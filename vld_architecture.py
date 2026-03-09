import torch
import torch.nn as nn
import torch.nn.functional as F

class VLDHead(nn.Module):
    def __init__(self, in_channels=128):
        super(VLDHead, self).__init__()
        
        # 1. Keypoint heatmap branch to predict 2D center point
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # 2. Regression head to predict (du, dv, dh, dw)
        self.regression_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1)
        )

    def forward(self, x):
        # ---------------- Heatmap Branch ----------------
        heatmap = self.heatmap_branch(x)
        heatmap = torch.sigmoid(heatmap)
        heatmap = torch.clamp(heatmap, min=1e-4, max=1.0 - 1e-4) # Clip as shown in the diagram
        
        # MaxPool 3x3 for center point peak detection (NMS)
        hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        # Keep only the peak points
        keep = (hmax == heatmap).float()
        heatmap_out = heatmap * keep

        # ---------------- Regression Branch ----------------
        regression = self.regression_branch(x)
        
        # Replicating the Slicing/Post-processing seen in the diagram
        # Slicing the 4 channels: first 2 for sub-pixel offsets (du, dv), next 2 for size (dh, dw)
        offsets = regression[:, 0:2, :, :]
        sizes = regression[:, 2:4, :, :]
        
        # Apply sigmoid to offsets to constrain them between 0 and 1
        offsets = torch.sigmoid(offsets)
        
        # Concat them back (or return separately depending on loss function needs)
        regression_out = torch.cat([offsets, sizes], dim=1)

        return heatmap_out, regression_out

if __name__ == "__main__":
    # Create the model
    model = VLDHead(in_channels=128)
    model.eval() # Set to evaluation mode for exporting

    # Create dummy input based on the dimensions in the diagram
    # Y = 64x64x128 in diagram (assuming B=1, C=128, H=64, W=64, although Netron might show standard C,H,W shapes)
    dummy_input = torch.randn(1, 128, 64, 64)

    # Export to ONNX
    onnx_path = "vld_architecture.onnx"
    print(f"Exporting PyTorch model to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['dla34_features'], 
        output_names=['heatmap_pred', 'regression_pred']
    )
    print("Export complete! You can now visualize this ONNX file using Netron.")
