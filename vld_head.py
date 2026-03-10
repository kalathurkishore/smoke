import torch
import torch.nn as nn
import torch.nn.functional as F

class VLDHead(nn.Module):
    """
    VLD Architecture Proposal Head
    Based on DLA-34 features.
    Predicts 2D center point and regression offsets (Δu, Δv, Δh, Δw).
    """

    def __init__(self, in_channels=64, hidden_channels=128):
        super(VLDHead, self).__init__()
        
        # 1. Base Convolution (Shared Features)
        # Diagram: Conv (64->128, kernel=3, padding=1, stride=1) -> ReLU
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Keypoint Heatmap Branch -> Predicts 2D center point
        # Diagram: Conv (128->128, k=3) -> ReLU -> Conv (128->1, k=1) -> Sigmoid -> Clip
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, stride=1)
        )
        
        # 3. Regression Branch -> Predicts (Δu, Δv, Δh, Δw)
        # Diagram: Conv (128->128, k=3) -> ReLU -> Conv (128->4, k=1)
        self.regression_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        """
        Forward pass for VLD Head.
        Returns:
            heatmap_out: Peak-extracted 2D center point predictions
            regression_out: Concatenated (Δu, Δv, exp(Δh), exp(Δw))
        """
        # --- Shared Features ---
        base_features = self.base_conv(x)
        
        # --- Keypoint Heatmap Branch ---
        hm = self.heatmap_branch(base_features)
        hm = torch.sigmoid(hm)
        
        # Clip values (from diagram: min=0.001, max=0.999)
        hm = torch.clamp(hm, min=1e-3, max=1 - 1e-3)
        
        # Local Max Pooling for Peak Extraction
        # Diagram nodes: MaxPool -> Equal -> Cast -> Mul
        # This acts as a standard CenterNet-style Non-Maximum Suppression (NMS)
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        keep = (hmax == hm).float()  # Equivalent to Equal + Cast
        heatmap_out = hm * keep      # Equivalent to Mul
        
        # --- Regression Branch ---
        reg = self.regression_branch(base_features)
        
        # Slicing channels into (Δu, Δv) and (Δh, Δw) 
        # as indicated by the Slice [0:2] and Slice [2:4] nodes
        du_dv = reg[:, 0:2, :, :]  # first two channels
        dh_dw = reg[:, 2:4, :, :]  # last two channels
        
        # Activation for dimension offsets 
        # (Diagram shows a short node like 'Exp' to enforce strictly positive sizes)
        dh_dw = torch.exp(dh_dw) 
        
        # Combine back into a single tensor
        regression_out = torch.cat([du_dv, dh_dw], dim=1)
        
        return heatmap_out, regression_out


def build_vld_training_optimizer(model, lr=1e-4):
    """
    Implements the Training Strategy:
    - Freeze learned backbone and other heads
    - Fine-tune only VLD head
    """
    # 1. Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze only the VLD head parameters
    for param in model.vld_head.parameters():
        param.requires_grad = True
        
    # 3. Create optimizer with only the trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    return optimizer
