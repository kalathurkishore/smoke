import torch
import timm

print("Creating DLA-34 model from timm...")
# Standard DLA-34
model = timm.create_model('dla34', pretrained=False)
model.eval()

# Dummy input matching typical dimensions (e.g. 1 x 3 x 224 x 224 or larger)
dummy_input = torch.randn(1, 3, 256, 256)

onnx_path = "/home/kishore/Smoke/dla34_architecture.onnx"
print(f"Exporting DLA-34 model to {onnx_path}...")

try:
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input_image'], 
        output_names=['dla34_features']
    )
    print("Export complete!")
except Exception as e:
    print(f"Error during ONNX export: {e}")
