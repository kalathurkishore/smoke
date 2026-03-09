import sys
import os

# Add SMOKE directory to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SMOKE'))

import torch
from smoke.modeling.detector import build_detection_model
from smoke.config import cfg

cfg.merge_from_file("SMOKE/configs/smoke_gn_vector.yaml")
cfg.MODEL.DEVICE = "cpu"

full_model = build_detection_model(cfg)
full_model.eval()

# To visualize the neural architecture, we only need the backbone and predictor.
# The post_processor expects camera intrinsics in `targets` which isn't part of the NN structure.
class SMOKENeuralNet(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.backbone = full_model.backbone
        self.predictor = full_model.heads.predictor

    def forward(self, x):
        features = self.backbone(x)
        return self.predictor(features)

model = SMOKENeuralNet(full_model)
model.eval()

# B x C x H x W (3x384x1280 is a typical image size used in KITTI)
x = torch.randn(1, 3, 384, 1280)

onnx_path = "smoke_full_architecture.onnx"
print(f"Exporting full SMOKE model to {onnx_path}...")

torch.onnx.export(
    model, 
    x, 
    onnx_path, 
    export_params=True, 
    opset_version=14, 
    do_constant_folding=True,
    input_names=['image_input'], 
    output_names=['detection_outputs']
)
print("Export complete!")

