# VLD Training Trigger — Step-by-Step Guide

This document provides the **exact steps** to initiate VLD head training inside the SMOKE repository.

---

## Prerequisites

### 1. Environment Setup

```bash
cd /home/kishore/Smoke/SMOKE-master

# Verify PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify SMOKE imports work
PYTHONPATH=. python -c "from smoke.modeling.heads.vld_head import VLDHead; print('VLDHead imported successfully')"
```

### 2. Dataset (KITTI)

The KITTI 3D object detection dataset must be organized as:

```
datasets/kitti/
├── training/
│   ├── image_2/        # RGB images (*.png)
│   ├── label_2/        # Annotations (*.txt)
│   ├── calib/           # Camera calibration (*.txt)
│   └── ImageSets/
│       ├── train.txt
│       └── val.txt
```

Update the dataset path in `smoke/config/paths_catalog.py` if needed.

### 3. Pretrained SMOKE Weights

Download the pretrained SMOKE model checkpoint from the original repo or use your existing trained model.

Place at:
```
tools/logs/model_final.pth
```

Update the config:
```yaml
MODEL:
  WEIGHT: "tools/logs/model_final.pth"
```

---

## Training Configuration

### Config File: `configs/smoke_gn_vector.yaml`

The existing config should work. Key parameters to verify:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BACKBONE.CONV_BODY` | `DLA-34-DCN` | Backbone architecture |
| `BACKBONE.BACKBONE_OUT_CHANNELS` | 64 | Feature channels to VLD head |
| `BACKBONE.DOWN_RATIO` | 4 | Spatial downsampling factor |
| `SOLVER.BASE_LR` | 0.00025 | Learning rate |
| `SOLVER.MAX_ITERATION` | 14500 | Training iterations |
| `DATASETS.MAX_OBJECTS` | 30 | Max objects per image |

---

## Training Execution

### Step 1: Verify Model Builds

```bash
cd /home/kishore/Smoke/SMOKE-master

PYTHONPATH=. python -c "
from smoke.config import cfg
from smoke.modeling.detector import build_detection_model

cfg.merge_from_file('configs/smoke_gn_vector.yaml')
model = build_detection_model(cfg)

# Check VLD head exists
assert hasattr(model, 'vld_head'), 'VLD head not attached!'

# Count parameters
total = sum(p.numel() for p in model.parameters())
vld = sum(p.numel() for p in model.vld_head.parameters())
print(f'Total params: {total:,}')
print(f'VLD head params: {vld:,}')
print(f'VLD is {vld/total*100:.1f}% of total')
print('Model built successfully!')
"
```

### Step 2: Verify Freeze Strategy

```bash
PYTHONPATH=. python -c "
from smoke.config import cfg
from smoke.modeling.detector import build_detection_model

cfg.merge_from_file('configs/smoke_gn_vector.yaml')
model = build_detection_model(cfg)

# Apply freeze
for p in model.parameters():
    p.requires_grad = False
for p in model.vld_head.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable:,}')
print('Only VLD head is trainable!')
"
```

### Step 3: Launch Training

```bash
cd /home/kishore/Smoke/SMOKE-master

# Single GPU training
PYTHONPATH=. python tools/plain_train_net.py \
    --config-file configs/smoke_gn_vector.yaml \
    --num-gpus 1

# Multi-GPU training (if available)
PYTHONPATH=. python tools/plain_train_net.py \
    --config-file configs/smoke_gn_vector.yaml \
    --num-gpus 2
```

### Step 4: Monitor Training

Watch for these log entries:

```
eta: HH:MM:SS  iter: N  loss: X.XX  hm_loss: X.XX  reg_loss: X.XX  vld_hm_loss: X.XX  vld_reg_loss: X.XX  lr: 0.00025000
```

**Healthy training indicators**:
- `vld_hm_loss` should **decrease** from ~2-5 down to <1 over iterations
- `vld_reg_loss` should **decrease** steadily
- No `NaN` or `inf` values in any loss

**Warning signs**:
- `vld_hm_loss` stays constant → heatmap not learning (check target generation)
- `NaN` in loss → numerical instability (check exp clamping)
- Loss explodes → learning rate too high (try 1e-5)

---

## Post-Training Verification

### Step 5: Verify Saved Checkpoint

```bash
PYTHONPATH=. python -c "
import torch
ckpt = torch.load('tools/logs/model_final.pth', map_location='cpu')

# Check VLD head weights exist in checkpoint
vld_keys = [k for k in ckpt['model'].keys() if 'vld_head' in k]
print(f'VLD head weights in checkpoint: {len(vld_keys)}')
for k in vld_keys:
    print(f'  {k}: {ckpt[\"model\"][k].shape}')
"
```

### Step 6: Run Inference Test

```bash
PYTHONPATH=. python -c "
import torch
from smoke.config import cfg
from smoke.modeling.detector import build_detection_model

cfg.merge_from_file('configs/smoke_gn_vector.yaml')
model = build_detection_model(cfg)
model.eval()

# Load trained weights
ckpt = torch.load('tools/logs/model_final.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])

# Dummy inference
x = torch.randn(1, 3, 384, 1280)
with torch.no_grad():
    outputs = model(x)

print('VLD Heatmap shape:', outputs['vld_heatmap'].shape)
print('VLD Regression shape:', outputs['vld_regression'].shape)
print('Heatmap peak value:', outputs['vld_heatmap'].max().item())
print('Inference successful!')
"
```

---

## Requirement Fulfillment Checklist

After training completes, verify each ARAS proposal requirement:

| # | Requirement | How to Verify | Command |
|---|------------|---------------|---------|
| 1 | DLA-34 features | Check backbone type in config | `grep CONV_BODY configs/*.yaml` |
| 2 | Last backbone layer | Check `return y[-1]` in `dla.py` L274 | `grep "return y" smoke/modeling/backbone/dla.py` |
| 3 | Keypoint heatmap (2D center) | Check heatmap output shape `[B,1,H,W]` | Run inference test above |
| 4 | Regression (Δu,Δv,Δh,Δw) | Check regression shape `[B,4,H,W]` | Run inference test above |
| 5 | 2D image space offsets | Check `vld_target.py` uses 2D bbox coords | `cat smoke/modeling/utils/vld_target.py` |
| 6 | Freeze backbone | Check `requires_grad=False` in train script | `grep requires_grad tools/plain_train_net.py` |
| 7 | Fine-tune VLD only | Check only VLD params are trainable | Run freeze verification above |
| 8 | No neck | No FPN/PAN in detector | `grep -r "FPN\|PAN\|neck" smoke/modeling/` |
| 9 | Lightweight head | Count VLD parameters (<200K) | Run model build test above |
| 10 | No anchors | No anchor generation in VLD head | `grep -r "anchor" smoke/modeling/heads/vld_head.py` |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Missing PYTHONPATH | Run with `PYTHONPATH=.` |
| `RuntimeError: size mismatch` | Wrong `in_channels` | Verify `BACKBONE_OUT_CHANNELS=64` |
| `NaN in loss` | exp overflow | Verify `torch.clamp(max=10)` in vld_head.py |
| `Loss not decreasing` | Targets not generated | Check kitti.py adds `bboxes` field |
| `CUDA out of memory` | Batch too large | Reduce `SOLVER.IMS_PER_BATCH` |
