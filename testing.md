# VLD Architecture — Comprehensive Test Plan

This document covers **every testable aspect** of the VLD head implementation to ensure correctness before training.

---

## Test 1: VLD Head — Shape Validation

**What**: Verify the head produces correct output tensor shapes.

```python
import torch
import sys
sys.path.insert(0, '/home/kishore/Smoke/SMOKE-master')

from smoke.modeling.heads.vld_head import VLDHead

# DLA-34 SMOKE output: 64 channels (from config BACKBONE_OUT_CHANNELS)
model = VLDHead(in_channels=64)
x = torch.randn(2, 64, 96, 320)  # batch=2, KITTI feature map size

heatmap, regression = model(x)

assert heatmap.shape == (2, 1, 96, 320), f"Heatmap shape wrong: {heatmap.shape}"
assert regression.shape == (2, 4, 96, 320), f"Regression shape wrong: {regression.shape}"
print("[PASS] Test 1: Output shapes correct")
```

**Expected**: `[2,1,96,320]` and `[2,4,96,320]`

---

## Test 2: Heatmap — Value Range

**What**: Verify sigmoid + clamp produces values in [1e-4, 1-1e-4].

```python
assert heatmap.min().item() >= 1e-4, f"Heatmap min too low: {heatmap.min()}"
assert heatmap.max().item() <= 1 - 1e-4, f"Heatmap max too high: {heatmap.max()}"
print("[PASS] Test 2: Heatmap range valid")
```

**Why this matters**: `log(0)` in focal loss = NaN. Clamping prevents this.

---

## Test 3: Heatmap — Bias Initialization

**What**: Verify the heatmap bias is initialized to -2.19.

```python
bias = model.heatmap_head[-1].bias.data
assert torch.allclose(bias, torch.tensor([-2.19])), f"Bias wrong: {bias}"
print("[PASS] Test 3: Heatmap bias initialization correct")
```

**Why this matters**: Without this, initial heatmap predictions will be ~0.5 everywhere, causing massive focal loss at step 0 and unstable early training.

---

## Test 4: Regression — Dimensions Are Positive

**What**: Verify channels 2-3 (Δh, Δw) are strictly positive after exp().

```python
dh_dw = regression[:, 2:4]
assert (dh_dw > 0).all(), "Dimension channels must be positive"
print("[PASS] Test 4: Regression dimensions are positive")
```

---

## Test 5: Regression — Exp Clamping (Numerical Stability)

**What**: Verify exp(clamp(·, max=10)) prevents overflow.

```python
# Feed extreme values
extreme_input = torch.randn(1, 64, 10, 10) * 100  # very large activations
h, r = model(extreme_input)
assert torch.isfinite(r).all(), "Regression has inf/nan values!"
print("[PASS] Test 5: Exp clamping prevents overflow")
```

---

## Test 6: Gradient Flow (Backward Pass)

**What**: Verify gradients propagate through the entire head.

```python
model.zero_grad()
loss = heatmap.mean() + regression.mean()
loss.backward()

for name, param in model.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
    assert torch.isfinite(param.grad).all(), f"Gradient NaN/Inf for {name}"

print("[PASS] Test 6: Gradient flow verified for all parameters")
```

---

## Test 7: Focal Loss — Correctness

**What**: Verify focal loss computation with known inputs.

```python
from smoke.modeling.loss.vld_loss import focal_loss

# Perfect prediction: pred matches gt
pred = torch.tensor([[[[0.9999]]]]) # predicted center
gt = torch.tensor([[[[1.0]]]])       # ground truth center
loss = focal_loss(pred, gt)
assert loss.item() < 0.01, f"Perfect prediction should have near-zero loss: {loss}"

# Wrong prediction: pred=0.5 where gt=1
pred_wrong = torch.tensor([[[[0.5]]]])
loss_wrong = focal_loss(pred_wrong, gt)
assert loss_wrong > loss, "Wrong prediction should have higher loss"

print("[PASS] Test 7: Focal loss behavior correct")
```

---

## Test 8: Focal Loss — Negative Weighting

**What**: Verify `(1-gt)^4` down-weights negatives near centers.

```python
# Background pixel far from center (gt=0)
pred_bg = torch.tensor([[[[0.01]]]])
gt_far = torch.tensor([[[[0.0]]]])
loss_far = focal_loss(pred_bg, gt_far)

# Background pixel near center (gt=0.8, from Gaussian tail)
gt_near = torch.tensor([[[[0.8]]]])
loss_near = focal_loss(pred_bg, gt_near)

# Near-center negative should have LOWER loss (down-weighted by (1-0.8)^4 = 0.0016)
assert loss_near < loss_far, "Near-center negatives should be down-weighted"
print("[PASS] Test 8: Focal loss negative weighting correct")
```

---

## Test 9: Regression Loss — Mask Behavior

**What**: Verify L1 loss only computes at masked positions.

```python
from smoke.modeling.loss.vld_loss import regression_loss

pred = torch.ones(1, 4, 10, 10) * 5.0
gt = torch.zeros(1, 4, 10, 10)
mask = torch.zeros(1, 1, 10, 10)
mask[0, 0, 5, 5] = 1  # only one active position

loss = regression_loss(pred, gt, mask)
assert loss.item() > 0, "Loss should be nonzero at masked position"

# Zero mask should give zero loss
zero_mask = torch.zeros(1, 1, 10, 10)
zero_loss = regression_loss(pred, gt, zero_mask)
assert zero_loss.item() == 0, "Zero mask should give zero loss"
print("[PASS] Test 9: Regression mask behavior correct")
```

---

## Test 10: Target Generation — Gaussian Heatmap

**What**: Verify Gaussian heatmap targets are generated correctly.

```python
from smoke.modeling.utils.vld_target import generate_vld_targets
import numpy as np

# Single bbox at center of a 96x320 feature map (already in feature-map coords)
bboxes = np.array([[155, 43, 165, 53]])  # [x1,y1,x2,y2]
hm, reg, mask = generate_vld_targets(bboxes, output_size=(96, 320), stride=1)

# Center should be at (160, 48)
center_x, center_y = 160, 48
assert hm[center_y, center_x] > 0.5, f"Heatmap peak too low at center: {hm[center_y, center_x]}"
assert mask[0, center_y, center_x] == 1, "Mask should be 1 at center"
assert reg[0, center_y, center_x] != 0 or reg[1, center_y, center_x] != 0, "Offsets should be set"
print("[PASS] Test 10: Target generation correct")
```

---

## Test 11: Target Generation — Adaptive Gaussian Radius

**What**: Verify larger objects get larger Gaussian radius.

```python
from smoke.modeling.utils.heatmap import gaussian_radius

r_small = gaussian_radius((5, 5))
r_large = gaussian_radius((50, 50))

assert r_large > r_small, f"Large object should have larger radius: {r_small} vs {r_large}"
print("[PASS] Test 11: Adaptive radius scales with object size")
```

---

## Test 12: Backbone Feature Compatibility

**What**: Verify VLD head accepts actual backbone output dimensions.

```python
from smoke.config import cfg

# Check configured output channels
print(f"Backbone output channels: {cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS}")
print(f"Backbone down ratio: {cfg.MODEL.BACKBONE.DOWN_RATIO}")

model = VLDHead(in_channels=cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS)
x = torch.randn(1, cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS, 96, 320)
h, r = model(x)
assert h.shape[1] == 1 and r.shape[1] == 4
print("[PASS] Test 12: Head compatible with configured backbone channels")
```

---

## Test 13: Freeze Strategy Validation

**What**: Verify only VLD head parameters are trainable.

```python
from smoke.modeling.detector import build_detection_model

model = build_detection_model(cfg)

# Simulate freeze strategy
for param in model.parameters():
    param.requires_grad = False
for param in model.vld_head.parameters():
    param.requires_grad = True

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable

print(f"Total parameters: {total:,}")
print(f"Trainable (VLD only): {trainable:,}")
print(f"Frozen: {frozen:,}")

assert trainable < total * 0.1, "VLD head should be <10% of total params"
print("[PASS] Test 13: Freeze strategy correct")
```

---

## Test 14: Parallel Head Independence

**What**: Verify VLD head does not interfere with SMOKE head outputs.

```python
model = build_detection_model(cfg)
model.eval()

x = torch.randn(1, 3, 384, 1280)

# VLD outputs should exist alongside SMOKE
outputs = model(x)
assert "vld_heatmap" in outputs, "VLD heatmap missing from outputs"
assert "vld_regression" in outputs, "VLD regression missing from outputs"
assert "smoke_result" in outputs, "SMOKE result missing from outputs"
print("[PASS] Test 14: Parallel heads produce independent outputs")
```

---

## Running All Tests

Save all tests to a single file and execute:

```bash
cd /home/kishore/Smoke/SMOKE-master
PYTHONPATH=. python tests/test_vld_complete.py
```

All 14 tests should print `[PASS]`. Any failure indicates a bug that must be fixed before training.
