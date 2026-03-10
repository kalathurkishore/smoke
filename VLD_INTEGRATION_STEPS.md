# Integrating VLD Head into SMOKE Architecture

This guide details the explicit codebase modifications required to successfully mount the ARAS VLD Head (Visual Light Detection) into an existing fork of the SMOKE (Single-Stage Monocular 3D Object Detection) architecture.

## 1. Required New Files
Ensure these 4 files are present in your SMOKE repository exactly as detailed in the technical documentation:
- `smoke/modeling/heads/vld_head.py`: The network layers for the dual heatmap/regression branch.
- `smoke/modeling/loss/vld_loss.py`: The designated focal and L1 loss components.
- `smoke/modeling/utils/heatmap.py`: The adaptive CenterNet Gaussian calculation logic.
- `smoke/modeling/utils/vld_target.py`: Target processing logic for dynamic radii scaling.

---

## 2. Modifications to the Dataloader
**File:** `smoke/data/datasets/kitti.py`

You must pass the 2D bounding boxes down into the Target configurations so that your target generator can construct the heatmap arrays during network execution.

**In `__getitem__()` during array allocation:**
```python
bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
```

**Inside the annotations iteration loop `for i, a in enumerate(anns):`:**
```python
bboxes[i] = box2d
```

**At the end of `__getitem__()` when adding fields to `target`:**
```python
target.add_field("bboxes", bboxes)
```

---

## 3. Modifications to the Primary Detector
**File:** `smoke/modeling/detector/keypoint_detector.py`

This is where you branch the computational load from SMOKE's heads into the parallel VLD Head.

**Imports at the top of the file:**
```python
from ..heads.vld_head import VLDHead
from ..loss.vld_loss import focal_loss, regression_loss
from ..utils.vld_target import generate_vld_targets
```

**Inside the `__init__` constructor:**
```python
self.vld_head = VLDHead(in_channels=self.backbone.out_channels) 
# Note: DO NOT hardcode 256. Access the out_channels directly from the initialized backbone.
```

**Inside the `forward()` method logic:**
Instead of returning just `detector_losses` natively, calculate the VLD properties. Ensure you access the final emitted representation layer.
```python
# Extract the final DLA-34 layer (crucial step!)
vld_feat = features[0] if isinstance(features, (list, tuple)) else features
vld_heatmap, vld_reg = self.vld_head(vld_feat)

if self.training:
    losses = dict(detector_losses)

    output_size = (vld_feat.shape[2], vld_feat.shape[3]) # H, W
    gt_heatmaps, gt_regs, gt_masks = [], [], []
    
    for t in targets:
        bboxes = t.get_field("bboxes")
        valid_mask = t.get_field("reg_mask") == 1
        hm, reg, mask = generate_vld_targets(bboxes[valid_mask].cpu().numpy(), output_size, stride=1)
        gt_heatmaps.append(hm)
        gt_regs.append(reg)
        gt_masks.append(mask)

    # Cast to GPU variables
    gt_heatmaps = torch.stack(gt_heatmaps).unsqueeze(1).to(vld_heatmap.device)
    gt_regs = torch.stack(gt_regs).to(vld_reg.device)
    gt_masks = torch.stack(gt_masks).to(vld_reg.device)

    # Accumulate Loss Output
    losses["vld_hm_loss"] = focal_loss(vld_heatmap, gt_heatmaps)
    losses["vld_reg_loss"] = 0.1 * regression_loss(vld_reg, gt_regs, gt_masks)
    return losses
    
return {
    "vld_heatmap": vld_heatmap,
    "vld_regression": vld_reg,
    "smoke_result": result
}
```

---

## 4. Modifications to the Training Loop (Freezing)
**File:** `tools/plain_train_net.py`

The ARAS training strategy necessitates freezing previous components sequentially so network gradients converge cleanly within the VLD parameters alone.

**Inside the `train()` definition function (BEFORE Optimizer Init):**
```python
def train(cfg, model, device, distributed):
    # Completely freeze the full backbone + SMOKE regression heads
    for param in model.parameters():
        param.requires_grad = False
    
    # Isolate active tuning specifically for VLD components
    _model = model.module if distributed else model
    if hasattr(_model, 'vld_head'):
        for param in _model.vld_head.parameters():
            param.requires_grad = True

    # Standard execution continues
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
```

---

*This guide mirrors the precise edits and logic merged successfully under commit `4c540e9` internally.*
