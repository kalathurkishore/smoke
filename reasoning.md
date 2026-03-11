# VLD Architecture — Design Reasoning

## 1. Problem Statement

The ARAS project requires detecting **2D visible light sources** (headlights, tail lights, indicators) from monocular camera images. The detector must be:
- Lightweight enough for real-time inference
- Accurate enough for downstream ADAS pipeline consumption
- Compatible with the existing SMOKE 3D detection backbone (DLA-34)

---

## 2. Why Center-Based Detection (Not Anchor-Based)?

### Anchor-based detectors (YOLO, Faster R-CNN)
- Require **predefined anchor boxes** with tuned aspect ratios
- Generate **thousands of proposals** that must be filtered via NMS
- Anchors designed for vehicles/pedestrians are **poorly suited for small lights**

### Center-based detectors (CenterNet-style, our VLD)
- Predict a **single heatmap peak per object** — no anchor tuning
- Naturally handle **small, point-like objects** (lights are essentially bright points)
- Regression outputs are **direct offsets** — no proposal generation overhead
- The architecture is **inherently simpler** with fewer hyperparameters

**Decision**: Center-based detection is the correct choice because vehicle lights are small, point-like objects where center localization is highly effective.

---

## 3. Why Reuse the DLA-34 Backbone?

The SMOKE model already trains a DLA-34 backbone on driving scenes (KITTI dataset). This backbone has **already learned**:
- Road scene semantics (lane markings, vehicles, pedestrians)
- Illumination patterns (shadows, reflections, light sources)
- Geometric structures (car shapes, dimensions)

**Decision**: Reuse the pretrained DLA-34 backbone and freeze it. Training only the VLD head on top:
- Saves **weeks** of backbone training
- Requires **less labeled VLD data** (transfer learning)
- Uses **less GPU memory** (only head gradients stored)
- Produces **stable training** (backbone features don't drift)

---

## 4. Why No Neck (FPN/PAN)?

Modern detectors (YOLOv5, EfficientDet) use Feature Pyramid Networks (FPN) to handle multi-scale objects. However:

| Factor | With Neck | Without Neck |
|--------|-----------|--------------|
| Parameter count | High | Low |
| Latency | Higher | Lower |
| Multi-scale detection | Excellent | Sufficient for lights |
| Implementation complexity | High | Low |

Vehicle lights occupy a **narrow size range** (small to medium). Multi-scale detection is unnecessary. The DLA-34 already performs internal multi-scale fusion via its tree architecture.

**Decision**: Skip the neck entirely. Connect head directly to backbone output.

---

## 5. VLD Head Architecture Reasoning

### 5.1 Shared Convolution Layer

```
Backbone (64 channels) → Conv(64 → 128) → ReLU
```

**Why**: The backbone output has only 64 channels, which may be too sparse for two independent tasks. A shared conv layer:
- Increases feature capacity to 128 channels
- Learns **task-specific feature refinement** before branching
- Acts as a bottleneck that **prevents overfitting** on the small VLD dataset

### 5.2 Heatmap Branch

```
Conv(128→128) → ReLU → Conv(128→1) → Sigmoid → Clamp
```

**Key design choices**:
- **1 output channel**: Single class (light source). Multi-class would use N channels.
- **Sigmoid activation**: Maps output to [0,1] probability range
- **Clamp(1e-4, 1-1e-4)**: Prevents `log(0)` in focal loss computation
- **Bias init = -2.19**: Sets initial sigmoid output to `σ(-2.19) ≈ 0.1`, which means the model starts by predicting "no object everywhere." This is critical because:
  - Heatmaps are **extremely sparse** (99%+ background)
  - Without this init, the model starts at 0.5 which creates massive focal loss at step 0
  - This trick comes from the CenterNet and CornerNet papers

### 5.3 Regression Branch

```
Conv(128→128) → ReLU → Conv(128→4)
```

The 4 output channels encode:

| Channel | Symbol | Meaning | Activation |
|---------|--------|---------|-----------|
| 0 | Δu | Sub-pixel x offset from integer center | None (raw) |
| 1 | Δv | Sub-pixel y offset from integer center | None (raw) |
| 2 | Δh | Object height | exp(clamp(·, max=10)) |
| 3 | Δw | Object width | exp(clamp(·, max=10)) |

**Why exp() for height/width**: Dimensions must be strictly positive. The exp function maps any real value to (0, ∞). The clamp(max=10) prevents exp overflow (e^10 ≈ 22026, sufficient for any pixel dimension).

**Why raw output for offsets**: Offsets can be negative or positive (sub-pixel correction), so no activation is needed.

---

## 6. Loss Function Reasoning

### 6.1 Focal Loss for Heatmap

```
L = -(1-p)^α · log(p)          for positive pixels (gt=1)
  = -(1-gt)^β · p^α · log(1-p)  for negative pixels (gt<1)
```

Where α=2, β=4 (CenterNet defaults).

**Why focal loss (not BCE)**:
- Heatmaps are **extremely imbalanced** (1 positive pixel per object vs thousands of background pixels)
- Focal loss **down-weights easy negatives** via the `p^2` term
- The `(1-gt)^4` term reduces penalty for background pixels **near** object centers (Gaussian tail region), teaching the model that "close to center is partially correct"

### 6.2 L1 Loss for Regression

```
L = |pred - gt| · mask
```

**Why L1 (not L2)**:
- L1 is **less sensitive to outliers** — a single wrong prediction won't dominate the loss
- Used with a **binary mask** so loss is computed only at detected center locations
- Weight factor of **0.1** balances regression against heatmap loss (heatmap learning is primary)

---

## 7. Training Strategy Reasoning

### 7.1 Why Freeze Everything First

```python
for param in model.parameters():
    param.requires_grad = False
```

- The DLA-34 backbone is pretrained on driving scenes
- The SMOKE heads are pretrained for 3D detection
- **Both are useful as-is** — unfreezing them would risk catastrophic forgetting

### 7.2 Why Unfreeze Only VLD Head

```python
for param in model.vld_head.parameters():
    param.requires_grad = True
```

- Only VLD head weights need to learn the new light detection task
- Results in **~100K trainable parameters** (vs ~18M total model parameters)
- Training converges in **minutes** instead of hours
- Can use a **higher learning rate** since only a small head is being optimized

---

## 8. Parallel Head Architecture Reasoning

```
DLA-34 Backbone
      │
      ▼
Feature Map [B,64,H,W]
      │
 ┌────┴────────────┐
 │                  │
SMOKE Head         VLD Head
(3D detection)     (2D light detection)
```

**Why parallel (not sequential)**:
- Both heads **independently consume the same features** — no dependency
- SMOKE head continues to work for 3D vehicle detection
- VLD head adds light detection **without modifying** the original pipeline
- At inference time, either head can be disabled independently

---

## 9. Summary: How Each Proposal Requirement Is Met

| Requirement | Design Decision | Justification |
|-------------|----------------|---------------|
| DLA-34 features | Reuse frozen pretrained backbone | Transfer learning, stability |
| Last layer of backbone | DLA returns final fused `y[-1]` tensor | Richest semantic features |
| Keypoint heatmap (2D center) | 1-channel sigmoid output + focal loss | Sparse detection, CenterNet-proven |
| Regression (Δu, Δv, Δh, Δw) | 4-channel output, exp for sizes | Direct offset prediction |
| 2D image space | Targets in feature-map coordinates | No 3D projection needed |
| Freeze backbone + heads | `requires_grad = False` globally | Preserve learned features |
| Fine-tune VLD only | `requires_grad = True` for VLD | Fast convergence, minimal data |
| No neck | Direct backbone → head | Reduced latency and complexity |
| Lightweight head | 3 conv layers per branch | ~100K parameters |
| No anchors | Center-based detection | Better for small point-like objects |
