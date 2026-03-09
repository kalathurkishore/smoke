# 🚗 SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

> **CVPRW 2020** | [Paper (PDF)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Liu_SMOKE_Single-Stage_Monocular_3D_Object_Detection_via_Keypoint_Estimation_CVPRW_2020_paper.pdf) | [arXiv](https://arxiv.org/abs/2002.10111) | [Official Code](https://github.com/lzccccc/SMOKE)

SMOKE is a **single-stage**, **anchor-free** monocular 3D object detection method that predicts 3D bounding boxes from a single RGB image by combining **keypoint estimation** with **regressed 3D variables**. It eliminates the need for a separate 2D detection stage, achieving state-of-the-art results on the KITTI benchmark with a runtime of ~30ms per frame.

---

## 📑 Table of Contents

- [1. Problem Statement & Motivation](#1-problem-statement--motivation)
- [2. Background Fundamentals](#2-background-fundamentals)
- [3. SMOKE Architecture Overview](#3-smoke-architecture-overview)
- [4. Component Deep Dive](#4-component-deep-dive)
- [5. Loss Functions](#5-loss-functions)
- [6. Training & Inference Pipeline](#6-training--inference-pipeline)
- [7. Results & Benchmarks](#7-results--benchmarks)
- [8. Setup & Usage](#8-setup--usage)
- [9. Key Takeaways](#9-key-takeaways)
- [10. References](#10-references)

---

## 1. Problem Statement & Motivation

### What Problem Does SMOKE Solve?

**Monocular 3D Object Detection** = predicting the **3D position, dimensions, and orientation** of objects (cars, pedestrians, cyclists) from a **single 2D RGB camera image**.

This is critical for:
- **Autonomous driving** — understanding where objects are in 3D space
- **Infrastructure-less navigation** — no LiDAR, radar, or stereo cameras needed
- **Cost efficiency** — a single cheap camera vs. expensive multi-sensor setups

### Why Is This Hard?

A single 2D image inherently **loses depth information** during the projection from 3D world → 2D image plane. Recovering the lost dimension is an **ill-posed inverse problem**.

```
3D World                    2D Image
┌─────────────┐             ┌───────────┐
│  Car at      │  Camera     │           │
│  (x,y,z)    │ ──────────→ │  Car at   │
│  w=1.8m     │  Projection  │  (u,v)    │
│  h=1.5m     │             │  pixel    │
│  l=4.5m     │             │  coords   │
│  yaw=30°    │             │           │
└─────────────┘             └───────────┘
   7 unknowns                 depth is LOST
```

### Why Not Use Existing Two-Stage Methods?

Before SMOKE, the dominant approach was:

```
Stage 1: 2D Detector (e.g., Faster R-CNN)
    ↓ generates 2D bounding box proposals
Stage 2: 3D Estimator (R-CNN head)
    ↓ uses cropped regions to predict 3D params
Output: 3D bounding boxes
```

**Problems with two-stage approaches:**
1. **Redundancy** — 2D bounding boxes are not needed for 3D detection
2. **Noise propagation** — errors in 2D detection cascade into 3D estimation
3. **Misalignment** — the center of a 2D bounding box ≠ the projected center of the 3D bounding box
4. **Complexity** — requires R-CNN architecture, NMS, anchor generation, etc.

### SMOKE's Key Insight

> *"The 2D detection network is redundant and introduces non-negligible noise for 3D detection."*

SMOKE removes 2D detection entirely and instead:
1. Detects the **projected 3D center** as a keypoint on a heatmap
2. Regresses **3D variables** (depth, dimensions, orientation) directly
3. Combines both to construct the **3D bounding box**

---

## 2. Background Fundamentals

### 2.1 Camera Projection Model (Pinhole Camera)

Understanding how 3D points map to 2D pixels is essential for SMOKE.

**The Pinhole Camera Model:**

```
         3D Point (X, Y, Z) in world coordinates
                    │
                    ▼
      ┌─────────────────────────────┐
      │   Camera Intrinsic Matrix K │
      │                             │
      │   K = │ fx  0  cx │         │
      │       │ 0  fy  cy │         │
      │       │ 0   0   1 │         │
      └─────────────────────────────┘
                    │
                    ▼
         2D Point (u, v) on image plane
```

**The projection equation:**

```
┌ u ┐       ┌ fx  0  cx ┐   ┌ X ┐
│ v │ = 1/Z │ 0  fy  cy │ × │ Y │
└ 1 ┘       └ 0   0   1 ┘   └ Z ┘

Where:
  (X, Y, Z) = 3D point in camera coordinates
  (u, v)    = 2D pixel coordinates on image
  fx, fy    = focal lengths (in pixels)
  cx, cy    = principal point (image center)
  Z         = depth (distance from camera)
```

**Why this matters for SMOKE:** Given the keypoint `(u, v)` on the image and the predicted depth `Z`, SMOKE can **back-project** to recover the 3D location `(X, Y, Z)` using the known camera matrix `K`.

### 2.2 3D Bounding Box Representation (7 Degrees of Freedom)

Every 3D bounding box in autonomous driving is described by **7 DoF**:

```
┌─────────────────────────────────────────────────────┐
│                   7 DoF of a 3D Box                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Location:    (x, y, z)  — 3D center in camera      │
│               coordinates (3 DoF)                   │
│                                                     │
│  Dimensions:  (h, w, l)  — height, width, length    │
│               of the box (3 DoF)                    │
│                                                     │
│  Orientation: θ (yaw)    — rotation around the      │
│               vertical (Y) axis (1 DoF)              │
│               (pitch & roll ≈ 0 in driving)         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**The 8-corner representation:**

Given the 7 DoF, SMOKE converts them to the **8 corners** of the 3D cuboid:

```
        3────────────2
       /|           /|
      / |          / |          Y (up)
     0────────────1  |          │
     |  7─────────|──6          │
     | /          | /           └───── X (right)
     |/           |/           /
     4────────────5           Z (forward/depth)

Each corner is a 3D point → project all 8 to 2D → 16 values total
```

### 2.3 Keypoint Detection (CenterNet Approach)

SMOKE builds on the **CenterNet** philosophy:

> *"Represent each object as a single point — the center of its bounding box."*

**How keypoint heatmaps work:**

```
Input Image (H × W)              Heatmap (H/4 × W/4)
┌──────────────────┐             ┌──────────────────┐
│                  │             │                  │
│    ┌──────┐      │  Network    │      ○           │
│    │ Car  │      │ ────────→   │   (Gaussian      │
│    └──────┘      │             │    peak at        │
│                  │             │    projected      │
│         ┌────┐   │             │    3D center)     │
│         │Ped │   │             │            ○      │
│         └────┘   │             │                   │
└──────────────────┘             └──────────────────┘

Each class gets its own heatmap channel (C channels total)
Peak locations = detected object centers
```

**Key Difference from CenterNet:** CenterNet uses the **2D bounding box center** as the keypoint. SMOKE uses the **projected 3D bounding box center** — this is a *virtual keypoint* that may not even lie inside the 2D bounding box.

```
            2D Box Center (CenterNet)
                    ↓
        ┌───────────●───────────┐
        │                       │
        │           ★           │  ← Projected 3D Center (SMOKE)
        │           │           │     (can be offset from 2D center!)
        │           │           │
        └───────────┘───────────┘

★ is more geometrically meaningful for 3D reconstruction
```

### 2.4 Focal Loss (Background)

The heatmap is mostly zeros (background) with only a few positive peaks. This causes a **class imbalance** problem.

**Focal Loss** (Lin et al., 2017) addresses this:

```
Standard Cross-Entropy:  L = -log(p)

Focal Loss:  L = -α(1-p)^γ × log(p)

Where:
  α  = balancing factor for class frequency
  γ  = focusing parameter (typically 2)
  (1-p)^γ = DOWN-weights easy/well-classified examples
           = UP-weights hard/misclassified examples
```

SMOKE uses a **penalty-reduced variant** where ground-truth is not binary (0/1) but a Gaussian-splattered heatmap.

### 2.5 Deep Layer Aggregation (DLA)

**DLA** is a feature extraction architecture that aggregates features across layers and scales more effectively than standard ResNets.

```
Standard ResNet:
  Layer1 → Layer2 → Layer3 → Layer4 → Output
  (Each layer only sees the previous layer)

DLA (Deep Layer Aggregation):
  Layer1 ──→ Layer2 ──→ Layer3 ──→ Layer4
     │          │          │          │
     └──────────┴──────────┴──────────┘
          Iterative Deep Aggregation (IDA)
          + Hierarchical Deep Aggregation (HDA)

  = Combines features from ALL scales and depths
  = Better multi-scale feature representation
```

**DLA-34** specifically has 34 layers and uses two aggregation strategies:
- **IDA (Iterative Deep Aggregation):** Merges adjacent scales progressively
- **HDA (Hierarchical Deep Aggregation):** Merges features in a tree structure across the network depth

---

## 3. SMOKE Architecture Overview

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMOKE Architecture                               │
│                                                                     │
│  Input Image (1242 × 375)                                          │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │     Backbone: Modified DLA-34       │                           │
│  │  (with Deformable Convolutions)     │                           │
│  │                                     │                           │
│  │  Input:  H × W × 3                 │                           │
│  │  Output: H/4 × W/4 × 256           │                           │
│  └────────────────┬────────────────────┘                           │
│                   │                                                 │
│          Feature Map (shared)                                       │
│           ┌───────┴───────┐                                        │
│           ▼               ▼                                        │
│  ┌────────────────┐ ┌──────────────────┐                           │
│  │  Keypoint Head │ │  Regression Head │                           │
│  │                │ │                  │                           │
│  │ Predicts:      │ │ Predicts 8 params│                           │
│  │ Heatmap of     │ │ per keypoint:    │                           │
│  │ projected 3D   │ │                  │                           │
│  │ centers        │ │ • δz (depth)     │                           │
│  │                │ │ • δx,δy (offset) │                           │
│  │ Output:        │ │ • δh,δw,δl (dim) │                           │
│  │ H/4 × W/4 × C │ │ • sin θ, cos θ   │                           │
│  │ (C = classes)  │ │                  │                           │
│  └────────┬───────┘ │ Output:          │                           │
│           │         │ H/4 × W/4 × 8   │                           │
│           │         └────────┬─────────┘                           │
│           └───────┬──────────┘                                     │
│                   ▼                                                 │
│         ┌─────────────────────┐                                    │
│         │  3D Box Constructor │                                    │
│         │                     │                                    │
│         │  Combines keypoint  │                                    │
│         │  + regression params│                                    │
│         │  + camera matrix K  │                                    │
│         │  → 8-corner 3D box  │                                    │
│         └─────────────────────┘                                    │
│                   │                                                 │
│                   ▼                                                 │
│         3D Bounding Boxes Output                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Output | Dimensions |
|-------|-------|--------|------------|
| Input | RGB Image | — | `H × W × 3` |
| Backbone (DLA-34 + DCN) | `H × W × 3` | Feature Map | `H/4 × W/4 × 256` |
| Keypoint Head | Feature Map | Class Heatmaps | `H/4 × W/4 × C` |
| Regression Head | Feature Map | 3D Parameters | `H/4 × W/4 × 8` |
| Box Constructor | Heatmap peaks + 3D params + K | 3D Bounding Boxes | `N × 7` (per object) |

---

## 4. Component Deep Dive

### 4.1 Backbone: Modified DLA-34 with Deformable Convolutions

#### What is DLA-34?

DLA-34 is the feature extraction backbone. It processes the raw image to produce rich, multi-scale feature representations.

**Why DLA-34 and not ResNet or VGG?**

| Feature | ResNet | VGG | DLA-34 |
|---------|--------|-----|--------|
| Multi-scale aggregation | ❌ Only skip connections | ❌ None | ✅ IDA + HDA |
| Parameter efficiency | ✅ | ❌ | ✅ |
| Fine-grained features | Limited | Limited | ✅ Strong |
| Suitable for dense prediction | Moderate | Poor | ✅ Excellent |

#### Deformable Convolutions (DCN) — The Modification

SMOKE replaces **all hierarchical aggregation nodes** in DLA-34 with **Deformable Convolutional Networks (DCNv2)**.

**Standard Convolution vs. Deformable Convolution:**

```
Standard Convolution (3×3):            Deformable Convolution (3×3):
  ● ● ●                                  ●   ●     ●
  ● ● ●  (fixed grid)                      ●  ●  ●
  ● ● ●                                 ●    ●   ●
                                         (learned offsets — adapts to shape!)
```

**Why Deformable Convolutions?**
- Objects appear at different **scales and aspect ratios**
- Cars seen from different **viewpoints** have varying shapes
- DCN learns to **adapt its sampling locations** to the object geometry
- This improves feature extraction for irregularly shaped objects

**How DCN Works (Mathematical Detail):**

```
Standard Conv at location p₀:
  y(p₀) = Σ w(pₙ) · x(p₀ + pₙ)        for pₙ in fixed grid R

Deformable Conv:
  y(p₀) = Σ w(pₙ) · x(p₀ + pₙ + Δpₙ)  for pₙ in R
                               ↑
                      learned offset (fractional, uses bilinear interpolation)

DCNv2 adds modulation:
  y(p₀) = Σ w(pₙ) · x(p₀ + pₙ + Δpₙ) · Δmₙ
                                          ↑
                              learned modulation scalar [0,1]
                              (controls importance of each sample point)
```

#### Backbone Output

The backbone produces a feature map at **1/4 resolution**:
- Input: `1242 × 375 × 3` (KITTI image)
- Output: `310 × 93 × 256` (feature channels)

This 4× downsampling balances **spatial resolution** (needed for accurate keypoint localization) with **computational efficiency**.

### 4.2 Keypoint Classification Head

This branch predicts **where objects are** in the image.

#### Architecture

```
Feature Map (H/4 × W/4 × 256)
    │
    ▼
Conv2D (3×3, 256 → 256) + ReLU + BN
    │
    ▼
Conv2D (1×1, 256 → C)
    │
    ▼
Sigmoid Activation
    │
    ▼
Heatmap (H/4 × W/4 × C)

Where C = number of object classes (e.g., 3 for Car, Pedestrian, Cyclist)
```

#### Ground Truth Heatmap Generation

For each object with projected 3D center at pixel `(cx, cy)`:

```
Ground truth heatmap value at location (i, j):

  Y(i,j) = exp( -(i - cx̃)² + (j - cỹ)²  )
            ─────────────────────────────────
                       2σ²

Where:
  (cx̃, cỹ) = downsampled center (cx/4, cy/4)
  σ         = radius proportional to object size
              (uses corner-based radius computation)

This creates a soft Gaussian "blob" centered at each object:

  0.0  0.0  0.1  0.3  0.1  0.0  0.0
  0.0  0.1  0.5  0.8  0.5  0.1  0.0
  0.1  0.5  0.9  1.0  0.9  0.5  0.1   ← peak = object center
  0.0  0.1  0.5  0.8  0.5  0.1  0.0
  0.0  0.0  0.1  0.3  0.1  0.0  0.0
```

#### Why Projected 3D Center, Not 2D Box Center?

```
Side View of a Car:
                                    ┌─────────────────────┐
                                    │    2D Bounding Box   │
  3D Bounding Box:                  │                      │
  ┌───────────┐                     │  ★ = 3D center proj  │
  │     ★     │ ──── projects to ──→│                      │
  │  (true    │                     │  ● = 2D box center   │
  │  center)  │                     │                      │
  └───────────┘                     └─────────────────────┘

The ★ and ● can be significantly different!
Using ★ gives geometrically correct 3D reconstruction.
Using ● introduces systematic bias.
```

### 4.3 3D Regression Head

This branch predicts **what the 3D properties are** for each detected keypoint.

#### Architecture

```
Feature Map (H/4 × W/4 × 256)
    │
    ▼
Conv2D (3×3, 256 → 256) + ReLU + BN
    │
    ▼
Conv2D (1×1, 256 → 8)
    │
    ▼
8-channel output (H/4 × W/4 × 8)
```

#### The 8 Regression Parameters

For each pixel location on the feature map, the network outputs 8 values:

| Parameter | Symbol | Description | Encoding |
|-----------|--------|-------------|----------|
| Depth offset | `δ_z` | Distance from camera | `z = 1/(σ(δ_z) + ε)` — ensures positive depth |
| Keypoint offset X | `δ_x` | Sub-pixel offset for center | Added to quantized center |
| Keypoint offset Y | `δ_y` | Sub-pixel offset for center | Added to quantized center |
| Dimension residual H | `δ_h` | Height relative to class mean | `h = h̄ · e^(δ_h)` |
| Dimension residual W | `δ_w` | Width relative to class mean | `w = w̄ · e^(δ_w)` |
| Dimension residual L | `δ_l` | Length relative to class mean | `l = l̄ · e^(δ_l)` |
| Orientation sin | `sin(θ)` | Yaw angle sine component | `θ = arctan2(sin θ, cos θ)` |
| Orientation cos | `cos(θ)` | Yaw angle cosine component | Avoids angle discontinuity |

#### Why These Specific Encodings?

**Depth encoding:** `z = 1/(σ(δ_z) + ε)`
- Sigmoid `σ` bounds output to (0, 1)
- Inverse mapping ensures depth is always **positive**
- Better gradient behavior than direct regression for large depth values

**Dimension encoding:** `h = h̄ · e^(δ_h)`
- `h̄` is the **class-specific mean dimension** from training data (e.g., average car height = 1.53m)
- The network only learns the **residual** `δ_h` (small deviation from mean)
- Exponential ensures dimensions are always **positive**
- Much easier to learn small residuals than absolute values

**Orientation encoding:** `sin(θ), cos(θ)` instead of `θ` directly
- The angle `θ` has a **discontinuity** at ±π (wraps around)
- `sin` and `cos` are **continuous** — no gradient explosion at boundaries
- `arctan2` recovers the angle unambiguously from both components

### 4.4 3D Bounding Box Construction

Given a detected keypoint and its 8 regression parameters, SMOKE builds the 3D box:

```
Step 1: Get keypoint location
  (u, v) = peak location on heatmap + sub-pixel offset (δ_x, δ_y)

Step 2: Recover 3D center location
  Given depth z and camera matrix K:
  ┌ X ┐       ┌ fx  0  cx ┐⁻¹   ┌ u·z ┐
  │ Y │ = K⁻¹ │ 0  fy  cy │   × │ v·z │
  └ Z ┘       └ 0   0   1 ┘     └  z  ┘

  X = (u - cx) · z / fx
  Y = (v - cy) · z / fy
  Z = z

Step 3: Recover dimensions
  h = h̄ · exp(δ_h)
  w = w̄ · exp(δ_w)
  l = l̄ · exp(δ_l)

Step 4: Recover orientation
  θ = arctan2(sin θ, cos θ)

Step 5: Construct 8 corners of the 3D box
  Using rotation matrix R(θ) and dimensions (h, w, l),
  compute corners relative to center, then translate
  to world position (X, Y, Z).

Step 6: Project 8 corners back to 2D
  For each corner (Xc, Yc, Zc):
  uc = fx · Xc/Zc + cx
  vc = fy · Yc/Zc + cy
  
  → 8 corners × 2 coords = 16 projected values
```

---

## 5. Loss Functions

### 5.1 Keypoint Classification Loss

Uses **penalty-reduced focal loss** (from CornerNet/CenterNet):

```
For each pixel (i,j) on heatmap:

If Y(i,j) = 1 (positive — object center):
  L_cls = -(1 - Ŝ(i,j))^α · log(Ŝ(i,j))

If Y(i,j) < 1 (negative — background or near object):
  L_cls = -(1 - Y(i,j))^β · Ŝ(i,j)^α · log(1 - Ŝ(i,j))

Where:
  Ŝ(i,j) = predicted score
  Y(i,j)  = ground truth (Gaussian-splattered)
  α = 2    (focal loss exponent)
  β = 4    (penalty reduction for near-center points)
```

**Why penalty-reduced?** Points near the object center (where `Y(i,j)` is close to 1 but not exactly 1) should receive **less penalty** for predicting positive — they're almost correct. The `(1 - Y(i,j))^β` term handles this.

The total keypoint loss is normalized by the number of objects `N`:

```
L_keypoint = (1/N) · Σ L_cls(i,j)
```

### 5.2 3D Regression Loss: Disentangled L1 Loss

This is a **key contribution** of the paper.

#### The Problem with Naive Regression

If you directly regress all 8 parameters with an L1 loss:
```
L_naive = |δ_z - δ_z*| + |δ_h - δ_h*| + ... + |cos θ - cos θ*|
```

**Problem:** Different parameters have different scales and different impacts on the final 3D box. A small error in depth has a MUCH larger effect on 3D box accuracy than a small error in dimensions. This makes training unstable.

#### The Disentangling Solution

**Core idea:** Instead of regressing parameters independently, regress the **final 8 projected corners** of the 3D box, but compute the loss in a **disentangled** manner.

**Step 1: Group the 8 parameters**

```
Group 1: [δ_z]           — Depth
Group 2: [δ_x, δ_y, δ_h, δ_w, δ_l]  — Location offset + Dimensions
Group 3: [sin θ, cos θ]  — Orientation
```

**Step 2: For each group, compute a separate 3D box**

```
Box from Group 1: Use predicted δ_z + ground-truth for Groups 2 & 3
Box from Group 2: Use predicted δ_x,δ_y,δ_h,δ_w,δ_l + ground-truth for Groups 1 & 3
Box from Group 3: Use predicted sin θ, cos θ + ground-truth for Groups 1 & 2
```

**Step 3: Compute L1 loss on projected corners for each box**

```
L_group_k = (1/16) · Σ |corner_pred - corner_gt|    (16 = 8 corners × 2D coords)
```

**Step 4: Average across groups**

```
L_regression = (1/3) · (L_group_1 + L_group_2 + L_group_3)
```

#### Why Disentangling Works

```
Without Disentangling:                  With Disentangling:
┌─────────────────────────┐            ┌─────────────────────────┐
│ All 8 params jointly    │            │ Group 1: Only depth     │
│ affect corner positions  │            │ affects corner error    │
│                         │            │                         │
│ Gradient is MIXED:      │            │ Gradient is CLEAN:      │
│ depth error ← affects → │            │ depth error → only      │
│ dimension gradient      │            │ updates depth params    │
│ and vice versa          │            │                         │
│                         │            │ Training converges      │
│ Training is UNSTABLE    │            │ FASTER and BETTER       │
└─────────────────────────┘            └─────────────────────────┘
```

### 5.3 Total Loss

```
L_total = L_keypoint + L_regression

No weighting needed — the disentangling naturally balances the contributions!
```

---

## 6. Training & Inference Pipeline

### 6.1 Training Details

| Setting | Value |
|---------|-------|
| **Dataset** | KITTI 3D Object Detection |
| **Input Resolution** | 1242 × 375 (original KITTI) |
| **Backbone** | DLA-34 (pretrained on ImageNet) |
| **Optimizer** | Adam (lr = 2.5 × 10⁻⁴) |
| **Batch Size** | 8 (on 4 GPUs) |
| **Training Epochs** | 72 |
| **Learning Rate Schedule** | Decay at epochs 36 and 54 |
| **Data Augmentation** | Random horizontal flip, random scaling, color jitter |
| **Normalization** | Group Normalization (GN), not Batch Normalization |

### 6.2 Data Handling

```
KITTI Dataset Structure:
├── training/
│   ├── image_2/        ← RGB images (left camera)
│   ├── label_2/        ← 3D annotations (class, dims, location, rotation)
│   └── calib/          ← Camera calibration matrices (K matrix)
└── testing/
    ├── image_2/
    └── calib/

Important: Objects whose 3D centers project outside the image boundary
are discarded (~5% of all objects). This is a design choice for training
stability.
```

### 6.3 Inference Pipeline

```
Input Image
    │
    ▼
┌──────────────────┐
│ DLA-34 Backbone  │ ← Single forward pass (~30ms total)
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Heatmap   Regression
    │         │
    ▼         │
 ┌────────┐   │
 │ Extract│   │
 │ Top-K  │   │  ← Take top 50 peaks (no NMS needed!)
 │ peaks  │   │
 └───┬────┘   │
     │        │
     ▼        ▼
 ┌─────────────────┐
 │ Gather matching │
 │ regression vals │
 │ at peak locs    │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Construct 3D    │
 │ bounding boxes  │
 │ (no NMS or      │
 │  post-proc!)    │
 └────────┬────────┘
          │
          ▼
 3D Bounding Boxes per object
```

**No NMS (Non-Maximum Suppression):** Since each object produces exactly one heatmap peak, there are no duplicate detections to suppress. This simplifies inference significantly.

### 6.4 Why No Post-Processing?

| Method | Requires NMS? | Requires Refinement? | Extra Data? |
|--------|:---:|:---:|:---:|
| M3D-RPN | ✅ | ✅ | ❌ |
| MonoDIS | ✅ | ❌ | ❌ |
| Deep3DBox | ✅ | ✅ | ❌ |
| Pseudo-LiDAR | ✅ | ✅ | ✅ (depth maps) |
| **SMOKE** | **❌** | **❌** | **❌** |

---

## 7. Results & Benchmarks

### 7.1 KITTI 3D Object Detection (Car Class)

AP_{3D} at IoU = 0.7:

| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| Deep3DBox | 5.85 | 4.10 | 3.84 |
| MonoPSR | 12.75 | 11.48 | 8.59 |
| MonoDIS | 11.06 | 7.60 | 6.37 |
| M3D-RPN | 14.76 | 9.71 | 7.42 |
| **SMOKE** | **14.03** | **9.76** | **7.84** |

### 7.2 Birds-Eye View (BEV) Evaluation

AP_{BEV} at IoU = 0.7:

| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| Deep3DBox | 9.99 | 7.71 | 5.30 |
| MonoDIS | 18.45 | 12.58 | 10.66 |
| M3D-RPN | 21.02 | 13.67 | 10.23 |
| **SMOKE** | **20.83** | **14.49** | **12.75** |

### 7.3 Key Performance Metrics

| Metric | Value |
|--------|-------|
| **Runtime** | ~30ms per frame |
| **FPS** | ~33 FPS (real-time!) |
| **GPU** | NVIDIA TITAN Xp |
| **Depth error** | < 5% (~3m at 60m distance) |
| **No extra data** | Only monocular RGB + calibration |

---

## 8. Setup & Usage

### 8.1 Environment Requirements

```bash
# Tested environment
Ubuntu 16.04 / 18.04
Python 3.7+
PyTorch 1.3.1+
CUDA 10.0+
```

### 8.2 Installation

```bash
# Clone the repository
git clone https://github.com/lzccccc/SMOKE.git
cd SMOKE

# Install dependencies
pip install -r requirements.txt

# Install SMOKE package
python setup.py develop
```

### 8.3 Dataset Preparation (KITTI)

```bash
# Download KITTI 3D Object Detection dataset
# from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

# Organize data as:
SMOKE/
└── datasets/
    └── kitti/
        ├── training/
        │   ├── image_2/       # Left color images
        │   ├── label_2/       # Training labels
        │   └── calib/         # Calibration files
        └── testing/
            ├── image_2/
            └── calib/
```

### 8.4 Training

```bash
# Single GPU
python tools/plain_train_net.py --config-file configs/smoke_gn_vector.yaml

# Multi-GPU (4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/plain_train_net.py \
    --config-file configs/smoke_gn_vector.yaml
```

### 8.5 Evaluation

```bash
# Run evaluation
python tools/plain_train_net.py \
    --eval-only \
    --config-file configs/smoke_gn_vector.yaml \
    MODEL.WEIGHT path/to/checkpoint.pth
```

---

## 9. Key Takeaways

### What Makes SMOKE Special?

| Innovation | Impact |
|-----------|--------|
| **Single-stage** (no 2D detection) | Removes noise, simplifies pipeline |
| **Projected 3D center as keypoint** | Geometrically correct representation |
| **Disentangled L1 loss** | Faster convergence, better accuracy |
| **No NMS, no post-processing** | Clean, simple inference |
| **8-corner unified regression** | All 3D params trained jointly via geometry |

### Limitations

1. **Requires camera calibration** — the intrinsic matrix `K` must be known
2. **Occlusion handling** is limited — heavily occluded objects may have poor 3D center projection
3. **Depth accuracy decreases** with distance — inherent to monocular methods
4. **KITTI-specific** — performance on other datasets may vary
5. **No temporal information** — single-frame, doesn't exploit video sequences

### Comparison with Other Paradigms

```
┌──────────────────────────────────────────────────────────────────┐
│                 3D Object Detection Methods                      │
├──────────────────────┬───────────────────────────────────────────┤
│  LiDAR-based         │  Most accurate, but expensive sensors    │
│  (PointPillars, etc.)│  directly process 3D point clouds        │
├──────────────────────┼───────────────────────────────────────────┤
│  Stereo-based        │  Two cameras → compute depth via         │
│  (Pseudo-LiDAR)      │  disparity → 3D detection                │
├──────────────────────┼───────────────────────────────────────────┤
│  Monocular Two-stage │  2D detect → crop → 3D estimate          │
│  (Deep3DBox, etc.)   │  (redundant 2D step)                     │
├──────────────────────┼───────────────────────────────────────────┤
│  Monocular One-stage │  Direct 3D from single image             │
│  ★ SMOKE ★           │  (simplest, fastest, competitive)        │
└──────────────────────┴───────────────────────────────────────────┘
```

---

## 10. References

1. **SMOKE Paper:** Liu, Z., Wu, Z., & Tóth, R. (2020). *SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation.* CVPRW 2020.
2. **CenterNet:** Zhou, X., Wang, D., & Krähenbühl, P. (2019). *Objects as Points.* arXiv:1904.07850.
3. **DLA:** Yu, F., et al. (2018). *Deep Layer Aggregation.* CVPR 2018.
4. **Deformable ConvNets v2:** Zhu, X., et al. (2019). *Deformable ConvNets v2.* CVPR 2019.
5. **Focal Loss:** Lin, T.Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
6. **MonoDIS:** Simonelli, A., et al. (2019). *Disentangling Monocular 3D Object Detection.* ICCV 2019.
7. **KITTI Benchmark:** Geiger, A., et al. (2012). *Are We Ready for Autonomous Driving?* CVPR 2012.

---

## 📄 Citation

```bibtex
@inproceedings{liu2020smoke,
  title     = {SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation},
  author    = {Liu, Zechen and Wu, Zizhang and T{\'o}th, Roland},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2020}
}
```

---

## 📊 Architecture Diagram Summary

```
                    ┌──────────────────────────────────────┐
                    │         Single RGB Image             │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │   Modified DLA-34 + DCN Backbone     │
                    │   (ImageNet pretrained, 4× downsample)│
                    └──────────────────┬───────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
               ┌──────────▼──────────┐   ┌─────────▼──────────┐
               │  Keypoint Branch    │   │  Regression Branch  │
               │                     │   │                     │
               │  Conv → Conv → σ    │   │  Conv → Conv        │
               │                     │   │                     │
               │  Output: C-channel  │   │  Output: 8-channel  │
               │  heatmap            │   │  3D parameters      │
               └──────────┬──────────┘   └─────────┬──────────┘
                          │                         │
                          │  ┌──────────────────┐   │
                          └──│  Peak Detection   │───┘
                             │  (Top-K, no NMS)  │
                             └────────┬─────────┘
                                      │
                             ┌────────▼─────────┐
                             │  Back-projection  │
                             │  + 3D Box Build   │
                             │  (using K matrix) │
                             └────────┬─────────┘
                                      │
                             ┌────────▼─────────┐
                             │   3D Bounding     │
                             │   Boxes Output    │
                             └──────────────────┘
```
