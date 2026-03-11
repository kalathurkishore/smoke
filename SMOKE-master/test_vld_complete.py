"""
VLD Architecture — Complete Test Suite
All 14 tests must print [PASS] before training.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"[PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        failed += 1


# ───────────────────────────────────────────────
# Test 1: VLD Head — Shape Validation
# ───────────────────────────────────────────────
def test_shape_validation():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    x = torch.randn(2, 64, 96, 320)
    heatmap, regression = model(x)

    assert heatmap.shape == (2, 1, 96, 320), f"Heatmap shape wrong: {heatmap.shape}"
    assert regression.shape == (2, 4, 96, 320), f"Regression shape wrong: {regression.shape}"


# ───────────────────────────────────────────────
# Test 2: Heatmap — Value Range
# ───────────────────────────────────────────────
def test_heatmap_range():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    x = torch.randn(2, 64, 96, 320)
    heatmap, _ = model(x)

    assert heatmap.min().item() >= 1e-4, f"Heatmap min too low: {heatmap.min()}"
    assert heatmap.max().item() <= 1 - 1e-4, f"Heatmap max too high: {heatmap.max()}"


# ───────────────────────────────────────────────
# Test 3: Heatmap — Bias Initialization
# ───────────────────────────────────────────────
def test_heatmap_bias():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    bias = model.heatmap_head[-1].bias.data
    assert torch.allclose(bias, torch.tensor([-2.19])), f"Bias wrong: {bias}"


# ───────────────────────────────────────────────
# Test 4: Regression — Dimensions Are Positive
# ───────────────────────────────────────────────
def test_regression_positive():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    x = torch.randn(2, 64, 96, 320)
    _, regression = model(x)

    dh_dw = regression[:, 2:4]
    assert (dh_dw > 0).all(), "Dimension channels must be positive"


# ───────────────────────────────────────────────
# Test 5: Regression — Exp Clamping
# ───────────────────────────────────────────────
def test_exp_clamping():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    extreme_input = torch.randn(1, 64, 10, 10) * 100
    _, r = model(extreme_input)
    assert torch.isfinite(r).all(), "Regression has inf/nan values!"


# ───────────────────────────────────────────────
# Test 6: Gradient Flow
# ───────────────────────────────────────────────
def test_gradient_flow():
    from smoke.modeling.heads.vld_head import VLDHead

    model = VLDHead(in_channels=64)
    x = torch.randn(2, 64, 96, 320)
    heatmap, regression = model(x)

    model.zero_grad()
    loss = heatmap.mean() + regression.mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient NaN/Inf for {name}"


# ───────────────────────────────────────────────
# Test 7: Focal Loss — Correctness
# ───────────────────────────────────────────────
def test_focal_loss_correctness():
    from smoke.modeling.loss.vld_loss import focal_loss

    pred = torch.tensor([[[[0.9999]]]])
    gt = torch.tensor([[[[1.0]]]])
    loss = focal_loss(pred, gt)
    assert loss.item() < 0.01, f"Perfect prediction should have near-zero loss: {loss}"

    pred_wrong = torch.tensor([[[[0.5]]]])
    loss_wrong = focal_loss(pred_wrong, gt)
    assert loss_wrong > loss, "Wrong prediction should have higher loss"


# ───────────────────────────────────────────────
# Test 8: Focal Loss — Negative Weighting
# ───────────────────────────────────────────────
def test_focal_loss_negative_weighting():
    from smoke.modeling.loss.vld_loss import focal_loss

    pred_bg = torch.tensor([[[[0.01]]]])
    gt_far = torch.tensor([[[[0.0]]]])
    loss_far = focal_loss(pred_bg, gt_far)

    gt_near = torch.tensor([[[[0.8]]]])
    loss_near = focal_loss(pred_bg, gt_near)

    assert loss_near < loss_far, "Near-center negatives should be down-weighted"


# ───────────────────────────────────────────────
# Test 9: Regression Loss — Mask Behavior
# ───────────────────────────────────────────────
def test_regression_loss_mask():
    from smoke.modeling.loss.vld_loss import regression_loss

    pred = torch.ones(1, 4, 10, 10) * 5.0
    gt = torch.zeros(1, 4, 10, 10)
    mask = torch.zeros(1, 1, 10, 10)
    mask[0, 0, 5, 5] = 1

    loss = regression_loss(pred, gt, mask)
    assert loss.item() > 0, "Loss should be nonzero at masked position"

    zero_mask = torch.zeros(1, 1, 10, 10)
    zero_loss = regression_loss(pred, gt, zero_mask)
    assert zero_loss.item() == 0, "Zero mask should give zero loss"


# ───────────────────────────────────────────────
# Test 10: Target Generation — Gaussian Heatmap
# ───────────────────────────────────────────────
def test_target_generation():
    from smoke.modeling.utils.vld_target import generate_vld_targets

    bboxes = np.array([[155.0, 43.0, 165.0, 53.0]])
    hm, reg, mask = generate_vld_targets(bboxes, output_size=(96, 320), stride=1)

    center_x, center_y = 160, 48
    assert hm[center_y, center_x] > 0.5, f"Heatmap peak too low: {hm[center_y, center_x]}"
    assert mask[0, center_y, center_x] == 1, "Mask should be 1 at center"


# ───────────────────────────────────────────────
# Test 11: Adaptive Gaussian Radius
# ───────────────────────────────────────────────
def test_adaptive_radius():
    from smoke.modeling.utils.heatmap import gaussian_radius

    r_small = gaussian_radius((5, 5))
    r_large = gaussian_radius((50, 50))
    assert r_large > r_small, f"Large obj should have larger radius: {r_small} vs {r_large}"


# ───────────────────────────────────────────────
# Test 12: Backbone Channel Compatibility
# ───────────────────────────────────────────────
def test_backbone_channels():
    from smoke.modeling.heads.vld_head import VLDHead

    # SMOKE default: BACKBONE_OUT_CHANNELS = 64
    model = VLDHead(in_channels=64)
    x = torch.randn(1, 64, 96, 320)
    h, r = model(x)
    assert h.shape[1] == 1, f"Heatmap channels wrong: {h.shape[1]}"
    assert r.shape[1] == 4, f"Regression channels wrong: {r.shape[1]}"


# ───────────────────────────────────────────────
# Test 13: Freeze Strategy
# ───────────────────────────────────────────────
def test_freeze_strategy():
    from smoke.modeling.heads.vld_head import VLDHead

    # Simulate a minimal model
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 64, 3)
            self.vld_head = VLDHead(in_channels=64)

    model = FakeModel()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.vld_head.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable > 0, "VLD head should have trainable parameters"
    assert trainable < total, "Not all parameters should be trainable"

    # Verify backbone is frozen
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone should be frozen"

    # Verify VLD head is trainable
    for p in model.vld_head.parameters():
        assert p.requires_grad, "VLD head should be trainable"


# ───────────────────────────────────────────────
# Test 14: Parallel Head Independence
# ───────────────────────────────────────────────
def test_parallel_heads():
    from smoke.modeling.heads.vld_head import VLDHead

    # Verify VLD head works independently
    vld = VLDHead(in_channels=64)
    feat = torch.randn(1, 64, 96, 320)

    hm1, reg1 = vld(feat)

    # Run again with different input
    feat2 = torch.randn(1, 64, 96, 320)
    hm2, reg2 = vld(feat2)

    # Outputs should differ for different inputs
    assert not torch.allclose(hm1, hm2), "Different inputs should give different heatmaps"
    assert not torch.allclose(reg1, reg2), "Different inputs should give different regressions"

    # Verify outputs are deterministic for same input
    hm3, reg3 = vld(feat)
    assert torch.allclose(hm1, hm3), "Same input should give same heatmap"
    assert torch.allclose(reg1, reg3), "Same input should give same regression"


# ───────────────────────────────────────────────
# Run all tests
# ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VLD Architecture — Complete Test Suite")
    print("=" * 60)
    print()

    run_test("Test  1: VLD Head Shape Validation", test_shape_validation)
    run_test("Test  2: Heatmap Value Range", test_heatmap_range)
    run_test("Test  3: Heatmap Bias Initialization", test_heatmap_bias)
    run_test("Test  4: Regression Dimensions Positive", test_regression_positive)
    run_test("Test  5: Exp Clamping Stability", test_exp_clamping)
    run_test("Test  6: Gradient Flow", test_gradient_flow)
    run_test("Test  7: Focal Loss Correctness", test_focal_loss_correctness)
    run_test("Test  8: Focal Loss Negative Weighting", test_focal_loss_negative_weighting)
    run_test("Test  9: Regression Loss Mask", test_regression_loss_mask)
    run_test("Test 10: Target Generation", test_target_generation)
    run_test("Test 11: Adaptive Gaussian Radius", test_adaptive_radius)
    run_test("Test 12: Backbone Channel Compatibility", test_backbone_channels)
    run_test("Test 13: Freeze Strategy", test_freeze_strategy)
    run_test("Test 14: Parallel Head Independence", test_parallel_heads)

    print()
    print("=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed}")
    print("=" * 60)

    if failed > 0:
        print("\n⚠️  FIX ALL FAILURES BEFORE TRAINING!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Ready for training.")
        sys.exit(0)
