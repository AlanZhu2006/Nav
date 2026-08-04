"""Unit tests for the decoder-gate fusion math + logit-space teacher curriculum."""

import pytest
import torch

from internnav.model.basemodel.memnav.decgate_schedule import (
    DECGATE_FUSIONS, Z_CLAMP, blend_decoder_gate_logit, branch_bias_values,
    decgate_fusion_code, linear_teacher_ratio)


def test_fusion_codes_roundtrip():
    for i, mode in enumerate(DECGATE_FUSIONS):
        assert decgate_fusion_code(mode) == float(i)
    with pytest.raises(ValueError):
        decgate_fusion_code("complementary")


def test_symmetric_bias_zero_common_mode():
    z = torch.tensor([-5.0, 0.0, 3.0])
    rev, nov = branch_bias_values(z, "symmetric")
    assert torch.allclose(rev + nov, torch.zeros_like(z))       # zero common mode
    assert torch.allclose(rev - nov, z)                         # difference == logit


def test_residual_bias_leaves_novel_untouched():
    z = torch.tensor([-5.0, 0.0, 3.0])
    rev, nov = branch_bias_values(z, "residual")
    assert torch.allclose(nov, torch.zeros_like(z))
    assert torch.allclose(rev, z)


def test_bias_clamped_at_rail():
    z = torch.tensor([-100.0, 100.0])
    rev, _ = branch_bias_values(z, "residual")
    assert torch.allclose(rev, torch.tensor([-Z_CLAMP, Z_CLAMP]))


def test_value_scale_has_no_bias_path():
    with pytest.raises(ValueError):
        branch_bias_values(torch.zeros(2), "value_scale")


def test_teacher_ratio_schedule():
    assert linear_teacher_ratio(0, 1.0, 0.0, 500) == 1.0
    assert linear_teacher_ratio(250, 1.0, 0.0, 500) == pytest.approx(0.5)
    assert linear_teacher_ratio(500, 1.0, 0.0, 500) == 0.0
    assert linear_teacher_ratio(10_000, 1.0, 0.0, 500) == 0.0   # clamped past the end
    assert linear_teacher_ratio(123, 1.0, 0.0, 0) == 0.0        # steps=0 == disabled
    with pytest.raises(ValueError):
        linear_teacher_ratio(0, 1.5, 0.0, 500)
    with pytest.raises(ValueError):
        linear_teacher_ratio(0, 1.0, 0.0, -1)


def test_blend_endpoints_and_gradient_handover():
    z_pred = torch.tensor([-4.0, -1.0, 2.0], requires_grad=True)
    is_rev = torch.tensor([1.0, 0.0, 1.0])
    # r=0: identical object semantics — exactly the inference path
    assert blend_decoder_gate_logit(z_pred, is_rev, 0.0, 3.0) is z_pred
    # r=1: pure GT routing at ±teacher_z
    z1 = blend_decoder_gate_logit(z_pred, is_rev, 1.0, 3.0)
    assert torch.allclose(z1, torch.tensor([3.0, -3.0, 3.0]))
    # mid-blend: gradient to z_pred is scaled by exactly (1 - r)
    r = 0.7
    zm = blend_decoder_gate_logit(z_pred, is_rev, r, 3.0)
    zm.sum().backward()
    assert torch.allclose(z_pred.grad, torch.full_like(z_pred, 1.0 - r))
    with pytest.raises(ValueError):
        blend_decoder_gate_logit(z_pred, is_rev, 1.5, 3.0)
    with pytest.raises(ValueError):
        blend_decoder_gate_logit(z_pred, is_rev, 0.5, 0.0)
