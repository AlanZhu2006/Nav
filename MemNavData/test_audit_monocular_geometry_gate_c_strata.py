import torch

from MemNavData.audit_monocular_geometry_gate_c_strata import (
    scale_stratum,
    selected_candidate_disagreement,
)


def test_scale_stratum_is_fail_closed_and_disjoint():
    assert scale_stratum({"scale_valid": False, "scale_clamped": True}) == "scale_invalid"
    assert scale_stratum({"scale_valid": True, "scale_clamped": True}) == "scale_valid_clamped"
    assert scale_stratum({"scale_valid": True, "scale_clamped": False}) == "scale_valid_unclamped"


def test_selected_candidate_disagreement_uses_cumulative_planar_endpoint():
    candidates = torch.zeros(2, 2, 4, 3)
    candidates[:, 0, :, 0] = 1.0
    candidates[:, 1, :, 1] = 1.0
    teacher = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    student = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    metrics = selected_candidate_disagreement(candidates, student, teacher)
    assert metrics["critic_top1_agreement"].tolist() == [1.0, 1.0]
    assert metrics["selected_endpoint_l2_m"].tolist() == [0.0, 0.0]
    assert metrics["selected_heading_abs_error_deg"].tolist() == [0.0, 0.0]

    swapped = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    metrics = selected_candidate_disagreement(candidates, swapped, teacher)
    assert metrics["critic_top1_agreement"].tolist() == [0.0, 0.0]
    assert torch.allclose(
        metrics["selected_endpoint_l2_m"], torch.tensor([2.0**0.5, 2.0**0.5])
    )
    assert torch.allclose(
        metrics["selected_heading_abs_error_deg"], torch.tensor([90.0, 90.0])
    )
