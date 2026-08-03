import pytest
import torch

from internnav.model.basemodel.memnav.gate_fusion import (
    branch_log_weights,
    gate_fusion_code,
)
from internnav.model.basemodel.memnav.retrieval_losses import (
    multi_positive_retrieval_losses,
)


def test_complementary_fusion_is_exact_historical_mask():
    gate = torch.tensor([0.2, 0.8], requires_grad=True)
    revisit, visual = branch_log_weights(gate, gate_fusion_code("complementary"))
    assert torch.allclose(revisit, torch.log(gate))
    assert torch.allclose(visual, torch.log1p(-gate))


def test_residual_fusion_preserves_visual_and_gate_gradient():
    gate = torch.tensor([0.2, 0.8], requires_grad=True)
    revisit, visual = branch_log_weights(gate, gate_fusion_code("residual"))
    assert torch.allclose(revisit, torch.log(gate))
    assert torch.equal(visual, torch.zeros_like(gate))

    revisit.sum().backward()
    assert torch.allclose(gate.grad, gate.reciprocal())


def test_gate_fusion_rejects_unknown_mode_and_nonscalar_code():
    with pytest.raises(ValueError):
        gate_fusion_code("hard-switch")
    with pytest.raises(ValueError):
        branch_log_weights(torch.tensor([0.5]), torch.tensor([0.0, 1.0]))


def test_set_loss_matches_original_multi_positive_objective():
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, -0.5, 1.5]])
    pos = torch.tensor([[True, True, False], [False, True, False]])
    neg = torch.tensor([[False, False, True], [True, False, True]])
    set_loss, _top1, valid = multi_positive_retrieval_losses(logits, pos, neg)

    expected = (
        logits.logsumexp(-1)
        - logits.masked_fill(~pos, torch.finfo(logits.dtype).min).logsumexp(-1)
    ).mean()
    assert valid.tolist() == [True, True]
    assert set_loss == pytest.approx(expected.item())


def test_top1_margin_targets_live_argmax_and_has_correct_gradient():
    logits = torch.tensor([[0.0, 1.0, 0.7], [0.8, 0.0, 0.3]], requires_grad=True)
    pos = torch.tensor([[True, False, False], [True, False, False]])
    neg = ~pos
    _set_loss, top1_loss, _valid = multi_positive_retrieval_losses(
        logits, pos, neg, top1_margin=0.2
    )
    # Row 0 violates the margin by 1.2; row 1 already has exactly 0.5 margin.
    assert top1_loss.item() == pytest.approx(0.6)
    top1_loss.backward()
    assert logits.grad[0, 0] < 0       # optimizer raises the best positive
    assert logits.grad[0, 1] > 0       # optimizer lowers the best negative
    assert torch.equal(logits.grad[1], torch.zeros(3))


def test_empty_retrieval_rows_return_finite_differentiable_zero():
    logits = torch.randn(2, 4, requires_grad=True)
    pos = torch.zeros(2, 4, dtype=torch.bool)
    neg = torch.ones(2, 4, dtype=torch.bool)
    set_loss, top1_loss, valid = multi_positive_retrieval_losses(logits, pos, neg)
    assert not valid.any()
    assert set_loss.item() == 0.0
    assert top1_loss.item() == 0.0
    (set_loss + top1_loss).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
