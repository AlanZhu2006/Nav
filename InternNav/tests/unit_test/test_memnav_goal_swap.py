import pytest
import torch

from internnav.model.basemodel.memnav.goal_swap import goal_swap_margin_metrics


def test_goal_swap_hinge_is_zero_after_margin():
    target = torch.zeros(2, 2, 1)
    correct = torch.zeros_like(target, requires_grad=True)
    swapped = torch.full_like(target, 0.5, requires_grad=True)

    metrics = goal_swap_margin_metrics(
        correct, swapped, target, torch.tensor([True, False]), margin=0.1
    )

    assert metrics["loss"].item() == pytest.approx(0.0)
    assert metrics["error_gap"].item() == pytest.approx(0.25)
    assert metrics["output_rms"].item() == pytest.approx(0.5)
    assert metrics["active_count"].item() == 1


def test_goal_swap_hinge_penalizes_goal_ignoring_and_backpropagates():
    target = torch.zeros(1, 3, 2)
    correct = torch.zeros_like(target, requires_grad=True)
    swapped = torch.full_like(target, 0.1, requires_grad=True)

    metrics = goal_swap_margin_metrics(
        correct, swapped, target, torch.tensor([True]), margin=0.05
    )
    metrics["loss"].backward()

    assert metrics["loss"].item() == pytest.approx(0.04)
    assert metrics["error_gap"].item() == pytest.approx(0.01)
    assert swapped.grad is not None
    assert torch.isfinite(swapped.grad).all()
    assert swapped.grad.abs().sum().item() > 0


def test_goal_swap_empty_mask_returns_differentiable_zero():
    target = torch.zeros(2, 2, 1)
    correct = torch.randn_like(target, requires_grad=True)
    swapped = torch.randn_like(target, requires_grad=True)

    metrics = goal_swap_margin_metrics(
        correct, swapped, target, torch.zeros(2, dtype=torch.bool), margin=0.05
    )
    metrics["loss"].backward()

    assert metrics["loss"].item() == 0.0
    assert correct.grad is not None
    assert swapped.grad is not None
    assert correct.grad.abs().sum().item() == 0.0
    assert swapped.grad.abs().sum().item() == 0.0


def test_goal_swap_rejects_invalid_shapes_and_margin():
    x = torch.zeros(2, 3, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        goal_swap_margin_metrics(x, x[:1], x, torch.ones(2), 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        goal_swap_margin_metrics(x, x, x, torch.ones(2), -0.1)
