"""Counterfactual goal-conditioning objective for MemNav's Novel branch.

The expert action belongs to the real image goal.  A different image goal has no
action label, so we do not invent one.  Instead, with current state, diffusion
noise and timestep held fixed, require the real goal to explain its expert noise
target better than the swapped goal by a small margin.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def goal_swap_margin_metrics(
    correct_prediction: torch.Tensor,
    swapped_prediction: torch.Tensor,
    target_noise: torch.Tensor,
    active_mask: torch.Tensor,
    margin: float,
) -> dict[str, torch.Tensor]:
    """Return a bounded ranking loss and diagnostics for counterfactual goals.

    All predictions have shape ``[B, T, A]``. ``active_mask`` selects Novel rows
    whose swapped image is genuinely different.  The desired inequality is::

        mse(swapped_goal, target) - mse(correct_goal, target) >= margin

    The hinge becomes zero once that inequality holds, preventing an incentive
    to make the counterfactual output grow without bound.
    """
    if correct_prediction.shape != swapped_prediction.shape:
        raise ValueError(
            "correct/swapped prediction shape mismatch: "
            f"{tuple(correct_prediction.shape)} != {tuple(swapped_prediction.shape)}"
        )
    if correct_prediction.shape != target_noise.shape:
        raise ValueError(
            "prediction/target shape mismatch: "
            f"{tuple(correct_prediction.shape)} != {tuple(target_noise.shape)}"
        )
    if correct_prediction.ndim < 2:
        raise ValueError("goal-swap predictions must include batch and feature axes")
    if margin < 0:
        raise ValueError(f"goal-swap margin must be non-negative, got {margin}")

    reduce_dims = tuple(range(1, correct_prediction.ndim))
    correct_error = (correct_prediction - target_noise).square().mean(dim=reduce_dims)
    swapped_error = (swapped_prediction - target_noise).square().mean(dim=reduce_dims)
    error_gap = swapped_error - correct_error
    output_rms = (swapped_prediction - correct_prediction).square().mean(
        dim=reduce_dims
    ).sqrt()

    active = active_mask.to(device=correct_prediction.device, dtype=torch.bool).reshape(-1)
    if active.numel() != correct_prediction.shape[0]:
        raise ValueError(
            f"active_mask has {active.numel()} rows for batch {correct_prediction.shape[0]}"
        )
    active_count = active.sum()
    if bool(active_count.item()):
        loss = F.relu(float(margin) - error_gap[active]).mean()
        mean_gap = error_gap[active].mean()
        mean_rms = output_rms[active].mean()
    else:
        # Preserve a valid zero-gradient graph for DDP batches with no Novel row.
        loss = (correct_prediction.sum() + swapped_prediction.sum()) * 0.0
        mean_gap = error_gap.new_zeros(())
        mean_rms = output_rms.new_zeros(())

    return {
        "loss": loss,
        "error_gap": mean_gap,
        "output_rms": mean_rms,
        "active_count": active_count,
        "correct_error": correct_error,
        "swapped_error": swapped_error,
    }
