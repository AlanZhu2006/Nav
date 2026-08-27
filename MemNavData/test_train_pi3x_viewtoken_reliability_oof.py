import numpy as np
import torch

from MemNavData.train_pi3x_viewtoken_reliability_oof import (
    ViewTokenReliabilityHead,
    _picks,
)


def test_viewtoken_head_masks_padding_and_returns_two_logits() -> None:
    model = ViewTokenReliabilityHead(8, model_dim=16, layers=1, heads=4).eval()
    descriptors = torch.randn(3, 5, 8)
    roles = torch.tensor([[0, 1, 2, 3, -1]] * 3)
    ages = torch.tensor([[0.0, 0.3, 1.0, -1.0, 0.0]] * 3)
    valid = roles >= 0
    with torch.inference_mode():
        action, support = model(descriptors, roles, ages, valid)
    assert action.shape == support.shape == (3,)
    assert torch.isfinite(action).all()


def test_picks_selects_highest_score_with_rank_tiebreak() -> None:
    rows = [
        {"session_id": "a", "scene": "s", "session_label": 1,
         "candidate_rank": 1, "candidate_label": 1,
         "navigation_action_label": 1, "bearing_error_deg": 2.0,
         "row_index": 0},
        {"session_id": "a", "scene": "s", "session_label": 1,
         "candidate_rank": 0, "candidate_label": 0,
         "navigation_action_label": 0, "bearing_error_deg": 80.0,
         "row_index": 1},
    ]
    picks = _picks(rows, {0: 0.5, 1: 0.5})
    assert picks[0]["selected_row_index"] == 1
