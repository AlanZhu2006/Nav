import numpy as np
import torch

from MemNavData.train_pi3x_spatial_reliability_crossfit_oof import (
    Pi3XSpatialReliabilityHead,
    _fixed_proposal_picks,
)


def test_spatial_head_shapes_and_padding() -> None:
    model = Pi3XSpatialReliabilityHead(
        8, model_dim=16, layers=1, heads=4
    ).eval()
    batch, views = 3, 4
    descriptors = torch.randn(batch, views, 8)
    roles = torch.tensor([[0, 1, 2, -1]] * batch)
    age = torch.tensor([[0.0, 0.5, 1.0, 0.0]] * batch)
    valid = roles >= 0
    world = torch.randn(batch, views, 3, 4, 3)
    local = torch.randn(batch, views, 3, 4, 3)
    confidence = torch.rand(batch, views, 3, 4, 1)
    poses = torch.randn(batch, views, 3, 4)
    with torch.inference_mode():
        action, support = model(
            descriptors, roles, age, valid,
            world, local, confidence, poses,
        )
    assert action.shape == support.shape == (batch,)
    assert torch.isfinite(action).all()


def test_fixed_proposal_uses_overlap_not_learned_score() -> None:
    rows = [
        {"session_id": "a", "scene": "s", "session_label": 1,
         "row_index": 0, "candidate_rank": 0, "candidate_label": 0,
         "navigation_action_label": 0, "bearing_error_deg": 80.0,
         "raw_pi3x_overlap": 0.9},
        {"session_id": "a", "scene": "s", "session_label": 1,
         "row_index": 1, "candidate_rank": 1, "candidate_label": 1,
         "navigation_action_label": 1, "bearing_error_deg": 2.0,
         "raw_pi3x_overlap": 0.2},
    ]
    picks = _fixed_proposal_picks(rows, {0: 0.1, 1: 0.99}, {"s"})
    assert picks[0]["selected_row_index"] == 0
    assert picks[0]["score"] == 0.1
