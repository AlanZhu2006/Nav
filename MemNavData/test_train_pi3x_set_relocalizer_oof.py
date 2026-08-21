import torch

from MemNavData.train_pi3x_set_relocalizer_oof import (
    Pi3XSetRelocalizer,
    _multi_target_nll,
    _sessions,
)


def test_set_relocalizer_emits_k_plus_reject_and_support() -> None:
    model = Pi3XSetRelocalizer(
        8, model_dim=16, view_layers=1, set_layers=1, heads=4,
        max_candidates=3,
    ).eval()
    descriptors = torch.randn(2, 3, 5, 8)
    roles = torch.tensor([[[0, 1, 2, 3, -1]] * 3] * 2)
    age = torch.tensor([[[0.0, 0.5, 1.0, -1.0, 0.0]] * 3] * 2)
    valid = roles >= 0
    dino = torch.randn(2, 3)
    with torch.inference_mode():
        hypotheses, support = model(descriptors, roles, age, valid, dino)
    assert hypotheses.shape == (2, 4)
    assert support.shape == (2, 3)
    assert torch.isfinite(hypotheses).all()


def test_multi_target_nll_rewards_any_valid_candidate_or_reject() -> None:
    target = torch.tensor([
        [True, False, True, False],
        [False, False, False, True],
    ])
    good = torch.tensor([[4.0, -2.0, 3.0, -2.0], [-2.0, -2.0, -2.0, 4.0]])
    bad = -good
    assert _multi_target_nll(good, target) < _multi_target_nll(bad, target)


def test_sessions_preserve_nonzero_frozen_candidate_ranks() -> None:
    rows = [
        {"scene": "scene", "session_id": "session", "session_label": 1,
         "candidate_rank": rank}
        for rank in (4, 2, 9, 3)
    ]
    sessions = _sessions(rows, {"scene"})
    assert [[rows[index]["candidate_rank"] for index in session]
            for session in sessions] == [[2, 3, 4, 9]]
