import pytest

from MemNavData.summarize_revisit_phase_ablation import (
    architecture_decision,
    scene_cluster_interval,
)


def test_scene_cluster_interval_is_deterministic():
    values = {
        "a": [1.0, 0.0],
        "b": [0.0, -1.0],
        "c": [1.0, 1.0],
        "d": [0.0, 0.0],
    }
    left = scene_cluster_interval(values, seed=7, resamples=2_000)
    right = scene_cluster_interval(values, seed=7, resamples=2_000)
    assert left == right
    assert -1.0 <= left[0] <= left[1] <= 1.0


@pytest.mark.parametrize(
    ("gains", "losses", "branch"),
    [
        (1, 0, "advance_known_revisit_direct_to_fresh_confirmation"),
        (4, 1, "inconclusive_build_cluster_geometry_ablation"),
        (1, 2, "retain_geometry_as_required_safety_expert"),
        (0, 0, "inconclusive_build_cluster_geometry_ablation"),
    ],
)
def test_architecture_decision_is_frozen(gains, losses, branch):
    result = architecture_decision(gains=gains, losses=losses)
    assert result["branch"] == branch
    assert result["authorize_paper_claim"] is False
    assert result["authorize_blind_eval"] is False
