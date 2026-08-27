import numpy as np
import pytest

from MemNavData.revisit_action_shadow import (
    ACTION_SHADOW_KEYS,
    paired_action_shadow_diagnostics,
    unavailable_action_shadow,
)
from MemNavData.summarize_revisit_action_shadow import read_metric


def _response(endpoint: float, *, seed: int = 7, read_only: bool = False):
    candidates = np.zeros((1, 2, 3, 3), dtype=float)
    candidates[0, 0, -1, 0] = endpoint
    candidates[0, 1, -1, 0] = endpoint / 2.0
    payload = {
        "trajectory": candidates[:, 0].tolist(),
        "all_trajectory": candidates.tolist(),
        "all_values": [[0.2, -0.1]],
        "diffusion_seed": seed,
    }
    if read_only:
        payload.update(
            memory_mutated=False,
            queue_hashes_before=["abc"],
            queue_hashes_after=["abc"],
        )
    return payload


def test_paired_shadow_reports_same_fifo_ratios_without_decision():
    result = paired_action_shadow_diagnostics(
        _response(0.2),
        _response(2.0, read_only=True),
        expected_seed=7,
        pointgoal_distance_m=4.0,
        stop_threshold=-0.5,
    )
    assert set(result) == set(ACTION_SHADOW_KEYS)
    assert result["revisit_action_shadow_available"] is True
    assert result[
        "revisit_action_shadow_endpoint_mean_ratio_memory_over_native"
    ] == pytest.approx(0.1)
    assert result["revisit_action_shadow_endpoint_to_pointgoal_ratio"] == (
        pytest.approx(0.0375)
    )
    assert result["revisit_action_shadow_queue_hash_match"] is True
    assert not any("decision" in key or "activate" in key for key in result)


def test_shadow_rejects_fifo_mutation_and_seed_mismatch():
    changed = _response(2.0, read_only=True)
    changed["queue_hashes_after"] = ["changed"]
    with pytest.raises(ValueError, match="changed FIFO"):
        paired_action_shadow_diagnostics(
            _response(0.2), changed, expected_seed=7,
            pointgoal_distance_m=1.0, stop_threshold=-0.5,
        )
    with pytest.raises(ValueError, match="seed mismatch"):
        paired_action_shadow_diagnostics(
            _response(0.2, seed=8), _response(2.0, read_only=True),
            expected_seed=7, pointgoal_distance_m=1.0,
            stop_threshold=-0.5,
        )


def test_unavailable_receipt_is_explicit_and_complete():
    result = unavailable_action_shadow("pose_unavailable")
    assert set(result) == set(ACTION_SHADOW_KEYS)
    assert result["revisit_action_shadow_available"] is False
    assert result["revisit_action_shadow_reason"] == "pose_unavailable"


def test_reference_metric_selects_one_episode_from_scene_csv(tmp_path):
    metric = tmp_path / "metric.csv"
    metric.write_text(
        "episode,seed,reached_B\n"
        "episode_0000,11,0\n"
        "episode_0001,12,1\n",
        encoding="utf-8",
    )
    assert read_metric(metric, "episode_0001")["seed"] == "12"
    with pytest.raises(RuntimeError, match="found 0"):
        read_metric(metric, "episode_0002")
