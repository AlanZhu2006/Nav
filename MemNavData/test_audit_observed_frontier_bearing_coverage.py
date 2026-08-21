import math

import numpy as np

from MemNavData.audit_observed_frontier_bearing_coverage import (
    circular_error_deg,
    cluster_bootstrap_episode_mean,
    episode_record,
    heading_resultant,
    path_initial_bearing,
    summarize_early_windows,
    summarize_state_rows,
)


def _state(error, top1, *, plan=0, candidates=2):
    return {
        "plan_index": plan,
        "oracle_bearing_deg": 0.0,
        "reachable_candidate_count": candidates,
        "candidate_oracle_error_deg": error,
        "fixed_top1_error_deg": top1,
        "current_heading_error_deg": 50.0,
        "candidate_heading_resultant": 0.5 if candidates else None,
        "scene": "scene",
        "episode": "episode_0000",
    }


def test_circular_error_wraps_and_path_bearing_uses_habitat_convention():
    assert math.isclose(circular_error_deg(math.radians(179),
                                           math.radians(-179)), 2.0)
    start = np.array([0.0, 0.0, 0.0])
    assert math.isclose(path_initial_bearing(
        [start, [0.0, 0.0, -1.0]], start), 0.0)
    assert math.isclose(path_initial_bearing(
        [start, [-1.0, 0.0, 0.0]], start), math.pi / 2)


def test_resultant_distinguishes_collapsed_and_opposed_headings():
    assert math.isclose(heading_resultant([0.0, 0.0, 0.0]), 1.0)
    assert heading_resultant([0.0, math.pi]) < 1e-12
    assert heading_resultant([]) is None


def test_state_and_episode_summaries_keep_oracle_separate_from_top1():
    states = [_state(10.0, 80.0, plan=0), _state(40.0, 20.0, plan=1)]
    state_summary = summarize_state_rows(states)
    assert state_summary["candidate_oracle_within_30_count"] == 1
    assert state_summary["fixed_top1_within_30_count"] == 1
    assert state_summary["current_heading_within_30_count"] == 0
    record = episode_record("scene", "episode_0000", False, states)
    assert record["candidate_oracle_any_within_30"]
    assert record["candidate_oracle_first_within_30_plan"] == 0
    assert record["fixed_top1_any_within_30"]
    assert record["candidate_oracle_within_30_fraction"] == 0.5


def test_early_window_summary_does_not_overweight_late_states():
    states = [_state(80.0, 80.0, plan=0), _state(10.0, 10.0, plan=4)]
    summary = summarize_early_windows(states, windows=(1, 8))
    assert summary["first_1_plans"]["candidate_oracle_state_rate"] == 0.0
    assert summary["first_1_plans"]["candidate_oracle_episode_any_count"] == 0
    assert summary["first_8_plans"]["candidate_oracle_state_rate"] == 0.5
    assert summary["first_8_plans"]["candidate_oracle_episode_any_count"] == 1


def test_scene_bootstrap_samples_scene_clusters_not_individual_episodes():
    rows = [
        {"scene": "a", "value": 1.0},
        {"scene": "a", "value": 1.0},
        {"scene": "b", "value": 0.0},
    ]
    interval = cluster_bootstrap_episode_mean(
        rows, "value", resamples=100, seed=7)
    assert interval["scene_clusters"] == 2
    assert 0.0 <= interval["lower_95"] <= interval["upper_95"] <= 1.0
