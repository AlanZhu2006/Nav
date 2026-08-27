import math

import numpy as np
import pytest

from MemNavData.navdp_goal_contrast import (
    exact_mcnemar_p,
    goal_contrast_diagnostics,
    summarize_goal_contrast,
    trajectory_heading_deg,
)


def _candidate(heading_deg: float) -> list[list[float]]:
    theta = math.radians(heading_deg)
    return [
        [0.2 * math.cos(theta), 0.2 * math.sin(theta), 0.0],
        [0.8 * math.cos(theta), 0.8 * math.sin(theta), 0.0],
    ]


def _response() -> dict:
    return {
        "all_trajectory": [[
            _candidate(0.0), _candidate(90.0), _candidate(-180.0),
        ]],
        "goal_contrast": {
            "score_semantics": "nogoal_mse_minus_goal_mse",
            "is_calibrated_likelihood": False,
            "goal_advantage": [[0.1, 0.5, 0.2]],
            "normalized_goal_advantage": [[0.01, 0.05, 0.02]],
            "control_goal_advantage": [[0.6, 0.1, 0.2]],
        },
    }


def test_goal_contrast_selects_correct_and_control_independently():
    result = goal_contrast_diagnostics(
        _response(), requested_heading_deg=90.0)
    assert result["goal_candidate_index"] == 1
    assert result["goal_selected_request_error_deg"] == pytest.approx(0.0)
    assert result["control_candidate_index"] == 0
    assert result["control_selected_request_error_deg"] == pytest.approx(90.0)
    assert result["goal_vs_control_at_goal_choice"] == pytest.approx(0.4)


def test_goal_contrast_rejects_likelihood_overclaim():
    response = _response()
    response["goal_contrast"]["is_calibrated_likelihood"] = True
    with pytest.raises(ValueError, match="must not claim"):
        goal_contrast_diagnostics(response)


def test_heading_uses_last_endpoint_beyond_threshold():
    assert trajectory_heading_deg(_candidate(-90.0)) == pytest.approx(-90.0)
    with pytest.raises(ValueError, match="shape"):
        trajectory_heading_deg(np.zeros((2, 2)))


def test_goal_contrast_summary_keeps_request_and_execution_separate():
    rows = [{
        "scene": "a",
        "goal_request_error_deg": 10.0,
        "goal_executed_error_deg": 80.0,
        "control_request_error_deg": 90.0,
        "control_executed_error_deg": 20.0,
        "goal_chosen_direction_deg": 0.0,
        "control_chosen_direction_deg": 90.0,
        "goal_oracle_request_rank": 1,
        "control_oracle_request_rank": 3,
        "direction_count": 8,
        "random_request_hit_probability": 0.125,
        "goal_score_margin": 0.2,
        "goal_vs_control_at_goal_choice": 0.4,
    }]
    report = summarize_goal_contrast(rows, 30.0)
    assert report["goal_request_hits"] == 1
    assert report["goal_executed_hits"] == 0
    assert report["control_request_hits"] == 0
    assert report["control_executed_hits"] == 1
    assert report["goal_vs_control_request_gains"] == 1
    assert report["goal_vs_control_executed_losses"] == 1


def test_exact_mcnemar_small_sample_values():
    assert exact_mcnemar_p(3, 0) == pytest.approx(0.25)
    assert exact_mcnemar_p(2, 0) == pytest.approx(0.5)
    assert exact_mcnemar_p(0, 0) == pytest.approx(1.0)
