import math

from MemNavData.audit_xnavdp_direction_execution import (
    _exact_mcnemar_two_sided,
    summarize_states,
)


def test_exact_mcnemar_handles_dominance_and_tie():
    assert _exact_mcnemar_two_sided(0, 0) == 1.0
    assert math.isclose(_exact_mcnemar_two_sided(3, 0), 0.25)
    assert _exact_mcnemar_two_sided(1, 1) == 1.0


def test_summary_keeps_actuator_and_cross_request_q_separate():
    states = [{
        "scene": "s1",
        "xnav_oracle_request_executed_error_deg": 10.0,
        "base_oracle_request_executed_error_deg": 80.0,
        "xnav_request_execution_ceiling_error_deg": 10.0,
        "xnav_candidate_execution_ceiling_error_deg": 5.0,
        "xnav_q_chosen_executed_error_deg": 90.0,
    }]
    directions = [{
        "selected_request_error_deg": 15.0,
        "best_candidate_request_error_deg": 5.0,
    }]
    report = summarize_states(states, directions, threshold_deg=30.0)
    assert report["xnav_oracle_request_hits"] == 1
    assert report["base_mixed_oracle_request_hits"] == 0
    assert report["xnav_vs_base_gains"] == 1
    assert report["xnav_cross_request_q_hits"] == 0
    assert report["selected_request_fidelity_hits"] == 1
