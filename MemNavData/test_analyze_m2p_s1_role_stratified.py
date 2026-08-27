import pandas as pd

from MemNavData.analyze_m2p_s1_role_stratified import (
    causal_state,
    exact_mcnemar_p,
    partial_futility_decision,
    quadrant_summary,
    recall_gap_band,
)


def test_causal_state_is_role_explicit():
    assert causal_state("x/goal_b_t0/factual") == "novel_b_t0"
    assert causal_state("x/goal_b_midpoint_t1/factual") == (
        "novel_b_midpoint_t1")
    assert causal_state("x/goal_c_t0/factual") == "true_revisit_c_t0"


def test_quadrants_keep_complementary_hypotheses_separate():
    frame = pd.DataFrame({
        "anchored_error_deg": [1.0, 1.0, 90.0, 90.0],
        "full_prefix_error_deg": [1.0, 90.0, 1.0, 90.0],
    })
    result = quadrant_summary(frame, anchored_column="anchored_error_deg")
    assert result["both_good"] == 1
    assert result["anchored_only_good"] == 1
    assert result["full_prefix_only_good"] == 1
    assert result["neither_good"] == 1
    assert result["oracle_union_cdf30"] == 0.75


def test_mcnemar_and_gap_bands():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(3, 0) == 0.25
    assert recall_gap_band(None) == "no_teacher_support"
    assert recall_gap_band(32) == "within_gct_window_le32"
    assert recall_gap_band(33) == "mid_gap_33_128"
    assert recall_gap_band(129) == "long_gap_gt128"


def test_partial_gate_only_stops_catastrophic_or_invalid_runs():
    quadrants = {
        "full_prefix_only_good": 0,
        "anchored_only_good": 5,
        "full_prefix_cdf30": 0.5,
        "anchored_cdf30": 0.75,
    }
    stopped = partial_futility_decision(
        scenes=10, primary_quadrants=quadrants,
        identity_ok=True, production_parity_ok=True)
    assert stopped["continue_full40"] is False
    assert stopped["reason"] == (
        "catastrophic_full_prefix_dominance_failure")
    assert stopped["stop_reasons"] == [
        "catastrophic_full_prefix_dominance_failure"]

    not_futile = dict(quadrants, full_prefix_only_good=1)
    continued = partial_futility_decision(
        scenes=10, primary_quadrants=not_futile,
        identity_ok=True, production_parity_ok=True)
    assert continued["continue_full40"] is True
    assert continued["reason"] == "not_futile_requires_full40"

    parity_failure = partial_futility_decision(
        scenes=10, primary_quadrants=not_futile,
        identity_ok=True, production_parity_ok=False)
    assert parity_failure["continue_full40"] is False
    assert parity_failure["reason"] == "production_cdf30_parity_failure"
    assert parity_failure["stop_reasons"] == [
        "production_cdf30_parity_failure"]

    two_failures = partial_futility_decision(
        scenes=10, primary_quadrants=quadrants,
        identity_ok=True, production_parity_ok=False)
    assert two_failures["stop_reasons"] == [
        "production_cdf30_parity_failure",
        "catastrophic_full_prefix_dominance_failure",
    ]

    one_scene = partial_futility_decision(
        scenes=1, primary_quadrants=quadrants,
        identity_ok=True, production_parity_ok=True)
    assert one_scene["continue_full40"] is None
