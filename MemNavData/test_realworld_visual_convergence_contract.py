import copy

import pytest

from MemNavData.realworld_visual_convergence_contract import (
    ArrivalContractError,
    CALIBRATION_SCHEMA_VERSION,
    ScaleFreeConvergenceLatch,
    ScaleFreeConvergenceRule,
    VISUAL_AUDIT_SCHEMA_VERSION,
    evaluate_scale_free_observation,
    validate_labeled_population,
)


GOAL = "a" * 64
FRAME = "b" * 64


def rule():
    return ScaleFreeConvergenceRule(
        min_fundamental_inliers=32,
        min_reference_hull_coverage=0.25,
        min_query_hull_coverage=0.25,
        max_identity_flow_median_diag=0.05,
        max_affine_corner_identity_diag=0.08,
        max_abs_affine_rotation_deg=8.0,
        min_affine_scale=0.85,
        max_affine_scale=1.15,
        consecutive_hold_observations=3,
    )


def passing_row():
    return {
        "schema_version": VISUAL_AUDIT_SCHEMA_VERSION,
        "certificate_precheck_passed": True,
        "fundamental_inliers": 96,
        "fundamental_reference_hull_coverage": 0.55,
        "fundamental_query_hull_coverage": 0.52,
        "identity_flow_median_diag": 0.02,
        "affine_valid": True,
        "affine_corner_identity_max_diag": 0.04,
        "affine_rotation_deg": 2.0,
        "affine_scale": 1.02,
        # These diagnostic fields must never influence the decision.
        "predicted_distance_m": 0.01,
        "metric_scale_available": True,
    }


def test_observation_requires_proof_before_near_identity():
    row = passing_row()
    row["certificate_precheck_passed"] = False
    decision = evaluate_scale_free_observation(row, rule())
    assert decision.passed is False
    assert decision.reason == "two_view_precheck_rejected"
    assert decision.metric_translation_consumed is False


def test_observation_rejects_each_untrusted_view_residual():
    cases = {
        "identity_flow_median_diag": (0.06, "identity_flow_too_large"),
        "affine_corner_identity_max_diag": (
            0.09,
            "affine_identity_error_too_large",
        ),
        "affine_rotation_deg": (9.0, "affine_rotation_too_large"),
        "affine_scale": (1.2, "affine_scale_outside_range"),
    }
    for field, (value, reason) in cases.items():
        row = passing_row()
        row[field] = value
        decision = evaluate_scale_free_observation(row, rule())
        assert decision.passed is False
        assert decision.reason == reason
        assert decision.metric_translation_consumed is False


def test_metric_translation_cannot_change_visual_decision():
    near = passing_row()
    far = copy.deepcopy(near)
    near["predicted_distance_m"] = 0.001
    far["predicted_distance_m"] = 1000.0
    near_decision = evaluate_scale_free_observation(near, rule())
    far_decision = evaluate_scale_free_observation(far, rule())
    assert near_decision == far_decision
    assert near_decision.passed is True
    assert near_decision.metric_translation_consumed is False


@pytest.mark.parametrize("value", [32.0, "32", True])
def test_rule_rejects_non_integer_inlier_threshold(value):
    kwargs = rule().__dict__.copy()
    kwargs["min_fundamental_inliers"] = value
    with pytest.raises(ArrivalContractError, match="integer >= 8"):
        ScaleFreeConvergenceRule(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("consecutive_hold_observations", 3.0, "integer >= 2"),
        ("maximum_frame_gap", 1.0, "positive integer"),
    ],
)
def test_rule_rejects_integer_valued_floats(field, value, message):
    kwargs = rule().__dict__.copy()
    kwargs[field] = value
    with pytest.raises(ArrivalContractError, match=message):
        ScaleFreeConvergenceRule(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "min_reference_hull_coverage",
        "max_identity_flow_median_diag",
        "min_affine_scale",
    ],
)
def test_rule_rejects_numeric_strings(field):
    kwargs = rule().__dict__.copy()
    kwargs[field] = "0.5"
    with pytest.raises(ArrivalContractError, match="finite JSON number"):
        ScaleFreeConvergenceRule(**kwargs)


def test_rule_receipt_normalizes_json_numbers_to_float():
    kwargs = rule().__dict__.copy()
    kwargs["min_query_hull_coverage"] = 0
    parsed = ScaleFreeConvergenceRule(**kwargs)
    assert parsed.receipt()["min_query_hull_coverage"] == 0.0
    assert type(parsed.receipt()["min_query_hull_coverage"]) is float


@pytest.mark.parametrize("frame_index", [1.0, "1", True])
def test_latch_rejects_non_integer_frame_index(frame_index):
    latch = ScaleFreeConvergenceLatch(rule())
    with pytest.raises(ArrivalContractError, match="non-negative integer"):
        latch.update(
            goal_sha256=GOAL,
            frame_index=frame_index,
            observation=passing_row(),
            terminal_hold_active=True,
        )


def test_latch_requires_hold_and_three_causal_observations():
    latch = ScaleFreeConvergenceLatch(rule())
    request = latch.update(
        goal_sha256=GOAL,
        frame_index=10,
        observation=passing_row(),
        terminal_hold_active=False,
    )
    assert request.disposition == "request_hold"
    assert request.shadow_stop_authorized is False
    for frame in (11, 12):
        decision = latch.update(
            goal_sha256=GOAL,
            frame_index=frame,
            observation=passing_row(),
            terminal_hold_active=True,
        )
        assert decision.disposition == "hold"
        assert decision.runtime_stop_authorized is False
    decision = latch.update(
        goal_sha256=GOAL,
        frame_index=13,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert decision.disposition == "shadow_stop"
    assert decision.shadow_stop_authorized is True
    # Physical promotion is a different release and remains disabled here.
    assert decision.runtime_stop_authorized is False


def test_latch_cannot_skip_request_hold_from_cold_start():
    latch = ScaleFreeConvergenceLatch(rule())
    cold = latch.update(
        goal_sha256=GOAL,
        frame_index=1,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert cold.disposition == "request_hold"
    assert cold.streak == 0
    assert cold.shadow_stop_authorized is False


def test_latch_resets_on_gap_failure_and_goal_change():
    latch = ScaleFreeConvergenceLatch(rule())
    request = latch.update(
        goal_sha256=GOAL,
        frame_index=1,
        observation=passing_row(),
        terminal_hold_active=False,
    )
    assert request.disposition == "request_hold"
    first = latch.update(
        goal_sha256=GOAL,
        frame_index=2,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert first.streak == 1
    gap = latch.update(
        goal_sha256=GOAL,
        frame_index=4,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert gap.disposition == "request_hold"
    assert gap.streak == 0
    rearmed = latch.update(
        goal_sha256=GOAL,
        frame_index=5,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert rearmed.streak == 1
    failed_row = passing_row()
    failed_row["identity_flow_median_diag"] = 0.5
    failed = latch.update(
        goal_sha256=GOAL,
        frame_index=6,
        observation=failed_row,
        terminal_hold_active=True,
    )
    assert failed.streak == 0
    changed = latch.update(
        goal_sha256="c" * 64,
        frame_index=7,
        observation=passing_row(),
        terminal_hold_active=True,
    )
    assert changed.disposition == "request_hold"
    assert changed.streak == 0


def population():
    def sample(sample_id, location, split, distance, yaw=0.0):
        return {
            "sample_id": sample_id,
            "location_id": location,
            "split": split,
            "distance_m": distance,
            "yaw_deg": yaw,
            "goal_sha256": GOAL,
            "frame_sha256": FRAME,
        }

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "success_radius_m": 0.85,
        "samples": [
            sample("cal-pos", "office-a", "calibration", 0.25),
            sample("cal-neg", "office-a", "calibration", 1.0),
            sample("conf-pos", "office-b", "confirmation", 0.5),
            sample("conf-neg", "office-b", "confirmation", 1.0),
        ],
    }


def test_population_enforces_location_disjoint_calibration():
    receipt = validate_labeled_population(population())
    assert receipt["location_disjoint"] is True
    assert receipt["threshold_selection_reads_confirmation"] is False
    assert receipt["counts"]["confirmation"] == {
        "positive": 1,
        "negative": 1,
        "samples": 2,
    }


def test_population_rejects_location_leakage_and_one_class_split():
    leaked = population()
    leaked["samples"][2]["location_id"] = "office-a"
    with pytest.raises(ArrivalContractError, match="location leakage"):
        validate_labeled_population(leaked)

    one_class = population()
    one_class["samples"][3]["distance_m"] = 0.5
    with pytest.raises(ArrivalContractError, match="positive and negative"):
        validate_labeled_population(one_class)
