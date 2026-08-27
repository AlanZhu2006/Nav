import math
import hashlib

import pytest

from controller_portability_contract import (
    CEC_BEARING_EXECUTOR,
    CEC_POINTGOAL_UNITS,
    CEC_PROOF_HYBRID,
    MAP_BACKEND_DIAGNOSTIC,
    NATIVE_IMAGEGOAL,
    ROLE_FREE_CEC_FULL,
    ComparisonPlan,
    fixed_bearing_payload,
    is_headline_eligible,
    project_cec_proof,
    validate_comparison_plan,
)


def plan(controller, protocol, depth, population, reject, **kwargs):
    return ComparisonPlan(
        controller=controller,
        protocol=protocol,
        depth_source=depth,
        query_population=population,
        reject_policy=reject,
        **kwargs,
    )


def test_navdp_is_the_only_current_full_role_free_controller():
    valid = plan(
        "navdp", ROLE_FREE_CEC_FULL, "monocular_sidecar",
        "mixed_role", "native_exact")
    assert validate_comparison_plan(valid).display_name == "NavDP"

    for controller, depth in (
            ("vint", "none"),
            ("iplanner", "monocular_sidecar"),
            ("viplanner", "monocular_sidecar")):
        with pytest.raises(ValueError, match="both CEC branches"):
            validate_comparison_plan(plan(
                controller, ROLE_FREE_CEC_FULL, depth,
                "mixed_role", "native_exact"))


def test_vint_is_a_native_imagegoal_baseline_not_a_bearing_executor():
    native = plan(
        "vint", NATIVE_IMAGEGOAL, "none", "mixed_role",
        "not_applicable")
    assert validate_comparison_plan(native).display_name == "ViNT"
    assert is_headline_eligible(native)

    with pytest.raises(ValueError, match="cannot consume"):
        validate_comparison_plan(plan(
            "vint", CEC_BEARING_EXECUTOR, "none", "revisit_only",
            "score_uncovered"))


@pytest.mark.parametrize("controller", ["navdp", "iplanner", "viplanner"])
@pytest.mark.parametrize("depth", ["metric_sensor", "monocular_sidecar"])
def test_pointgoal_controllers_can_enter_the_bearing_executor_tier(
        controller, depth):
    candidate = plan(
        controller, CEC_BEARING_EXECUTOR, depth, "revisit_only",
        "score_uncovered")
    assert validate_comparison_plan(candidate).key == controller
    assert is_headline_eligible(candidate)


def test_pointgoal_only_controller_cannot_hide_rejects_with_other_fallback():
    with pytest.raises(ValueError, match="score CEC reject as uncovered"):
        validate_comparison_plan(plan(
            "iplanner", CEC_BEARING_EXECUTOR, "metric_sensor",
            "revisit_only", "native_exact"))


def test_role_and_oracle_leaks_are_rejected():
    base = dict(
        controller="navdp",
        protocol=ROLE_FREE_CEC_FULL,
        depth_source="monocular_sidecar",
        query_population="mixed_role",
        reject_policy="native_exact",
    )
    with pytest.raises(ValueError, match="role labels"):
        validate_comparison_plan(ComparisonPlan(
            **base, role_label_visible=True))
    with pytest.raises(ValueError, match="oracle pose"):
        validate_comparison_plan(ComparisonPlan(
            **base, uses_oracle_pose=True))


def test_depth_contract_is_explicit():
    with pytest.raises(ValueError, match="requires an explicit depth source"):
        validate_comparison_plan(plan(
            "iplanner", CEC_BEARING_EXECUTOR, "none",
            "revisit_only", "score_uncovered"))
    with pytest.raises(ValueError, match="does not consume controller depth"):
        validate_comparison_plan(plan(
            "vint", NATIVE_IMAGEGOAL, "monocular_sidecar",
            "mixed_role", "not_applicable"))


def test_ego_planner_is_an_isolated_non_headline_map_diagnostic():
    diagnostic = plan(
        "ego_planner", MAP_BACKEND_DIAGNOSTIC, "metric_map", "any",
        "not_applicable")
    assert validate_comparison_plan(diagnostic).display_name == "EGO-Planner"
    assert not is_headline_eligible(diagnostic)

    with pytest.raises(ValueError, match="not a native ImageGoal"):
        validate_comparison_plan(plan(
            "ego_planner", NATIVE_IMAGEGOAL, "metric_map", "mixed_role",
            "not_applicable"))


def test_fixed_bearing_payload_preserves_forward_left_and_radius():
    point = [1.5, 2.0]
    payload = fixed_bearing_payload(point)
    assert payload == {"goal_x": [1.5], "goal_y": [2.0]}
    assert math.hypot(payload["goal_x"][0], payload["goal_y"][0]) == 2.5

    for invalid in ([2.4, 0.0], [float("nan"), 0.0], [True, 2.5], [2.5]):
        with pytest.raises(ValueError):
            fixed_bearing_payload(invalid)


@pytest.mark.parametrize(
    ("controller", "depth", "headline"),
    [
        ("navdp", "monocular_sidecar", True),
        ("vint", "none", True),
        ("iplanner", "metric_sensor", True),
        ("viplanner", "metric_sensor", True),
        ("ego_planner", "metric_map", False),
    ],
)
def test_every_controller_has_an_explicit_cec_proof_hybrid(
        controller, depth, headline):
    candidate = plan(
        controller, CEC_PROOF_HYBRID, depth, "mixed_role",
        "shared_native_exact", fallback_controller="navdp")
    assert validate_comparison_plan(candidate).key == controller
    assert is_headline_eligible(candidate) is headline


def accepted_proof(**updates):
    proof = {
        "certified_relocalization_schema_version": 2,
        "frame_idx": 40,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 12,
        "direction_vector": [3.0, 4.0],
        "pointgoal_units": CEC_POINTGOAL_UNITS,
        "certificate": {"accepted": True},
    }
    proof.update(updates)
    return proof


@pytest.mark.parametrize("controller", ["iplanner", "viplanner"])
def test_cec_bearing_projection_is_shared_and_scale_free(controller):
    projected = project_cec_proof(controller, accepted_proof())
    assert projected.takeover
    assert projected.adapter == "bearing_pointgoal"
    assert projected.endpoint == "pointgoal_step"
    assert projected.payload["goal_x"] == [1.5]
    assert projected.payload["goal_y"] == [2.0]
    assert projected.payload["cec_selected_anchor"] == 12


def test_navdp_preserves_the_frozen_cec_mixed_goal_adapter():
    projected = project_cec_proof("navdp", accepted_proof())
    assert projected.takeover
    assert projected.adapter == "bearing_mixedgoal"
    assert projected.endpoint == "navdp_step_ip_mixgoal"
    assert projected.payload["goal_x"] == [1.5]
    assert projected.payload["goal_y"] == [2.0]
    assert projected.payload["preserve_original_imagegoal"] is True


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_short_context_controller_consumes_the_certified_anchor_not_an_uncertified_goal(
        controller):
    anchor = b"certified-anchor-jpeg"
    anchor_sha = hashlib.sha256(anchor).hexdigest()
    projected = project_cec_proof(
        controller,
        accepted_proof(selected_anchor_image_sha256=anchor_sha),
        anchor_jpeg=anchor,
    )
    assert projected.takeover
    assert projected.adapter == "verified_anchor_imagegoal"
    assert projected.endpoint == "imagegoal_step"
    assert projected.payload["cec_anchor_sha256"] == anchor_sha
    with pytest.raises(ValueError, match="does not match"):
        project_cec_proof(
            controller,
            accepted_proof(selected_anchor_image_sha256="0" * 64),
            anchor_jpeg=anchor,
        )


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_short_context_controller_shadow_only_needs_no_anchor_fetch(
        controller):
    """forced-reject-native audits a shadow takeover without paying for the
    (expensive) certified anchor fetch, since the takeover is never granted."""
    projected = project_cec_proof(
        controller, accepted_proof(), anchor_jpeg=None, shadow_only=True)
    assert projected.takeover
    assert projected.adapter == "verified_anchor_imagegoal"
    assert projected.payload["shadow_anchor_unresolved"] is True
    assert "cec_anchor_sha256" not in projected.payload
    with pytest.raises(ValueError, match="requires the certified anchor"):
        project_cec_proof(
            controller, accepted_proof(), anchor_jpeg=None,
            shadow_only=False)


def test_ego_consumes_only_the_local_cec_goal_and_remains_non_headline():
    projected = project_cec_proof("ego_planner", accepted_proof())
    assert projected.adapter == "bearing_metric_map_goal"
    assert projected.endpoint == "metric_map_goal"
    assert projected.payload["local_metric_goal"] == [1.5, 2.0, 0.0]
    assert projected.payload["occupancy_required"] is True


@pytest.mark.parametrize(
    "controller",
    ["navdp", "vint", "gnm", "nomad", "iplanner", "viplanner", "ego_planner"])
def test_cec_rejection_selects_the_same_native_action_fallback(controller):
    projected = project_cec_proof(controller, {
        "certified_relocalization_schema_version": 2,
        "frame_idx": 40,
        "ok": True,
        "accepted": False,
        "reason": "no_causal_candidate",
        "selected_anchor": None,
        "direction_vector": None,
        "pointgoal_units": None,
        "certificate": None,
    })
    assert not projected.takeover
    assert projected.controller == "navdp"
    assert projected.adapter == "shared_native_exact"
    assert projected.payload["fallback_this_action"] is True


def test_cec_hybrid_cannot_hide_its_shared_fallback_or_role_label():
    with pytest.raises(ValueError, match="shared exact native fallback"):
        validate_comparison_plan(plan(
            "vint", CEC_PROOF_HYBRID, "none", "mixed_role",
            "native_exact", fallback_controller="navdp"))
    with pytest.raises(ValueError, match="freezes mono NavDP"):
        validate_comparison_plan(plan(
            "vint", CEC_PROOF_HYBRID, "none", "mixed_role",
            "shared_native_exact", fallback_controller="vint"))
    with pytest.raises(ValueError, match="privileged"):
        project_cec_proof("vint", accepted_proof(query_role="revisit"),
                          anchor_jpeg=b"anchor")


def test_vint_cec_can_use_the_same_controller_as_exact_reject_fallback():
    candidate = plan(
        "vint", CEC_PROOF_HYBRID, "none", "mixed_role",
        "controller_native_exact", fallback_controller="vint")
    assert validate_comparison_plan(candidate).key == "vint"

    rejected = project_cec_proof(
        "vint",
        {
            "certified_relocalization_schema_version": 2,
            "frame_idx": 40,
            "ok": True,
            "accepted": False,
            "reason": "no_causal_candidate",
            "selected_anchor": None,
            "direction_vector": None,
            "pointgoal_units": None,
            "certificate": None,
        },
        reject_policy="controller_native_exact",
    )
    assert rejected.takeover is False
    assert rejected.controller == "vint"
    assert rejected.adapter == "controller_native_exact"

    with pytest.raises(ValueError, match="same controller"):
        validate_comparison_plan(plan(
            "vint", CEC_PROOF_HYBRID, "none", "mixed_role",
            "controller_native_exact", fallback_controller="navdp"))
    with pytest.raises(ValueError, match="cannot provide controller-native"):
        validate_comparison_plan(plan(
            "iplanner", CEC_PROOF_HYBRID, "metric_sensor", "mixed_role",
            "controller_native_exact", fallback_controller="iplanner"))


def test_fractional_anchor_is_not_silently_truncated():
    with pytest.raises(ValueError, match="integer"):
        project_cec_proof("iplanner", accepted_proof(selected_anchor=12.7))
