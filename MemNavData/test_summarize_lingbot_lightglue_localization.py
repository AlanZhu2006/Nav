import json

import pandas as pd

from MemNavData.summarize_lingbot_lightglue_localization import (
    has_certificate,
    is_actionable,
    summarize,
    wilson,
)


def good_pnp(**overrides):
    value = {
        "status": "ok",
        "inliers": 40,
        "query_inlier_coverage": 0.2,
        "reference_inlier_coverage": 0.2,
        "reprojection_rmse_px": 1.0,
        "relative_position_error_m": 0.2,
        "relative_position_direction_error_deg": 5.0,
        "relative_rotation_error_deg": 3.0,
    }
    value.update(overrides)
    return value


def test_actionability_is_pose_accuracy_not_covisibility():
    assert is_actionable(good_pnp())
    assert not is_actionable(good_pnp(relative_position_error_m=1.0))


def test_metric_pointgoal_actionability_does_not_gate_camera_yaw_or_short_bearing():
    assert is_actionable(
        good_pnp(
            relative_position_direction_error_deg=170.0,
            relative_rotation_error_deg=170.0))


def test_certificate_fails_closed_on_tiny_image_support():
    assert has_certificate(good_pnp())
    assert not has_certificate(good_pnp(query_inlier_coverage=0.01))
    assert not has_certificate(good_pnp(reference_inlier_coverage=0.01))
    assert not has_certificate(good_pnp(inliers=8))


def test_wilson_interval_is_bounded():
    interval = wilson(9, 10)
    assert interval is not None
    assert 0.0 <= interval[0] < 0.9 < interval[1] <= 1.0


def test_hpc_gate_requires_safe_cross_scene_coverage_and_pose_gain():
    rows = []
    for index in range(5):
        pnp = good_pnp()
        rows.append({
            "session_id": f"scene_{index}/episode/goal",
            "scene": f"scene_{index}",
            "label": 1,
            "teacher_covis": 0.8,
            "hypotheses_json": json.dumps([{
                "offset": 0,
                "target_relative_distance_m": 2.0,
                "pnp_lightglue": pnp,
            }]),
            "relative_position_error_m_center": 1.0,
            "relative_position_direction_error_deg_center": 40.0,
            "relative_rotation_error_deg_center": 20.0,
        })
    result = summarize(
        pd.DataFrame(rows), {"config": {"pnp_lightglue": True}})
    assert result["hpc_effectiveness_gate"]["all_checks_pass"]
    assert result["hpc_effectiveness_gate"]["certified_actionable_scenes"] == 5


def test_stratification_is_posthoc_and_preserves_confusion_counts():
    rows = []
    cases = [
        ("low", 0.0, 0.0, 20, 100, good_pnp()),
        ("boundary", 0.3, 0.4, 80, 120,
         good_pnp(relative_position_error_m=1.2)),
        ("strong", 0.8, 0.9, 30, 50,
         good_pnp(query_inlier_coverage=0.01)),
    ]
    for name, selected_covis, max_covis, candidate, decision, pnp in cases:
        rows.append({
            "session_id": f"scene_{name}/episode/goal",
            "scene": f"scene_{name}",
            "label": 1 if selected_covis > 0.5 else 0,
            "teacher_covis": selected_covis,
            "session_max_covis": max_covis,
            "candidate_frame": candidate,
            "causal_decision_frame": decision,
            "causal_state_name": f"state_{name}",
            "hypotheses_json": json.dumps([{
                "offset": 0,
                "pnp_lightglue": pnp,
            }]),
            "relative_position_error_m_center": 1.0,
            "relative_position_direction_error_deg_center": 40.0,
            "relative_rotation_error_deg_center": 20.0,
        })
    result = summarize(
        pd.DataFrame(rows), {"config": {"pnp_lightglue": True}})
    strata = result["stratified_actionability"]
    assert strata["selected_anchor_support"][
        "strict_or_low_le_0p10"]["true_positive"] == 1
    assert strata["selected_anchor_support"][
        "boundary_0p10_to_0p50"]["false_positive"] == 1
    assert strata["selected_anchor_support"][
        "strong_gt_0p50"]["false_negative"] == 1
    assert strata["history_gap"]["within_lingbot_window_le_32"][
        "sessions"] == 1
    assert strata["history_gap"]["delayed_33_to_96"]["sessions"] == 2
