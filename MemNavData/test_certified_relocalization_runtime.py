import numpy as np

from certified_relocalization_runtime import (
    CERTIFICATE_MAX_REPROJECTION_RMSE_PX,
    CERTIFICATE_MIN_INLIERS,
    CERTIFICATE_MIN_QUERY_COVERAGE,
    CERTIFICATE_MIN_REFERENCE_COVERAGE,
    STRICT_AUTHORITY_POLICY,
    UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
    candidate_rank_key,
    certificate_decision,
    fundamental_can_reach_certificate,
    fundamental_support,
    operational_authority_decision,
    rank_candidates,
    runtime_contract,
    scale_free_relative_xy,
)


def valid_pnp(**updates):
    payload = {
        "status": "ok",
        "inliers": CERTIFICATE_MIN_INLIERS,
        "query_inlier_coverage": CERTIFICATE_MIN_QUERY_COVERAGE,
        "reference_inlier_coverage": CERTIFICATE_MIN_REFERENCE_COVERAGE,
        "reprojection_rmse_px": CERTIFICATE_MAX_REPROJECTION_RMSE_PX,
        "pose9": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    }
    payload.update(updates)
    return payload


def test_certificate_is_atomic_and_inclusive_at_frozen_boundaries():
    accepted = certificate_decision(valid_pnp())
    assert accepted["accepted"] is True
    assert accepted["failed_checks"] == []

    for field, value, expected in (
        ("inliers", CERTIFICATE_MIN_INLIERS - 1, "minimum_inliers"),
        ("query_inlier_coverage", 0.049, "minimum_query_coverage"),
        ("reference_inlier_coverage", 0.049,
         "minimum_reference_coverage"),
        ("reprojection_rmse_px", 2.001, "maximum_reprojection_rmse"),
    ):
        rejected = certificate_decision(valid_pnp(**{field: value}))
        assert rejected["accepted"] is False
        assert expected in rejected["failed_checks"]


def test_certificate_rejects_missing_nonfinite_and_bad_status():
    assert certificate_decision({})["accepted"] is False
    assert certificate_decision(valid_pnp(status="ransac_failed"))[
        "accepted"] is False
    assert certificate_decision(valid_pnp(
        reprojection_rmse_px=float("nan")))["accepted"] is False


def test_authority_ablation_changes_only_intervention_thresholds():
    weak_but_finite = valid_pnp(
        status="insufficient_inliers",
        inliers=8,
        query_inlier_coverage=0.01,
        reference_inlier_coverage=0.01,
        reprojection_rmse_px=14.0,
    )
    strict = operational_authority_decision(
        weak_but_finite, policy=STRICT_AUTHORITY_POLICY)
    unthresholded = operational_authority_decision(
        weak_but_finite,
        policy=UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
    )
    assert strict["accepted"] is False
    assert strict["certificate_thresholds_enforced"] is True
    assert unthresholded["accepted"] is True
    assert unthresholded["reason"] == "pnp_pose_available"
    assert unthresholded["certificate_thresholds_enforced"] is False
    assert unthresholded["strict_certificate"] == strict["strict_certificate"]


def test_unthresholded_authority_still_requires_a_finite_pose():
    for payload in (
        {},
        valid_pnp(pose9=None),
        valid_pnp(pose9=[0.0] * 8),
        valid_pnp(pose9=[0.0] * 8 + [float("nan")]),
    ):
        decision = operational_authority_decision(
            payload, policy=UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY)
        assert decision["accepted"] is False
        assert decision["reason"] == "pnp_pose_unavailable"


def test_authority_policy_is_closed_to_unregistered_values():
    with np.testing.assert_raises(ValueError):
        operational_authority_decision(valid_pnp(), policy="accept_everything")


def test_rank_is_frozen_lexicographic_and_prefers_earlier_tie():
    base = {
        "fundamental_inliers": 20,
        "fundamental_query_grid_coverage": 0.5,
        "fundamental_query_hull_coverage": 0.2,
        "lightglue_score_median": 0.8,
        "dino_cosine": 0.9,
    }
    candidates = [
        {**base, "anchor": 20},
        {**base, "anchor": 10},
        {**base, "anchor": 30, "fundamental_inliers": 21,
         "dino_cosine": 0.1},
    ]
    ranked = rank_candidates(candidates)
    assert [row["anchor"] for row in ranked] == [30, 10, 20]
    assert candidate_rank_key(candidates[0]) < candidate_rank_key(candidates[1])


def test_fundamental_precheck_is_only_monotone_certificate_bounds():
    evidence = {
        "fundamental_inliers": 16,
        "fundamental_query_hull_coverage": 0.05,
        "fundamental_reference_hull_coverage": 0.05,
    }
    assert fundamental_can_reach_certificate(evidence) == (
        True, "precheck_passed")
    for field in evidence:
        broken = dict(evidence)
        broken[field] = 0
        possible, reason = fundamental_can_reach_certificate(broken)
        assert possible is False
        assert reason == f"precheck_{field}"


def test_fundamental_support_reports_distributed_correspondences():
    # A synthetic translated stereo pair with broad image support.
    xx, yy = np.meshgrid(np.linspace(10, 90, 8), np.linspace(10, 70, 6))
    reference = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    query = reference + np.array([4.0, 0.5], dtype=np.float32)
    result = fundamental_support(
        reference, query, np.ones(len(reference)), (80, 100), (80, 100))
    assert result["lightglue_matches"] == len(reference)
    assert result["fundamental_inliers"] >= 40
    assert result["fundamental_query_hull_coverage"] > 0.5
    assert result["fundamental_reference_hull_coverage"] > 0.5


def test_runtime_contract_exposes_fallback_and_not_binary_semantics():
    contract = runtime_contract()
    assert contract["candidate_top_k"] == 8
    assert contract["candidate_min_gap"] == 4
    assert contract["candidate_lifecycle"] == "frozen_at_first_goal_query"
    assert contract["empty_candidate_semantics"] == (
        "cached_native_abstention")
    assert contract["fallback"] == "native_imagegoal"
    assert contract["metric_distance_certified"] is False
    assert contract["output"] == "scale_free_relative_bearing"
    assert contract["default_authority_policy"] == STRICT_AUTHORITY_POLICY
    assert contract["diagnostic_authority_policies"] == [
        UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY]
    assert "unknown" in contract["semantic_claim"]


def test_scale_free_relative_xy_matches_memnav_axis_and_ignores_fov():
    current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 99.0, -5.0]
    goal = [1.0, 7.0, 2.0, 0.0, 0.0, 0.0, 1.0, -8.0, 3.0]
    assert np.allclose(scale_free_relative_xy(current, goal), [2.0, -1.0])


def test_scale_free_relative_xy_preserves_bearing_under_global_scale():
    current = np.asarray(
        [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    goal = np.asarray(
        [4.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    original = np.asarray(scale_free_relative_xy(current, goal))
    current_scaled = current.copy()
    goal_scaled = goal.copy()
    current_scaled[:3] *= 7.3
    goal_scaled[:3] *= 7.3
    scaled = np.asarray(scale_free_relative_xy(current_scaled, goal_scaled))
    assert np.allclose(scaled, 7.3 * original)
    assert np.allclose(
        scaled / np.linalg.norm(scaled), original / np.linalg.norm(original))


def test_scale_free_relative_xy_rejects_invalid_pose():
    valid = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    with np.testing.assert_raises(ValueError):
        scale_free_relative_xy(valid[:-1], valid)
    zero_quaternion = list(valid)
    zero_quaternion[3:7] = [0.0, 0.0, 0.0, 0.0]
    with np.testing.assert_raises(ValueError):
        scale_free_relative_xy(zero_quaternion, valid)
