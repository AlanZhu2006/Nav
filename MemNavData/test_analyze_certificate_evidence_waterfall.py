import json

import pytest

from MemNavData.analyze_certificate_evidence_waterfall import analyze


def pnp(*, status="ok", error=0.2, inliers=32, query=0.2,
        reference=0.2, rmse=1.0):
    return {
        "status": status,
        "inliers": inliers,
        "query_inlier_coverage": query,
        "reference_inlier_coverage": reference,
        "reprojection_rmse_px": rmse,
        "relative_position_error_m": error,
        "predicted_relative_xy_m": [1.0, 0.0] if status == "ok" else None,
    }


def endpoint(index, value, *, positive=True, strict=False):
    return {
        "session_id": f"session_{index}",
        "scene": f"scene_{index % 2}",
        "candidate_frame": str(index + 10),
        "candidate_path": f"/{index}.jpg",
        "candidate_selection_origin": "lightglue_fundamental_rank_v1",
        "session_has_positive": str(positive),
        "session_is_strict_no_match": str(strict),
        "causal_state_name": "goal_c_t0" if positive else "goal_b_t0",
        "hypotheses_json": json.dumps([{
            "offset": 0,
            "pnp_lightglue": value,
        }]),
    }


def static(index, *, inliers=32, query=0.2, reference=0.2):
    return {
        "session_id": f"session_{index}",
        "candidate_frame": str(index + 10),
        "candidate_path": f"/{index}.jpg",
        "fundamental_inliers": str(inliers),
        "fundamental_query_hull_coverage": str(query),
        "fundamental_reference_hull_coverage": str(reference),
    }


def test_waterfall_is_cumulative_and_separates_precheck_from_certificate():
    endpoints = [
        endpoint(0, pnp()),
        endpoint(1, pnp(error=2.0), positive=False, strict=True),
        endpoint(2, pnp(inliers=8)),
        endpoint(3, pnp(query=0.01)),
        endpoint(4, pnp(reference=0.01)),
        endpoint(5, pnp(rmse=4.0)),
        endpoint(6, pnp(status="failed")),
    ]
    statics = [static(index) for index in range(len(endpoints))]
    # This row never passes even the correspondence precheck.
    statics[6]["fundamental_inliers"] = "8"

    result = analyze(endpoints, statics)
    stages = result["stages"]
    assert stages["geometry_ranked_candidate"]["accepted"] == 7
    assert stages["fundamental_precheck"]["accepted"] == 6
    assert stages["precheck_plus_pnp_pose"]["accepted"] == 6
    assert stages["plus_pnp_inliers"]["accepted"] == 5
    assert stages["plus_query_coverage"]["accepted"] == 4
    assert stages["plus_reference_coverage"]["accepted"] == 3
    assert stages["full_certificate"]["accepted"] == 2
    assert stages["precheck_plus_pnp_pose"]["false_positive"] == 1
    assert stages["full_certificate"]["false_positive"] == 1
    assert stages["full_certificate"]["open_set_support_audit"][
        "strict_no_match"] == {"sessions": 1, "authorized": 1}
    sensitivity = result["threshold_sensitivity"]["sweeps"]
    assert [row["value"] for row in sensitivity["min_inliers"]] == [
        8, 12, 16, 24, 32]
    assert next(
        row for row in sensitivity["symmetric_hull_coverage"]
        if row["value"] == 0.05)["accepted"] == 2


def test_selected_static_candidate_must_match_exact_path():
    row = endpoint(0, pnp())
    candidate = static(0)
    candidate["candidate_path"] = "/changed.jpg"
    with pytest.raises(RuntimeError, match="path changed"):
        analyze([row], [candidate])


def test_manifest_must_bind_exact_session_universe():
    row = endpoint(0, pnp())
    manifest = {
        "schema_version": "train40_certificate_challenge_manifest_v1",
        "sessions": ["another_session"],
    }
    with pytest.raises(RuntimeError, match="frozen manifest"):
        analyze([row], [static(0)], manifest)
