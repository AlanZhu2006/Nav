import json

import pandas as pd

from MemNavData.independent_verify_train40_certificate_challenge import verify


def pnp(*, error=0.2, support=0.2):
    return {
        "status": "ok",
        "inliers": 32,
        "query_inlier_coverage": support,
        "reference_inlier_coverage": support,
        "reprojection_rmse_px": 1.0,
        "relative_position_error_m": error,
    }


def test_independent_recount_matches_all_strata():
    rows = []
    sessions = []
    for index in range(480):
        session = f"train/scene_{index % 40:02d}/episode/{index}"
        sessions.append(session)
        value = pnp(
            error=1.0 if index == 1 else 0.2,
            support=0.01 if index == 2 else 0.2,
        )
        rows.append({
            "session_id": session,
            "scene": f"scene_{index % 40:02d}",
            "teacher_covis": (0.0, 0.3, 0.8)[index % 3],
            "session_max_covis": (0.0, 0.4, 0.9)[index % 3],
            "candidate_frame": 20,
            "causal_decision_frame": (40, 80, 140)[index % 3],
            "causal_state_name": ("b0", "b1", "c0")[index % 3],
            "hypotheses_json": json.dumps([{
                "offset": 0, "pnp_lightglue": value,
            }]),
        })
    frame = pd.DataFrame(rows)

    # Build the expected audit with a first pass through the verifier's simple
    # count vocabulary. This fixture intentionally supplies only the fields the
    # independent comparison consumes.
    from MemNavData.independent_verify_train40_certificate_challenge import (
        actionable,
        certificate,
        counts,
        grouped_counts,
        history_band,
        support_band,
    )
    records = []
    for row in frame.itertuples(index=False):
        value = json.loads(row.hypotheses_json)[0]["pnp_lightglue"]
        records.append({
            "scene": row.scene,
            "certificate": certificate(value),
            "actionable": actionable(value),
            "selected": support_band(row.teacher_covis),
            "session": support_band(row.session_max_covis),
            "history": history_band(
                row.causal_decision_frame - row.candidate_frame),
            "state": row.causal_state_name,
        })
    expected = pd.DataFrame(records)
    overall = counts(expected)
    audit = {
        "rows": overall["sessions"],
        "scenes": overall["scenes"],
        "ground_truth_actionable": overall["actionable"],
        "certificate": overall,
        "stratified_actionability": {
            "selected_anchor_support": grouped_counts(expected, "selected"),
            "session_max_support": grouped_counts(expected, "session"),
            "history_gap": grouped_counts(expected, "history"),
            "causal_state": grouped_counts(expected, "state"),
        },
    }
    manifest = {
        "schema_version": "train40_certificate_challenge_manifest_v1",
        "session_universe_sha256": "a" * 64,
        "sessions": sessions,
    }
    result = verify(frame, audit, manifest)
    assert result["overall"]["sessions"] == 480
    assert result["overall"]["false_positive"] == 1
    assert result["overall"]["false_negative"] == 1
