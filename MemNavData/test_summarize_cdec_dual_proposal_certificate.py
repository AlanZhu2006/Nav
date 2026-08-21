import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from MemNavData.summarize_cdec_dual_proposal_certificate import (
    audit, cascade, paired, same_anchor_repeatability,
)


def row(certificate, actionable):
    return {
        "scene": "s", "teacher_candidate_label": 1,
        "candidate_frame": 1, "candidate_path": "/same.jpg",
        "session_has_positive": True, "session_is_strict_no_match": False,
        "certificate": certificate, "actionable": actionable,
        "certified_actionable": certificate and actionable,
        "certificate_false_positive": certificate and not actionable,
        "pnp_status": "ok" if certificate else "ransac_failed",
        "pnp_inliers": 32 if certificate else 0,
    }


def test_reject_only_cascade_uses_fallback_without_overriding_accept():
    primary = {"a": row(True, True), "b": row(False, False)}
    fallback = {"a": row(True, False), "b": row(True, True)}
    summary, selected = cascade(primary, fallback)
    assert selected["a"]["actionable"] is True
    assert selected["b"]["actionable"] is True
    assert summary["fallback_rescues"] == 1
    assert summary["second_certificate_invocations"] == 1


def test_paired_counts_directional_discordance():
    first = {"a": row(True, True), "b": row(False, False)}
    second = {"a": row(False, False), "b": row(True, True)}
    result = paired(first, second, "certified_actionable")
    assert result["gains"] == 1
    assert result["losses"] == 1
    assert result["exact_mcnemar_p"] == 1.0


def test_same_anchor_repeatability_detects_certificate_decision_drift():
    first = {"stable": row(True, True), "drift": row(True, True)}
    second = {"stable": row(True, True), "drift": row(False, False)}
    result = same_anchor_repeatability(first, second)
    assert result["same_anchor_sessions"] == 2
    assert result["decision_equal"] == 1
    assert result["decision_mismatches"] == 1
    assert result["mismatch_session_ids"] == ["drift"]


def test_geometry_first_reject_only_cascade_cannot_replace_accepted_geometry():
    geometry = {
        "accepted": row(True, True),
        "rejected": row(False, False),
    }
    learned = {
        # This harmful accepted proposal must never override geometry.
        "accepted": row(True, False),
        # This useful proposal may rescue only the rejected case.
        "rejected": row(True, True),
    }
    summary, selected = cascade(geometry, learned)
    comparison = paired(selected, geometry, "certified_actionable")
    assert selected["accepted"]["selected_source"] == "primary"
    assert selected["accepted"]["actionable"] is True
    assert selected["rejected"]["selected_source"] == "fallback"
    assert comparison["gains"] == 1
    assert comparison["losses"] == 0
    assert summary["fallback_rescues"] == 1


def test_dual_audit_fails_closed_on_incomplete_hardware_paired_rows():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = root / "rows.csv"
        report = root / "report.json"
        pd.DataFrame([{
            "session_id": "s", "candidate_selection_origin":
            "lightglue_fundamental_rank_v1",
        }]).to_csv(rows, index=False)
        report.write_text(json.dumps({
            "config": {
                "selection_mode": "cdec_geometry_dual_ranked",
                "neighbor_offsets": [0], "full_replay": True,
            }
        }))
        args = argparse.Namespace(
            dual_rows=rows, dual_report=report,
            geometry_rows=None, geometry_report=None,
            cdec_rows=None, cdec_report=None)
        try:
            audit(args)
        except RuntimeError as error:
            assert "480 rows per proposal" in str(error)
        else:
            raise AssertionError("incomplete dual rows were accepted")
