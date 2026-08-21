import csv
import json

from MemNavData.compare_pi3x_spatial_proof_to_certificate import (
    _mcnemar_exact,
    compare,
)


def _pnp(*, accepted: bool, error: float) -> str:
    return json.dumps([{
        "offset": 0,
        "pnp_lightglue": {
            "status": "ok" if accepted else "ransac_failed",
            "inliers": 20 if accepted else 0,
            "query_inlier_coverage": 0.1 if accepted else 0.0,
            "reference_inlier_coverage": 0.1 if accepted else 0.0,
            "reprojection_rmse_px": 1.0 if accepted else None,
            "relative_position_direction_error_deg": error,
        },
    }])


def test_endpoint_aligned_comparison(tmp_path):
    certificate = tmp_path / "certificate.csv"
    learned = tmp_path / "learned.csv"
    certificate_rows = [
        {
            "session_id": "positive",
            "scene": "s0",
            "session_has_positive": "True",
            "session_is_strict_no_match": "False",
            "hypotheses_json": _pnp(accepted=True, error=10.0),
        },
        {
            "session_id": "negative",
            "scene": "s1",
            "session_has_positive": "False",
            "session_is_strict_no_match": "True",
            "hypotheses_json": _pnp(accepted=True, error=120.0),
        },
        {
            "session_id": "ambiguous",
            "scene": "s2",
            "session_has_positive": "False",
            "session_is_strict_no_match": "False",
            "hypotheses_json": _pnp(accepted=False, error=45.0),
        },
    ]
    learned_rows = [
        {
            "session_id": "positive",
            "scene": "s0",
            "selected": "True",
            "session_label_reporting_only": "1",
            "navigation_action_label_reporting_only": "1",
            "bearing_error_deg_reporting_only": "5.0",
            "accepted": "True",
        },
        {
            "session_id": "negative",
            "scene": "s1",
            "selected": "True",
            "session_label_reporting_only": "0",
            "navigation_action_label_reporting_only": "0",
            "bearing_error_deg_reporting_only": "20.0",
            "accepted": "False",
        },
        {
            "session_id": "ambiguous",
            "scene": "s2",
            "selected": "True",
            "session_label_reporting_only": "-1",
            "navigation_action_label_reporting_only": "-1",
            "bearing_error_deg_reporting_only": "45.0",
            "accepted": "False",
        },
    ]
    for path, rows in ((certificate, certificate_rows), (learned, learned_rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = compare(certificate, learned)
    assert result["population"] == {
        "sessions": 3,
        "scenes": 3,
        "positive_sessions": 1,
        "strict_negative_sessions": 1,
        "ambiguous_sessions_excluded": 1,
    }
    assert result["learned_spatial_proof"]["correct_positive_accepts"] == 1
    assert result["learned_spatial_proof"]["strict_negative_false_accepts"] == 0
    assert result["certificate"]["correct_positive_accepts"] == 1
    assert result["certificate"]["strict_negative_false_accepts"] == 1
    assert result["certificate"]["accepted_bearing_catastrophic_gt90deg"] == 1


def test_exact_mcnemar():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(12, 0) == 0.00048828125
