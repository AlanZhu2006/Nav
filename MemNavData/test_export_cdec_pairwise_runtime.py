import json

import pytest

from MemNavData.export_cdec_pairwise_runtime import build_artifact
from MemNavData.patch_temporal_router import directional_patch_feature_names
from MemNavData.train_cdec_pairwise_ranker_oof import SCHEMA_VERSION


def report():
    count = len(directional_patch_feature_names())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "development_or_blind_read": False,
            "activation_or_NULL_learned": False,
            "groups": "scene",
        },
        "coverage": {"scenes": 40, "sessions": 480},
        "inputs": {
            "rows_csv_sha256": "a" * 64,
            "patch_cache_sha256": "b" * 64,
        },
        "selection_artifact": {"sha256": "c" * 64},
        "deployment_fit_on_all_train_scenes": {
            "selected_C": 3.0,
            "feature_names": list(directional_patch_feature_names()),
            "coefficient": [0.0] * count,
            "mean": [0.0] * count,
            "scale": [1.0] * count,
        },
    }


def test_export_is_unapproved_and_preserves_authority(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()))
    artifact = build_artifact(
        report(), report_path=path, report_sha256="d" * 64)
    assert artifact["deployment_approved"] is False
    assert artifact["runtime_semantics"]["cascade"] == (
        "geometry_proposal_then_learned_on_certificate_reject")
    assert artifact["runtime_semantics"][
        "accepted_geometry_can_be_overridden"] is False


def test_export_refuses_any_development_or_blind_read(tmp_path):
    bad = report()
    bad["protocol"]["development_or_blind_read"] = True
    with pytest.raises(ValueError):
        build_artifact(
            bad, report_path=tmp_path / "report.json",
            report_sha256="d" * 64)
