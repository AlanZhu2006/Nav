from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_role_pair_length_constructibility import audit_manifest, sha256_file


def _write_manifest(path: Path, distances: list[tuple[str, float]]) -> None:
    queries = [
        {
            "analysis_role": role,
            "query_id": f"query_{index}",
            "geodesic_from_a_end_m": distance,
        }
        for index, (role, distance) in enumerate(distances)
    ]
    payload = {
        "schema_version": "shared_online_role_pair_v1_20260814",
        "contract": {
            "runtime_role_visibility": "none",
            "minimum_query_geodesic_m": 2.0,
            "maximum_query_geodesic_m": 50.0,
        },
        "episodes": [{
            "scene": "scene_a",
            "episode": "episode_0000",
            "pairs": [{"pair_id": "pair_00", "queries": queries}],
        }],
    }
    path.write_text(json.dumps(payload))


def test_requested_bin_boundaries_and_roles(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [
        ("novel", 19.999), ("revisit", 20.0),
        ("novel", 29.999), ("revisit", 30.0),
        ("novel", 50.0),
    ])
    result = audit_manifest(manifest, sha256_file(manifest))
    assert result["requested_bins"]["0_to_20_m"]["all"] == 1
    assert result["requested_bins"]["20_to_30_m"]["all"] == 2
    assert result["requested_bins"]["30_to_50_m"]["all"] == 2
    assert result["table3_constructible_from_this_population"] is True
    assert result["navigation_outcomes_read"] is False


def test_missing_long_bins_fail_constructibility_not_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [("novel", 8.9), ("revisit", 3.1)])
    result = audit_manifest(manifest)
    assert result["verified"] is True
    assert result["table3_constructible_from_this_population"] is False
    assert result["requested_bins"]["0_to_20_m"]["all"] == 2


def test_hash_or_role_change_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [("novel", 4.0), ("revisit", 3.0)])
    with pytest.raises(RuntimeError, match="SHA-256"):
        audit_manifest(manifest, "0" * 64)
    payload = json.loads(manifest.read_text())
    payload["episodes"][0]["pairs"][0]["queries"][0]["analysis_role"] = "oracle"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="analysis role"):
        audit_manifest(manifest)
