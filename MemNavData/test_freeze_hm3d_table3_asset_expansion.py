from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeze_hm3d_table3_asset_expansion import freeze, sha256_file


def _asset(root: Path, directory: str, scene: str) -> None:
    target = root / directory
    target.mkdir(parents=True)
    (target / f"{scene}.basis.glb").write_bytes((scene + "-glb").encode())
    (target / f"{scene}.basis.navmesh").write_bytes((scene + "-nav").encode())


def test_freeze_is_disjoint_hash_bound_and_outcome_blind(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"scenes": ["parent_scene"]}))
    first, second = tmp_path / "first", tmp_path / "second"
    _asset(first, "001-scene_b", "scene_b")
    _asset(second, "002-scene_a", "scene_a")
    out = tmp_path / "expansion.json"
    result = freeze(
        parent_manifest=parent,
        expected_parent_sha256=sha256_file(parent),
        asset_roots=[first, second], expected_count=2, out=out,
    )
    assert result["scenes"] == ["scene_a", "scene_b"]
    assert result["scene_overlap_with_parent"] == 0
    assert result["navigation_outcomes_read"] is False
    assert out.with_name("expansion.json.sha256").is_file()


def test_overlap_or_count_mismatch_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"scenes": ["same"]}))
    root = tmp_path / "assets"
    _asset(root, "001-same", "same")
    with pytest.raises(RuntimeError, match="overlaps"):
        freeze(
            parent_manifest=parent,
            expected_parent_sha256=sha256_file(parent),
            asset_roots=[root], expected_count=1,
            out=tmp_path / "bad.json",
        )
