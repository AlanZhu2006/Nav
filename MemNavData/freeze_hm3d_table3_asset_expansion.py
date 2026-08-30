#!/usr/bin/env python3
"""Freeze additional HM3D scene assets for the long-range capacity audit.

The freezer reads only GLB/NavMesh bytes and the already sealed parent scene
identities.  It never opens an episode, history, query, or policy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "hm3d_table3_asset_expansion_v1_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )


def freeze(
    *, parent_manifest: Path, expected_parent_sha256: str,
    asset_roots: list[Path], expected_count: int, out: Path,
) -> dict[str, Any]:
    require(sha256_file(parent_manifest) == expected_parent_sha256,
            "parent manifest SHA-256 changed")
    parent = json.loads(parent_manifest.read_text())
    parent_scenes = {str(scene) for scene in parent["scenes"]}
    require(parent_scenes, "parent manifest has no scenes")
    require(asset_roots, "no expansion asset roots provided")
    assets: dict[str, dict[str, Any]] = {}
    source_rows = []
    for root in sorted(path.resolve() for path in asset_roots):
        require(root.is_dir(), f"asset root missing: {root}")
        navmeshes = sorted(root.glob("*/*.basis.navmesh"))
        require(navmeshes, f"asset root has no HM3D navmeshes: {root}")
        source_rows.append({"root": str(root), "navmesh_files": len(navmeshes)})
        for navmesh in navmeshes:
            scene = navmesh.name.removesuffix(".basis.navmesh")
            require(scene and scene not in parent_scenes,
                    f"expansion scene overlaps sealed parent: {scene}")
            require(scene not in assets, f"duplicate expansion scene: {scene}")
            glb = navmesh.with_name(scene + ".basis.glb")
            require(glb.is_file(), f"missing GLB for {scene}")
            assets[scene] = {
                "directory": navmesh.parent.name,
                "glb_path": str(glb.resolve()),
                "glb_bytes": glb.stat().st_size,
                "glb_sha256": sha256_file(glb),
                "navmesh_path": str(navmesh.resolve()),
                "navmesh_bytes": navmesh.stat().st_size,
                "navmesh_sha256": sha256_file(navmesh),
            }
    require(len(assets) == expected_count,
            f"expansion scene count {len(assets)} != {expected_count}")
    scenes = sorted(assets)
    result = {
        "schema_version": SCHEMA,
        "scope": "additional HM3D assets for result-blind Table-3 capacity",
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": expected_parent_sha256,
        "parent_scene_count": len(parent_scenes),
        "source_roots": source_rows,
        "scene_count": len(scenes),
        "scenes": scenes,
        "assets": {scene: assets[scene] for scene in scenes},
        "scene_overlap_with_parent": 0,
        "episode_files_read": False,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "policy_evaluation_authorized": False,
    }
    _atomic_write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--asset-root", action="append", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        parent_manifest=args.parent_manifest,
        expected_parent_sha256=args.expected_parent_sha256,
        asset_roots=args.asset_root,
        expected_count=args.expected_count,
        out=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
