#!/usr/bin/env python3
"""Audit the public MemoNav MP3D multi-goal benchmark without running policy.

The upstream MemoNav repository currently publishes episode JSONs but no
training/evaluation implementation.  This audit therefore records exactly
what is available, which MP3D assets are locally present, and the fields that
must be resolved before claiming a contract-compatible comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


SPLITS = ("1goal", "2goal", "3goal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError(f"{path}: expected a top-level episode list")
    return payload


def discover_assets(roots: list[Path]) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in ("*.glb", "*.basis.glb"):
            for path in sorted(root.rglob(suffix)):
                scene = path.name.removesuffix(".basis.glb").removesuffix(".glb")
                assets.setdefault(scene, path.resolve())
    return assets


def audit(repo: Path, asset_roots: list[Path]) -> dict:
    data_root = repo / "image-goal-nav-dataset/mp3d/test"
    if not data_root.is_dir():
        raise RuntimeError("MemoNav MP3D test dataset is missing")
    readme = repo / "README.md"
    source_files = sorted(repo.rglob("*.py"))
    assets = discover_assets(asset_roots)
    split_rows = {}
    all_scenes: set[str] = set()
    for split in SPLITS:
        files = sorted((data_root / split).glob("*.json.gz"))
        rows = []
        file_receipts = []
        for path in files:
            current = load_rows(path)
            rows.extend(current)
            file_receipts.append({
                "file": path.name,
                "episodes": len(current),
                "sha256": sha256_file(path),
            })
        expected_goals = int(split.removesuffix("goal"))
        scenes = {
            Path(str(row["scene_id"])).stem
            for row in rows
        }
        all_scenes.update(scenes)
        split_rows[split] = {
            "files": len(files),
            "episodes": len(rows),
            "scenes": len(scenes),
            "goal_count_matches": all(
                len(row.get("goals", [])) == expected_goals for row in rows
            ),
            "episodes_with_any_goal_rotation": sum(
                any("rotation" in goal for goal in row.get("goals", []))
                for row in rows
            ),
            "files_receipt": file_receipts,
        }
    present = sorted(scene for scene in all_scenes if scene in assets)
    missing = sorted(all_scenes - set(assets))
    return {
        "schema_version": "memonav_external_benchmark_readiness_v1_20260814",
        "scope": "episode_and_asset_readiness_only_no_policy_outcome",
        "upstream_repo": str(repo.resolve()),
        "upstream_readme_sha256": sha256_file(readme),
        "published_python_source_files": len(source_files),
        "upstream_evaluation_code_available": bool(source_files),
        "split_audit": split_rows,
        "benchmark_scenes": len(all_scenes),
        "asset_roots": [str(path.resolve()) for path in asset_roots],
        "locally_present_scenes": present,
        "locally_present_scene_count": len(present),
        "missing_scenes": missing,
        "missing_scene_count": len(missing),
        "full_scene_coverage": not missing,
        "contract_blockers": [
            "upstream evaluation implementation is not published",
            "episode goals contain positions but no goal-camera rotations",
            "published VGM-family evaluation uses panoramic observations",
        ],
        "direct_published_score_comparison_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.repo, args.asset_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: result[key] for key in (
            "benchmark_scenes", "locally_present_scene_count",
            "missing_scene_count", "full_scene_coverage",
            "upstream_evaluation_code_available",
            "direct_published_score_comparison_authorized",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
