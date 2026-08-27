#!/usr/bin/env python3
"""Prepare a frozen fresh160 double-Revisit benchmark before policy evaluation.

Every genuine, native-controlled Goal-A success in the immutable fresh160 run
is attempted.  Goal-B/Goal-C construction uses only geometry, rendering and
co-visibility; no navigation arm is executed by this program.  A scene-balanced
round-robin then freezes exactly ``target_count`` constructible episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import build_shared_online_double_revisit_v2 as builder
from audit_shared_online_double_revisit import audit as audit_benchmark
from materialize_online_a_traces import (
    SCHEMA_VERSION as ONLINE_SCHEMA,
    discover_candidates,
    materialize_one,
    native_control_audit,
)


SCHEMA_VERSION = "shared_online_double_revisit_fresh_preparation_v1_20260813"
EXPECTED_UPSTREAM_SCHEMA = "revisit_fresh_confirmation_manifest_v1"
EXPECTED_CONSTRUCTION_FAILURE_PREFIXES = (
    "online-A trace has too few source-anchor candidates",
    "no route-negative online-A B/C source pair exists",
    "no source pair supports both route-negative V0 and V1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def construction_contract() -> dict[str, Any]:
    """The V2 contract frozen by the successful four-episode local gate."""
    return {
        "minimum_eligible_online_frame": 39,
        "source_anchor_end_margin_frames": 39,
        "source_anchor_stride_frames": 8,
        "minimum_anchor_gap_frames": 32,
        "minimum_leg_geodesic_m": 2.0,
        "target_leg_geodesic_m": 3.0,
        "reference_path_sample_step_m": 0.25,
        "reference_path_c_tail_max_covis": 0.08,
        "preferred_reference_path_c_tail_max_covis": 0.05,
        "v0_min_self_covis": 0.95,
        "v1_min_translation_m": 0.20,
        "v1_max_translation_m": 0.50,
        "v1_min_yaw_delta_deg": 10.0,
        "v1_max_yaw_delta_deg": 25.0,
        "v1_min_source_frame_covis": 0.45,
        "v1_min_max_online_a_covis": 0.50,
        "v1_max_max_online_a_covis": 0.98,
        "v1_max_argmax_gap_frames": 20,
        "v1_min_pixel_mae": 5.0,
    }


def is_expected_construction_failure(error: BaseException) -> bool:
    message = str(error)
    return isinstance(error, RuntimeError) and any(
        message.startswith(prefix)
        for prefix in EXPECTED_CONSTRUCTION_FAILURE_PREFIXES
    )


def round_robin_select(
    valid_rows: list[dict[str, Any]],
    scene_order: list[str],
    target_count: int,
) -> list[dict[str, Any]]:
    """Take one episode per scene per round, with stable episode ordering."""
    if target_count < 1:
        raise ValueError("target_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        grouped[str(row["scene"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (str(row["episode"]), int(row["candidate_index"])))

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < target_count:
        added = False
        for scene in scene_order:
            rows = grouped.get(scene, [])
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == target_count:
                    return selected
        if not added:
            break
        depth += 1
    return selected


def verify_source_episode(
    source_manifest: dict[str, Any],
    *,
    scene: str,
    episode: str,
    asset_root: Path,
    episode_root: Path,
) -> dict[str, str]:
    by_episode = {
        str(row["episode"]): row for row in source_manifest["episodes"][scene]
    }
    require(episode in by_episode, f"episode absent from source manifest: {scene}/{episode}")
    row = by_episode[episode]
    source = episode_root / scene / episode
    paths = {
        "metadata": source / "meta" / "gen_meta.json",
        "parquet": source / "data" / "chunk-000" / "episode_000000.parquet",
        "goal": source / "goal_image.jpg",
    }
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing source {name}: {path}")
        require(
            sha256_file(path) == str(row["files"][name]["sha256"]),
            f"source {name} changed: {scene}/{episode}",
        )
    asset = asset_root / scene / f"{scene}.glb"
    require(asset.is_file() and not asset.is_symlink(), f"missing scene asset: {asset}")
    require(
        sha256_file(asset) == str(source_manifest["assets"][scene]["sha256"]),
        f"scene asset changed: {scene}",
    )
    return {name: sha256_file(path) for name, path in paths.items()}


def write_online_manifest(root: Path, receipt: dict[str, Any]) -> str:
    payload = {
        "schema_version": ONLINE_SCHEMA,
        "purpose": "one audited online-A candidate for frozen B/C construction",
        "selection": {
            "requested_count": 1,
            "eligible_count": 1,
            "all_goal_a_successes_attempted_by_parent": True,
            "goals_frozen": False,
        },
        "episodes": [receipt],
    }
    path = root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    digest = sha256_file(path)
    (root / "manifest.json.sha256").write_text(digest + "  manifest.json\n")
    return digest


def remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~0o222)
    mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, mode & ~0o222)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = args.original_run_root / "data_manifest.json"
    require(source_manifest_path.is_file(), "fresh160 source manifest is missing")
    source_manifest_sha = sha256_file(source_manifest_path)
    require(
        source_manifest_sha == args.expected_source_manifest_sha,
        "fresh160 source manifest SHA changed",
    )
    source_manifest = json.loads(source_manifest_path.read_text())
    # Historical manifests used more than one descriptive schema string.  The
    # immutable SHA and structural checks below are the authoritative identity.
    require(
        isinstance(source_manifest.get("scenes"), list)
        and isinstance(source_manifest.get("episodes"), dict),
        "unexpected fresh160 manifest structure",
    )
    scene_order = [str(scene) for scene in source_manifest["scenes"]]
    require(len(scene_order) == 20 and len(set(scene_order)) == 20,
            "fresh160 must contain 20 distinct scenes")
    asset_root = Path(source_manifest["paths"]["asset_root"])
    episode_root = Path(source_manifest["paths"]["episode_root"])
    trace_root = args.original_run_root / "scenes"
    require(trace_root.is_dir(), "fresh160 trace root is missing")

    # margin=0/gap=1/score=0 is deliberately non-selective: construction, not
    # this diagnostic, decides whether two formal Revisit goals exist.
    candidates = discover_candidates(
        trace_root, margin=0, min_gap=1, min_score_m=0.0
    )
    scene_rank = {scene: index for index, scene in enumerate(scene_order)}
    candidates.sort(
        key=lambda row: (scene_rank.get(row.scene, 10**9), row.episode, str(row.path))
    )
    require(
        len(candidates) == args.expected_goal_a_successes,
        f"expected {args.expected_goal_a_successes} native Goal-A successes, "
        f"found {len(candidates)}",
    )
    require(
        all(row.scene in scene_rank for row in candidates),
        "trace contains a scene outside the frozen source manifest",
    )

    if args.out.exists():
        raise FileExistsError(f"preparation output already exists: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "INCOMPLETE").write_text(
        "No evaluator may consume this directory while this marker exists.\n"
    )
    candidates_root = args.out / "candidates"
    candidates_root.mkdir()
    contract = construction_contract()
    statuses: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        scene = candidate.scene
        episode = candidate.episode
        require(native_control_audit(candidate.payload)["ok"],
                f"Goal-A trace was not native-controlled: {scene}/{episode}")
        source_hashes = verify_source_episode(
            source_manifest,
            scene=scene,
            episode=episode,
            asset_root=asset_root,
            episode_root=episode_root,
        )
        key = f"{index:03d}_{scene}_{episode}"
        candidate_root = candidates_root / key
        online_root = candidate_root / "online"
        online_episode = online_root / scene / episode
        online_episode.mkdir(parents=True)
        receipt = materialize_one(
            candidate,
            asset_root=asset_root,
            episode_root=episode_root,
            destination=online_episode,
        )
        online_manifest_sha = write_online_manifest(online_root, receipt)
        benchmark_episode = candidate_root / "benchmark" / scene / episode
        benchmark_episode.mkdir(parents=True)
        status = {
            "candidate_index": index,
            "scene": scene,
            "episode": episode,
            "trace_path": str(candidate.path.resolve()),
            "trace_sha256": sha256_file(candidate.path),
            "online_manifest_sha256": online_manifest_sha,
            "online_steps": len(candidate.payload["poses"]),
            "source_hashes": source_hashes,
            "constructible": False,
            "construction_failure": None,
        }
        try:
            benchmark = builder.build_episode(
                online_episode, benchmark_episode, contract
            )
        except BaseException as error:
            if not is_expected_construction_failure(error):
                status["unexpected_error"] = traceback.format_exc()
                statuses.append(status)
                raise
            status["construction_failure"] = str(error)
            shutil.rmtree(candidate_root)
        else:
            status["constructible"] = True
            status["benchmark_sha256"] = benchmark["benchmark_sha256"]
            status["candidate_root"] = str(candidate_root.resolve())
            valid_rows.append(status)
        statuses.append(status)
        print(
            json.dumps(
                {
                    "candidate": index,
                    "scene": scene,
                    "episode": episode,
                    "constructible": status["constructible"],
                    "reason": status["construction_failure"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected = round_robin_select(valid_rows, scene_order, args.target_count)
    require(
        len(selected) == args.target_count,
        f"only {len(selected)} constructible episodes for target {args.target_count}",
    )
    selected_scenes = {str(row["scene"]) for row in selected}
    require(
        len(selected_scenes) >= args.minimum_selected_scenes,
        f"selection spans only {len(selected_scenes)} scenes; require "
        f"{args.minimum_selected_scenes}",
    )
    selected_keys = {
        (str(row["scene"]), str(row["episode"])) for row in selected
    }

    benchmark_root = args.out / "benchmark"
    benchmark_root.mkdir()
    manifest_episodes = []
    for row in selected:
        candidate_root = Path(str(row["candidate_root"]))
        scene, episode = str(row["scene"]), str(row["episode"])
        source = candidate_root / "benchmark" / scene / episode
        destination = benchmark_root / scene / episode
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        payload = json.loads((source / "benchmark.json").read_text())
        payload["benchmark_sha256"] = sha256_file(source / "benchmark.json")
        manifest_episodes.append(payload)

    manifest = {
        "schema_version": builder.SCHEMA_VERSION,
        "purpose": (
            "fresh160 internal, scene-balanced double-Revisit benchmark; "
            "frozen before all four-arm navigation outcomes"
        ),
        "source_fresh160_manifest_sha256": source_manifest_sha,
        "contract": contract,
        "selection": {
            "target_count": args.target_count,
            "minimum_selected_scenes": args.minimum_selected_scenes,
            "candidate_population": "all native-controlled Goal-A successes",
            "candidate_count": len(candidates),
            "constructible_count": len(valid_rows),
            "selection_rule": (
                "scene order from immutable fresh160 manifest; one lexicographic "
                "episode per scene per round until target_count"
            ),
            "selected_scene_count": len(selected_scenes),
            "no_navigation_outcomes_observed": True,
        },
        "episodes": manifest_episodes,
    }
    manifest_path = benchmark_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    manifest_sha = sha256_file(manifest_path)
    (benchmark_root / "manifest.json.sha256").write_text(
        manifest_sha + "  manifest.json\n"
    )
    benchmark_audit = audit_benchmark(benchmark_root)
    require(benchmark_audit["ok"], "independent benchmark audit failed")
    require(benchmark_audit["episodes"] == args.target_count,
            "benchmark audit episode count changed")

    for candidate_dir in list(candidates_root.iterdir()):
        parts = candidate_dir.name.split("_", 2)
        if len(parts) != 3:
            raise RuntimeError(f"unexpected candidate directory: {candidate_dir}")
        key = (parts[1], parts[2])
        if key not in selected_keys:
            shutil.rmtree(candidate_dir)
        else:
            shutil.rmtree(candidate_dir / "benchmark")

    report = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "candidate_count": len(candidates),
        "constructible_count": len(valid_rows),
        "construction_failure_count": len(candidates) - len(valid_rows),
        "target_count": args.target_count,
        "selected_scene_count": len(selected_scenes),
        "selected": [
            {
                "selection_index": index,
                "candidate_index": int(row["candidate_index"]),
                "scene": str(row["scene"]),
                "episode": str(row["episode"]),
            }
            for index, row in enumerate(selected)
        ],
        "candidate_status": statuses,
        "benchmark_manifest_sha256": manifest_sha,
        "benchmark_audit": benchmark_audit,
        "causal_seal": {
            "navigation_arms_executed": 0,
            "selection_observed_navigation_outcomes": False,
            "next_allowed_operation": "four_arm_closed_loop_evaluation",
        },
    }
    report_path = args.out / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.out / "preparation_report.json.sha256").write_text(
        sha256_file(report_path) + "  preparation_report.json\n"
    )
    (args.out / "benchmark_audit.json").write_text(
        json.dumps(benchmark_audit, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    (args.out / "INCOMPLETE").unlink()
    (args.out / "SEALED").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark_manifest_sha256": manifest_sha,
                "episodes": args.target_count,
                "scenes": len(selected_scenes),
                "navigation_outcomes_observed": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    remove_write_bits(args.out)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--expected-goal-a-successes", type=int, default=120)
    parser.add_argument("--minimum-selected-scenes", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_count < 1 or args.expected_goal_a_successes < args.target_count:
        raise ValueError("invalid target/candidate counts")
    report = prepare(args)
    print(
        json.dumps(
            {
                "status": "sealed",
                "benchmark_manifest_sha256": report[
                    "benchmark_manifest_sha256"
                ],
                "candidate_count": report["candidate_count"],
                "constructible_count": report["constructible_count"],
                "episodes": report["target_count"],
                "scenes": report["selected_scene_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
