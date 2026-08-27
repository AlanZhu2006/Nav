#!/usr/bin/env python3
"""Fail-closed finalization of the feasibility-limited fresh20 benchmark.

The preregistered fresh40 preparation attempted every one of the 120 native
Goal-A successes, constructed 20 episodes, and then stopped before writing a
manifest because its power target was not met.  This program does *not*
reconstruct, relax, or select using navigation outcomes.  It verifies the
complete failed-job record and causally seals exactly those 20 already-built
episodes as a separately labelled, lower-power internal gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import build_shared_online_double_revisit_v2 as builder
from audit_shared_online_double_revisit import audit as audit_benchmark
from prepare_shared_online_double_revisit_fresh import (
    construction_contract,
    round_robin_select,
)


SCHEMA_VERSION = "shared_online_double_revisit_fresh20_finalization_v1_20260813"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~0o222)
    mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, mode & ~0o222)


def parse_statuses(path: Path, expected_candidates: int) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "candidate" not in row:
            continue
        index = int(row["candidate"])
        require(index not in by_index, f"duplicate candidate status {index}")
        require(
            set(row) == {"candidate", "constructible", "episode", "reason", "scene"},
            f"unexpected status fields for candidate {index}",
        )
        by_index[index] = row
    require(
        set(by_index) == set(range(expected_candidates)),
        "failed preparation log does not contain the complete candidate population",
    )
    return [by_index[index] for index in range(expected_candidates)]


def verify_candidate(
    candidates_root: Path, row: dict[str, Any]
) -> dict[str, Any]:
    index = int(row["candidate"])
    scene = str(row["scene"])
    episode = str(row["episode"])
    root = candidates_root / f"{index:03d}_{scene}_{episode}"
    require(root.is_dir() and not root.is_symlink(), f"candidate directory missing: {root}")
    online_root = root / "online"
    online_episode = online_root / scene / episode
    benchmark_episode = root / "benchmark" / scene / episode
    for path in (
        online_root / "manifest.json",
        online_root / "manifest.json.sha256",
        online_episode / "receipt.json",
        online_episode / "online_a_trace.json",
        benchmark_episode / "benchmark.json",
    ):
        require(path.is_file() and not path.is_symlink(), f"candidate asset missing: {path}")
    stored_online_sha = (online_root / "manifest.json.sha256").read_text().split()[0]
    require(
        stored_online_sha == sha256_file(online_root / "manifest.json"),
        f"online manifest changed: {scene}/{episode}",
    )
    online_manifest = json.loads((online_root / "manifest.json").read_text())
    require(len(online_manifest.get("episodes", [])) == 1, "bad online manifest")
    receipt = online_manifest["episodes"][0]
    require(
        (str(receipt["scene"]), str(receipt["episode"])) == (scene, episode),
        f"online identity changed: {scene}/{episode}",
    )
    benchmark = json.loads((benchmark_episode / "benchmark.json").read_text())
    require(
        (str(benchmark["scene"]), str(benchmark["episode"])) == (scene, episode),
        f"benchmark identity changed: {scene}/{episode}",
    )
    require(
        Path(benchmark["source_online_episode"]).resolve() == online_episode.resolve(),
        f"benchmark points to a different online history: {scene}/{episode}",
    )
    return {
        "candidate_index": index,
        "scene": scene,
        "episode": episode,
        "candidate_root": str(root.resolve()),
        "online_manifest_sha256": stored_online_sha,
        "benchmark_sha256": sha256_file(benchmark_episode / "benchmark.json"),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    prepared = args.prepared_root.resolve()
    run_root = prepared.parent
    require(prepared.is_dir() and not prepared.is_symlink(), "prepared root missing")
    require((prepared / "INCOMPLETE").is_file(), "original INCOMPLETE marker missing")
    require(not (prepared / "SEALED").exists(), "benchmark was already sealed")
    require(not (prepared / "benchmark").exists(), "benchmark output already exists")
    require(not (prepared / "benchmark.partial").exists(), "partial finalization exists")
    require(not (run_root / "scenes").exists(), "navigation outputs already exist")
    for path in (
        args.source_manifest,
        args.failed_stdout,
        args.failed_stderr,
        args.original_submission,
    ):
        require(path.is_file() and not path.is_symlink(), f"missing immutable evidence: {path}")
    require(
        sha256_file(args.source_manifest) == args.expected_source_manifest_sha,
        "source fresh160 manifest changed",
    )
    expected_error = (
        f"RuntimeError: only {args.expected_constructible} constructible episodes "
        f"for target {args.preregistered_target}"
    )
    require(expected_error in args.failed_stderr.read_text(errors="replace"),
            "failed-job reason differs from the declared power failure")
    statuses = parse_statuses(args.failed_stdout, args.expected_candidates)
    constructible_statuses = [row for row in statuses if row["constructible"] is True]
    require(
        len(constructible_statuses) == args.expected_constructible,
        "constructible count changed",
    )
    candidates_root = prepared / "candidates"
    actual_dirs = {path.name for path in candidates_root.iterdir() if path.is_dir()}
    expected_dirs = {
        f"{int(row['candidate']):03d}_{row['scene']}_{row['episode']}"
        for row in constructible_statuses
    }
    require(actual_dirs == expected_dirs, "candidate directories differ from failed-job log")
    valid_rows = [verify_candidate(candidates_root, row) for row in constructible_statuses]

    source_manifest = json.loads(args.source_manifest.read_text())
    scene_order = [str(scene) for scene in source_manifest["scenes"]]
    selected = round_robin_select(valid_rows, scene_order, args.expected_constructible)
    require(len(selected) == args.expected_constructible, "selection count changed")
    selected_scenes = {str(row["scene"]) for row in selected}
    require(
        len(selected_scenes) == args.expected_scenes,
        f"selected scene count changed: {len(selected_scenes)}",
    )

    benchmark_partial = prepared / "benchmark.partial"
    benchmark_partial.mkdir()
    manifest_episodes = []
    for row in selected:
        candidate_root = Path(str(row["candidate_root"]))
        scene, episode = str(row["scene"]), str(row["episode"])
        source = candidate_root / "benchmark" / scene / episode
        destination = benchmark_partial / scene / episode
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        payload = json.loads((source / "benchmark.json").read_text())
        payload["benchmark_sha256"] = sha256_file(source / "benchmark.json")
        manifest_episodes.append(payload)

    evidence = {
        "failed_preparation_job_id": args.failed_job_id,
        "failed_stdout": str(args.failed_stdout.resolve()),
        "failed_stdout_sha256": sha256_file(args.failed_stdout),
        "failed_stderr": str(args.failed_stderr.resolve()),
        "failed_stderr_sha256": sha256_file(args.failed_stderr),
        "original_submission": str(args.original_submission.resolve()),
        "original_submission_sha256": sha256_file(args.original_submission),
        "verified_failure": expected_error,
    }
    manifest = {
        "schema_version": builder.SCHEMA_VERSION,
        "purpose": (
            "feasibility-limited fresh20 internal double-Revisit gate; exact "
            "strict constructs from the failed preregistered fresh40 preparation; "
            "frozen before all four-arm navigation outcomes"
        ),
        "source_fresh160_manifest_sha256": args.expected_source_manifest_sha,
        "contract": construction_contract(),
        "selection": {
            "target_count": args.expected_constructible,
            "preregistered_power_target": args.preregistered_target,
            "formal_power_target_met": False,
            "candidate_population": "all native-controlled Goal-A successes",
            "candidate_count": args.expected_candidates,
            "constructible_count": args.expected_constructible,
            "selection_rule": (
                "all strict constructs retained; manifest ordering is the frozen "
                "fresh160 scene round-robin used by the preregistered preparer"
            ),
            "selected_scene_count": len(selected_scenes),
            "no_navigation_outcomes_observed": True,
            "failed_preparation_evidence": evidence,
        },
        "episodes": manifest_episodes,
    }
    manifest_path = benchmark_partial / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    manifest_sha = sha256_file(manifest_path)
    (benchmark_partial / "manifest.json.sha256").write_text(
        manifest_sha + "  manifest.json\n"
    )
    audit = audit_benchmark(benchmark_partial)
    require(audit["ok"], "independent benchmark audit failed")
    require(audit["episodes"] == args.expected_constructible, "audit count changed")
    benchmark_partial.rename(prepared / "benchmark")

    inferential_scope = (
        "feasibility-limited internal fresh20 architecture/causal gate on 13 "
        "previously consumed fresh160 scenes; the preregistered fresh40 power "
        "target failed and this is neither powered confirmation nor paper-final"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": args.expected_source_manifest_sha,
        "candidate_count": args.expected_candidates,
        "constructible_count": args.expected_constructible,
        "construction_failure_count": args.expected_candidates - args.expected_constructible,
        "target_count": args.expected_constructible,
        "preregistered_power_target": args.preregistered_target,
        "formal_power_target_met": False,
        "inferential_scope": inferential_scope,
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
        "failed_preparation_evidence": evidence,
        "benchmark_manifest_sha256": manifest_sha,
        "benchmark_audit": audit,
        "causal_seal": {
            "navigation_arms_executed": 0,
            "selection_observed_navigation_outcomes": False,
            "next_allowed_operation": "four_arm_closed_loop_feasibility_evaluation",
        },
    }
    report_path = prepared / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (prepared / "preparation_report.json.sha256").write_text(
        sha256_file(report_path) + "  preparation_report.json\n"
    )
    (prepared / "benchmark_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (prepared / "POWER_AMENDMENT_FRESH20.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "reason": "strict constructible population smaller than power target",
                "preregistered_target": args.preregistered_target,
                "strict_constructible_population": args.expected_constructible,
                "formal_power_target_met": False,
                "inferential_scope": inferential_scope,
                "failed_preparation_evidence": evidence,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    # The copied benchmark is now authoritative; retain only causal online-A
    # histories under candidates, exactly as the original preparer would.
    for row in selected:
        shutil.rmtree(Path(str(row["candidate_root"])) / "benchmark")
    (prepared / "INCOMPLETE").unlink()
    (prepared / "SEALED").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark_manifest_sha256": manifest_sha,
                "episodes": args.expected_constructible,
                "scenes": len(selected_scenes),
                "formal_power_target_met": False,
                "navigation_outcomes_observed": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    remove_write_bits(prepared)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha", required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--original-submission", type=Path, required=True)
    parser.add_argument("--failed-job-id", type=int, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--expected-constructible", type=int, default=20)
    parser.add_argument("--expected-scenes", type=int, default=13)
    parser.add_argument("--preregistered-target", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(
        0 < args.expected_constructible < args.preregistered_target
        <= args.expected_candidates,
        "invalid feasibility/power counts",
    )
    report = finalize(args)
    print(
        json.dumps(
            {
                "status": "sealed_feasibility_limited",
                "benchmark_manifest_sha256": report["benchmark_manifest_sha256"],
                "episodes": report["target_count"],
                "scenes": report["selected_scene_count"],
                "formal_power_target_met": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
