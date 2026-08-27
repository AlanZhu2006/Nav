#!/usr/bin/env python3
"""Freeze receipts for the consumed-pool CDEC closed-loop comparison.

The preparation step deliberately treats the previously generated Goal-A
traces as opaque byte strings.  It verifies their exact names and hashes, but
does not deserialize a trace, inspect a target, or inspect a navigation
outcome.  Runtime replay remains responsible for the full semantic trace
validation before an episode can execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


SCHEMA_VERSION = "cdec_consumed_closed_loop_preparation_v1_20260813"
TRACE_RECEIPT_SCHEMA_VERSION = (
    "cdec_consumed_goal_a_trace_receipt_v1_20260813")
EXPECTED_SCENES = 20
EXPECTED_EPISODES_PER_SCENE = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(),
            f"not a physical file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    audit = manifest.get("audit")
    guards = manifest.get("data_role_guards")
    require(isinstance(audit, Mapping) and audit.get("status") == "ok",
            "source manifest audit failed")
    require(audit.get("development_read") is False
            and audit.get("blind_read") is False,
            "source manifest read development or blind data")
    require(isinstance(guards, Mapping)
            and guards.get("blind_allowed") is False,
            "source manifest permits blind access")
    scenes = manifest.get("scenes")
    require(isinstance(scenes, list) and len(scenes) == EXPECTED_SCENES,
            "comparison requires exactly 20 scenes")
    require(len(set(map(str, scenes))) == EXPECTED_SCENES,
            "source manifest contains duplicate scenes")
    raw_episodes = manifest.get("episodes")
    require(isinstance(raw_episodes, Mapping), "manifest episodes are missing")
    episodes: dict[str, list[str]] = {}
    for raw_scene in scenes:
        scene = str(raw_scene)
        rows = raw_episodes.get(scene)
        require(isinstance(rows, list)
                and len(rows) == EXPECTED_EPISODES_PER_SCENE,
                f"{scene}: expected eight episodes")
        ids = []
        for row in rows:
            require(isinstance(row, Mapping), f"{scene}: malformed episode row")
            episode = row.get("episode")
            require(isinstance(episode, str)
                    and episode.startswith("episode_"),
                    f"{scene}: malformed episode id")
            ids.append(episode)
        require(len(set(ids)) == len(ids), f"{scene}: duplicate episode ids")
        episodes[scene] = ids
    return list(map(str, scenes)), episodes


def prepare(
    *,
    source_manifest: Path,
    expected_manifest_sha256: str,
    source_dependency_receipt: Path,
    expected_dependency_receipt_sha256: str,
    trace_run_root: Path,
    trace_run_report: Path,
    expected_trace_run_report_sha256: str,
    run_root: Path,
) -> dict[str, Any]:
    require(not run_root.exists(), f"run root already exists: {run_root}")
    manifest_sha = sha256_file(source_manifest)
    require(manifest_sha == expected_manifest_sha256,
            "source manifest SHA256 changed")
    dependency_sha = sha256_file(source_dependency_receipt)
    require(dependency_sha == expected_dependency_receipt_sha256,
            "source dependency receipt SHA256 changed")
    trace_report_sha = sha256_file(trace_run_report)
    require(trace_report_sha == expected_trace_run_report_sha256,
            "trace-source report SHA256 changed")
    require(trace_run_root.is_dir() and not trace_run_root.is_symlink(),
            "trace run root is not a physical directory")
    trace_manifest = trace_run_root / "data_manifest.json"
    require(sha256_file(trace_manifest) == manifest_sha,
            "trace source and requested manifest differ")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    scenes, episodes = validate_manifest(manifest)
    dependency = json.loads(
        source_dependency_receipt.read_text(encoding="utf-8"))
    require(dependency.get("manifest_sha256") == manifest_sha,
            "dependency receipt is not bound to the manifest")
    required_dependencies = {
        "gatecurr600", "navdp_checkpoint", "lingbot_map_long"}
    require(set((dependency.get("dependencies") or {}).keys())
            == required_dependencies,
            "dependency receipt universe changed")

    trace_scenes: dict[str, Any] = {}
    for index, scene in enumerate(scenes):
        scene_root = trace_run_root / "scenes" / f"{index:02d}_{scene}"
        trace_root = scene_root / "trace_source"
        require(trace_root.is_dir() and not trace_root.is_symlink(),
                f"missing physical trace root: {scene}")
        summary = trace_root / "summary.json"
        summary_sha = sha256_file(summary)
        expected_names = {
            f"{episode}_leg1_trace.json" for episode in episodes[scene]}
        actual_names = {
            path.name for path in trace_root.glob("*_leg1_trace.json")
            if path.is_file() and not path.is_symlink()}
        require(actual_names == expected_names,
                f"{scene}: Goal-A trace identity/count changed")
        trace_scenes[scene] = {
            "scene_index": index,
            "trace_root": str(trace_root.resolve()),
            "summary_sha256": summary_sha,
            # sha256_file is intentionally the only operation performed on
            # each trace; no JSON decoder is called here.
            "episodes": {
                episode: sha256_file(
                    trace_root / f"{episode}_leg1_trace.json")
                for episode in episodes[scene]
            },
        }

    trace_receipt = {
        "schema_version": TRACE_RECEIPT_SCHEMA_VERSION,
        "scope": "opaque hashes of consumed-pool Goal-A replay traces",
        "manifest_sha256": manifest_sha,
        "trace_source_run_root": str(trace_run_root.resolve()),
        "trace_source_report_sha256": trace_report_sha,
        "trace_payload_decoded": False,
        "trace_bytes_hashed_only": True,
        "episode_target_or_outcome_fields_accessed": False,
        # Backward-compatible, explicit semantic guard consumed by the
        # fail-closed summarizer.
        "episode_target_or_outcome_read": False,
        "development_read": False,
        "blind_read": False,
        "scenes": trace_scenes,
    }

    run_root.mkdir(parents=True)
    try:
        artifacts = {
            "data_manifest.json": source_manifest.read_bytes(),
            "dependency_receipt.json": source_dependency_receipt.read_bytes(),
            "trace_receipt.json": json_bytes(trace_receipt),
        }
        artifact_shas = {}
        for name, payload in artifacts.items():
            target = run_root / name
            write_new(target, payload)
            digest = hashlib.sha256(payload).hexdigest()
            artifact_shas[name] = digest
            write_new(run_root / f"{name}.sha256",
                      f"{digest}  {name}\n".encode())
        preparation = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared",
            "scenes": len(scenes),
            "episodes": sum(map(len, episodes.values())),
            "development_read": False,
            "blind_read": False,
            "target_or_outcome_fields_accessed_while_freezing_traces": False,
            "artifact_sha256": artifact_shas,
        }
        write_new(run_root / "preparation.json", json_bytes(preparation))
        preparation_sha = sha256_file(run_root / "preparation.json")
        write_new(run_root / "preparation.json.sha256",
                  f"{preparation_sha}  preparation.json\n".encode())
    except BaseException:
        shutil.rmtree(run_root)
        raise
    return preparation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-dependency-receipt", type=Path, required=True)
    parser.add_argument("--expected-dependency-receipt-sha256", required=True)
    parser.add_argument("--trace-run-root", type=Path, required=True)
    parser.add_argument("--trace-run-report", type=Path, required=True)
    parser.add_argument("--expected-trace-run-report-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        source_manifest=args.source_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        source_dependency_receipt=args.source_dependency_receipt,
        expected_dependency_receipt_sha256=(
            args.expected_dependency_receipt_sha256),
        trace_run_root=args.trace_run_root,
        trace_run_report=args.trace_run_report,
        expected_trace_run_report_sha256=(
            args.expected_trace_run_report_sha256),
        run_root=args.run_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
