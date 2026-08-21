#!/usr/bin/env python3
"""Freeze the complete train40 relocalization session universe.

The challenge deliberately selects every eligible train session.  Selection
therefore cannot depend on co-visibility labels, geometry outcomes, or method
scores.  The input geometry table is used only as an already-audited inventory
of session identities; candidate-level labels are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


SCHEMA_VERSION = "train40_certificate_challenge_manifest_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_universe_sha256(sessions: list[str]) -> str:
    payload = "".join(f"{session}\n" for session in sessions).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
        evidence: pd.DataFrame, *, evidence_sha256: str,
        teacher_sha256: str, expected_sessions: int,
        expected_scenes: int) -> dict:
    required = {"session_id", "scene", "kind", "split_role"}
    missing = required - set(evidence.columns)
    if missing:
        raise RuntimeError(f"inventory lacks columns: {sorted(missing)}")
    if set(evidence["split_role"].astype(str)) != {"train"}:
        raise RuntimeError("non-train rows entered the challenge inventory")
    if set(evidence["kind"].astype(str)) != {
            "manifest_causal_goal_localization_train"}:
        raise RuntimeError("unexpected task kind entered the challenge inventory")
    identity = evidence[["session_id", "scene"]].drop_duplicates().copy()
    scene_per_session = identity.groupby("session_id")["scene"].nunique()
    if not scene_per_session.eq(1).all():
        raise RuntimeError("a session maps to multiple scenes")
    sessions = sorted(identity["session_id"].astype(str).unique())
    scenes = sorted(identity["scene"].astype(str).unique())
    if len(sessions) != expected_sessions:
        raise RuntimeError(
            f"session universe changed: {len(sessions)} != {expected_sessions}")
    if len(scenes) != expected_scenes:
        raise RuntimeError(
            f"scene universe changed: {len(scenes)} != {expected_scenes}")
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_algorithm": "complete_sorted_train_session_universe_v1",
        "selection_uses_labels": False,
        "scope": (
            "train-only exhaustive actionability characterization; not a "
            "scene-disjoint confirmation and not a closed-loop SR result"),
        "source_inventory_sha256": evidence_sha256,
        "source_teacher_sha256": teacher_sha256,
        "selected_session_count": len(sessions),
        "selected_scene_count": len(scenes),
        "session_universe_sha256": session_universe_sha256(sessions),
        "sessions": sessions,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-evidence", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--expected-sessions", type=int, default=480)
    parser.add_argument("--expected-scenes", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.teacher_sha256) != 64:
        raise ValueError("teacher SHA256 must contain 64 hexadecimal characters")
    evidence = pd.read_csv(
        args.geometry_evidence,
        usecols=["session_id", "scene", "kind", "split_role"],
    )
    manifest = build_manifest(
        evidence,
        evidence_sha256=sha256_file(args.geometry_evidence),
        teacher_sha256=args.teacher_sha256,
        expected_sessions=args.expected_sessions,
        expected_scenes=args.expected_scenes,
    )
    atomic_json(args.out, manifest)
    print(json.dumps({
        "out": str(args.out.resolve()),
        "sessions": manifest["selected_session_count"],
        "scenes": manifest["selected_scene_count"],
        "session_universe_sha256": manifest["session_universe_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
