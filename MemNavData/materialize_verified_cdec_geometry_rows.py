#!/usr/bin/env python3
"""Reuse the independently verified CDEC geometry rows without another GPU run.

The CDEC dual-proposal collector already evaluated the frozen geometry proposal
on the complete train40 session universe.  This tool extracts that arm only
after checking the raw collector/report hashes against the independent CDEC
verification receipt.  It does not select sessions or fit a threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


EXPECTED_SESSION_COUNT = 480
EXPECTED_ORIGIN = "lightglue_fundamental_rank_v1"
EXPECTED_SELECTION_MODE = "cdec_geometry_dual_ranked"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def select_geometry_rows(
    rows: Iterable[Mapping[str, str]],
    expected_sessions: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    selected: list[dict[str, str]] = []
    origin_counts: dict[str, int] = {}
    for raw in rows:
        row = dict(raw)
        origin = str(row.get("candidate_selection_origin", ""))
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if origin == EXPECTED_ORIGIN:
            selected.append(row)
    session_ids = [str(row.get("session_id", "")) for row in selected]
    if len(session_ids) != len(set(session_ids)):
        raise RuntimeError("geometry extraction contains duplicate sessions")
    observed = set(session_ids)
    if observed != expected_sessions:
        raise RuntimeError(
            "geometry extraction differs from frozen session universe: "
            f"missing={sorted(expected_sessions - observed)} "
            f"extra={sorted(observed - expected_sessions)}")
    return selected, dict(sorted(origin_counts.items()))


def verify_provenance(
    *,
    dual_rows: Path,
    collector_report: Path,
    official_report: Path,
    independent_verification: Path,
    manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    collector = load_object(collector_report)
    official = load_object(official_report)
    independent = load_object(independent_verification)
    frozen_manifest = load_object(manifest)

    if independent.get("verified") is not True:
        raise RuntimeError("source CDEC independent verification did not pass")
    scope = independent.get("scope", {})
    if (scope.get("train40_only") is not True
            or scope.get("development_or_blind_read") is not False
            or scope.get("closed_loop") is not False):
        raise RuntimeError("source CDEC verification scope changed")
    inputs = independent.get("inputs", {})
    expected_hashes = {
        dual_rows: inputs.get("dual_rows_sha256"),
        collector_report: inputs.get("collector_report_sha256"),
        official_report: inputs.get("official_report_sha256"),
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError(f"verified source hash changed: {path}")

    config = collector.get("config", {})
    required_config = {
        "selection_mode": EXPECTED_SELECTION_MODE,
        "neighbor_offsets": [0],
        "full_replay": True,
        "pnp_lightglue": True,
        "top_k": 8,
        "candidate_min_gap": 4,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"source collector config changed: {key}={config.get(key)!r}")

    if (official.get("status") !=
            "train_only_oof_proposal_certificate_audit_not_closed_loop"):
        raise RuntimeError("source CDEC official report status changed")
    geometry = official.get("policies", {}).get("geometry", {})
    reconstructed = independent.get(
        "reconstructed", {}).get("policies", {}).get("geometry", {})
    if geometry != reconstructed:
        raise RuntimeError("official and independent geometry summaries differ")
    if int(geometry.get("sessions", -1)) != EXPECTED_SESSION_COUNT:
        raise RuntimeError("source geometry policy is not the full train40 set")

    sessions = frozen_manifest.get("sessions")
    if (frozen_manifest.get("schema_version") !=
            "train40_certificate_challenge_manifest_v1"
            or not isinstance(sessions, list)
            or len(sessions) != EXPECTED_SESSION_COUNT
            or len(set(map(str, sessions))) != EXPECTED_SESSION_COUNT):
        raise RuntimeError("frozen train40 challenge manifest changed")
    return collector, independent, frozen_manifest


def atomic_csv(path: Path, fieldnames: list[str],
               rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="raise",
                lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    _collector, independent, manifest = verify_provenance(
        dual_rows=args.dual_rows,
        collector_report=args.collector_report,
        official_report=args.official_report,
        independent_verification=args.independent_verification,
        manifest=args.session_manifest,
    )
    with args.dual_rows.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("dual collector CSV has no header")
        fieldnames = list(reader.fieldnames)
        selected, origin_counts = select_geometry_rows(
            reader, set(map(str, manifest["sessions"])))
    if len(selected) != EXPECTED_SESSION_COUNT:
        raise RuntimeError("geometry extraction count changed")
    atomic_csv(args.out_rows, fieldnames, selected)
    receipt = {
        "schema_version": "verified_cdec_geometry_reuse_v1_20260814",
        "status": "materialized_verified_existing_measurements_not_new_gpu_run",
        "scope": {
            "train40_only": True,
            "development_or_blind_read": False,
            "closed_loop": False,
            "threshold_fitting": False,
            "session_selection": "complete_frozen_manifest_universe",
        },
        "source_independent_verification_schema": independent.get(
            "schema_version"),
        "selection_origin": EXPECTED_ORIGIN,
        "source_origin_counts": origin_counts,
        "selected_sessions": len(selected),
        "selected_scenes": len({row["scene"] for row in selected}),
        "inputs": {
            "dual_rows_sha256": sha256_file(args.dual_rows),
            "collector_report_sha256": sha256_file(args.collector_report),
            "official_report_sha256": sha256_file(args.official_report),
            "independent_verification_sha256": sha256_file(
                args.independent_verification),
            "session_manifest_sha256": sha256_file(args.session_manifest),
        },
        "output": {
            "geometry_rows": str(args.out_rows.resolve()),
            "geometry_rows_sha256": sha256_file(args.out_rows),
        },
    }
    atomic_json(args.out_receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-rows", type=Path, required=True)
    parser.add_argument("--collector-report", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--independent-verification", type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--out-rows", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
            args.dual_rows, args.collector_report, args.official_report,
            args.independent_verification, args.session_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
