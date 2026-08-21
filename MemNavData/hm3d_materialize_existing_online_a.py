#!/usr/bin/env python3
"""Materialize every eligible saved HM3D native-A trace, fail closed.

This wrapper is intentionally executed with the exact parent HM3D runtime at
the front of PYTHONPATH.  In particular, rendering and JPEG encoding come from
the source that wrote the saved trace hashes, while query construction happens
later with the frozen Final14 builder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from deterministic_eval_protocol import validate_leg1_trace
from materialize_online_a_traces import (
    SCHEMA_VERSION,
    TraceCandidate,
    best_separated_pair,
    materialize_one,
    native_control_audit,
    sha256_file,
)


MINIMUM_FRAME = 39
END_MARGIN = 16


def materialize_all(
    trace_root: Path,
    asset_root: Path,
    episode_root: Path,
    out: Path,
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    paths = sorted(trace_root.glob("*_leg1_trace.json"))
    if not paths:
        raise RuntimeError("saved HM3D scene has no Goal-A traces")
    candidates: list[TraceCandidate] = []
    attrition = []
    for path in paths:
        payload = json.loads(path.read_text())
        validate_leg1_trace(payload)
        identity = {
            "scene": str(payload["source_scene"]),
            "episode": str(payload["episode"]),
            "trace_sha256": sha256_file(path),
        }
        if not payload["reached"]:
            attrition.append({**identity, "stage": "online_a_eligibility",
                              "reason": "native_a_failed"})
            continue
        control = native_control_audit(payload)
        if not control["ok"]:
            raise RuntimeError(f"Goal-A was not native-only: {identity}")
        poses = payload["poses"]
        if len(poses) <= MINIMUM_FRAME + END_MARGIN:
            attrition.append({
                **identity,
                "stage": "online_a_eligibility",
                "reason": "insufficient_history_for_runtime_anchor_contract",
                "pose_count": len(poses),
            })
            continue
        # The pair is only a materialization diagnostic.  The frozen Final14
        # builder independently searches and validates the actual Revisit.
        pair = best_separated_pair(poses, margin=END_MARGIN, min_gap=1)
        if pair is None:
            attrition.append({**identity, "stage": "online_a_eligibility",
                              "reason": "no_interior_diagnostic_pair"})
            continue
        candidates.append(TraceCandidate(path, payload, *pair))

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    receipts = []
    try:
        for candidate in candidates:
            destination = temporary / candidate.scene / candidate.episode
            destination.mkdir(parents=True)
            try:
                receipt = materialize_one(
                    candidate,
                    asset_root=asset_root,
                    episode_root=episode_root,
                    destination=destination,
                )
            except Exception as error:
                shutil.rmtree(destination, ignore_errors=True)
                attrition.append({
                    "scene": candidate.scene,
                    "episode": candidate.episode,
                    "trace_sha256": sha256_file(candidate.path),
                    "stage": "online_a_materialization",
                    "reason": f"{type(error).__name__}: {error}",
                })
            else:
                receipts.append(receipt)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "all eligible saved native NavDP HM3D Goal-A histories for "
                "a role-unknown mixed Novel/Revisit safety extension"
            ),
            "trace_root": str(trace_root.resolve()),
            "selection": {
                "requested_count": None,
                "eligible_count": len(candidates),
                "all_eligible_traces_attempted": True,
                "minimum_runtime_eligible_frame": MINIMUM_FRAME,
                "anchor_end_margin": END_MARGIN,
                "goals_frozen": False,
                "query_outcomes_read": False,
            },
            "source_trace_count": len(paths),
            "attrition": attrition,
            "episodes": receipts,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (temporary / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "source_traces": len(paths),
        "eligible": len(candidates),
        "materialized": len(receipts),
        "attrition": len(attrition),
        "manifest_sha256": sha256_file(out / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_all(
        args.trace_root, args.asset_root, args.episode_root, args.out
    ), sort_keys=True))


if __name__ == "__main__":
    main()
