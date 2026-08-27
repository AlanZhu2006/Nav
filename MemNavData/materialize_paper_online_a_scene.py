#!/usr/bin/env python3
"""Materialize every eligible native Goal-A trace in one paper array task.

Unlike the pilot selector, this wrapper never asks for a desired count and
never drops an eligible trace to improve downstream constructibility.  Native
failures, short histories, and renderer/hash failures are retained as explicit
attrition rows.
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
    discover_single_anchor_candidates,
    materialize_one,
    sha256_file,
)

PAPER_MINIMUM_ELIGIBLE_FRAME = 39
PAPER_ANCHOR_END_MARGIN = 16


def materialize(
    trace_root: Path,
    asset_root: Path,
    episode_root: Path,
    out: Path,
    asset_map: dict[str, Path] | None = None,
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    trace_paths = sorted(trace_root.glob("*_leg1_trace.json"))
    if not trace_paths:
        raise RuntimeError("no native Goal-A traces were written")
    trace_payloads = {}
    for path in trace_paths:
        payload = json.loads(path.read_text())
        validate_leg1_trace(payload)
        identity = (str(payload["source_scene"]), str(payload["episode"]))
        if identity in trace_payloads:
            raise RuntimeError(f"duplicate trace identity {identity}")
        trace_payloads[identity] = (path, payload)

    candidates = discover_single_anchor_candidates(
        trace_root,
        minimum_frame=PAPER_MINIMUM_ELIGIBLE_FRAME,
        end_margin=PAPER_ANCHOR_END_MARGIN,
    )
    candidate_index = {
        (candidate.scene, candidate.episode): candidate
        for candidate in candidates
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    receipts = []
    attrition = []
    try:
        for identity, (trace_path, payload) in sorted(trace_payloads.items()):
            if identity not in candidate_index:
                attrition.append({
                    "scene": identity[0],
                    "episode": identity[1],
                    "trace_sha256": sha256_file(trace_path),
                    "stage": "online_a_eligibility",
                    "reason": (
                        "native_a_failed" if not payload["reached"]
                        else "insufficient_native_history_for_anchor_contract"
                    ),
                })
                continue
            candidate = candidate_index[identity]
            destination = temporary / candidate.scene / candidate.episode
            try:
                receipt = materialize_one(
                    candidate,
                    asset_root=asset_root,
                    episode_root=episode_root,
                    destination=destination,
                    asset_map=asset_map,
                )
            except Exception as error:
                shutil.rmtree(destination, ignore_errors=True)
                attrition.append({
                    "scene": identity[0],
                    "episode": identity[1],
                    "trace_sha256": sha256_file(trace_path),
                    "stage": "online_a_materialization",
                    "reason": f"{type(error).__name__}: {error}",
                })
            else:
                receipts.append(receipt)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "exhaustive audited online NavDP Goal-A histories from one "
                "frozen paper source scene"
            ),
            "trace_root": str(trace_root.resolve()),
            "selection": {
                "requested_count": None,
                "eligible_count": len(candidates),
                "distinct_scene_first": False,
                "anchor_margin": PAPER_MINIMUM_ELIGIBLE_FRAME,
                "anchor_end_margin": PAPER_ANCHOR_END_MARGIN,
                "anchor_requirement": "one_runtime_eligible_frame",
                "minimum_anchor_gap_frames": None,
                "minimum_preselection_score_m": None,
                "goals_frozen": False,
                "all_eligible_traces_attempted": True,
            },
            "source_trace_count": len(trace_paths),
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
        "source_traces": len(trace_paths),
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
    parser.add_argument(
        "--asset-map-json",
        type=Path,
        help=("optional JSON object mapping stable scene labels to explicit "
              "Habitat stage/asset paths for cross-dataset evaluation"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asset_map = None
    if args.asset_map_json is not None:
        raw_map = json.loads(args.asset_map_json.read_text())
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("asset map must be a non-empty JSON object")
        asset_map = {
            str(scene): Path(path).resolve() for scene, path in raw_map.items()
        }
        for scene, path in asset_map.items():
            if not scene or not path.is_file():
                raise FileNotFoundError(f"invalid asset mapping {scene}: {path}")
    print(json.dumps(materialize(
        args.trace_root, args.asset_root, args.episode_root, args.out, asset_map
    ), sort_keys=True))


if __name__ == "__main__":
    main()
