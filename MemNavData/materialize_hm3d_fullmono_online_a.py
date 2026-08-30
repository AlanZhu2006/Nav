"""Materialize every eligible full-mono HM3D Goal-A trace with explicit assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from deterministic_eval_protocol import validate_leg1_trace
    from materialize_online_a_traces import (
        SCHEMA_VERSION,
        SingleAnchorTraceCandidate,
        materialize_one,
        native_control_audit,
        sha256_file,
    )
except ImportError:
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    from MemNavData.materialize_online_a_traces import (
        SCHEMA_VERSION,
        SingleAnchorTraceCandidate,
        materialize_one,
        native_control_audit,
        sha256_file,
    )


MINIMUM_FRAME = 39
END_MARGIN = 16


def _candidate(path: Path, payload: dict[str, Any]) -> SingleAnchorTraceCandidate:
    poses = payload["poses"]
    stop = len(poses) - END_MARGIN
    if MINIMUM_FRAME >= stop:
        raise ValueError("insufficient_history_for_runtime_anchor_contract")
    points = np.asarray([[pose["x"], pose["z"]] for pose in poses], dtype=float)
    endpoint = points[-1]
    ranked = sorted(
        (-float(np.linalg.norm(points[frame] - endpoint)), frame)
        for frame in range(MINIMUM_FRAME, stop)
    )
    negative_distance, frame = ranked[0]
    return SingleAnchorTraceCandidate(
        path=path,
        payload=payload,
        score_m=-negative_distance,
        anchor=frame,
        distance_to_end_m=-negative_distance,
    )


def materialize_scene(
    *,
    trace_root: Path,
    scene: str,
    asset: Path,
    episode_root: Path,
    source_episode_order: list[str],
    out: Path,
    purpose: str | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    if not asset.is_file():
        raise FileNotFoundError(asset)
    paths = sorted(trace_root.rglob("*_leg1_trace.json"))
    if len(paths) != len(source_episode_order):
        raise RuntimeError(
            f"{scene}: expected {len(source_episode_order)} traces, found {len(paths)}"
        )
    by_episode: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        validate_leg1_trace(payload)
        if str(payload["source_scene"]) != scene:
            raise RuntimeError(f"{scene}: trace scene identity changed")
        episode = str(payload["episode"])
        if episode in by_episode:
            raise RuntimeError(f"{scene}: duplicate trace {episode}")
        by_episode[episode] = (path, payload)
    if set(by_episode) != set(source_episode_order):
        raise RuntimeError(f"{scene}: trace/source episode population changed")

    candidates: list[SingleAnchorTraceCandidate] = []
    attrition: list[dict[str, Any]] = []
    for episode in source_episode_order:
        path, payload = by_episode[episode]
        identity = {
            "scene": scene,
            "episode": episode,
            "trace_sha256": sha256_file(path),
        }
        if not payload["reached"]:
            attrition.append({**identity, "stage": "online_a_eligibility",
                              "reason": "mono_a_failed"})
            continue
        control = native_control_audit(payload)
        if not control["ok"]:
            raise RuntimeError(f"{scene}/{episode}: mono Goal-A was not native")
        try:
            candidates.append(_candidate(path, payload))
        except ValueError as error:
            attrition.append({
                **identity,
                "stage": "online_a_eligibility",
                "reason": str(error),
                "pose_count": len(payload["poses"]),
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    receipts = []
    try:
        for candidate in candidates:
            destination = temporary / scene / candidate.episode
            destination.mkdir(parents=True)
            # Any render, hash, parquet, or asset failure is infrastructure,
            # not scientific attrition.  Fail the scene rather than silently
            # shrinking the population.
            receipts.append(materialize_one(
                candidate,
                asset_root=asset.parent,
                episode_root=episode_root,
                destination=destination,
                asset_map={scene: asset},
            ))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": purpose or (
                "actual-online monocular HM3D Goal-A histories for a "
                "role-unknown full-monocular mixed-role evaluation"
            ),
            "trace_root": str(trace_root.resolve()),
            "selection": {
                "source_episode_order": source_episode_order,
                "source_trace_count": len(paths),
                "eligible_count": len(candidates),
                "all_eligible_traces_materialized": len(receipts) == len(candidates),
                "minimum_runtime_eligible_frame": MINIMUM_FRAME,
                "anchor_end_margin": END_MARGIN,
                "goals_frozen": False,
                "query_outcomes_read": False,
                "explicit_parent_asset_path": str(asset.resolve()),
                "explicit_parent_asset_sha256": sha256_file(asset),
            },
            "attrition": attrition,
            "episodes": receipts,
        }
        path = temporary / "manifest.json"
        path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "manifest.json.sha256").write_text(
            sha256_file(path) + "  manifest.json\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "source_traces": len(paths),
        "goal_a_successes": sum(
            int(payload["reached"]) for _path, payload in by_episode.values()
        ),
        "eligible": len(candidates),
        "materialized": len(receipts),
        "attrition": attrition,
        "manifest_sha256": sha256_file(out / "manifest.json"),
    }


__all__ = ["END_MARGIN", "MINIMUM_FRAME", "materialize_scene"]
