#!/usr/bin/env python3
"""Measure DINO route-address information without invoking geometric proof.

The query construction poses are analysis-only.  Runtime receives the frozen
causal RGB stream and query RGB bytes, and returns only its normal temporally
diverse historical shortlist.  This diagnostic separates place-address
information from local-feature/PnP visibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import requests


SCHEMA_VERSION = "local_route_address_audit_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_path(mirror_root: Path, remote_path: str) -> Path:
    path = Path(remote_path)
    require(path.is_absolute() and path.parts[:2] == ("/", "scratch"),
            "source path is outside the sealed scratch mirror")
    return mirror_root.joinpath(*path.parts[1:])


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    require(isinstance(payload, dict), f"{label} returned non-object JSON")
    return payload


def planar_error(first: dict[str, Any], second: list[float]) -> float:
    return math.hypot(
        float(first["x"]) - float(second[0]),
        float(first["z"]) - float(second[2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32),
            "address audit is sealed to indices 0,16,32")
    require(not arguments.out.exists(), "address-audit output exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    history = mirror_path(arguments.mirror_root.resolve(),
                          item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(sha256(receipt_path) == item["online_a_receipt_sha256"]
            and sha256(trace_path) == item["online_a_trace_sha256"],
            "mirrored causal history changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    hashes = receipt["rgb_frame_hashes"]
    require(len(poses) == len(hashes) == int(item["online_a_steps"]),
            "history length changed")
    history_rgb = []
    for pose, expected in zip(poses, hashes):
        path = history / "rgb" / f"{int(pose['step']):06d}.jpg"
        require(sha256(path) == expected == pose["jpg_sha256"],
                f"history RGB changed at {pose['step']}")
        history_rgb.append(path)

    query_set = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    require(int(query_set.get("history_index", -1)) == arguments.history_index
            and query_set.get("runtime_evaluator_pose_visible") is False,
            "query-set identity/information contract changed")
    queries = query_set.get("queries")
    require(isinstance(queries, list) and len(queries) == 12,
            "address audit expects 12 queries")
    query_root = arguments.query_set.parent
    for query in queries:
        path = query_root / query["rgb"]
        require(sha256(path) == query["rgb_sha256"],
                f"query RGB changed: {query['rgb']}")

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(history_rgb) + len(queries) + 1,
        },
        timeout=120,
    ), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")
    for frame_index, frame in enumerate(history_rgb):
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=120,
        ), f"memory frame {frame_index}")
        require(int(response.get("frame_idx", -1)) == frame_index,
                "history replay index diverged")

    rows = []
    causal_ceiling = len(history_rgb) - 1
    for query in reversed(queries):
        path = query_root / query["rgb"]
        payload = path.read_bytes()
        probe = checked_json(session.post(
            f"{base}/retrieval_probe_step",
            files={
                "image": ("image.jpg", payload, "image/jpeg"),
                "goal": ("goal.jpg", payload, "image/jpeg"),
            },
            data={"candidate_ceiling_override": str(causal_ceiling)},
            timeout=180,
        ), f"query {query['query_index']}")
        candidates = probe.get("certified_visual_candidates")
        require(isinstance(candidates, list) and candidates,
                "DINO historical shortlist is empty")
        expected_center = float(query["construction_expected_center_frame"])
        enriched = []
        for rank, candidate in enumerate(candidates):
            anchor = int(candidate["anchor"])
            require(0 <= anchor < len(poses),
                    "DINO address escaped causal history")
            enriched.append({
                "rank": rank,
                "anchor": anchor,
                "score": float(candidate["score"]),
                "temporal_error_frames_analysis_only": abs(
                    anchor - expected_center),
                "position_error_m_analysis_only": planar_error(
                    poses[anchor], query["construction_floor_position"]),
            })
        top = enriched[0]
        best = min(enriched,
                   key=lambda row: row["position_error_m_analysis_only"])
        rows.append({
            "query_index": int(query["query_index"]),
            "runtime_input": "query_rgb_only",
            "expected_center_frame_analysis_only": expected_center,
            "top1": top,
            "best_position_in_top_k": best,
            "shortlist": enriched,
        })

    top_errors = [row["top1"]["position_error_m_analysis_only"]
                  for row in rows]
    best_errors = [
        row["best_position_in_top_k"]["position_error_m_analysis_only"]
        for row in rows]
    anchors = [row["top1"]["anchor"] for row in rows]
    large_order_violations = sum(
        current > previous + 16
        for previous, current in zip(anchors, anchors[1:])
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "DINO route-address diagnostic only; construction poses are "
            "analysis-only and no geometry proof, control, actions, or SR "
            "is executed"
        ),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(arguments.query_set),
        "base_yaw_offset_deg": query_set.get("base_yaw_offset_deg", 0.0),
        "runtime_evaluator_pose_visible": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "query_count": len(rows),
        "top1_within_1m_count": sum(error <= 1.0 for error in top_errors),
        "top_k_contains_within_1m_count": sum(
            error <= 1.0 for error in best_errors),
        "median_top1_position_error_m": median(top_errors),
        "median_best_top_k_position_error_m": median(best_errors),
        "large_top1_order_violation_count": large_order_violations,
        "queries": rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    path = arguments.out / "address_audit.json"
    path.write_bytes(encoded)
    (arguments.out / "address_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  address_audit.json\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "top1_within_1m": result["top1_within_1m_count"],
        "top_k_contains_within_1m": result[
            "top_k_contains_within_1m_count"],
        "median_top1_position_error_m": result[
            "median_top1_position_error_m"],
        "out": str(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
