#!/usr/bin/env python3
"""RGB-only local re-anchor gate for long-range episodic route following.

Habitat poses are present only in the sealed query-set construction receipt and
are used after each runtime response to score localization error.  The MemNav
server receives only the causal RGB history and one perturbed RGB query at a
time.  No controller, action, navigation success, or runtime role label is
used by this mechanism gate.
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


SCHEMA_VERSION = "local_episodic_reanchor_gate_v1_20260902"
QUERY_SCHEMA_VERSIONS = {
    "hm3d_episodic_reanchor_query_set_v2_20260902",
    "hm3d_episodic_reanchor_query_set_v3_20260902",
}
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
MIN_ACCEPTED = 9
MIN_LOCALIZED_WITHIN_1M = 9
MAX_MEDIAN_POSITION_ERROR_M = 0.75
MAX_LARGE_ORDER_VIOLATIONS = 1
ORDER_TOLERANCE_FRAMES = 16
MIN_ACCEPTED_PER_TEMPORAL_THIRD = 2


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
            f"only sealed /scratch paths may be mirrored: {remote_path}")
    return mirror_root.joinpath(*path.parts[1:])


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    require(isinstance(payload, dict), f"{label} returned non-object JSON")
    return payload


def planar_error_m(pose: dict[str, Any], position: list[float]) -> float:
    return math.hypot(
        float(pose["x"]) - float(position[0]),
        float(pose["z"]) - float(position[2]),
    )


def finite_pose9(response: dict[str, Any]) -> bool:
    pose9 = response.get("pnp", {}).get("pose9")
    return (
        isinstance(pose9, list)
        and len(pose9) == 9
        and all(isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in pose9)
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
            "local gate is sealed to indices 0,16,32")
    require(not arguments.out.exists(), "output already exists")
    mirror = arguments.mirror_root.resolve()
    manifest_path = arguments.manifest.resolve()
    query_set_path = arguments.query_set.resolve()
    require(sha256(manifest_path) == FROZEN_MANIFEST_SHA256,
            "frozen source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]

    query_set = json.loads(query_set_path.read_text(encoding="utf-8"))
    require(query_set.get("schema_version") in QUERY_SCHEMA_VERSIONS,
            "wrong query-set schema")
    require(int(query_set.get("history_index", -1)) == arguments.history_index,
            "query set and selected history differ")
    require(query_set.get("source_manifest_sha256") == FROZEN_MANIFEST_SHA256,
            "query set was built from another population")
    require(query_set.get("runtime_evaluator_pose_visible") is False,
            "query set violates the runtime information contract")
    queries = query_set.get("queries")
    require(isinstance(queries, list) and len(queries) == 12,
            "gate is preregistered for exactly 12 queries")

    history = mirror_path(mirror, item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(
        sha256(receipt_path) == item["online_a_receipt_sha256"]
        and sha256(trace_path) == item["online_a_trace_sha256"],
        "mirrored causal history differs from the frozen source",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    hashes = receipt["rgb_frame_hashes"]
    require(len(poses) == len(hashes) == int(item["online_a_steps"]),
            "history frame count changed")
    rgb_paths: list[Path] = []
    for pose, expected_hash in zip(poses, hashes):
        frame = history / "rgb" / f"{int(pose['step']):06d}.jpg"
        require(sha256(frame) == expected_hash == pose["jpg_sha256"],
                f"history RGB changed at step {pose['step']}")
        rgb_paths.append(frame)

    query_root = query_set_path.parent
    for row in queries:
        image_path = query_root / row["rgb"]
        require(sha256(image_path) == row["rgb_sha256"],
                f"query RGB changed: {row['rgb']}")

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(rgb_paths) + len(queries) + 1,
        },
        timeout=120,
    ), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")

    for frame_index, frame in enumerate(rgb_paths):
        payload = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=120,
        ), f"memory frame {frame_index}")
        require(int(payload.get("frame_idx", -1)) == frame_index,
                "server history index diverged during replay")

    # Query in descending historical order, matching a return along the taught
    # route.  The causal ceiling excludes every diagnostic query, including
    # those appended earlier in this loop.
    rows: list[dict[str, Any]] = []
    causal_ceiling = len(rgb_paths) - 1
    for query in reversed(queries):
        query_path = query_root / query["rgb"]
        query_bytes = query_path.read_bytes()
        probe = checked_json(session.post(
            f"{base}/retrieval_probe_step",
            files={
                "image": ("image.jpg", query_bytes, "image/jpeg"),
                "goal": ("goal.jpg", query_bytes, "image/jpeg"),
            },
            data={"candidate_ceiling_override": str(causal_ceiling)},
            timeout=180,
        ), f"probe query {query['query_index']}")
        candidates = probe.get("certified_visual_candidates")
        require(isinstance(candidates, list), "probe omitted CEC candidates")
        response = checked_json(session.post(
            f"{base}/certified_relocalize",
            files={"goal": ("goal.jpg", query_bytes, "image/jpeg")},
            data={
                "candidates": json.dumps(candidates),
                "graph_rescue": "0",
                "learned_rescue": "0",
                "proposal_order": "geometry_first",
                "authority_policy": "strict_certificate",
                "guidance_mode": "endpoint_bearing",
            },
            timeout=300,
        ), f"certificate query {query['query_index']}")
        require(response.get("ok") is True,
                f"certificate runtime failed: {response.get('reason')}")
        accepted = response.get("accepted") is True
        selected_anchor = response.get("selected_anchor")
        expected_center = float(query["construction_expected_center_frame"])
        position_error = None
        temporal_error = None
        if accepted:
            require(isinstance(selected_anchor, int),
                    "accepted proof omitted selected anchor")
            require(0 <= selected_anchor < len(poses),
                    "selected anchor escaped frozen history")
            position_error = planar_error_m(
                poses[selected_anchor],
                query["construction_floor_position"])
            temporal_error = abs(float(selected_anchor) - expected_center)
        rows.append({
            "query_index": int(query["query_index"]),
            "runtime_input": "perturbed_rgb_only",
            "expected_center_frame_analysis_only": expected_center,
            "accepted": accepted,
            "reason": response.get("reason"),
            "selected_anchor": selected_anchor,
            "selected_dino_rank": response.get("selected_dino_rank"),
            "candidate_count": len(candidates),
            "dino_top1_anchor": (
                candidates[0].get("anchor") if candidates else None),
            "position_error_m_analysis_only": position_error,
            "temporal_error_frames_analysis_only": temporal_error,
            "pnp_pose9_finite": finite_pose9(response) if accepted else None,
            "certificate": response.get("certificate"),
        })

    accepted_rows = [row for row in rows if row["accepted"]]
    errors = [float(row["position_error_m_analysis_only"])
              for row in accepted_rows]
    local_rows = [row for row in accepted_rows
                  if float(row["position_error_m_analysis_only"]) <= 1.0]
    large_order_violations = 0
    selected_sequence = [int(row["selected_anchor"])
                         for row in accepted_rows]
    for previous, current in zip(selected_sequence, selected_sequence[1:]):
        if current > previous + ORDER_TOLERANCE_FRAMES:
            large_order_violations += 1

    third_accepts = []
    for lower in (0, 4, 8):
        third_accepts.append(sum(
            1 for row in accepted_rows
            if lower <= int(row["query_index"]) < lower + 4
        ))
    median_error = median(errors) if errors else None
    checks = {
        "accepted_at_least_9_of_12": len(accepted_rows) >= MIN_ACCEPTED,
        "localized_within_1m_at_least_9_of_12": (
            len(local_rows) >= MIN_LOCALIZED_WITHIN_1M),
        "accepted_median_position_error_at_most_0_75m": (
            median_error is not None
            and median_error <= MAX_MEDIAN_POSITION_ERROR_M),
        "large_order_violations_at_most_1": (
            large_order_violations <= MAX_LARGE_ORDER_VIOLATIONS),
        "at_least_2_accepts_in_each_temporal_third": all(
            count >= MIN_ACCEPTED_PER_TEMPORAL_THIRD
            for count in third_accepts),
        "all_accepted_pnp_pose9_finite": all(
            row["pnp_pose9_finite"] is True for row in accepted_rows),
    }
    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "claim_boundary": (
            "RGB-only local relocalization mechanism gate; no controller, "
            "actions, SR, or deployable long-range claim"
        ),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(query_set_path),
        "runtime_evaluator_pose_visible": False,
        "runtime_role_forwarded": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "query_count": len(rows),
        "accepted_count": len(accepted_rows),
        "localized_within_1m_count": len(local_rows),
        "accepted_median_position_error_m": median_error,
        "large_order_violation_count": large_order_violations,
        "accepted_per_temporal_third": third_accepts,
        "frozen_gate": {
            "minimum_accepted": MIN_ACCEPTED,
            "minimum_localized_within_1m": MIN_LOCALIZED_WITHIN_1M,
            "maximum_median_position_error_m": (
                MAX_MEDIAN_POSITION_ERROR_M),
            "maximum_large_order_violations": (
                MAX_LARGE_ORDER_VIOLATIONS),
            "order_tolerance_frames": ORDER_TOLERANCE_FRAMES,
            "minimum_accepted_per_temporal_third": (
                MIN_ACCEPTED_PER_TEMPORAL_THIRD),
        },
        "checks": checks,
        "queries": rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    result_path = arguments.out / "gate_result.json"
    result_path.write_bytes(encoded)
    (arguments.out / "gate_result.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  gate_result.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "accepted": len(accepted_rows),
        "localized_within_1m": len(local_rows),
        "median_position_error_m": median_error,
        "large_order_violations": large_order_violations,
        "output": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
