#!/usr/bin/env python3
"""End-to-end RGB mechanism gate for proof-once route-coordinate tracking.

The runtime sees one strict CEC target proof, a sealed causal RGB history, and
then only current RGB observations.  Construction poses are read after each
response for audit metrics and are never included in an HTTP request.  No
NavDP controller, Habitat action, success label, runtime role, fallback, or
distance switch is part of this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import requests

from MemNavData.run_local_episodic_reanchor_gate import (
    FROZEN_MANIFEST_SHA256,
    QUERY_SCHEMA_VERSION,
    checked_json,
    mirror_path,
    planar_error_m,
    require,
    sha256,
)


SCHEMA_VERSION = "local_route_coordinate_gate_v1_20260902"
FROZEN_QUERY_COUNT = 24
MAX_POSITION_ERROR_M = 1.0
MAX_MEDIAN_POSITION_ERROR_M = 0.75
MAX_FINAL_REMAINING_M = 5.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index == 16,
            "route-coordinate gate is frozen to medium history index 16")
    require(not arguments.out.exists(), "output already exists")
    mirror = arguments.mirror_root.resolve()
    manifest_path = arguments.manifest.resolve()
    query_set_path = arguments.query_set.resolve()
    require(sha256(manifest_path) == FROZEN_MANIFEST_SHA256,
            "frozen source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    query_set = json.loads(query_set_path.read_text(encoding="utf-8"))
    require(query_set.get("schema_version") == QUERY_SCHEMA_VERSION,
            "wrong corrected query-set schema")
    require(int(query_set.get("history_index", -1)) == arguments.history_index,
            "query set and selected history differ")
    require(query_set.get("runtime_evaluator_pose_visible") is False,
            "query set violates runtime information contract")
    queries = query_set.get("queries")
    require(isinstance(queries, list) and len(queries) == FROZEN_QUERY_COUNT,
            "gate is preregistered for exactly 24 queries")

    history = mirror_path(mirror, item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(
        sha256(receipt_path) == item["online_a_receipt_sha256"]
        and sha256(trace_path) == item["online_a_trace_sha256"],
        "mirrored causal history differs from frozen source",
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
    for query in queries:
        image = query_root / query["rgb"]
        require(sha256(image) == query["rgb_sha256"],
                f"query RGB changed: {query['rgb']}")

    population = manifest_path.parent
    episode_root = population / str(item["scene"]) / str(item["episode"])
    role_pair_path = episode_root / "role_pairs.json"
    require(sha256(role_pair_path) == item["role_pairs_sha256"],
            "role-pair sidecar changed")
    role_pair = json.loads(role_pair_path.read_text(encoding="utf-8"))
    revisit = [
        query for query in role_pair["pairs"][0]["queries"]
        if query["analysis_role"] == "revisit"
    ]
    require(len(revisit) == 1, "offline construction selection is not unique")
    target = revisit[0]
    target_path = episode_root / target["goal_rgb"]
    require(sha256(target_path) == target["goal_rgb_sha256"],
            "target goal RGB changed")
    target_bytes = target_path.read_bytes()

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(rgb_paths) + len(queries) + 2,
        },
        timeout=120,
    ), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")

    for frame_index, frame in enumerate(rgb_paths):
        payload = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=180,
        ), f"memory frame {frame_index}")
        require(int(payload.get("frame_idx", -1)) == frame_index,
                "server history index diverged during replay")

    causal_ceiling = len(rgb_paths) - 1
    tail_bytes = rgb_paths[-1].read_bytes()
    probe = checked_json(session.post(
        f"{base}/retrieval_probe_step",
        files={
            "image": ("image.jpg", tail_bytes, "image/jpeg"),
            "goal": ("goal.jpg", target_bytes, "image/jpeg"),
        },
        data={"candidate_ceiling_override": str(causal_ceiling)},
        timeout=180,
    ), "initial target probe")
    candidates = probe.get("certified_visual_candidates")
    require(isinstance(candidates, list), "target probe omitted candidates")
    target_proof = checked_json(session.post(
        f"{base}/certified_relocalize",
        files={"goal": ("goal.jpg", target_bytes, "image/jpeg")},
        data={
            "candidates": json.dumps(candidates),
            "proposal_order": "geometry_first",
            "authority_policy": "strict_certificate",
            "guidance_mode": "episodic_path_field",
            "graph_rescue": "0",
            "learned_rescue": "0",
        },
        timeout=300,
    ), "initial target certificate")
    require(target_proof.get("ok") is True,
            f"target proof runtime failed: {target_proof.get('reason')}")
    require(target_proof.get("accepted") is True,
            f"target certificate rejected: {target_proof.get('reason')}")
    initial_total_m = float(target_proof["path_progress_m"]) + float(
        target_proof["path_remaining_m"])

    rows: list[dict[str, Any]] = []
    stopped_reason = None
    for query in reversed(queries):
        query_path = query_root / query["rgb"]
        query_bytes = query_path.read_bytes()
        # This append is still conditioned on the original target goal.  It
        # keeps the one goal session and one target authority immutable.
        checked_json(session.post(
            f"{base}/retrieval_probe_step",
            files={
                "image": ("image.jpg", query_bytes, "image/jpeg"),
                "goal": ("goal.jpg", target_bytes, "image/jpeg"),
            },
            data={"candidate_ceiling_override": str(causal_ceiling)},
            timeout=180,
        ), f"append route query {query['query_index']}")
        response = checked_json(session.post(
            f"{base}/episodic_path_field_step",
            files={"goal": ("goal.jpg", target_bytes, "image/jpeg")},
            timeout=300,
        ), f"route-coordinate query {query['query_index']}")
        selected_anchor = response.get("local_witness", {}).get(
            "selected_anchor")
        position_error = None
        if isinstance(selected_anchor, int) and 0 <= selected_anchor < len(poses):
            position_error = planar_error_m(
                poses[selected_anchor],
                query["construction_floor_position"],
            )
        row = {
            "query_index": int(query["query_index"]),
            "runtime_input": "current_rgb_and_authorized_target_goal_only",
            "state_updated": response.get("state_updated") is True,
            "status": response.get("status"),
            "reason": response.get("reason"),
            "selected_anchor": selected_anchor,
            "position_error_m_analysis_only": position_error,
            "path_progress_m": response.get("path_progress_m"),
            "path_remaining_m": response.get("path_remaining_m"),
            "path_cross_track_m": response.get("path_cross_track_m"),
            "direction_vector": response.get("direction_vector"),
            "endpoint_fallback_available": response.get(
                "endpoint_fallback_available"),
            "native_fallback_available": response.get(
                "native_fallback_available"),
            "distance_gate_present": response.get("distance_gate_present"),
            "local_observation_authority": response.get(
                "local_observation_authority"),
            "runtime_ms": response.get("runtime_ms"),
        }
        rows.append(row)
        if not row["state_updated"]:
            stopped_reason = str(row["reason"])
            break

    updated = [row for row in rows if row["state_updated"]]
    errors = [float(row["position_error_m_analysis_only"])
              for row in updated
              if row["position_error_m_analysis_only"] is not None]
    progress = [float(row["path_progress_m"]) for row in updated]
    final_remaining = (
        float(updated[-1]["path_remaining_m"]) if updated else None)
    checks = {
        "all_24_route_observations_update_state": (
            len(updated) == FROZEN_QUERY_COUNT),
        "all_updates_have_anchor_within_1m": (
            len(errors) == FROZEN_QUERY_COUNT
            and all(error <= MAX_POSITION_ERROR_M for error in errors)),
        "median_anchor_error_at_most_0_75m": (
            bool(errors) and median(errors) <= MAX_MEDIAN_POSITION_ERROR_M),
        "route_progress_is_monotone": all(
            second + 1e-9 >= first
            for first, second in zip(progress, progress[1:])),
        "final_remaining_at_most_5m": (
            final_remaining is not None
            and final_remaining <= MAX_FINAL_REMAINING_M),
        "no_fallback_or_distance_gate": all(
            row["endpoint_fallback_available"] is False
            and row["native_fallback_available"] is False
            and row["distance_gate_present"] is False
            and row["local_observation_authority"] is False
            for row in rows),
        "all_active_directions_finite": all(
            row["status"] == "complete"
            or (isinstance(row["direction_vector"], list)
                and len(row["direction_vector"]) == 2)
            for row in updated),
    }
    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "claim_boundary": (
            "RGB-only route-coordinate mechanism gate; no NavDP actions or SR"
        ),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(query_set_path),
        "target_goal_sha256": target["goal_rgb_sha256"],
        "target_certificate_accepted_once": True,
        "target_anchor": target_proof.get("selected_anchor"),
        "initial_route_length_m": initial_total_m,
        "runtime_role_forwarded": False,
        "runtime_evaluator_pose_visible": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "query_count_planned": FROZEN_QUERY_COUNT,
        "query_count_executed": len(rows),
        "state_update_count": len(updated),
        "stopped_reason": stopped_reason,
        "median_position_error_m": median(errors) if errors else None,
        "maximum_position_error_m": max(errors) if errors else None,
        "final_remaining_m": final_remaining,
        "checks": checks,
        "frozen_gate": {
            "required_state_updates": FROZEN_QUERY_COUNT,
            "maximum_each_position_error_m": MAX_POSITION_ERROR_M,
            "maximum_median_position_error_m": (
                MAX_MEDIAN_POSITION_ERROR_M),
            "maximum_final_remaining_m": MAX_FINAL_REMAINING_M,
            "fallback_or_distance_gate_permitted": False,
        },
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
        "updates": len(updated),
        "planned": FROZEN_QUERY_COUNT,
        "median_position_error_m": result["median_position_error_m"],
        "final_remaining_m": final_remaining,
        "stopped_reason": stopped_reason,
        "output": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
