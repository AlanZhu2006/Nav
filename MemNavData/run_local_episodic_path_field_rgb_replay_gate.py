#!/usr/bin/env python3
"""Deployment-only RGB replay gate for the episodic path field.

This local gate does not instantiate Habitat or NavDP and never computes a
navigation outcome.  It replays one sealed causal survey into the real MemNav /
LingBot server, opens the hidden-role Revisit goal only for offline smoke
selection, and then feeds observations from the demonstrated return in reverse
temporal order.  The purpose is narrow: establish that live LingBot poses,
first-40 scale, CEC localization, and monotone path-field readout compose before
spending a formal HPC allocation.

The reversed observations are a deployment-geometry replay, not a navigation
policy and not efficacy evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import requests

from MemNavData.run_hm3d_episodic_path_field_gate import (
    audit_path_field_plans,
)


SCHEMA_VERSION = "local_episodic_path_field_rgb_replay_gate_v1_20260902"
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
            f"only sealed /scratch paths may be mirrored: {remote_path}")
    return mirror_root.joinpath(*path.parts[1:])


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    require(isinstance(payload, dict), f"{label} returned non-object JSON")
    return payload


def plan_receipt(response: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "role_label_visible": False,
        "certified_relocalization_ok": response.get("ok"),
        "certified_relocalization_accepted": response.get("accepted"),
        "certified_relocalization_guidance_mode": response.get(
            "guidance_mode"),
    }
    receipt.update({
        key: value for key, value in response.items()
        if key.startswith("episodic_path_field_") or key.startswith("path_")
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-readouts", type=int, default=80)
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32),
            "local gate is sealed to indices 0,16,32")
    require(arguments.stride >= 1 and arguments.max_readouts >= 1,
            "stride and max-readouts must be positive")
    require(not arguments.out.exists(), "output already exists")
    arguments.mirror_root = arguments.mirror_root.resolve()
    arguments.manifest = arguments.manifest.resolve()
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen source manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    require(len(manifest.get("episodes", [])) == 48,
            "source population changed")
    item = manifest["episodes"][arguments.history_index]

    history = mirror_path(arguments.mirror_root, item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(
        sha256(receipt_path) == item["online_a_receipt_sha256"]
        and sha256(trace_path) == item["online_a_trace_sha256"],
        "mirrored causal history differs from the frozen source",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    require(
        receipt.get("history_source")
        == "controlled_causal_rgb_geodesic_survey"
        and trace.get("source_hybrid_route") == "causal_survey"
        and trace.get("reached") is True,
        "history is not the sealed successful causal survey",
    )
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

    population = arguments.manifest.parent
    episode_root = (
        population / str(item["scene"]) / str(item["episode"])
    )
    role_pair_path = episode_root / "role_pairs.json"
    require(sha256(role_pair_path) == item["role_pairs_sha256"],
            "role-pair sidecar changed")
    role_pair = json.loads(role_pair_path.read_text(encoding="utf-8"))
    revisit = [
        query for query in role_pair["pairs"][0]["queries"]
        if query["analysis_role"] == "revisit"
    ]
    require(len(revisit) == 1, "offline smoke selection is not one Revisit")
    query = revisit[0]
    goal_path = episode_root / query["goal_rgb"]
    require(sha256(goal_path) == query["goal_rgb_sha256"],
            "Revisit goal RGB changed")
    goal_bytes = goal_path.read_bytes()

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(rgb_paths) + arguments.max_readouts + 1,
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

    raw_responses: list[dict[str, Any]] = []
    plan_receipts: list[dict[str, Any]] = []
    replayed_history_indices: list[int] = []
    candidate_fingerprint: str | None = None
    selected_anchor: int | None = None

    # First decision repeats the causal tail, matching the evaluator's first
    # query observation after the frozen history replay.
    candidate_indices = [len(rgb_paths) - 1]
    candidate_indices.extend(range(
        len(rgb_paths) - 1 - arguments.stride,
        -1,
        -arguments.stride,
    ))
    for source_index in candidate_indices[:arguments.max_readouts]:
        current_bytes = rgb_paths[source_index].read_bytes()
        probe = checked_json(session.post(
            f"{base}/retrieval_probe_step",
            files={
                "image": ("image.jpg", current_bytes, "image/jpeg"),
                "goal": ("goal.jpg", goal_bytes, "image/jpeg"),
            },
            timeout=180,
        ), f"probe at source frame {source_index}")
        candidates = probe.get("certified_visual_candidates")
        require(isinstance(candidates, list), "probe omitted CEC candidates")
        serialized = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        if candidate_fingerprint is None:
            candidate_fingerprint = fingerprint
        else:
            require(fingerprint == candidate_fingerprint,
                    "goal-session candidate contract changed during replay")
        response = checked_json(session.post(
            f"{base}/certified_relocalize",
            files={"goal": ("goal.jpg", goal_bytes, "image/jpeg")},
            data={
                "candidates": json.dumps(candidates),
                "graph_rescue": "0",
                "learned_rescue": "0",
                "proposal_order": "geometry_first",
                "authority_policy": "strict_certificate",
                "guidance_mode": "episodic_path_field",
            },
            timeout=300,
        ), f"certificate at source frame {source_index}")
        require(response.get("ok") is True,
                f"certificate runtime failed: {response.get('reason')}")
        require(response.get("accepted") is True,
                f"Revisit certificate rejected: {response.get('reason')}")
        if selected_anchor is None:
            selected_anchor = int(response["selected_anchor"])
        else:
            require(int(response["selected_anchor"]) == selected_anchor,
                    "certified anchor changed during replay")
        raw_responses.append(response)
        plan_receipts.append(plan_receipt(response))
        replayed_history_indices.append(source_index)
        if response.get("episodic_path_field_status") == "complete":
            break

    audit = audit_path_field_plans(plan_receipts)
    require(audit["final_progress_m"] + 1e-6 >= audit["initial_progress_m"],
            "live path field made no valid monotone receipt")
    arguments.out.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "local_rgb_replay_gate_passed",
        "claim_boundary": (
            "deployment geometry replay only; reversed demonstrated RGB is "
            "not a navigation policy or an SR result"
        ),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "bin_name": item["bin_name"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_history_receipt_sha256": item["online_a_receipt_sha256"],
        "source_history_trace_sha256": item["online_a_trace_sha256"],
        "source_role_pair_sha256": item["role_pairs_sha256"],
        "source_goal_rgb_sha256": query["goal_rgb_sha256"],
        "analysis_only_query_selection": "revisit",
        "runtime_role_forwarded": False,
        "candidate_fingerprint_sha256": candidate_fingerprint,
        "selected_anchor": selected_anchor,
        "stride": arguments.stride,
        "maximum_readouts": arguments.max_readouts,
        "replayed_history_indices": replayed_history_indices,
        "path_field_audit": audit,
        "navigation_controller_executed": False,
        "navigation_success_read": False,
        "navigation_sr_computed": False,
        "readouts": plan_receipts,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    result_path = arguments.out / "gate_result.json"
    result_path.write_bytes(encoded)
    (arguments.out / "gate_result.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  gate_result.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "history_index": arguments.history_index,
        "selected_anchor": selected_anchor,
        "audit": audit,
        "output": str(arguments.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
