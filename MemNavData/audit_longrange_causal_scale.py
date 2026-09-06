#!/usr/bin/env python3
"""Compare causal first-40 and first-64 LingBot scale on sealed histories.

Only the first 64 online-A RGB observations are replayed.  The server returns
model-derived receipts; Habitat pose, route distance, query images, controller
actions, goal role, and success are never sent to the runtime.  This is a
mechanism audit, not a navigation evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests


SCHEMA_VERSION = "longrange_causal_scale_audit_v1_20260902"
RECEIPT_SCHEMA_VERSION = "lingbot_causal_metric_scale_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
PREFIX_COUNT = 64


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-indices", type=int, nargs="+", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(not args.out.exists(), "output exists")
    require(sha256(args.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    require(len(set(args.history_indices)) == len(args.history_indices),
            "history indices must be unique")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    session = requests.Session()
    base = f"http://127.0.0.1:{args.port}"
    rows = []

    for history_index in args.history_indices:
        require(0 <= history_index < len(episodes), "history index out of range")
        item = episodes[history_index]
        history = mirror_path(args.mirror_root.resolve(), item["online_a_episode"])
        receipt_path = history / "receipt.json"
        trace_path = history / "online_a_trace.json"
        require(sha256(receipt_path) == item["online_a_receipt_sha256"],
                "online-A receipt changed")
        require(sha256(trace_path) == item["online_a_trace_sha256"],
                "online-A trace changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        poses = trace["poses"]
        require(len(poses) >= PREFIX_COUNT, "history has fewer than 64 frames")
        require(len(receipt["rgb_frame_hashes"]) >= PREFIX_COUNT,
                "RGB receipt has fewer than 64 hashes")

        reset = checked_json(session.post(
            f"{base}/navigator_reset",
            json={
                "camera_height": float(receipt["camera_height_m"]),
                "camera_intrinsic": receipt["camera_intrinsic"],
                "seed": int(receipt["episode_seed"]),
                "episode_len": PREFIX_COUNT + 1,
            }, timeout=120), "reset")
        require(reset.get("algo") == "memnav", "wrong runtime server")

        replay_hashes = []
        for frame_index, (pose, expected) in enumerate(zip(
                poses[:PREFIX_COUNT], receipt["rgb_frame_hashes"][:PREFIX_COUNT])):
            path = history / "rgb" / f"{int(pose['step']):06d}.jpg"
            actual = sha256(path)
            require(actual == expected == pose["jpg_sha256"],
                    f"history RGB changed at prefix frame {frame_index}")
            response = checked_json(session.post(
                f"{base}/memory_step",
                files={"image": ("image.jpg", path.read_bytes(), "image/jpeg")},
                timeout=120), f"history frame {frame_index}")
            require(int(response.get("frame_idx", -1)) == frame_index,
                    "history replay index diverged")
            replay_hashes.append(actual)

        scale = checked_json(session.post(
            f"{base}/causal_metric_scale_query", timeout=300), "scale query")
        require(scale.get("schema_version") == RECEIPT_SCHEMA_VERSION,
                "wrong scale receipt schema")
        require(int(scale.get("stream_observation_count", -1)) == PREFIX_COUNT,
                "scale query changed or misreported stream length")
        require(scale.get("runtime_evaluator_pose_visible") is False,
                "runtime evaluator pose boundary changed")
        require(scale.get("metric_depth_sensor_consumed") is False,
                "metric depth sensor entered scale audit")
        rows.append({
            "history_index": int(history_index),
            "scene": str(item["scene"]),
            "episode": str(item["episode"]),
            "online_a_receipt_sha256": item["online_a_receipt_sha256"],
            "online_a_trace_sha256": item["online_a_trace_sha256"],
            "replayed_rgb_sha256": replay_hashes,
            "scale_receipt": scale,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "model-only causal scale mechanism audit; no controller or SR"),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "prefix_count": PREFIX_COUNT,
        "history_indices": [int(value) for value in args.history_indices],
        "rows": rows,
    }
    args.out.mkdir(parents=True)
    result = args.out / "causal_scale_audit.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    (args.out / "causal_scale_audit.json.sha256").write_text(
        f"{sha256(result)}  {result.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "out": str(result.resolve()),
        "scales": [{
            "history_index": row["history_index"],
            "first40": row["scale_receipt"]["first40"].get(
                "metric_scale_m_per_raw"),
            "first64": row["scale_receipt"]["first64"].get(
                "metric_scale_m_per_raw"),
            "first64_available": row["scale_receipt"]["first64"].get(
                "available"),
        } for row in rows],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
