#!/usr/bin/env python3
"""Compare full and readout-only LingBot depth on one causal RGB prefix.

The script never calls a goal-conditioned or planning endpoint.  It appends
the same immutable online RGB frames in order and seals compact, frame-bound
depth receipts at predeclared checkpoints.  A second invocation can require
bit-exact depth PNG equivalence to a previously written receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests

from MemNavData.monocular_depth_runtime import (
    decode_monocular_depth_payload,
)


SCHEMA_VERSION = "depth_observation_only_equivalence_v1_20260903"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    require(isinstance(result, dict), f"{label} returned non-object JSON")
    require("error" not in result, f"{label} failed: {result.get('error')}")
    return result


def compact_depth_receipt(
        payload: dict[str, Any], expected_image_sha256: str) -> dict[str, Any]:
    depth, metadata = decode_monocular_depth_payload(
        payload, expected_image_sha256=expected_image_sha256)
    return {
        "frame_index": int(metadata["frame_index"]),
        "image_sha256": str(metadata["image_sha256"]),
        "depth_shape": [int(value) for value in metadata["depth_shape"]],
        "scale_state": str(metadata["scale_state"]),
        "scale_active": bool(metadata["scale_active"]),
        "scale_valid": bool(metadata["scale_valid"]),
        "scale_receipt_sha256": metadata["scale_receipt_sha256"],
        "depth_png_sha256": str(metadata["depth_png_sha256"]),
        "decoded_depth_sha256": hashlib.sha256(
            np.asarray(depth, dtype=np.float32).tobytes()).hexdigest(),
        "depth_nonzero_fraction": float(
            metadata["depth_nonzero_fraction"]),
        "depth_nonzero_median_m": (
            None if metadata["depth_nonzero_median_m"] is None
            else float(metadata["depth_nonzero_median_m"])),
        "metric_depth_sensor_consumed": bool(
            metadata["metric_depth_sensor_consumed"]),
        "relative_depth_model": str(metadata["relative_depth_model"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--episode-len", type=int, required=True)
    parser.add_argument("--checkpoints", default="40,127,255")
    parser.add_argument(
        "--expected-mode", choices=("full", "depth-only"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(not arguments.out.exists(), "output already exists")
    receipt_path = arguments.history_dir / "receipt.json"
    trace_path = arguments.history_dir / "online_a_trace.json"
    require(receipt_path.is_file() and trace_path.is_file(),
            "history receipt or trace is missing")
    require(sha256(receipt_path) == arguments.expected_receipt_sha256,
            "history receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rows = trace["poses"]
    hashes = receipt["rgb_frame_hashes"]
    require(len(rows) == len(hashes), "history trace/hash count mismatch")

    checkpoints = sorted({
        int(value) for value in arguments.checkpoints.split(",")
        if value.strip()
    })
    require(checkpoints and checkpoints[0] >= 40,
            "checkpoints must start at or after active frame 40")
    require(checkpoints[-1] < len(rows), "checkpoint exceeds history")
    require(arguments.episode_len > checkpoints[-1],
            "episode length must exceed every replayed frame")

    base = arguments.server_url.rstrip("/")
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": int(arguments.episode_len),
        }, timeout=180), "reset")
    expected_depth_only = arguments.expected_mode == "depth-only"
    require(bool(reset.get("depth_observation_only")) == expected_depth_only,
            "server readout-only mode differs from the declared invocation")

    depth_receipts: list[dict[str, Any]] = []
    checkpoint_set = set(checkpoints)
    for frame_index in range(checkpoints[-1] + 1):
        row = rows[frame_index]
        require(int(row["step"]) == frame_index,
                "history steps are not contiguous")
        frame = arguments.history_dir / "rgb" / f"{frame_index:06d}.jpg"
        digest = sha256(frame)
        require(digest == hashes[frame_index] == row["jpg_sha256"],
                f"history RGB changed at frame {frame_index}")
        materialize = frame_index in checkpoint_set
        append = checked_json(session.post(
            f"{base}/memory_step",
            data={"materialize_monocular_depth": "1" if materialize else "0"},
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=240), f"frame {frame_index}")
        require(int(append.get("frame_idx", -1)) == frame_index,
                "server frame index diverged")
        require(append.get("image_sha256") == digest,
                "server image digest diverged")
        if materialize:
            token = append.get("monocular_depth_transaction_token")
            require(isinstance(token, str), "depth transaction token missing")
            payload = checked_json(session.post(
                f"{base}/monocular_depth_query",
                data={
                    "expected_image_sha256": digest,
                    "expected_frame_index": str(frame_index),
                    "monocular_depth_transaction_token": token,
                }, timeout=240), f"depth {frame_index}")
            depth_receipts.append(compact_depth_receipt(payload, digest))

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": arguments.expected_mode,
        "history_receipt_sha256": arguments.expected_receipt_sha256,
        "history_trace_sha256": sha256(trace_path),
        "episode_len": int(arguments.episode_len),
        "flow_threshold": float(reset["flow_threshold"]),
        "checkpoints": checkpoints,
        "depth_receipts": depth_receipts,
        "planning_endpoint_called": False,
        "metric_depth_sensor_consumed": False,
        "runtime_evaluator_pose_visible": False,
    }
    if arguments.reference is not None:
        reference = json.loads(
            arguments.reference.read_text(encoding="utf-8"))
        require(reference.get("schema_version") == SCHEMA_VERSION,
                "reference schema changed")
        comparable = (
            "history_receipt_sha256",
            "history_trace_sha256",
            "episode_len",
            "flow_threshold",
            "checkpoints",
            "depth_receipts",
        )
        mismatches = [
            key for key in comparable if result[key] != reference.get(key)
        ]
        result["reference_sha256"] = sha256(arguments.reference)
        result["equivalent_to_reference"] = not mismatches
        result["reference_mismatches"] = mismatches
        require(not mismatches,
                "depth-observation-only output differs from full mode: "
                + ", ".join(mismatches))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
