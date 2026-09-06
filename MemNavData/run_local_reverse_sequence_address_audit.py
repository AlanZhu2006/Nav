#!/usr/bin/env python3
"""Measure causal sequence-level addressability on a reverse visual route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import requests


SCHEMA_VERSION = "local_reverse_sequence_address_audit_v1_20260902"
QUERY_SCHEMA_VERSION = "hm3d_reverse_route_odometry_queries_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
MAX_ADVANCE_GRID = (8, 16, 24, 32, 48, 64)


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
    result = response.json()
    require(isinstance(result, dict), f"{label} returned non-object JSON")
    return result


def causal_monotone_readout(
    similarity: np.ndarray, *, maximum_advance: int
) -> list[int]:
    """Online Viterbi readout with one monotone bounded transition model."""

    require(similarity.ndim == 2 and similarity.shape[0] > 0
            and similarity.shape[1] > 1, "invalid similarity matrix")
    width = similarity.shape[1]
    advance = int(maximum_advance)
    require(1 <= advance < width, "invalid maximum advance")
    score = np.full(width, -np.inf, dtype=np.float64)
    score[0] = 0.0  # the goal switch is continuous with the history tail
    readouts: list[int] = []
    for observation in similarity:
        standard = float(np.std(observation))
        normalized = (
            observation - float(np.mean(observation))) / max(standard, 1e-6)
        predecessor = np.full(width, -np.inf, dtype=np.float64)
        for delta in range(advance + 1):
            predecessor[delta:] = np.maximum(
                predecessor[delta:], score[:width - delta])
        score = predecessor + normalized
        readouts.append(int(np.argmax(score)))
    return readouts


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
            "sequence audit is sealed to indices 0,16,32")
    require(not arguments.out.exists(), "sequence-audit output exists")
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
            "causal history changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    history_rgb = []
    for row, expected in zip(poses, receipt["rgb_frame_hashes"]):
        path = history / "rgb" / f"{int(row['step']):06d}.jpg"
        require(sha256(path) == expected == row["jpg_sha256"],
                f"history RGB changed at {row['step']}")
        history_rgb.append(path)

    query_set = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    require(query_set.get("schema_version") == QUERY_SCHEMA_VERSION
            and int(query_set.get("history_index", -1))
            == arguments.history_index,
            "wrong reverse-route query set")
    require(query_set.get("runtime_evaluator_pose_visible") is False,
            "query set violates the information contract")
    queries = query_set["queries"]
    query_root = arguments.query_set.parent
    for row in queries:
        path = query_root / row["rgb"]
        require(sha256(path) == row["rgb_sha256"],
                f"query RGB changed: {row['rgb']}")

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(history_rgb) + len(queries) + 1,
        }, timeout=120), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")
    for index, frame in enumerate(history_rgb):
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=120), f"history frame {index}")
        require(int(response.get("frame_idx", -1)) == index,
                "history replay index diverged")
    for offset, row in enumerate(queries):
        path = query_root / row["rgb"]
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", path.read_bytes(), "image/jpeg")},
            timeout=120), f"query frame {offset}")
        require(int(response.get("frame_idx", -1))
                == len(history_rgb) + offset,
                "query replay index diverged")
    similarity_receipt = checked_json(session.post(
        f"{base}/causal_visual_similarity_query",
        data={"history_count": str(len(history_rgb))}, timeout=180),
        "DINO similarity")
    similarity = np.asarray(
        similarity_receipt["similarity"], dtype=np.float64)
    require(similarity.shape == (len(queries), len(history_rgb)),
            "DINO similarity matrix has the wrong shape")

    target = int(query_set["target_history_frame_analysis_only"])
    route_indices = np.arange(
        len(history_rgb) - 1, target - 1, -1, dtype=np.int64)
    route_queries = [row for row in queries if row["kind"] == "reverse_route"]
    route_query_indices = [int(row["query_index"]) for row in route_queries]
    route_similarity = similarity[route_query_indices][:, route_indices]

    def position_error(anchor: int, query: dict) -> float:
        pose = poses[int(anchor)]
        floor = query["construction_floor_position"]
        return math.hypot(float(pose["x"]) - float(floor[0]),
                          float(pose["z"]) - float(floor[2]))

    static_anchors = [
        int(route_indices[int(np.argmax(observation))])
        for observation in route_similarity
    ]
    static_errors = [
        position_error(anchor, query)
        for anchor, query in zip(static_anchors, route_queries)
    ]
    methods = {}
    for maximum_advance in MAX_ADVANCE_GRID:
        state = causal_monotone_readout(
            route_similarity, maximum_advance=maximum_advance)
        anchors = [int(route_indices[value]) for value in state]
        errors = [
            position_error(anchor, query)
            for anchor, query in zip(anchors, route_queries)
        ]
        methods[str(maximum_advance)] = {
            "maximum_advance_history_frames": maximum_advance,
            "within_1m_count": sum(error <= 1.0 for error in errors),
            "within_2m_count": sum(error <= 2.0 for error in errors),
            "median_position_error_m": median(errors),
            "final_position_error_m": errors[-1],
            "selected_anchors": anchors,
            "position_errors_m_analysis_only": errors,
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "DINO causal-sequence information audit only; construction "
            "poses score addresses after inference and no control/SR runs"),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(arguments.query_set),
        "runtime_evaluator_pose_visible": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "route_query_count": len(route_queries),
        "static_top1": {
            "within_1m_count": sum(error <= 1.0 for error in static_errors),
            "within_2m_count": sum(error <= 2.0 for error in static_errors),
            "median_position_error_m": median(static_errors),
            "selected_anchors": static_anchors,
            "position_errors_m_analysis_only": static_errors,
        },
        "causal_monotone_viterbi": methods,
    }
    arguments.out.mkdir(parents=True)
    matrix_path = arguments.out / "dino_similarity.npz"
    np.savez_compressed(
        matrix_path, similarity=similarity.astype(np.float16),
        route_indices=route_indices,
        route_query_indices=np.asarray(route_query_indices, dtype=np.int64))
    result["dino_similarity_npz_sha256"] = sha256(matrix_path)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output = arguments.out / "sequence_address_audit.json"
    output.write_bytes(encoded)
    (arguments.out / "sequence_address_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  sequence_address_audit.json\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "static_within_1m": result["static_top1"]["within_1m_count"],
        "static_median_m": result["static_top1"][
            "median_position_error_m"],
        "viterbi": {
            key: {
                "within_1m": value["within_1m_count"],
                "median_m": value["median_position_error_m"],
            }
            for key, value in methods.items()
        },
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
