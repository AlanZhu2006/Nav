#!/usr/bin/env python3
"""Freeze the factual-B visual-support population for lifelong NNR.

This is a result-blind constructibility audit.  It reads only the sealed NNR
benchmark, its strict-v4 source assets, and the byte-identical factual online
A/B traces.  It never reads a C, B2, or C2 navigation result.

For every source episode, the exact Goal-B RGB-D camera is reconstructed from
the frozen source metadata.  Its visible 3-D points are projected into every
frame that native NavDP actually observed during factual leg B.  Episodes with
maximum projected co-visibility >= 0.20 form the pre-registered supported
population; >= 0.50 is reported as the strong-support subset.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import build_shared_online_double_revisit as online_goal
import build_shared_online_novel_revisit as nnr_builder
from deterministic_eval_protocol import file_sha256
from generate_twoleg import cam_to_world_hab, covis_curve, make_sim, render


SCHEMA_VERSION = "lifelong_factual_b_support_v1_20260821"
SUPPORTED_COVIS = 0.20
STRONG_COVIS = 0.50


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supported-covis", type=float, default=SUPPORTED_COVIS)
    parser.add_argument("--strong-covis", type=float, default=STRONG_COVIS)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def goal_b_observation(simulator, benchmark: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source = Path(benchmark["source_episode"])
    metadata_path = source / "meta/gen_meta.json"
    goal_path = source / "goal_1.jpg"
    require(metadata_path.is_file() and goal_path.is_file(), "source Goal-B assets missing")
    require(
        file_sha256(metadata_path) == benchmark["source_metadata_sha256"],
        "source metadata changed",
    )
    require(
        nnr_builder.bytes_sha256(goal_path.read_bytes()) == benchmark["goal_b_sha256"],
        "source Goal-B JPEG changed",
    )
    metadata = load_json(metadata_path)
    goal = metadata["goals"][0]
    require(goal.get("name") == "B" and goal.get("kind") == "novel", "source Goal-B role changed")
    camera_height = float(benchmark["camera_height_m"])
    floor = nnr_builder.data_to_hab(goal["pos"])
    camera = floor + np.asarray([0.0, camera_height, 0.0], dtype=np.float64)
    yaw = float(goal["yaw_habitat"])
    rgb, depth = render(simulator, camera, yaw)
    rendered_sha = nnr_builder.bytes_sha256(online_goal.jpeg_bytes(rgb))
    require(rendered_sha == benchmark["goal_b_sha256"], "re-rendered Goal-B JPEG changed")
    return camera, rgb, depth, yaw


def audit_episode(simulator, benchmark_root: Path, manifest_row: dict) -> dict:
    scene = str(manifest_row["scene"])
    episode = str(manifest_row["episode"])
    benchmark_path = benchmark_root / scene / episode / "benchmark.json"
    require(benchmark_path.is_file(), "sealed benchmark episode missing")
    require(
        file_sha256(benchmark_path) == manifest_row["benchmark_sha256"],
        "benchmark episode changed",
    )
    benchmark = load_json(benchmark_path)
    require(benchmark["scene"] == scene and benchmark["episode"] == episode, "benchmark identity changed")
    require(benchmark.get("construction_uses_c_navigation_outcomes") is False, "source construction read C outcomes")

    scene_asset = Path(benchmark["source_scene_asset"])
    require(scene_asset.is_file(), "source scene asset missing")
    require(
        file_sha256(scene_asset) == benchmark["source_scene_asset_sha256"],
        "source scene asset changed",
    )
    trace_root = Path(benchmark["trace_root"])
    trace_b_path = trace_root / benchmark["online_b_trace"]
    require(trace_b_path.is_file(), "factual online-B trace missing")
    require(
        file_sha256(trace_b_path) == benchmark["online_b_trace_sha256"],
        "factual online-B trace changed",
    )
    goal_b = Path(benchmark["source_episode"]) / "goal_1.jpg"
    trace_b = nnr_builder.load_trace(
        trace_b_path,
        episode=episode,
        scene=scene,
        goal=goal_b.read_bytes(),
    )
    require(trace_b["reached"] is True, "factual online-B prefix did not reach B")
    camera_height = float(benchmark["camera_height_m"])
    b_history = nnr_builder.trace_history(
        simulator, trace_b, camera_height=camera_height
    )
    goal_camera, _goal_rgb, goal_depth, goal_yaw = goal_b_observation(
        simulator, benchmark
    )
    goal_points = online_goal.goal_world_points(
        goal_depth, goal_camera, goal_yaw
    )
    curve = covis_curve(
        goal_points, b_history["transforms"], b_history["depths"]
    )
    require(len(curve) == int(benchmark["online_b_steps"]), "online-B curve length changed")
    require(len(curve) > 0 and np.isfinite(curve).all(), "invalid online-B co-visibility curve")
    argmax = int(np.argmax(curve))
    maximum = float(curve[argmax])
    a_steps = int(benchmark["online_a_steps"])
    pose = trace_b["poses"][argmax]
    b_goal_floor = goal_camera - np.asarray([0.0, camera_height, 0.0])
    anchor_floor = np.asarray([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    return {
        "scene": scene,
        "episode": episode,
        "source_population_index": int(manifest_row["source_population_index"]),
        "benchmark_sha256": manifest_row["benchmark_sha256"],
        "online_b_trace_sha256": benchmark["online_b_trace_sha256"],
        "goal_b_sha256": benchmark["goal_b_sha256"],
        "online_a_steps": a_steps,
        "online_b_steps": int(benchmark["online_b_steps"]),
        "factual_b_max_covis": maximum,
        "factual_b_argmax_local_frame": argmax,
        "factual_b_argmax_global_memory_frame": a_steps + argmax,
        "factual_b_endpoint_covis": float(curve[-1]),
        "factual_b_frames_ge_0p20": int(np.sum(curve >= SUPPORTED_COVIS)),
        "factual_b_frames_ge_0p50": int(np.sum(curve >= STRONG_COVIS)),
        "argmax_position_error_to_goal_m": float(np.linalg.norm(anchor_floor - b_goal_floor)),
        "supported": bool(maximum >= SUPPORTED_COVIS),
        "strong_support": bool(maximum >= STRONG_COVIS),
        "selection_reads_query_navigation_outcomes": False,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parsed = parse_args()
    require(abs(parsed.supported_covis - SUPPORTED_COVIS) <= 1e-12, "supported threshold is frozen at 0.20")
    require(abs(parsed.strong_covis - STRONG_COVIS) <= 1e-12, "strong threshold is frozen at 0.50")
    benchmark_root = parsed.benchmark_root.resolve()
    output = parsed.out.resolve()
    require((benchmark_root / "SEALED").is_file(), "NNR benchmark is not sealed")
    require((benchmark_root / "manifest.json.sha256").is_file(), "NNR manifest receipt missing")
    require(not output.exists(), "support-audit output already exists")
    manifest_path = benchmark_root / "manifest.json"
    manifest = load_json(manifest_path)
    source_rows = list(manifest["accepted"])
    require(bool(source_rows), "sealed NNR population is empty")

    records: list[dict] = []
    current_scene = None
    simulator = None
    try:
        for row in source_rows:
            scene = str(row["scene"])
            if scene != current_scene:
                if simulator is not None:
                    simulator.close()
                scene_asset = Path(row["source_scene_asset"])
                require(scene_asset.is_file(), "manifest scene asset missing")
                simulator = make_sim(str(scene_asset), "", agent_radius=0.30)
                current_scene = scene
            assert simulator is not None
            record = audit_episode(simulator, benchmark_root, row)
            records.append(record)
            print(
                f"[{record['scene']}/{record['episode']}] "
                f"max-covis={record['factual_b_max_covis']:.4f} "
                f"support={int(record['supported'])} strong={int(record['strong_support'])}"
            )
    finally:
        if simulator is not None:
            simulator.close()

    supported = [row for row in records if row["supported"]]
    rejected = [row for row in records if not row["supported"]]
    scene_counts = Counter(row["scene"] for row in supported)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_benchmark_root": str(benchmark_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_population": len(records),
        "source_scenes": len({row["scene"] for row in records}),
        "supported_threshold_covis_inclusive": SUPPORTED_COVIS,
        "strong_threshold_covis_inclusive": STRONG_COVIS,
        "selection_inputs": [
            "sealed benchmark and source Goal-B camera",
            "byte-identical factual online-A/B traces",
            "GT depth used only for pre-outcome constructibility audit",
        ],
        "selection_reads_query_navigation_outcomes": False,
        "claim_scope": "internal lifelong memory accumulation mechanism; consumed NNR scenes",
        "supported_population": len(supported),
        "supported_scenes": len(scene_counts),
        "strong_population": sum(row["strong_support"] for row in records),
        "strong_scenes": len({row["scene"] for row in records if row["strong_support"]}),
        "supported_scene_counts": dict(sorted(scene_counts.items())),
        "accepted": supported,
        "excluded_for_insufficient_factual_b_support": rejected,
        "all_records": records,
    }
    output.mkdir(parents=True)
    atomic_json(output / "population.json", payload)
    receipt = file_sha256(output / "population.json")
    (output / "population.json.sha256").write_text(
        f"{receipt}  population.json\n", encoding="utf-8"
    )
    (output / "SEALED").write_text(
        "factual-B support population frozen before multi-leg query evaluation\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "source_population": len(records),
        "supported_population": len(supported),
        "supported_scenes": len(scene_counts),
        "strong_population": payload["strong_population"],
        "population_sha256": receipt,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
