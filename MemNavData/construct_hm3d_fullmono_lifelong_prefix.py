#!/usr/bin/env python3
"""Turn one frozen actual-mono A/B rollout into a lifelong query benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import build_shared_online_double_revisit as history_tools
import build_shared_online_role_pairs as pair_tools
from deterministic_eval_protocol import validate_leg1_trace
from generate_twoleg import covis_curve, make_sim, render
from hm3d_fullmono_lifelong import PREFIX_SCHEMA, load_protocol, require, sha256_file


COMPLETION_SCHEMA = "hm3d_fullmono_lifelong_prefix_fragment_v1_20260824"
DEPTH_TOLERANCE_M = 0.30


def find_role(item: dict, role: str) -> dict:
    rows = [
        query for pair in item["pairs"] for query in pair["queries"]
        if query["analysis_role"] == role
    ]
    require(len(rows) == 1, f"expected exactly one {role} query")
    return rows[0]


def rendered_b_history(simulator, trace: dict, camera_height: float) -> dict:
    depths = []
    transforms = []
    for pose in trace["poses"]:
        floor = np.asarray(
            [pose[axis] for axis in ("x", "y", "z")], dtype=np.float64
        )
        camera = floor + np.asarray([0.0, camera_height, 0.0])
        rgb, depth = render(simulator, camera, float(pose["yaw"]))
        require(
            hashlib.sha256(history_tools.jpeg_bytes(rgb)).hexdigest()
            == pose["jpg_sha256"],
            f"factual B RGB mismatch at step {pose['step']}",
        )
        depths.append(depth)
        transforms.append(history_tools.cam_to_world_hab(camera, float(pose["yaw"])))
    return {"depths": depths, "transforms": transforms}


def construct(
    *,
    protocol_path: Path,
    ab_root: Path,
    b_root: Path,
    history_index: int,
    out: Path,
) -> dict:
    protocol = load_protocol(protocol_path)
    manifest_path = ab_root / "role_pairs/manifest.json"
    population_path = ab_root / "population_receipt.json"
    require((ab_root / "SEALED").is_file(), "A/B population is not sealed")
    require(sha256_file(manifest_path)
            == json.loads(population_path.read_text())["benchmark_manifest_sha256"],
            "A/B population manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(0 <= history_index < len(manifest["episodes"]),
            "history index outside A/B population")
    item = manifest["episodes"][history_index]
    scene, episode = str(item["scene"]), str(item["episode"])
    label = f"{history_index:03d}_{scene}_{episode}"
    factual_root = b_root / label
    completion_path = factual_root / "completion.json"
    require(completion_path.is_file(), f"{label}: factual B completion missing")
    factual = json.loads(completion_path.read_text())
    require(factual.get("status") == "complete", f"{label}: B incomplete")
    require(factual["protocol_sha256"] == sha256_file(protocol_path),
            f"{label}: B used another protocol")
    require(factual["benchmark_manifest_sha256"] == sha256_file(manifest_path),
            f"{label}: B used another A/B population")
    require(factual.get("controller") == "frozen_navdp_native_sidecar",
            f"{label}: factual B controller changed")
    require(factual.get("navdp_depth_source") == "monocular_sidecar",
            f"{label}: factual B was not monocular")
    require(int(factual.get("metric_depth_sensor_reads", -1)) == 0,
            f"{label}: factual B consumed metric depth")
    depth_audit = factual.get("depth_audit")
    require(isinstance(depth_audit, dict)
            and int(depth_audit.get("metric_sensor_plan_count", -1)) == 0
            and int(depth_audit.get("monocular_receipt_plan_count", 0)) > 0
            and int(depth_audit.get("monocular_scale_hash_count", 0)) == 1,
            f"{label}: factual B monocular audit is incomplete")
    require(not out.exists(), f"prefix fragment exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    try:
        base_completion = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "complete",
            "history_index": history_index,
            "scene": scene,
            "episode": episode,
            "protocol_sha256": sha256_file(protocol_path),
            "AB_manifest_sha256": sha256_file(manifest_path),
            "factual_B_completion_sha256": sha256_file(completion_path),
            "query_navigation_outcomes_read": False,
            "eligible": False,
        }
        if not factual["reached_B"]:
            base_completion["attrition_reason"] = "actual_mono_B_failed"
            completion = base_completion
        else:
            trace_path = Path(factual["B_trace_path"])
            require(sha256_file(trace_path) == factual["B_trace_sha256"],
                    f"{label}: factual B trace changed")
            trace = json.loads(trace_path.read_text())
            validate_leg1_trace(trace, expected_episode=episode,
                                expected_source_scene=scene)
            source_a = Path(item["online_a_episode"])
            receipt_a = json.loads((source_a / "receipt.json").read_text())
            scene_file = Path(receipt_a["source_asset"])
            require(sha256_file(scene_file) == receipt_a["source_asset_sha256"],
                    f"{label}: scene asset changed")
            episode_ab = ab_root / "role_pairs" / scene / episode
            query_b = find_role(item, "novel")
            query_c = find_role(item, "revisit")
            goal_b_rgb = episode_ab / query_b["goal_rgb"]
            goal_b_depth = episode_ab / query_b["goal_depth"]
            goal_c_rgb = episode_ab / query_c["goal_rgb"]
            goal_c_depth = episode_ab / query_c["goal_depth"]
            for path, digest in (
                (goal_b_rgb, query_b["goal_rgb_sha256"]),
                (goal_b_depth, query_b["goal_depth_sha256"]),
                (goal_c_rgb, query_c["goal_rgb_sha256"]),
                (goal_c_depth, query_c["goal_depth_sha256"]),
            ):
                require(path.is_file() and sha256_file(path) == digest,
                        f"{label}: query asset changed: {path.name}")
            simulator = make_sim(str(scene_file), "", agent_radius=0.30)
            try:
                b_history = rendered_b_history(
                    simulator, trace, float(receipt_a["camera_height_m"])
                )
                b_floor = np.asarray(query_b["floor_position"], dtype=np.float64)
                b_camera = b_floor + np.asarray(
                    [0.0, float(receipt_a["camera_height_m"]), 0.0]
                )
                b_depth = history_tools.read_depth_png(goal_b_depth)
                b_points = history_tools.goal_world_points(
                    b_depth, b_camera, float(query_b["yaw_rad"])
                )
                b_curve = covis_curve(
                    b_points,
                    b_history["transforms"],
                    b_history["depths"],
                    tol=DEPTH_TOLERANCE_M,
                )
                b_support = float(b_curve.max()) if len(b_curve) else 0.0
                b_support_frame = int(np.argmax(b_curve)) if len(b_curve) else None
                actual_b_end = np.asarray(trace["end_position"], dtype=np.float64)
                c_floor = np.asarray(query_c["floor_position"], dtype=np.float64)
                b_end_to_c = pair_tools.query_geometry(
                    simulator.pathfinder, actual_b_end, c_floor
                )
            finally:
                simulator.close()
            reasons = []
            if b_support < float(protocol["factual_b_collection"][
                "B_goal_support_by_factual_B_minimum_inclusive"
            ]):
                reasons.append("B_goal_not_supported_by_factual_B")
            if b_end_to_c is None or not (
                float(protocol["factual_b_collection"][
                    "actual_B_end_to_C_geodesic_band_m"
                ][0]) <= float(b_end_to_c[0]) <=
                float(protocol["factual_b_collection"][
                    "actual_B_end_to_C_geodesic_band_m"
                ][1])
            ):
                reasons.append("actual_B_end_to_C_geodesic_outside_band")
            if reasons:
                completion = {
                    **base_completion,
                    "attrition_reason": ",".join(reasons),
                    "B_goal_max_factual_B_covis": b_support,
                    "actual_B_end_to_C_geodesic_m": (
                        None if b_end_to_c is None else float(b_end_to_c[0])
                    ),
                }
            else:
                benchmark_root = temporary / "benchmark"
                goals = benchmark_root / "goals"
                goals.mkdir(parents=True)
                assets = {}
                for name, rgb, depth in (
                    ("B", goal_b_rgb, goal_b_depth),
                    ("C", goal_c_rgb, goal_c_depth),
                ):
                    rgb_out, depth_out = goals / f"{name}.jpg", goals / f"{name}_depth.png"
                    shutil.copy2(rgb, rgb_out)
                    shutil.copy2(depth, depth_out)
                    assets[name] = {
                        "rgb": str(rgb_out.relative_to(benchmark_root)),
                        "rgb_sha256": sha256_file(rgb_out),
                        "depth": str(depth_out.relative_to(benchmark_root)),
                        "depth_sha256": sha256_file(depth_out),
                    }
                trace_out = benchmark_root / f"{episode}_legB_trace.json"
                shutil.copy2(trace_path, trace_out)
                factual_completion_out = (
                    benchmark_root / "factual_B_completion.json"
                )
                shutil.copy2(completion_path, factual_completion_out)
                benchmark = {
                    "schema_version": PREFIX_SCHEMA,
                    "scene": scene,
                    "episode": episode,
                    "history_index": history_index,
                    "protocol_sha256": sha256_file(protocol_path),
                    "AB_manifest_sha256": sha256_file(manifest_path),
                    "source_online_A_episode": str(source_a.resolve()),
                    "source_online_A_episode_id": str(
                        item.get("lifelong_construction", {}).get(
                            "recipient_episode", episode
                        )
                    ),
                    "source_online_A_receipt_sha256": sha256_file(
                        source_a / "receipt.json"
                    ),
                    "source_online_A_trace_sha256": sha256_file(
                        source_a / "online_a_trace.json"
                    ),
                    "source_scene_asset": str(scene_file.resolve()),
                    "source_scene_asset_sha256": sha256_file(scene_file),
                    "online_A_steps": int(item["online_a_steps"]),
                    "online_B_steps": int(trace["steps"]),
                    "online_B_trace": trace_out.name,
                    "online_B_trace_sha256": sha256_file(trace_out),
                    "factual_B_completion": factual_completion_out.name,
                    "factual_B_completion_sha256": sha256_file(
                        factual_completion_out
                    ),
                    "episode_seed": int(trace["episode_seed"]),
                    "runtime_role_visibility": "none",
                    "query_outcomes_read": False,
                    "goals": {
                        "B": {
                            **assets["B"],
                            "floor_position": query_b["floor_position"],
                            "yaw_rad": float(query_b["yaw_rad"]),
                            "max_online_A_covis": float(
                                query_b["max_online_a_covis"]
                            ),
                            "max_factual_B_covis": b_support,
                            "max_factual_B_covis_frame": b_support_frame,
                            "factual_B_covis_curve": [
                                float(value) for value in b_curve
                            ],
                        },
                        "C": {
                            **assets["C"],
                            "floor_position": query_c["floor_position"],
                            "yaw_rad": float(query_c["yaw_rad"]),
                            "max_online_A_covis": float(
                                query_c["max_online_a_covis"]
                            ),
                            "max_online_A_covis_frame": int(
                                query_c["max_online_a_covis_frame"]
                            ),
                        },
                    },
                    "actual_B_end_to_C_geodesic_m": float(b_end_to_c[0]),
                    "B_goal_strong_support": b_support >= float(
                        protocol["factual_b_collection"][
                            "B_goal_support_strong_threshold_inclusive"
                        ]
                    ),
                }
                benchmark_path = benchmark_root / "benchmark.json"
                benchmark_path.write_text(json.dumps(
                    benchmark, indent=2, sort_keys=True, allow_nan=False
                ) + "\n")
                completion = {
                    **base_completion,
                    "eligible": True,
                    "B_goal_max_factual_B_covis": b_support,
                    "B_goal_max_factual_B_covis_frame": b_support_frame,
                    "B_goal_strong_support": benchmark["B_goal_strong_support"],
                    "actual_B_end_to_C_geodesic_m": float(b_end_to_c[0]),
                    "benchmark_sha256": sha256_file(benchmark_path),
                }
        completion_path_out = temporary / "completion.json"
        completion_path_out.write_text(json.dumps(
            completion, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "completion.json.sha256").write_text(
            sha256_file(completion_path_out) + "  completion.json\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ab-root", type=Path, required=True)
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = construct(
        protocol_path=args.protocol,
        ab_root=args.ab_root,
        b_root=args.b_root,
        history_index=args.history_index,
        out=args.out,
    )
    print(json.dumps({
        "history_index": args.history_index,
        "scene": result["scene"],
        "episode": result["episode"],
        "eligible": result["eligible"],
        "attrition_reason": result.get("attrition_reason"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
