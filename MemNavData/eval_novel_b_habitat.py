#!/usr/bin/env python3
"""Strict start->A->B evaluation for Novel-B NavDP upper bounds.

The input episodes are the frozen three-leg A/B/C episodes, but this evaluator
never executes Goal C.  It runs one of three Goal-B controllers after an
identical native ImageGoal Goal-A prefix:

* ``native_imagegoal``: native NavDP ImageGoal;
* ``oracle_short_1p25m``: a point 1.25 m ahead on Habitat's geodesic;
* ``oracle_final_point``: a point 100 m ahead, which saturates at the final
  Goal-B endpoint for every benchmark path.

``--novel-b-arm`` is parsed here before delegating the remaining CLI to the
audited two-leg evaluator module.  The runner invokes this file three times
against one live NavDP server and the paired summarizer rejects any difference
in the complete serialized Goal-A record.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ARMS = (
    "native_imagegoal",
    "oracle_short_1p25m",
    "oracle_final_point",
)
ORACLE_SUBGOAL_METRES = {
    "native_imagegoal": None,
    "oracle_short_1p25m": 1.25,
    "oracle_final_point": 100.0,
}
PROTOCOL = "novel_b_upper_bound_v1"
SCHEMA_VERSION = 1


def _extract_arm(argv: list[str]) -> tuple[str | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--novel-b-arm", choices=ARMS)
    namespace, remaining = parser.parse_known_args(argv[1:])
    return namespace.novel_b_arm, [argv[0], *remaining]


NOVEL_B_ARM, sys.argv = _extract_arm(sys.argv)

# These imports intentionally happen only after the evaluator-specific option
# has been removed.  eval_2leg_habitat owns the shared controller CLI.
import eval_2leg_habitat as base  # noqa: E402
from conditional_c_protocol import world_goal_to_local  # noqa: E402
from global_subgoal_protocol import polyline_subgoal  # noqa: E402


args = base.args

# The frozen base evaluator predates request timeouts.  Scope a bounded wrapper
# to this process so a dead/livelocked server cannot consume the whole Slurm
# allocation.  Model initialization is allowed a generous read timeout.
HTTP_TIMEOUT_S = (10.0, 600.0)
_requests_post = base.requests.post


def _post_with_timeout(*post_args, **post_kwargs):
    post_kwargs.setdefault("timeout", HTTP_TIMEOUT_S)
    return _requests_post(*post_args, **post_kwargs)


base.requests.post = _post_with_timeout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def jsonable(value: Any) -> Any:
    """Convert rollout state to strict, canonical-JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        require(math.isfinite(value), "rollout contains a non-finite float")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"rollout contains unsupported value {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(
        jsonable(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def leg_spl(leg: dict, geodesic_m: float) -> float:
    path_m = (
        leg["path_len_at_reach"]
        if leg.get("path_len_at_reach") is not None
        else leg["path_len"]
    )
    return base.spl(leg["reached"], geodesic_m, path_m)


def empty_goal_b(position: np.ndarray, yaw: float, goal_xz: np.ndarray) -> dict:
    return {
        "attempted": False,
        "reached": False,
        "path_len": 0.0,
        "path_len_at_reach": None,
        "step_at_reach": None,
        "steps": 0,
        "diagnostic_steps": 0,
        "plans": [],
        "memory_trace": [],
        "rollout_trace": [],
        "end_pos": np.asarray(position, dtype=np.float64).copy(),
        "end_psi": float(yaw),
        "final_goal_dist_m": float(
            np.linalg.norm(np.asarray(position)[[0, 2]] - goal_xz)
        ),
    }


def run_oracle_goal_b(
    sim,
    pathfinder,
    start_position: np.ndarray,
    start_yaw: float,
    goal_jpg: bytes,
    goal_xz: np.ndarray,
    episode_seed: int,
    subgoal_distance_m: float,
) -> dict:
    """Run NavDP with a privileged point on the live shortest path."""
    position = np.asarray(start_position, dtype=np.float64).copy()
    yaw = float(start_yaw)
    path_len = 0.0
    path_len_at_reach = None
    step_at_reach = None
    way_world = None
    plans: list[dict] = []
    history: list[np.ndarray] = []
    rollout_trace: list[dict] = []
    success_dist = float(args.success_dist)

    def result(steps: int) -> dict:
        return {
            "attempted": True,
            "reached": path_len_at_reach is not None,
            "path_len": float(path_len),
            "path_len_at_reach": path_len_at_reach,
            "step_at_reach": step_at_reach,
            "steps": int(steps),
            "diagnostic_steps": int(steps),
            "plans": plans,
            "memory_trace": [],
            "rollout_trace": rollout_trace,
            "end_pos": position.copy(),
            "end_psi": float(yaw),
            "final_goal_dist_m": float(
                np.linalg.norm(position[[0, 2]] - goal_xz)
            ),
        }

    for step in range(args.max_steps):
        rgb, depth = base.render(
            sim,
            position + np.asarray([0.0, base.CAM_H, 0.0]),
            yaw,
        )
        frame = base.jpg_bytes(rgb)
        rollout_trace.append({
            "step": int(step),
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "yaw": float(yaw),
            "jpg_sha256": hashlib.sha256(frame).hexdigest(),
        })

        if step % args.exec_horizon == 0:
            goal3 = np.asarray(
                [goal_xz[0], position[1], goal_xz[1]], dtype=np.float64
            )
            ok, remaining_m, path_points = base.geodesic(
                pathfinder, position, goal3
            )
            require(
                ok and np.isfinite(remaining_m) and bool(path_points),
                "oracle Goal-B geodesic is invalid",
            )
            subgoal = polyline_subgoal(path_points, subgoal_distance_m)
            local_goal = world_goal_to_local(
                subgoal[[0, 2]], position[[0, 2]], yaw
            )
            request_seed = base.diffusion_plan_seed(
                int(episode_seed), 1, len(plans)
            )
            response = base.requests.post(
                f"{base.BASE}/navdp_step_ip_mixgoal",
                files={
                    "image": ("image.jpg", frame),
                    "image_goal": ("goal.jpg", goal_jpg),
                    "depth": ("depth.png", base.depth_png_bytes(depth)),
                },
                data={
                    "goal_data": json.dumps({
                        "goal_x": [float(local_goal[0])],
                        "goal_y": [float(local_goal[1])],
                    }),
                    "diffusion_seed": str(request_seed),
                },
            )
            response.raise_for_status()
            response_json = response.json()
            require(
                int(response_json.get("diffusion_seed", -1)) == request_seed,
                "NavDP server did not echo the oracle Goal-B diffusion seed",
            )
            normalized = base.normalize_navdp_response(response_json)
            way, selector = base.select_plan_trajectory(
                normalized,
                position,
                yaw,
                pathfinder,
                goal_xz,
                trajectory_selector="server",
            )
            way_world = base.waypoints_to_world(
                way, position[[0, 2]], yaw
            )
            subgoal_is_final = bool(
                np.linalg.norm(subgoal - np.asarray(path_points[-1])) <= 1e-9
            )
            if subgoal_distance_m == ORACLE_SUBGOAL_METRES["oracle_final_point"]:
                require(
                    subgoal_is_final,
                    "100 m oracle cap did not reach the final Goal-B endpoint",
                )
            plans.append({
                "step": int(step),
                "current_x": float(position[0]),
                "current_z": float(position[2]),
                "current_yaw": float(yaw),
                "evaluation_gt_goal_distance_m": float(
                    np.linalg.norm(position[[0, 2]] - goal_xz)
                ),
                "remaining_geodesic_m": float(remaining_m),
                "oracle_subgoal_distance_cap_m": float(subgoal_distance_m),
                "oracle_subgoal_world": subgoal.tolist(),
                "oracle_subgoal_local": local_goal.tolist(),
                "oracle_subgoal_euclidean_m": float(
                    np.linalg.norm(subgoal - position)
                ),
                "oracle_subgoal_is_final_endpoint": subgoal_is_final,
                "pose_controller": "oracle_habitat_geodesic_image_point_mix",
                "diffusion_seed": normalized.get("diffusion_seed"),
                "requested_diffusion_seed": request_seed,
                "navdp_critic_max": normalized.get("navdp_critic_max"),
                **selector,
            })

        if way_world is not None:
            position, yaw, distance = base.pursuit_step(
                position, yaw, way_world, pathfinder
            )
            path_len += distance
        history.append(position[[0, 2]].copy())
        if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
            path_len_at_reach = float(path_len)
            step_at_reach = int(step + 1)
            return result(step + 1)
        if (
            len(history) > args.stuck_window
            and np.linalg.norm(history[-1] - history[-args.stuck_window])
            < args.stuck_dist
        ):
            return result(step + 1)
    return result(args.max_steps)


def validate_protocol() -> None:
    require(NOVEL_B_ARM is not None, "--novel-b-arm is required")
    require(args.server_backend == "navdp", "Novel-B benchmark requires NavDP")
    require(
        float(args.navdp_stop_threshold) == -0.5,
        "NavDP stop threshold must remain frozen at -0.5",
    )
    require(args.leg1_mode == "policy", "Goal A must use live policy rollout")
    require(args.leg1_goal_source == "own", "Goal-A swapping is forbidden")
    require(not args.stop_after_leg1, "Novel-B benchmark must execute Goal B")
    require(not args.write_leg1_trace, "shared/replayed Goal-A traces are forbidden")
    require(not args.shared_leg1_trace_root, "shared Goal-A traces are forbidden")
    require(not args.reset_memory, "Goal A->B must carry NavDP short memory")
    require(
        args.navdp_goal_switch_reset == "carry",
        "Goal A->B must carry NavDP short memory",
    )
    require(args.retrieval_override == "off", "GT retrieval is forbidden")
    require(args.gate_override is None, "gate overrides are forbidden")
    require(not args.probe_leg1_memory, "Goal-A memory probes are forbidden")
    require(args.terminal_uturn == "off", "terminal maneuvers are forbidden")
    require(
        args.terminal_visual_refine == "off",
        "terminal visual refinement is forbidden",
    )
    require(args.arrival_shadow == "off", "arrival shadow is forbidden")
    require(
        args.trajectory_selector == "server"
        and args.trajectory_selector_scope == "all",
        "trajectory-selector oracles are forbidden",
    )
    require(
        args.oracle_candidate_seed_count == 1,
        "candidate pooling is forbidden",
    )
    require(
        args.oracle_global_subgoal_m == 0.0,
        "use --novel-b-arm instead of --oracle_global_subgoal_m",
    )
    require(
        getattr(args, "oracle_observed_frontier", "off") == "off",
        "observed-frontier interventions are forbidden",
    )
    require(
        args.deterministic_plan_seeds,
        "paired evaluation requires --deterministic_plan_seeds",
    )
    require(
        args.leg1_success_dist is None
        or float(args.leg1_success_dist) == float(args.success_dist),
        "Goal A and Goal B must use the same success radius",
    )


def main() -> None:
    validate_protocol()
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    require(not any(output_root.iterdir()), f"output is not empty: {output_root}")

    episode_dirs = sorted(
        Path(path)
        for path in glob.glob(os.path.join(args.episode_root, "episode_*"))
        if os.path.isfile(os.path.join(path, "meta", "gen_meta.json"))
    )
    if args.episode_ids:
        wanted = {
            value.strip() for value in args.episode_ids.split(",") if value.strip()
        }
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
        require(
            {path.name for path in episode_dirs} == wanted,
            "one or more requested episode IDs are missing",
        )
    if args.episodes:
        episode_dirs = episode_dirs[: args.episodes]
    require(bool(episode_dirs), "no three-leg episodes selected")

    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    metrics: list[dict] = []
    print(
        f"[novel-b] arm={NOVEL_B_ARM} episodes={len(episode_dirs)} "
        f"scene={Path(args.scene).stem}"
    )
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            metadata_path = episode_dir / "meta" / "gen_meta.json"
            metadata = json.loads(metadata_path.read_text())
            require(int(metadata.get("n_legs", -1)) == 3, "episode is not 3-leg")
            require(len(metadata.get("switches", [])) == 2, "switches are invalid")
            require(len(metadata.get("goals", [])) == 2, "goals are invalid")
            require(
                metadata.get("scene") == Path(args.scene).name,
                "episode/scene mismatch",
            )
            goal_b, goal_c = metadata["goals"]
            require(goal_b.get("kind") == "novel", "Goal B must be Novel")
            require(goal_c.get("kind") == "revisit", "Goal C must be Revisit")

            parquet_path = (
                episode_dir / "data/chunk-000/episode_000000.parquet"
            )
            rows = pd.read_parquet(parquet_path)
            require(
                len(rows) == int(metadata["n_frames"]),
                "parquet frame count mismatch",
            )
            intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
            camera_intrinsic = np.stack([
                np.asarray(row, dtype=np.float64) for row in intrinsic_raw
            ])
            switch_a, _switch_b = [int(value) for value in metadata["switches"]]
            require(0 < switch_a < len(rows), "Goal-A switch is out of bounds")

            a_hab = base.data_to_hab(metadata["A"])
            b_hab = base.data_to_hab(goal_b["pos"])
            a_xz = a_hab[[0, 2]]
            b_xz = b_hab[[0, 2]]
            start_floor, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])

            ok_a, geo_a, _path_a = base.geodesic(
                pathfinder, start_floor, a_hab
            )
            ok_b, geo_b, _path_b = base.geodesic(
                pathfinder, a_hab, b_hab
            )
            require(
                ok_a and np.isfinite(geo_a) and ok_b and np.isfinite(geo_b),
                "A/B benchmark geodesic is invalid",
            )
            geo_a = float(geo_a)
            geo_b = float(geo_b)

            rgb_root = (
                episode_dir / "videos/chunk-000/observation.images.rgb"
            )
            image_a = (rgb_root / f"{switch_a - 1}.jpg").read_bytes()
            image_b = (episode_dir / "goal_1.jpg").read_bytes()
            episode_seed = int(args.seed + episode_index)

            reset_backend = base.srv_reset(
                camera_height=float(
                    metadata.get("camera_height_m", base.CAM_H)
                ),
                seed=episode_seed,
                episode_len=int(metadata["n_frames"]),
                camera_intrinsic=camera_intrinsic,
            )
            require(reset_backend == "navdp", "unexpected policy reset backend")
            leg_a = base.run_policy_leg(
                sim,
                pathfinder,
                start_floor,
                start_yaw,
                image_a,
                a_xz,
                geo_a,
                None,
                terminal_mode="off",
                forced_gate=None,
                policy_backend=None,
                success_dist=args.success_dist,
                episode_seed=episode_seed,
                leg_index=0,
            )
            position = np.asarray(leg_a["end_pos"], dtype=np.float64)
            yaw = float(leg_a["end_psi"])

            leg_b = empty_goal_b(position, yaw, b_xz)
            if leg_a["reached"]:
                subgoal_m = ORACLE_SUBGOAL_METRES[NOVEL_B_ARM]
                if subgoal_m is None:
                    leg_b = base.run_policy_leg(
                        sim,
                        pathfinder,
                        position,
                        yaw,
                        image_b,
                        b_xz,
                        geo_b,
                        None,
                        terminal_mode="off",
                        goal_yaw=float(goal_b["yaw_habitat"]),
                        camera_intrinsic=camera_intrinsic,
                        forced_gate=None,
                        policy_backend=None,
                        success_dist=args.success_dist,
                        episode_seed=episode_seed,
                        leg_index=1,
                    )
                    leg_b["attempted"] = True
                else:
                    leg_b = run_oracle_goal_b(
                        sim,
                        pathfinder,
                        position,
                        yaw,
                        image_b,
                        b_xz,
                        episode_seed,
                        subgoal_m,
                    )

            goal_a_record = jsonable(leg_a)
            goal_b_record = jsonable(leg_b)
            goal_a_sha = canonical_sha256(goal_a_record)
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "scene": Path(args.scene).stem,
                "episode": episode_dir.name,
                "seed": episode_seed,
                "arm": NOVEL_B_ARM,
                "goal_a_sha256": goal_a_sha,
                "geodesic_m": {"A": geo_a, "B": geo_b},
                "goal_a": goal_a_record,
                "goal_b": goal_b_record,
            }
            write_json(
                output_root / f"{episode_dir.name}_audit.json", artifact
            )

            metric = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "scene": Path(args.scene).stem,
                "episode": episode_dir.name,
                "seed": episode_seed,
                "arm": NOVEL_B_ARM,
                "server_backend": args.server_backend,
                "navdp_stop_threshold": float(args.navdp_stop_threshold),
                "goal_A_controller": "native_imagegoal",
                "goal_B_controller": NOVEL_B_ARM,
                "oracle_subgoal_m": ORACLE_SUBGOAL_METRES[NOVEL_B_ARM],
                "deterministic_plan_seeds": int(args.deterministic_plan_seeds),
                "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
                "success_dist_m": float(args.success_dist),
                "max_steps": int(args.max_steps),
                "exec_horizon": int(args.exec_horizon),
                "reached_A": int(bool(leg_a["reached"])),
                "reached_B": int(bool(leg_b["reached"])),
                "B_attempted": int(bool(leg_b["attempted"])),
                "spl_A": leg_spl(leg_a, geo_a),
                "spl_B": leg_spl(leg_b, geo_b),
                "geo_A": geo_a,
                "geo_B": geo_b,
                "len_A": float(leg_a["path_len"]),
                "len_B": float(leg_b["path_len"]),
                "len_B_at_reach": leg_b.get("path_len_at_reach"),
                "steps_A": int(leg_a["steps"]),
                "steps_B": int(leg_b["steps"]),
                "final_dist_A": float(leg_a["final_goal_dist_m"]),
                "final_dist_B": float(leg_b["final_goal_dist_m"]),
                "goal_A_plan_count": len(leg_a["plans"]),
                "goal_B_plan_count": len(leg_b["plans"]),
                "goal_a_sha256": goal_a_sha,
            }
            metrics.append(metric)
            with (output_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{episode_dir.name}] arm={NOVEL_B_ARM} "
                f"A={metric['reached_A']} B={metric['reached_B']} "
                f"A_sha={goal_a_sha[:12]}"
            )

        eligible = [row for row in metrics if row["reached_A"]]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "arm": NOVEL_B_ARM,
            "episodes": len(metrics),
            "goal_A_successes": len(eligible),
            "goal_B_eligible": len(eligible),
            "goal_B_successes": sum(row["reached_B"] for row in eligible),
            "goal_B_sr_given_A": (
                sum(row["reached_B"] for row in eligible) / len(eligible)
                if eligible
                else None
            ),
        }
        write_json(output_root / "summary.json", summary)
        print("[novel-b] done", summary)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
