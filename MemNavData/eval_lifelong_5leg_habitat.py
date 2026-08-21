#!/usr/bin/env python3
"""Five-leg actual-online test of accumulating CEC memory.

The source population is the strict role-paired three-leg benchmark.  Without
creating or reading a runtime role label, this evaluator executes the fixed
goal sequence A -> B -> C -> B -> C:

* A is the initial ImageGoal;
* B was Novel with respect to the A prefix;
* C revisits the old A trajectory;
* the second B tests memory acquired while actually navigating the first B;
* the second C tests repeated long-horizon reuse after another goal switch.

The same JPEG is deliberately reused for B and C.  This makes the benchmark a
strict test of goal-session lifecycle: each reappearance must receive a fresh
causal candidate ceiling while the long-term RGB stream remains intact.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from eval_3leg_habitat import empty_leg, leg_spl, mean_or_none, require
from multigoal_benchmark_contract import (
    ROLE_SEQUENCE,
    ROLE_SYMMETRIC_PROTOCOL,
    RoleSymmetryObservation,
    validate_role_symmetric_contract,
)


args = base.args
LEG_NAMES = ("A", "B", "C", "B2", "C2")
ANALYSIS_ROLES = (
    "initial_imagegoal",
    "novel_at_first_presentation",
    "revisit_of_initial_history",
    "revisit_of_online_acquired_B",
    "repeated_revisit_C",
)
PROTOCOL = "actual_online_lifelong_abcbc_v1_20260821"


def cec_stats(plans: list[dict]) -> dict:
    decisions = [
        bool(plan["cec_takeover"])
        for plan in plans
        if plan.get("cec_takeover") is not None
    ]
    anchors = [
        int(plan["cec_selected_anchor"])
        for plan in plans
        if (plan.get("cec_takeover") is True
            and plan.get("cec_selected_anchor") is not None)
    ]
    return {
        "decisions": len(decisions),
        "takeovers": sum(decisions),
        "any_takeover": bool(any(decisions)),
        "anchors": anchors,
    }


def validate_goal_session(
        leg_name: str,
        plans: list[dict],
        expected_session_index: int,
        history_scope: str,
        initial_leg_ceiling: int | None) -> dict:
    require(bool(plans), f"leg {leg_name} produced no policy decision")
    first = plans[0]
    require(
        first.get("cec_goal_session_expected_start") is True,
        f"leg {leg_name} was not recognized as a new hub goal session",
    )
    require(
        first.get("cec_goal_session_started") is True,
        f"leg {leg_name} did not open a new MemNav goal session",
    )
    require(
        int(first.get("cec_goal_session_index", -1))
        == int(expected_session_index),
        f"leg {leg_name} goal-session index is not contiguous",
    )
    require(
        first.get("cec_long_term_memory_preserved") is True,
        f"leg {leg_name} did not attest preserved long-term memory",
    )
    for later in plans[1:]:
        require(
            later.get("cec_goal_session_expected_start") is False
            and later.get("cec_goal_session_started") is False,
            f"leg {leg_name} reopened its goal session within one query",
        )
        require(
            int(later.get("cec_goal_session_index", -1))
            == int(expected_session_index),
            f"leg {leg_name} changed goal-session index within one query",
        )

    goal_start = int(first["cec_goal_start_frame"])
    ceiling = int(first["cec_candidate_ceiling"])
    if leg_name == "A" or history_scope == "all_prior":
        require(
            ceiling == goal_start - 1,
            f"leg {leg_name} did not use its natural causal ceiling",
        )
    else:
        require(initial_leg_ceiling is not None, "missing online-A ceiling")
        require(
            ceiling == int(initial_leg_ceiling),
            f"leg {leg_name} escaped the initial-leg-only ablation",
        )
    return {
        "goal_start_frame": goal_start,
        "candidate_ceiling": ceiling,
        "goal_session_index": int(expected_session_index),
    }


def load_episode_contract(
        simulator, pathfinder, episode_dir: Path) -> dict:
    metadata = json.loads(
        (episode_dir / "meta/gen_meta.json").read_text())
    require(
        metadata.get("gen_protocol") == ROLE_SYMMETRIC_PROTOCOL,
        "lifelong source must use the strict role-paired protocol",
    )
    require(
        tuple(metadata.get("role_sequence") or ()) == ROLE_SEQUENCE,
        "lifelong source must be initial/Novel/Revisit",
    )
    require(int(metadata.get("n_legs", -1)) == 3, "source is not three-leg")
    require(len(metadata.get("switches", [])) == 2, "invalid source switches")
    require(len(metadata.get("goals", [])) == 2, "invalid source goals")

    rows = pd.read_parquet(
        episode_dir / "data/chunk-000/episode_000000.parquet")
    require(len(rows) == int(metadata["n_frames"]), "source frame count changed")
    intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack([
        np.asarray(row, dtype=np.float64) for row in intrinsic_raw
    ])
    switch_a, switch_b = [int(value) for value in metadata["switches"]]
    rgb_root = episode_dir / "videos/chunk-000/observation.images.rgb"

    a_hab = base.data_to_hab(metadata["A"])
    goal_b, goal_c = metadata["goals"]
    b_hab = base.data_to_hab(goal_b["pos"])
    c_hab = base.data_to_hab(goal_c["pos"])
    start_floor, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])

    distances = []
    for source, target in (
            (start_floor, a_hab),
            (a_hab, b_hab),
            (b_hab, c_hab),
            (c_hab, b_hab),
            (b_hab, c_hab)):
        ok, distance, _path = base.geodesic(pathfinder, source, target)
        require(ok and np.isfinite(distance), "five-leg geodesic is invalid")
        distances.append(float(distance))

    image_a = (rgb_root / f"{switch_a - 1}.jpg").read_bytes()
    image_b = (episode_dir / "goal_1.jpg").read_bytes()
    image_c = (episode_dir / "goal_2.jpg").read_bytes()
    image_b_terminal = (rgb_root / f"{switch_b - 1}.jpg").read_bytes()
    a_terminal, _ = base.parquet_pose_hab(rows.iloc[switch_a - 1]["action"])
    b_terminal, b_terminal_yaw = base.parquet_pose_hab(
        rows.iloc[switch_b - 1]["action"])
    observation = RoleSymmetryObservation(
        geo_a_m=distances[0],
        geo_b_m=distances[1],
        initial_pose_error_m=float(np.linalg.norm(
            start_floor - base.data_to_hab(metadata["start"]))),
        a_terminal_pose_error_m=float(np.linalg.norm(a_terminal - a_hab)),
        b_terminal_pose_error_m=float(np.linalg.norm(b_terminal - b_hab)),
        b_terminal_yaw_error_deg=abs(float(np.degrees(base.wrap_angle(
            b_terminal_yaw - float(goal_b["yaw_habitat"]))))),
        goal_b_matches_terminal_rgb=(image_b == image_b_terminal),
    )
    contract = validate_role_symmetric_contract(metadata, observation)
    require(contract["ok"], "strict source contract failed: " + "; ".join(
        contract["issues"]))
    return {
        "metadata": metadata,
        "intrinsic": intrinsic,
        "start_floor": start_floor,
        "start_yaw": float(start_yaw),
        "positions": (a_hab, b_hab, c_hab, b_hab, c_hab),
        "images": (image_a, image_b, image_c, image_b, image_c),
        "goal_yaws": (
            None,
            float(goal_b["yaw_habitat"]),
            float(goal_c["yaw_habitat"]),
            float(goal_b["yaw_habitat"]),
            float(goal_c["yaw_habitat"]),
        ),
        "geodesics": tuple(distances),
        "contract": contract,
    }


def main() -> None:
    require(
        args.server_backend == "cec_portability",
        "five-leg accumulation test requires the role-free CEC portability hub",
    )
    require(args.leg1_mode == "policy", "five-leg A must be actual online policy")
    require(not args.reset_memory, "five-leg evaluation must preserve memory")
    require(
        args.navdp_goal_switch_reset == "carry",
        "five-leg evaluation carries the controller's short context",
    )
    require(args.lifelong_sequence == "natural_abcbc", "unknown sequence")
    require(args.terminal_uturn == "off", "position SR disables terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "position SR disables terminal visual refinement",
    )
    require(
        args.trajectory_selector == "server"
        and args.trajectory_selector_scope == "all",
        "five-leg evaluation requires the frozen server selector",
    )
    require(args.deterministic_plan_seeds, "paired plan seeds are required")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output must be empty")
    episode_dirs = sorted(
        Path(path) for path in glob.glob(
            os.path.join(args.episode_root, "episode_*"))
        if Path(path, "meta/gen_meta.json").is_file()
    )
    if args.episode_ids:
        selected = {
            item.strip() for item in args.episode_ids.split(",") if item.strip()
        }
        episode_dirs = [path for path in episode_dirs if path.name in selected]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no five-leg source episodes selected")

    simulator = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            source = load_episode_contract(simulator, pathfinder, episode_dir)
            metadata = source["metadata"]
            episode_seed = int(args.seed) + episode_index
            base.srv_reset(
                camera_height=float(metadata.get("camera_height_m", base.CAM_H)),
                seed=episode_seed,
                episode_len=int(args.max_steps) * len(LEG_NAMES),
                camera_intrinsic=source["intrinsic"],
            )

            position = source["start_floor"].copy()
            yaw = float(source["start_yaw"])
            legs = []
            session_receipts = []
            initial_leg_ceiling = None
            prefix_alive = True
            for leg_index, (name, goal_position, goal_image, goal_yaw, geo) in enumerate(
                    zip(
                        LEG_NAMES,
                        source["positions"],
                        source["images"],
                        source["goal_yaws"],
                        source["geodesics"],
                    )):
                goal_xz = np.asarray(goal_position, dtype=np.float64)[[0, 2]]
                if prefix_alive:
                    ceiling_override = None
                    if (leg_index > 0
                            and args.lifelong_history_scope
                            == "initial_leg_only"):
                        require(
                            initial_leg_ceiling is not None,
                            "online-A memory boundary is unavailable",
                        )
                        ceiling_override = int(initial_leg_ceiling)
                    leg = base.run_policy_leg(
                        simulator,
                        pathfinder,
                        position,
                        yaw,
                        goal_image,
                        goal_xz,
                        geo,
                        None,
                        terminal_mode="off",
                        goal_yaw=goal_yaw,
                        camera_intrinsic=source["intrinsic"],
                        forced_gate=args.gate_override,
                        policy_backend=None,
                        episode_seed=episode_seed,
                        leg_index=leg_index,
                        candidate_ceiling_override=ceiling_override,
                    )
                    receipt = validate_goal_session(
                        name,
                        leg["plans"],
                        leg_index + 1,
                        args.lifelong_history_scope,
                        initial_leg_ceiling,
                    )
                    session_receipts.append(receipt)
                    if leg_index == 0:
                        memory_indices = [
                            int(row["frame_idx"])
                            for row in leg["memory_trace"]
                            if row.get("frame_idx") is not None
                        ]
                        require(bool(memory_indices), "online A wrote no CEC history")
                        require(
                            memory_indices == list(range(
                                memory_indices[0], memory_indices[-1] + 1)),
                            "online-A memory indices are not contiguous",
                        )
                        initial_leg_ceiling = int(memory_indices[-1])
                else:
                    leg = empty_leg(position, yaw, goal_xz)
                legs.append(leg)
                position = np.asarray(leg["end_pos"], dtype=np.float64)
                yaw = float(leg["end_psi"])
                prefix_alive = prefix_alive and bool(leg["reached"])

            reached = [bool(leg["reached"]) for leg in legs]
            completed = 0
            for success in reached:
                if not success:
                    break
                completed += 1
            statistics = [cec_stats(leg["plans"]) for leg in legs]
            post_a_use = [
                any(anchor > int(initial_leg_ceiling)
                    for anchor in statistics[index]["anchors"])
                if initial_leg_ceiling is not None else False
                for index in range(len(legs))
            ]
            total_path = sum(float(leg["path_len"]) for leg in legs)
            joint = all(reached)
            metric = {
                "episode": episode_dir.name,
                "seed": episode_seed,
                "protocol": PROTOCOL,
                "history_scope": args.lifelong_history_scope,
                "runtime_role_visible": 0,
                "actual_online_A": 1,
                "long_term_memory_preserved": 1,
                "initial_leg_candidate_ceiling": initial_leg_ceiling,
                "reached_A": int(reached[0]),
                "reached_B": int(reached[1]),
                "reached_C": int(reached[2]),
                "reached_B2": int(reached[3]),
                "reached_C2": int(reached[4]),
                "evaluated_A": 1,
                "evaluated_B": int(reached[0]),
                "evaluated_C": int(all(reached[:2])),
                "evaluated_B2": int(all(reached[:3])),
                "evaluated_C2": int(all(reached[:4])),
                "goals_completed_before_first_failure": completed,
                "joint_success": int(joint),
                "joint_spl": base.spl(
                    joint, sum(source["geodesics"]), total_path),
                "used_post_A_memory_B2": int(post_a_use[3]),
                "used_post_A_memory_C2": int(post_a_use[4]),
                "cec_takeovers_A": statistics[0]["takeovers"],
                "cec_takeovers_B": statistics[1]["takeovers"],
                "cec_takeovers_C": statistics[2]["takeovers"],
                "cec_takeovers_B2": statistics[3]["takeovers"],
                "cec_takeovers_C2": statistics[4]["takeovers"],
            }
            for index, name in enumerate(LEG_NAMES):
                metric[f"geo_{name}"] = source["geodesics"][index]
                metric[f"len_{name}"] = float(legs[index]["path_len"])
                metric[f"steps_{name}"] = int(legs[index]["steps"])
                metric[f"spl_{name}"] = leg_spl(
                    legs[index], source["geodesics"][index])
                metric[f"final_dist_{name}"] = float(
                    legs[index]["final_goal_dist_m"])
            metrics.append(metric)

            plans_payload = {
                "schema": "actual_online_lifelong_abcbc_plans_v1",
                "protocol": PROTOCOL,
                "analysis_roles_not_visible_to_policy": dict(
                    zip(LEG_NAMES, ANALYSIS_ROLES)),
                "history_scope": args.lifelong_history_scope,
                "initial_leg_candidate_ceiling": initial_leg_ceiling,
                "goal_session_receipts": session_receipts,
                "plans": {
                    name: legs[index]["plans"]
                    for index, name in enumerate(LEG_NAMES)
                },
                "memory_traces": {
                    name: legs[index]["memory_trace"]
                    for index, name in enumerate(LEG_NAMES)
                },
                "rollout_traces": {
                    name: legs[index]["rollout_trace"]
                    for index, name in enumerate(LEG_NAMES)
                },
            }
            (output / f"{episode_dir.name}_plans.json").write_text(
                json.dumps(plans_payload, indent=2, sort_keys=True) + "\n")

            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{episode_dir.name}] history={args.lifelong_history_scope} "
                f"survival={completed}/5 reached={''.join(str(int(x)) for x in reached)}"
            )

        summary = {
            "schema": "actual_online_lifelong_abcbc_summary_v1",
            "protocol": PROTOCOL,
            "episodes": len(metrics),
            "scene": Path(args.scene).stem,
            "history_scope": args.lifelong_history_scope,
            "runtime_role_visible": False,
            "goal_sequence": list(LEG_NAMES),
            "analysis_roles": dict(zip(LEG_NAMES, ANALYSIS_ROLES)),
            "survival_rate_after_leg": {
                name: mean_or_none([
                    float(all(bool(row[f"reached_{prefix}"])
                              for prefix in LEG_NAMES[:index + 1]))
                    for row in metrics
                ])
                for index, name in enumerate(LEG_NAMES)
            },
            "mean_goals_completed_before_first_failure": mean_or_none([
                row["goals_completed_before_first_failure"] for row in metrics
            ]),
            "joint_SR": mean_or_none([
                row["joint_success"] for row in metrics
            ]),
            "mean_joint_spl": mean_or_none([
                row["joint_spl"] for row in metrics
            ]),
            "B2_post_A_memory_use_rate_given_evaluated": mean_or_none([
                row["used_post_A_memory_B2"] for row in metrics
                if row["evaluated_B2"]
            ]),
            "C2_post_A_memory_use_rate_given_evaluated": mean_or_none([
                row["used_post_A_memory_C2"] for row in metrics
                if row["evaluated_C2"]
            ]),
            "claim_scope": (
                "internal sequential capability; paired all-prior versus "
                "initial-leg-only confirmation is required before a paper claim"
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("[eval5leg] done", summary)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
