#!/usr/bin/env python3
"""Lifelong CEC evaluation after a frozen factual online A->B prefix.

The source is the sealed actual-online NNR benchmark.  Its factual A and B
rollouts are replayed byte-for-byte into one causal memory stream before the
three evaluated queries C -> B2 -> C2:

* C is the benchmark's controlled revisit of factual online A;
* B2 returns to the factual Novel-B goal after B has become history;
* C2 repeats C after an intervening goal switch.

The all-prior and initial-leg-only arms are identical through C.  At B2 the
former may retrieve factual B frames while the latter remains capped at the A
boundary.  Runtime role labels are never sent to the controller.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
import eval_3leg_habitat as multigoal
import eval_shared_online_novel_revisit as nnr
from deterministic_eval_protocol import file_sha256


args = base.args
RESULT_SCHEMA = "shared_online_lifelong_nnr_eval_v1_20260821"
QUERY_NAMES = ("C", "B2", "C2")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def path_remaps() -> list[tuple[Path, Path]]:
    result = []
    for raw in args.shared_path_remap:
        require("=" in raw, "shared path remap must be FROM=TO")
        source, target = raw.split("=", 1)
        require(bool(source) and bool(target), "shared path remap is empty")
        source_path = Path(source)
        target_path = Path(target)
        require(
            source_path.is_absolute() and target_path.is_absolute(),
            "shared path remap endpoints must be absolute",
        )
        result.append((source_path, target_path))
    return sorted(result, key=lambda item: len(str(item[0])), reverse=True)


def remap_sealed_path(value: str | Path) -> Path:
    original = Path(value)
    for source, target in path_remaps():
        try:
            suffix = original.relative_to(source)
        except ValueError:
            continue
        return target / suffix
    return original


def source_assets(benchmark: dict) -> tuple[Path, bytes, bytes, np.ndarray]:
    """Read a hash-identical source through an optional local path mirror."""
    source = remap_sealed_path(benchmark["source_episode"])
    metadata_path = source / "meta/gen_meta.json"
    parquet_path = source / "data/chunk-000/episode_000000.parquet"
    require(
        file_sha256(metadata_path) == benchmark["source_metadata_sha256"],
        "mirrored source metadata changed",
    )
    require(
        file_sha256(parquet_path)
        == benchmark["source_strict_v4_audit"]["source_parquet_sha256"],
        "mirrored source parquet changed",
    )
    metadata = json.loads(metadata_path.read_text())
    rows = pd.read_parquet(parquet_path)
    intrinsic = np.stack([
        np.asarray(row, dtype=np.float64)
        for row in rows.iloc[0]["observation.camera_intrinsic"]
    ])
    switch_a = int(metadata["switches"][0])
    rgb_root = source / "videos/chunk-000/observation.images.rgb"
    goal_a = (rgb_root / f"{switch_a - 1}.jpg").read_bytes()
    goal_b = (source / "goal_1.jpg").read_bytes()
    require(
        base.bytes_sha256(goal_a) == benchmark["goal_a_sha256"],
        "mirrored Goal A changed",
    )
    require(
        base.bytes_sha256(goal_b) == benchmark["goal_b_sha256"],
        "mirrored Goal B changed",
    )
    return source, goal_a, goal_b, intrinsic


def validate_cli() -> None:
    nnr.validate_cli()
    require(
        args.shared_online_nnr_arm == "cec_portability",
        "lifelong NNR currently requires the role-free CEC portability arm",
    )
    require(
        args.lifelong_history_scope
        in ("all_prior", "initial_leg_only", "forced_reject_native"),
        "unknown lifelong history scope",
    )


def validate_query_session(
    name: str,
    plans: list[dict],
    expected_index: int,
    expected_ceiling: int,
) -> dict:
    require(bool(plans), f"query {name} produced no decisions")
    first = plans[0]
    require(
        first.get("cec_goal_session_expected_start") is True,
        f"query {name} was not recognized as a new hub goal session",
    )
    require(
        first.get("cec_goal_session_started") is True,
        f"query {name} did not open a new MemNav goal session",
    )
    require(
        int(first.get("cec_goal_session_index", -1)) == expected_index,
        f"query {name} has a non-contiguous session index",
    )
    require(
        first.get("cec_long_term_memory_preserved") is True,
        f"query {name} did not attest preserved long-term memory",
    )
    require(
        int(first.get("cec_candidate_ceiling", -2)) == int(expected_ceiling),
        f"query {name} used the wrong causal candidate ceiling",
    )
    for later in plans[1:]:
        require(
            later.get("cec_goal_session_expected_start") is False
            and later.get("cec_goal_session_started") is False,
            f"query {name} reopened a session within one goal",
        )
        require(
            int(later.get("cec_goal_session_index", -1)) == expected_index,
            f"query {name} changed session index within one goal",
        )
        require(
            int(later.get("cec_candidate_ceiling", -2))
            == int(expected_ceiling),
            f"query {name} changed its frozen candidate ceiling",
        )
    return {
        "query": name,
        "goal_session_index": expected_index,
        "goal_start_frame": int(first["cec_goal_start_frame"]),
        "candidate_ceiling": int(expected_ceiling),
    }


def validate_forced_reject(name: str, plans: list[dict]) -> None:
    """Every decision of the shared-native baseline must refuse takeover."""
    for row in plans:
        if row.get("cec_takeover") is None:
            continue
        require(
            row.get("cec_forced_reject_native") is True,
            f"query {name}: hub is not in force-reject-native mode",
        )
        require(
            row.get("cec_takeover") is False,
            f"query {name}: forced-reject arm granted a takeover",
        )
        require(
            row.get("cec_action_state") in ("fallback", "forced_reject"),
            f"query {name}: forced-reject arm left the fallback controller",
        )


def validate_not_forced(name: str, plans: list[dict]) -> None:
    for row in plans:
        if row.get("cec_takeover") is None:
            continue
        require(
            row.get("cec_forced_reject_native") is not True,
            f"query {name}: hub unexpectedly runs force-reject-native",
        )


def plan_stats(plans: list[dict], a_ceiling: int, b_ceiling: int) -> dict:
    decisions = [
        row for row in plans if row.get("cec_takeover") is not None
    ]
    accepted = [row for row in decisions if row.get("cec_takeover") is True]
    shadow = [
        row for row in decisions if row.get("cec_shadow_takeover") is True
    ]
    anchors = [
        int(row["cec_selected_anchor"])
        for row in accepted if row.get("cec_selected_anchor") is not None
    ]
    latency = [
        float(row["cec_total_decision_ms"])
        for row in decisions if row.get("cec_total_decision_ms") is not None
    ]
    return {
        "decisions": len(decisions),
        "takeovers": len(accepted),
        "shadow_takeovers": len(shadow),
        "anchors": anchors,
        "used_A_anchor": any(anchor <= a_ceiling for anchor in anchors),
        "used_factual_B_anchor": any(
            a_ceiling < anchor <= b_ceiling for anchor in anchors
        ),
        "decision_ms_median": (
            float(np.median(latency)) if latency else None
        ),
        "decision_ms_max": max(latency) if latency else None,
    }


def ensure_contiguous_memory(legs: list[tuple[str, dict]]) -> tuple[int, int]:
    indexed = []
    for name, leg in legs:
        values = [
            int(row["frame_idx"])
            for row in leg["memory_trace"]
            if row.get("frame_idx") is not None
        ]
        require(bool(values), f"{name} wrote no causal memory")
        indexed.extend(values)
    require(
        indexed == list(range(indexed[0], indexed[-1] + 1)),
        "causal A/B/query memory contains a missing or duplicate frame",
    )
    return indexed[0], indexed[-1]


def run_query(
    simulator,
    pathfinder,
    start_position,
    start_yaw,
    goal_image: bytes,
    goal_position: np.ndarray,
    goal_yaw: float,
    camera_intrinsic: np.ndarray,
    episode_seed: int,
    leg_index: int,
    candidate_ceiling: int | None,
) -> tuple[dict, float]:
    ok, geodesic_m, _ = base.geodesic(
        pathfinder, start_position, goal_position)
    require(ok and np.isfinite(geodesic_m), "query geodesic is invalid")
    leg = base.run_policy_leg(
        simulator,
        pathfinder,
        start_position,
        start_yaw,
        goal_image,
        goal_position[[0, 2]],
        float(geodesic_m),
        None,
        terminal_mode="off",
        goal_yaw=float(goal_yaw),
        camera_intrinsic=camera_intrinsic,
        policy_backend=nnr.controller_backend(),
        episode_seed=episode_seed,
        leg_index=leg_index,
        candidate_ceiling_override=candidate_ceiling,
    )
    return leg, float(geodesic_m)


def main() -> None:
    validate_cli()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")

    scene_file = Path(args.scene).resolve()
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene_file.stem, "scene/benchmark mismatch")
    manifest = nnr.load_manifest(scene_root)
    manifest_rows = {row["episode"]: row for row in manifest["accepted"]}
    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "benchmark.json").is_file()
    )
    if args.episode_ids:
        wanted = {
            item.strip() for item in args.episode_ids.split(",")
            if item.strip()
        }
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no lifelong NNR episodes selected")

    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        for episode_dir in episode_dirs:
            benchmark, benchmark_sha = nnr.load_benchmark(
                episode_dir, manifest_rows)
            source, image_a, image_b, intrinsic = source_assets(benchmark)
            require(
                file_sha256(scene_file)
                == benchmark["source_scene_asset_sha256"],
                "scene asset changed",
            )
            trace_root = Path(args.shared_leg1_trace_root).resolve()
            sealed_trace_root = remap_sealed_path(benchmark["trace_root"])
            require(
                str(trace_root) == str(sealed_trace_root.resolve()),
                "trace root differs from sealed benchmark",
            )
            trace_a_path = trace_root / benchmark["online_a_trace"]
            trace_b_path = trace_root / benchmark["online_b_trace"]
            require(
                file_sha256(trace_a_path) == benchmark["online_a_trace_sha256"],
                "online-A trace changed",
            )
            require(
                file_sha256(trace_b_path) == benchmark["online_b_trace_sha256"],
                "online-B trace changed",
            )

            metadata = json.loads((source / "meta/gen_meta.json").read_text())
            rows = pd.read_parquet(
                source / "data/chunk-000/episode_000000.parquet")
            start_position, start_yaw = base.parquet_pose_hab(
                rows.iloc[0]["action"])
            b_goal = metadata["goals"][0]
            b_floor = base.data_to_hab(b_goal["pos"])
            b_yaw = float(b_goal["yaw_habitat"])
            c_goal = benchmark["goal_c"]
            c_floor = np.asarray(c_goal["floor_position"], dtype=np.float64)
            c_yaw = float(c_goal["yaw_rad"])
            image_c_path = episode_dir / benchmark["goal_c_asset"]["rgb"]
            require(
                file_sha256(image_c_path)
                == benchmark["goal_c_asset"]["rgb_sha256"],
                "Goal C image changed",
            )
            image_c = image_c_path.read_bytes()
            episode_seed = int(benchmark["episode_seed"])
            base.srv_reset(
                camera_height=float(benchmark["camera_height_m"]),
                seed=episode_seed,
                episode_len=(
                    int(benchmark["online_a_steps"])
                    + int(benchmark["online_b_steps"])
                    + 3 * int(args.max_steps)
                ),
                camera_intrinsic=intrinsic,
            )

            leg_a, trace_a_sha = base.replay_shared_leg1(
                simulator, trace_root, episode_dir.name, episode_seed,
                image_a, start_position, start_yaw)
            require(leg_a["reached"], "sealed online A is not successful")
            require(
                len(leg_a["memory_trace"])
                == int(benchmark["online_a_steps"]),
                "online-A replay omitted causal memory frames",
            )
            leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
                simulator, trace_root, episode_dir.name, episode_seed,
                image_b, leg_a["end_pos"], leg_a["end_psi"])
            require(leg_b["reached"], "sealed online B is not successful")
            require(
                len(leg_b["memory_trace"])
                == int(benchmark["online_b_steps"]),
                "online-B replay omitted causal memory frames",
            )
            require(
                trace_a_sha == benchmark["online_a_trace_sha256"]
                and trace_b_sha == benchmark["online_b_trace_sha256"],
                "factual prefix hash mismatch",
            )
            a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
            b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
            require(b_ceiling > a_ceiling, "online B did not extend memory")
            require(
                a_ceiling + 1 == int(benchmark["online_a_steps"])
                and b_ceiling + 1 == (
                    int(benchmark["online_a_steps"])
                    + int(benchmark["online_b_steps"])
                ),
                "factual A/B replay memory indices are not contiguous",
            )
            require(
                int(c_goal["max_online_a_covis_frame"])
                <= int(a_ceiling),
                "Goal C support escaped the factual A boundary",
            )

            reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
            leg_c, geo_c = run_query(
                simulator, pathfinder, leg_b["end_pos"], leg_b["end_psi"],
                image_c, c_floor, c_yaw, intrinsic, episode_seed, 2,
                a_ceiling)
            legs = {"C": leg_c}
            geodesics = {"C": geo_c}
            receipts = []
            if leg_c["plans"]:
                receipts.append(validate_query_session(
                    "C", leg_c["plans"], 1, a_ceiling))

            prefix_alive = bool(leg_c["reached"])
            if prefix_alive:
                b2_ceiling = (
                    b_ceiling
                    if args.lifelong_history_scope
                    in ("all_prior", "forced_reject_native")
                    else a_ceiling
                )
                leg_b2, geo_b2 = run_query(
                    simulator, pathfinder, leg_c["end_pos"], leg_c["end_psi"],
                    image_b, b_floor, b_yaw, intrinsic, episode_seed, 3,
                    b2_ceiling)
                receipts.append(validate_query_session(
                    "B2", leg_b2["plans"], 2, b2_ceiling))
            else:
                leg_b2 = multigoal.empty_leg(
                    leg_c["end_pos"], leg_c["end_psi"], b_floor[[0, 2]])
                geo_b2 = float("nan")
            legs["B2"] = leg_b2
            geodesics["B2"] = geo_b2

            prefix_alive = prefix_alive and bool(leg_b2["reached"])
            if prefix_alive:
                c2_override = (
                    None
                    if args.lifelong_history_scope
                    in ("all_prior", "forced_reject_native")
                    else a_ceiling
                )
                leg_c2, geo_c2 = run_query(
                    simulator, pathfinder, leg_b2["end_pos"], leg_b2["end_psi"],
                    image_c, c_floor, c_yaw, intrinsic, episode_seed, 4,
                    c2_override)
                expected_c2_ceiling = (
                    int(leg_c2["plans"][0]["cec_goal_start_frame"]) - 1
                    if c2_override is None else a_ceiling
                )
                receipts.append(validate_query_session(
                    "C2", leg_c2["plans"], 3, expected_c2_ceiling))
            else:
                leg_c2 = multigoal.empty_leg(
                    leg_b2["end_pos"], leg_b2["end_psi"], c_floor[[0, 2]])
                geo_c2 = float("nan")
            legs["C2"] = leg_c2
            geodesics["C2"] = geo_c2

            for name in QUERY_NAMES:
                if not legs[name]["plans"]:
                    continue
                if args.lifelong_history_scope == "forced_reject_native":
                    validate_forced_reject(name, legs[name]["plans"])
                else:
                    validate_not_forced(name, legs[name]["plans"])
            memory_legs = [("A", leg_a), ("B", leg_b)] + [
                (name, legs[name]) for name in QUERY_NAMES
                if legs[name]["memory_trace"]
            ]
            memory_first, memory_last = ensure_contiguous_memory(memory_legs)
            stats = {
                name: plan_stats(legs[name]["plans"], a_ceiling, b_ceiling)
                for name in QUERY_NAMES
            }
            reached = [bool(legs[name]["reached"]) for name in QUERY_NAMES]
            completed = 0
            for success in reached:
                if not success:
                    break
                completed += 1
            metric = {
                "result_schema": RESULT_SCHEMA,
                "scene": scene_file.stem,
                "episode": episode_dir.name,
                "benchmark_sha256": benchmark_sha,
                "history_scope": args.lifelong_history_scope,
                "runtime_role_visible": 0,
                "online_A_trace_sha256": trace_a_sha,
                "online_B_trace_sha256": trace_b_sha,
                "online_A_candidate_ceiling": a_ceiling,
                "online_B_candidate_ceiling": b_ceiling,
                "memory_first_frame": memory_first,
                "memory_last_frame": memory_last,
                "reached_C": int(reached[0]),
                "reached_B2": int(reached[1]),
                "reached_C2": int(reached[2]),
                "evaluated_B2": int(reached[0]),
                "evaluated_C2": int(all(reached[:2])),
                "queries_completed_before_first_failure": completed,
                "query_joint_success": int(all(reached)),
                "B2_used_factual_B_anchor": int(
                    stats["B2"]["used_factual_B_anchor"]),
                "C2_reopened_goal_session": int(len(receipts) == 3),
                "navdp_short_fifo_reset_before_C": 1,
                "navdp_short_fifo_reset_receipt": json.dumps(
                    reset_receipt, sort_keys=True),
            }
            for name in QUERY_NAMES:
                leg = legs[name]
                metric[f"geo_{name}"] = geodesics[name]
                metric[f"steps_{name}"] = int(leg["steps"])
                metric[f"len_{name}"] = float(leg["path_len"])
                metric[f"final_dist_{name}"] = float(
                    leg["final_goal_dist_m"])
                metric[f"cec_takeovers_{name}"] = stats[name]["takeovers"]
                metric[f"cec_shadow_takeovers_{name}"] = stats[name][
                    "shadow_takeovers"]
                metric[f"cec_decision_ms_median_{name}"] = stats[name][
                    "decision_ms_median"]
                metric[f"cec_decision_ms_max_{name}"] = stats[name][
                    "decision_ms_max"]
            metrics.append(metric)

            (output / f"{episode_dir.name}_plans.json").write_text(
                json.dumps({
                    "result_schema": RESULT_SCHEMA,
                    "history_scope": args.lifelong_history_scope,
                    "runtime_role_visible": False,
                    "goal_session_receipts": receipts,
                    "frozen_legA": leg_a["plans"],
                    "frozen_legB": leg_b["plans"],
                    "queries": {
                        name: legs[name]["plans"] for name in QUERY_NAMES
                    },
                    "rollout_traces": {
                        "A": leg_a["rollout_trace"],
                        "B": leg_b["rollout_trace"],
                        **{
                            name: legs[name]["rollout_trace"]
                            for name in QUERY_NAMES
                        },
                    },
                    "memory_traces": {
                        "A": leg_a["memory_trace"],
                        "B": leg_b["memory_trace"],
                        **{
                            name: legs[name]["memory_trace"]
                            for name in QUERY_NAMES
                        },
                    },
                }, sort_keys=True, allow_nan=False) + "\n")
            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{episode_dir.name}] history={args.lifelong_history_scope} "
                f"queries={''.join(str(int(value)) for value in reached)} "
                f"B-memory={int(stats['B2']['used_factual_B_anchor'])}"
            )

        summary = {
            "result_schema": RESULT_SCHEMA,
            "history_scope": args.lifelong_history_scope,
            "episodes": len(metrics),
            "runtime_role_visible": False,
            "frozen_actual_online_prefix": "A_then_Novel_B",
            "query_sequence": list(QUERY_NAMES),
            "C_success": sum(row["reached_C"] for row in metrics),
            "B2_success_given_C": sum(
                row["reached_B2"] for row in metrics if row["reached_C"]),
            "C2_success_given_CB2": sum(
                row["reached_C2"] for row in metrics
                if row["reached_C"] and row["reached_B2"]),
            "query_joint_success": sum(
                row["query_joint_success"] for row in metrics),
            "B2_factual_B_anchor_use_given_evaluated": sum(
                row["B2_used_factual_B_anchor"] for row in metrics
                if row["evaluated_B2"]),
            "claim_scope": (
                "pipeline/lifecycle pilot until factual-B visual support is "
                "sealed and paired all-prior versus initial-only is complete"
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
