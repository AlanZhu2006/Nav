#!/usr/bin/env python3
"""Role-free closed-loop evaluation after an exact frozen online-A replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from audit_shared_online_role_pairs import audit as audit_shared_benchmark
from audit_hm3d_table3_length_role_pairs import audit as audit_table3_benchmark
from shared_online_double_revisit_runtime import replay_online_a, sha256_file
from shared_online_role_pair_contract import runtime_query as shared_runtime_query
from hm3d_table3_length_contract import runtime_query as table3_runtime_query


args = base.args
RESULT_SCHEMA = "shared_online_role_pair_closed_loop_v2_depth_audit_20260819"
CEC_LATENCY_FIELDS = (
    "cec_probe_ms",
    "cec_certificate_ms",
    "cec_projection_ms",
    "cec_controller_ms",
    "cec_depth_sidecar_ms",
    "cec_context_shadow_ms",
    "cec_total_decision_ms",
)


def audit_benchmark(root: Path) -> dict:
    if args.role_pair_scope == "table3_length":
        return audit_table3_benchmark(root)
    return audit_shared_benchmark(root)


def runtime_query(query: dict) -> dict:
    if args.role_pair_scope == "table3_length":
        return table3_runtime_query(query)
    return shared_runtime_query(query)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def resolve_arm() -> tuple[str, str | None]:
    if args.server_backend == "cec_portability":
        require(args.hybrid_route == "phase", "CEC hub owns all routing")
        require(
            args.revisit_adapter == "legacy_metric",
            "CEC hub owns the proof-bound adapter",
        )
        require(args.novel_port is None, "CEC hub uses one public endpoint")
        return "cec_portability", "navdp"
    if args.server_backend == "navdp":
        require(args.hybrid_route == "phase", "native arm must use phase label")
        require(args.novel_port is None, "native arm must use one NavDP server")
        return "native", None
    require(args.server_backend == "hybrid_pose", "unsupported server backend")
    require(args.novel_port is not None, "hybrid arm requires --novel_port")
    if args.hybrid_route == "native_sidecar":
        require(
            args.revisit_adapter == "legacy_metric",
            "native-sidecar arm must not enable a Revisit adapter",
        )
        return "native_sidecar", "navdp"
    if args.hybrid_route == "phase":
        if args.revisit_adapter == "legacy_metric":
            return "raw_direct", "navdp_mix"
        if args.revisit_adapter == "raw_fixed_bearing_v1":
            return "raw_fixed_bearing", "navdp_mix"
        raise RuntimeError(
            "phase ablation requires legacy_metric or raw_fixed_bearing_v1"
        )
    if args.hybrid_route == "memory_geometry":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "geometry-fixed arm requires the frozen bearing adapter",
        )
        return "geometry_fixed", "navdp_auto"
    if args.hybrid_route == "certified_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "certified arm requires the frozen bearing adapter",
        )
        return "certified", "navdp_auto"
    if args.hybrid_route == "certified_unthresholded_witness":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "unthresholded witness arm requires the frozen bearing adapter",
        )
        require(
            args.certified_cdec_rescue == "off",
            "unthresholded witness must keep the frozen geometry proposal",
        )
        return "unthresholded_witness", "navdp_auto"
    if args.hybrid_route == "certified_semantic_first":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "semantic-first arm requires the frozen bearing adapter",
        )
        return "semantic_first_certified", "navdp_auto"
    if args.hybrid_route == "learned_pi3x_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "learned Pi3X arm requires the frozen bearing adapter",
        )
        return "learned_pi3x_spatial", "navdp_auto"
    raise RuntimeError(
        "role-pair evaluator supports only "
        "native/native-sidecar/raw/geometry/certified/"
        "unthresholded-witness/semantic-first/"
        "learned-pi3x"
    )


def validate_cli() -> tuple[str, str | None]:
    arm, backend = resolve_arm()
    require(args.leg1_mode == "shared_trace", "online A requires shared_trace")
    require(
        not args.shared_leg1_trace_root,
        "online A is bound by role_pairs.json, not legacy trace root",
    )
    require(args.leg1_goal_source == "own", "Goal-A swapping is forbidden")
    require(not args.write_leg1_trace, "frozen online A cannot be rewritten")
    require(not args.stop_after_leg1, "query rollout cannot stop after replay")
    require(not args.reset_memory, "online-A memory must be preserved")
    require(args.terminal_uturn == "off", "position SR forbids terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "position SR forbids terminal visual refinement",
    )
    require(args.retrieval_override == "off", "retrieval oracle is forbidden")
    require(args.gate_override is None, "gate oracle is forbidden")
    require(args.trajectory_selector == "server", "trajectory oracle is forbidden")
    require(args.oracle_candidate_seed_count == 1, "candidate oracle is forbidden")
    require(args.oracle_global_subgoal_m == 0.0, "global oracle is forbidden")
    require(args.oracle_observed_frontier == "off", "frontier oracle is forbidden")
    require(args.deterministic_plan_seeds, "paired queries require fixed seeds")
    require(args.agent_radius == 0.30, "benchmark uses a 0.30 m agent radius")
    require(args.exec_horizon == 8, "formal NavDP execution horizon is eight")
    require(args.certified_cdec_rescue == "off", "CDEC rescue is out of scope")
    require(args.certified_stagnation_graph == "off", "graph rescue is out of scope")
    if args.cec_initial_bearing_alignment != "off":
        require(
            arm == "cec_portability",
            "CEC bearing alignment requires the proof-carrying portability hub",
        )
        require(
            args.role_pair_query_role == "all",
            "CEC bearing alignment cannot receive a runtime role filter",
        )
        if args.cec_initial_bearing_alignment == "first_certified":
            require(
                args.role_pair_scope == "consumed_integration",
                "ideal CEC bearing alignment is a consumed mechanism only",
            )
            require(
                bool(args.role_pair_query_manifest),
                "ideal bearing alignment requires a frozen consumed subset",
            )
        else:
            require(
                args.role_pair_scope in (
                    "consumed_integration", "paper_heldout",
                    "paper_replication"),
                "bounded bearing alignment has an unsupported scope",
            )
            if args.role_pair_scope == "consumed_integration":
                require(
                    bool(args.role_pair_query_manifest),
                    "consumed bounded smoke requires a frozen query subset",
                )
            else:
                require(
                    not args.role_pair_query_manifest,
                    "held-out bounded formal must run its complete population",
                )
    require(
        args.revisit_controller == "navdp_mixed",
        "legacy evaluator controller label must remain neutral",
    )
    if arm == "learned_pi3x_spatial":
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.expected_pi3x_model_sha256)
            is not None,
            "learned arm requires a pinned Pi3X model hash",
        )
        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                args.expected_pi3x_proof_manifest_sha256,
            )
            is not None,
            "learned arm requires a pinned proof-manifest hash",
        )
    if args.role_pair_query_role != "all":
        require(
            args.role_pair_scope == "consumed_integration",
            "role filtering is permitted only for consumed development",
        )
    if args.role_pair_query_manifest:
        require(
            args.role_pair_scope == "consumed_integration",
            "query-manifest filtering is permitted only for consumed ablation",
        )
        require(
            args.role_pair_query_role == "all",
            "query manifest cannot be combined with role filtering",
        )
    if arm != "cec_portability":
        base.validate_revisit_adapter_configuration(
            mode=args.revisit_adapter,
            server_backend=args.server_backend,
            revisit_controller=args.revisit_controller,
            router_is_automatic_geometry=(
                args.hybrid_route in base.AUTO_HYBRID_ROUTES
            ),
            router_is_certified_relocalization=(
                args.hybrid_route in base.SCALE_FREE_RELOCALIZATION_ROUTES
            ),
        )
    return arm, backend


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_query_selection(
    benchmark_audit: dict,
) -> tuple[set[tuple[str, str, str]], dict | None]:
    """Load a deployment-evidence query subset without exposing roles."""

    if not args.role_pair_query_manifest:
        return set(), None
    path = Path(args.role_pair_query_manifest).resolve()
    require(path.is_file(), "role-pair query manifest is missing")
    payload = json.loads(path.read_text())
    require(
        payload.get("schema_version")
        == "cec_first_decision_accepted_population_v1_20260827",
        "role-pair query manifest schema changed",
    )
    require(
        payload.get("source_benchmark_manifest_sha256")
        == benchmark_audit["manifest_sha256"],
        "query manifest is bound to a different benchmark",
    )
    entries = payload.get("queries")
    require(isinstance(entries, list) and entries,
            "query manifest contains no selected queries")
    selected: set[tuple[str, str, str]] = set()
    for entry in entries:
        require(isinstance(entry, dict), "query manifest entry is invalid")
        require(not ({"analysis_role", "role", "query_role"} & set(entry)),
                "query manifest contains a role label")
        identity = (
            str(entry.get("scene")), str(entry.get("episode")),
            str(entry.get("query_id")),
        )
        require(all(value and value != "None" for value in identity),
                "query manifest identity is incomplete")
        require(identity not in selected, "duplicate query manifest identity")
        selected.add(identity)
    return selected, {
        "path": str(path),
        "sha256": sha256_path(path),
        "selected_total": len(selected),
    }


def load_episode(episode_dir: Path, expected_scene: str) -> dict:
    payload = json.loads((episode_dir / "role_pairs.json").read_text())
    require(payload["scene"] == expected_scene, "role-pair scene mismatch")
    source = Path(payload["online_a_episode"])
    require(source.is_dir(), "online-A source is missing")
    require(
        sha256_file(source / "receipt.json") == payload["online_a_receipt_sha256"],
        "online-A receipt hash changed",
    )
    require(
        sha256_file(source / "online_a_trace.json")
        == payload["online_a_trace_sha256"],
        "online-A trace hash changed",
    )
    receipt = json.loads((source / "receipt.json").read_text())
    trace = json.loads((source / "online_a_trace.json").read_text())
    require(trace.get("reached") is True, "frozen online A did not succeed")
    require(
        len(trace["poses"]) == int(payload["online_a_steps"]),
        "online-A trace length changed",
    )
    return {
        "benchmark": payload,
        "source": source,
        "receipt": receipt,
        "trace": trace,
    }


def replay_prefix(frozen: dict) -> tuple[dict, dict]:
    replay = replay_online_a(
        frozen,
        memory_step=base.srv_memory,
        navdp_replay_step=base.srv_navdp_memory_replay,
    )
    trace = frozen["trace"]
    leg = {
        "reached": True,
        "path_len": float(trace["path_len"]),
        "steps": int(trace["steps"]),
        "plans": trace["plans"],
        "memory_trace": replay["memory_trace"],
        "rollout_trace": trace["poses"],
        "end_pos": np.asarray(trace["end_position"], dtype=np.float64),
        "end_psi": float(trace["end_yaw"]),
    }
    return leg, replay


def router_counts(plans: list[dict]) -> dict:
    active = [
        plan.get("router_active")
        if plan.get("router_active") is not None
        else plan.get("cec_takeover")
        for plan in plans
    ]
    return {
        "router_active_plans": sum(value is True for value in active),
        "certificate_accept_plans": sum(
            (
                plan.get("certified_relocalization_accepted") is True
                or plan.get("cec_takeover") is True
            )
            for plan in plans
        ),
        "learned_pi3x_accept_plans": sum(
            plan.get("learned_pi3x_relocalization_accepted") is True
            for plan in plans
        ),
        "learned_pi3x_initial_inference_plans": sum(
            plan.get("learned_pi3x_initial_candidate_selection_cached")
            is False
            for plan in plans
        ),
        "adapter_takeover_plans": sum(
            (
                plan.get("revisit_adapter_takeover") is True
                or plan.get("cec_takeover") is True
            )
            for plan in plans
        ),
        "cec_fallback_plans": sum(
            plan.get("cec_takeover") is False for plan in plans
        ),
        "runtime_failure_plans": sum(
            (
                plan.get("certified_relocalization_reason")
                == "certificate_endpoint_failure"
            )
            or (plan.get("learned_pi3x_relocalization_ok") is False)
            or (plan.get("cec_reason") == "certificate_endpoint_failure")
            for plan in plans
        ),
    }


def depth_counts(plans: list[dict]) -> dict:
    """Summarize the NavDP observation-depth wire contract.

    The raw plan payload remains authoritative.  These scalar columns make a
    fail-closed factorial summary possible without trusting arm labels or
    re-running the policy.
    """

    receipts = [
        plan.get("monocular_depth_receipt")
        for plan in plans
        if isinstance(plan.get("monocular_depth_receipt"), dict)
    ]
    scale_hashes = {
        str(receipt["scale_receipt_sha256"])
        for receipt in receipts
        if receipt.get("scale_receipt_sha256")
    }
    return {
        "navdp_depth_source": args.navdp_depth_source,
        "metric_depth_sensor_consumed_any": int(any(
            plan.get("metric_depth_sensor_consumed") is True
            for plan in plans
        )),
        "monocular_receipt_plans": len(receipts),
        "monocular_active_receipt_plans": sum(
            receipt.get("scale_active") is True for receipt in receipts
        ),
        "monocular_scale_receipt_hashes": len(scale_hashes),
    }


def cec_latency_counts(plans: list[dict]) -> dict:
    result: dict[str, float | int | None] = {}
    for field in CEC_LATENCY_FIELDS:
        values = [
            float(plan[field]) for plan in plans
            if isinstance(plan.get(field), (int, float))
            and math.isfinite(float(plan[field]))
            and float(plan[field]) >= 0.0
        ]
        result[f"{field}_count"] = len(values)
        result[f"{field}_mean"] = mean_or_none(values)
        result[f"{field}_max"] = max(values) if values else None
    return result


def main() -> None:
    arm, policy_backend = validate_cli()
    if args.contract_dry_run:
        print(
            "[eval-role-pair] contract_dry_run OK: "
            f"arm={arm} backend={policy_backend} "
            f"route={args.hybrid_route} adapter={args.revisit_adapter} "
            f"depth={args.navdp_depth_source} role_visibility=none"
        )
        return
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output must be empty")

    scene_file = Path(args.scene).resolve()
    scene = base.SCENE_IDENTITY
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene, "episode root must be a scene directory")
    benchmark_root = scene_root.parent
    benchmark_audit = audit_benchmark(benchmark_root)
    require(benchmark_audit["ok"], "benchmark-wide audit failed")
    selected_queries, selection_receipt = load_query_selection(benchmark_audit)
    selected_seen: set[tuple[str, str, str]] = set()
    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "role_pairs.json").is_file()
    )
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[: args.episodes]
    require(bool(episode_dirs), "no role-pair episodes selected")
    selected_episode_names = {path.name for path in episode_dirs}

    if args.pinned_navmesh:
        pinned_navmesh = Path(args.pinned_navmesh).resolve()
        require(
            pinned_navmesh.is_file()
            and sha256_path(pinned_navmesh)
            == args.expected_pinned_navmesh_sha256,
            "pinned runtime navmesh receipt changed",
        )
        simulator = base.make_sim(
            str(scene_file), str(pinned_navmesh),
            agent_radius=args.agent_radius, recompute_navmesh=False,
        )
    else:
        simulator = base.make_sim(
            str(scene_file), "", agent_radius=args.agent_radius,
        )
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        query_index = 0
        for episode_dir in episode_dirs:
            frozen = load_episode(episode_dir, scene)
            receipt = frozen["receipt"]
            require(
                sha256_file(scene_file) == receipt["source_asset_sha256"],
                "scene asset hash differs from online-A materialization",
            )
            if (receipt.get("history_source")
                    == "controlled_causal_rgb_geodesic_survey"):
                camera_intrinsic = np.asarray(
                    receipt.get("camera_intrinsic"), dtype=np.float64
                )
                require(
                    camera_intrinsic.shape == (3, 3)
                    and np.isfinite(camera_intrinsic).all(),
                    "causal-survey camera intrinsic changed",
                )
                require(
                    int(receipt.get("episode_seed", -1))
                    == int(frozen["trace"].get("episode_seed", -2)) >= 0,
                    "causal-survey seed receipt changed",
                )
            else:
                source_parquet = (
                    Path(receipt["source_episode"])
                    / "data/chunk-000/episode_000000.parquet"
                )
                require(
                    sha256_file(source_parquet)
                    == receipt["source_parquet_sha256"],
                    "source parquet hash changed",
                )
                rows = pd.read_parquet(source_parquet)
                intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
                camera_intrinsic = np.stack(
                    [np.asarray(row, dtype=np.float64)
                     for row in intrinsic_raw]
                )
            camera_height = float(receipt["camera_height_m"])
            require(
                math.isclose(camera_height, float(base.CAM_H), abs_tol=1e-12),
                "online-A camera height differs from evaluator camera height",
            )
            episode_seed = int(frozen["trace"]["episode_seed"])
            for pair in frozen["benchmark"]["pairs"]:
                for stored_query in pair["queries"]:
                    if (args.role_pair_query_role != "all"
                            and stored_query["analysis_role"]
                            != args.role_pair_query_role):
                        continue
                    query_identity = (
                        scene, episode_dir.name, str(stored_query["query_id"]),
                    )
                    if selected_queries and query_identity not in selected_queries:
                        continue
                    if selected_queries:
                        selected_seen.add(query_identity)
                    # The runtime projection deliberately drops analysis_role,
                    # co-visibility and all construction diagnostics.  The role
                    # remains available below only for stratified scoring.
                    query = runtime_query(stored_query)
                    goal_rgb_path = episode_dir / query["goal_rgb"]
                    goal_depth_path = episode_dir / query["goal_depth"]
                    require(
                        sha256_file(goal_rgb_path) == query["goal_rgb_sha256"]
                        and sha256_file(goal_depth_path)
                        == query["goal_depth_sha256"],
                        "runtime query asset hash changed",
                    )
                    goal_jpg = goal_rgb_path.read_bytes()
                    goal_floor = np.asarray(query["floor_position"], dtype=np.float64)
                    goal_xz = goal_floor[[0, 2]]
                    goal_yaw = float(query["yaw_rad"])

                    base.srv_reset(
                        camera_height=camera_height,
                        seed=episode_seed,
                        episode_len=int(frozen["benchmark"]["online_a_steps"])
                        + int(args.max_steps),
                        camera_intrinsic=camera_intrinsic,
                        causal_history_sha256=frozen["benchmark"][
                            "online_a_trace_sha256"],
                    )
                    leg_a, replay = replay_prefix(frozen)
                    position = np.asarray(leg_a["end_pos"], dtype=np.float64)
                    yaw = float(leg_a["end_psi"])
                    ok, geo_distance, _path = base.geodesic(
                        pathfinder, position, goal_floor
                    )
                    require(ok and np.isfinite(geo_distance), "query geodesic failed")
                    require(
                        abs(
                            float(geo_distance)
                            - float(stored_query["geodesic_from_a_end_m"])
                        )
                        <= 0.05,
                        "stored/measured query geodesic mismatch",
                    )
                    leg = base.run_policy_leg(
                        simulator,
                        pathfinder,
                        position,
                        yaw,
                        goal_jpg,
                        goal_xz,
                        float(geo_distance),
                        None,
                        terminal_mode="off",
                        goal_yaw=goal_yaw,
                        camera_intrinsic=camera_intrinsic,
                        policy_backend=policy_backend,
                        episode_seed=episode_seed,
                        leg_index=1,
                    )
                    counts = router_counts(leg["plans"])
                    depth = depth_counts(leg["plans"])
                    latency = cec_latency_counts(leg["plans"])
                    query_trace_payload = base.leg1_trace_payload(
                        episode=episode_dir.name,
                        episode_seed=episode_seed,
                        goal_jpg=goal_jpg,
                        goal_source_episode=episode_dir.name,
                        source_scene=scene,
                        leg=leg,
                    )
                    metric = {
                        "scene": scene,
                        "episode": episode_dir.name,
                        "pair_id": pair["pair_id"],
                        "query_id": query["query_id"],
                        "analysis_role": stored_query["analysis_role"],
                        "arm": arm,
                        "seed": episode_seed,
                        "shared_A_frames": replay["online_frames"],
                        "shared_A_decision_frames": replay["decision_frames"],
                        "shared_A_hashes_ok": int(replay["all_rgb_hashes_verified"]),
                        "shared_A_diffusion_samples": replay[
                            "diffusion_samples_during_replay"
                        ],
                        "reached": int(bool(leg["reached"])),
                        "geodesic_m": float(geo_distance),
                        "path_len_m": float(leg["path_len"]),
                        "steps": int(leg["steps"]),
                        "final_goal_dist_m": float(leg["final_goal_dist_m"]),
                        "end_x_m": float(leg["end_pos"][0]),
                        "end_y_m": float(leg["end_pos"][1]),
                        "end_z_m": float(leg["end_pos"][2]),
                        "end_yaw_rad": float(leg["end_psi"]),
                        "termination_reason": leg.get("termination_reason"),
                        "cec_initial_bearing_alignment_mode": (
                            leg["cec_initial_bearing_alignment_mode"]),
                        "cec_initial_bearing_alignment_count": int(
                            leg["cec_initial_bearing_alignment_count"]),
                        "cec_initial_bearing_alignment_turn_deg": (
                            leg["cec_initial_bearing_alignment_turn_deg"]),
                        "cec_initial_bearing_alignment_action_count": int(
                            leg["cec_initial_bearing_alignment_action_count"]),
                        **counts,
                        **depth,
                        **latency,
                    }
                    metrics.append(metric)
                    plan_path = output / f"{episode_dir.name}_{query['query_id']}_plans.json"
                    plan_path.write_text(
                        json.dumps(
                            {
                                "schema_version": RESULT_SCHEMA,
                                "arm": arm,
                                "query_runtime_fields": sorted(query),
                                "analysis_role_not_forwarded": True,
                                "replay": replay,
                                "legA": leg_a["plans"],
                                "query_leg": leg["plans"],
                                "memory_traces": {
                                    "legA": leg_a["memory_trace"],
                                    "query": leg["memory_trace"],
                                },
                                "rollout_traces": {
                                    "legA": leg_a["rollout_trace"],
                                    "query": leg["rollout_trace"],
                                },
                                "cec_initial_bearing_alignment_trace": (
                                    leg[
                                        "cec_initial_bearing_alignment_trace"]),
                                # Canonical factual-prefix construction needs
                                # the exact terminal state returned by the
                                # controller, not an endpoint inferred from the
                                # last pre-action observation.  These fields are
                                # diagnostics for ordinary role-pair runs and a
                                # sealed source for later multi-leg replay.
                                "query_result": {
                                    "reached": bool(leg["reached"]),
                                    "path_len_m": float(leg["path_len"]),
                                    "path_len_at_reach_m": (
                                        None
                                        if leg.get("path_len_at_reach") is None
                                        else float(leg["path_len_at_reach"])
                                    ),
                                    "step_at_reach": leg.get("step_at_reach"),
                                    "steps": int(leg["steps"]),
                                    "termination_reason": leg.get(
                                        "termination_reason"
                                    ),
                                    "blocked_step_count": int(
                                        leg.get("blocked_step_count", 0)
                                    ),
                                    "final_goal_dist_m": float(
                                        leg["final_goal_dist_m"]
                                    ),
                                    "end_position": [
                                        float(value) for value in leg["end_pos"]
                                    ],
                                    "end_yaw_rad": float(leg["end_psi"]),
                                },
                                "query_trace_payload": query_trace_payload,
                            },
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    query_index += 1
                    print(
                        f"[{scene}/{episode_dir.name}/{pair['pair_id']}/"
                        f"{stored_query['analysis_role']}/{arm}] "
                        f"success={metric['reached']} steps={metric['steps']}"
                    )
                    with (output / "metric.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                        writer.writeheader()
                        writer.writerows(metrics)

        roles = {
            role: [row for row in metrics if row["analysis_role"] == role]
            for role in ("novel", "revisit")
        }
        if selected_queries:
            expected_for_invocation = {
                identity for identity in selected_queries
                if identity[0] == scene
                and identity[1] in selected_episode_names
            }
            require(bool(expected_for_invocation),
                    "query manifest selected nothing for this invocation")
            require(selected_seen == expected_for_invocation,
                    "selected query manifest was not realized exactly")
        scope_map = {
            "consumed_integration": (
                "consumed-scene integration unless externally promoted"
            ),
            "paper_heldout": "paper held-out role-pair evaluation",
            "paper_replication": (
                "paper reused-scene/history new-query replication"
            ),
            "replica_cross_dataset": (
                "Replica cross-dataset role-pair evaluation"
            ),
            "table3_length": (
                "HM3D causal-RGB Novel/Revisit evaluation by geodesic length"
            ),
        }
        summary = {
            "schema_version": RESULT_SCHEMA,
            "scope": scope_map[args.role_pair_scope],
            "role_pair_scope": args.role_pair_scope,
            "role_pair_query_role": args.role_pair_query_role,
            "role_pair_query_selection": selection_receipt,
            "arm": arm,
            "server_backend": args.server_backend,
            "hybrid_route": args.hybrid_route,
            "revisit_adapter": args.revisit_adapter,
            "revisit_controller": args.revisit_controller,
            "benchmark_manifest_sha256": benchmark_audit["manifest_sha256"],
            "scene": scene,
            "queries": len(metrics),
            "role_counts": {role: len(rows) for role, rows in roles.items()},
            "SR": mean_or_none([row["reached"] for row in metrics]),
            "SR_by_role": {
                role: mean_or_none([row["reached"] for row in rows])
                for role, rows in roles.items()
            },
            "router_active_episodes_by_role": {
                role: sum(row["router_active_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "certificate_accept_episodes_by_role": {
                role: sum(row["certificate_accept_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "learned_pi3x_accept_episodes_by_role": {
                role: sum(
                    row["learned_pi3x_accept_plans"] > 0 for row in rows
                )
                for role, rows in roles.items()
            },
            "adapter_takeover_episodes_by_role": {
                role: sum(row["adapter_takeover_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "cec_fallback_plans_by_role": {
                role: sum(row["cec_fallback_plans"] for row in rows)
                for role, rows in roles.items()
            },
            "runtime_failure_plans": sum(
                row["runtime_failure_plans"] for row in metrics
            ),
            "navdp_depth_source": args.navdp_depth_source,
            "metric_depth_sensor_consumed_episodes": sum(
                row["metric_depth_sensor_consumed_any"] > 0
                for row in metrics
            ),
            "monocular_receipt_plans": sum(
                row["monocular_receipt_plans"] for row in metrics
            ),
            "monocular_active_receipt_plans": sum(
                row["monocular_active_receipt_plans"] for row in metrics
            ),
            "cec_latency_ms": {
                field: {
                    "count": sum(
                        row[f"{field}_count"] for row in metrics
                    ),
                    "mean": (
                        sum(
                            row[f"{field}_mean"]
                            * row[f"{field}_count"]
                            for row in metrics
                            if row[f"{field}_mean"] is not None
                        )
                        / sum(row[f"{field}_count"] for row in metrics)
                        if sum(row[f"{field}_count"] for row in metrics)
                        else None
                    ),
                    "max": max(
                        (
                            row[f"{field}_max"] for row in metrics
                            if row[f"{field}_max"] is not None
                        ),
                        default=None,
                    ),
                }
                for field in CEC_LATENCY_FIELDS
            },
            "shared_A_all_hashes_ok": all(
                row["shared_A_hashes_ok"] for row in metrics
            ),
            "shared_A_total_diffusion_samples": sum(
                row["shared_A_diffusion_samples"] for row in metrics
            ),
            "runtime_role_visibility": "none",
            "deterministic_plan_seeds": bool(args.deterministic_plan_seeds),
            "max_steps": int(args.max_steps),
            "exec_horizon": int(args.exec_horizon),
            "cec_initial_bearing_alignment_mode": (
                args.cec_initial_bearing_alignment),
            "cec_initial_bearing_alignment_episodes": sum(
                row["cec_initial_bearing_alignment_count"] > 0
                for row in metrics
            ),
            "cec_initial_bearing_alignment_actions": sum(
                row["cec_initial_bearing_alignment_action_count"]
                for row in metrics
            ),
            "certified_cdec_rescue": args.certified_cdec_rescue,
            "certified_stagnation_graph": args.certified_stagnation_graph,
            "memnav_server_info": dict(base.MEMNAV_SERVER_INFO),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print("[shared-online-role-pair] done", summary)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
