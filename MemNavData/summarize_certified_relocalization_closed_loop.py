#!/usr/bin/env python3
"""Fail-closed four-arm summary for certified relocalization closed loop."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.build_revisit_fresh_manifest import sha256_file
from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    exact_sign_p,
    load_arm,
    paired_summary,
    percentile,
    require,
    truth,
)


ARMS = (
    "certified_relocalization",
    "known_revisit_direct",
    "geometry_router",
    "native",
)
WILLIAMS_ORDERS = (
    ("certified_relocalization", "known_revisit_direct", "native",
     "geometry_router"),
    ("known_revisit_direct", "geometry_router", "certified_relocalization",
     "native"),
    ("geometry_router", "native", "known_revisit_direct",
     "certified_relocalization"),
    ("native", "certified_relocalization", "geometry_router",
     "known_revisit_direct"),
)
BOOTSTRAP_CHUNK = 10_000


def _number(value: Any, *, integer: bool = False) -> float | int:
    require(value not in (None, ""), "required numeric receipt is missing")
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite numeric receipt: {value!r}")
    return int(parsed) if integer else parsed


def validate_summary(path: Path, arm: str, episode_count: int) -> None:
    require(path.is_file(), f"missing arm summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": episode_count,
        "max_steps": 500,
        "exec_horizon": 8,
        "leg1_mode": "shared_trace",
        "write_leg1_trace": False,
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
    }
    if arm == "certified_relocalization":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "certified_relocalization",
            "revisit_controller": "navdp_mixed",
            "revisit_adapter": "verified_bearing_v1",
            "revisit_adapter_fixed_radius_m": 2.5,
            "graph_subgoal_spacing_m": 0.0,
            "graph_subgoal_arrival_m": 0.6,
        })
    elif arm == "known_revisit_direct":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "revisit_controller": "navdp_mixed",
            "revisit_adapter": "legacy_metric",
            "graph_subgoal_spacing_m": 0.0,
            "graph_subgoal_arrival_m": 0.6,
        })
    elif arm == "geometry_router":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "memory_geometry",
            "revisit_controller": "navdp_mixed",
            "revisit_adapter": "legacy_metric",
            "graph_subgoal_spacing_m": 0.0,
            "graph_subgoal_arrival_m": 0.6,
            "router_visual_floor": 0.88,
            "router_min_matches": 20,
            "router_min_inliers": 12,
            "router_min_inlier_ratio": 0.5,
            "router_confirm_plans": 2,
            "router_verify_top_k": 8,
        })
    else:
        expected.update({
            "server_backend": "navdp",
            "hybrid_route": "phase",
            "revisit_adapter": "legacy_metric",
            "graph_subgoal_spacing_m": None,
            "graph_subgoal_arrival_m": None,
        })
    for field, wanted in expected.items():
        require(summary.get(field) == wanted,
                f"{path}: {field}={summary.get(field)!r}, expected {wanted!r}")

    if arm == "certified_relocalization":
        server = summary.get("certified_relocalization_server")
        contract = (server or {}).get("runtime_contract")
        required_contract = {
            "schema_version": 3,
            "geometry_certificate_version": 2,
            "candidate_top_k": 8,
            "candidate_min_gap": 4,
            "minimum_anchor": 8,
            "candidate_lifecycle": "frozen_at_first_goal_query",
            "empty_candidate_semantics": "cached_native_abstention",
            "output": "scale_free_relative_bearing",
            "pointgoal_units": "lingbot_raw_direction_only",
            "metric_distance_certified": False,
            "controller_adapter": "verified_bearing_v1_fixed_2.5m",
            "fallback": "native_imagegoal",
        }
        require(isinstance(server, dict) and server.get("enabled") is True,
                f"{path}: certified server is not enabled")
        require(isinstance(contract, dict), f"{path}: runtime contract missing")
        for field, wanted in required_contract.items():
            require(contract.get(field) == wanted,
                    f"{path}: contract {field} changed")


def validate_certified_episode(
    arm_root: Path,
    metric: dict[str, str],
    *,
    reached_a: bool,
) -> dict[str, Any]:
    episode = metric["episode"]
    plans_path = arm_root / f"{episode}_plans.json"
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    plans = payload.get("legB")
    require(isinstance(plans, list), f"{plans_path}: legB plans missing")

    requests = int(_number(
        metric["certified_relocalization_request_count"], integer=True))
    uncached = int(_number(
        metric["certified_relocalization_uncached_count"], integer=True))
    accepts = int(_number(
        metric["certified_relocalization_accept_count"], integer=True))
    failures = int(_number(
        metric["certified_relocalization_runtime_failure_count"], integer=True))
    takeovers = int(_number(
        metric["revisit_adapter_takeover_plan_count"], integer=True))
    abstains = int(_number(
        metric["revisit_adapter_abstain_plan_count"], integer=True))
    require(requests == len(plans),
            f"{plans_path}: request count differs from B plan count")
    require(takeovers + abstains == len(plans),
            f"{plans_path}: adapter decisions do not cover B plans")
    if reached_a:
        require(requests > 0, f"{plans_path}: A-success episode never planned B")
        require(uncached == 1,
                f"{plans_path}: certificate must execute exactly once per goal")
        require(failures == 0,
                f"{plans_path}: certificate transport/runtime failure")
    else:
        require(requests == 0 and uncached == 0 and accepts == 0,
                f"{plans_path}: B ran despite shared A failure")

    accepted_plans = 0
    selected_dino_ranks = []
    uncached_ms = []
    pnp_inliers = []
    for plan in plans:
        require(plan.get("certified_relocalization_metric_scale") is None,
                f"{plans_path}: metric scale leaked into certified runtime")
        require(plan.get("revisit_adapter_mode") == "verified_bearing_v1",
                f"{plans_path}: wrong adapter mode")
        require(plan.get("revisit_adapter_source") ==
                "lightglue_lingbot_pnp_v2_scale_free",
                f"{plans_path}: wrong bearing source")
        accepted = plan.get("certified_relocalization_accepted") is True
        takeover = plan.get("revisit_adapter_takeover") is True
        require(accepted == takeover,
                f"{plans_path}: certificate/takeover mismatch")
        if accepted:
            accepted_plans += 1
            require(plan.get("certified_relocalization_ok") is True,
                    f"{plans_path}: accepted request is not ok")
            require(plan.get("certified_relocalization_pointgoal_units") ==
                    "lingbot_raw_direction_only",
                    f"{plans_path}: scale-free unit receipt missing")
            require(plan.get("memory_unbounded_pointgoal_units") ==
                    "lingbot_raw_direction_only",
                    f"{plans_path}: adapter unit receipt missing")
            require(plan.get("memory_unbounded_pointgoal_distance_m") is None,
                    f"{plans_path}: raw arbitrary norm was mislabeled metres")
            require(math.isclose(
                float(plan["memory_controller_pointgoal_distance_m"]),
                2.5, abs_tol=1e-9),
                f"{plans_path}: fixed controller radius changed")
            require(plan.get("pose_controller") == "navdp_image_point_mix",
                    f"{plans_path}: accepted bearing did not use mixed NavDP")
        else:
            require(plan.get("revisit_adapter_controller_contract") ==
                    "native_imagegoal",
                    f"{plans_path}: rejection did not fail closed")
            require(plan.get("pose_controller") == "navdp_image_router",
                    f"{plans_path}: rejection did not execute native NavDP")
        rank = plan.get("router_selected_candidate_dino_rank")
        if rank not in (None, "") and plan.get(
                "certified_relocalization_cached") is False:
            selected_dino_ranks.append(int(rank))
        if plan.get("certified_relocalization_cached") is False:
            latency = plan.get("certified_relocalization_uncached_ms")
            if latency not in (None, ""):
                uncached_ms.append(float(latency))
            pnp = plan.get("certified_relocalization_pnp")
            if isinstance(pnp, dict) and pnp.get("inliers") is not None:
                pnp_inliers.append(int(pnp["inliers"]))
    require(accepted_plans == accepts,
            f"{plans_path}: accepted-plan count receipt mismatch")
    require(accepted_plans == takeovers,
            f"{plans_path}: takeover count receipt mismatch")
    return {
        "requests": requests,
        "accepted_plans": accepted_plans,
        "takeover_episode": accepted_plans > 0,
        "fallback_episode": reached_a and accepted_plans == 0,
        "selected_dino_ranks": selected_dino_ranks,
        "uncached_ms": uncached_ms,
        "pnp_inliers": pnp_inliers,
    }


def conditional_paired(
    left_name: str,
    right_name: str,
    left: dict[tuple[str, str], dict],
    right: dict[tuple[str, str], dict],
    expected: set[tuple[str, str]],
) -> dict[str, Any]:
    eligible = []
    gains = []
    losses = []
    both = neither = 0
    for key in sorted(expected):
        require(left[key]["reached_a"] == right[key]["reached_a"],
                f"shared Goal-A outcome differs: {key}")
        if not left[key]["reached_a"]:
            continue
        eligible.append(key)
        lval = bool(left[key]["reached_b"])
        rval = bool(right[key]["reached_b"])
        if lval and rval:
            both += 1
        elif rval:
            gains.append(key)
        elif lval:
            losses.append(key)
        else:
            neither += 1
    discordant = len(gains) + len(losses)
    return {
        "left": left_name,
        "right": right_name,
        "eligible_shared_novel_success": len(eligible),
        "outcomes": {
            "both_revisit_success": both,
            "left_only_revisit_success": len(losses),
            "right_only_revisit_success": len(gains),
            "neither_revisit_success": neither,
        },
        "risk_difference_right_minus_left": (
            (len(gains) - len(losses)) / len(eligible) if eligible else None),
        "mcnemar_exact_two_sided_p": exact_sign_p(len(gains), discordant),
        "gains": [
            {"scene": scene, "episode": episode}
            for scene, episode in gains],
        "losses": [
            {"scene": scene, "episode": episode}
            for scene, episode in losses],
    }


def cluster_interval(
    scenes: list[str],
    episode_ids: dict[str, list[str]],
    left: dict[tuple[str, str], dict],
    right: dict[tuple[str, str], dict],
    *,
    conditional: bool,
    seed: int,
    resamples: int,
) -> list[float]:
    numerators = []
    denominators = []
    for scene in scenes:
        numerator = 0.0
        denominator = 0
        for episode in episode_ids[scene]:
            key = (scene, episode)
            if conditional and not left[key]["reached_a"]:
                continue
            denominator += 1
            target = "reached_b" if conditional else "joint"
            numerator += float(right[key][target]) - float(left[key][target])
        numerators.append(numerator)
        denominators.append(denominator)
    nums = np.asarray(numerators, dtype=float)
    dens = np.asarray(denominators, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, resamples, BOOTSTRAP_CHUNK):
        count = min(BOOTSTRAP_CHUNK, resamples - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        sampled_den = dens[indices].sum(axis=1)
        valid = sampled_den > 0
        samples.append(
            nums[indices].sum(axis=1)[valid] / sampled_den[valid])
    values = np.concatenate(samples)
    require(values.size > 0, "cluster bootstrap has no eligible sample")
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def decision_branch(primary: dict[str, Any]) -> str:
    delta = float(primary["joint_sr_delta_right_minus_left"])
    pvalue = float(primary["mcnemar_exact_two_sided_p"])
    low, high = primary["scene_cluster_bootstrap_risk_difference_95"]
    if delta > 0.0 and pvalue < 0.05 and low > 0.0:
        return (
            "certified_router_has_closed_loop_value_"
            "seek_fresh_scene_open_set_confirmation")
    if delta < 0.0 and pvalue < 0.05 and high < 0.0:
        return "reject_certified_router_retain_known_role_system"
    return "inconclusive_do_not_retune_on_consumed_pool"


def summarize(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["audit"]["status"] == "ok", "data manifest audit failed")
    require(manifest["data_role_guards"]["blind_allowed"] is False,
            "manifest does not prohibit blind data")
    scenes = manifest["scenes"]
    episode_ids = {
        scene: [row["episode"] for row in manifest["episodes"][scene]]
        for scene in scenes
    }
    expected = {
        (scene, episode)
        for scene in scenes for episode in episode_ids[scene]
    }
    require(len(scenes) == 20 and len(expected) == 160,
            "formal closed loop requires 20 scenes / 160 episodes")
    require(all(len(values) == 8 for values in episode_ids.values()),
            "formal closed loop requires eight episodes per scene")

    manifest_sha = sha256_file(manifest_path)
    rows = {arm: {} for arm in ARMS}
    arm_orders = {}
    certified_audit = []
    for index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        contract_path = scene_root / "scene_contract.json"
        require(contract_path.is_file(), f"missing scene contract: {scene}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_order = list(WILLIAMS_ORDERS[index % len(WILLIAMS_ORDERS)])
        require(contract.get("schema_version") ==
                "certified_relocalization_closed_loop_v1",
                f"scene contract schema changed: {scene}")
        require(contract.get("scene") == scene, f"scene identity changed: {scene}")
        require(contract.get("scene_index") == index,
                f"scene index changed: {scene}")
        require(contract.get("manifest_sha256") == manifest_sha,
                f"manifest receipt mismatch: {scene}")
        require(contract.get("arm_order") == expected_order,
                f"Williams arm order changed: {scene}")
        arm_orders[scene] = expected_order

        trace_root = scene_root / "trace_source"
        trace_summary = json.loads(
            (trace_root / "summary.json").read_text(encoding="utf-8"))
        for field, wanted in {
            "episodes": 8,
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "leg1_mode": "policy",
            "stop_after_leg1": True,
            "write_leg1_trace": True,
            "deterministic_plan_seeds": True,
        }.items():
            require(trace_summary.get(field) == wanted,
                    f"trace source {scene}: {field} changed")
        for episode in episode_ids[scene]:
            require((trace_root / f"{episode}_leg1_trace.json").is_file(),
                    f"missing trace: {scene}/{episode}")

        for arm in ARMS:
            arm_root = scene_root / arm
            validate_summary(arm_root / "summary.json", arm, 8)
            loaded = load_arm(scene_root, arm, scene)
            rows[arm].update(loaded)
            if arm == "certified_relocalization":
                with (arm_root / "metric.csv").open(newline="") as handle:
                    metrics = list(csv.DictReader(handle))
                require(len(metrics) == 8,
                        f"{scene}: certified metric row count changed")
                for metric in metrics:
                    key = (scene, metric["episode"])
                    certified_audit.append(validate_certified_episode(
                        arm_root, metric, reached_a=loaded[key]["reached_a"]))

    for arm in ARMS:
        require(set(rows[arm]) == expected,
                f"{arm}: result keys differ from manifest")
    for scene, episode in sorted(expected):
        trace = (
            run_root / "scenes" / f"{scenes.index(scene):02d}_{scene}" /
            "trace_source" / f"{episode}_leg1_trace.json")
        trace_sha = sha256_file(trace)
        for arm in ARMS:
            require(rows[arm][(scene, episode)]["leg1_trace_sha256"] == trace_sha,
                    f"{arm}: shared trace SHA differs: {scene}/{episode}")

    analysis = manifest["analysis"]
    seed = int(analysis["cluster_bootstrap_seed"])
    resamples = int(analysis["cluster_bootstrap_resamples"])
    specs = {
        "certified_minus_native": ("native", "certified_relocalization"),
        "certified_minus_geometry": (
            "geometry_router", "certified_relocalization"),
        "certified_minus_known_revisit_direct": (
            "known_revisit_direct", "certified_relocalization"),
        "direct_minus_native": ("native", "known_revisit_direct"),
        "geometry_minus_native": ("native", "geometry_router"),
    }
    contrasts = {}
    for offset, (name, (left_name, right_name)) in enumerate(specs.items()):
        paired = paired_summary(
            left_name, right_name, rows[left_name], rows[right_name], expected)
        conditional = conditional_paired(
            left_name, right_name, rows[left_name], rows[right_name], expected)
        paired["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
            scenes, episode_ids, rows[left_name], rows[right_name],
            conditional=False, seed=seed + 2 * offset, resamples=resamples)
        conditional[
            "scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
                scenes, episode_ids, rows[left_name], rows[right_name],
                conditional=True, seed=seed + 2 * offset + 1,
                resamples=resamples)
        contrasts[name] = {"joint": paired, "conditional_b": conditional}

    latencies = [
        value for audit in certified_audit for value in audit["uncached_ms"]]
    ranks = [
        value for audit in certified_audit
        for value in audit["selected_dino_ranks"]]
    inliers = [
        value for audit in certified_audit for value in audit["pnp_inliers"]]
    primary = contrasts["certified_minus_native"]["joint"]
    branch = decision_branch(primary)
    return {
        "scope": (
            "frozen certified-router evaluation on the consumed fresh-episode "
            "20-scene replication pool; not fresh-scene or final confirmation"),
        "question": (
            "Does a scale-free LightGlue+LingBot-PnP history certificate "
            "improve closed-loop SR over native NavDP, and how much of the "
            "known-Revisit direct upper bound does it retain?"),
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "manifest_sha256": manifest_sha,
            "source_bundle_receipt_sha256": sha256_file(
                run_root / "source_bundle.sha256"),
            "training_scene_overlap": manifest["audit"][
                "training_scene_overlap"],
            "shared_native_goal_a_trace": True,
            "balanced_williams_arm_order": True,
            "runtime_metric_scale_used": False,
            "development_read": False,
            "blind_read": False,
        },
        "arm_order_by_scene": arm_orders,
        "arms": {
            arm: arm_summary(
                [rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "certified_runtime": {
            "a_success_episodes": sum(
                rows["certified_relocalization"][key]["reached_a"]
                for key in expected),
            "takeover_episodes": sum(
                audit["takeover_episode"] for audit in certified_audit),
            "fallback_episodes_after_a_success": sum(
                audit["fallback_episode"] for audit in certified_audit),
            "accepted_plans": sum(
                audit["accepted_plans"] for audit in certified_audit),
            "uncached_calls": len(latencies),
            "uncached_latency_ms": {
                "mean": float(np.mean(latencies)) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies, default=None),
            },
            "selected_dino_rank": {
                "p50": percentile(ranks, 0.50),
                "p95": percentile(ranks, 0.95),
                "max": max(ranks, default=None),
            },
            "pnp_inliers": {
                "p50": percentile(inliers, 0.50),
                "p05": percentile(inliers, 0.05),
                "min": min(inliers, default=None),
            },
        },
        "contrasts": contrasts,
        "decision": {
            "primary_contrast": "certified_minus_native",
            "branch": branch,
            "authorize_threshold_or_radius_retuning_on_this_pool": False,
            "authorize_blind_eval": False,
            "authorize_paper_claim": False,
            "episodes_consumed_for_future_tuning": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    report = summarize(args.manifest, args.run_root)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "arms": report["arms"],
        "certified_runtime": report["certified_runtime"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
