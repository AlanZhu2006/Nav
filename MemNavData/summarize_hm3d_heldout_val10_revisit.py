#!/usr/bin/env python3
"""Fail-closed summary for the frozen HM3D held-out val10 Revisit transfer study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MANIFEST_SCHEMA_V1 = "hm3d_heldout_val10_causal_revisit_manifest_v1_20260816"
MANIFEST_SCHEMA_V2 = "hm3d_heldout_val10_causal_revisit_manifest_v2_20260816"
SCENE_SCHEMA = "hm3d_heldout_val10_revisit_scene_contract_v1_20260816"
RUNTIME_REPAIR_SCENE_SCHEMA = (
    "hm3d_heldout_runtime_repair_scene_v1_20260816")
SCENE_SCHEMAS = {SCENE_SCHEMA, RUNTIME_REPAIR_SCENE_SCHEMA}
REPORT_SCHEMA_V1 = "hm3d_heldout_val10_revisit_summary_v1_20260816"
REPORT_SCHEMA_V2 = "hm3d_heldout_val10_revisit_summary_v2_20260816"
ARMS = (
    "native",
    "raw_fixed_oracle_role",
    "geometry_router",
    "certified_relocalization",
)
WILLIAMS_ORDERS = (
    ("certified_relocalization", "raw_fixed_oracle_role", "native",
     "geometry_router"),
    ("raw_fixed_oracle_role", "geometry_router",
     "certified_relocalization", "native"),
    ("geometry_router", "native", "raw_fixed_oracle_role",
     "certified_relocalization"),
    ("native", "certified_relocalization", "geometry_router",
     "raw_fixed_oracle_role"),
)
BOOTSTRAP_CHUNK = 10_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_scene_contract_schema(contract: dict[str, Any], scene: str) -> str:
    """Accept only the frozen legacy or runtime-repair contract lineage."""
    schema = contract.get("schema_version")
    require(schema in SCENE_SCHEMAS,
            f"{scene}: scene contract schema_version changed")
    marker = contract.get("runtime_repair_method_change")
    if schema == RUNTIME_REPAIR_SCENE_SCHEMA:
        require(marker is False,
                f"{scene}: runtime repair method-change guard changed")
    else:
        require(marker in {None, False},
                f"{scene}: unexpected method-change guard")
    return str(schema)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str) and value.strip().lower() in {
            "true", "yes", "y"}:
        return True
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite truth value: {value!r}")
    return parsed > 0.5


def integer(value: Any) -> int:
    require(value not in (None, ""), "missing integer receipt")
    parsed = float(value)
    require(math.isfinite(parsed) and parsed.is_integer(),
            f"invalid integer receipt: {value!r}")
    return int(parsed)


def finite(value: Any) -> float:
    require(value not in (None, ""), "missing numeric receipt")
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite receipt: {value!r}")
    return parsed


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def read_metrics(path: Path, expected: list[str]) -> dict[str, dict[str, str]]:
    require(path.is_file(), f"missing metric file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require([row.get("episode") for row in rows] == expected,
            f"episode identity/order changed: {path}")
    return {str(row["episode"]): row for row in rows}


def validate_common_summary(
    summary: dict[str, Any],
    *,
    path: Path,
    episode_count: int,
    base_seed: int,
) -> None:
    expected = {
        "episodes": episode_count,
        "max_steps": 500,
        "exec_horizon": 8,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
        "deterministic_plan_seeds": True,
        "base_seed": base_seed,
        "leg1_goal_source": "own",
        "certified_cdec_rescue": "off",
        "certified_stagnation_graph": "off",
        "retrieval_override": "off",
    }
    for field, wanted in expected.items():
        require(summary.get(field) == wanted,
                f"{path}: {field}={summary.get(field)!r}, expected {wanted!r}")


def validate_trace_summary(
    path: Path, *, episode_count: int, base_seed: int,
) -> None:
    require(path.is_file(), f"missing trace summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_common_summary(
        summary, path=path, episode_count=episode_count, base_seed=base_seed)
    expected = {
        "server_backend": "hybrid_pose",
        "hybrid_route": "phase",
        "leg1_mode": "policy",
        "stop_after_leg1": True,
        "write_leg1_trace": True,
        "revisit_controller": "navdp_mixed",
        "revisit_adapter": "legacy_metric",
    }
    for field, wanted in expected.items():
        require(summary.get(field) == wanted,
                f"{path}: {field} changed")


def validate_arm_summary(
    path: Path,
    arm: str,
    *,
    episode_count: int,
    base_seed: int,
) -> dict[str, Any]:
    require(path.is_file(), f"missing arm summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_common_summary(
        summary, path=path, episode_count=episode_count, base_seed=base_seed)
    expected: dict[str, Any] = {
        "leg1_mode": "shared_trace",
        "stop_after_leg1": False,
        "write_leg1_trace": False,
        "revisit_controller": "navdp_mixed",
    }
    if arm == "native":
        expected.update({
            "server_backend": "navdp",
            "hybrid_route": "phase",
            "revisit_adapter": "legacy_metric",
            "revisit_adapter_fixed_radius_m": None,
        })
    elif arm == "raw_fixed_oracle_role":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "revisit_adapter": "raw_fixed_bearing_v1",
            "revisit_adapter_fixed_radius_m": None,
        })
    elif arm == "geometry_router":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "memory_geometry",
            "revisit_adapter": "legacy_metric",
            "revisit_adapter_fixed_radius_m": None,
            "router_visual_floor": 0.88,
            "router_min_matches": 20,
            "router_min_inliers": 12,
            "router_min_inlier_ratio": 0.5,
            "router_confirm_plans": 2,
            "router_verify_top_k": 8,
        })
    elif arm == "certified_relocalization":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "certified_relocalization",
            "revisit_adapter": "verified_bearing_v1",
            "revisit_adapter_fixed_radius_m": 2.5,
        })
    else:  # pragma: no cover - guarded by the frozen constant
        raise RuntimeError(f"unknown arm: {arm}")
    for field, wanted in expected.items():
        require(summary.get(field) == wanted,
                f"{path}: {field}={summary.get(field)!r}, expected {wanted!r}")
    if arm == "certified_relocalization":
        server = summary.get("certified_relocalization_server")
        require(isinstance(server, dict) and server.get("enabled") is True,
                f"{path}: certificate server not enabled")
        contract = server.get("runtime_contract")
        require(isinstance(contract, dict), f"{path}: runtime contract missing")
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
        for field, wanted in required_contract.items():
            require(contract.get(field) == wanted,
                    f"{path}: runtime contract {field} changed")
    return summary


def validate_certified_episode(
    arm_root: Path,
    metric: dict[str, str],
    *,
    reached_a: bool,
) -> dict[str, Any]:
    episode = str(metric["episode"])
    plans_path = arm_root / f"{episode}_plans.json"
    require(plans_path.is_file(), f"missing plan receipt: {plans_path}")
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    plans = payload.get("legB")
    require(isinstance(plans, list), f"{plans_path}: legB plans missing")

    requests = integer(metric["certified_relocalization_request_count"])
    uncached = integer(metric["certified_relocalization_uncached_count"])
    accepts = integer(metric["certified_relocalization_accept_count"])
    failures = integer(metric["certified_relocalization_runtime_failure_count"])
    takeovers = integer(metric["revisit_adapter_takeover_plan_count"])
    abstains = integer(metric["revisit_adapter_abstain_plan_count"])
    require(requests == len(plans),
            f"{plans_path}: request count differs from plan count")
    require(takeovers + abstains == len(plans),
            f"{plans_path}: adapter decisions do not cover plans")
    if reached_a:
        require(requests > 0, f"{plans_path}: Goal-B was never planned")
        require(uncached == 1,
                f"{plans_path}: certificate must execute once per goal")
        require(failures == 0, f"{plans_path}: certificate runtime failure")
    else:
        require(requests == uncached == accepts == failures == 0,
                f"{plans_path}: certificate ran after Goal-A failure")

    accepted_plans = 0
    uncached_ms: list[float] = []
    pnp_inliers: list[int] = []
    dino_ranks: list[int] = []
    for plan in plans:
        require(plan.get("certified_relocalization_metric_scale") is None,
                f"{plans_path}: metric scale leaked")
        require(plan.get("revisit_adapter_mode") == "verified_bearing_v1",
                f"{plans_path}: adapter mode changed")
        accepted = plan.get("certified_relocalization_accepted") is True
        takeover = plan.get("revisit_adapter_takeover") is True
        require(accepted == takeover,
                f"{plans_path}: certificate/takeover mismatch")
        if accepted:
            accepted_plans += 1
            require(plan.get("certified_relocalization_ok") is True,
                    f"{plans_path}: accepted proposal is not valid")
            require(plan.get("certified_relocalization_pointgoal_units") ==
                    "lingbot_raw_direction_only",
                    f"{plans_path}: scale-free receipt missing")
            require(plan.get("memory_unbounded_pointgoal_distance_m") is None,
                    f"{plans_path}: arbitrary magnitude mislabeled as metres")
            require(math.isclose(finite(
                plan["memory_controller_pointgoal_distance_m"]), 2.5,
                abs_tol=1e-9), f"{plans_path}: controller radius changed")
            require(plan.get("pose_controller") == "navdp_image_point_mix",
                    f"{plans_path}: accepted bearing did not use mixed NavDP")
        else:
            require(plan.get("revisit_adapter_controller_contract") ==
                    "native_imagegoal",
                    f"{plans_path}: rejection did not fail closed")
            require(plan.get("pose_controller") == "navdp_image_router",
                    f"{plans_path}: rejection did not use native NavDP")
        if plan.get("certified_relocalization_cached") is False:
            value = plan.get("certified_relocalization_uncached_ms")
            if value not in (None, ""):
                uncached_ms.append(finite(value))
            pnp = plan.get("certified_relocalization_pnp")
            if isinstance(pnp, dict) and pnp.get("inliers") is not None:
                pnp_inliers.append(integer(pnp["inliers"]))
            rank = plan.get("router_selected_candidate_dino_rank")
            if rank not in (None, ""):
                dino_ranks.append(integer(rank))
    require(accepted_plans == accepts == takeovers,
            f"{plans_path}: accepted/takeover receipts disagree")
    if reached_a:
        require(accepted_plans in {0, len(plans)},
                f"{plans_path}: atomic cached decision changed within goal")
    return {
        "requests": requests,
        "accepted_plans": accepted_plans,
        "abstained_plans": abstains,
        "runtime_failures": failures,
        "takeover_episode": reached_a and accepted_plans > 0,
        "fallback_episode": reached_a and accepted_plans == 0,
        "uncached_ms": uncached_ms,
        "pnp_inliers": pnp_inliers,
        "dino_ranks": dino_ranks,
    }


def paired(
    left_name: str,
    right_name: str,
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    keys: Iterable[tuple[str, str]],
    *,
    conditional_b: bool,
) -> dict[str, Any]:
    eligible: list[tuple[str, str]] = []
    gains: list[tuple[str, str]] = []
    losses: list[tuple[str, str]] = []
    both = neither = 0
    target = "reached_b" if conditional_b else "joint"
    for key in sorted(keys):
        require(left[key]["reached_a"] == right[key]["reached_a"],
                f"Goal-A pairing differs for {key}")
        if conditional_b and not left[key]["reached_a"]:
            continue
        eligible.append(key)
        lval = bool(left[key][target])
        rval = bool(right[key][target])
        if lval and rval:
            both += 1
        elif rval:
            gains.append(key)
        elif lval:
            losses.append(key)
        else:
            neither += 1
    count = len(eligible)
    return {
        "left": left_name,
        "right": right_name,
        "estimand": ("goal_b_success_given_shared_goal_a_success"
                      if conditional_b else "joint_goal_b_success"),
        "eligible": count,
        "left_successes": sum(bool(left[key][target]) for key in eligible),
        "right_successes": sum(bool(right[key][target]) for key in eligible),
        "both_success": both,
        "right_only_gain": len(gains),
        "left_only_loss": len(losses),
        "neither_success": neither,
        "risk_difference_right_minus_left": (
            (len(gains) - len(losses)) / count if count else None),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
            len(gains), len(losses)),
        "gains": [{"scene": scene, "episode": episode}
                  for scene, episode in gains],
        "losses": [{"scene": scene, "episode": episode}
                   for scene, episode in losses],
    }


def cluster_interval(
    scenes: list[str],
    episode_ids: dict[str, list[str]],
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    *,
    conditional_b: bool,
    seed: int,
    resamples: int,
) -> list[float]:
    numerators: list[float] = []
    denominators: list[int] = []
    target = "reached_b" if conditional_b else "joint"
    for scene in scenes:
        numerator = 0.0
        denominator = 0
        for episode in episode_ids[scene]:
            key = (scene, episode)
            if conditional_b and not left[key]["reached_a"]:
                continue
            require(left[key]["reached_a"] == right[key]["reached_a"],
                    f"Goal-A pairing differs for {key}")
            denominator += 1
            numerator += (float(right[key][target]) -
                          float(left[key][target]))
        numerators.append(numerator)
        denominators.append(denominator)
    nums = np.asarray(numerators, dtype=float)
    dens = np.asarray(denominators, dtype=float)
    require(len(scenes) > 0 and resamples > 0,
            "cluster bootstrap has no input")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for start in range(0, resamples, BOOTSTRAP_CHUNK):
        count = min(BOOTSTRAP_CHUNK, resamples - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        sampled_den = dens[indices].sum(axis=1)
        valid = sampled_den > 0
        samples.append(nums[indices].sum(axis=1)[valid] /
                       sampled_den[valid])
    values = np.concatenate(samples)
    require(values.size > 0, "conditional bootstrap has no eligible sample")
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def arm_outcome(rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    a_successes = sum(bool(row["reached_a"]) for row in rows.values())
    joint = sum(bool(row["joint"]) for row in rows.values())
    b_successes = sum(bool(row["reached_b"]) for row in rows.values()
                      if row["reached_a"])
    b_steps = [row["steps_b"] for row in rows.values() if row["reached_a"]]
    return {
        "episodes": total,
        "goal_a_successes": a_successes,
        "goal_a_sr": a_successes / total if total else None,
        "joint_successes": joint,
        "joint_sr": joint / total if total else None,
        "goal_b_successes_given_a": b_successes,
        "goal_b_eligible": a_successes,
        "goal_b_sr_given_a": b_successes / a_successes if a_successes else None,
        "mean_goal_b_steps_given_a": (
            float(np.mean(b_steps)) if b_steps else None),
    }


def summarize(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema_version")
    require(manifest_schema in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2},
            "manifest schema changed")
    require(manifest.get("audit", {}).get("status") == "ok",
            "manifest audit failed")
    require(manifest.get("audit", {}).get("no_mp3d_evaluation") is True,
            "manifest does not prohibit MP3D")
    require(manifest.get("frozen_guards", {}).get(
        "no_scene_or_episode_filtering_after_outcomes") is True,
        "outcome-filtering guard absent")
    scenes = [str(value) for value in manifest.get("scenes", [])]
    require(len(scenes) == 10 and manifest.get("scene_count") == 10,
            "selected population must retain all ten frozen scenes")
    episode_count = int(manifest.get("episodes_per_scene", 0))
    require(episode_count == 4, "per-constructible-scene count changed")
    episode_ids = {
        scene: [str(row["episode"]) for row in manifest["episodes"][scene]]
        for scene in scenes
    }
    constructible_indices = [
        index for index, scene in enumerate(scenes) if episode_ids[scene]
    ]
    constructible_scenes = [scenes[index] for index in constructible_indices]
    if manifest_schema == MANIFEST_SCHEMA_V1:
        require(constructible_indices == list(range(10)) and
                manifest.get("episode_count") == 40,
                "V1 transfer requires all 40 frozen episodes")
    else:
        require(constructible_indices == [0, 1, 2, 3, 4, 5, 6, 7, 9] and
                manifest.get("evaluation_scene_indices") ==
                constructible_indices and
                manifest.get("constructible_scene_count") == 9 and
                manifest.get("episode_count") == 36,
                "V2 construction-attrition population changed")
        attrition = manifest.get("construction_attrition", {})
        require(attrition.get("target_met") is False and
                attrition.get("underpowered") is True and
                attrition.get("navigation_outcomes_read") is False and
                len(attrition.get("receipts", [])) == 1,
                "V2 construction attrition receipt is invalid")
        require(manifest.get("frozen_guards", {}).get(
            "failed_scene_retained_as_explicit_attrition") is True,
            "V2 attrition guard absent")
    require(all(len(episode_ids[scene]) == 4
                for scene in constructible_scenes) and
            all(len(episode_ids[scene]) == 0
                for scene in scenes if scene not in constructible_scenes),
            "constructible-scene balance changed")
    expected_keys = {(scene, episode) for scene in scenes
                     for episode in episode_ids[scene]}
    require(len(expected_keys) == manifest.get("episode_count"),
            "manifest episode count differs from identities")
    manifest_sha = sha256_file(manifest_path)
    base_seed = int(manifest["evaluation"]["base_seed"])

    all_rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        arm: {} for arm in ARMS}
    certified_audit: list[dict[str, Any]] = []
    fallback_mismatches: list[dict[str, str]] = []
    for index, scene in enumerate(scenes):
        if not episode_ids[scene]:
            continue
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        contract_path = scene_root / "scene_contract.json"
        require(contract_path.is_file(), f"missing scene contract: {scene}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        validate_scene_contract_schema(contract, scene)
        expected_contract = {
            "scene": scene,
            "scene_index": index,
            "manifest_sha256": manifest_sha,
            "arm_order": list(WILLIAMS_ORDERS[index % 4]),
            "actual_online_goal_a_trace": True,
            "certified_runtime_role_label_visible": False,
            "raw_fixed_role_oracle": True,
        }
        for field, wanted in expected_contract.items():
            require(contract.get(field) == wanted,
                    f"{scene}: scene contract {field} changed")

        trace_root = scene_root / "trace_source"
        validate_trace_summary(
            trace_root / "summary.json", episode_count=episode_count,
            base_seed=base_seed)
        trace_metrics = read_metrics(
            trace_root / "metric.csv", episode_ids[scene])
        trace_shas: dict[str, str] = {}
        for episode in episode_ids[scene]:
            trace_path = trace_root / f"{episode}_leg1_trace.json"
            require(trace_path.is_file(), f"missing actual-online trace: {trace_path}")
            trace_shas[episode] = sha256_file(trace_path)
            require(trace_metrics[episode].get("leg1_trace_sha256") ==
                    trace_shas[episode], f"trace SHA receipt differs: {scene}/{episode}")

        scene_metrics: dict[str, dict[str, dict[str, str]]] = {}
        for arm in ARMS:
            arm_root = scene_root / arm
            validate_arm_summary(
                arm_root / "summary.json", arm,
                episode_count=episode_count, base_seed=base_seed)
            metrics = read_metrics(arm_root / "metric.csv", episode_ids[scene])
            scene_metrics[arm] = metrics
            for episode in episode_ids[scene]:
                metric = metrics[episode]
                reached_a = truth(metric.get("reached_A"))
                reached_b = truth(metric.get("reached_B"))
                require(reached_a == truth(
                    trace_metrics[episode].get("reached_A")),
                    f"{arm}: Goal-A outcome differs: {scene}/{episode}")
                require(metric.get("leg1_trace_sha256") == trace_shas[episode],
                        f"{arm}: Goal-A trace differs: {scene}/{episode}")
                steps_b = integer(metric.get("steps_B"))
                if not reached_a:
                    require(not reached_b and steps_b == 0,
                            f"{arm}: Goal-B ran after Goal-A failure")
                key = (scene, episode)
                all_rows[arm][key] = {
                    "reached_a": reached_a,
                    "reached_b": reached_b,
                    "joint": reached_a and reached_b,
                    "steps_b": steps_b,
                    "raw": metric,
                }
                if arm == "certified_relocalization":
                    audit = validate_certified_episode(
                        arm_root, metric, reached_a=reached_a)
                    audit.update({"scene": scene, "episode": episode})
                    certified_audit.append(audit)

        for episode in episode_ids[scene]:
            key = (scene, episode)
            audit = next(row for row in certified_audit
                         if row["scene"] == scene and row["episode"] == episode)
            if not audit["fallback_episode"]:
                continue
            certified_raw = all_rows["certified_relocalization"][key]["raw"]
            native_raw = all_rows["native"][key]["raw"]
            fields = (
                "reached_B", "steps_B", "termination_reason_B", "len_B",
                "final_dist_B", "blocked_steps_B",
            )
            changed = [field for field in fields
                       if certified_raw.get(field) != native_raw.get(field)]
            if changed:
                fallback_mismatches.append({
                    "scene": scene,
                    "episode": episode,
                    "fields": ",".join(changed),
                })

    for arm in ARMS:
        require(set(all_rows[arm]) == expected_keys,
                f"{arm}: result population differs from manifest")
    require(len(certified_audit) == len(expected_keys),
            "certificate audit does not cover the ITT population")
    require(not fallback_mismatches,
            f"certified fallback differs from native: {fallback_mismatches}")

    analysis = manifest["analysis"]
    seed = int(analysis["cluster_bootstrap_seed"])
    resamples = int(analysis["cluster_bootstrap_resamples"])
    contrast_specs = {
        "certified_minus_native": ("native", "certified_relocalization"),
        "certified_minus_raw_fixed_oracle_role": (
            "raw_fixed_oracle_role", "certified_relocalization"),
        "certified_minus_geometry": (
            "geometry_router", "certified_relocalization"),
        "raw_fixed_oracle_role_minus_native": (
            "native", "raw_fixed_oracle_role"),
        "geometry_minus_native": ("native", "geometry_router"),
    }
    contrasts: dict[str, Any] = {}
    for offset, (name, (left_name, right_name)) in enumerate(
            contrast_specs.items()):
        joint = paired(
            left_name, right_name, all_rows[left_name], all_rows[right_name],
            expected_keys, conditional_b=False)
        conditional = paired(
            left_name, right_name, all_rows[left_name], all_rows[right_name],
            expected_keys, conditional_b=True)
        joint["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
            constructible_scenes, episode_ids,
            all_rows[left_name], all_rows[right_name],
            conditional_b=False, seed=seed + 2 * offset,
            resamples=resamples)
        conditional[
            "scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
                constructible_scenes, episode_ids,
                all_rows[left_name], all_rows[right_name],
                conditional_b=True, seed=seed + 2 * offset + 1,
                resamples=resamples)
        contrasts[name] = {"joint": joint, "conditional_b": conditional}

    eligible_a = sum(audit["takeover_episode"] or audit["fallback_episode"]
                     for audit in certified_audit)
    takeover_episodes = sum(audit["takeover_episode"]
                            for audit in certified_audit)
    fallback_episodes = sum(audit["fallback_episode"]
                            for audit in certified_audit)
    require(eligible_a == takeover_episodes + fallback_episodes,
            "certificate decisions do not partition A-success episodes")
    latencies = [value for audit in certified_audit
                 for value in audit["uncached_ms"]]
    inliers = [value for audit in certified_audit
               for value in audit["pnp_inliers"]]
    ranks = [value for audit in certified_audit
             for value in audit["dino_ranks"]]
    certificate = {
        "goal_a_eligible_episodes": eligible_a,
        "takeover_episodes": takeover_episodes,
        "fallback_episodes": fallback_episodes,
        "takeover_coverage_given_a": (
            takeover_episodes / eligible_a if eligible_a else None),
        "exact_native_fallback_episodes": fallback_episodes,
        "fallback_behavior_mismatch_count": len(fallback_mismatches),
        "requests": sum(audit["requests"] for audit in certified_audit),
        "accepted_plans": sum(audit["accepted_plans"]
                              for audit in certified_audit),
        "abstained_plans": sum(audit["abstained_plans"]
                               for audit in certified_audit),
        "runtime_failures": sum(audit["runtime_failures"]
                                for audit in certified_audit),
        "uncached_latency_ms": {
            "n": len(latencies),
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "pnp_inliers": {
            "n": len(inliers),
            "median": percentile([float(value) for value in inliers], 0.5),
        },
        "selected_dino_rank": {
            "n": len(ranks),
            "median": percentile([float(value) for value in ranks], 0.5),
            "maximum": max(ranks) if ranks else None,
        },
    }
    return {
        "schema_version": (
            REPORT_SCHEMA_V1 if manifest_schema == MANIFEST_SCHEMA_V1
            else REPORT_SCHEMA_V2),
        "scope": (
            "frozen causal-visual Revisit transfer on the constructible "
            "population from ten selected outcome-disjoint HM3D v0.2 val "
            "scenes; not MP3D, not an official GOAT/MemoNav score"),
        "estimand": (
            "closed-loop utility of certified history bearing under a frozen "
            "NavDP controller, with raw fixed bearing as an explicitly "
            "role-oracle upper comparator"),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "scene_count": len(constructible_scenes),
        "selected_scene_count": len(scenes),
        "constructible_scene_count": len(constructible_scenes),
        "episode_count": len(expected_keys),
        "construction_attrition": manifest.get("construction_attrition"),
        "actual_online_goal_a_trace": True,
        "intention_to_treat": True,
        "no_mp3d_evaluation": True,
        "arms": {arm: arm_outcome(all_rows[arm]) for arm in ARMS},
        "contrasts": contrasts,
        "certificate_audit": certificate,
        "interpretation_guards": {
            "raw_fixed_oracle_role_is_not_deployable": True,
            "no_novel_open_set_safety_claim_from_this_revisit_only_protocol": True,
            "no_threshold_tuning_on_hm3d_heldout_val10": True,
            "all_goal_a_failures_retained": True,
            "no_outcome_filtered_scene_or_episode": True,
            "pre_navigation_construction_attrition_reported": True,
            "original_scene_indices_and_arm_orders_preserved": True,
        },
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(args.manifest.resolve(), args.run_root.resolve())
    write_exclusive(args.out.resolve(), report)
    receipt = args.out.resolve().with_suffix(args.out.suffix + ".sha256")
    receipt.write_text(
        f"{sha256_file(args.out.resolve())}  {args.out.resolve().name}\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "report": str(args.out.resolve()),
        "episodes": report["episode_count"],
        "arms": {name: value["joint_successes"]
                 for name, value in report["arms"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
