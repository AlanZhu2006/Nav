#!/usr/bin/env python3
"""Fail-closed audit and paired summary for the Novel-A bearing gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

try:  # direct script execution from MemNavData/
    from deterministic_eval_protocol import diffusion_plan_seed
    from novel_a_bearing_gate import ARMS, require, rotated_arm_order
except ModuleNotFoundError:  # package import in unit tests
    from MemNavData.deterministic_eval_protocol import diffusion_plan_seed
    from MemNavData.novel_a_bearing_gate import ARMS, require, rotated_arm_order


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "1.0"):
        return True
    if normalized in ("false", "0", "0.0"):
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def finite_float(value: Any, field: str) -> float:
    converted = float(value)
    require(math.isfinite(converted), f"{field} is non-finite")
    return converted


def wilson(successes: int, total: int,
           z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total
        + z * z / (4.0 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def exact_mcnemar_two_sided(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = min(gains, losses)
    probability = 2.0 * sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2 ** discordant)
    return float(min(1.0, probability))


def scene_cluster_bootstrap(
    rows_by_arm: dict[str, dict[tuple[str, str], dict]],
    right_arm: str,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    native = rows_by_arm["native"]
    right = rows_by_arm[right_arm]
    scenes = sorted({scene for scene, _episode in native})
    scene_differences = []
    for scene in scenes:
        keys = sorted(key for key in native if key[0] == scene)
        require(keys, f"scene has no episodes: {scene}")
        scene_differences.append(float(np.mean([
            float(right[key]["reached"]) - float(native[key]["reached"])
            for key in keys
        ])))
    values = np.asarray(scene_differences, dtype=float)
    rng = np.random.default_rng(int(seed))
    # Chunking fixes memory use while retaining the exact frozen RNG stream.
    samples = []
    remaining = int(resamples)
    while remaining:
        count = min(remaining, 20_000)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(np.mean(values[indices], axis=1))
        remaining -= count
    distribution = np.concatenate(samples)
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def validate_plan_trace(
    path: Path,
    *,
    scene_index: int,
    scene: str,
    episode: str,
    episode_index: int,
    episode_seed: int,
    arm: str,
    arm_position: int,
    expected_order: tuple[str, ...],
    exec_horizon: int,
) -> dict:
    payload = json.loads(path.read_text())
    require(payload["scene_index"] == scene_index, "plan scene index mismatch")
    require(payload["scene"] == scene, "plan scene mismatch")
    require(payload["episode"] == episode, "plan episode mismatch")
    require(payload["episode_index"] == episode_index,
            "plan episode index mismatch")
    require(payload["episode_seed"] == episode_seed, "plan seed mismatch")
    require(payload["arm"] == arm, "plan arm mismatch")
    require(payload["arm_position"] == arm_position,
            "plan arm position mismatch")
    require(tuple(payload["arm_order"]) == expected_order,
            "plan arm order mismatch")
    plans = payload.get("plans")
    require(isinstance(plans, list) and plans, "plan trace is empty")
    for index, plan in enumerate(plans):
        prefix = f"{scene}/{episode}/{arm}/plan{index}"
        require(plan.get("plan_index") == index,
                f"{prefix}: plan indices are not dense")
        require(plan.get("step") == index * exec_horizon,
                f"{prefix}: replan cadence changed")
        expected_seed = diffusion_plan_seed(episode_seed, 0, index)
        require(plan.get("requested_diffusion_seed") == expected_seed,
                f"{prefix}: requested seed changed")
        require(plan.get("native_diffusion_seed") == expected_seed,
                f"{prefix}: native seed echo mismatch")
        source = plan.get("trajectory_source")
        require(source in ("native", "oracle_token"),
                f"{prefix}: invalid trajectory source")
        if source == "oracle_token":
            require(arm == "oracle_token_periodic",
                    f"{prefix}: token trajectory leaked into another arm")
            require(plan.get("token_request_deg") is not None,
                    f"{prefix}: token request missing")
            require(plan.get("token_diffusion_seed") == expected_seed,
                    f"{prefix}: token seed mismatch")
            before = plan.get("token_queue_hashes_before")
            after = plan.get("token_queue_hashes_after")
            require(isinstance(before, list) and before and before == after,
                    f"{prefix}: FIFO content changed during resample")
            require(isinstance(plan.get("token_shadow"), dict),
                    f"{prefix}: token critic shadow missing")
        else:
            require(plan.get("token_diffusion_seed") is None,
                    f"{prefix}: abstain/native path sampled a token")
            require(plan.get("token_queue_hashes_before") is None
                    and plan.get("token_queue_hashes_after") is None,
                    f"{prefix}: native path contains token FIFO audit")
        if arm != "ideal_periodic_yaw":
            require(float(plan.get("ideal_turn_deg") or 0.0) == 0.0,
                    f"{prefix}: ideal yaw leaked into another arm")
        require(0 < int(plan.get("executed_steps")) <= exec_horizon,
                f"{prefix}: invalid executed step count")
        finite_float(plan.get("path_m"), f"{prefix}.path_m")
    return payload


def load_and_audit(
    run_root: Path,
    manifest: dict,
    protocol: dict,
    *,
    allow_smoke: bool,
) -> tuple[dict[str, dict[tuple[str, str], dict]], dict]:
    selected_scenes = manifest["selection"]["selected_scenes"]
    if allow_smoke:
        scene_roots = sorted((run_root / "scenes").glob("[0-9][0-9]_*"))
        require(scene_roots, "smoke root contains no scene outputs")
    else:
        scene_roots = [
            run_root / "scenes" / f"{index:02d}_{scene}"
            for index, scene in enumerate(selected_scenes)]
        require(all(path.is_dir() for path in scene_roots),
                "formal run is missing scene directories")
        actual_roots = sorted((run_root / "scenes").glob("[0-9][0-9]_*"))
        require(actual_roots == sorted(scene_roots),
                "formal run has missing or extra scene directories")

    rows_by_arm: dict[str, dict[tuple[str, str], dict]] = {
        arm: {} for arm in ARMS}
    expected_protocol_sha = protocol["_sha256"]
    expected_manifest_sha = protocol["manifest"]["sha256"]
    expected_inputs_sha = protocol["input_overlay"]["sha256"]
    for scene_root in scene_roots:
        prefix, scene = scene_root.name.split("_", 1)
        scene_index = int(prefix)
        require(selected_scenes[scene_index] == scene,
                "scene directory and manifest order disagree")
        run_meta_path = scene_root / "run_meta.json"
        csv_path = scene_root / "bearing_arms.csv"
        require(run_meta_path.is_file() and csv_path.is_file(),
                f"incomplete scene output: {scene_root}")
        run_meta = json.loads(run_meta_path.read_text())
        require(run_meta.get("status") == "complete", "scene is not complete")
        require(run_meta.get("formal") is (not allow_smoke),
                "formal/smoke marker mismatch")
        require(run_meta.get("scene_index") == scene_index
                and run_meta.get("scene") == scene,
                "run metadata scene mismatch")
        require(tuple(run_meta.get("arms", [])) == ARMS,
                "run metadata arm contract changed")
        require(run_meta.get("protocol_sha256") == expected_protocol_sha,
                "run used a different bearing protocol")
        require(run_meta.get("manifest_sha256") == expected_manifest_sha,
                "run used a different benchmark manifest")
        require(run_meta.get("input_overlay_sha256") == expected_inputs_sha,
                "run used a different Goal-A input overlay")
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not allow_smoke:
            require(len(rows) == 3 * protocol["evaluation"]["episodes_per_scene"],
                    f"formal arm coverage mismatch: {scene}")
        seen = set()
        for row in rows:
            episode = row["episode"]
            episode_index = int(row["episode_index"])
            arm = row["arm"]
            key3 = (episode, arm)
            require(key3 not in seen, f"duplicate arm row: {scene} {key3}")
            seen.add(key3)
            require(arm in ARMS, f"unknown arm row: {arm}")
            require(row["scene"] == scene
                    and int(row["scene_index"]) == scene_index,
                    "metric scene mismatch")
            require(truth(row["formal"]) is (not allow_smoke),
                    "metric formal/smoke marker mismatch")
            require(row["protocol_sha256"] == expected_protocol_sha,
                    "metric protocol SHA mismatch")
            require(row["manifest_sha256"] == expected_manifest_sha,
                    "metric manifest SHA mismatch")
            require(row["input_overlay_sha256"] == expected_inputs_sha,
                    "metric input-overlay SHA mismatch")
            expected_order = rotated_arm_order(scene_index, episode_index)
            arm_position = int(row["arm_position"])
            require(json.loads(row["arm_order"]) == list(expected_order),
                    "metric arm order mismatch")
            require(expected_order[arm_position] == arm,
                    "metric arm position mismatch")
            episode_seed = int(row["seed"])
            require(episode_seed == protocol["evaluation"]["base_seed"]
                    + episode_index, "metric episode seed mismatch")
            plans_path = scene_root / row["plans_file"]
            require(plans_path.is_file(), f"missing plans file: {plans_path}")
            validate_plan_trace(
                plans_path,
                scene_index=scene_index,
                scene=scene,
                episode=episode,
                episode_index=episode_index,
                episode_seed=episode_seed,
                arm=arm,
                arm_position=arm_position,
                expected_order=expected_order,
                exec_horizon=int(protocol["evaluation"]["execution_horizon"]),
            )
            key = (scene, episode)
            require(key not in rows_by_arm[arm], f"duplicate metric key: {key}")
            rows_by_arm[arm][key] = {
                "scene": scene,
                "episode": episode,
                "episode_index": episode_index,
                "seed": episode_seed,
                "reached": truth(row["reached"]),
                "geo_a": finite_float(row["geo_A"], "geo_A"),
                "path_m": finite_float(row["path_len_m"], "path_len_m"),
                "final_dist_m": finite_float(
                    row["final_dist_m"], "final_dist_m"),
                "steps": int(row["steps"]),
                "plan_count": int(row["plan_count"]),
                "ideal_turn_count": int(row["ideal_turn_count"]),
                "ideal_turn_abs_deg": finite_float(
                    row["ideal_turn_abs_deg"], "ideal_turn_abs_deg"),
                "token_plan_count": int(row["token_plan_count"]),
                "token_path_m": finite_float(
                    row["token_path_m"], "token_path_m"),
                "token_disabled_reason": row.get("token_disabled_reason") or None,
                "goal_jpg_sha256": row["goal_jpg_sha256"],
            }

    expected_keys = set(rows_by_arm["native"])
    require(expected_keys, "no audited episodes")
    for arm in ARMS:
        require(set(rows_by_arm[arm]) == expected_keys,
                f"paired coverage mismatch for {arm}")
    for key in expected_keys:
        reference = rows_by_arm["native"][key]
        for arm in ARMS[1:]:
            row = rows_by_arm[arm][key]
            require(row["seed"] == reference["seed"],
                    f"paired seed mismatch: {key}")
            require(row["goal_jpg_sha256"] == reference["goal_jpg_sha256"],
                    f"paired Goal-A image mismatch: {key}")
            require(math.isclose(row["geo_a"], reference["geo_a"],
                                 rel_tol=0.0, abs_tol=1e-9),
                    f"paired geodesic mismatch: {key}")
    expected_count = (len(scene_roots)
                      * protocol["evaluation"]["episodes_per_scene"])
    if not allow_smoke:
        require(len(expected_keys) == protocol["evaluation"]["episodes"],
                "formal 40-episode coverage failed")
        require(len(scene_roots) == protocol["evaluation"]["scenes"],
                "formal 20-scene coverage failed")
    return rows_by_arm, {
        "status": "ok",
        "formal": not allow_smoke,
        "scenes": len(scene_roots),
        "episodes": len(expected_keys),
        "expected_episode_capacity": expected_count,
        "paired_seed_goal_geodesic_match": True,
        "policy_training_overlap": sorted(
            set(selected_scenes) & set(manifest["training_scenes"])),
    }


def arm_summary(rows: dict[tuple[str, str], dict]) -> dict:
    values = list(rows.values())
    successes = sum(row["reached"] for row in values)
    spl_values = [
        row["geo_a"] / max(row["geo_a"], row["path_m"], 1e-9)
        if row["reached"] else 0.0 for row in values]
    return {
        "episodes": len(values),
        "successes": successes,
        "sr": successes / len(values),
        "wilson_95": wilson(successes, len(values)),
        "mean_spl": float(statistics.fmean(spl_values)),
        "mean_final_distance_m": float(statistics.fmean(
            row["final_dist_m"] for row in values)),
        "mean_path_m": float(statistics.fmean(
            row["path_m"] for row in values)),
    }


def paired_summary(
    rows_by_arm: dict[str, dict[tuple[str, str], dict]],
    right_arm: str,
    protocol: dict,
) -> dict:
    native = rows_by_arm["native"]
    right = rows_by_arm[right_arm]
    gains, losses, both, neither = [], [], [], []
    for key in sorted(native):
        left_success = native[key]["reached"]
        right_success = right[key]["reached"]
        if right_success and not left_success:
            gains.append(key)
        elif left_success and not right_success:
            losses.append(key)
        elif left_success:
            both.append(key)
        else:
            neither.append(key)
    bootstrap = protocol["bootstrap"]
    return {
        "episodes": len(native),
        "gain_count": len(gains),
        "loss_count": len(losses),
        "net_gain_count": len(gains) - len(losses),
        "paired_risk_difference": (len(gains) - len(losses)) / len(native),
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
            len(gains), len(losses)),
        "scene_cluster_bootstrap_95": scene_cluster_bootstrap(
            rows_by_arm,
            right_arm,
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
        ),
        "gains": [f"{scene}/{episode}" for scene, episode in gains],
        "losses": [f"{scene}/{episode}" for scene, episode in losses],
        "both_success": len(both),
        "neither_success": len(neither),
    }


def gate_decision(primary: dict) -> str:
    gains = int(primary["gain_count"])
    losses = int(primary["loss_count"])
    net = int(primary["net_gain_count"])
    if gains <= 2 or net <= 0:
        return "no_go"
    if gains >= 4 and losses <= 1 and net >= 3:
        return "go_to_unseen_526_pool_not_paper_confirmation"
    return "ambiguous_retest_disjoint_before_building_frontier_ranker"


def summarize(rows_by_arm: dict[str, dict[tuple[str, str], dict]],
              protocol: dict, audit: dict) -> dict:
    arms = {arm: arm_summary(rows_by_arm[arm]) for arm in ARMS}
    primary = paired_summary(rows_by_arm, "ideal_periodic_yaw", protocol)
    secondary = paired_summary(rows_by_arm, "oracle_token_periodic", protocol)
    token_rows = list(rows_by_arm["oracle_token_periodic"].values())
    secondary["actuation"] = {
        "activated_episodes": sum(
            row["token_plan_count"] > 0 for row in token_rows),
        "token_plans": sum(row["token_plan_count"] for row in token_rows),
        "mean_token_path_m": float(statistics.fmean(
            row["token_path_m"] for row in token_rows)),
        "max_burst_exhausted_episodes": sum(
            row["token_disabled_reason"] == "max_burst_exhausted"
            for row in token_rows),
    }
    decision = (gate_decision(primary) if audit.get("formal") is True
                else "not_evaluated_transport_smoke")
    return {
        "protocol_version": protocol["protocol_version"],
        "benchmark_role": protocol["benchmark_role"],
        "audit": audit,
        "arms": arms,
        "primary_ideal_vs_native": primary,
        "conditional_secondary_token_vs_native": secondary,
        "decision": decision,
        "interpretation_guard": (
            "ideal is a privileged mechanism upper bound; token still uses "
            "oracle bearing; this internal set cannot confirm a paper claim"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    protocol["_sha256"] = sha256(args.protocol)
    require(sha256(args.manifest) == protocol["manifest"]["sha256"],
            "summarizer manifest SHA mismatch")
    require(not (set(manifest["selection"]["selected_scenes"])
                 & set(manifest["training_scenes"])),
            "evaluation scenes overlap policy training")
    rows_by_arm, audit = load_and_audit(
        args.run_root, manifest, protocol, allow_smoke=args.allow_smoke)
    result = summarize(rows_by_arm, protocol, audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
