#!/usr/bin/env python3
"""Fail-closed summary for the fresh-episode Revisit confirmation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.build_revisit_fresh_manifest import sha256_file
from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    exact_sign_p,
    load_arm,
    paired_summary,
    require,
)


ARMS = ("geometry_router", "known_revisit_direct", "native")
BOOTSTRAP_CHUNK = 10_000


def decision_branch(
    delta: float,
    pvalue: float,
    interval: list[float],
    direct_sr: float,
    native_sr: float,
) -> str:
    direct_wins = delta > 0 and pvalue < 0.05 and interval[0] > 0
    geometry_wins = delta < 0 and pvalue < 0.05 and interval[1] < 0
    if direct_wins and direct_sr >= native_sr:
        return "replace_geometry_hard_gate_then_seek_fresh_scene_confirmation"
    if geometry_wins:
        return "retain_geometry_hard_gate"
    return "inconclusive_keep_geometry_and_do_not_retune_on_these_episodes"


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
    if arm == "geometry_router":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "memory_geometry",
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
    elif arm == "known_revisit_direct":
        expected.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "revisit_adapter": "legacy_metric",
            "graph_subgoal_spacing_m": 0.0,
            "graph_subgoal_arrival_m": 0.6,
        })
    else:
        expected.update({
            "server_backend": "navdp",
            "hybrid_route": "phase",
            # eval_2leg_habitat records its parser default even for the pure
            # NavDP backend.  The adapter is unreachable without
            # hybrid_pose, but the audit must still match the raw receipt.
            "revisit_adapter": "legacy_metric",
            "graph_subgoal_spacing_m": None,
            "graph_subgoal_arrival_m": None,
        })
    for field, wanted in expected.items():
        require(summary.get(field) == wanted,
                f"{path}: {field}={summary.get(field)!r}, expected {wanted!r}")


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
            (len(gains) - len(losses)) / len(eligible) if eligible else None
        ),
        "mcnemar_exact_two_sided_p": exact_sign_p(len(gains), discordant),
        "gains": [{"scene": scene, "episode": episode} for scene, episode in gains],
        "losses": [{"scene": scene, "episode": episode} for scene, episode in losses],
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
    values = []
    for start in range(0, resamples, BOOTSTRAP_CHUNK):
        count = min(BOOTSTRAP_CHUNK, resamples - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        sampled_den = dens[indices].sum(axis=1)
        valid = sampled_den > 0
        values.append(nums[indices].sum(axis=1)[valid] / sampled_den[valid])
    samples = np.concatenate(values)
    require(samples.size > 0, "cluster bootstrap has no eligible sample")
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


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
    per_scene = int(manifest["episodes_per_scene"])
    require(all(len(values) == per_scene for values in episode_ids.values()),
            "manifest has unequal episode counts")
    expected = {
        (scene, episode) for scene in scenes for episode in episode_ids[scene]
    }
    require(len(expected) == 160, "formal confirmation requires 160 episodes")

    permutations = list(itertools.permutations(ARMS))
    rows = {arm: {} for arm in ARMS}
    arm_orders = {}
    for index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        contract_path = scene_root / "scene_contract.json"
        require(contract_path.is_file(), f"missing scene contract: {scene}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_order = list(permutations[index % len(permutations)])
        require(contract.get("scene") == scene, f"scene contract identity: {scene}")
        require(contract.get("scene_index") == index, f"scene contract index: {scene}")
        require(contract.get("arm_order") == expected_order, f"arm order changed: {scene}")
        require(contract.get("manifest_sha256") == sha256_file(manifest_path),
                f"manifest receipt mismatch: {scene}")
        trace_root = scene_root / "trace_source"
        trace_summary = json.loads((trace_root / "summary.json").read_text())
        for field, wanted in {
            "episodes": per_scene,
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "leg1_mode": "policy",
            "stop_after_leg1": True,
            "write_leg1_trace": True,
            "deterministic_plan_seeds": True,
        }.items():
            require(trace_summary.get(field) == wanted,
                    f"trace source {scene} changed {field}")
        for episode in episode_ids[scene]:
            require((trace_root / f"{episode}_leg1_trace.json").is_file(),
                    f"missing trace: {scene}/{episode}")
        arm_orders[scene] = expected_order
        for arm in ARMS:
            validate_summary(scene_root / arm / "summary.json", arm, per_scene)
            rows[arm].update(load_arm(scene_root, arm, scene))

    for arm in ARMS:
        require(set(rows[arm]) == expected, f"{arm}: result keys differ from manifest")
    for key in sorted(expected):
        trace_path = (
            run_root / "scenes" / f"{scenes.index(key[0]):02d}_{key[0]}" /
            "trace_source" / f"{key[1]}_leg1_trace.json"
        )
        trace_sha = sha256_file(trace_path)
        for arm in ARMS:
            require(rows[arm][key]["leg1_trace_sha256"] == trace_sha,
                    f"{arm}: source trace hash mismatch: {key}")

    analysis = manifest["analysis"]
    seed = int(analysis["cluster_bootstrap_seed"])
    resamples = int(analysis["cluster_bootstrap_resamples"])
    contrast_specs = {
        "direct_minus_geometry": ("geometry_router", "known_revisit_direct"),
        "direct_minus_native": ("native", "known_revisit_direct"),
        "geometry_minus_native": ("native", "geometry_router"),
    }
    contrasts = {}
    for offset, (name, (left_name, right_name)) in enumerate(contrast_specs.items()):
        paired = paired_summary(
            left_name, right_name, rows[left_name], rows[right_name], expected)
        conditional = conditional_paired(
            left_name, right_name, rows[left_name], rows[right_name], expected)
        paired["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
            scenes, episode_ids, rows[left_name], rows[right_name],
            conditional=False, seed=seed + offset * 2, resamples=resamples)
        conditional["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
            scenes, episode_ids, rows[left_name], rows[right_name],
            conditional=True, seed=seed + offset * 2 + 1, resamples=resamples)
        contrasts[name] = {"joint": paired, "conditional_b": conditional}

    primary = contrasts["direct_minus_geometry"]["joint"]
    primary_ci = primary["scene_cluster_bootstrap_risk_difference_95"]
    delta = float(primary["joint_sr_delta_right_minus_left"])
    pvalue = float(primary["mcnemar_exact_two_sided_p"])
    direct_sr = arm_summary([rows["known_revisit_direct"][key] for key in sorted(expected)])["joint"]["sr"]
    native_sr = arm_summary([rows["native"][key] for key in sorted(expected)])["joint"]["sr"]
    branch = decision_branch(delta, pvalue, primary_ci, direct_sr, native_sr)

    return {
        "scope": manifest["scope"],
        "question": "Does known-Revisit raw-DINO direct beat the geometry hard gate on fresh episodes?",
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "manifest_sha256": sha256_file(manifest_path),
            "source_bundle_receipt_sha256": sha256_file(run_root / "source_bundle.sha256"),
            "training_scene_overlap": manifest["audit"]["training_scene_overlap"],
            "shared_native_goal_a_trace": True,
            "balanced_arm_order": True,
            "development_read": False,
            "blind_read": False,
        },
        "arm_order_by_scene": arm_orders,
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "contrasts": contrasts,
        "decision": {
            "branch": branch,
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
        encoding="utf-8",
    )
    print(json.dumps({"arms": report["arms"], "decision": report["decision"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
