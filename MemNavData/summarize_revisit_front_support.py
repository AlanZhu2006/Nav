#!/usr/bin/env python3
"""Audit the paired known-Revisit direct/front-support controller experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    load_arm,
    paired_summary,
    require,
)
from MemNavData.summarize_revisit_phase_ablation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    scene_cluster_interval,
)


ARMS = ("known_revisit_direct", "front_support_residual")
EXPECTED_ADAPTER = {
    "known_revisit_direct": "legacy_metric",
    "front_support_residual": "navdp_front_support_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indices(raw: str, scene_count: int) -> list[int]:
    if not raw.strip():
        return list(range(scene_count))
    values = [int(value) for value in raw.split(",")]
    require(len(values) == len(set(values)), "duplicate scene index")
    require(
        all(0 <= value < scene_count for value in values),
        "scene index outside manifest",
    )
    return sorted(values)


def validate_arm_summary(root: Path, arm: str) -> None:
    path = root / "summary.json"
    require(path.is_file(), f"missing summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": 2,
        "max_steps": 500,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "server_backend": "hybrid_pose",
        "hybrid_route": "phase",
        "leg1_mode": "shared_trace",
        "write_leg1_trace": False,
        "revisit_controller": "navdp_mixed",
        "revisit_adapter": EXPECTED_ADAPTER[arm],
    }
    for field, wanted in expected.items():
        require(
            summary.get(field) == wanted,
            f"{root}: summary {field} differs from front-support protocol",
        )


def support_receipts(
    run_root: Path,
    selected: list[tuple[int, str]],
    episodes: dict[str, list[str]],
) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    controllers: Counter[str] = Counter()
    behind_by_episode: dict[str, int] = {}
    first_supported_step: dict[str, int | None] = {}
    total_plans = 0
    for index, scene in selected:
        arm_root = (
            run_root / "scenes" / f"{index:02d}_{scene}"
            / "front_support_residual"
        )
        for episode in episodes[scene]:
            payload = json.loads(
                (arm_root / f"{episode}_plans.json").read_text(
                    encoding="utf-8")
            )
            plans = payload["legB"]
            total_plans += len(plans)
            key = f"{scene}/{episode}"
            behind = 0
            supported_steps = []
            for plan in plans:
                reason = str(plan.get("revisit_adapter_reason"))
                controller = str(plan.get("revisit_adapter_controller_contract"))
                reasons[reason] += 1
                controllers[controller] += 1
                if reason == "pointgoal_behind_navdp_support":
                    behind += 1
                    require(
                        plan.get("revisit_adapter_takeover") is False,
                        f"{key}: behind PointGoal unexpectedly took over",
                    )
                    require(
                        plan.get("revisit_controller") == "navdp_mixed",
                        f"{key}: configured controller changed",
                    )
                if reason == "pointgoal_inside_navdp_support":
                    require(
                        plan.get("revisit_adapter_takeover") is True,
                        f"{key}: supported PointGoal did not take over",
                    )
                    supported_steps.append(int(plan["step"]))
            behind_by_episode[key] = behind
            first_supported_step[key] = (
                min(supported_steps) if supported_steps else None
            )
    return {
        "total_goal_b_plans": total_plans,
        "reason_counts": dict(sorted(reasons.items())),
        "controller_contract_counts": dict(sorted(controllers.items())),
        "behind_fallback_plans": reasons["pointgoal_behind_navdp_support"],
        "episodes_with_behind_fallback": sum(
            count > 0 for count in behind_by_episode.values()
        ),
        "behind_fallback_by_episode": behind_by_episode,
        "first_supported_step_by_episode": first_supported_step,
    }


def summarize(
    manifest_path: Path,
    run_root: Path,
    scene_indices: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_scenes = manifest["selection"]["selected_scenes"]
    require(len(all_scenes) == 20, "front-support pool requires 20 scenes")
    indices = parse_indices(scene_indices, len(all_scenes))
    selected = [(index, all_scenes[index]) for index in indices]
    episodes = {
        scene: [record["episode"] for record in manifest["episodes"][scene]]
        for _, scene in selected
    }
    require(
        all(values == ["episode_0000", "episode_0001"]
            for values in episodes.values()),
        "front-support run requires two frozen episodes per selected scene",
    )
    expected = {
        (scene, episode)
        for _, scene in selected
        for episode in episodes[scene]
    }
    rows = {arm: {} for arm in ARMS}
    for index, scene in selected:
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        for arm in ARMS:
            validate_arm_summary(scene_root / arm, arm)
            rows[arm].update(load_arm(scene_root, arm, scene))
    for arm in ARMS:
        require(set(rows[arm]) == expected, f"{arm} rows differ from protocol")

    paired = paired_summary(
        "known_revisit_direct",
        "front_support_residual",
        rows["known_revisit_direct"],
        rows["front_support_residual"],
        expected,
    )
    receipts = support_receipts(run_root, selected, episodes)
    is_full = indices == list(range(20))
    is_t0 = indices == [6]
    if is_t0:
        require(
            all_scenes[6] == "pLe4wQe7qrG",
            "manifest index 6 is no longer the frozen pLe scene",
        )
        require(
            receipts["behind_fallback_plans"] > 0,
            "T0 did not exercise the frozen behind-support fallback",
        )
    if is_full:
        deltas = {
            scene: [
                float(rows["front_support_residual"][(scene, episode)]["joint"])
                - float(rows["known_revisit_direct"][(scene, episode)]["joint"])
                for episode in episodes[scene]
            ]
            for _, scene in selected
        }
        paired["scene_cluster_bootstrap"] = {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "risk_difference_95": scene_cluster_interval(deltas),
        }

    gains = int(paired["outcomes"]["right_only_joint_success"])
    losses = int(paired["outcomes"]["left_only_joint_success"])
    return {
        "scope": (
            "full consumed 20-scene controller-support ablation"
            if is_full else
            "T0 pLe controller-support transport check"
            if is_t0 else
            "partial consumed-pool controller-support diagnostic"
        ),
        "question": (
            "Does failing closed on PointGoals whose forward component NavDP "
            "would clip preserve direct-memory gains without native-success harm?"
        ),
        "audit": {
            "status": "ok",
            "stage": "T1_full" if is_full else "T0_pLe" if is_t0 else "partial",
            "scene_indices": indices,
            "scenes": len(selected),
            "episodes": len(expected),
            "same_server_process_required_by_launcher": True,
            "shared_trace_seed_and_goal_a_contract": True,
            "training_scene_overlap": sorted(
                {scene for _, scene in selected}
                & set(manifest["training_scenes"])
            ),
            "manifest_sha256": sha256_file(manifest_path),
            "outer_source_inputs_sha256": sha256_file(
                run_root / "ablation_source_inputs.sha256"
            ),
            "runner_source_inputs_sha256": sha256_file(
                run_root / "source_inputs.sha256"
            ),
        },
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "paired_support_minus_direct": paired,
        "support_receipts": receipts,
        "decision": {
            "transport_valid": True,
            "full_pool_complete": is_full,
            "advance_architecture": (
                is_full and gains > losses and losses == 0
            ),
            "support_only": gains,
            "direct_only_native_success_harm": losses,
            "authorize_blind_eval": False,
            "authorize_paper_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scene-indices", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    report = summarize(args.manifest, args.run_root, args.scene_indices)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
