#!/usr/bin/env python3
"""Strict paired summary for the Phase-B ranking-only P0 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from MemNavData.summarize_expanded_navdp_router_eval import (
        arm_summary,
        load_arm,
        paired_summary,
        require,
    )
except ModuleNotFoundError:  # direct script invocation
    from summarize_expanded_navdp_router_eval import (  # type: ignore
        arm_summary,
        load_arm,
        paired_summary,
        require,
    )


ARMS = ("geometry_router", "learned_rank_geometry", "navdp_native")


def integer(row: dict[str, str], name: str) -> int:
    value = row.get(name)
    return 0 if value in (None, "") else int(float(value))


def load_phase_b_audit(
    scene_root: Path, expected_checkpoint_sha256: str
) -> tuple[dict[str, int | float], list[dict]]:
    arm_root = scene_root / "learned_rank_geometry"
    summary_path = arm_root / "summary.json"
    metric_path = arm_root / "metric.csv"
    require(summary_path.is_file(), f"missing learned summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    ranker = summary.get("phase_b_ranker") or {}
    require(
        ranker.get("checkpoint_sha256") == expected_checkpoint_sha256,
        f"Phase-B checkpoint mismatch: {summary_path}",
    )
    require(
        ranker.get("deployment_approved") is False
        and ranker.get("allow_unapproved") is True,
        f"Phase-B experimental approval state changed: {summary_path}",
    )
    require(
        ranker.get("activation_semantics")
        == "diagnostic_only_geometry_gate_unchanged",
        f"Phase-B activation semantics changed: {summary_path}",
    )
    require(
        summary.get("phase_b_p0_transport_valid") is True,
        f"Phase-B transport audit failed: {summary_path}",
    )
    with metric_path.open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    counters: dict[str, int | float] = {
        "requests": sum(integer(row, "phase_b_rank_request_count")
                        for row in metrics),
        "successes": sum(integer(row, "phase_b_rank_success_count")
                         for row in metrics),
        "uncached_candidate_sets": sum(
            integer(row, "phase_b_uncached_rank_count") for row in metrics),
        "fallbacks": sum(integer(row, "phase_b_rank_fallback_count")
                         for row in metrics),
        "activation_violations": sum(
            integer(row, "phase_b_activation_violation_count")
            for row in metrics),
        "order_changes": sum(integer(row, "phase_b_order_change_count")
                             for row in metrics),
        "uncached_ranking_ms": sum(float(
            row.get("phase_b_uncached_ranking_ms") or 0.0)
            for row in metrics),
    }
    require(
        counters["successes"] == counters["requests"]
        and counters["fallbacks"] == 0
        and counters["activation_violations"] == 0,
        f"Phase-B metric audit failed: {metric_path}",
    )

    candidate_sets = []
    for row in metrics:
        episode = row["episode"]
        plans = json.loads(
            (arm_root / f"{episode}_plans.json").read_text())
        for plan in [*plans.get("legA", []), *plans.get("legB", [])]:
            if (plan.get("router_phase_b_requested") is not True
                    or plan.get("router_phase_b_cached") is not False):
                continue
            require(
                plan.get("router_phase_b_success") is True
                and plan.get("router_phase_b_activation_used") is False,
                f"invalid Phase-B plan audit: {arm_root} {episode}",
            )
            dino_order = plan.get("router_candidate_order_dino") or []
            learned_order = plan.get("router_candidate_order_used") or []
            require(
                sorted(dino_order) == sorted(learned_order)
                and len(dino_order) == len(set(dino_order)),
                f"Phase-B changed shortlist membership: {arm_root} {episode}",
            )
            candidate_sets.append({
                "episode": episode,
                "step": int(plan["step"]),
                "candidate_count": len(dino_order),
                "order_changed": dino_order != learned_order,
                "dino_order": dino_order,
                "learned_order": learned_order,
                "selected_anchor": plan.get("router_selected_anchor"),
                "selected_learned_rank": plan.get(
                    "router_selected_candidate_rank"),
                "selected_dino_rank": plan.get(
                    "router_selected_candidate_dino_rank"),
            })
    require(
        len(candidate_sets) == counters["uncached_candidate_sets"],
        f"Phase-B plan/metric candidate-set count mismatch: {arm_root}",
    )
    return counters, candidate_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-phase-b-sha256", required=True)
    args = parser.parse_args()
    require(
        len(args.expected_phase_b_sha256) == 64
        and all(character in "0123456789abcdef"
                for character in args.expected_phase_b_sha256),
        "expected Phase-B SHA256 is malformed",
    )

    manifest = json.loads(args.manifest.read_text())
    scenes = manifest["selection"]["selected_scenes"]
    require(len(scenes) == 20, "P0 manifest is not the frozen 20-scene pool")
    require(
        not set(scenes) & set(manifest["training_scenes"]),
        "P0 scene pool overlaps router training scenes",
    )
    expected = {
        (scene, record["episode"])
        for scene in scenes for record in manifest["episodes"][scene]
    }
    require(len(expected) == 40, "P0 manifest does not contain 40 episodes")

    rows = {arm: {} for arm in ARMS}
    phase_totals: dict[str, int | float] = {}
    candidate_sets = []
    for index, scene in enumerate(scenes):
        scene_root = args.run_root / "scenes" / f"{index:02d}_{scene}"
        for arm in ARMS:
            rows[arm].update(load_arm(scene_root, arm, scene))
        counters, sets = load_phase_b_audit(
            scene_root, args.expected_phase_b_sha256)
        for name, value in counters.items():
            phase_totals[name] = phase_totals.get(name, 0) + value
        for item in sets:
            candidate_sets.append({"scene": scene, **item})

    for arm in ARMS:
        require(set(rows[arm]) == expected,
                f"{arm} result keys differ from the P0 manifest")
    require(phase_totals.get("requests", 0) > 0,
            "P0 run never exercised learned ranking")
    require(phase_totals.get("uncached_candidate_sets", 0) > 0,
            "P0 run produced no uncached candidate set")

    result = {
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "policy_training_overlap": [],
            "shared_goal_a_trace_pairing": True,
            "same_local_server_process_required_by_runner": True,
            "phase_b_checkpoint_sha256": args.expected_phase_b_sha256,
            "model_activation_used": False,
        },
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "pairwise": {
            "learned_vs_geometry": paired_summary(
                "geometry_router", "learned_rank_geometry",
                rows["geometry_router"], rows["learned_rank_geometry"],
                expected,
            ),
            "learned_vs_native": paired_summary(
                "navdp_native", "learned_rank_geometry",
                rows["navdp_native"], rows["learned_rank_geometry"],
                expected,
            ),
            "geometry_vs_native": paired_summary(
                "navdp_native", "geometry_router",
                rows["navdp_native"], rows["geometry_router"], expected,
            ),
        },
        "phase_b_ranking": {
            **phase_totals,
            "mean_uncached_ranking_ms": (
                phase_totals["uncached_ranking_ms"]
                / phase_totals["uncached_candidate_sets"]
            ),
            "order_change_rate": (
                phase_totals["order_changes"]
                / phase_totals["uncached_candidate_sets"]
            ),
            "candidate_sets": candidate_sets,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
