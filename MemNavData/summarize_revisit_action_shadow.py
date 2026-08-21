#!/usr/bin/env python3
"""Summarize the frozen seven-episode Revisit actionability shadow probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


TARGETS = (
    (6, "pLe4wQe7qrG", "episode_0001", "direct_loss"),
    (4, "yqstnuAEVhm", "episode_0001", "direct_gain"),
    (11, "uNb9QFRL6hY", "episode_0000", "direct_gain"),
    (13, "ac26ZMwG7aT", "episode_0000", "direct_gain"),
    (15, "qoiz87JEwZ2", "episode_0000", "direct_gain"),
    (17, "i5noydFURQK", "episode_0000", "direct_gain"),
    (19, "gZ6f7yhEvPG", "episode_0000", "direct_gain"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric(path: Path, episode: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("episode") == episode
        ]
    require(
        len(rows) == 1,
        f"{path} must contain exactly one row for {episode}; found {len(rows)}",
    )
    return rows[0]


def finite_values(plans: list[dict], field: str, limit: int | None = None):
    values = []
    for plan in plans[:limit]:
        value = plan.get(field)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def average(plans: list[dict], field: str, limit: int | None = None):
    values = finite_values(plans, field, limit)
    return mean(values) if values else None


def ratio(numerator: float | None, denominator: float | None):
    if numerator is None or denominator is None or denominator <= 1e-8:
        return None
    return numerator / denominator


def episode_summary(
    run_root: Path,
    reference_root: Path,
    index: int,
    scene: str,
    episode: str,
    prior_outcome: str,
) -> dict[str, Any]:
    target_root = run_root / "targets" / f"{index:02d}_{scene}_{episode}"
    summary_path = target_root / "summary.json"
    plans_path = target_root / f"{episode}_plans.json"
    require(summary_path.is_file(), f"missing {summary_path}")
    require(plans_path.is_file(), f"missing {plans_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("episodes") == 1, f"{target_root}: episode count")
    require(summary.get("server_backend") == "hybrid_pose", "backend changed")
    require(summary.get("hybrid_route") == "phase", "route changed")
    require(summary.get("revisit_controller") == "navdp_mixed", "controller changed")
    require(
        summary.get("revisit_action_shadow") == "native_counterfactual",
        "shadow mode changed",
    )
    require(summary.get("deterministic_plan_seeds") is True, "seed mode changed")

    metric = read_metric(target_root / "metric.csv", episode)
    reference_dir = reference_root / "scenes" / f"{index:02d}_{scene}"
    reference = read_metric(
        reference_dir / "known_revisit_direct" / "metric.csv", episode
    )
    require(metric["seed"] == reference["seed"], "episode seed differs")
    require(
        metric["leg1_trace_sha256"] == reference["leg1_trace_sha256"],
        "shared Goal-A trace differs",
    )
    factual_fields = (
        "geo_A",
        "geo_B",
        "reached_A",
        "reached_B",
        "steps_A",
        "steps_B",
        "len_A",
        "len_B",
        "terminal_final_goal_dist_m",
        "termination_reason_A",
        "termination_reason_B",
    )
    for field in factual_fields:
        require(metric[field] == reference[field], f"factual field differs: {field}")

    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    plans = payload["legB"]
    require(bool(plans), f"{scene}/{episode}: no Goal-B plans")
    available = [
        plan for plan in plans
        if plan.get("revisit_action_shadow_available") is True
    ]
    unavailable = [
        plan for plan in plans
        if plan.get("revisit_action_shadow_available") is False
    ]
    require(len(available) + len(unavailable) == len(plans), "shadow receipt missing")
    for plan in available:
        require(
            plan.get("revisit_action_shadow_memory_mutated") is False,
            "shadow reported memory mutation",
        )
        require(
            plan.get("revisit_action_shadow_queue_hash_match") is True,
            "shadow FIFO fingerprints differ",
        )
        require(
            int(plan["revisit_action_shadow_seed"])
            == int(plan["requested_diffusion_seed"]),
            "shadow plan seed differs",
        )

    fields = {
        "memory_mean": (
            "revisit_action_shadow_memory_candidate_endpoint_length_mean"
        ),
        "native_mean": (
            "revisit_action_shadow_native_candidate_endpoint_length_mean"
        ),
        "memory_zero": (
            "revisit_action_shadow_memory_zero_candidate_fraction"
        ),
        "native_zero": (
            "revisit_action_shadow_native_zero_candidate_fraction"
        ),
        "memory_selected": (
            "revisit_action_shadow_memory_selected_endpoint_m"
        ),
        "native_selected": (
            "revisit_action_shadow_native_selected_endpoint_m"
        ),
        "pointgoal_ratio": (
            "revisit_action_shadow_endpoint_to_pointgoal_ratio"
        ),
    }

    def window(limit: int | None):
        result = {name: average(available, field, limit) for name, field in fields.items()}
        result.update(
            endpoint_mean_ratio=ratio(result["memory_mean"], result["native_mean"]),
            selected_endpoint_ratio=ratio(
                result["memory_selected"], result["native_selected"]
            ),
            memory_all_zero_plans=sum(
                float(plan[fields["memory_zero"]]) >= 1.0 - 1e-12
                for plan in available[:limit]
            ),
            native_all_zero_plans=sum(
                float(plan[fields["native_zero"]]) >= 1.0 - 1e-12
                for plan in available[:limit]
            ),
            plans=min(len(available), limit) if limit is not None else len(available),
        )
        return result

    first_active_sequence = [
        {
            "step": int(plan["step"]),
            "endpoint_mean_ratio_memory_over_native": plan.get(
                "revisit_action_shadow_endpoint_mean_ratio_memory_over_native"
            ),
            "endpoint_to_pointgoal_ratio": plan.get(
                "revisit_action_shadow_endpoint_to_pointgoal_ratio"
            ),
            "memory_zero_candidate_fraction": plan.get(
                "revisit_action_shadow_memory_zero_candidate_fraction"
            ),
            "native_zero_candidate_fraction": plan.get(
                "revisit_action_shadow_native_zero_candidate_fraction"
            ),
        }
        for plan in available[:4]
    ]

    return {
        "scene": scene,
        "episode": episode,
        "prior_outcome": prior_outcome,
        "factual_reached_b": bool(float(metric["reached_B"])),
        "reference_reached_b": bool(float(reference["reached_B"])),
        "factual_matches_reference": True,
        "factual_final_distance_m": float(metric["terminal_final_goal_dist_m"]),
        "reference_final_distance_m": float(reference["terminal_final_goal_dist_m"]),
        "plans": len(plans),
        "available_shadow_plans": len(available),
        "unavailable_shadow_plans": len(unavailable),
        "first_memory_active_step": int(available[0]["step"]),
        "unavailable_shadow_reasons": dict(Counter(
            str(plan.get("revisit_action_shadow_reason")) for plan in unavailable
        )),
        "first_four_memory_active": window(4),
        "first_four_memory_active_sequence": first_active_sequence,
        "all_plans": window(None),
    }


def summarize(run_root: Path, reference_root: Path) -> dict[str, Any]:
    rows = [
        episode_summary(run_root, reference_root, *target)
        for target in TARGETS
    ]
    loss = next(row for row in rows if row["prior_outcome"] == "direct_loss")
    gains = [row for row in rows if row["prior_outcome"] == "direct_gain"]
    ordered = sorted(
        rows,
        key=lambda row: (
            float("inf")
            if row["first_four_memory_active"]["endpoint_mean_ratio"] is None
            else row["first_four_memory_active"]["endpoint_mean_ratio"]
        ),
    )
    return {
        "scope": "seven selected discordant episodes; mechanism-only shadow",
        "audit": {
            "status": "ok",
            "targets": len(rows),
            "all_fifo_fingerprints_match": True,
            "all_shadow_seeds_match": True,
            "factual_trajectories_matching_reference": sum(
                row["factual_matches_reference"] for row in rows
            ),
            "total_factual_plans": sum(row["plans"] for row in rows),
            "paired_shadow_plans": sum(
                row["available_shadow_plans"] for row in rows
            ),
            "explicitly_unavailable_shadow_plans": sum(
                row["unavailable_shadow_plans"] for row in rows
            ),
            "unavailable_shadow_reasons": dict(sum(
                (
                    Counter(row["unavailable_shadow_reasons"])
                    for row in rows
                ),
                Counter(),
            )),
            "source_inputs_sha256": sha256_file(
                run_root / "source_inputs.sha256"
            ),
            "report_generator_sha256": sha256_file(Path(__file__)),
        },
        "descriptive_mechanism": {
            "window_definition": "first four memory-active replans",
            "loss_first_four_endpoint_mean_ratio": loss[
                "first_four_memory_active"
            ][
                "endpoint_mean_ratio"
            ],
            "gain_first_four_endpoint_mean_ratios": [
                row["first_four_memory_active"]["endpoint_mean_ratio"]
                for row in gains
            ],
            "loss_first_four_endpoint_to_pointgoal_sequence": [
                item["endpoint_to_pointgoal_ratio"]
                for item in loss["first_four_memory_active_sequence"]
            ],
            "gain_first_four_endpoint_to_pointgoal_sequences": {
                row["scene"]: [
                    item["endpoint_to_pointgoal_ratio"]
                    for item in row["first_four_memory_active_sequence"]
                ]
                for row in gains
            },
            "loss_ratio_rank_low_to_high": ordered.index(loss) + 1,
            "loss_first_four_memory_all_zero_plans": loss[
                "first_four_memory_active"
            ]["memory_all_zero_plans"],
            "loss_first_four_native_all_zero_plans": loss[
                "first_four_memory_active"
            ]["native_all_zero_plans"],
            "is_posthoc_selected_mechanism_set": True,
            "defines_actionability_threshold": False,
        },
        "targets": rows,
        "decision": {
            "deployment_approved": False,
            "online_arbiter_approved": False,
            "paper_claim_approved": False,
            "next_step": (
                "interpret mechanism and, if proceeding, freeze a separate "
                "full-pool persistence contract without tuning on these targets"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    report = summarize(args.run_root, args.reference_root)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
