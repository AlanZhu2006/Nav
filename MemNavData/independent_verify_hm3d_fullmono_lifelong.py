#!/usr/bin/env python3
"""Independent raw-file verifier for full-mono lifelong accumulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from hm3d_fullmono_lifelong import ARMS, QUERY_NAMES, require, sha256_file
from independent_verify_shared_online_lifelong_nnr import (
    accepted_anchors,
    causal_plan_projection,
    memory_indices,
    verify_forced_reject_plans,
)


SCHEMA = "hm3d_fullmono_lifelong_independent_verification_v1_20260824"


def independent_exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    require(gains >= 0 and losses >= 0, "negative discordant count")
    if discordant == 0:
        return 1.0
    smaller = min(int(gains), int(losses))
    probability = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * probability)


def independent_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def independent_cluster_interval(
    records: list[dict], *, draws: int, seed: int
) -> dict:
    by_scene: dict[str, list[float]] = {}
    for record in records:
        by_scene.setdefault(str(record["scene"]), []).append(
            float(record["difference"])
        )
    scenes = sorted(by_scene)
    require(bool(scenes), "independent cluster bootstrap has no scenes")
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = []
        for _slot in scenes:
            sample.extend(by_scene[scenes[generator.randrange(len(scenes))]])
        estimates.append(sum(sample) / len(sample))
    return {
        "scene_clusters": len(scenes),
        "draws": draws,
        "seed": seed,
        "risk_difference": sum(
            float(record["difference"]) for record in records
        ) / len(records),
        "percentile_95_CI": [
            independent_quantile(estimates, 0.025),
            independent_quantile(estimates, 0.975),
        ],
    }


def independent_arm_counts(rows: list[dict]) -> dict:
    episodes = len(rows)
    evaluated_b2 = sum(int(row["evaluated_B2"]) for row in rows)
    evaluated_c2 = sum(int(row["evaluated_C2"]) for row in rows)
    return {
        "episodes": episodes,
        "C": {
            "success": sum(int(row["reached_C"]) for row in rows),
            "evaluated": episodes,
        },
        "B2_given_C": {
            "success": sum(int(row["reached_B2"]) for row in rows),
            "evaluated": evaluated_b2,
        },
        "C2_given_C_B2": {
            "success": sum(int(row["reached_C2"]) for row in rows),
            "evaluated": evaluated_c2,
        },
        "prefix_survival": {
            str(k): sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in rows
            ) for k in (1, 2, 3)
        },
        "query_joint": {
            "success": sum(int(row["query_joint_success"]) for row in rows),
            "evaluated": episodes,
        },
        "B2_factual_B_anchor": {
            "used": sum(int(row["B2_used_factual_B_anchor"]) for row in rows),
            "evaluated": evaluated_b2,
        },
    }


def independent_prefix_comparison(
    first: dict[tuple[str, str], dict],
    second: dict[tuple[str, str], dict],
    *,
    first_name: str,
    second_name: str,
) -> dict:
    require(set(first) == set(second), "independent paired populations differ")
    output = {}
    for k in (1, 2, 3):
        records = []
        gains = losses = 0
        for scene, episode in sorted(first):
            a = int(first[(scene, episode)][
                "queries_completed_before_first_failure"
            ]) >= k
            b = int(second[(scene, episode)][
                "queries_completed_before_first_failure"
            ]) >= k
            difference = a - b
            gains += difference == 1
            losses += difference == -1
            records.append({
                "scene": scene, "episode": episode,
                "difference": difference,
            })
        output[str(k)] = {
            "endpoint": f"survived_at_least_{k}_queries",
            first_name: sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in first.values()
            ),
            second_name: sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in second.values()
            ),
            "n": len(records),
            "paired_gains": gains,
            "paired_losses": losses,
            "exact_McNemar_p": independent_exact_mcnemar(gains, losses),
            "scene_cluster_bootstrap": independent_cluster_interval(
                records, draws=100000, seed=20260824 + k
            ),
        }
    return output


def independent_mono_depth_audit(
    plans: list[dict], *, allow_first40_bootstrap: bool
) -> dict:
    """Validate the monocular depth transaction without collector helpers."""

    require(bool(plans), "full-mono prefix has no NavDP plans")
    scale_hashes: set[str] = set()
    bootstrap = active = 0
    for plan in plans:
        require(plan.get("navdp_depth_source") == "monocular_sidecar",
                "prefix plan depth source is not monocular_sidecar")
        require(plan.get("metric_depth_sensor_consumed") is not True,
                "prefix plan consumed simulator metric depth")
        receipt = plan.get("monocular_depth_receipt")
        require(isinstance(receipt, dict),
                "prefix plan omitted its monocular depth receipt")
        require(receipt.get("depth_contract")
                == "raw_lingbot_depth_first40_v1",
                "prefix monocular depth contract changed")
        require(receipt.get("metric_depth_sensor_consumed") is False,
                "prefix mono receipt reports metric sensor consumption")
        require(receipt.get("image_sha256")
                and receipt.get("depth_png_sha256"),
                "prefix mono receipt omitted RGB/depth content hashes")
        frame = int(receipt.get("frame_index", -1))
        if frame < 40:
            require(allow_first40_bootstrap,
                    "post-A factual B unexpectedly used bootstrap depth")
            require(float(receipt.get("depth_nonzero_fraction", -1.0)) == 0.0
                    and receipt.get("scale_active") is False,
                    "first-40 bootstrap was not exact zero depth")
            bootstrap += 1
            continue
        require(receipt.get("scale_active") is True,
                "mono depth remained inactive after frame 40")
        scale = receipt.get("scale_receipt")
        require(isinstance(scale, dict), "active mono receipt omitted scale")
        require(scale.get("scale_evidence_contract")
                == "causal_first_prefix_rgb_only_v1",
                "mono scale evidence contract changed")
        require(scale.get("whole_episode_ground_cache_consumed") is False,
                "mono scale consumed whole-episode future evidence")
        scale_hash = receipt.get("scale_receipt_sha256")
        require(bool(scale_hash), "mono scale receipt hash is missing")
        scale_hashes.add(str(scale_hash))
        active += 1
    require(active > 0, "full-mono prefix never reached active mono depth")
    require(len(scale_hashes) == 1,
            "full-mono prefix used more than one frozen scale receipt")
    return {
        "metric_sensor_plan_count": 0,
        "monocular_receipt_plan_count": len(plans),
        "monocular_scale_hash_count": len(scale_hashes),
        "bootstrap_plan_count": bootstrap,
        "active_plan_count": active,
    }


def verify_fullmono_prefix(population_path: Path, item: dict) -> dict:
    """Re-audit raw A/B depth receipts from the sealed query population."""

    population_root = population_path.parent.resolve()
    benchmark_path = (population_root / item["benchmark"]).resolve()
    require(
        benchmark_path == population_root
        or population_root in benchmark_path.parents,
        "benchmark path escaped the sealed population",
    )
    require(benchmark_path.is_file(), "sealed benchmark is missing")
    require(sha256_file(benchmark_path) == item["benchmark_sha256"],
            "sealed benchmark hash changed")
    benchmark = json.loads(benchmark_path.read_text())
    require(benchmark["scene"] == item["scene"]
            and benchmark["episode"] == item["episode"],
            "sealed benchmark identity changed")

    source_a = Path(benchmark["source_online_A_episode"])
    receipt_a_path = source_a / "receipt.json"
    trace_a_path = source_a / "online_a_trace.json"
    require(sha256_file(receipt_a_path)
            == benchmark["source_online_A_receipt_sha256"],
            "online-A receipt changed")
    require(sha256_file(trace_a_path)
            == benchmark["source_online_A_trace_sha256"],
            "online-A trace changed")
    receipt_a = json.loads(receipt_a_path.read_text())
    trace_a = json.loads(trace_a_path.read_text())
    source_episode = str(benchmark.get(
        "source_online_A_episode_id", item["episode"]
    ))
    require(receipt_a["scene"] == item["scene"]
            and receipt_a["episode"] == source_episode,
            "online-A receipt identity changed")
    require(trace_a.get("reached") is True
            and trace_a.get("source_hybrid_route") == "native_sidecar",
            "online-A was not a successful native-sidecar rollout")
    audit_a = independent_mono_depth_audit(
        trace_a["plans"], allow_first40_bootstrap=True
    )

    trace_b_path = benchmark_path.parent / benchmark["online_B_trace"]
    require(sha256_file(trace_b_path) == benchmark["online_B_trace_sha256"],
            "online-B trace changed")
    trace_b = json.loads(trace_b_path.read_text())
    require(trace_b.get("reached") is True
            and trace_b.get("source_hybrid_route") == "native_sidecar",
            "online-B was not a successful native-sidecar rollout")
    audit_b_full = independent_mono_depth_audit(
        trace_b["plans"], allow_first40_bootstrap=False
    )
    audit_b = {
        key: audit_b_full[key] for key in (
            "metric_sensor_plan_count", "monocular_receipt_plan_count",
            "monocular_scale_hash_count",
        )
    }

    completion_path = benchmark_path.parent / benchmark[
        "factual_B_completion"
    ]
    require(sha256_file(completion_path)
            == benchmark["factual_B_completion_sha256"],
            "factual-B completion changed")
    completion = json.loads(completion_path.read_text())
    require(completion.get("controller") == "frozen_navdp_native_sidecar"
            and completion.get("navdp_depth_source") == "monocular_sidecar"
            and int(completion.get("metric_depth_sensor_reads", -1)) == 0,
            "factual-B completion violates full-mono control")
    require(completion.get("depth_audit") == audit_b,
            "stored factual-B depth audit differs from raw plans")
    return {
        "A_plan_receipts": len(trace_a["plans"]),
        "B_plan_receipts": len(trace_b["plans"]),
        "A_depth_audit": audit_a,
        "B_depth_audit": audit_b,
    }


def load_raw_arm(
    records: list[dict], evaluation_root: Path, arm: str
) -> tuple[dict, dict, dict]:
    """Read immutable evaluator outputs, never aggregate-owned copies."""

    evaluation_root = evaluation_root.resolve()
    rows, plans, compute = {}, {}, {}
    for record in records:
        root = Path(record["run_root"]).resolve()
        require(root == evaluation_root or evaluation_root in root.parents,
                f"{arm}: raw run root escaped evaluation root")
        require(root.name == arm, f"{arm}: raw run root scope changed")
        metric_path = root / "result/metric.csv"
        plans_path = root / "result" / f"{record['episode']}_plans.json"
        summary_path = root / "result/summary.json"
        compute_path = root / "compute_identity.json"
        require(sha256_file(metric_path) == record["metric_sha256"],
                f"{arm}: raw metric hash changed")
        require(sha256_file(plans_path) == record["plans_sha256"],
                f"{arm}: raw plans hash changed")
        require(sha256_file(summary_path) == record["summary_sha256"],
                f"{arm}: raw summary hash changed")
        require(sha256_file(compute_path)
                == record["compute_identity_sha256"],
                f"{arm}: raw compute identity hash changed")
        with metric_path.open(newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        require(len(raw_rows) == 1, f"{arm}: raw metric row count changed")
        row = raw_rows[0]
        identity = (str(row["scene"]), str(row["episode"]))
        require(identity == (str(record["scene"]), str(record["episode"])),
                f"{arm}: raw input identity changed")
        require(identity not in rows, f"{arm}: duplicate raw identity")
        rows[identity] = row
        plans[identity] = json.loads(plans_path.read_text())
        compute[identity] = json.loads(compute_path.read_text())
    require(bool(rows), f"{arm}: no raw evaluation inputs")
    return rows, plans, compute


def same_pose(first: list[dict], second: list[dict]) -> bool:
    if not first or not second:
        return False
    keys = ("x", "y", "z", "yaw")
    return all(math.isclose(float(first[0][key]), float(second[0][key]),
                            rel_tol=0.0, abs_tol=1e-9) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    population_receipt = (
        args.population.parent / "population.json.sha256"
    ).read_text().split()
    require(population_receipt and population_receipt[0] == sha256_file(args.population),
            "population receipt changed")
    population = json.loads(args.population.read_text())
    require(population["selection_reads_C_B2_C2_navigation_outcomes"] is False,
            "population was selected using query outcomes")
    summary_path = args.aggregate / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows_by_arm, plans_by_arm, compute_by_arm = {}, {}, {}
    for arm in ARMS:
        (rows_by_arm[arm], plans_by_arm[arm],
         compute_by_arm[arm]) = load_raw_arm(
             summary["raw_inputs"][arm], args.evaluation_root, arm
         )
    identities = sorted((row["scene"], row["episode"])
                        for row in population["accepted"])
    require(all(set(rows_by_arm[arm]) == set(identities) for arm in ARMS),
            "raw arm population differs from sealed population")

    paired = []
    forced_decisions = forced_shadow = 0
    causal_c_hash_matches = 0
    prefix_a_plan_receipts = prefix_b_plan_receipts = 0
    paired_compute_matches = same_node_forced_matches = 0
    population_items = {
        (str(row["scene"]), str(row["episode"])): row
        for row in population["accepted"]
    }
    for identity in identities:
        prefix_audit = verify_fullmono_prefix(
            args.population, population_items[identity]
        )
        prefix_a_plan_receipts += int(prefix_audit["A_plan_receipts"])
        prefix_b_plan_receipts += int(prefix_audit["B_plan_receipts"])
        all_compute = compute_by_arm["all_prior"][identity]
        initial_compute = compute_by_arm["initial_leg_only"][identity]
        forced_compute = compute_by_arm["forced_reject_native"][identity]
        for scope, payload in (
            ("all_prior", all_compute),
            ("initial_leg_only", initial_compute),
            ("forced_reject_native", forced_compute),
        ):
            require(payload.get("schema_version")
                    == "cec_compute_identity_v1_20260824",
                    f"{identity}/{scope}: compute receipt schema changed")
            require(payload.get("runtime_scope") == scope,
                    f"{identity}/{scope}: compute scope changed")
        require(all_compute["host"] == initial_compute["host"]
                and all_compute["gpu_uuid"] == initial_compute["gpu_uuid"]
                and all_compute.get("cuda_visible_devices")
                == initial_compute.get("cuda_visible_devices"),
                f"{identity}: primary CEC pair did not share GPU/node")
        for process in ("memnav", "navdp", "cec_hub"):
            require(all_compute[process] == initial_compute[process],
                    f"{identity}: primary pair did not share {process} process")
        require(set(all_compute["paired_scope_order"])
                == {"all_prior", "initial_leg_only"}
                and all_compute["paired_scope_order"]
                == initial_compute["paired_scope_order"],
                f"{identity}: primary paired order receipt changed")
        paired_compute_matches += 1
        require(forced_compute["host"] == all_compute["host"]
                and forced_compute["gpu_uuid"] == all_compute["gpu_uuid"]
                and forced_compute.get("cuda_visible_devices")
                == all_compute.get("cuda_visible_devices"),
                f"{identity}: forced baseline ran on another GPU/node")
        require(any(forced_compute[process] != all_compute[process]
                    for process in ("memnav", "navdp", "cec_hub")),
                f"{identity}: forced baseline did not use a fresh fail-closed process")
        same_node_forced_matches += 1
        payloads = {arm: plans_by_arm[arm][identity] for arm in ARMS}
        rows = {arm: rows_by_arm[arm][identity] for arm in ARMS}
        for arm in ARMS:
            payload = payloads[arm]
            row = rows[arm]
            require(payload["runtime_role_visible"] is False
                    and int(row["runtime_role_visible"]) == 0,
                    f"{identity}/{arm}: role label leaked")
            indices = memory_indices(payload)
            require(indices == list(range(indices[0], indices[-1] + 1)),
                    f"{identity}/{arm}: causal memory is not contiguous")
            require(int(row["metric_depth_reads_queries"]) == 0,
                    f"{identity}/{arm}: metric depth consumed")
            a_ceiling = int(row["online_A_candidate_ceiling"])
            b_ceiling = int(row["online_B_candidate_ceiling"])
            receipts = payload["goal_session_receipts"]
            require(receipts and int(receipts[0]["candidate_ceiling"]) == a_ceiling,
                    f"{identity}/{arm}: C escaped A history")
            for name in QUERY_NAMES:
                plans = payload["queries"][name]
                for plan in plans:
                    require(plan.get("metric_depth_sensor_consumed") is not True,
                            f"{identity}/{arm}/{name}: metric depth consumed")
                anchors = accepted_anchors(plans)
                if anchors:
                    require(max(anchors) <= int(plans[0]["cec_candidate_ceiling"]),
                            f"{identity}/{arm}/{name}: anchor escaped ceiling")
            if int(row["evaluated_B2"]):
                expected = b_ceiling if arm in (
                    "all_prior", "forced_reject_native"
                ) else a_ceiling
                require(int(receipts[1]["candidate_ceiling"]) == expected,
                        f"{identity}/{arm}: B2 candidate ceiling changed")
            if arm == "forced_reject_native":
                for name in QUERY_NAMES:
                    checked = verify_forced_reject_plans(payload["queries"][name])
                    forced_decisions += checked["decisions"]
                    forced_shadow += checked["shadow_takeovers"]
        # The treatment begins at B2.  A/B replay and the entire executed C
        # prefix must therefore be action-identical between the two CEC arms.
        all_payload = payloads["all_prior"]
        initial_payload = payloads["initial_leg_only"]
        require(all_payload["memory_traces"]["A"]
                == initial_payload["memory_traces"]["A"]
                and all_payload["memory_traces"]["B"]
                == initial_payload["memory_traces"]["B"],
                f"{identity}: factual A/B replay differs")
        require(causal_plan_projection(all_payload["queries"]["C"])
                == causal_plan_projection(initial_payload["queries"]["C"]),
                f"{identity}: C action causality differs before treatment")
        require(all_payload["rollout_traces"]["C"]
                == initial_payload["rollout_traces"]["C"],
                f"{identity}: C physical prefix differs before treatment")
        causal_c_hash_matches += 1
        require(int(rows["all_prior"]["reached_C"])
                == int(rows["initial_leg_only"]["reached_C"]),
                f"{identity}: C outcome differs before treatment")
        if int(rows["all_prior"]["reached_C"]):
            require(same_pose(
                all_payload["rollout_traces"]["B2"],
                initial_payload["rollout_traces"]["B2"],
            ), f"{identity}: paired B2 starts from another pose")
            difference = (
                int(rows["all_prior"]["reached_B2"])
                - int(rows["initial_leg_only"]["reached_B2"])
            )
            paired.append({
                "scene": identity[0], "episode": identity[1],
                "all_prior": int(rows["all_prior"]["reached_B2"]),
                "initial_leg_only": int(rows["initial_leg_only"]["reached_B2"]),
                "difference": difference,
            })
    gains = sum(row["difference"] == 1 for row in paired)
    losses = sum(row["difference"] == -1 for row in paired)
    recomputed = {
        "estimable": bool(paired),
        "n": len(paired),
        "all_prior_success": sum(row["all_prior"] for row in paired),
        "initial_leg_only_success": sum(row["initial_leg_only"] for row in paired),
        "paired_gains": gains,
        "paired_losses": losses,
        "exact_McNemar_p": (
            independent_exact_mcnemar(gains, losses) if paired else None
        ),
        "scene_cluster_bootstrap": (
            independent_cluster_interval(
                paired, draws=100000, seed=20260824
            ) if paired else None
        ),
    }
    stored = summary["primary_B2_after_shared_C"]
    for key in (
        "estimable", "n", "all_prior_success", "initial_leg_only_success",
        "paired_gains", "paired_losses", "exact_McNemar_p",
        "scene_cluster_bootstrap",
    ):
        require(stored[key] == recomputed[key], f"stored primary {key} differs")
    recomputed_arms = {
        arm: independent_arm_counts(list(rows_by_arm[arm].values()))
        for arm in ARMS
    }
    require(summary["arms"] == recomputed_arms,
            "stored per-arm endpoints differ from raw metrics")
    recomputed_prefix = {
        "all_prior_vs_initial_leg_only": independent_prefix_comparison(
            rows_by_arm["all_prior"], rows_by_arm["initial_leg_only"],
            first_name="all_prior", second_name="initial_leg_only",
        ),
        "all_prior_vs_forced_reject_native": independent_prefix_comparison(
            rows_by_arm["all_prior"], rows_by_arm["forced_reject_native"],
            first_name="all_prior", second_name="forced_reject_native",
        ),
    }
    require(summary["paired_prefix_survival"] == recomputed_prefix,
            "stored paired prefix-survival statistics differ from raw metrics")
    verification = {
        "schema_version": SCHEMA,
        "verified": True,
        "population_sha256": sha256_file(args.population),
        "aggregate_summary_sha256": sha256_file(summary_path),
        "episodes": len(identities),
        "scenes": len({identity[0] for identity in identities}),
        "C_causal_prefix_matches": causal_c_hash_matches,
        "primary_pair_same_GPU_process_receipts": paired_compute_matches,
        "forced_baseline_same_GPU_node_receipts": same_node_forced_matches,
        "forced_reject_decisions": forced_decisions,
        "forced_reject_shadow_takeovers": forced_shadow,
        "actual_fullmono_prefix": {
            "episodes": len(identities),
            "A_plan_receipts_verified": prefix_a_plan_receipts,
            "B_plan_receipts_verified": prefix_b_plan_receipts,
            "metric_depth_reads": 0,
        },
        "metric_depth_reads": 0,
        "runtime_role_visible": False,
        "arms": recomputed_arms,
        "paired_prefix_survival": recomputed_prefix,
        "primary_B2_after_shared_C": recomputed,
    }
    require(not args.out.exists(), f"verification output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        verification, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
