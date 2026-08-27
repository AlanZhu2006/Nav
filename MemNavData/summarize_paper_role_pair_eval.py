#!/usr/bin/env python3
"""Aggregate the sealed paired role-query paper evaluation.

The legacy parent run and the prospective learned amendment intentionally use
different five-arm sets.  ``--include-learned-pi3x`` selects the latter; no
arm is inferred from observed outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_PROTOCOLS = ("support_controlled", "natural_direction")
FINAL14_PROTOCOLS = ("natural_direction", "hard_support")
# Backward-compatible public constant used by legacy fixture generators.
PROTOCOLS = LEGACY_PROTOCOLS
PARENT_ARMS = (
    "native", "raw_direct", "raw_fixed_bearing", "geometry_fixed",
    "certified",
)
LEARNED_AMENDMENT_ARMS = (
    "native", "raw_fixed_bearing", "geometry_fixed", "certified",
    "learned_pi3x_spatial",
)
ROLES = ("novel", "revisit")
FALLBACK_PLAN_KEYS = (
    "step",
    "requested_diffusion_seed",
    "diffusion_seed",
    "server_selected_idx",
    "trajectory_candidate_count",
    "selected_trajectory_sha256",
)
FALLBACK_METRIC_KEYS = (
    "reached", "steps", "path_len_m", "final_goal_dist_m",
    "termination_reason",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def population_layout(population: dict[str, Any]) -> dict[str, Any]:
    """Return protocol-specific sizes without conflating the hard subset."""

    if (
        population.get("schema_version")
        == "final14_role_pair_population_v1_20260817"
    ):
        declared = population["populations"]
        return {
            "mode": "final14_standard_natural_plus_hard_subset",
            "protocols": FINAL14_PROTOCOLS,
            "primary_protocol": "natural_direction",
            "expected": {
                "natural_direction": {
                    **declared["natural_standard"],
                    "analysis_roles": ["novel", "revisit"],
                },
                "hard_support": {
                    **declared["hard_support"],
                    "analysis_roles": ["revisit"],
                },
            },
            "all_targets_required": True,
        }
    histories = int(population["role_pair_constructible_histories"])
    scenes = int(population["role_pair_scene_count"])
    target_histories = int(population.get("target_histories", 20))
    target_scenes = int(population.get("target_scenes", 12))
    target_met = histories >= target_histories and scenes >= target_scenes
    return {
        "mode": "legacy_matched_role_pairs",
        "protocols": LEGACY_PROTOCOLS,
        "primary_protocol": LEGACY_PROTOCOLS[0],
        "expected": {
            protocol: {
                "histories": histories,
                "scenes": scenes,
                "target_histories": target_histories,
                "target_scenes": target_scenes,
                "target_met": target_met,
                "underpowered_if_target_not_met": not target_met,
                "analysis_roles": ["novel", "revisit"],
            }
            for protocol in LEGACY_PROTOCOLS
        },
        "all_targets_required": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(gains, losses) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def spl(record: dict[str, Any]) -> float:
    if not record["success"]:
        return 0.0
    geodesic = float(record["geodesic_m"])
    return geodesic / max(
        geodesic, float(record["path_len_m"]), 1e-12
    )


def cluster_interval(
    records: list[dict[str, Any]], left: str, right: str
) -> list[float]:
    scenes = sorted({str(record["scene"]) for record in records})
    require(bool(scenes), "cluster bootstrap population is empty")
    by_scene = {
        scene: [
            record for record in records if record["scene"] == scene
        ]
        for scene in scenes
    }
    rng = np.random.default_rng(20260814)
    samples = np.empty(100_000, dtype=np.float64)
    for index in range(samples.size):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        numerator = 0.0
        denominator = 0
        for scene in chosen:
            rows = by_scene[str(scene)]
            numerator += sum(
                int(row["outcomes"][right])
                - int(row["outcomes"][left])
                for row in rows
            )
            denominator += len(rows)
        samples[index] = numerator / denominator
    return [
        float(value)
        for value in np.quantile(samples, [0.025, 0.975])
    ]


def contrast(
    records: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    require(bool(records), "paired contrast population is empty")
    gains = [
        row for row in records
        if row["outcomes"][right] and not row["outcomes"][left]
    ]
    losses = [
        row for row in records
        if row["outcomes"][left] and not row["outcomes"][right]
    ]
    return {
        "n": len(records),
        "scene_count": len({row["scene"] for row in records}),
        "left": left,
        "right": right,
        "left_successes": sum(
            row["outcomes"][left] for row in records
        ),
        "right_successes": sum(
            row["outcomes"][right] for row in records
        ),
        "gains": len(gains),
        "losses": len(losses),
        "risk_difference_pp": (
            100.0 * (len(gains) - len(losses)) / len(records)
        ),
        "exact_mcnemar_two_sided_p": exact_mcnemar(
            len(gains), len(losses)
        ),
        "scene_cluster_bootstrap_risk_difference_95": [
            100.0 * value
            for value in cluster_interval(records, left, right)
        ],
        "gain_identities": [
            [row["scene"], row["episode"], row["role"]]
            for row in gains
        ],
        "loss_identities": [
            [row["scene"], row["episode"], row["role"]]
            for row in losses
        ],
    }


def finite_values(plans: list[dict], key: str) -> list[float]:
    values = []
    for plan in plans:
        value = plan.get(key)
        if value is not None:
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def distribution(
    values: list[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0, "mean": None, "median": None, "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def exact_native_fallback(
    native_row: dict[str, str],
    learned_row: dict[str, str],
    native_payload: dict[str, Any],
    learned_payload: dict[str, Any],
) -> dict[str, bool]:
    native_plans = native_payload["query_leg"]
    learned_plans = learned_payload["query_leg"]
    plan_equal = (
        len(native_plans) == len(learned_plans)
        and all(
            native_plan.get(key) == learned_plan.get(key)
            for native_plan, learned_plan in zip(
                native_plans, learned_plans
            )
            for key in FALLBACK_PLAN_KEYS
        )
    )
    rollout_equal = (
        native_payload["rollout_traces"]["query"]
        == learned_payload["rollout_traces"]["query"]
    )
    metric_equal = all(
        native_row[key] == learned_row[key]
        for key in FALLBACK_METRIC_KEYS
    )
    return {
        "plan_equal": plan_equal,
        "rollout_equal": rollout_equal,
        "metric_equal": metric_equal,
        "exact": plan_equal and rollout_equal and metric_equal,
    }


def _learned_query_diagnostics(
    *,
    native_row: dict[str, str],
    learned_row: dict[str, str],
    native_payload: dict[str, Any],
    learned_payload: dict[str, Any],
) -> dict[str, Any]:
    plans = learned_payload["query_leg"]
    requests = [
        plan for plan in plans
        if plan.get("learned_pi3x_relocalization_ok") is not None
    ]
    initial = [
        plan for plan in requests
        if plan.get(
            "learned_pi3x_initial_candidate_selection_cached"
        ) is False
    ]
    accepted = [
        plan for plan in requests
        if plan.get("learned_pi3x_relocalization_accepted") is True
    ]
    errors = []
    for plan in accepted:
        value = plan.get(
            "learned_pi3x_evaluation_gt_bearing_error_deg"
        )
        require(
            value is not None and math.isfinite(float(value)),
            "accepted learned plan lacks finite evaluation-only GT bearing "
            "error",
        )
        error = float(value)
        require(0.0 <= error <= 180.0, "invalid learned bearing error")
        errors.append(error)

    accept_count = int(learned_row["learned_pi3x_accept_plans"])
    takeover_count = int(learned_row["adapter_takeover_plans"])
    initial_count = int(
        learned_row["learned_pi3x_initial_inference_plans"]
    )
    runtime_failures = int(learned_row["runtime_failure_plans"])
    require(
        len(accepted) == accept_count,
        "learned raw-plan and metric accept counts differ",
    )
    require(
        len(initial) == initial_count,
        "learned raw-plan and metric initial-inference counts differ",
    )

    initial_accept = (
        len(initial) == 1
        and initial[0].get("learned_pi3x_relocalization_accepted") is True
    )
    sticky_rejection_violation = bool(
        len(initial) == 1 and not initial_accept and accepted
    )
    post_accept_abstentions = (
        sum(
            plan.get("learned_pi3x_relocalization_accepted") is not True
            for plan in requests
        )
        if initial_accept else 0
    )
    fully_abstained = accept_count == 0 and takeover_count == 0
    fallback = (
        exact_native_fallback(
            native_row, learned_row, native_payload, learned_payload
        )
        if fully_abstained else None
    )
    return {
        "request_plans": len(requests),
        "initial_inference_plans": initial_count,
        "accept_plans": accept_count,
        "takeover_plans": takeover_count,
        "runtime_failure_plans": runtime_failures,
        "initial_accept": initial_accept,
        "accept_takeover_match": accept_count == takeover_count,
        "sticky_rejection_violation": sticky_rejection_violation,
        "post_accept_abstention_plans": post_accept_abstentions,
        "fully_abstained": fully_abstained,
        "exact_native_fallback": fallback,
        "accepted_bearing_errors_deg": errors,
        "first_query_latency_ms": finite_values(
            initial, "learned_pi3x_relocalization_ms"
        ),
        "one_anchor_update_latency_ms": finite_values(
            [
                plan for plan in accepted
                if plan.get(
                    "learned_pi3x_initial_candidate_selection_cached"
                ) is True
            ],
            "learned_pi3x_relocalization_ms",
        ),
        "peak_gpu_memory_allocated_bytes": finite_values(
            requests, "learned_pi3x_peak_gpu_memory_allocated_bytes"
        ),
    }


def summarize(
    root: Path,
    *,
    excluded_scenes: tuple[str, ...] = (),
    include_learned_pi3x: bool = False,
) -> dict[str, Any]:
    arms = (
        LEARNED_AMENDMENT_ARMS
        if include_learned_pi3x else PARENT_ARMS
    )
    benchmark_root = root / "benchmarks"
    require((benchmark_root / "SEALED").is_file(), "benchmark is not sealed")
    receipt_file = benchmark_root / "BENCHMARK_FILES.sha256"
    require(
        sha256_file(receipt_file)
        == (benchmark_root / "BENCHMARK_FILES.sha256.sha256")
        .read_text().split()[0],
        "benchmark aggregate receipt changed",
    )
    population = json.loads(
        (benchmark_root / "population_receipt.json").read_text()
    )
    require(
        population["policy_outcomes_read"] is False,
        "construction read policy outcomes",
    )
    layout = population_layout(population)
    protocols = tuple(layout["protocols"])
    for protocol in protocols:
        require(
            int(layout["expected"][protocol]["histories"]) > 0,
            f"{protocol} paper population is empty",
        )

    records_by_protocol: dict[str, list[dict[str, Any]]] = {}
    latency_by_protocol: dict[str, dict[str, list[float]]] = {}
    rejections_by_protocol: dict[str, Counter] = {}
    completion_hashes = []
    learned_artifact_identities: set[tuple[str, str]] = set()
    for protocol in protocols:
        expected_population = int(
            layout["expected"][protocol]["histories"]
        )
        expected_scenes = int(layout["expected"][protocol]["scenes"])
        manifest = json.loads(
            (benchmark_root / protocol / "manifest.json").read_text()
        )
        require(
            len(manifest["episodes"]) == expected_population,
            "protocol population differs",
        )
        records = []
        latency = {
            "certificate_cached_ms": [],
            "certificate_uncached_ms": [],
            "geometry_verification_ms": [],
            "episode_wall_seconds": [],
            "learned_first_query_ms": [],
            "learned_one_anchor_update_ms": [],
            "learned_peak_gpu_memory_allocated_bytes": [],
            "learned_stored_online_history_frames": [],
        }
        rejections: Counter[str] = Counter()
        for index, episode_row in enumerate(manifest["episodes"]):
            scene = str(episode_row["scene"])
            episode = str(episode_row["episode"])
            include_in_analysis = scene not in excluded_scenes
            episode_root = (
                root / "evaluation" / protocol
                / f"{index:03d}_{scene}_{episode}"
            )
            completion_path = episode_root / "completion.json"
            completion_sha_path = episode_root / "completion.json.sha256"
            require(
                completion_path.is_file() and completion_sha_path.is_file(),
                f"missing {protocol}/{index}",
            )
            require(
                sha256_file(completion_path)
                == completion_sha_path.read_text().split()[0],
                f"completion receipt changed: {protocol}/{index}",
            )
            completion_hashes.append(sha256_file(completion_path))
            completion = json.loads(completion_path.read_text())
            require(
                completion["prefix_equality"] is True,
                "paired prefix equality failed",
            )
            require(
                completion["runtime_role_visibility"] == "none",
                "runtime role leak",
            )
            if include_in_analysis:
                for value in completion["wall_time_seconds"].values():
                    latency["episode_wall_seconds"].append(float(value))

            if include_learned_pi3x:
                episode_contract = json.loads(
                    (episode_root / "episode_contract.json").read_text()
                )
                require(
                    tuple(episode_contract.get("arms") or ()) == arms,
                    f"{protocol}/{index}: prospective arm set changed",
                )
                declared = episode_contract.get("learned_pi3x") or {}
                completed = completion.get("learned_pi3x") or {}
                identity = (
                    str(declared.get("model_sha256")),
                    str(declared.get("proof_manifest_sha256")),
                )
                require(
                    identity == (
                        str(completed.get("model_sha256")),
                        str(completed.get("proof_manifest_sha256")),
                    )
                    and all(len(value) == 64 for value in identity),
                    f"{protocol}/{index}: learned artifact identity changed",
                )
                learned_artifact_identities.add(identity)

            arm_metrics: dict[str, dict[str, dict[str, str]]] = {}
            arm_payloads: dict[str, dict[str, dict[str, Any]]] = {}
            for arm in arms:
                with (episode_root / arm / "metric.csv").open(
                    newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                require(
                    len(rows) == 2,
                    f"{protocol}/{index}/{arm}: metric count",
                )
                arm_metrics[arm] = {
                    row["analysis_role"]: row for row in rows
                }
                require(
                    set(arm_metrics[arm]) == set(ROLES),
                    "role set changed",
                )
                arm_payloads[arm] = {}
                for role, row in arm_metrics[arm].items():
                    plan_path = (
                        episode_root / arm
                        / f"{episode}_{row['query_id']}_plans.json"
                    )
                    payload = json.loads(plan_path.read_text())
                    require(
                        payload["analysis_role_not_forwarded"] is True,
                        "role was forwarded",
                    )
                    arm_payloads[arm][role] = payload

            for role in ROLES:
                native = arm_metrics["native"][role]
                outcomes = {}
                per_arm = {}
                for arm in arms:
                    row = arm_metrics[arm][role]
                    require(
                        all(
                            row[field] == native[field]
                            for field in (
                                "scene", "episode", "pair_id", "query_id",
                                "analysis_role", "seed", "shared_A_frames",
                                "shared_A_decision_frames", "geodesic_m",
                            )
                        ),
                        f"{protocol}/{index}/{role}/{arm}: paired identity "
                        "changed",
                    )
                    outcomes[arm] = bool(int(row["reached"]))
                    per_arm[arm] = {
                        "success": outcomes[arm],
                        "path_len_m": float(row["path_len_m"]),
                        "steps": int(row["steps"]),
                        "final_goal_dist_m": float(
                            row["final_goal_dist_m"]
                        ),
                        "termination_reason": row["termination_reason"],
                        "router_active_plans": int(
                            row["router_active_plans"]
                        ),
                        "certificate_accept_plans": int(
                            row["certificate_accept_plans"]
                        ),
                        "adapter_takeover_plans": int(
                            row["adapter_takeover_plans"]
                        ),
                        "runtime_failure_plans": int(
                            row["runtime_failure_plans"]
                        ),
                    }
                learned_diagnostics = None
                if include_learned_pi3x:
                    learned_diagnostics = _learned_query_diagnostics(
                        native_row=native,
                        learned_row=arm_metrics[
                            "learned_pi3x_spatial"
                        ][role],
                        native_payload=arm_payloads["native"][role],
                        learned_payload=arm_payloads[
                            "learned_pi3x_spatial"
                        ][role],
                    )
                    per_arm["learned_pi3x_spatial"].update({
                        key: learned_diagnostics[key]
                        for key in (
                            "request_plans", "initial_inference_plans",
                            "accept_plans", "takeover_plans",
                            "runtime_failure_plans", "initial_accept",
                            "accept_takeover_match",
                            "sticky_rejection_violation",
                            "post_accept_abstention_plans",
                            "fully_abstained", "exact_native_fallback",
                            "accepted_bearing_errors_deg",
                        )
                    })
                    if include_in_analysis:
                        latency["learned_first_query_ms"].extend(
                            learned_diagnostics["first_query_latency_ms"]
                        )
                        latency["learned_one_anchor_update_ms"].extend(
                            learned_diagnostics[
                                "one_anchor_update_latency_ms"
                            ]
                        )
                        latency[
                            "learned_peak_gpu_memory_allocated_bytes"
                        ].extend(
                            learned_diagnostics[
                                "peak_gpu_memory_allocated_bytes"
                            ]
                        )
                        latency[
                            "learned_stored_online_history_frames"
                        ].append(float(native["shared_A_frames"]))
                records.append({
                    "protocol": protocol,
                    "scene": scene,
                    "episode": episode,
                    "pair_id": native["pair_id"],
                    "query_id": native["query_id"],
                    "role": role,
                    "geodesic_m": float(native["geodesic_m"]),
                    "outcomes": outcomes,
                    "arms": per_arm,
                })

                certified_plans = arm_payloads["certified"][role][
                    "query_leg"
                ]
                if include_in_analysis:
                    # ``uncached_relocalization_ms`` remains embedded in every
                    # later cache-hit response because it is part of the frozen
                    # first-query result.  Do not therefore collect timing by
                    # key presence alone: that repeats one expensive
                    # localization once per navigation replan and makes the
                    # distribution replan-weighted.  Split by the explicit
                    # cache lifecycle flag instead.
                    certificate_first_queries = [
                        plan for plan in certified_plans
                        if plan.get("certified_relocalization_cached") is False
                    ]
                    certificate_cached_updates = [
                        plan for plan in certified_plans
                        if plan.get("certified_relocalization_cached") is True
                    ]
                    latency["certificate_cached_ms"].extend(
                        finite_values(
                            certificate_cached_updates,
                            "certified_relocalization_ms",
                        )
                    )
                    latency["certificate_uncached_ms"].extend(
                        finite_values(
                            certificate_first_queries,
                            "certified_relocalization_uncached_ms",
                        )
                    )
                    for plan in certified_plans:
                        reason = plan.get(
                            "certified_relocalization_reason"
                        )
                        if (
                            reason
                            and plan.get(
                                "certified_relocalization_accepted"
                            ) is not True
                        ):
                            rejections[str(reason)] += 1
                    latency["geometry_verification_ms"].extend(
                        finite_values(
                            arm_payloads["geometry_fixed"][role][
                                "query_leg"
                            ],
                            "router_verification_total_ms",
                        )
                    )

        require(
            len(records) == 2 * expected_population,
            "query population incomplete",
        )
        require(
            len({row["scene"] for row in records}) == expected_scenes,
            "scene population differs",
        )
        records_by_protocol[protocol] = [
            record for record in records
            if record["scene"] not in excluded_scenes
        ]
        require(
            bool(records_by_protocol[protocol]),
            "analysis population is empty after scene exclusion",
        )
        latency_by_protocol[protocol] = latency
        rejections_by_protocol[protocol] = rejections

    if include_learned_pi3x:
        require(
            len(learned_artifact_identities) == 1,
            "learned artifact identity differs across paired episodes",
        )

    protocol_results = {}
    for protocol, records in records_by_protocol.items():
        metrics = {}
        for arm in arms:
            metrics[arm] = {}
            for role_filter in ("all", *ROLES):
                subset = (
                    records if role_filter == "all" else
                    [r for r in records if r["role"] == role_filter]
                )
                successes = sum(r["outcomes"][arm] for r in subset)
                spl_values = []
                for record in subset:
                    item = dict(record["arms"][arm])
                    item["geodesic_m"] = record["geodesic_m"]
                    spl_values.append(spl(item))
                metrics[arm][role_filter] = {
                    "n": len(subset),
                    "successes": successes,
                    "SR": successes / len(subset),
                    "SPL": float(np.mean(spl_values)),
                }
        contrasts = {}
        for baseline in ("native", "raw_fixed_bearing", "geometry_fixed"):
            key = f"certified_minus_{baseline}"
            contrasts[key] = {
                role_filter: contrast(
                    records if role_filter == "all" else [
                        r for r in records if r["role"] == role_filter
                    ],
                    baseline,
                    "certified",
                )
                for role_filter in ("all", *ROLES)
            }
        if include_learned_pi3x:
            for baseline in ("native", "certified"):
                key = f"learned_pi3x_spatial_minus_{baseline}"
                contrasts[key] = {
                    role_filter: contrast(
                        records if role_filter == "all" else [
                            r for r in records
                            if r["role"] == role_filter
                        ],
                        baseline,
                        "learned_pi3x_spatial",
                    )
                    for role_filter in ("all", *ROLES)
                }

        certified_novel = [r for r in records if r["role"] == "novel"]
        rejected_novel = [
            r for r in certified_novel
            if r["arms"]["certified"]["certificate_accept_plans"] == 0
            and r["arms"]["certified"]["adapter_takeover_plans"] == 0
        ]
        exact_rejected = 0
        for record in rejected_novel:
            native = record["arms"]["native"]
            certified = record["arms"]["certified"]
            exact = (
                native["success"] == certified["success"]
                and native["steps"] == certified["steps"]
                and native["termination_reason"]
                == certified["termination_reason"]
                and math.isclose(
                    native["path_len_m"], certified["path_len_m"],
                    abs_tol=1e-9,
                )
                and math.isclose(
                    native["final_goal_dist_m"],
                    certified["final_goal_dist_m"],
                    abs_tol=1e-9,
                )
            )
            exact_rejected += int(exact)

        protocol_result: dict[str, Any] = {
            "records": len(records),
            "scenes": len({row["scene"] for row in records}),
            "analysis_roles": layout["expected"][protocol][
                "analysis_roles"
            ],
            "duplicated_novel_is_instrumentation_only": (
                protocol == "hard_support"
            ),
            "metrics": metrics,
            "contrasts": contrasts,
            "certificate_safety": {
                "novel_queries": len(certified_novel),
                "novel_accept_episodes": sum(
                    r["arms"]["certified"]["certificate_accept_plans"] > 0
                    for r in certified_novel
                ),
                "novel_takeover_episodes": sum(
                    r["arms"]["certified"]["adapter_takeover_plans"] > 0
                    for r in certified_novel
                ),
                "fully_rejected_novel_queries": len(rejected_novel),
                "fully_rejected_exact_native_queries": exact_rejected,
                "revisit_activation_queries": sum(
                    r["arms"]["certified"]["adapter_takeover_plans"] > 0
                    for r in records if r["role"] == "revisit"
                ),
                "runtime_failure_plans": sum(
                    r["arms"]["certified"]["runtime_failure_plans"]
                    for r in records
                ),
                "rejection_reasons": dict(sorted(
                    rejections_by_protocol[protocol].items()
                )),
            },
            "latency": {
                key: distribution(values)
                for key, values in latency_by_protocol[protocol].items()
                if include_learned_pi3x or not key.startswith("learned_")
            },
        }
        if include_learned_pi3x:
            learned_records = [
                r["arms"]["learned_pi3x_spatial"] for r in records
            ]
            learned_novel_records = [
                r for r in records if r["role"] == "novel"
            ]
            learned_revisit_records = [
                r for r in records if r["role"] == "revisit"
            ]
            all_errors = [
                error
                for item in learned_records
                for error in item["accepted_bearing_errors_deg"]
            ]
            abstained = [
                item for item in learned_records
                if item["fully_abstained"]
            ]
            protocol_result["learned_pi3x_safety"] = {
                "queries": len(learned_records),
                "initial_inference_contract_violations": sum(
                    item["initial_inference_plans"] != 1
                    for item in learned_records
                ),
                "accept_takeover_mismatch_queries": sum(
                    not item["accept_takeover_match"]
                    for item in learned_records
                ),
                "sticky_rejection_violations": sum(
                    item["sticky_rejection_violation"]
                    for item in learned_records
                ),
                "post_accept_abstention_plans": sum(
                    item["post_accept_abstention_plans"]
                    for item in learned_records
                ),
                "runtime_failure_plans": sum(
                    item["runtime_failure_plans"]
                    for item in learned_records
                ),
                "fully_abstained_queries": len(abstained),
                "fully_abstained_exact_native_queries": sum(
                    item["exact_native_fallback"]["exact"]
                    for item in abstained
                ),
                "accepted_bearing_error": distribution(all_errors),
                "accepted_bearing_errors_over_90_deg": sum(
                    error > 90.0 for error in all_errors
                ),
                "novel_queries": len(learned_novel_records),
                "novel_accept_queries": sum(
                    r["arms"]["learned_pi3x_spatial"]["accept_plans"] > 0
                    for r in learned_novel_records
                ),
                "novel_takeover_queries": sum(
                    r["arms"]["learned_pi3x_spatial"]["takeover_plans"] > 0
                    for r in learned_novel_records
                ),
                "novel_takeover_identities": [
                    [r["scene"], r["episode"], r["role"]]
                    for r in learned_novel_records
                    if r["arms"]["learned_pi3x_spatial"][
                        "takeover_plans"
                    ] > 0
                ],
                "revisit_queries": len(learned_revisit_records),
                "revisit_activation_queries": sum(
                    r["arms"]["learned_pi3x_spatial"]["takeover_plans"] > 0
                    for r in learned_revisit_records
                ),
                "successful_takeover_queries": sum(
                    r["outcomes"]["learned_pi3x_spatial"]
                    and r["arms"]["learned_pi3x_spatial"][
                        "takeover_plans"
                    ] > 0
                    for r in records
                ),
                "successful_fallback_queries": sum(
                    r["outcomes"]["learned_pi3x_spatial"]
                    and r["arms"]["learned_pi3x_spatial"][
                        "fully_abstained"
                    ]
                    for r in records
                ),
            }
        protocol_results[protocol] = protocol_result

    primary_protocol = str(layout["primary_protocol"])
    histories = len(records_by_protocol[primary_protocol]) // 2
    scenes = len({
        row["scene"] for row in records_by_protocol[primary_protocol]
    })
    primary_expected = layout["expected"][primary_protocol]
    target_histories = int(primary_expected["target_histories"])
    target_scenes = int(primary_expected["target_scenes"])
    target_met = (
        histories >= target_histories and scenes >= target_scenes
    )
    protocol_populations = {}
    for protocol in protocols:
        protocol_histories = len(records_by_protocol[protocol]) // 2
        protocol_scenes = len({
            row["scene"] for row in records_by_protocol[protocol]
        })
        declared = layout["expected"][protocol]
        protocol_target_met = (
            protocol_histories >= int(declared["target_histories"])
            and protocol_scenes >= int(declared["target_scenes"])
        )
        protocol_populations[protocol] = {
            "histories": protocol_histories,
            "scenes": protocol_scenes,
            "full_histories": int(declared["histories"]),
            "full_scenes": int(declared["scenes"]),
            "target_histories": int(declared["target_histories"]),
            "target_scenes": int(declared["target_scenes"]),
            "target_met": protocol_target_met,
            "underpowered_if_target_not_met": not protocol_target_met,
            "analysis_roles": list(declared["analysis_roles"]),
        }
    all_population_targets_met = all(
        row["target_met"] for row in protocol_populations.values()
    )
    learned_qualification = None
    learned_identity = None
    if include_learned_pi3x:
        learned_identity = {
            "model_sha256": next(iter(learned_artifact_identities))[0],
            "proof_manifest_sha256": next(
                iter(learned_artifact_identities)
            )[1],
        }
        natural = protocol_results["natural_direction"]
        l1_contrast = natural["contrasts"][
            "learned_pi3x_spatial_minus_native"
        ]["revisit"]
        l2_contrast = natural["contrasts"][
            "learned_pi3x_spatial_minus_certified"
        ]["revisit"]
        novel_contrast = natural["contrasts"][
            "learned_pi3x_spatial_minus_native"
        ]["novel"]
        safety = natural["learned_pi3x_safety"]
        l1_pass = (
            l1_contrast["risk_difference_pp"] > 0.0
            and l1_contrast[
                "scene_cluster_bootstrap_risk_difference_95"
            ][0] > 0.0
        )
        l2_pass = (
            l2_contrast["risk_difference_pp"] >= -5.0
            and l2_contrast[
                "scene_cluster_bootstrap_risk_difference_95"
            ][0] > -10.0
        )
        l3_pass = (
            safety["runtime_failure_plans"] == 0
            and safety["initial_inference_contract_violations"] == 0
            and safety["accept_takeover_mismatch_queries"] == 0
            and safety["sticky_rejection_violations"] == 0
            and safety["post_accept_abstention_plans"] == 0
            and safety["fully_abstained_queries"]
            == safety["fully_abstained_exact_native_queries"]
            and novel_contrast["gains"] >= novel_contrast["losses"]
            and safety["accepted_bearing_errors_over_90_deg"] == 0
        )
        learned_qualification = {
            "evaluation_protocol": "natural_direction",
            "L1_useful_revisit_control": {
                "pass": l1_pass,
                "positive_net_gain": (
                    l1_contrast["gains"] > l1_contrast["losses"]
                ),
                "scene_cluster_ci_lower_above_zero": (
                    l1_contrast[
                        "scene_cluster_bootstrap_risk_difference_95"
                    ][0] > 0.0
                ),
            },
            "L2_noninferior_to_cec": {
                "pass": l2_pass,
                "margin_pp": -10.0,
                "point_estimate_floor_pp": -5.0,
            },
            "L3_novel_safety_and_exact_fallback": {
                "pass": l3_pass,
                "novel_net_nonnegative": (
                    novel_contrast["gains"] >= novel_contrast["losses"]
                ),
                "all_fully_abstained_queries_exact": (
                    safety["fully_abstained_queries"]
                    == safety["fully_abstained_exact_native_queries"]
                ),
            },
            "all_primary_gates_pass": l1_pass and l2_pass and l3_pass,
            "population_target_met": all_population_targets_met,
            "eligible_for_primary_method_promotion": (
                all_population_targets_met and l1_pass and l2_pass and l3_pass
            ),
        }

    return {
        "schema_version": (
            "paper_role_pair_evaluation_summary_v2_learned_20260817"
            if include_learned_pi3x else
            "paper_role_pair_evaluation_summary_v1_20260814"
        ),
        "include_learned_pi3x": include_learned_pi3x,
        "learned_pi3x_artifacts": learned_identity,
        "benchmark_files_receipt_sha256": sha256_file(receipt_file),
        "population_receipt_sha256": sha256_file(
            benchmark_root / "population_receipt.json"
        ),
        "completion_receipt_aggregate_sha256": hashlib.sha256(
            "\n".join(sorted(completion_hashes)).encode()
        ).hexdigest(),
        "population": {
            "layout": layout["mode"],
            "primary_protocol": primary_protocol,
            "histories": histories,
            "scenes": scenes,
            "full_histories": int(primary_expected["histories"]),
            "full_scenes": int(primary_expected["scenes"]),
            "excluded_scenes": list(excluded_scenes),
            "target_histories": target_histories,
            "target_scenes": target_scenes,
            "target_met": target_met,
            "underpowered_if_target_not_met": not all_population_targets_met,
            "all_protocol_targets_met": all_population_targets_met,
            "by_protocol": protocol_populations,
        },
        "arms": list(arms),
        "protocols": protocol_results,
        "learned_pi3x_qualification": learned_qualification,
        "all_required_outputs_complete": True,
        "runtime_role_visibility": "none",
        "statistical_unit": "paired query; scene-cluster uncertainty",
        "bootstrap_resamples": 100_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--include-learned-pi3x", action="store_true")
    args = parser.parse_args()
    result = summarize(
        args.root,
        excluded_scenes=tuple(sorted(set(args.exclude_scene))),
        include_learned_pi3x=args.include_learned_pi3x,
    )
    encoded = json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded)
    output = {
        "population": result["population"],
        "natural_certified": result["protocols"][
            "natural_direction"
        ]["metrics"]["certified"],
        "complete": result["all_required_outputs_complete"],
    }
    if args.include_learned_pi3x:
        output["natural_learned_pi3x_spatial"] = result["protocols"][
            "natural_direction"
        ]["metrics"]["learned_pi3x_spatial"]
        output["learned_qualification"] = result[
            "learned_pi3x_qualification"
        ]
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
