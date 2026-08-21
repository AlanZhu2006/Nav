#!/usr/bin/env python3
"""Independent raw-output verification for the paper role-pair run.

This reader deliberately does not import the primary summarizer.  It derives
counts, paired effects, learned lifecycle, GT bearing safety, and exact native
fallback directly from metric CSVs, plan payloads, and rollout receipts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_PROTOCOLS = ("support_controlled", "natural_direction")
FINAL14_PROTOCOLS = ("natural_direction", "hard_support")
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
PLAN_KEYS = (
    "step", "requested_diffusion_seed", "diffusion_seed",
    "server_selected_idx", "trajectory_candidate_count",
    "selected_trajectory_sha256",
)
METRIC_KEYS = (
    "reached", "steps", "path_len_m", "final_goal_dist_m",
    "termination_reason",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def independent_population_layout(population: dict[str, Any]) -> dict[str, Any]:
    """Decode populations independently of the primary summarizer."""

    if (
        population.get("schema_version")
        == "final14_role_pair_population_v1_20260817"
    ):
        declared = population["populations"]
        return {
            "protocols": FINAL14_PROTOCOLS,
            "primary": "natural_direction",
            "expected": {
                "natural_direction": {
                    "histories": int(declared["natural_standard"]["histories"]),
                    "scenes": int(declared["natural_standard"]["scenes"]),
                    "target_met": bool(declared["natural_standard"]["target_met"]),
                },
                "hard_support": {
                    "histories": int(declared["hard_support"]["histories"]),
                    "scenes": int(declared["hard_support"]["scenes"]),
                    "target_met": bool(declared["hard_support"]["target_met"]),
                },
            },
        }
    histories = int(population["role_pair_constructible_histories"])
    scenes = int(population["role_pair_scene_count"])
    return {
        "protocols": LEGACY_PROTOCOLS,
        "primary": LEGACY_PROTOCOLS[0],
        "expected": {
            protocol: {
                "histories": histories,
                "scenes": scenes,
                "target_met": bool(population.get("target_met", False)),
            }
            for protocol in LEGACY_PROTOCOLS
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    count = gains + losses
    if count == 0:
        return 1.0
    probability = 2.0 * sum(
        math.comb(count, index)
        for index in range(min(gains, losses) + 1)
    ) / (2**count)
    return min(1.0, probability)


def cluster_interval(
    records: list[dict[str, Any]], left: str, right: str
) -> list[float]:
    scenes = sorted({str(row["scene"]) for row in records})
    require(bool(scenes), "empty cluster population")
    grouped = {
        scene: [row for row in records if row["scene"] == scene]
        for scene in scenes
    }
    rng = np.random.default_rng(20260814)
    draws = np.empty(100_000, dtype=np.float64)
    for index in range(draws.size):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        numerator = 0
        denominator = 0
        for scene in selected:
            rows = grouped[str(scene)]
            numerator += sum(
                int(row["outcomes"][right])
                - int(row["outcomes"][left])
                for row in rows
            )
            denominator += len(rows)
        draws[index] = 100.0 * numerator / denominator
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def exact_fallback(
    native_row: dict[str, str],
    learned_row: dict[str, str],
    native_payload: dict[str, Any],
    learned_payload: dict[str, Any],
) -> bool:
    left = native_payload["query_leg"]
    right = learned_payload["query_leg"]
    plans_equal = (
        len(left) == len(right)
        and all(
            a.get(key) == b.get(key)
            for a, b in zip(left, right)
            for key in PLAN_KEYS
        )
    )
    rollout_equal = (
        native_payload["rollout_traces"]["query"]
        == learned_payload["rollout_traces"]["query"]
    )
    metrics_equal = all(
        native_row[key] == learned_row[key] for key in METRIC_KEYS
    )
    return plans_equal and rollout_equal and metrics_equal


def learned_diagnostics(
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
    accept_count = int(learned_row["learned_pi3x_accept_plans"])
    takeover_count = int(learned_row["adapter_takeover_plans"])
    initial_count = int(
        learned_row["learned_pi3x_initial_inference_plans"]
    )
    require(len(accepted) == accept_count, "raw learned accepts differ")
    require(len(initial) == initial_count, "raw learned initial count differs")
    errors = []
    for plan in accepted:
        value = plan.get(
            "learned_pi3x_evaluation_gt_bearing_error_deg"
        )
        require(
            value is not None and math.isfinite(float(value)),
            "accepted learned plan has no finite bearing error",
        )
        error = float(value)
        require(0.0 <= error <= 180.0, "bearing error outside [0,180]")
        errors.append(error)
    initial_accept = (
        len(initial) == 1
        and initial[0].get("learned_pi3x_relocalization_accepted") is True
    )
    fully_abstained = accept_count == 0 and takeover_count == 0
    return {
        "initial_count": initial_count,
        "accept_count": accept_count,
        "takeover_count": takeover_count,
        "runtime_failures": int(learned_row["runtime_failure_plans"]),
        "accept_takeover_match": accept_count == takeover_count,
        "sticky_rejection_violation": bool(
            len(initial) == 1 and not initial_accept and accepted
        ),
        "post_accept_abstentions": (
            sum(
                plan.get("learned_pi3x_relocalization_accepted") is not True
                for plan in requests
            )
            if initial_accept else 0
        ),
        "fully_abstained": fully_abstained,
        "exact_fallback": (
            exact_fallback(
                native_row, learned_row, native_payload, learned_payload
            )
            if fully_abstained else None
        ),
        "errors": errors,
    }


def verify(
    root: Path,
    summary_path: Path,
    *,
    excluded_scenes: tuple[str, ...] = (),
    include_learned_pi3x: bool = False,
) -> dict[str, Any]:
    arms = (
        LEARNED_AMENDMENT_ARMS
        if include_learned_pi3x else PARENT_ARMS
    )
    summary = json.loads(summary_path.read_text())
    require(tuple(summary["arms"]) == arms, "summary arm set differs")
    require(
        bool(summary.get("include_learned_pi3x", False))
        == include_learned_pi3x,
        "summary learned mode differs",
    )
    population = json.loads(
        (root / "benchmarks/population_receipt.json").read_text()
    )
    layout = independent_population_layout(population)
    protocols = tuple(layout["protocols"])
    primary_protocol = str(layout["primary"])
    n_histories = int(layout["expected"][primary_protocol]["histories"])
    require(
        summary["population"].get("full_histories", n_histories)
        == n_histories,
        "full population differs",
    )
    require(
        summary["population"].get("excluded_scenes", [])
        == list(excluded_scenes),
        "scene exclusion differs",
    )
    checked = []
    all_records: dict[str, list[dict[str, Any]]] = {}
    learned_identities: set[tuple[str, str]] = set()
    learned_raw_by_protocol: dict[str, list[dict[str, Any]]] = {}
    for protocol in protocols:
        protocol_histories = int(layout["expected"][protocol]["histories"])
        manifest = json.loads(
            (root / f"benchmarks/{protocol}/manifest.json").read_text()
        )
        require(
            len(manifest["episodes"]) == protocol_histories,
            "protocol population differs",
        )
        records = []
        learned_raw = []
        for index, source in enumerate(manifest["episodes"]):
            scene = str(source["scene"])
            episode = str(source["episode"])
            episode_root = (
                root / f"evaluation/{protocol}/{index:03d}_{scene}_{episode}"
            )
            receipt = episode_root / "completion.json"
            require(
                sha256_file(receipt)
                == (episode_root / "completion.json.sha256")
                .read_text().split()[0],
                "completion receipt changed",
            )
            completion = json.loads(receipt.read_text())
            if include_learned_pi3x:
                contract = json.loads(
                    (episode_root / "episode_contract.json").read_text()
                )
                require(
                    tuple(contract.get("arms") or ()) == arms,
                    "episode learned arm set differs",
                )
                declared = contract.get("learned_pi3x") or {}
                completed = completion.get("learned_pi3x") or {}
                identity = (
                    str(declared.get("model_sha256")),
                    str(declared.get("proof_manifest_sha256")),
                )
                require(
                    identity == (
                        str(completed.get("model_sha256")),
                        str(completed.get("proof_manifest_sha256")),
                    ),
                    "learned artifact identity differs",
                )
                learned_identities.add(identity)
            arm_rows = {}
            arm_payloads = {}
            for arm in arms:
                with (episode_root / arm / "metric.csv").open(
                    newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                require(len(rows) == 2, "metric population differs")
                arm_rows[arm] = {
                    row["analysis_role"]: row for row in rows
                }
                arm_payloads[arm] = {}
                for role, row in arm_rows[arm].items():
                    payload = json.loads((
                        episode_root / arm
                        / f"{episode}_{row['query_id']}_plans.json"
                    ).read_text())
                    require(
                        payload["analysis_role_not_forwarded"] is True,
                        "analysis role reached runtime",
                    )
                    arm_payloads[arm][role] = payload
            for role in ROLES:
                reference = arm_rows["native"][role]
                outcomes = {
                    arm: bool(int(arm_rows[arm][role]["reached"]))
                    for arm in arms
                }
                for arm in arms:
                    row = arm_rows[arm][role]
                    require(
                        (
                            row["scene"], row["episode"], row["pair_id"],
                            row["query_id"], row["analysis_role"],
                            row["seed"], row["geodesic_m"],
                        ) == (
                            reference["scene"], reference["episode"],
                            reference["pair_id"], reference["query_id"],
                            reference["analysis_role"], reference["seed"],
                            reference["geodesic_m"],
                        ),
                        "paired identity differs",
                    )
                record = {
                    "scene": scene,
                    "episode": episode,
                    "role": role,
                    "outcomes": outcomes,
                }
                if include_learned_pi3x:
                    diagnostic = learned_diagnostics(
                        reference,
                        arm_rows["learned_pi3x_spatial"][role],
                        arm_payloads["native"][role],
                        arm_payloads["learned_pi3x_spatial"][role],
                    )
                    record["learned"] = diagnostic
                    learned_raw.append(record)
                records.append(record)
        require(
            len(records) == 2 * protocol_histories,
            "record count differs",
        )
        records = [
            row for row in records
            if row["scene"] not in excluded_scenes
        ]
        require(
            bool(records),
            "verification population is empty after exclusion",
        )
        if include_learned_pi3x:
            learned_raw = [
                row for row in learned_raw
                if row["scene"] not in excluded_scenes
            ]
        all_records[protocol] = records
        learned_raw_by_protocol[protocol] = learned_raw

        reported = summary["protocols"][protocol]
        for arm in arms:
            for role_filter in ("all", *ROLES):
                subset = (
                    records if role_filter == "all" else
                    [r for r in records if r["role"] == role_filter]
                )
                successes = sum(
                    row["outcomes"][arm] for row in subset
                )
                metric = reported["metrics"][arm][role_filter]
                require(metric["n"] == len(subset), "reported N differs")
                require(
                    metric["successes"] == successes,
                    "reported successes differ",
                )
                require(
                    math.isclose(
                        metric["SR"], successes / len(subset),
                        abs_tol=1e-15,
                    ),
                    "reported SR differs",
                )
        contrast_specs = [
            (f"certified_minus_{baseline}", baseline, "certified")
            for baseline in (
                "native", "raw_fixed_bearing", "geometry_fixed"
            )
        ]
        if include_learned_pi3x:
            contrast_specs.extend([
                (
                    "learned_pi3x_spatial_minus_native",
                    "native", "learned_pi3x_spatial",
                ),
                (
                    "learned_pi3x_spatial_minus_certified",
                    "certified", "learned_pi3x_spatial",
                ),
            ])
        for name, left, right in contrast_specs:
            for role_filter in ("all", *ROLES):
                subset = (
                    records if role_filter == "all" else
                    [r for r in records if r["role"] == role_filter]
                )
                gains = sum(
                    r["outcomes"][right]
                    and not r["outcomes"][left]
                    for r in subset
                )
                losses = sum(
                    r["outcomes"][left]
                    and not r["outcomes"][right]
                    for r in subset
                )
                item = reported["contrasts"][name][role_filter]
                require(
                    item["gains"] == gains and item["losses"] == losses,
                    "discordance differs",
                )
                require(
                    math.isclose(
                        item["exact_mcnemar_two_sided_p"],
                        exact_mcnemar(gains, losses),
                        abs_tol=1e-15,
                    ),
                    "McNemar p differs",
                )
                expected_ci = cluster_interval(subset, left, right)
                require(
                    np.allclose(
                        item[
                            "scene_cluster_bootstrap_risk_difference_95"
                        ],
                        expected_ci,
                        atol=1e-12,
                    ),
                    "cluster interval differs",
                )
                checked.append(f"{protocol}:{name}:{role_filter}")

        if include_learned_pi3x:
            raw = learned_raw_by_protocol[protocol]
            reported_safety = reported["learned_pi3x_safety"]
            errors = [
                error for row in raw for error in row["learned"]["errors"]
            ]
            abstained = [
                row for row in raw if row["learned"]["fully_abstained"]
            ]
            expected = {
                "queries": len(raw),
                "initial_inference_contract_violations": sum(
                    row["learned"]["initial_count"] != 1 for row in raw
                ),
                "accept_takeover_mismatch_queries": sum(
                    not row["learned"]["accept_takeover_match"]
                    for row in raw
                ),
                "sticky_rejection_violations": sum(
                    row["learned"]["sticky_rejection_violation"]
                    for row in raw
                ),
                "post_accept_abstention_plans": sum(
                    row["learned"]["post_accept_abstentions"]
                    for row in raw
                ),
                "runtime_failure_plans": sum(
                    row["learned"]["runtime_failures"] for row in raw
                ),
                "fully_abstained_queries": len(abstained),
                "fully_abstained_exact_native_queries": sum(
                    row["learned"]["exact_fallback"] for row in abstained
                ),
                "accepted_bearing_errors_over_90_deg": sum(
                    error > 90.0 for error in errors
                ),
                "novel_queries": sum(row["role"] == "novel" for row in raw),
                "novel_accept_queries": sum(
                    row["role"] == "novel"
                    and row["learned"]["accept_count"] > 0
                    for row in raw
                ),
                "novel_takeover_queries": sum(
                    row["role"] == "novel"
                    and row["learned"]["takeover_count"] > 0
                    for row in raw
                ),
                "revisit_queries": sum(
                    row["role"] == "revisit" for row in raw
                ),
                "revisit_activation_queries": sum(
                    row["role"] == "revisit"
                    and row["learned"]["takeover_count"] > 0
                    for row in raw
                ),
            }
            for key, value in expected.items():
                require(
                    reported_safety[key] == value,
                    f"reported learned safety differs: {key}",
                )
            reported_error = reported_safety["accepted_bearing_error"]
            require(
                reported_error["n"] == len(errors),
                "reported bearing-error count differs",
            )
            require(
                (
                    reported_error["maximum"] is None and not errors
                ) or math.isclose(
                    float(reported_error["maximum"]), max(errors),
                    abs_tol=1e-12,
                ),
                "reported maximum bearing error differs",
            )

    analysis_histories = len(all_records[primary_protocol]) // 2
    require(
        summary["population"]["histories"] == analysis_histories,
        "analysis population differs",
    )
    analysis_histories_by_protocol = {
        protocol: len(all_records[protocol]) // 2 for protocol in protocols
    }
    reported_by_protocol = summary["population"].get("by_protocol")
    if reported_by_protocol is not None:
        for protocol in protocols:
            require(
                int(reported_by_protocol[protocol]["histories"])
                == analysis_histories_by_protocol[protocol],
                f"{protocol} analysis population differs",
            )
    learned_gate_checks = None
    if include_learned_pi3x:
        require(
            len(learned_identities) == 1,
            "learned artifact identity changed across episodes",
        )
        identity = next(iter(learned_identities))
        require(
            summary["learned_pi3x_artifacts"] == {
                "model_sha256": identity[0],
                "proof_manifest_sha256": identity[1],
            },
            "summary learned identity differs",
        )
        natural = summary["protocols"]["natural_direction"]
        l1 = natural["contrasts"][
            "learned_pi3x_spatial_minus_native"
        ]["revisit"]
        l2 = natural["contrasts"][
            "learned_pi3x_spatial_minus_certified"
        ]["revisit"]
        novel = natural["contrasts"][
            "learned_pi3x_spatial_minus_native"
        ]["novel"]
        safety = natural["learned_pi3x_safety"]
        l1_pass = (
            l1["risk_difference_pp"] > 0.0
            and l1["scene_cluster_bootstrap_risk_difference_95"][0] > 0.0
        )
        l2_pass = (
            l2["risk_difference_pp"] >= -5.0
            and l2["scene_cluster_bootstrap_risk_difference_95"][0] > -10.0
        )
        l3_pass = (
            safety["runtime_failure_plans"] == 0
            and safety["initial_inference_contract_violations"] == 0
            and safety["accept_takeover_mismatch_queries"] == 0
            and safety["sticky_rejection_violations"] == 0
            and safety["post_accept_abstention_plans"] == 0
            and safety["fully_abstained_queries"]
            == safety["fully_abstained_exact_native_queries"]
            and novel["gains"] >= novel["losses"]
            and safety["accepted_bearing_errors_over_90_deg"] == 0
        )
        qualification = summary["learned_pi3x_qualification"]
        require(
            qualification["L1_useful_revisit_control"]["pass"] == l1_pass,
            "L1 gate differs",
        )
        require(
            qualification["L2_noninferior_to_cec"]["pass"] == l2_pass,
            "L2 gate differs",
        )
        require(
            qualification[
                "L3_novel_safety_and_exact_fallback"
            ]["pass"] == l3_pass,
            "L3 gate differs",
        )
        require(
            qualification["all_primary_gates_pass"]
            == (l1_pass and l2_pass and l3_pass),
            "combined learned gate differs",
        )
        expected_population_gate = (
            all(
                bool(row["target_met"])
                for row in reported_by_protocol.values()
            )
            if reported_by_protocol is not None
            else bool(summary["population"]["target_met"])
        )
        require(
            qualification["population_target_met"]
            == expected_population_gate,
            "learned population gate differs",
        )
        learned_gate_checks = {
            "L1": l1_pass,
            "L2": l2_pass,
            "L3": l3_pass,
        }

    return {
        "schema_version": (
            "paper_role_pair_independent_verification_v2_learned_20260817"
            if include_learned_pi3x else
            "paper_role_pair_independent_verification_v1_20260814"
        ),
        "verified": True,
        "include_learned_pi3x": include_learned_pi3x,
        "summary_sha256": sha256_file(summary_path),
        "population_receipt_sha256": sha256_file(
            root / "benchmarks/population_receipt.json"
        ),
        "histories": n_histories,
        "analysis_histories": analysis_histories,
        "analysis_histories_by_protocol": analysis_histories_by_protocol,
        "excluded_scenes": list(excluded_scenes),
        "queries_per_protocol": 2 * analysis_histories,
        "queries_by_protocol": {
            protocol: 2 * count
            for protocol, count in analysis_histories_by_protocol.items()
        },
        "checked_contrasts": checked,
        "learned_gate_checks": learned_gate_checks,
        "source_reader": (
            "raw metric.csv, plan payloads, rollout traces, episode contracts "
            "and completion receipts; no summarizer import"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--include-learned-pi3x", action="store_true")
    args = parser.parse_args()
    result = verify(
        args.root,
        args.summary,
        excluded_scenes=tuple(sorted(set(args.exclude_scene))),
        include_learned_pi3x=args.include_learned_pi3x,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.write_text(encoded)
    print(json.dumps({
        "verified": True,
        "histories": result["histories"],
        "learned_gate_checks": result["learned_gate_checks"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
