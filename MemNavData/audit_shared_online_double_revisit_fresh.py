#!/usr/bin/env python3
"""Independent outcome audit for the fresh160 double-Revisit four-arm gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SCHEMA = "shared_online_double_revisit_closed_loop_v3_leg_scope_20260813"
EXPECTED_VARIANT = "v1_controlled_pose_perturbation"
EXPECTED_ARMS = {
    "native": {
        "server_backend": "navdp",
        "hybrid_route": "phase",
        "scope": "both",
        "backends": {"B": None, "C": None},
        "C_long_memory_enabled": False,
    },
    "full_memory": {
        "server_backend": "hybrid_pose",
        "hybrid_route": "phase",
        "scope": "both",
        "backends": {"B": "navdp_mix", "C": "navdp_mix"},
        "C_long_memory_enabled": True,
    },
    "memory_b_native_c": {
        "server_backend": "hybrid_pose",
        "hybrid_route": "phase",
        "scope": "b_only",
        "backends": {"B": "navdp_mix", "C": "navdp"},
        "C_long_memory_enabled": False,
    },
    "certified": {
        "server_backend": "hybrid_pose",
        "hybrid_route": "certified_relocalization",
        "scope": "both",
        "backends": {"B": "navdp_auto", "C": "navdp_auto"},
        "C_long_memory_enabled": True,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False", ""):
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def read_single_metric(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"expected one metric row: {path}")
    return rows[0]


def exact_mcnemar(first: list[bool], second: list[bool]) -> dict[str, Any]:
    require(len(first) == len(second), "paired vectors differ in length")
    gains = sum(a and not b for a, b in zip(first, second))
    losses = sum(b and not a for a, b in zip(first, second))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(gains, losses) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "N": len(first),
        "first_success": sum(first),
        "second_success": sum(second),
        "risk_difference_pp": (
            100.0 * (sum(first) - sum(second)) / len(first) if first else None
        ),
        "gain": gains,
        "loss": losses,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
    }


def scene_cluster_bootstrap(
    rows: list[tuple[str, bool, bool]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    require(bool(rows), "cluster bootstrap population is empty")
    require(resamples >= 1000, "too few bootstrap resamples")
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene, first, second in rows:
        grouped[scene].append(float(first) - float(second))
    scenes = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(scenes), size=len(scenes))
        values = [value for item in sampled for value in grouped[scenes[item]]]
        draws[index] = float(np.mean(values))
    return {
        "clusters": len(scenes),
        "episodes": len(rows),
        "seed": seed,
        "resamples": resamples,
        "risk_difference_pp": 100.0
        * float(np.mean([float(a) - float(b) for _, a, b in rows])),
        "ci95_pp": [
            100.0 * float(np.quantile(draws, 0.025)),
            100.0 * float(np.quantile(draws, 0.975)),
        ],
    }


def audit(
    run_root: Path,
    *,
    expected_manifest_sha: str,
    expected_episodes: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    benchmark_root = run_root / "prepared" / "benchmark"
    manifest_path = benchmark_root / "manifest.json"
    require(sha256_file(manifest_path) == expected_manifest_sha,
            "benchmark manifest SHA changed")
    manifest = read_json(manifest_path)
    require(len(manifest.get("episodes", [])) == expected_episodes,
            "benchmark episode count changed")
    require(
        manifest.get("selection", {}).get("no_navigation_outcomes_observed") is True,
        "benchmark was not sealed before outcome generation",
    )
    frozen_identities = [
        (str(row["scene"]), str(row["episode"]))
        for row in manifest["episodes"]
    ]
    require(len(set(frozen_identities)) == expected_episodes,
            "benchmark contains duplicate identities")

    scenes_root = run_root / "scenes"
    episode_dirs = sorted(path for path in scenes_root.iterdir() if path.is_dir())
    require(len(episode_dirs) == expected_episodes,
            f"expected {expected_episodes} episode outputs, found {len(episode_dirs)}")

    records: list[dict[str, Any]] = []
    certificate_reasons: Counter[str] = Counter()
    certificate_requests = Counter()
    certificate_accepts = Counter()
    observed_identities = []

    for episode_dir in episode_dirs:
        contract = read_json(episode_dir / "episode_contract.json")
        index = int(contract["selection_index"])
        require(0 <= index < expected_episodes, f"bad selection index: {episode_dir}")
        expected_identity = frozen_identities[index]
        require(
            (contract["scene"], contract["episode"]) == expected_identity,
            f"selection identity changed: {episode_dir}",
        )
        require(contract["benchmark_manifest_sha256"] == expected_manifest_sha,
                f"episode contract manifest changed: {episode_dir}")
        observed_identities.append(expected_identity)
        arm_payloads: dict[str, dict[str, Any]] = {}
        identities = set()
        for arm, expected in EXPECTED_ARMS.items():
            arm_root = episode_dir / arm
            require(arm_root.is_dir(), f"missing arm {episode_dir.name}/{arm}")
            summary = read_json(arm_root / "summary.json")
            run_contract = read_json(arm_root / "run_contract.json")
            require(summary.get("schema_version") == EXPECTED_SCHEMA,
                    f"schema changed: {episode_dir.name}/{arm}")
            require(summary.get("benchmark_manifest_sha256") == expected_manifest_sha,
                    f"manifest changed: {episode_dir.name}/{arm}")
            require(summary.get("variant") == EXPECTED_VARIANT,
                    f"variant changed: {episode_dir.name}/{arm}")
            require(summary.get("deterministic_plan_seeds") is True,
                    f"non-deterministic seeds: {episode_dir.name}/{arm}")
            require(summary.get("shared_A_all_hashes_ok") is True,
                    f"shared-A hash failure: {episode_dir.name}/{arm}")
            require(summary.get("shared_A_total_diffusion_samples") == 0,
                    f"shared-A sampled diffusion: {episode_dir.name}/{arm}")
            require(summary.get("navdp_goal_switch_reset") == "before_c",
                    f"reset changed: {episode_dir.name}/{arm}")
            require(summary.get("server_backend") == expected["server_backend"],
                    f"server backend changed: {episode_dir.name}/{arm}")
            require(summary.get("hybrid_route") == expected["hybrid_route"],
                    f"route changed: {episode_dir.name}/{arm}")
            require(summary.get("known_revisit_scope") == expected["scope"],
                    f"leg scope changed: {episode_dir.name}/{arm}")
            require(summary.get("policy_backends") == expected["backends"],
                    f"policy backend changed: {episode_dir.name}/{arm}")
            require(summary.get("C_long_memory_enabled") is
                    expected["C_long_memory_enabled"],
                    f"C memory flag changed: {episode_dir.name}/{arm}")
            require(run_contract.get("C_history") == "initial_leg_only",
                    f"C history changed: {episode_dir.name}/{arm}")
            require(run_contract.get("C_candidate_ceiling") in
                    ("online_A_boundary", "disabled"),
                    f"C ceiling contract missing: {episode_dir.name}/{arm}")

            metric = read_single_metric(arm_root / "metric.csv")
            plan_files = sorted(arm_root.glob("episode_*_plans.json"))
            require(len(plan_files) == 1,
                    f"expected one plan receipt: {episode_dir.name}/{arm}")
            plans = read_json(plan_files[0])
            identities.add((metric["scene"], metric["episode"], int(metric["seed"])))
            require((metric["scene"], metric["episode"]) == expected_identity,
                    f"metric identity changed: {episode_dir.name}/{arm}")
            require(as_bool(metric["shared_A_hashes_ok"]),
                    f"metric shared-A hash failed: {episode_dir.name}/{arm}")
            require(int(metric["shared_A_replay_diffusion_samples"]) == 0,
                    f"metric shared-A sampled: {episode_dir.name}/{arm}")
            reached_b = as_bool(metric["reached_B"])
            c_input_ok = as_bool(metric["c_effective_input_contract_ok"])
            c_evaluated = as_bool(metric["C_evaluated"])
            require(c_evaluated == (reached_b and c_input_ok),
                    f"C censoring contract changed: {episode_dir.name}/{arm}")
            arm_payloads[arm] = {
                "summary": summary,
                "contract": run_contract,
                "metric": metric,
                "plans": plans,
            }
        require(len(identities) == 1, f"arm identity mismatch: {episode_dir.name}")
        require(next(iter(identities))[2] == int(contract["episode_seed"]),
                f"arm seed mismatch: {episode_dir.name}")

        full = arm_payloads["full_memory"]
        ablation = arm_payloads["memory_b_native_c"]
        b_prefix_equal = {
            "plans": full["plans"]["legB"] == ablation["plans"]["legB"],
            "rollout": full["plans"]["rollout_traces"]["legB"]
            == ablation["plans"]["rollout_traces"]["legB"],
            "memory": full["plans"]["memory_traces"]["legB"]
            == ablation["plans"]["memory_traces"]["legB"],
        }
        require(all(b_prefix_equal.values()),
                f"causal B prefix differs: {episode_dir.name}")
        for field in (
            "reached_B",
            "c_tail_contract_ok",
            "c_tail_max_covis",
            "c_effective_input_contract_ok",
            "C_evaluated",
        ):
            require(full["metric"][field] == ablation["metric"][field],
                    f"causal-arm pre-C field differs ({field}): {episode_dir.name}")

        certified = arm_payloads["certified"]
        a_ceiling = int(certified["metric"]["A_candidate_ceiling"])
        for leg in ("legB", "legC"):
            for plan in certified["plans"][leg]:
                certificate_requests[leg] += 1
                accepted = plan.get("certified_relocalization_accepted") is True
                certificate_accepts[leg] += int(accepted)
                certificate_reasons[str(plan.get("certified_relocalization_reason"))] += 1
                ceiling = plan.get("candidate_ceiling")
                if leg == "legC" and ceiling is not None:
                    require(int(ceiling) <= a_ceiling,
                            f"certified C exceeded A memory: {episode_dir.name}")

        record = {
            "selection_index": index,
            "scene": expected_identity[0],
            "episode": expected_identity[1],
            "seed": int(contract["episode_seed"]),
            "B_prefix_equal": b_prefix_equal,
            "arms": {},
        }
        for arm, payload in arm_payloads.items():
            metric = payload["metric"]
            record["arms"][arm] = {
                "B": as_bool(metric["reached_B"]),
                "C_input_ok": as_bool(metric["c_effective_input_contract_ok"]),
                "C_evaluated": as_bool(metric["C_evaluated"]),
                "C": as_bool(metric["reached_C"]),
                "joint": as_bool(metric["joint_success"]),
                "steps_B": int(metric["steps_B"]),
                "steps_C": int(metric["steps_C"]),
                "final_dist_B": float(metric["final_dist_B"]),
                "final_dist_C": float(metric["final_dist_C"]),
            }
        records.append(record)

    require(sorted(observed_identities) == sorted(frozen_identities),
            "completed identities differ from frozen selection")
    records.sort(key=lambda row: int(row["selection_index"]))

    arm_summary = {}
    for arm in EXPECTED_ARMS:
        values = [record["arms"][arm] for record in records]
        eligible = [value for value in values if value["C_evaluated"]]
        arm_summary[arm] = {
            "episodes": len(values),
            "B_success": sum(value["B"] for value in values),
            "C_eligible": len(eligible),
            "C_success": sum(value["C"] for value in eligible),
            "joint_success": sum(value["joint"] for value in values),
        }

    causal_eligible = [
        record for record in records
        if record["arms"]["full_memory"]["C_evaluated"]
    ]
    primary_rows = [
        (
            record["scene"],
            record["arms"]["full_memory"]["C"],
            record["arms"]["memory_b_native_c"]["C"],
        )
        for record in causal_eligible
    ]
    require(bool(primary_rows), "no episodes eligible for the primary C contrast")

    def contrast(first_arm: str, second_arm: str, outcome: str) -> dict[str, Any]:
        first = [record["arms"][first_arm][outcome] for record in records]
        second = [record["arms"][second_arm][outcome] for record in records]
        result = exact_mcnemar(first, second)
        result["scene_cluster_bootstrap"] = scene_cluster_bootstrap(
            [
                (record["scene"], a, b)
                for record, a, b in zip(records, first, second)
            ],
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        return result

    primary = exact_mcnemar(
        [first for _, first, _ in primary_rows],
        [second for _, _, second in primary_rows],
    )
    primary["scene_cluster_bootstrap"] = scene_cluster_bootstrap(
        primary_rows, seed=bootstrap_seed, resamples=bootstrap_resamples
    )
    preparation = read_json(run_root / "prepared" / "preparation_report.json")
    inferential_scope = preparation.get(
        "inferential_scope",
        (
            "statistically powered internal fresh-target gate on previously "
            "consumed fresh160 scenes; not paper-final fresh-scene confirmation"
        ),
    )

    return {
        "schema_version": "shared_online_double_revisit_fresh_audit_v1_20260813",
        "scope": inferential_scope,
        "formal_power_target_met": bool(
            preparation.get("formal_power_target_met", expected_episodes >= 40)
        ),
        "audit_ok": True,
        "run_root": str(run_root.resolve()),
        "benchmark_manifest_sha256": expected_manifest_sha,
        "episodes": len(records),
        "scene_clusters": len({record["scene"] for record in records}),
        "construction": {
            "candidate_count": preparation["candidate_count"],
            "constructible_count": preparation["constructible_count"],
            "construction_failure_count": preparation["construction_failure_count"],
            "selected_scene_count": preparation["selected_scene_count"],
            "selection_observed_navigation_outcomes": False,
        },
        "arm_summary": arm_summary,
        "primary_contrast": {
            "name": "full_memory_C_minus_memory_B_native_C_given_identical_B_and_valid_C_input",
            **primary,
        },
        "secondary_contrasts": {
            "full_memory_joint_minus_native": contrast(
                "full_memory", "native", "joint"
            ),
            "certified_joint_minus_native": contrast(
                "certified", "native", "joint"
            ),
            "certified_joint_minus_full_memory": contrast(
                "certified", "full_memory", "joint"
            ),
        },
        "certificate": {
            "requests_B": certificate_requests["legB"],
            "accepted_B": certificate_accepts["legB"],
            "requests_C": certificate_requests["legC"],
            "accepted_C": certificate_accepts["legC"],
            "reasons": dict(sorted(certificate_reasons.items())),
        },
        "all_causal_B_prefixes_exact": all(
            all(record["B_prefix_equal"].values()) for record in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--expected-episodes", type=int, default=40)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.run_root,
        expected_manifest_sha=args.expected_manifest_sha,
        expected_episodes=args.expected_episodes,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
