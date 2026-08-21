#!/usr/bin/env python3
"""Independent audit of the shared-online double-Revisit four-arm gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "shared_online_double_revisit_closed_loop_v3_leg_scope_20260813"
EXPECTED_MANIFEST_SHA = (
    "95f5cbb311c10f3f6604eca47632cefea4b77b80d9f1e0e6ec93c1056c30786f"
)
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
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "N": len(first),
        "first_success": sum(first),
        "second_success": sum(second),
        "gain": gains,
        "loss": losses,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
    }


def audit(run_root: Path) -> dict[str, Any]:
    scenes_root = run_root / "scenes"
    require(scenes_root.is_dir(), "run root has no scenes directory")
    scene_dirs = sorted(path for path in scenes_root.iterdir() if path.is_dir())
    require(bool(scene_dirs), "no completed scene directories")

    records: list[dict[str, Any]] = []
    certificate_reasons: Counter[str] = Counter()
    certificate_requests = Counter()
    certificate_accepts = Counter()

    for scene_dir in scene_dirs:
        arm_payloads: dict[str, dict[str, Any]] = {}
        identities = set()
        for arm, expected in EXPECTED_ARMS.items():
            arm_root = scene_dir / arm
            require(arm_root.is_dir(), f"missing arm {scene_dir.name}/{arm}")

            # Read protocol receipts before outcome rows.
            summary = read_json(arm_root / "summary.json")
            contract = read_json(arm_root / "run_contract.json")
            require(summary.get("schema_version") == EXPECTED_SCHEMA,
                    f"schema changed: {scene_dir.name}/{arm}")
            require(summary.get("benchmark_manifest_sha256") ==
                    EXPECTED_MANIFEST_SHA,
                    f"manifest changed: {scene_dir.name}/{arm}")
            require(summary.get("variant") ==
                    "v1_controlled_pose_perturbation",
                    f"variant changed: {scene_dir.name}/{arm}")
            require(summary.get("deterministic_plan_seeds") is True,
                    f"non-deterministic seeds: {scene_dir.name}/{arm}")
            require(summary.get("shared_A_all_hashes_ok") is True,
                    f"shared-A hash audit failed: {scene_dir.name}/{arm}")
            require(summary.get("shared_A_total_diffusion_samples") == 0,
                    f"shared-A replay sampled diffusion: {scene_dir.name}/{arm}")
            require(summary.get("navdp_goal_switch_reset") == "before_c",
                    f"reset contract changed: {scene_dir.name}/{arm}")
            require(summary.get("server_backend") == expected["server_backend"],
                    f"server backend changed: {scene_dir.name}/{arm}")
            require(summary.get("hybrid_route") == expected["hybrid_route"],
                    f"hybrid route changed: {scene_dir.name}/{arm}")
            require(summary.get("known_revisit_scope") == expected["scope"],
                    f"leg scope changed: {scene_dir.name}/{arm}")
            require(summary.get("policy_backends") == expected["backends"],
                    f"policy backends changed: {scene_dir.name}/{arm}")
            require(summary.get("C_long_memory_enabled") is
                    expected["C_long_memory_enabled"],
                    f"C memory flag changed: {scene_dir.name}/{arm}")
            require(contract.get("C_history") == "initial_leg_only",
                    f"C history changed: {scene_dir.name}/{arm}")
            require(contract.get("navdp_goal_switch_reset") == "before_c",
                    f"run reset changed: {scene_dir.name}/{arm}")

            metric = read_single_metric(arm_root / "metric.csv")
            plan_files = sorted(arm_root.glob("episode_*_plans.json"))
            require(len(plan_files) == 1,
                    f"expected one plans file: {scene_dir.name}/{arm}")
            plans = read_json(plan_files[0])
            identities.add((metric["scene"], metric["episode"], metric["seed"]))
            require(as_bool(metric["shared_A_hashes_ok"]),
                    f"metric shared-A hash failed: {scene_dir.name}/{arm}")
            require(int(metric["shared_A_replay_diffusion_samples"]) == 0,
                    f"metric shared-A replay sampled: {scene_dir.name}/{arm}")
            require(as_bool(metric["c_effective_input_contract_ok"]),
                    f"C input contract failed: {scene_dir.name}/{arm}")
            arm_payloads[arm] = {
                "summary": summary,
                "contract": contract,
                "metric": metric,
                "plans": plans,
            }

        require(len(identities) == 1, f"arm identity mismatch: {scene_dir.name}")

        full = arm_payloads["full_memory"]["plans"]
        ablation = arm_payloads["memory_b_native_c"]["plans"]
        b_prefix_equal = {
            "plans": full["legB"] == ablation["legB"],
            "rollout": (full["rollout_traces"]["legB"] ==
                        ablation["rollout_traces"]["legB"]),
            "memory": (full["memory_traces"]["legB"] ==
                       ablation["memory_traces"]["legB"]),
        }
        require(all(b_prefix_equal.values()),
                f"B prefix differs: {scene_dir.name} {b_prefix_equal}")

        certified = arm_payloads["certified"]
        a_ceiling = int(certified["metric"]["A_candidate_ceiling"])
        for leg in ("legB", "legC"):
            for plan in certified["plans"][leg]:
                certificate_requests[leg] += 1
                accepted = plan.get("certified_relocalization_accepted") is True
                certificate_accepts[leg] += int(accepted)
                certificate_reasons[str(
                    plan.get("certified_relocalization_reason"))] += 1
                ceiling = plan.get("candidate_ceiling")
                if leg == "legC" and ceiling is not None:
                    require(int(ceiling) <= a_ceiling,
                            f"C exceeded A boundary: {scene_dir.name}")

        record = {
            "scene_directory": scene_dir.name,
            "scene": next(iter(identities))[0],
            "episode": next(iter(identities))[1],
            "seed": int(next(iter(identities))[2]),
            "B_prefix_equal": b_prefix_equal,
            "arms": {},
        }
        for arm, payload in arm_payloads.items():
            metric = payload["metric"]
            record["arms"][arm] = {
                "B": as_bool(metric["reached_B"]),
                "C_evaluated": as_bool(metric["C_evaluated"]),
                "C": as_bool(metric["reached_C"]),
                "joint": as_bool(metric["joint_success"]),
                "steps_B": int(metric["steps_B"]),
                "steps_C": int(metric["steps_C"]),
                "final_dist_B": float(metric["final_dist_B"]),
                "final_dist_C": float(metric["final_dist_C"]),
            }
        records.append(record)

    arm_summary = {}
    for arm in EXPECTED_ARMS:
        values = [record["arms"][arm] for record in records]
        c_eligible = [value for value in values if value["C_evaluated"]]
        arm_summary[arm] = {
            "episodes": len(values),
            "B_success": sum(value["B"] for value in values),
            "C_eligible": len(c_eligible),
            "C_success": sum(value["C"] for value in c_eligible),
            "joint_success": sum(value["joint"] for value in values),
            "mean_steps_B": sum(value["steps_B"] for value in values) / len(values),
            "mean_steps_C_when_evaluated": (
                sum(value["steps_C"] for value in c_eligible) / len(c_eligible)
                if c_eligible else None
            ),
        }

    eligible = [
        record for record in records
        if record["arms"]["full_memory"]["C_evaluated"]
        and record["arms"]["memory_b_native_c"]["C_evaluated"]
    ]
    contrasts = {
        "full_memory_C_minus_memory_B_native_C": exact_mcnemar(
            [record["arms"]["full_memory"]["C"] for record in eligible],
            [record["arms"]["memory_b_native_c"]["C"]
             for record in eligible],
        ),
        "certified_joint_minus_native": exact_mcnemar(
            [record["arms"]["certified"]["joint"] for record in records],
            [record["arms"]["native"]["joint"] for record in records],
        ),
        "certified_joint_minus_full_memory": exact_mcnemar(
            [record["arms"]["certified"]["joint"] for record in records],
            [record["arms"]["full_memory"]["joint"] for record in records],
        ),
    }

    return {
        "schema_version": "shared_online_double_revisit_gate_audit_v1_20260813",
        "scope": "N=4 causal mechanism gate; not an SR estimate",
        "audit_ok": True,
        "run_root": str(run_root.resolve()),
        "benchmark_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "scenes": len(records),
        "arm_summary": arm_summary,
        "contrasts": contrasts,
        "certificate": {
            "requests_B": certificate_requests["legB"],
            "accepted_B": certificate_accepts["legB"],
            "requests_C": certificate_requests["legC"],
            "accepted_C": certificate_accepts["legC"],
            "reasons": dict(sorted(certificate_reasons.items())),
        },
        "all_B_prefixes_exact": all(
            all(record["B_prefix_equal"].values()) for record in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.run_root)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
