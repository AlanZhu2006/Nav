#!/usr/bin/env python3
"""Independently apply the frozen four-arm consumed role-pair readiness gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_shared_online_role_pair_smoke import audit as integration_audit


ARMS = ("native", "raw_direct", "raw_fixed_bearing", "certified")
ROLES = ("novel", "revisit")
RADIUS_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, f"{path}: expected two query rows")
    indexed = {str(row["analysis_role"]): row for row in rows}
    require(set(indexed) == set(ROLES), f"{path}: role set changed")
    return indexed


def success(row: dict[str, str]) -> bool:
    return bool(int(row["reached"]))


def plans_for(scene_root: Path, arm: str, row: dict[str, str]) -> list[dict]:
    path = (
        scene_root / arm /
        f"{row['episode']}_{row['query_id']}_plans.json"
    )
    payload = json.loads(path.read_text())
    plans = payload.get("query_leg")
    require(isinstance(plans, list), f"{path}: missing query plan list")
    return plans


def audit(root: Path) -> dict:
    base = integration_audit(root)
    require(tuple(base["arms"]) == ARMS, "four-arm order changed")
    require(base["scenes"] == 4, "gate requires exactly four scenes")
    require(base["queries_per_arm"] == 8, "gate requires eight queries per arm")

    scene_roots = sorted(
        path for path in (root / "scenes").iterdir() if path.is_dir()
    )
    role_outcomes: dict[str, dict[str, list[bool]]] = {
        arm: {role: [] for role in ROLES} for arm in ARMS
    }
    certified_revisit_activations = 0
    fixed_takeovers = 0
    fixed_radius_checks = 0
    per_scene = []
    for scene_root in scene_roots:
        metrics = {
            arm: read_metrics(scene_root / arm / "metric.csv")
            for arm in ARMS
        }
        for arm in ARMS:
            for role in ROLES:
                role_outcomes[arm][role].append(success(metrics[arm][role]))

        revisit = metrics["certified"]["revisit"]
        revisit_active = (
            int(revisit["certificate_accept_plans"]) > 0
            and int(revisit["adapter_takeover_plans"]) > 0
        )
        certified_revisit_activations += int(revisit_active)

        for role in ROLES:
            fixed_row = metrics["raw_fixed_bearing"][role]
            query_plans = plans_for(
                scene_root, "raw_fixed_bearing", fixed_row
            )
            takeover_plans = [
                plan for plan in query_plans
                if plan.get("revisit_adapter_takeover") is True
            ]
            require(bool(takeover_plans), f"{scene_root}: fixed arm inert on {role}")
            fixed_takeovers += len(takeover_plans)
            for plan in takeover_plans:
                require(
                    plan.get("revisit_adapter_mode")
                    == "raw_fixed_bearing_v1",
                    f"{scene_root}: fixed adapter mode changed",
                )
                require(
                    plan.get("revisit_adapter_reason")
                    == "raw_uncertified_fixed_bearing",
                    f"{scene_root}: fixed adapter reason changed",
                )
                for field in (
                    "memory_controller_pointgoal_distance_m",
                    "memory_pointgoal_fixed_radius_m",
                ):
                    value = float(plan[field])
                    require(
                        math.isfinite(value)
                        and abs(value - RADIUS_M) <= 1e-6,
                        f"{scene_root}: {field} is not fixed 2.5 m",
                    )
                    fixed_radius_checks += 1
        per_scene.append({
            "scene_root": scene_root.name,
            "certified_revisit_activated": revisit_active,
            "outcomes": {
                arm: {
                    role: success(metrics[arm][role]) for role in ROLES
                }
                for arm in ARMS
            },
        })

    certified_novel = role_outcomes["certified"]["novel"]
    native_novel = role_outcomes["native"]["novel"]
    revisit_cert = role_outcomes["certified"]["revisit"]
    revisit_native = role_outcomes["native"]["revisit"]
    revisit_gains = sum(
        right and not left
        for left, right in zip(revisit_native, revisit_cert)
    )
    revisit_losses = sum(
        left and not right
        for left, right in zip(revisit_native, revisit_cert)
    )
    gate_checks = {
        "integration_audit_ok": bool(base["ok"]),
        "novel_zero_accepts": base["novel_certified_accept_plans"] == 0,
        "novel_exact_fallback_all_scenes": (
            base["novel_certified_exact_fallback_scenes"] == 4
        ),
        "novel_zero_paired_losses": all(
            (not left) or right
            for left, right in zip(native_novel, certified_novel)
        ),
        "revisit_activation_all_scenes": certified_revisit_activations == 4,
        "revisit_gain_at_least_one": revisit_gains >= 1,
        "revisit_losses_at_most_one": revisit_losses <= 1,
        "fixed_bearing_contract_exercised": fixed_takeovers > 0,
        "fixed_bearing_all_radius_checks_passed": fixed_radius_checks > 0,
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "shared_online_role_pair_four_arm_gate_v1_20260814",
        "root": str(root.resolve()),
        "source_run_contract_sha256": sha256_file(root / "run_contract.json"),
        "passed": passed,
        "paper_final_unlock_authorized": passed,
        "gate_checks": gate_checks,
        "paired_certified_minus_native": {
            "novel_gains": sum(
                right and not left
                for left, right in zip(native_novel, certified_novel)
            ),
            "novel_losses": sum(
                left and not right
                for left, right in zip(native_novel, certified_novel)
            ),
            "revisit_gains": revisit_gains,
            "revisit_losses": revisit_losses,
        },
        "certified_revisit_activation_scenes": certified_revisit_activations,
        "fixed_takeover_plans": fixed_takeovers,
        "fixed_radius_scalar_checks": fixed_radius_checks,
        "role_outcomes": role_outcomes,
        "per_scene": per_scene,
        "integration_audit": base,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
