#!/usr/bin/env python3
"""Outcome-neutral integrity audit for local role-pair navigation runs.

Unlike the consumed readiness-gate auditor, this reader never treats method
performance (Novel rejection, Revisit activation, or SR gain) as a validity
criterion.  It checks the frozen inputs, pairing, causal-history replay,
role-hiding and runtime integrity, then independently recomputes outcomes and
paired McNemar counts from raw CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROLES = ("novel", "revisit")
FORBIDDEN_RUNTIME_FIELDS = {
    "analysis_role",
    "max_online_a_covis",
    "covis_curve",
    "geodesic_from_a_end_m",
    "initial_path_bearing_rad",
}
SHA256_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, f"{path}: expected exactly two role rows")
    indexed = {str(row["analysis_role"]): row for row in rows}
    require(set(indexed) == set(ROLES), f"{path}: role set changed")
    return indexed


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, losses) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def verify_source_inputs(receipt: Path) -> dict[str, Any]:
    rows = []
    for line_number, raw in enumerate(receipt.read_text().splitlines(), start=1):
        match = SHA256_LINE.fullmatch(raw)
        require(match is not None, f"{receipt}:{line_number}: malformed sha256 line")
        expected, filename = match.groups()
        path = Path(filename)
        require(path.is_file(), f"source input missing: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"source input changed: {path}")
        rows.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    require(bool(rows), "source-input receipt is empty")
    return {
        "receipt_sha256": sha256_file(receipt),
        "files_verified": len(rows),
        "bytes_verified": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }


def outcome(row: dict[str, str]) -> bool:
    return bool(int(row["reached"]))


def contrast(records: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    gains = [
        record for record in records
        if record["outcomes"][right] and not record["outcomes"][left]
    ]
    losses = [
        record for record in records
        if record["outcomes"][left] and not record["outcomes"][right]
    ]
    return {
        "n": len(records),
        "left": left,
        "right": right,
        "left_successes": sum(record["outcomes"][left] for record in records),
        "right_successes": sum(record["outcomes"][right] for record in records),
        "gains": len(gains),
        "losses": len(losses),
        "risk_difference_pp": (
            100.0 * (len(gains) - len(losses)) / len(records)
            if records else None
        ),
        "exact_mcnemar_two_sided_p": exact_mcnemar(len(gains), len(losses)),
        "gain_identities": [
            [record["scene"], record["episode"], record["role"]]
            for record in gains
        ],
        "loss_identities": [
            [record["scene"], record["episode"], record["role"]]
            for record in losses
        ],
    }


def audit(root: Path) -> dict[str, Any]:
    contract_path = root / "run_contract.json"
    input_receipt_path = root / "source_inputs.sha256"
    contract = json.loads(contract_path.read_text())
    require(
        contract.get("schema_version")
        == "shared_online_role_pair_local_smoke_v1_20260814",
        "run-contract schema changed",
    )
    require(contract.get("role_visible_to_runtime") is False, "role exposed to runtime")
    require(contract.get("blind_data_read") is False, "run contract claims blind-data read")
    arms = tuple(str(value) for value in contract.get("arms", ()))
    require("native" in arms and "certified" in arms, "required arms missing")
    require(len(arms) == len(set(arms)), "duplicate arm declaration")
    selected = list(contract.get("selected_identities") or ())
    require(bool(selected), "selected population is empty")

    input_audit = verify_source_inputs(input_receipt_path)
    scene_roots = sorted(
        path for path in (root / "scenes").iterdir() if path.is_dir()
    )
    require(len(scene_roots) == len(selected), "scene-output population is incomplete")

    records: list[dict[str, Any]] = []
    runtime_failures = 0
    certificate = {
        role: {"accept_episodes": 0, "accept_plans": 0,
               "takeover_episodes": 0, "takeover_plans": 0}
        for role in ROLES
    }
    rejected_novel = 0
    rejected_novel_exact_native = 0
    completion_rows = []
    for index, (scene_root, selected_identity) in enumerate(zip(scene_roots, selected)):
        require(scene_root.name.startswith(f"{index:02d}_"), "output order changed")
        per_arm_metrics = {}
        per_arm_summaries = {}
        per_arm_plans = {}
        for arm in arms:
            arm_root = scene_root / arm
            summary = json.loads((arm_root / "summary.json").read_text())
            require(summary["arm"] == arm, f"{scene_root}/{arm}: arm mismatch")
            require(summary["runtime_role_visibility"] == "none",
                    f"{scene_root}/{arm}: role visibility changed")
            require(summary["shared_A_all_hashes_ok"] is True,
                    f"{scene_root}/{arm}: shared history hash failed")
            require(int(summary["shared_A_total_diffusion_samples"]) == 0,
                    f"{scene_root}/{arm}: shared history replay sampled diffusion")
            require(int(summary["runtime_failure_plans"]) == 0,
                    f"{scene_root}/{arm}: runtime failure occurred")
            require(int(summary["max_steps"]) == int(contract["max_steps"]),
                    f"{scene_root}/{arm}: step budget changed")
            metrics = read_metrics(arm_root / "metric.csv")
            per_arm_metrics[arm] = metrics
            per_arm_summaries[arm] = summary
            per_arm_plans[arm] = {}
            for role, row in metrics.items():
                require(int(row["runtime_failure_plans"]) == 0,
                        f"{scene_root}/{arm}/{role}: runtime failure")
                plan_path = arm_root / f"{row['episode']}_{row['query_id']}_plans.json"
                plan = json.loads(plan_path.read_text())
                require(plan["analysis_role_not_forwarded"] is True,
                        f"{plan_path}: role-forwarding receipt failed")
                runtime_fields = set(plan["query_runtime_fields"])
                require(not runtime_fields.intersection(FORBIDDEN_RUNTIME_FIELDS),
                        f"{plan_path}: analysis field leaked to runtime")
                per_arm_plans[arm][role] = plan

        reference = per_arm_metrics["native"]
        for role in ROLES:
            reference_row = reference[role]
            identity = (
                reference_row["scene"], reference_row["episode"],
                reference_row["pair_id"], reference_row["query_id"],
                reference_row["analysis_role"], reference_row["seed"],
                reference_row["shared_A_frames"],
                reference_row["shared_A_decision_frames"],
                reference_row["geodesic_m"],
            )
            for arm in arms:
                row = per_arm_metrics[arm][role]
                candidate = (
                    row["scene"], row["episode"], row["pair_id"],
                    row["query_id"], row["analysis_role"], row["seed"],
                    row["shared_A_frames"], row["shared_A_decision_frames"],
                    row["geodesic_m"],
                )
                require(candidate == identity,
                        f"{scene_root}/{role}: arm inputs are not paired")
            require(
                [reference_row["scene"], reference_row["episode"]]
                == list(selected_identity),
                f"{scene_root}: selected identity changed",
            )
            outcomes = {arm: outcome(per_arm_metrics[arm][role]) for arm in arms}
            records.append({
                "scene": reference_row["scene"],
                "episode": reference_row["episode"],
                "pair_id": reference_row["pair_id"],
                "query_id": reference_row["query_id"],
                "role": role,
                "outcomes": outcomes,
                "per_arm": {
                    arm: {
                        "success": outcomes[arm],
                        "steps": int(per_arm_metrics[arm][role]["steps"]),
                        "path_len_m": float(per_arm_metrics[arm][role]["path_len_m"]),
                        "final_goal_dist_m": float(
                            per_arm_metrics[arm][role]["final_goal_dist_m"]
                        ),
                        "certificate_accept_plans": int(
                            per_arm_metrics[arm][role]["certificate_accept_plans"]
                        ),
                        "adapter_takeover_plans": int(
                            per_arm_metrics[arm][role]["adapter_takeover_plans"]
                        ),
                    }
                    for arm in arms
                },
            })

            certified_row = per_arm_metrics["certified"][role]
            accepts = int(certified_row["certificate_accept_plans"])
            takeovers = int(certified_row["adapter_takeover_plans"])
            certificate[role]["accept_plans"] += accepts
            certificate[role]["takeover_plans"] += takeovers
            certificate[role]["accept_episodes"] += int(accepts > 0)
            certificate[role]["takeover_episodes"] += int(takeovers > 0)
            runtime_failures += int(certified_row["runtime_failure_plans"])
            if role == "novel" and accepts == 0 and takeovers == 0:
                rejected_novel += 1
                certified_plan = per_arm_plans["certified"][role]
                native_plan = per_arm_plans["native"][role]
                metric_equal = all(
                    certified_row[field] == reference_row[field]
                    for field in (
                        "reached", "steps", "path_len_m",
                        "final_goal_dist_m", "termination_reason",
                    )
                )
                trace_equal = (
                    certified_plan["rollout_traces"]
                    == native_plan["rollout_traces"]
                )
                rejected_novel_exact_native += int(metric_equal and trace_equal)

        completion_rows.append({
            "index": index,
            "scene": reference["novel"]["scene"],
            "episode": reference["novel"]["episode"],
            "arm_summary_sha256": {
                arm: sha256_file(scene_root / arm / "summary.json") for arm in arms
            },
            "arm_metric_sha256": {
                arm: sha256_file(scene_root / arm / "metric.csv") for arm in arms
            },
        })

    metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role_filter in ("all", *ROLES):
            subset = (
                records if role_filter == "all"
                else [record for record in records if record["role"] == role_filter]
            )
            successes = sum(record["outcomes"][arm] for record in subset)
            metrics[arm][role_filter] = {
                "n": len(subset),
                "successes": successes,
                "SR": successes / len(subset),
            }

    contrasts = {}
    for baseline in arms:
        if baseline == "certified":
            continue
        contrasts[f"certified_minus_{baseline}"] = {
            role_filter: contrast(
                records if role_filter == "all"
                else [record for record in records if record["role"] == role_filter],
                baseline,
                "certified",
            )
            for role_filter in ("all", *ROLES)
        }

    return {
        "schema_version": "local_role_pair_outcome_neutral_audit_v1_20260814",
        "valid": True,
        "scope": contract["scope"],
        "run_root": str(root.resolve()),
        "run_contract_sha256": sha256_file(contract_path),
        "auditor_sha256": sha256_file(Path(__file__)),
        "benchmark_manifest_sha256": contract["benchmark_manifest_sha256"],
        "max_steps": int(contract["max_steps"]),
        "histories": len(scene_roots),
        "queries": len(records),
        "arms": list(arms),
        "runtime_role_visibility": "none",
        "online_a_hashes_verified": True,
        "online_a_diffusion_samples": 0,
        "runtime_failure_plans": runtime_failures,
        "source_inputs": input_audit,
        "certificate_observations_not_validity_gates": certificate,
        "rejected_novel_queries": rejected_novel,
        "rejected_novel_exact_native_queries": rejected_novel_exact_native,
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "completion_rows": completion_rows,
        "statistical_note": (
            "single-scene consumed cross-domain pilot; no scene-cluster interval "
            "and no scene-disjoint generalization claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded)
    print(json.dumps({
        "valid": result["valid"],
        "histories": result["histories"],
        "queries": result["queries"],
        "runtime_failure_plans": result["runtime_failure_plans"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
