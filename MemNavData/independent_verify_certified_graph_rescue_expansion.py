#!/usr/bin/env python3
"""Independent raw-CSV verifier for the certified graph fresh20 report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("direct", "budget_control", "rescue")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bool_value(value: Any) -> bool:
    if value in (True, 1, "1", "1.0", "true", "True"):
        return True
    if value in (False, 0, "0", "0.0", "false", "False", ""):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"expected one row: {path}")
    return rows[0]


def exact_p(gains: int, losses: int) -> float:
    n = gains + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def contrast(
        rows: list[dict[str, Any]], first: str, second: str, outcome: str,
        *, seed: int, resamples: int) -> dict[str, Any]:
    pairs = [
        (row["scene"], row[first][outcome], row[second][outcome])
        for row in rows
    ]
    gains = sum(a and not b for _, a, b in pairs)
    losses = sum(b and not a for _, a, b in pairs)
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene, a, b in pairs:
        grouped[scene].append(float(a) - float(b))
    scenes = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(scenes), size=len(scenes))
        values = [value for item in chosen for value in grouped[scenes[item]]]
        draws[index] = np.mean(values)
    return {
        "N": len(rows),
        "first_success": sum(a for _, a, _ in pairs),
        "second_success": sum(b for _, _, b in pairs),
        "risk_difference_pp": 100.0 * sum(
            float(a) - float(b) for _, a, b in pairs) / len(rows),
        "gain": gains,
        "loss": losses,
        "discordant": gains + losses,
        "exact_mcnemar_p": exact_p(gains, losses),
        "first_arm": first,
        "second_arm": second,
        "outcome": outcome,
        "scene_cluster_bootstrap": {
            "clusters": len(scenes), "episodes": len(rows),
            "seed": seed, "resamples": resamples,
            "risk_difference_pp": 100.0 * np.mean([
                float(a) - float(b) for _, a, b in pairs]),
            "ci95_pp": [
                100.0 * float(np.quantile(draws, 0.025)),
                100.0 * float(np.quantile(draws, 0.975)),
            ],
        },
    }


def verify(run_root: Path, report_path: Path, expected_report_sha: str) -> dict:
    require(sha256_file(report_path) == expected_report_sha,
            "official report hash changed")
    report = json.loads(report_path.read_text())
    require(report["schema_version"] ==
            "certified_stagnation_graph_fresh20_expansion_report_v1",
            "wrong report schema")
    require(sha256_file(Path(report["pilot_report"])) ==
            report["pilot_report_sha256"], "pilot report changed")
    rows = []
    for root in sorted((run_root / "scenes").iterdir()):
        contract = json.loads((root / "episode_contract.json").read_text())
        row = {
            "selection_index": int(contract["selection_index"]),
            "scene": contract["scene"], "episode": contract["episode"],
        }
        for arm in ARMS:
            metric = one_csv(root / arm / "metric.csv")
            require(metric["scene"] == row["scene"] and
                    metric["episode"] == row["episode"],
                    f"{root.name}/{arm}: identity mismatch")
            row[arm] = {
                "B": bool_value(metric["reached_B"]),
                "C": bool_value(metric["reached_C"]),
                "joint": bool_value(metric["joint_success"]),
                "graph_active_plans_B": int(
                    metric["certified_graph_active_plans_B"]),
                "graph_active_plans_C": int(
                    metric["certified_graph_active_plans_C"]),
            }
        rows.append(row)
    rows.sort(key=lambda row: row["selection_index"])
    require([row["selection_index"] for row in rows] == list(range(20)),
            "raw full20 indices changed")

    arm_summary = {}
    for arm in ARMS:
        arm_summary[arm] = {
            "B_success": sum(row[arm]["B"] for row in rows),
            "C_success": sum(row[arm]["C"] for row in rows),
            "joint_success": sum(row[arm]["joint"] for row in rows),
            "episodes": len(rows),
            "graph_active_plans_B": sum(
                row[arm]["graph_active_plans_B"] for row in rows),
            "graph_active_plans_C": sum(
                row[arm]["graph_active_plans_C"] for row in rows),
        }
    require(arm_summary == report["arm_summary"], "arm summary differs")

    specifications = {
        "primary_contrast": ("rescue", "budget_control", "joint"),
        "rescue_minus_direct_joint": ("rescue", "direct", "joint"),
        "budget_control_minus_direct_joint": (
            "budget_control", "direct", "joint"),
        "rescue_minus_budget_control_B": (
            "rescue", "budget_control", "B"),
    }
    checked = {}
    for name, spec in specifications.items():
        expected = (report["primary_contrast"] if name == "primary_contrast"
                    else report["secondary_contrasts"][name])
        bootstrap = expected["scene_cluster_bootstrap"]
        actual = contrast(
            rows, *spec, seed=int(bootstrap["seed"]),
            resamples=int(bootstrap["resamples"]))
        require(actual == expected, f"contrast differs: {name}")
        checked[name] = actual

    direct_failures = [
        row["selection_index"] for row in rows if not row["direct"]["joint"]]
    gains = [
        row["selection_index"] for row in rows
        if not row["direct"]["joint"] and row["rescue"]["joint"]]
    losses = [
        row["selection_index"] for row in rows
        if row["direct"]["joint"] and not row["rescue"]["joint"]]
    require(direct_failures == report["gate"]["direct_failure_indices"],
            "direct failure list differs")
    require(gains == report["gate"]["rescue_gain_indices"],
            "rescue gain list differs")
    require(losses == report["gate"]["rescue_loss_indices"],
            "rescue loss list differs")
    require(all(
        row["rescue"]["graph_active_plans_B"] > 0
        or row["rescue"]["graph_active_plans_C"] > 0
        for row in rows if row["selection_index"] in gains),
        "a raw rescue gain did not execute a graph")
    return {
        "schema_version": (
            "independent_certified_graph_expansion_verification_v1"),
        "verified_report": str(report_path),
        "verified_report_sha256": expected_report_sha,
        "raw_episode_count": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "arm_summary": arm_summary,
        "contrasts": checked,
        "direct_failure_indices": direct_failures,
        "rescue_gain_indices": gains,
        "rescue_loss_indices": losses,
        "verification": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root, args.report, args.expected_report_sha)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verification": result["verification"],
        "arm_summary": result["arm_summary"],
        "rescue_gain_indices": result["rescue_gain_indices"],
        "rescue_loss_indices": result["rescue_loss_indices"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
