#!/usr/bin/env python3
"""Aggregate the consumed HM3D fixed-vs-full-metric Revisit gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


SCHEMA = "hm3d_table1_metric_distance_revisit_summary_v1_20260901"
EPISODE_SCHEMA = "hm3d_table1_metric_distance_revisit_pair_v1_20260901"
ARMS = ("fixed_2p5m", "full_metric")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(gains, losses) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def spl(row: dict[str, Any], arm: str) -> float:
    if int(row["outcomes"][arm]) == 0:
        return 0.0
    shortest = float(row["geodesic_m"][arm])
    executed = float(row["path_len_m"][arm])
    return shortest / max(shortest, executed, 1e-12)


def cluster_interval(
    rows: list[dict[str, Any]], *, samples: int, seed: int,
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene"])].append(row)
    scenes = sorted(groups)
    numerator = np.asarray([
        sum(int(row["outcomes"]["full_metric"])
            - int(row["outcomes"]["fixed_2p5m"])
            for row in groups[scene])
        for scene in scenes
    ], dtype=np.float64)
    denominator = np.asarray([
        len(groups[scene]) for scene in scenes
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    chunk = 10_000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        draws = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        values[start:stop] = (
            numerator[draws].sum(axis=1)
            / denominator[draws].sum(axis=1)
        )
    return [
        100.0 * float(np.quantile(values, 0.025)),
        100.0 * float(np.quantile(values, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()

    manifest_path = args.benchmark_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    histories = manifest.get("episodes")
    require(isinstance(histories, list) and len(histories) == 28,
            "expected 28 frozen histories")
    require(len({str(row["scene"]) for row in histories}) == 21,
            "expected 21 frozen scene clusters")

    root = args.run_root / "evaluation" / "revisit_metric_distance"
    paths = sorted(root.glob("*/completion.json"))
    require(len(paths) == len(histories),
            f"expected {len(histories)} completions, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        receipt = path.with_suffix(path.suffix + ".sha256")
        fields = receipt.read_text().split()
        require(fields and fields[0] == sha256(path),
                f"completion receipt changed: {path}")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == EPISODE_SCHEMA,
                f"completion schema changed: {path}")
        require(row.get("benchmark_manifest_sha256")
                == args.expected_manifest_sha256,
                f"completion manifest changed: {path}")
        require(row.get("arms") == list(ARMS),
                f"arm set changed: {path}")
        require(row.get("same_server_pair") is True
                and row.get("runtime_role_visibility") == "none"
                and row.get("analysis_query_role") == "revisit"
                and row.get("prefix_equality") is True,
                f"pairing contract failed: {path}")
        index = int(row["history_index"])
        require(index not in seen and 0 <= index < len(histories),
                f"history coverage changed: {path}")
        seen.add(index)
        expected = histories[index]
        require(str(row["scene"]) == str(expected["scene"])
                and str(row["episode"]) == str(expected["episode"]),
                f"history identity changed: {path}")
        rows.append(row)
    require(seen == set(range(len(histories))),
            "history coverage is incomplete")
    rows.sort(key=lambda row: int(row["history_index"]))

    fixed_success = sum(int(row["outcomes"]["fixed_2p5m"]) for row in rows)
    metric_success = sum(int(row["outcomes"]["full_metric"]) for row in rows)
    gains = sum(
        int(row["outcomes"]["full_metric"]) == 1
        and int(row["outcomes"]["fixed_2p5m"]) == 0
        for row in rows
    )
    losses = sum(
        int(row["outcomes"]["full_metric"]) == 0
        and int(row["outcomes"]["fixed_2p5m"]) == 1
        for row in rows
    )
    both_success = [
        row for row in rows
        if int(row["outcomes"]["fixed_2p5m"]) == 1
        and int(row["outcomes"]["full_metric"]) == 1
    ]
    fixed_path = mean(float(row["path_len_m"]["fixed_2p5m"])
                      for row in both_success) if both_success else None
    metric_path = mean(float(row["path_len_m"]["full_metric"])
                       for row in both_success) if both_success else None
    fixed_steps = mean(float(row["steps"]["fixed_2p5m"])
                       for row in both_success) if both_success else None
    metric_steps = mean(float(row["steps"]["full_metric"])
                        for row in both_success) if both_success else None
    path_improvement = (
        (fixed_path - metric_path) / fixed_path
        if fixed_path not in (None, 0.0) else None
    )
    step_improvement = (
        (fixed_steps - metric_steps) / fixed_steps
        if fixed_steps not in (None, 0.0) else None
    )

    first_receipts = [
        row["first_takeover"] for row in rows
        if row.get("first_takeover") is not None
    ]
    all_distances = [
        float(distance)
        for row in rows
        for distance in row["metric_controller_distances_m"]
    ]
    cap_hits = sum(int(row["metric_native_support_cap_hits"]) for row in rows)
    net = gains - losses
    efficiency_gate = bool(
        cap_hits == 0
        and losses == 0
        and ((path_improvement is not None and path_improvement >= 0.10)
             or (step_improvement is not None and step_improvement >= 0.10))
    )
    if cap_hits > 0:
        decision = "stop_keep_fixed_2p5m_native_support_saturation"
    elif net >= 2 and losses <= 1:
        decision = "promote_to_independent_revisit_only_replication"
    elif net <= -2:
        decision = "reject_full_metric_distance"
    elif efficiency_gate:
        decision = "retain_as_efficiency_ablation_not_main_method"
    else:
        decision = "stop_keep_fixed_2p5m"

    summary = {
        "schema_version": SCHEMA,
        "claim_boundary": (
            "consumed-population promotion gate; not fresh confirmation and "
            "not a certified metric-safety claim"
        ),
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "histories": len(rows),
        "scene_clusters": len({str(row["scene"]) for row in rows}),
        "arms": list(ARMS),
        "success": {
            "fixed_2p5m": fixed_success,
            "full_metric": metric_success,
        },
        "success_rate": {
            "fixed_2p5m": fixed_success / len(rows),
            "full_metric": metric_success / len(rows),
        },
        "paired": {
            "gain": gains,
            "loss": losses,
            "net": net,
            "risk_difference_pp": 100.0 * net / len(rows),
            "mcnemar_exact_p": exact_mcnemar(gains, losses),
            "scene_cluster_bootstrap_ci95_pp": cluster_interval(
                rows, samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        },
        "secondary": {
            "spl": {
                arm: mean(spl(row, arm) for row in rows) for arm in ARMS
            },
            "both_success_n": len(both_success),
            "both_success_mean_path_len_m": {
                "fixed_2p5m": fixed_path,
                "full_metric": metric_path,
            },
            "both_success_mean_steps": {
                "fixed_2p5m": fixed_steps,
                "full_metric": metric_steps,
            },
            "paired_path_fraction_improvement": path_improvement,
            "paired_step_fraction_improvement": step_improvement,
        },
        "treatment_audit": {
            "histories_with_first_takeover": len(first_receipts),
            "first_takeover_differs_from_fixed": sum(
                bool(row["differs_from_fixed_2p5m"])
                for row in first_receipts
            ),
            "metric_takeover_plans": len(all_distances),
            "metric_controller_distance_m": {
                "minimum": min(all_distances) if all_distances else None,
                "median": float(np.median(all_distances))
                if all_distances else None,
                "maximum": max(all_distances) if all_distances else None,
            },
            "native_10m_support_cap_hits": cap_hits,
            "all_metric_payloads_unclipped": cap_hits == 0,
        },
        "preregistered_decision": decision,
        "decision_rule": {
            "promote": "paired net >= 2, losses <= 1, and zero 10m cap hits",
            "reject": "paired net <= -2",
            "otherwise": (
                "keep fixed 2.5m unless both-success path or steps improve "
                "by >=10% with zero losses; efficiency alone remains ablation"
            ),
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "scene",
        },
    }
    require(not args.out.exists(), "summary output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
