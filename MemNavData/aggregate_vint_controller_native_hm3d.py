#!/usr/bin/env python3
"""Aggregate mixed-role HM3D ViNT native-control versus ViNT+CEC pairs."""

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


SCHEMA = "vint_controller_native_hm3d_summary_v1_20260828"
AUDIT_SCHEMA = "vint_controller_native_pair_audit_v1_20260828"
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(int(gains), int(losses)) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def spl(row: dict[str, Any], prefix: str) -> float:
    if int(row[f"{prefix}_success"]) == 0:
        return 0.0
    shortest = float(row["initial_geodesic_m"])
    executed = float(row[f"{prefix}_path_len_m"])
    denominator = max(shortest, executed, 1e-12)
    return shortest / denominator


def scene_cluster_ci(
    rows: list[dict[str, Any]], *, samples: int = 100_000, seed: int = 20260828,
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene"])].append(row)
    scenes = sorted(groups)
    require(bool(scenes), "cannot bootstrap an empty population")
    numerators = np.asarray([
        sum(int(row["grant_success"]) - int(row["native_success"])
            for row in groups[scene])
        for scene in scenes
    ], dtype=np.float64)
    denominators = np.asarray(
        [len(groups[scene]) for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    chunk = 10_000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        draws = rng.integers(
            0, len(scenes), size=(stop - start, len(scenes)))
        values[start:stop] = (
            numerators[draws].sum(axis=1)
            / denominators[draws].sum(axis=1)
        )
    return [
        float(np.quantile(values, 0.025)) * 100.0,
        float(np.quantile(values, 0.975)) * 100.0,
    ]


def _population_index(manifest: dict[str, Any]) -> dict[tuple[str, str], int]:
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list) and episodes,
            "benchmark manifest contains no episodes")
    result: dict[tuple[str, str], int] = {}
    for index, entry in enumerate(episodes):
        identity = str(entry.get("scene")), str(entry.get("episode"))
        require(identity not in result, "benchmark history identity is duplicated")
        result[identity] = index
    return result


def aggregate(
    run_root: Path,
    benchmark_manifest: Path,
    *,
    expected_histories: int,
    expected_scenes: int,
    history_indices: tuple[int, ...] | None = None,
    claim_scope: str,
) -> dict[str, Any]:
    manifest = json.loads(benchmark_manifest.read_text())
    population = _population_index(manifest)
    expected_index_set = (
        set(range(len(population)))
        if history_indices is None else set(history_indices))
    require(len(expected_index_set) == expected_histories,
            "expected history-index set has the wrong size")

    audit_paths = sorted((run_root / "evaluation").glob(
        "*/vint/controller_native_pair_audit.json"))
    require(len(audit_paths) == expected_histories,
            f"pair audit count {len(audit_paths)} != {expected_histories}")
    cells = []
    rows: list[dict[str, Any]] = []
    realized_indices: set[int] = set()
    for path in audit_paths:
        cell = json.loads(path.read_text())
        require(cell.get("schema_version") == AUDIT_SCHEMA
                and cell.get("verified") is True,
                f"unverified pair audit: {path}")
        require(cell.get("controller") == "vint"
                and cell.get("reject_policy") == "controller_native_exact",
                f"wrong controller treatment: {path}")
        identity = str(cell["scene"]), str(cell["episode"])
        require(identity in population, f"cell not in benchmark manifest: {path}")
        history_index = population[identity]
        require(history_index in expected_index_set,
                f"cell is outside frozen history indices: {path}")
        require(history_index not in realized_indices,
                f"duplicate history cell: {path}")
        realized_indices.add(history_index)
        query_rows = cell.get("query_results")
        require(isinstance(query_rows, list) and len(query_rows) == 2
                and {row.get("analysis_role") for row in query_rows}
                == set(ROLES),
                f"cell is not balanced mixed-role: {path}")
        rows.extend(query_rows)
        cells.append({
            "history_index": history_index,
            "scene": identity[0],
            "episode": identity[1],
            "authority_order": cell["authority_order"],
            "audit_path": str(path),
            "audit_sha256": digest(path),
        })
    require(realized_indices == expected_index_set,
            "realized history-index set differs from frozen population")
    scenes = {str(row["scene"]) for row in rows}
    require(len(scenes) == expected_scenes,
            f"scene count {len(scenes)} != {expected_scenes}")
    require(len(rows) == expected_histories * 2,
            "mixed-role query denominator changed")

    order_counts = defaultdict(int)
    for cell in cells:
        order_counts[tuple(cell["authority_order"])] += 1
    if expected_histories > 1:
        require(set(order_counts) == {
            ("grant", "forced_reject_native"),
            ("forced_reject_native", "grant"),
        }, "both authority orders must be represented")
    require(max(order_counts.values()) - min(order_counts.values()) <= 1,
            "authority order is not balanced")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        gains = sum(int(row["paired_gain"]) for row in group)
        losses = sum(int(row["paired_loss"]) for row in group)
        return {
            "n": len(group),
            "native_success": sum(int(row["native_success"]) for row in group),
            "cec_success": sum(int(row["grant_success"]) for row in group),
            "native_sr": mean(int(row["native_success"]) for row in group),
            "cec_sr": mean(int(row["grant_success"]) for row in group),
            "risk_difference_pp": 100.0 * mean(
                int(row["grant_success"]) - int(row["native_success"])
                for row in group),
            "paired_gain": gains,
            "paired_loss": losses,
            "mcnemar_exact_p": exact_mcnemar(gains, losses),
            "native_spl": mean(spl(row, "native") for row in group),
            "cec_spl": mean(spl(row, "grant") for row in group),
            "native_mean_final_distance_m": mean(
                float(row["native_final_distance_m"]) for row in group),
            "cec_mean_final_distance_m": mean(
                float(row["grant_final_distance_m"]) for row in group),
            "native_mean_path_len_m": mean(
                float(row["native_path_len_m"]) for row in group),
            "cec_mean_path_len_m": mean(
                float(row["grant_path_len_m"]) for row in group),
            "native_mean_steps": mean(
                int(row["native_steps"]) for row in group),
            "cec_mean_steps": mean(int(row["grant_steps"]) for row in group),
            "cec_takeover_queries": sum(
                int(row["grant_takeover_plans"]) > 0 for row in group),
            "cec_takeover_plans": sum(
                int(row["grant_takeover_plans"]) for row in group),
            "first_shadow_accepts": sum(
                row["first_shadow_takeover"] is True for row in group),
        }

    by_role = {
        role: [row for row in rows if row["analysis_role"] == role]
        for role in ROLES
    }
    all_summary = summarize(rows)
    all_summary["scene_cluster_bootstrap_95ci_pp"] = scene_cluster_ci(rows)
    all_reject_rows = [
        row for row in rows if int(row["grant_takeover_plans"]) == 0]
    require(all(row.get("exact_fallback_trace_match") is True
                for row in all_reject_rows),
            "an all-reject query violated exact fallback")

    return {
        "schema_version": SCHEMA,
        "verified": True,
        "claim_scope": claim_scope,
        "run_root": str(run_root),
        "benchmark_manifest": str(benchmark_manifest),
        "benchmark_manifest_sha256": digest(benchmark_manifest),
        "controller": "vint",
        "treatment": "proof_bound_history_anchor_imagegoal",
        "reject_policy": "controller_native_exact",
        "histories": expected_histories,
        "scene_clusters": expected_scenes,
        "queries": len(rows),
        "role_counts": {role: len(by_role[role]) for role in ROLES},
        "history_indices": sorted(realized_indices),
        "authority_order_counts": {
            "/".join(key): value for key, value in sorted(order_counts.items())
        },
        "results": {
            "all": all_summary,
            **{role: summarize(by_role[role]) for role in ROLES},
        },
        "safety": {
            "novel_takeover_queries": sum(
                int(row["grant_takeover_plans"]) > 0
                for row in by_role["novel"]),
            "novel_takeover_plans": sum(
                int(row["grant_takeover_plans"])
                for row in by_role["novel"]),
            "all_reject_queries": len(all_reject_rows),
            "exact_fallback_trace_matches": sum(
                row.get("exact_fallback_trace_match") is True
                for row in all_reject_rows),
            "runtime_role_visibility": "none",
            "metric_depth_sensor_reads": 0,
        },
        "cells": cells,
    }


def parse_indices(raw: str) -> tuple[int, ...] | None:
    if not raw:
        return None
    values = tuple(int(value) for value in raw.split(","))
    require(len(values) == len(set(values)), "history indices are duplicated")
    require(all(value >= 0 for value in values), "history index is negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--expected-histories", type=int, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--history-indices", default="")
    parser.add_argument("--claim-scope", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.run_root.resolve(), args.benchmark_manifest.resolve(),
        expected_histories=args.expected_histories,
        expected_scenes=args.expected_scenes,
        history_indices=parse_indices(args.history_indices),
        claim_scope=args.claim_scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "verified": True,
        "histories": result["histories"],
        "native": result["results"]["all"]["native_success"],
        "cec": result["results"]["all"]["cec_success"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
