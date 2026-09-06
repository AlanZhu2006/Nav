#!/usr/bin/env python3
"""Aggregate the frozen 48-history endpoint-versus-visual-route comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.run_hm3d_monotone_visual_route_dev import (
    ARMS,
    SCHEMA_VERSION as EPISODE_SCHEMA,
)


SCHEMA_VERSION = "hm3d_monotone_visual_route_dev_result_v1_20260902"
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file()
            and sidecar.read_text(encoding="utf-8").split()
            == [digest, path.name],
            f"invalid SHA receipt: {path}")
    return digest


def mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def spl(success: int, geodesic: float, path_length: float) -> float:
    return (float(success) * float(geodesic)
            / max(float(geodesic), float(path_length), 1e-9))


def cluster_interval(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    seed: int,
) -> list[float]:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(
            float(row[left] - row[right]))
    scenes = sorted(by_scene)
    require(bool(scenes), "cluster interval has no scenes")
    rng = np.random.default_rng(seed)
    samples = np.empty(100_000, dtype=np.float64)
    for index in range(len(samples)):
        selected = rng.integers(0, len(scenes), size=len(scenes))
        values = [
            value
            for scene_index in selected
            for value in by_scene[scenes[int(scene_index)]]
        ]
        samples[index] = float(np.mean(values))
    return [100.0 * float(value)
            for value in np.quantile(samples, [0.025, 0.975])]


def contrast(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    seed: int,
) -> dict[str, Any]:
    require(bool(rows), "paired contrast is empty")
    gains = sum(row[left] == 1 and row[right] == 0 for row in rows)
    losses = sum(row[left] == 0 and row[right] == 1 for row in rows)
    return {
        "episodes": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "left_successes": sum(row[left] for row in rows),
        "right_successes": sum(row[right] for row in rows),
        "left_SR": sum(row[left] for row in rows) / len(rows),
        "right_SR": sum(row[right] for row in rows) / len(rows),
        "paired_gains": gains,
        "paired_losses": losses,
        "paired_net_gain": gains - losses,
        "mcnemar_exact_two_sided_p": mcnemar(gains, losses),
        "scene_cluster_bootstrap_risk_difference_pp_95ci": (
            cluster_interval(rows, left=left, right=right, seed=seed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "development result already exists")
    require(sha256(arguments.manifest)
            == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "development population must contain 48 histories")

    records: list[dict[str, Any]] = []
    completion_sha256: dict[str, str] = {}
    for index, item in enumerate(manifest["episodes"]):
        root = (arguments.run_root / "development" /
                "monotone_visual_route" /
                f"{index:03d}_{item['scene']}_{item['episode']}")
        path = root / "completion.json"
        digest = verify_sidecar(path)
        completion_sha256[str(path.relative_to(arguments.run_root))] = digest
        completion = json.loads(path.read_text(encoding="utf-8"))
        require(completion.get("schema_version") == EPISODE_SCHEMA
                and int(completion["history_index"]) == index
                and completion["scene"] == item["scene"]
                and completion["episode"] == item["episode"]
                and completion["arms"] == list(ARMS)
                and completion.get("prefix_equality") is True
                and completion.get("target_proof_equality") is True
                and completion.get("runtime_role_visibility") == "none"
                and completion.get("distance_gate_present") is False
                and completion.get(
                    "endpoint_or_native_fallback_after_authorization") is False,
                f"episode contract changed at population index {index}")
        for role in ROLES:
            row: dict[str, Any] = {
                "population_index": index,
                "scene": item["scene"],
                "episode": item["episode"],
                "bin_name": item["bin_name"],
                "role": role,
            }
            for arm in ARMS:
                row[arm] = int(completion["outcomes"][arm][role])
                row[f"{arm}_geodesic_m"] = float(
                    completion["geodesic_m"][arm][role])
                row[f"{arm}_path_len_m"] = float(
                    completion["path_len_m"][arm][role])
                row[f"{arm}_spl"] = spl(
                    row[arm], row[f"{arm}_geodesic_m"],
                    row[f"{arm}_path_len_m"])
                row[f"{arm}_final_distance_m"] = float(
                    completion["final_distance_m"][arm][role])
                row[f"{arm}_steps"] = int(
                    completion["query_steps"][arm][role])
                row[f"{arm}_accept_plans"] = int(
                    completion["certificate_accept_plans"][arm][role])
            route_audit = completion["runtime_audits"][
                "mono_cec_visual_route"][role]["route"]
            row["visual_route_updates"] = (
                0 if route_audit is None
                else int(route_audit["visual_route_update_count"]))
            row["visual_route_explicitly_authorized"] = route_audit is not None
            records.append(row)

    revisit = [row for row in records if row["role"] == "revisit"]
    novel = [row for row in records if row["role"] == "novel"]
    bins = [spec["name"] for spec in manifest["contract"]["bins_m"]]
    require(set(bins) == {"0_to_20_m", "20_to_30_m", "30_to_50_m"},
            "distance strata changed")
    primary = {
        "all": contrast(
            revisit, left="mono_cec_visual_route",
            right="mono_cec_endpoint", seed=60902),
    }
    for offset, name in enumerate(bins, start=1):
        subset = [row for row in revisit if row["bin_name"] == name]
        require(len(subset) == 16, f"{name}: expected 16 Revisit histories")
        primary[name] = contrast(
            subset, left="mono_cec_visual_route",
            right="mono_cec_endpoint", seed=60902 + offset)

    contextual = {
        "endpoint_cec_vs_native_revisit": contrast(
            revisit, left="mono_cec_endpoint", right="mono_native",
            seed=61001),
        "visual_route_cec_vs_native_revisit": contrast(
            revisit, left="mono_cec_visual_route", right="mono_native",
            seed=61002),
        "visual_route_cec_vs_native_novel": contrast(
            novel, left="mono_cec_visual_route", right="mono_native",
            seed=61003),
    }
    all_primary = primary["all"]
    mid = primary["20_to_30_m"]
    long = primary["30_to_50_m"]
    short = primary["0_to_20_m"]
    development_gate = {
        "net_gain_at_least_6": all_primary["paired_net_gain"] >= 6,
        "gains_at_least_twice_losses": (
            all_primary["paired_gains"] >=
            2 * all_primary["paired_losses"]),
        "positive_net_gain_20_to_30_m": mid["paired_net_gain"] > 0,
        "positive_net_gain_30_to_50_m": long["paired_net_gain"] > 0,
        "short_stratum_losses_at_most_2": short["paired_losses"] <= 2,
        "silent_endpoint_or_native_fallback_count": 0,
        "runtime_failure_rate": 0.0,
    }
    development_gate["passed"] = all(
        value is True
        for key, value in development_gate.items()
        if key not in {
            "silent_endpoint_or_native_fallback_count",
            "runtime_failure_rate",
        }
    ) and development_gate["silent_endpoint_or_native_fallback_count"] == 0 \
        and development_gate["runtime_failure_rate"] <= 0.05

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed-population development; not confirmation",
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "histories": 48,
        "hidden_role_queries": 96,
        "raw_arm_role_rows": 288,
        "arms": list(ARMS),
        "primary_route_vs_endpoint_revisit": primary,
        "contextual_contrasts": contextual,
        "mean_metrics_by_arm_revisit": {
            arm: {
                "SR": float(np.mean([row[arm] for row in revisit])),
                "SPL": float(np.mean([row[f"{arm}_spl"]
                                      for row in revisit])),
                "path_length_m": float(np.mean([
                    row[f"{arm}_path_len_m"] for row in revisit])),
                "final_distance_m": float(np.mean([
                    row[f"{arm}_final_distance_m"] for row in revisit])),
            }
            for arm in ARMS
        },
        "visual_route_authorized_revisit_histories": sum(
            row["visual_route_explicitly_authorized"] for row in revisit),
        "visual_route_updates": sum(
            row["visual_route_updates"] for row in revisit),
        "development_gate": development_gate,
        "bootstrap_resamples": 100_000,
        "records": records,
        "completion_sha256": completion_sha256,
        "partial_results_reported": False,
        "fallback_completion_used": False,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        f"{sha256(arguments.out)}  {arguments.out.name}\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "primary": primary,
        "development_gate": development_gate,
        "out": str(arguments.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
