#!/usr/bin/env python3
"""Summarize the consumed-development Novel causal-control experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from novel_memory_direction_control import ARMS, sha256_file, validate_control_manifest


SCHEMA_VERSION = "novel_memory_direction_summary_v1_20260816"
CONTRASTS = (
    ("raw_factual_history", "raw_randomized_bearing"),
    ("raw_factual_history", "raw_deranged_history"),
    ("raw_factual_history", "native"),
    ("raw_deranged_history", "native"),
    ("raw_randomized_bearing", "native"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def clustered_ci(
    rows: list[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    repetitions: int = 20000,
    seed: int = 20260816,
) -> list[float]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    scenes = sorted(by_scene)
    require(bool(scenes), "no scene clusters")
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = [rng.choice(scenes) for _ in scenes]
        numerator = 0.0
        denominator = 0
        for scene in sampled:
            for row in by_scene[scene]:
                numerator += row["outcomes"][arm_a] - row["outcomes"][arm_b]
                denominator += 1
        values.append(numerator / denominator)
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return [float(low), float(high)]


def read_completion(path: Path, manifest_sha: str) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing completion hash: {path.parent}")
    tokens = sidecar.read_text().split()
    require(len(tokens) == 2 and tokens[1] == path.name, "bad completion hash receipt")
    require(tokens[0] == sha256_file(path), f"completion changed: {path.parent}")
    payload = json.loads(path.read_text())
    require(
        payload.get("schema_version")
        == "novel_memory_direction_completion_v1_20260816",
        "completion schema changed",
    )
    require(payload.get("control_manifest_sha256") == manifest_sha, "manifest receipt changed")
    for key in (
        "prefix_equality", "factual_fifo_equality",
        "deranged_sidecar_verified", "randomized_bearing_verified",
        "zero_takeover_exact_fallback_verified",
    ):
        require(payload.get(key) is True, f"completion audit failed: {key}")
    require(payload.get("confirmation_claim_allowed") is False, "development result claims confirmation")
    require(set(payload["outcomes"]) == set(ARMS), "completion arm set changed")
    require(all(value in (0, 1) for value in payload["outcomes"].values()), "invalid outcome")
    return payload


def summarize(run_root: Path, manifest_path: Path, out: Path) -> dict[str, Any]:
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_control_manifest(manifest)
    require(
        manifest.get("evaluation_stage") == "consumed_development_mechanism_only",
        "wrong evaluation stage",
    )
    expected = {
        (str(row["scene"]), str(row["episode"])): row
        for row in manifest["episodes"]
    }
    completions = sorted((run_root / "evaluation").glob("*/completion.json"))
    require(len(completions) == len(expected), "formal population is incomplete")
    rows = []
    for path in completions:
        payload = read_completion(path, manifest_sha)
        identity = (str(payload["scene"]), str(payload["episode"]))
        require(identity in expected, f"unexpected completion: {identity}")
        contract = json.loads((path.parent / "episode_contract.json").read_text())
        require(contract["arm_order"] == expected[identity]["arm_order"], "arm order changed")
        rows.append({
            "scene": identity[0],
            "episode": identity[1],
            "query_id": str(payload["query_id"]),
            "outcomes": {arm: int(payload["outcomes"][arm]) for arm in ARMS},
            "takeover_plans": {
                arm: int(payload["takeover_plans"][arm]) for arm in ARMS
            },
            "fallback_plans": {
                arm: int(payload["fallback_plans"][arm]) for arm in ARMS
            },
            "plan_count": {
                arm: int(payload["plan_count"][arm]) for arm in ARMS
            },
            "geodesic_m": {
                arm: float(payload["geodesic_m"][arm]) for arm in ARMS
            },
            "path_length_m": {
                arm: float(payload["path_length_m"][arm]) for arm in ARMS
            },
            "final_euclidean_distance_m": {
                arm: float(payload["final_distance_m"][arm]) for arm in ARMS
            },
            "final_geodesic_m": {
                arm: float(payload["final_geodesic_m"][arm]) for arm in ARMS
            },
            "steps": {arm: int(payload["steps"][arm]) for arm in ARMS},
            "spl": {arm: float(payload["spl"][arm]) for arm in ARMS},
            "wall_time_seconds": {
                arm: float(payload["wall_time_seconds"][arm]) for arm in ARMS
            },
            "result_dir": str(path.parent.resolve()),
        })
    require(
        {(row["scene"], row["episode"]) for row in rows} == set(expected),
        "completion identities are not the frozen population",
    )

    arm_metrics = {}
    for arm in ARMS:
        successes = sum(row["outcomes"][arm] for row in rows)
        takeover_episodes = sum(row["takeover_plans"][arm] > 0 for row in rows)
        arm_metrics[arm] = {
            "successes": successes,
            "episodes": len(rows),
            "success_rate": successes / len(rows),
            "takeover_episodes": takeover_episodes,
            "takeover_coverage": takeover_episodes / len(rows),
            "takeover_plans": sum(row["takeover_plans"][arm] for row in rows),
            "fallback_plans": sum(row["fallback_plans"][arm] for row in rows),
            "plan_count": sum(row["plan_count"][arm] for row in rows),
            "mean_spl": sum(row["spl"][arm] for row in rows) / len(rows),
            "mean_path_length_m": (
                sum(row["path_length_m"][arm] for row in rows) / len(rows)
            ),
            "mean_final_geodesic_m": (
                sum(row["final_geodesic_m"][arm] for row in rows) / len(rows)
            ),
            "mean_steps": sum(row["steps"][arm] for row in rows) / len(rows),
            "wall_time_seconds": sum(row["wall_time_seconds"][arm] for row in rows),
        }

    contrasts = {}
    for arm_a, arm_b in CONTRASTS:
        gains = sum(
            row["outcomes"][arm_a] == 1 and row["outcomes"][arm_b] == 0
            for row in rows
        )
        losses = sum(
            row["outcomes"][arm_a] == 0 and row["outcomes"][arm_b] == 1
            for row in rows
        )
        differences = [
            row["outcomes"][arm_a] - row["outcomes"][arm_b]
            for row in rows
        ]
        contrasts[f"{arm_a}-minus-{arm_b}"] = {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "paired_gains": int(gains),
            "paired_losses": int(losses),
            "risk_difference": sum(differences) / len(differences),
            "exact_mcnemar_p": exact_mcnemar(int(gains), int(losses)),
            "scene_cluster_bootstrap_95ci": clustered_ci(rows, arm_a, arm_b),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_stage": "consumed_development_mechanism_only",
        "confirmation_claim_allowed": False,
        "method_or_threshold_selection_allowed": False,
        "decision_authority": "none_descriptive_causal_mechanism_only",
        "control_manifest": str(manifest_path.resolve()),
        "control_manifest_sha256": manifest_sha,
        "population": {
            "episodes": len(rows),
            "scenes": len({row["scene"] for row in rows}),
        },
        "arm_metrics": arm_metrics,
        "contrasts": contrasts,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    out.with_name(out.name + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n"
    )

    table_path = out.with_suffix(".csv")
    with table_path.open("w", newline="") as handle:
        fieldnames = ["scene", "episode", "query_id", *ARMS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scene": row["scene"], "episode": row["episode"],
                "query_id": row["query_id"], **row["outcomes"],
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_root, args.manifest, args.out)
    print(json.dumps({
        "population": summary["population"],
        "arm_metrics": summary["arm_metrics"],
        "contrasts": summary["contrasts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
