#!/usr/bin/env python3
"""Summarize the frozen reused-HM3D mixed Novel/Revisit evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = (
    "native", "raw_direct", "raw_fixed_bearing", "geometry_fixed", "certified"
)
ROLES = ("novel", "revisit")
FALLBACK_PLAN_KEYS = (
    "step", "requested_diffusion_seed", "diffusion_seed",
    "server_selected_idx", "trajectory_candidate_count",
    "selected_trajectory_sha256",
)
FALLBACK_METRIC_KEYS = (
    "reached", "steps", "path_len_m", "final_goal_dist_m", "termination_reason"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def cluster_interval(rows: list[dict], left: str, right: str) -> list[float]:
    scenes = sorted({row["scene"] for row in rows})
    grouped = {scene: [row for row in rows if row["scene"] == scene]
               for scene in scenes}
    rng = np.random.default_rng(20260818)
    values = np.empty(100_000, dtype=np.float64)
    for index in range(len(values)):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        numerator = 0
        denominator = 0
        for scene in chosen:
            group = grouped[str(scene)]
            numerator += sum(row["outcomes"][right] - row["outcomes"][left]
                             for row in group)
            denominator += len(group)
        values[index] = numerator / denominator
    return [100.0 * float(value)
            for value in np.quantile(values, [0.025, 0.975])]


def contrast(rows: list[dict], left: str, right: str) -> dict:
    gains = [row for row in rows
             if row["outcomes"][right] and not row["outcomes"][left]]
    losses = [row for row in rows
              if row["outcomes"][left] and not row["outcomes"][right]]
    return {
        "n": len(rows),
        "scene_count": len({row["scene"] for row in rows}),
        "left_successes": sum(row["outcomes"][left] for row in rows),
        "right_successes": sum(row["outcomes"][right] for row in rows),
        "gains": len(gains), "losses": len(losses),
        "risk_difference_pp": 100.0 * (len(gains) - len(losses)) / len(rows),
        "exact_mcnemar_two_sided_p": exact_mcnemar(len(gains), len(losses)),
        "scene_cluster_bootstrap_risk_difference_95":
            cluster_interval(rows, left, right),
        "gain_identities": [[row["scene"], row["episode"], row["role"]]
                            for row in gains],
        "loss_identities": [[row["scene"], row["episode"], row["role"]]
                            for row in losses],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def exact_fallback(
    native_row: dict[str, str], certified_row: dict[str, str],
    native_payload: dict, certified_payload: dict,
) -> bool:
    native_plans = native_payload["query_leg"]
    certified_plans = certified_payload["query_leg"]
    plans_equal = (
        len(native_plans) == len(certified_plans)
        and all(a.get(key) == b.get(key)
                for a, b in zip(native_plans, certified_plans)
                for key in FALLBACK_PLAN_KEYS)
    )
    rollout_equal = (
        native_payload["rollout_traces"]["query"]
        == certified_payload["rollout_traces"]["query"]
    )
    metric_equal = all(native_row[key] == certified_row[key]
                       for key in FALLBACK_METRIC_KEYS)
    return plans_equal and rollout_equal and metric_equal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    benchmark_root = args.root / "benchmarks/natural_direction"
    manifest = json.loads((benchmark_root / "manifest.json").read_text())
    expected = {(row["scene"], row["episode"]) for row in manifest["episodes"]}
    evaluation = args.root / "evaluation/natural_direction"
    episode_roots = sorted(path for path in evaluation.iterdir()
                           if path.is_dir() and path.name != "skipped")
    records = []
    observed = set()
    intervention = defaultdict(lambda: defaultdict(int))
    fallback = defaultdict(lambda: {"eligible": 0, "exact": 0})
    wall_times = defaultdict(list)
    for episode_root in episode_roots:
        completion_path = episode_root / "completion.json"
        if not completion_path.is_file():
            raise RuntimeError(f"incomplete evaluation: {episode_root}")
        if sha256_file(completion_path) != (
            episode_root / "completion.json.sha256"
        ).read_text().split()[0]:
            raise RuntimeError("completion hash mismatch")
        contract = json.loads((episode_root / "episode_contract.json").read_text())
        identity = (contract["scene"], contract["episode"])
        if identity in observed or identity not in expected:
            raise RuntimeError(f"unexpected or duplicate identity {identity}")
        observed.add(identity)
        completion = json.loads(completion_path.read_text())
        if completion["runtime_role_visibility"] != "none":
            raise RuntimeError("runtime role leaked")
        for arm, seconds in completion["wall_time_seconds"].items():
            wall_times[arm].append(float(seconds))
        metrics = {}
        payloads = {}
        for arm in ARMS:
            rows = read_csv(episode_root / arm / "metric.csv")
            if len(rows) != 2 or {row["analysis_role"] for row in rows} != set(ROLES):
                raise RuntimeError(f"{identity}/{arm}: role population changed")
            metrics[arm] = {row["analysis_role"]: row for row in rows}
            payloads[arm] = {}
            for row in rows:
                role = row["analysis_role"]
                path = episode_root / arm / (
                    f"{identity[1]}_{row['query_id']}_plans.json"
                )
                payload = json.loads(path.read_text())
                if payload["analysis_role_not_forwarded"] is not True:
                    raise RuntimeError("role forwarding audit failed")
                payloads[arm][role] = payload
        for role in ROLES:
            outcomes = {
                arm: int(metrics[arm][role]["reached"]) for arm in ARMS
            }
            records.append({"scene": identity[0], "episode": identity[1],
                            "role": role, "outcomes": outcomes})
            row = metrics["certified"][role]
            accepted = int(row["certificate_accept_plans"]) > 0
            takeover = int(row["adapter_takeover_plans"]) > 0
            intervention[role]["queries"] += 1
            intervention[role]["certificate_accept_queries"] += int(accepted)
            intervention[role]["takeover_queries"] += int(takeover)
            intervention[role]["runtime_failure_plans"] += int(
                row["runtime_failure_plans"])
            if not takeover:
                fallback[role]["eligible"] += 1
                fallback[role]["exact"] += int(exact_fallback(
                    metrics["native"][role], metrics["certified"][role],
                    payloads["native"][role], payloads["certified"][role],
                ))
    if observed != expected:
        raise RuntimeError(f"missing evaluations: {sorted(expected - observed)}")

    by_role = {role: [row for row in records if row["role"] == role]
               for role in ROLES}
    successes = {
        scope: {arm: sum(row["outcomes"][arm] for row in rows)
                for arm in ARMS}
        for scope, rows in {**by_role, "all": records}.items()
    }
    contrasts = {}
    for scope, rows in {**by_role, "all": records}.items():
        contrasts[scope] = {
            "certified_minus_native": contrast(rows, "native", "certified"),
            "certified_minus_raw_fixed": contrast(
                rows, "raw_fixed_bearing", "certified"),
            "certified_minus_geometry": contrast(
                rows, "geometry_fixed", "certified"),
        }
    result = {
        "schema_version": "hm3d_mixed_role_summary_v1_20260818",
        "scope": (
            "same-scene HM3D mixed-role safety extension; training-free but "
            "not a new scene-disjoint confirmation"
        ),
        "benchmark_manifest_sha256": sha256_file(
            benchmark_root / "manifest.json"),
        "histories": len(expected),
        "scene_count": len({scene for scene, _episode in expected}),
        "query_count": len(records),
        "role_counts": {role: len(rows) for role, rows in by_role.items()},
        "arms": list(ARMS),
        "arm_successes": successes,
        "contrasts": contrasts,
        "certified_intervention": {
            role: dict(values) for role, values in intervention.items()
        },
        "certified_exact_fallback": dict(fallback),
        "wall_time_seconds": {
            arm: {"median": float(np.median(values)),
                  "maximum": float(np.max(values))}
            for arm, values in wall_times.items()
        },
        "runtime_role_visibility": "none",
        "all_constructed_histories_evaluated": True,
    }
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "histories": result["histories"],
        "arm_successes": result["arm_successes"],
        "certified_intervention": result["certified_intervention"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
