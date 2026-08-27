#!/usr/bin/env python3
"""Independent recomputation of the fresh actual-online double-Revisit gate.

This verifier intentionally does not import the production audit module.  It
reconstructs the causal denominator and paired statistics directly from the
sealed manifest, four-arm metrics, and B-prefix receipts, then checks the
frozen HPC report field by field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


ARMS = ("native", "full_memory", "memory_b_native_c", "certified")
REPORT_SCHEMA = "shared_online_double_revisit_fresh_audit_v1_20260813"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def metric_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"metric file must have one row: {path}")
    return rows[0]


def boolean(value: Any) -> bool:
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False", ""):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def exact_pair(first: Iterable[bool], second: Iterable[bool]) -> dict[str, Any]:
    left = list(map(bool, first))
    right = list(map(bool, second))
    require(len(left) == len(right), "paired vectors differ in length")
    gain = sum(a and not b for a, b in zip(left, right))
    loss = sum(b and not a for a, b in zip(left, right))
    discordant = gain + loss
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(gain, loss) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    else:
        p_value = 1.0
    return {
        "N": len(left),
        "first_success": sum(left),
        "second_success": sum(right),
        "risk_difference_pp": (
            100.0 * (sum(left) - sum(right)) / len(left)
            if left else None),
        "gain": gain,
        "loss": loss,
        "discordant": discordant,
        "exact_mcnemar_p": float(p_value),
    }


def cluster_interval(rows: Iterable[tuple[str, bool, bool]], *, seed: int,
                     resamples: int) -> dict[str, Any]:
    values = list(rows)
    require(bool(values), "cluster population is empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene, first, second in values:
        grouped[str(scene)].append(float(first) - float(second))
    scenes = sorted(grouped)
    generator = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sampled = generator.integers(0, len(scenes), size=len(scenes))
        episode_values = [
            value for scene_index in sampled
            for value in grouped[scenes[int(scene_index)]]
        ]
        draws[index] = float(np.mean(episode_values))
    return {
        "clusters": len(scenes),
        "episodes": len(values),
        "seed": int(seed),
        "resamples": int(resamples),
        "risk_difference_pp": 100.0 * float(np.mean([
            float(first) - float(second)
            for _scene, first, second in values
        ])),
        "ci95_pp": [
            100.0 * float(np.quantile(draws, 0.025)),
            100.0 * float(np.quantile(draws, 0.975)),
        ],
    }


def _assert_close(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, float):
        require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12),
                f"report mismatch at {path}: {actual} != {expected}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"report list mismatch at {path}")
        for index, (observed, target) in enumerate(zip(actual, expected)):
            _assert_close(observed, target, f"{path}[{index}]")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict), f"report object mismatch at {path}")
        for key, target in expected.items():
            require(key in actual, f"report lacks {path}.{key}")
            _assert_close(actual[key], target, f"{path}.{key}")
    else:
        require(actual == expected,
                f"report mismatch at {path}: {actual!r} != {expected!r}")


def recompute(run_root: Path, *, expected_manifest_sha256: str,
              expected_episodes: int) -> tuple[dict[str, Any], list[dict]]:
    manifest_path = run_root / "prepared" / "benchmark" / "manifest.json"
    require(sha256(manifest_path) == expected_manifest_sha256,
            "benchmark manifest SHA changed")
    manifest = read_json(manifest_path)
    identities = [
        (str(row["scene"]), str(row["episode"]))
        for row in manifest.get("episodes", [])
    ]
    require(len(identities) == expected_episodes,
            "manifest episode count changed")
    require(len(set(identities)) == expected_episodes,
            "manifest identities are not unique")
    episode_dirs = sorted(
        path for path in (run_root / "scenes").iterdir() if path.is_dir())
    require(len(episode_dirs) == expected_episodes,
            "completed episode directory count changed")

    records = []
    observed = []
    for episode_dir in episode_dirs:
        contract = read_json(episode_dir / "episode_contract.json")
        index = int(contract["selection_index"])
        require(0 <= index < expected_episodes, "invalid selection index")
        identity = identities[index]
        require((str(contract["scene"]), str(contract["episode"])) == identity,
                f"episode identity changed: {episode_dir.name}")
        observed.append(identity)
        metrics = {}
        plans = {}
        arm_identities = set()
        for arm in ARMS:
            root = episode_dir / arm
            metric = metric_row(root / "metric.csv")
            matches = list(root.glob("episode_*_plans.json"))
            require(len(matches) == 1,
                    f"plan receipt count changed: {episode_dir.name}/{arm}")
            plan = read_json(matches[0])
            arm_identities.add((
                str(metric["scene"]), str(metric["episode"]),
                int(metric["seed"])))
            require((metric["scene"], metric["episode"]) == identity,
                    f"metric identity changed: {episode_dir.name}/{arm}")
            require(boolean(metric["shared_A_hashes_ok"]),
                    f"shared A hash failed: {episode_dir.name}/{arm}")
            require(int(metric["shared_A_replay_diffusion_samples"]) == 0,
                    f"shared A sampled diffusion: {episode_dir.name}/{arm}")
            reached_b = boolean(metric["reached_B"])
            input_ok = boolean(metric["c_effective_input_contract_ok"])
            evaluated = boolean(metric["C_evaluated"])
            require(evaluated == (reached_b and input_ok),
                    f"C censoring changed: {episode_dir.name}/{arm}")
            metrics[arm] = {
                "B": reached_b,
                "C_input_ok": input_ok,
                "C_evaluated": evaluated,
                "C": boolean(metric["reached_C"]),
                "joint": boolean(metric["joint_success"]),
            }
            plans[arm] = plan
        require(len(arm_identities) == 1,
                f"arm identities differ: {episode_dir.name}")
        require(next(iter(arm_identities))[2] == int(contract["episode_seed"]),
                f"arm seed differs: {episode_dir.name}")

        full_plan = plans["full_memory"]
        ablation_plan = plans["memory_b_native_c"]
        for field in ("legB",):
            require(full_plan[field] == ablation_plan[field],
                    f"B plans differ: {episode_dir.name}")
        for field in ("rollout_traces", "memory_traces"):
            require(full_plan[field]["legB"] == ablation_plan[field]["legB"],
                    f"B {field} differ: {episode_dir.name}")
        for field in ("B", "C_input_ok", "C_evaluated"):
            require(metrics["full_memory"][field]
                    == metrics["memory_b_native_c"][field],
                    f"pre-C causal field differs: {episode_dir.name}/{field}")
        records.append({
            "selection_index": index,
            "scene": identity[0],
            "episode": identity[1],
            "arms": metrics,
        })
    require(sorted(observed) == sorted(identities),
            "completed identities differ from manifest")
    records.sort(key=lambda row: row["selection_index"])

    arm_summary = {}
    for arm in ARMS:
        rows = [record["arms"][arm] for record in records]
        eligible = [row for row in rows if row["C_evaluated"]]
        arm_summary[arm] = {
            "episodes": len(rows),
            "B_success": sum(row["B"] for row in rows),
            "C_eligible": len(eligible),
            "C_success": sum(row["C"] for row in eligible),
            "joint_success": sum(row["joint"] for row in rows),
        }
    causal = [
        record for record in records
        if record["arms"]["full_memory"]["C_evaluated"]
    ]
    primary_rows = [(
        record["scene"],
        record["arms"]["full_memory"]["C"],
        record["arms"]["memory_b_native_c"]["C"],
    ) for record in causal]
    require(bool(primary_rows), "primary causal denominator is empty")
    primary = exact_pair(
        [row[1] for row in primary_rows], [row[2] for row in primary_rows])
    result = {
        "episodes": len(records),
        "scene_clusters": len({row["scene"] for row in records}),
        "arm_summary": arm_summary,
        "primary_contrast": primary,
    }
    return result, records


def verify(run_root: Path, *, expected_manifest_sha256: str,
           expected_episodes: int, expected_report_sha256: str | None,
           report_path: Path | None = None) -> dict:
    report_path = (run_root / "report.json" if report_path is None
                   else Path(report_path))
    require(report_path.is_file(), "frozen report is missing")
    report_sha = sha256(report_path)
    if expected_report_sha256 is not None:
        require(report_sha == expected_report_sha256,
                "frozen report SHA changed")
    report = read_json(report_path)
    require(report.get("schema_version") == REPORT_SCHEMA,
            "frozen report schema changed")
    require(report.get("audit_ok") is True, "frozen audit did not pass")
    require(report.get("benchmark_manifest_sha256")
            == expected_manifest_sha256, "report manifest SHA changed")
    computed, records = recompute(
        run_root, expected_manifest_sha256=expected_manifest_sha256,
        expected_episodes=expected_episodes)
    _assert_close(report["episodes"], computed["episodes"], "episodes")
    _assert_close(
        report["scene_clusters"], computed["scene_clusters"],
        "scene_clusters")
    _assert_close(report["arm_summary"], computed["arm_summary"],
                  "arm_summary")
    _assert_close(
        report["primary_contrast"], computed["primary_contrast"],
        "primary_contrast")

    primary_rows = [(
        record["scene"],
        record["arms"]["full_memory"]["C"],
        record["arms"]["memory_b_native_c"]["C"],
    ) for record in records
        if record["arms"]["full_memory"]["C_evaluated"]]
    cluster_report = report["primary_contrast"]["scene_cluster_bootstrap"]
    independent_interval = cluster_interval(
        primary_rows, seed=int(cluster_report["seed"]),
        resamples=int(cluster_report["resamples"]))
    _assert_close(cluster_report, independent_interval,
                  "primary_contrast.scene_cluster_bootstrap")

    report_records = {
        int(row["selection_index"]): row for row in report.get("records", [])}
    require(len(report_records) == expected_episodes,
            "frozen report record count changed")
    for record in records:
        frozen = report_records[record["selection_index"]]
        for arm in ARMS:
            for field in ("B", "C_input_ok", "C_evaluated", "C", "joint"):
                _assert_close(
                    frozen["arms"][arm][field],
                    record["arms"][arm][field],
                    f"records[{record['selection_index']}].{arm}.{field}")
    return {
        "schema_version": (
            "shared_online_double_revisit_independent_verification_v1_20260813"),
        "verified": True,
        "run_root": str(run_root.resolve()),
        "report_sha256": report_sha,
        "manifest_sha256": expected_manifest_sha256,
        "episodes": computed["episodes"],
        "scene_clusters": computed["scene_clusters"],
        "formal_power_target_met": report.get("formal_power_target_met"),
        "arm_summary": computed["arm_summary"],
        "primary_contrast": {
            **computed["primary_contrast"],
            "scene_cluster_bootstrap": independent_interval,
        },
        "development_or_blind_read": False,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument(
        "--report", type=Path,
        help="report to verify; defaults to RUN_ROOT/report.json",
    )
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = verify(
        args.run_root,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_episodes=args.expected_episodes,
        expected_report_sha256=args.expected_report_sha256,
        report_path=args.report)
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
