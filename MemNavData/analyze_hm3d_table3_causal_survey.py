#!/usr/bin/env python3
"""Aggregate the sealed HM3D causal-survey length evaluation.

This result is deliberately separate from the earlier actual-NavDP Goal-A
attempt.  Its history is a physically ordered causal RGB survey; query-time
NavDP and CEC consume RGB replay only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hm3d_table3_length_contract import validate_manifest


ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")
EXPECTED_EVALUATOR_ARM = {
    "mono_native": "native_sidecar",
    "mono_cec": "certified",
}
POPULATION_VARIANTS = {
    "query_population": {
        "verification_schema": (
            "hm3d_table3_causal_survey_population_verification_v1_20260830"
        ),
        "result_schema": "hm3d_table3_causal_survey_result_v1_20260830",
    },
    "merged_query_population": {
        "verification_schema": (
            "hm3d_table3_causal_survey_population_verification_v2_20260831"
        ),
        "result_schema": "hm3d_table3_causal_survey_result_v2_20260831",
    },
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(
        sidecar.is_file() and sidecar.read_text().split() == [digest, path.name],
        f"invalid SHA receipt: {path}",
    )
    return digest


def mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, losses) + 1)
    ) / 2**discordant
    return min(1.0, 2.0 * tail)


def spl(success: int, geodesic: float, path: float) -> float:
    return float(success) * geodesic / max(geodesic, path, 1e-9)


def cluster_interval(rows: list[dict], *, seed: int) -> list[float]:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(
            float(row["cec"] - row["native"])
        )
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
    return [100.0 * float(value) for value in np.quantile(samples, [0.025, 0.975])]


def read_metric_rows(root: Path, arm: str) -> dict[str, dict[str, str]]:
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(
        len(rows) == 2
        and {row["analysis_role"] for row in rows} == set(ROLES),
        f"{root}/{arm}: raw role rows changed",
    )
    return {row["analysis_role"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--population-relative-root",
        choices=tuple(POPULATION_VARIANTS),
        default="query_population",
        help="receipt-bound powered population inside --run-root",
    )
    args = parser.parse_args()
    require(not args.out.exists(), "causal-survey analysis exists")

    variant = POPULATION_VARIANTS[args.population_relative_root]
    population = args.run_root / args.population_relative_root
    verification_path = population / "independent_verification.json"
    verification_sha = verify_sidecar(verification_path)
    verification = json.loads(verification_path.read_text())
    require(
        verification.get("schema_version") == variant["verification_schema"]
        and verification.get("verified") is True
        and verification.get("formal_policy_evaluation_authorized") is True
        and verification.get("history_source")
        == "controlled_causal_rgb_geodesic_survey",
        "causal-survey population was not independently authorized",
    )
    manifest_path = population / "role_pairs/manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    require(
        manifest_sha == verification["benchmark_manifest_sha256"],
        "population verifier/manifest binding changed",
    )
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    require(len(manifest["episodes"]) == 48, "powered population changed")

    records: list[dict] = []
    for index, episode in enumerate(manifest["episodes"]):
        root = (
            args.run_root / "evaluation/natural_direction"
            / f"{index:03d}_{episode['scene']}_{episode['episode']}"
        )
        completion_path = root / "completion.json"
        verify_sidecar(completion_path)
        completion = json.loads(completion_path.read_text())
        expected_budget = max(
            600,
            math.ceil(
                2.5 * max(
                    float(query["geodesic_from_a_end_m"])
                    for query in episode["pairs"][0]["queries"]
                ) / 0.0376
            ),
        )
        require(
            completion.get("schema_version")
            == "hm3d_table3_causal_survey_history_v1_20260830"
            and int(completion["history_index"]) == index
            and completion["scene"] == episode["scene"]
            and completion["episode"] == episode["episode"]
            and completion["arms"] == list(ARMS)
            and completion.get("history_contract") == "causal_survey"
            and completion.get("shared_history_policy")
            == "controlled_causal_rgb_geodesic_survey_replay"
            and completion.get("runtime_geometry")
            == "content_addressed_pinned_navmesh"
            and completion.get("benchmark_manifest_sha256") == manifest_sha
            and completion.get("prefix_equality") is True
            and completion.get("runtime_role_visibility") == "none"
            and completion.get("smoke") is False
            and int(completion.get("max_steps", -1)) == expected_budget,
            f"paired completion contract changed at {index}",
        )
        raw = {arm: read_metric_rows(root, arm) for arm in ARMS}
        queries = {
            query["analysis_role"]: query
            for query in episode["pairs"][0]["queries"]
        }
        for role in ROLES:
            native, cec = raw["mono_native"][role], raw["mono_cec"][role]
            for arm, row in (("mono_native", native), ("mono_cec", cec)):
                require(
                    row["scene"] == episode["scene"]
                    and row["episode"] == episode["episode"]
                    and row["query_id"] == queries[role]["query_id"]
                    and row["arm"] == EXPECTED_EVALUATOR_ARM[arm]
                    and row["navdp_depth_source"] == "monocular_sidecar"
                    and int(row["metric_depth_sensor_consumed_any"]) == 0
                    and int(row["runtime_failure_plans"]) == 0
                    and int(row["shared_A_hashes_ok"]) == 1
                    and int(row["shared_A_diffusion_samples"]) == 0,
                    f"raw runtime contract changed at {index}/{arm}/{role}",
                )
            require(
                native["seed"] == cec["seed"]
                and native["shared_A_frames"] == cec["shared_A_frames"]
                and native["shared_A_decision_frames"]
                == cec["shared_A_decision_frames"]
                and abs(float(native["geodesic_m"]) - float(cec["geodesic_m"]))
                <= 1e-12
                and abs(
                    float(native["geodesic_m"])
                    - float(queries[role]["geodesic_from_a_end_m"])
                ) <= 0.05,
                f"paired geometry/history changed at {index}/{role}",
            )
            records.append({
                "population_index": index,
                "scene": episode["scene"],
                "episode": episode["episode"],
                "bin_name": episode["bin_name"],
                "role": role,
                "native": int(native["reached"]),
                "cec": int(cec["reached"]),
                "native_geodesic_m": float(native["geodesic_m"]),
                "cec_geodesic_m": float(cec["geodesic_m"]),
                "native_path_m": float(native["path_len_m"]),
                "cec_path_m": float(cec["path_len_m"]),
                "certificate_accept_plans": int(cec["certificate_accept_plans"]),
                "fully_rejected_exact_native": bool(
                    completion["fully_rejected_exact_native"][role]
                ),
            })

    bins = {}
    for bin_index, spec in enumerate(manifest["contract"]["bins_m"]):
        name = spec["name"]
        bin_rows = [row for row in records if row["bin_name"] == name]
        require(len(bin_rows) == 32, f"{name}: powered query count changed")
        bins[name] = {}
        for role_index, role in enumerate(("all", *ROLES)):
            rows = bin_rows if role == "all" else [
                row for row in bin_rows if row["role"] == role
            ]
            gains = sum(row["cec"] == 1 and row["native"] == 0 for row in rows)
            losses = sum(row["cec"] == 0 and row["native"] == 1 for row in rows)
            rejected = [row for row in rows if row["certificate_accept_plans"] == 0]
            bins[name][role] = {
                "queries": len(rows),
                "scene_clusters": len({row["scene"] for row in rows}),
                "mono_native_SR": sum(row["native"] for row in rows) / len(rows),
                "mono_cec_SR": sum(row["cec"] for row in rows) / len(rows),
                "risk_difference_pp": 100.0 * sum(
                    row["cec"] - row["native"] for row in rows
                ) / len(rows),
                "scene_cluster_bootstrap_95_pp": cluster_interval(
                    rows, seed=20260830 + 10 * bin_index + role_index
                ),
                "mono_native_SPL": sum(
                    spl(row["native"], row["native_geodesic_m"], row["native_path_m"])
                    for row in rows
                ) / len(rows),
                "mono_cec_SPL": sum(
                    spl(row["cec"], row["cec_geodesic_m"], row["cec_path_m"])
                    for row in rows
                ) / len(rows),
                "cec_vs_native_gains": gains,
                "cec_vs_native_losses": losses,
                "mcnemar_exact_p": mcnemar(gains, losses),
                "certificate_accept_queries": sum(
                    row["certificate_accept_plans"] > 0 for row in rows
                ),
                "certificate_reject_queries": len(rejected),
                "exact_fallback_queries_among_rejects": sum(
                    row["fully_rejected_exact_native"] for row in rejected
                ),
            }

    result = {
        "schema_version": variant["result_schema"],
        "scope": "controlled causal-RGB survey; not actual NavDP Goal-A history",
        "history_source": "controlled_causal_rgb_geodesic_survey",
        "population_verification_sha256": verification_sha,
        "population_relative_root": args.population_relative_root,
        "benchmark_manifest_sha256": manifest_sha,
        "histories": 48,
        "queries": 96,
        "raw_arm_role_rows": 192,
        "bins": bins,
        "records": records,
        "bootstrap_resamples": 100_000,
        "partial_results_reported": False,
        "fallback_completion_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )


if __name__ == "__main__":
    main()
