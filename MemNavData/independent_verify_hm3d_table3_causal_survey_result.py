#!/usr/bin/env python3
"""Recompute the causal-survey length result from all raw arm-role files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hm3d_table3_length_contract import (
    RUNTIME_VISIBLE_QUERY_FIELDS,
    validate_manifest,
)


ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")
EXPECTED_EVALUATOR_ARM = {
    "mono_native": "native_sidecar",
    "mono_cec": "certified",
}
POPULATION_VARIANTS = {
    "query_population": {
        "population_schema": (
            "hm3d_table3_causal_survey_population_verification_v1_20260830"
        ),
        "summary_schema": "hm3d_table3_causal_survey_result_v1_20260830",
        "verifier_schema": (
            "hm3d_table3_causal_survey_result_verification_v1_20260830"
        ),
    },
    "merged_query_population": {
        "population_schema": (
            "hm3d_table3_causal_survey_population_verification_v2_20260831"
        ),
        "summary_schema": "hm3d_table3_causal_survey_result_v2_20260831",
        "verifier_schema": (
            "hm3d_table3_causal_survey_result_verification_v2_20260831"
        ),
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


def close(first: float, second: float, label: str) -> None:
    require(abs(float(first) - float(second)) <= 1e-12,
            f"{label} does not reproduce")


def load_rows(root: Path, arm: str) -> dict[str, dict[str, str]]:
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(
        len(rows) == 2 and {row["analysis_role"] for row in rows} == set(ROLES),
        f"{root}/{arm}: raw role rows changed",
    )
    return {row["analysis_role"]: row for row in rows}


def audit_mono_plans(plans: list[dict], label: str) -> None:
    require(bool(plans), f"{label}: no policy plans")
    scale_hashes = set()
    for plan in plans:
        receipt = plan.get("monocular_depth_receipt")
        require(
            plan.get("navdp_depth_source") == "monocular_sidecar"
            and plan.get("metric_depth_sensor_consumed") is False
            and isinstance(receipt, dict)
            and receipt.get("depth_contract") == "raw_lingbot_depth_first40_v1"
            and receipt.get("metric_depth_sensor_consumed") is False
            and int(receipt.get("frame_index", -1)) >= 40
            and receipt.get("scale_active") is True,
            f"{label}: mono-depth wire contract changed",
        )
        scale = receipt.get("scale_receipt")
        require(
            isinstance(scale, dict)
            and scale.get("scale_evidence_contract")
            == "causal_first_prefix_rgb_only_v1"
            and scale.get("whole_episode_ground_cache_consumed") is False
            and receipt.get("scale_receipt_sha256"),
            f"{label}: causal scale receipt changed",
        )
        scale_hashes.add(str(receipt["scale_receipt_sha256"]))
    require(len(scale_hashes) == 1, f"{label}: query scale was not immutable")


def audit_raw_outcome(
    row: dict[str, str], payload: dict, label: str,
) -> tuple[int, int]:
    """Recompute success and intervention counts from the raw plan payload."""
    result = payload.get("query_result")
    require(isinstance(result, dict), f"{label}: query result is missing")
    reached = int(row["reached"])
    require(reached in (0, 1), f"{label}: invalid reached value")
    final_distance = float(row["final_goal_dist_m"])
    require(
        reached == int(final_distance <= 1.0)
        and bool(result.get("reached")) == bool(reached)
        and abs(float(result["final_goal_dist_m"]) - final_distance) <= 1e-12
        and abs(float(result["path_len_m"]) - float(row["path_len_m"])) <= 1e-12
        and int(result["steps"]) == int(row["steps"])
        and result["termination_reason"] == row["termination_reason"],
        f"{label}: success was not reproduced from raw final distance",
    )
    plans = payload["query_leg"]
    accepts = sum(
        plan.get("certified_relocalization_accepted") is True
        or plan.get("cec_takeover") is True
        for plan in plans
    )
    failures = sum(
        plan.get("certified_relocalization_reason")
        == "certificate_endpoint_failure"
        or plan.get("learned_pi3x_relocalization_ok") is False
        or plan.get("cec_reason") == "certificate_endpoint_failure"
        for plan in plans
    )
    require(
        accepts == int(row["certificate_accept_plans"])
        and failures == int(row["runtime_failure_plans"])
        and failures == 0
        and all(plan.get("role_label_visible") is False for plan in plans),
        f"{label}: intervention/runtime recount changed",
    )
    return accepts, failures


def replay_fingerprint(payload: dict) -> str:
    replay_fields = (
        "all_rgb_hashes_verified", "decision_frames", "decision_steps",
        "diffusion_samples_during_replay", "navdp_memory_size",
        "navdp_queue_lengths", "online_frames",
    )
    replay = {field: payload["replay"][field] for field in replay_fields}
    value = {
        "replay": replay,
        "legA": payload["legA"],
        "legA_memory": payload["memory_traces"]["legA"],
        "legA_rollout": payload["rollout_traces"]["legA"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exact_reject_fallback(cec: dict, native: dict, label: str) -> bool:
    cec_plans, native_plans = cec["query_leg"], native["query_leg"]
    require(len(cec_plans) == len(native_plans),
            f"{label}: rejected plan counts differ")
    for index, (left, right) in enumerate(zip(cec_plans, native_plans)):
        for field in (
            "requested_diffusion_seed", "diffusion_seed",
            "selected_trajectory_sha256",
        ):
            require(left.get(field) == right.get(field),
                    f"{label}/plan{index}: rejected {field} changed")
    require(
        cec["rollout_traces"]["query"] == native["rollout_traces"]["query"]
        and cec["query_result"] == native["query_result"],
        f"{label}: rejected physical trace differs",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--population-relative-root",
        choices=tuple(POPULATION_VARIANTS),
        default="query_population",
        help="receipt-bound powered population inside --run-root",
    )
    args = parser.parse_args()
    require(not args.out.exists(), "causal-survey result verification exists")
    variant = POPULATION_VARIANTS[args.population_relative_root]

    summary_sha = verify_sidecar(args.summary)
    summary = json.loads(args.summary.read_text())
    require(
        summary.get("schema_version") == variant["summary_schema"]
        and summary.get("scope")
        == "controlled causal-RGB survey; not actual NavDP Goal-A history"
        and summary.get("history_source")
        == "controlled_causal_rgb_geodesic_survey"
        and summary.get("histories") == 48
        and summary.get("queries") == 96
        and summary.get("raw_arm_role_rows") == 192
        and summary.get("bootstrap_resamples") == 100_000
        and summary.get("partial_results_reported") is False
        and summary.get("fallback_completion_used") is False,
        "causal-survey summary contract changed",
    )
    require(
        summary.get("population_relative_root")
        == args.population_relative_root,
        "summary population root changed",
    )

    population = args.run_root / args.population_relative_root
    verification_path = population / "independent_verification.json"
    verification_sha = verify_sidecar(verification_path)
    verification = json.loads(verification_path.read_text())
    manifest_path = population / "role_pairs/manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    require(
        verification.get("schema_version") == variant["population_schema"]
        and verification.get("verified") is True
        and verification.get("formal_policy_evaluation_authorized") is True
        and verification.get("history_source")
        == "controlled_causal_rgb_geodesic_survey"
        and verification.get("query_policy_outcomes_read") is False
        and verification.get("fallback_completion_allowed") is False
        and verification.get("benchmark_manifest_sha256") == manifest_sha
        and summary["population_verification_sha256"] == verification_sha
        and summary["benchmark_manifest_sha256"] == manifest_sha,
        "population/result provenance changed",
    )
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    require(len(manifest["episodes"]) == 48, "powered population changed")

    records: list[dict] = []
    raw_rows = 0
    completion_artifacts: dict[str, str] = {}
    raw_artifacts: dict[str, str] = {}

    def bind_artifact(target: dict[str, str], path: Path) -> str:
        relative = str(path.relative_to(args.run_root))
        digest = sha256(path)
        require(
            relative not in target or target[relative] == digest,
            f"artifact path changed during verification: {relative}",
        )
        target[relative] = digest
        return digest

    for index, episode in enumerate(manifest["episodes"]):
        root = (
            args.run_root / "evaluation/natural_direction"
            / f"{index:03d}_{episode['scene']}_{episode['episode']}"
        )
        completion_path = root / "completion.json"
        completion_digest = verify_sidecar(completion_path)
        bind_artifact(completion_artifacts, completion_path)
        require(
            completion_artifacts[str(completion_path.relative_to(args.run_root))]
            == completion_digest,
            f"completion digest changed at {index}",
        )
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
            and completion.get("runtime_navmesh_sha256")
            == episode["runtime_navmesh_sha256"]
            and completion.get("benchmark_manifest_sha256") == manifest_sha
            and completion.get("prefix_equality") is True
            and completion.get("runtime_role_visibility") == "none"
            and completion.get("smoke") is False
            and int(completion.get("max_steps", -1)) == expected_budget
            and float(completion.get("success_distance_m", -1)) == 1.0
            and int(completion.get("exec_horizon", -1)) == 8,
            f"completion contract changed at {index}",
        )
        raw = {arm: load_rows(root, arm) for arm in ARMS}
        for arm in ARMS:
            bind_artifact(raw_artifacts, root / arm / "metric.csv")
        queries = {
            query["analysis_role"]: query
            for query in episode["pairs"][0]["queries"]
        }
        replay_hash = None
        for role in ROLES:
            payloads = {}
            for arm in ARMS:
                row = raw[arm][role]
                raw_rows += 1
                metric_path = root / arm / "metric.csv"
                plan_path = root / arm / (
                    f"{episode['episode']}_{queries[role]['query_id']}_plans.json"
                )
                bind_artifact(raw_artifacts, metric_path)
                bind_artifact(raw_artifacts, plan_path)
                payload = json.loads(plan_path.read_text())
                payloads[arm] = payload
                require(
                    row["scene"] == episode["scene"]
                    and row["episode"] == episode["episode"]
                    and row["query_id"] == queries[role]["query_id"]
                    and row["analysis_role"] == role
                    and row["arm"] == EXPECTED_EVALUATOR_ARM[arm]
                    and row["navdp_depth_source"] == "monocular_sidecar"
                    and int(row["metric_depth_sensor_consumed_any"]) == 0
                    and int(row["runtime_failure_plans"]) == 0
                    and int(row["shared_A_hashes_ok"]) == 1
                    and int(row["shared_A_diffusion_samples"]) == 0
                    and payload.get("analysis_role_not_forwarded") is True
                    and set(payload.get("query_runtime_fields", []))
                    == set(RUNTIME_VISIBLE_QUERY_FIELDS)
                    and payload.get("arm") == EXPECTED_EVALUATOR_ARM[arm],
                    f"raw runtime contract changed at {index}/{arm}/{role}",
                )
                audit_mono_plans(payload["query_leg"], f"{index}/{arm}/{role}")
                audit_raw_outcome(row, payload, f"{index}/{arm}/{role}")
                current_replay_hash = replay_fingerprint(payload)
                if replay_hash is None:
                    replay_hash = current_replay_hash
                require(current_replay_hash == replay_hash,
                        f"shared RGB replay changed at {index}/{arm}/{role}")

            native, cec = raw["mono_native"][role], raw["mono_cec"][role]
            require(
                native["seed"] == cec["seed"]
                and native["shared_A_frames"] == cec["shared_A_frames"]
                and native["shared_A_decision_frames"]
                == cec["shared_A_decision_frames"]
                and int(native["shared_A_frames"]) == int(episode["online_a_steps"])
                and abs(float(native["geodesic_m"]) - float(cec["geodesic_m"]))
                <= 1e-12
                and abs(
                    float(native["geodesic_m"])
                    - float(queries[role]["geodesic_from_a_end_m"])
                ) <= 0.05,
                f"paired history/geometry changed at {index}/{role}",
            )
            accepted, _ = audit_raw_outcome(
                cec, payloads["mono_cec"], f"{index}/mono_cec/{role}/paired"
            )
            exact = False
            if accepted == 0:
                exact = exact_reject_fallback(
                    payloads["mono_cec"], payloads["mono_native"],
                    f"{index}/{role}",
                )
            require(
                bool(completion["fully_rejected_exact_native"][role]) == exact,
                f"completion fallback audit changed at {index}/{role}",
            )
            record = {
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
                "certificate_accept_plans": accepted,
                "fully_rejected_exact_native": exact,
            }
            require(record == summary["records"][len(records)],
                    f"summary record changed at {index}/{role}")
            records.append(record)

    require(raw_rows == 192 and len(records) == 96,
            "raw arm-role population is incomplete")
    require(
        len(completion_artifacts) == 48 and len(raw_artifacts) == 288,
        "raw artifact population is incomplete",
    )
    for bin_index, name in enumerate(
        ("0_to_20_m", "20_to_30_m", "30_to_50_m")
    ):
        bin_rows = [row for row in records if row["bin_name"] == name]
        require(len(bin_rows) == 32, f"{name}: raw query count changed")
        for role_index, role in enumerate(("all", *ROLES)):
            rows = bin_rows if role == "all" else [
                row for row in bin_rows if row["role"] == role
            ]
            reported = summary["bins"][name][role]
            gains = sum(row["cec"] == 1 and row["native"] == 0 for row in rows)
            losses = sum(row["cec"] == 0 and row["native"] == 1 for row in rows)
            rejected = [row for row in rows if row["certificate_accept_plans"] == 0]
            require(
                reported["queries"] == len(rows)
                and reported["scene_clusters"]
                == len({row["scene"] for row in rows})
                and reported["cec_vs_native_gains"] == gains
                and reported["cec_vs_native_losses"] == losses
                and reported["certificate_accept_queries"]
                == sum(row["certificate_accept_plans"] > 0 for row in rows)
                and reported["certificate_reject_queries"] == len(rejected)
                and reported["exact_fallback_queries_among_rejects"]
                == sum(row["fully_rejected_exact_native"] for row in rejected),
                f"{name}/{role}: count does not reproduce",
            )
            close(reported["mono_native_SR"],
                  sum(row["native"] for row in rows) / len(rows),
                  f"{name}/{role}/native SR")
            close(reported["mono_cec_SR"],
                  sum(row["cec"] for row in rows) / len(rows),
                  f"{name}/{role}/CEC SR")
            close(reported["risk_difference_pp"],
                  100.0 * sum(row["cec"] - row["native"] for row in rows)
                  / len(rows), f"{name}/{role}/risk difference")
            expected_ci = cluster_interval(
                rows, seed=20260830 + 10 * bin_index + role_index
            )
            for edge, expected in zip(
                reported["scene_cluster_bootstrap_95_pp"], expected_ci
            ):
                close(edge, expected, f"{name}/{role}/cluster interval")
            close(reported["mono_native_SPL"], sum(
                spl(row["native"], row["native_geodesic_m"], row["native_path_m"])
                for row in rows
            ) / len(rows), f"{name}/{role}/native SPL")
            close(reported["mono_cec_SPL"], sum(
                spl(row["cec"], row["cec_geodesic_m"], row["cec_path_m"])
                for row in rows
            ) / len(rows), f"{name}/{role}/CEC SPL")
            close(reported["mcnemar_exact_p"], mcnemar(gains, losses),
                  f"{name}/{role}/McNemar")

    result = {
        "schema_version": variant["verifier_schema"],
        "verified": True,
        "summary_sha256": summary_sha,
        "population_verification_sha256": verification_sha,
        "population_relative_root": args.population_relative_root,
        "benchmark_manifest_sha256": manifest_sha,
        "history_source": "controlled_causal_rgb_geodesic_survey",
        "histories": 48,
        "queries": 96,
        "raw_metric_rows": raw_rows,
        "completion_artifacts": len(completion_artifacts),
        "raw_artifacts": len(raw_artifacts),
        "completion_artifact_set_sha256": hashlib.sha256(
            "\n".join(
                f"{path}\t{digest}"
                for path, digest in sorted(completion_artifacts.items())
            ).encode()
        ).hexdigest(),
        "raw_artifact_set_sha256": hashlib.sha256(
            "\n".join(
                f"{path}\t{digest}"
                for path, digest in sorted(raw_artifacts.items())
            ).encode()
        ).hexdigest(),
        "partial_results_reported": False,
        "fallback_completion_used": False,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )


if __name__ == "__main__":
    main()
