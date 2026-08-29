#!/usr/bin/env python3
"""Independently verify raw HM3D Table-1 NavDP native/CEC outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA = "hm3d_table1_navdp_pair_verification_v1_20260829"
SUMMARY_SCHEMAS = {
    "HM3D": "hm3d_table1_navdp_pair_summary_v1_20260829",
    "MP3D": "mp3d_table1_navdp_pair_summary_v1_20260829",
    "HM3D_TABLE2": "hm3d_table2_leg3_navdp_pair_summary_v1_20260829",
}
VERIFICATION_SCHEMAS = {
    "HM3D": SCHEMA,
    "MP3D": "mp3d_table1_navdp_pair_verification_v1_20260829",
    "HM3D_TABLE2": (
        "hm3d_table2_leg3_navdp_pair_verification_v1_20260829"
    ),
}
ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(int(gains), int(losses)) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    units: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = str(row["scene"]), str(row["episode"]), str(row["role"])
        units.setdefault(key, {})[str(row["arm"])] = row
    require(all(set(value) == set(ARMS) for value in units.values()),
            "independent pairing is incomplete")
    gains = sum(
        value["mono_cec"]["reached"] == 1
        and value["mono_native"]["reached"] == 0
        for value in units.values()
    )
    losses = sum(
        value["mono_cec"]["reached"] == 0
        and value["mono_native"]["reached"] == 1
        for value in units.values()
    )
    native = [value["mono_native"] for value in units.values()]
    cec = [value["mono_cec"] for value in units.values()]

    def spl(row: dict[str, Any]) -> float:
        if int(row["reached"]) == 0:
            return 0.0
        shortest = float(row["geodesic_m"])
        return shortest / max(shortest, float(row["path_len_m"]), 1e-12)

    return {
        "n": len(units),
        "native_success": sum(int(row["reached"]) for row in native),
        "cec_success": sum(int(row["reached"]) for row in cec),
        "native_sr": mean(int(row["reached"]) for row in native),
        "cec_sr": mean(int(row["reached"]) for row in cec),
        "risk_difference_pp": 100.0 * mean(
            int(value["mono_cec"]["reached"])
            - int(value["mono_native"]["reached"])
            for value in units.values()
        ),
        "paired_gain": gains,
        "paired_loss": losses,
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
        "native_spl": mean(spl(row) for row in native),
        "cec_spl": mean(spl(row) for row in cec),
        "native_mean_final_distance_m": mean(
            float(row["final_goal_dist_m"]) for row in native),
        "cec_mean_final_distance_m": mean(
            float(row["final_goal_dist_m"]) for row in cec),
        "native_mean_path_len_m": mean(
            float(row["path_len_m"]) for row in native),
        "cec_mean_path_len_m": mean(
            float(row["path_len_m"]) for row in cec),
        "native_mean_steps": mean(int(row["steps"]) for row in native),
        "cec_mean_steps": mean(int(row["steps"]) for row in cec),
        "cec_takeover_queries": sum(
            int(row["certificate_accept_plans"]) > 0 for row in cec),
        "cec_takeover_plans": sum(
            int(row["certificate_accept_plans"]) for row in cec),
    }


def _compare_reported(observed: dict[str, Any], reported: dict[str, Any],
                      scope: str) -> None:
    for field, value in observed.items():
        require(field in reported, f"summary field missing: {scope}/{field}")
        expected = reported[field]
        if isinstance(value, float):
            require(math.isclose(value, float(expected), rel_tol=0.0,
                                 abs_tol=1e-12),
                    f"summary mismatch: {scope}/{field}")
        else:
            require(value == expected, f"summary mismatch: {scope}/{field}")


def verify(
    run_root: Path,
    benchmark_root: Path,
    construction_verification: Path,
    summary_path: Path,
    dataset: str = "HM3D",
) -> dict[str, Any]:
    require(dataset in SUMMARY_SCHEMAS, "unsupported Table-1 dataset")
    construction = json.loads(construction_verification.read_text())
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMAS[dataset]
            and summary.get("verified") is True,
            "summary is not a verified Table-1 NavDP summary")
    require(summary.get("dataset") == dataset,
            "summary dataset identity changed")
    require(construction.get("verified") is True
            and construction.get("formal_policy_evaluation_authorized") is True,
            "construction did not authorize formal evaluation")
    manifest_path = benchmark_root / "manifest.json"
    manifest_sha = digest(manifest_path)
    require(manifest_sha == construction["benchmark_manifest_sha256"]
            == summary["benchmark_manifest_sha256"],
            "manifest binding differs across construction and summary")
    require(digest(construction_verification)
            == summary["construction_verification_sha256"],
            "construction-verification receipt changed")
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    require(len(episodes) == int(summary["histories"])
            == int(construction["histories"]),
            "history denominator changed")
    table2 = dataset == "HM3D_TABLE2"
    if table2:
        require(
            construction.get("schema_version")
            == (
                "hm3d_table2_leg3_mixed_role_construction_"
                "verification_v1_20260829"
            ),
            "Table-2 construction verifier schema changed",
        )
        population_path = benchmark_root.parent / "population_receipt.json"
        require(
            digest(population_path)
            == construction.get("population_receipt_sha256"),
            "Table-2 population receipt changed",
        )
        population = json.loads(population_path.read_text())
        require(
            summary.get("conditional_on_factual_AB_success") is True
            and summary.get("unconditional_three_leg_joint_sr_reported")
            is False,
            "Table-2 estimand boundary changed",
        )
    else:
        population = None

    rows: list[dict[str, Any]] = []
    plan_receipts = 0
    exact_fallback_queries = 0
    novel_takeovers = 0
    revisit_takeovers = 0
    for index, item in enumerate(episodes):
        scene, episode = str(item["scene"]), str(item["episode"])
        queries = [query for pair in item["pairs"] for query in pair["queries"]]
        by_role = {str(query["analysis_role"]): query for query in queries}
        require(set(by_role) == set(ROLES) and len(queries) == 2,
                "manifest role pair changed")
        root = (run_root / "evaluation" / "natural_direction"
                / f"{index:03d}_{scene}_{episode}")
        completion_path = root / "completion.json"
        require((root / "completion.json.sha256").read_text().split()[0]
                == digest(completion_path),
                f"completion receipt changed at history {index}")
        completion = json.loads(completion_path.read_text())
        require(completion.get("arms") == list(ARMS)
                and completion.get("prefix_equality") is True,
                f"completion contract changed at history {index}")
        if table2:
            require(
                completion.get("schema_version")
                == "hm3d_table2_leg3_history_v1_20260829"
                and completion.get("history_contract") == "actual_ab"
                and completion.get("shared_history_policy")
                == "actual_mono_navdp_novel_A_then_novel_B_rgb_replay",
                f"Table-2 A/B replay contract changed at history {index}",
            )
            prefix_root = Path(item["online_a_episode"])
            receipt = json.loads((prefix_root / "receipt.json").read_text())
            trace = json.loads(
                (prefix_root / "online_a_trace.json").read_text()
            )
            prefix_a = int(receipt.get("prefix_A_steps", -1))
            prefix_b = int(receipt.get("prefix_B_steps", -1))
            require(
                receipt.get("prefix_receipt_schema")
                == "hm3d_table2_actual_mono_ab_prefix_v1_20260829"
                and receipt.get("prefix_semantics")
                == "actual_mono_Novel_A_then_Novel_B"
                and trace.get("prefix_semantics")
                == "exact_actual_mono_A_then_B_observation_concat"
                and prefix_a > 0 and prefix_b > 0
                and prefix_a + prefix_b == len(trace["poses"])
                and prefix_a == int(completion.get("prefix_A_steps", -2))
                and prefix_b == int(completion.get("prefix_B_steps", -2)),
                f"Table-2 raw A/B prefix failed audit at history {index}",
            )
        payloads: dict[tuple[str, str], dict[str, Any]] = {}
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as handle:
                arm_rows = list(csv.DictReader(handle))
            require(len(arm_rows) == 2
                    and {row["analysis_role"] for row in arm_rows}
                    == set(ROLES),
                    f"raw role denominator changed for history {index}/{arm}")
            for row in arm_rows:
                role = str(row["analysis_role"])
                final_distance = float(row["final_goal_dist_m"])
                reached = int(row["reached"])
                require(reached == int(final_distance < 1.0),
                        f"raw success mismatch at history {index}/{arm}/{role}")
                query_id = str(by_role[role]["query_id"])
                payload_path = root / arm / f"{episode}_{query_id}_plans.json"
                payload = json.loads(payload_path.read_text())
                require(payload.get("analysis_role_not_forwarded") is True,
                        f"role leaked at history {index}/{arm}/{role}")
                plans = payload.get("query_leg")
                require(isinstance(plans, list) and plans,
                        f"empty raw plan trace at history {index}/{arm}/{role}")
                scale_hashes = set()
                for plan in plans:
                    require(plan.get("navdp_depth_source")
                            == "monocular_sidecar",
                            f"depth source changed at {index}/{arm}/{role}")
                    require(plan.get("metric_depth_sensor_consumed") is False,
                            f"metric depth read at {index}/{arm}/{role}")
                    receipt = plan.get("monocular_depth_receipt")
                    require(isinstance(receipt, dict)
                            and int(receipt.get("frame_index", -1)) >= 40
                            and receipt.get("scale_active") is True,
                            f"invalid mono receipt at {index}/{arm}/{role}")
                    scale_hashes.add(str(receipt.get("scale_receipt_sha256")))
                    plan_receipts += 1
                require(len(scale_hashes) == 1 and "None" not in scale_hashes,
                        f"mono scale drift at {index}/{arm}/{role}")
                accept_plans = sum(
                    plan.get("certified_relocalization_accepted") is True
                    or plan.get("cec_takeover") is True for plan in plans
                )
                runtime_failures = sum(
                    plan.get("certified_relocalization_reason")
                    == "certificate_endpoint_failure"
                    or plan.get("learned_pi3x_relocalization_ok") is False
                    or plan.get("cec_reason")
                    == "certificate_endpoint_failure"
                    for plan in plans
                )
                require(runtime_failures
                        == int(row["runtime_failure_plans"]),
                        f"runtime-failure recount changed at "
                        f"{index}/{arm}/{role}")
                require(runtime_failures == 0,
                        f"certificate runtime failure at "
                        f"{index}/{arm}/{role}")
                require(accept_plans == int(row["certificate_accept_plans"]),
                        f"takeover recount changed at {index}/{arm}/{role}")
                payloads[(arm, role)] = payload
                rows.append({
                    "scene": scene,
                    "episode": episode,
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "geodesic_m": float(row["geodesic_m"]),
                    "path_len_m": float(row["path_len_m"]),
                    "steps": int(row["steps"]),
                    "final_goal_dist_m": final_distance,
                    "certificate_accept_plans": accept_plans,
                })
        for role in ROLES:
            cec = payloads[("mono_cec", role)]
            native = payloads[("mono_native", role)]
            accepted = any(
                plan.get("certified_relocalization_accepted") is True
                or plan.get("cec_takeover") is True
                for plan in cec["query_leg"]
            )
            if accepted:
                if role == "novel":
                    novel_takeovers += 1
                else:
                    revisit_takeovers += 1
                continue
            require(cec["rollout_traces"]["query"]
                    == native["rollout_traces"]["query"]
                    and cec["query_result"] == native["query_result"],
                    f"all-reject query was not exact fallback: {index}/{role}")
            exact_fallback_queries += 1

    require(len(rows) == len(episodes) * len(ARMS) * len(ROLES),
            "raw paired denominator changed")
    recomputed = {}
    for role in (*ROLES, "all"):
        selected = rows if role == "all" else [
            row for row in rows if row["role"] == role
        ]
        recomputed[role] = _summarize(selected)
        _compare_reported(
            recomputed[role], summary["results"][role], role,
        )
    require(novel_takeovers == int(summary["safety"]["novel_takeover_queries"])
            and revisit_takeovers
            == int(summary["safety"]["revisit_takeover_queries"]),
            "summary takeover counts do not reproduce")
    require(exact_fallback_queries == sum(
        int(value) for value in summary["safety"]
        ["fully_rejected_exact_native_by_role"].values()
    ), "summary exact-fallback count does not reproduce")
    result = {
        "schema_version": VERIFICATION_SCHEMAS[dataset],
        "verified": True,
        "authorized": True,
        "dataset": dataset,
        "claim_scope": summary["claim_scope"],
        "benchmark_manifest_sha256": manifest_sha,
        "construction_verification_sha256": digest(
            construction_verification),
        "summary_sha256": digest(summary_path),
        "histories": len(episodes),
        "scene_clusters": len({str(row["scene"]) for row in episodes}),
        "raw_queries_per_arm": 2 * len(episodes),
        "raw_metric_rows": len(rows),
        "raw_monocular_plan_receipts": plan_receipts,
        "novel_takeover_queries": novel_takeovers,
        "revisit_takeover_queries": revisit_takeovers,
        "fully_rejected_exact_native_queries": exact_fallback_queries,
        "success_recomputed_from_raw_final_distance": True,
        "success_distance_m": 1.0,
        "recomputed": recomputed,
    }
    if table2:
        assert population is not None
        observed_waterfall = {
            "source_A_successful_histories_entering_B": int(
                population["source_A_attempts"]
            ),
            "factual_AB_successful_prefixes": int(
                population["factual_AB_successful_prefixes"]
            ),
            "factual_AB_scene_clusters": int(
                population["factual_AB_scene_clusters"]
            ),
            "leg3_constructible_histories": int(
                population["leg3_constructible_histories"]
            ),
            "leg3_scene_clusters": int(
                population["leg3_scene_clusters"]
            ),
        }
        require(
            observed_waterfall == summary.get("factual_prefix_waterfall"),
            "Table-2 factual prefix waterfall does not reproduce",
        )
        segment_counts = {"A": 0, "B": 0}
        for item in episodes:
            segment = str(item.get("table2_selected_revisit_segment", ""))
            require(segment in segment_counts,
                    "Table-2 Revisit source segment changed")
            segment_counts[segment] += 1
        require(
            {key: value for key, value in segment_counts.items() if value}
            == summary.get("revisit_source_segment_counts"),
            "Table-2 Revisit segment counts do not reproduce",
        )
        result.update({
            "estimand": "Leg3_C_given_factual_successful_A_and_B",
            "factual_prefix_waterfall": observed_waterfall,
            "revisit_source_segment_counts": {
                key: value for key, value in segment_counts.items() if value
            },
            "unconditional_three_leg_joint_sr_reported": False,
        })
    return result


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--construction-verification", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=tuple(SUMMARY_SCHEMAS), default="HM3D",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_root.resolve(), args.benchmark_root.resolve(),
        args.construction_verification.resolve(), args.summary.resolve(),
        dataset=args.dataset,
    )
    write_exclusive(args.out.resolve(), result)
    print(json.dumps({
        "verified": True,
        "histories": result["histories"],
        "queries": result["raw_queries_per_arm"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
