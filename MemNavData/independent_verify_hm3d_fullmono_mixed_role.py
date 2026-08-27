#!/usr/bin/env python3
"""Independent raw-distance/depth audit for full-mono HM3D mixed roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("mono_native", "mono_raw_fixed", "mono_cec")
CONTRASTS = (
    ("mono_cec", "mono_native"),
    ("mono_cec", "mono_raw_fixed"),
)
SCHEMA = "hm3d_fullmono_mixed_role_independent_verification_v1_20260820"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def contrast(rows: list[dict[str, Any]], treatment: str,
             reference: str, role: str) -> dict[str, Any]:
    selected = [row for row in rows if role == "all" or row["role"] == role]
    units = {}
    for row in selected:
        key = (row["scene"], row["episode"], row["role"])
        units.setdefault(key, {})[row["arm"]] = row["reached"]
    require(all(treatment in value and reference in value
                for value in units.values()), "independent pairing incomplete")
    gains = sum(v[treatment] == 1 and v[reference] == 0 for v in units.values())
    losses = sum(v[treatment] == 0 and v[reference] == 1 for v in units.values())
    return {
        "n": len(units),
        "gains": gains,
        "losses": losses,
        "treatment_successes": sum(v[treatment] for v in units.values()),
        "reference_successes": sum(v[reference] for v in units.values()),
        "risk_difference_pp": 100.0 * sum(
            v[treatment] - v[reference] for v in units.values()
        ) / len(units),
        "exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "sealed benchmark changed before independent verification")
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(args.summary.read_text())
    require(summary["benchmark_manifest_sha256"] ==
            args.expected_manifest_sha256, "summary manifest mismatch")

    # Independently audit every frozen actual mono Goal-A trace, including failures.
    goal_a_plans = goal_a_sources = goal_a_successes = 0
    for scene_root in sorted((args.run_root / "goal_a/scenes").iterdir()):
        completion = json.loads((scene_root / "completion.json").read_text())
        goal_a_sources += int(completion["source_episode_count"])
        goal_a_successes += int(completion["goal_a_successes"])
        for record in completion["records"]:
            trace = json.loads(Path(record["trace_path"]).read_text())
            require(trace["source_hybrid_route"] == "native_sidecar",
                    "independent Goal-A route mismatch")
            for plan in trace["plans"]:
                require(plan.get("navdp_depth_source") == "monocular_sidecar",
                        "independent Goal-A depth source mismatch")
                require(plan.get("metric_depth_sensor_consumed") is False,
                        "independent Goal-A metric-depth read")
                receipt = plan.get("monocular_depth_receipt")
                require(isinstance(receipt, dict),
                        "independent Goal-A mono receipt missing")
                require(receipt.get("metric_depth_sensor_consumed") is False,
                        "independent Goal-A receipt reports metric read")
                goal_a_plans += 1
    require(goal_a_sources == int(summary["source_goal_a_episodes"]),
            "independent Goal-A source count changed")
    require(goal_a_successes == int(summary["goal_a_successes"]),
            "independent Goal-A success recount mismatch")

    rows = []
    query_plans = 0
    rejected_exact = 0
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = str(item["scene"]), str(item["episode"])
        root = (args.run_root / "evaluation/natural_direction" /
                f"{index:03d}_{scene}_{episode}")
        completion = json.loads((root / "completion.json").read_text())
        payloads = {}
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as handle:
                arm_rows = list(csv.DictReader(handle))
            require(len(arm_rows) == 2, f"{index}/{arm}: wrong query count")
            for row in arm_rows:
                role = row["analysis_role"]
                distance = float(row["final_goal_dist_m"])
                reached = int(row["reached"])
                require(reached == int(distance < 1.0),
                        f"{index}/{arm}/{role}: distance mismatch")
                payload = json.loads((
                    root / arm / f"{episode}_{row['query_id']}_plans.json"
                ).read_text())
                require(payload.get("analysis_role_not_forwarded") is True,
                        f"{index}/{arm}/{role}: role leak")
                payloads[(arm, role)] = payload
                for plan in payload["query_leg"]:
                    require(plan.get("navdp_depth_source") ==
                            "monocular_sidecar",
                            f"{index}/{arm}/{role}: depth source mismatch")
                    require(plan.get("metric_depth_sensor_consumed") is False,
                            f"{index}/{arm}/{role}: metric-depth read")
                    receipt = plan.get("monocular_depth_receipt")
                    require(isinstance(receipt, dict) and
                            int(receipt.get("frame_index", -1)) >= 40 and
                            receipt.get("scale_active") is True,
                            f"{index}/{arm}/{role}: active mono receipt invalid")
                    query_plans += 1
                rows.append({
                    "scene": scene, "episode": episode, "role": role,
                    "arm": arm, "reached": reached, "distance": distance,
                })
        for role in ("novel", "revisit"):
            cec = payloads[("mono_cec", role)]
            native = payloads[("mono_native", role)]
            if any(plan.get("certified_relocalization_accepted") is True
                   for plan in cec["query_leg"]):
                continue
            require(cec["rollout_traces"]["query"] ==
                    native["rollout_traces"]["query"],
                    f"{index}/{role}: rejected trace differs from native")
            rejected_exact += 1

    require(len(rows) == len(manifest["episodes"]) * 3 * 2,
            "independent query row count changed")
    recount = {}
    for role in ("novel", "revisit", "all"):
        recount[role] = {}
        for treatment, reference in CONTRASTS:
            key = f"{treatment}_minus_{reference}"
            value = contrast(rows, treatment, reference, role)
            recount[role][key] = value
            reported = summary["results"][role]["contrasts"][key]
            for field in (
                "n", "gains", "losses", "risk_difference_pp",
                "exact_mcnemar_two_sided_p",
            ):
                require(math.isclose(float(value[field]), float(reported[field]),
                                     rel_tol=0.0, abs_tol=1e-12),
                        f"summary/recount mismatch {role}/{key}/{field}")

    report = {
        "schema_version": SCHEMA,
        "verified": True,
        "authorized": True,
        "fresh_scene_generalization": bool(
            summary["fresh_scene_generalization"]),
        "scope": summary["scope"],
        "goal_a_sources": goal_a_sources,
        "goal_a_successes": goal_a_successes,
        "goal_a_mono_plan_receipts_verified": goal_a_plans,
        "raw_final_distance_records": len(rows),
        "query_mono_plan_receipts_verified": query_plans,
        "fully_rejected_exact_native_queries": rejected_exact,
        "success_recomputed_from_raw_final_distance": True,
        "success_distance_m": 1.0,
        "recount": recount,
    }
    require(not args.out.exists(), "verification output exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
