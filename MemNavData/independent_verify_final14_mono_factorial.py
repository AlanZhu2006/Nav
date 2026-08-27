#!/usr/bin/env python3
"""Independent raw-distance and plan verifier for Final14 mono factorial.

This module intentionally imports neither the primary summarizer nor its
statistics helpers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = (
    "mono_native", "mono_raw_fixed", "mono_cec",
    "metric_native", "metric_cec",
)
MONO_ARMS = {"mono_native", "mono_raw_fixed", "mono_cec"}
CEC_NATIVE = {"mono_cec": "mono_native", "metric_cec": "metric_native"}
SCHEMA = "final14_mono_factorial_independent_verification_v1_20260819"


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
    units: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in selected:
        key = (row["scene"], row["episode"], row["role"])
        units.setdefault(key, {})[row["arm"]] = row["reached"]
    require(all(treatment in values and reference in values
                for values in units.values()), "independent pairing incomplete")
    gains = sum(values[treatment] == 1 and values[reference] == 0
                for values in units.values())
    losses = sum(values[treatment] == 0 and values[reference] == 1
                 for values in units.values())
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


def assert_close(left: float, right: float, message: str) -> None:
    require(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12),
            message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest_path = Path(args.bench_root) / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "manifest changed before independent verification")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21, "history population changed")
    summary = json.loads(Path(args.summary).read_text())
    require(summary.get("fresh_confirmation") is False,
            "primary summary mislabeled consumed attribution")

    rows: list[dict[str, Any]] = []
    payloads: dict[tuple[int, str, str], dict[str, Any]] = {}
    mono_plan_count = 0
    metric_plan_count = 0
    rejected_exact_count = {"mono_cec": 0, "metric_cec": 0}
    for index, item in enumerate(manifest["episodes"]):
        scene = str(item["scene"])
        episode = str(item["episode"])
        root = (run_root / "evaluation" / "natural_direction" /
                f"{index:03d}_{scene}_{episode}")
        completion = root / "completion.json"
        require(completion.is_file(), f"missing completion for history {index}")
        receipt = (root / "completion.json.sha256").read_text().split()[0]
        require(sha256(completion) == receipt,
                f"history {index}: completion hash changed")

        reference_replay = None
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as handle:
                arm_rows = list(csv.DictReader(handle))
            require(len(arm_rows) == 2, f"history {index}/{arm}: wrong rows")
            for row in arm_rows:
                role = row["analysis_role"]
                distance = float(row["final_goal_dist_m"])
                reached = int(row["reached"])
                require(reached == int(distance < 1.0),
                        f"history {index}/{arm}/{role}: distance mismatch")
                payload_path = root / arm / f"{episode}_{row['query_id']}_plans.json"
                payload = json.loads(payload_path.read_text())
                require(payload.get("analysis_role_not_forwarded") is True,
                        f"history {index}/{arm}/{role}: role leak")
                key = (index, arm, role)
                payloads[key] = payload
                replay_signature = {
                    field: payload["replay"][field]
                    for field in (
                        "all_rgb_hashes_verified", "decision_frames",
                        "decision_steps", "diffusion_samples_during_replay",
                        "navdp_memory_size", "navdp_queue_lengths",
                        "online_frames",
                    )
                }
                if reference_replay is None:
                    reference_replay = replay_signature
                require(replay_signature == reference_replay,
                        f"history {index}/{arm}/{role}: replay changed")
                plans = payload["query_leg"]
                require(bool(plans), f"history {index}/{arm}/{role}: no plans")
                for plan in plans:
                    if arm in MONO_ARMS:
                        require(plan.get("navdp_depth_source") ==
                                "monocular_sidecar",
                                f"history {index}/{arm}/{role}: mono source changed")
                        require(plan.get("metric_depth_sensor_consumed") is False,
                                f"history {index}/{arm}/{role}: metric read")
                        receipt_payload = plan.get("monocular_depth_receipt")
                        require(isinstance(receipt_payload, dict),
                                f"history {index}/{arm}/{role}: mono receipt missing")
                        require(int(receipt_payload.get("frame_index", -1)) >= 40,
                                f"history {index}/{arm}/{role}: bootstrap used")
                        require(receipt_payload.get("scale_active") is True,
                                f"history {index}/{arm}/{role}: scale inactive")
                        require(receipt_payload.get("metric_depth_sensor_consumed")
                                is False,
                                f"history {index}/{arm}/{role}: receipt metric read")
                        mono_plan_count += 1
                    else:
                        require(plan.get("navdp_depth_source") == "metric_request",
                                f"history {index}/{arm}/{role}: metric source changed")
                        require(plan.get("metric_depth_sensor_consumed") is True,
                                f"history {index}/{arm}/{role}: metric read missing")
                        require(plan.get("monocular_depth_receipt") is None,
                                f"history {index}/{arm}/{role}: unexpected mono receipt")
                        metric_plan_count += 1
                rows.append({
                    "scene": scene,
                    "episode": episode,
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "distance": distance,
                })

        for cec_arm, native_arm in CEC_NATIVE.items():
            for role in ("novel", "revisit"):
                cec = payloads[(index, cec_arm, role)]
                native = payloads[(index, native_arm, role)]
                cec_plans = cec["query_leg"]
                if any(plan.get("certified_relocalization_accepted") is True
                       for plan in cec_plans):
                    continue
                native_plans = native["query_leg"]
                require(len(cec_plans) == len(native_plans),
                        f"history {index}/{cec_arm}/{role}: reject length changed")
                for cec_plan, native_plan in zip(
                        cec_plans, native_plans):
                    for field in (
                        "requested_diffusion_seed", "diffusion_seed",
                        "selected_trajectory_sha256",
                    ):
                        require(cec_plan.get(field) == native_plan.get(field),
                                f"history {index}/{cec_arm}/{role}: {field} changed")
                require(cec["rollout_traces"]["query"] ==
                        native["rollout_traces"]["query"],
                        f"history {index}/{cec_arm}/{role}: reject trace changed")
                rejected_exact_count[cec_arm] += 1

    require(len(rows) == 21 * 5 * 2, "raw result row count changed")
    require(mono_plan_count > 0 and metric_plan_count > 0,
            "depth arms were not independently exercised")

    recount: dict[str, Any] = {}
    for role in ("novel", "revisit", "all"):
        recount[role] = {}
        for treatment, reference in (
            ("mono_cec", "mono_native"),
            ("mono_cec", "mono_raw_fixed"),
            ("metric_cec", "metric_native"),
            ("mono_native", "metric_native"),
            ("mono_cec", "metric_cec"),
        ):
            key = f"{treatment}_minus_{reference}"
            value = contrast(rows, treatment, reference, role)
            recount[role][key] = value
            reported = summary["results"][role]["contrasts"][key]
            for field in (
                "n", "gains", "losses", "risk_difference_pp",
                "exact_mcnemar_two_sided_p",
            ):
                assert_close(value[field], reported[field],
                             f"reported/recount {role}/{key}/{field} mismatch")

    report = {
        "schema_version": SCHEMA,
        "verified": True,
        "authorized": True,
        "fresh_confirmation": False,
        "scope": "consumed_final14_query_controller_depth_attribution",
        "raw_final_distance_records": len(rows),
        "success_recomputed_from_raw_final_distance": True,
        "success_distance_m": 1.0,
        "mono_plan_receipts_verified": mono_plan_count,
        "metric_plan_receipts_verified": metric_plan_count,
        "fully_rejected_exact_native_queries": rejected_exact_count,
        "recount": recount,
        "known_gap": (
            "Original Goal-A history was generated by metric NavDP; this "
            "verifies mono query-controller behavior after causal RGB replay, "
            "not a full-mono Goal-A population."
        ),
    }
    out = Path(args.out)
    require(not out.exists(), "verification output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
