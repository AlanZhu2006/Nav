#!/usr/bin/env python3
"""Recompute the HM3D fixed-vs-full-metric gate from raw arm artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("fixed_2p5m", "full_metric")
SUMMARY_SCHEMA = "hm3d_table1_metric_distance_revisit_summary_v1_20260901"
EPISODE_SCHEMA = "hm3d_table1_metric_distance_revisit_pair_v1_20260901"
VERIFICATION_SCHEMA = (
    "hm3d_table1_metric_distance_revisit_verification_v1_20260901"
)
FIXED_RADIUS_M = 2.5
NATIVE_SUPPORT_RADIUS_M = 10.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-8)


def exact_mcnemar(gain: int, loss: int) -> float:
    discordant = gain + loss
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(gain, loss) + 1)) / 2 ** discordant
    return min(1.0, 2.0 * tail)


def verify(
    run_root: Path,
    benchmark_root: Path,
    expected_manifest_sha256: str,
    summary_path: Path,
) -> dict[str, Any]:
    manifest_path = benchmark_root / "manifest.json"
    require(digest(manifest_path) == expected_manifest_sha256,
            "benchmark manifest changed")
    histories = json.loads(manifest_path.read_text()).get("episodes")
    require(isinstance(histories, list) and len(histories) == 28,
            "expected the frozen 28-history population")
    require(len({str(item["scene"]) for item in histories}) == 21,
            "expected 21 frozen scene clusters")

    summary = json.loads(summary_path.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "summary schema changed")
    require(summary.get("benchmark_manifest_sha256")
            == expected_manifest_sha256,
            "summary manifest binding changed")

    outcomes: dict[str, list[int]] = {arm: [] for arm in ARMS}
    cap_hits = 0
    metric_plans = 0
    first_takeovers = 0
    first_different = 0
    raw_metric_rows = 0
    raw_plan_files = 0
    for index, item in enumerate(histories):
        scene = str(item["scene"])
        episode = str(item["episode"])
        root = (run_root / "evaluation" / "revisit_metric_distance"
                / f"{index:03d}_{scene}_{episode}")
        completion_path = root / "completion.json"
        require((root / "completion.json.sha256").read_text().split()[0]
                == digest(completion_path),
                f"completion receipt changed at history {index}")
        completion = json.loads(completion_path.read_text())
        require(completion.get("schema_version") == EPISODE_SCHEMA
                and completion.get("history_index") == index
                and str(completion.get("scene")) == scene
                and str(completion.get("episode")) == episode,
                f"history identity changed at {index}")
        require(completion.get("arms") == list(ARMS)
                and completion.get("same_server_pair") is True
                and completion.get("prefix_equality") is True
                and completion.get("runtime_role_visibility") == "none"
                and completion.get("analysis_query_role") == "revisit",
                f"paired contract changed at {index}")

        first = completion.get("first_takeover")
        if first is not None:
            first_takeovers += 1
            first_different += int(bool(first["differs_from_fixed_2p5m"]))

        observed_cap_hits = 0
        observed_metric_plans = 0
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 1 and rows[0].get("analysis_role") == "revisit",
                    f"raw role denominator changed at {index}/{arm}")
            row = rows[0]
            require(int(row.get("runtime_failure_plans", "-1")) == 0,
                    f"runtime failure at {index}/{arm}")
            reached = int(row["reached"])
            require(reached == int(float(row["final_goal_dist_m"]) < 1.0),
                    f"raw success mismatch at {index}/{arm}")
            require(reached == int(completion["outcomes"][arm]),
                    f"completion success mismatch at {index}/{arm}")
            outcomes[arm].append(reached)
            raw_metric_rows += 1

            plan_paths = list((root / arm).glob("*_plans.json"))
            require(len(plan_paths) == 1,
                    f"raw plan denominator changed at {index}/{arm}")
            payload = json.loads(plan_paths[0].read_text())
            require(payload.get("analysis_role_not_forwarded") is True,
                    f"role leaked at {index}/{arm}")
            plans = payload.get("query_leg")
            require(isinstance(plans, list) and plans,
                    f"empty raw plans at {index}/{arm}")
            raw_plan_files += 1
            for plan in plans:
                if plan.get("revisit_adapter_takeover") is not True:
                    continue
                controller = float(
                    plan["memory_controller_pointgoal_distance_m"])
                if arm == "fixed_2p5m":
                    require(close(controller, FIXED_RADIUS_M),
                            f"fixed arm radius changed at {index}")
                    continue
                scale = float(plan["memory_metric_scale_m_per_raw"])
                raw_norm = float(plan["memory_unbounded_pointgoal_norm"])
                unbounded = float(
                    plan["memory_unbounded_pointgoal_distance_m"])
                require(scale > 0.0 and close(unbounded, scale * raw_norm),
                        f"metric payload is not scale times norm at {index}")
                require(close(controller, min(unbounded,
                                              NATIVE_SUPPORT_RADIUS_M)),
                        f"metric support contract changed at {index}")
                observed_cap_hits += int(
                    unbounded > NATIVE_SUPPORT_RADIUS_M
                    and close(controller, NATIVE_SUPPORT_RADIUS_M)
                )
                observed_metric_plans += 1
        require(observed_cap_hits
                == int(completion["metric_native_support_cap_hits"]),
                f"cap-hit receipt mismatch at history {index}")
        require(observed_metric_plans
                == len(completion["metric_controller_distances_m"]),
                f"metric-plan receipt mismatch at history {index}")
        cap_hits += observed_cap_hits
        metric_plans += observed_metric_plans

    fixed_success = sum(outcomes["fixed_2p5m"])
    metric_success = sum(outcomes["full_metric"])
    gains = sum(metric == 1 and fixed == 0 for fixed, metric in zip(
        outcomes["fixed_2p5m"], outcomes["full_metric"]))
    losses = sum(metric == 0 and fixed == 1 for fixed, metric in zip(
        outcomes["fixed_2p5m"], outcomes["full_metric"]))
    require(summary["success"] == {
        "fixed_2p5m": fixed_success, "full_metric": metric_success,
    }, "summary success does not reproduce from raw metrics")
    require(summary["paired"]["gain"] == gains
            and summary["paired"]["loss"] == losses
            and summary["paired"]["net"] == gains - losses
            and close(summary["paired"]["mcnemar_exact_p"],
                      exact_mcnemar(gains, losses)),
            "summary paired result does not reproduce")
    audit = summary["treatment_audit"]
    require(int(audit["histories_with_first_takeover"]) == first_takeovers
            and int(audit["first_takeover_differs_from_fixed"])
            == first_different
            and int(audit["metric_takeover_plans"]) == metric_plans
            and int(audit["native_10m_support_cap_hits"]) == cap_hits,
            "summary treatment audit does not reproduce")

    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
        "benchmark_manifest_sha256": expected_manifest_sha256,
        "summary_sha256": digest(summary_path),
        "histories": len(histories),
        "scene_clusters": len({str(item["scene"]) for item in histories}),
        "raw_metric_rows": raw_metric_rows,
        "raw_plan_files": raw_plan_files,
        "success": {
            "fixed_2p5m": fixed_success,
            "full_metric": metric_success,
        },
        "paired_gain": gains,
        "paired_loss": losses,
        "metric_takeover_plans": metric_plans,
        "native_10m_support_cap_hits": cap_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(
        args.run_root, args.benchmark_root,
        args.expected_manifest_sha256, args.summary,
    )
    require(not args.out.exists(), "verification output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
