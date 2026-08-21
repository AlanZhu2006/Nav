#!/usr/bin/env python3
"""Independent raw-file recount for the MDTEC monocular x CEC composition.

The verifier deliberately does not import the summarizer's statistics code.
It also does not trust the compact outer ``depth_arms.csv`` success flag:
Goal-B success is independently thresholded from each arm's retained raw
``metric.csv/final_dist_B``.  This keeps already-running formal jobs
verifiable even when their compact CSV predates the final-distance column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = ("raw_native", "raw_cec")


def fail(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_mcnemar(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_metric_row(path, episode):
    fail(path.is_file(), f"missing raw metric file {path}")
    with path.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row.get("episode") == episode]
    fail(len(matches) == 1,
         f"expected one raw metric row for {episode} in {path}, got {len(matches)}")
    return matches[0]


def load(run_root, success_distance_m):
    rows = []
    for path in sorted((Path(run_root) / "scenes").glob("*_*")):
        csv_path = path / "depth_arms.csv"
        fail(csv_path.is_file(), f"missing depth_arms.csv under {path}")
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                fail(row["arm"] in ARMS, f"unknown arm {row['arm']}")
                plans_path = path / row["plans_file"]
                plans = json.loads(plans_path.read_text())
                fail(plans["arm"] == row["arm"], "plan identity mismatch")
                raw_metric = load_metric_row(
                    path / f"{row['episode']}_{row['arm']}" / "metric.csv",
                    row["episode"],
                )
                final_dist_b = float(raw_metric["final_dist_B"])
                fail(math.isfinite(final_dist_b),
                     f"non-finite Goal-B distance for {row['scene']}/"
                     f"{row['episode']}/{row['arm']}")
                reached_from_distance = int(final_dist_b < success_distance_m)
                fail(int(float(row["reached_B"])) == reached_from_distance,
                     f"Goal-B success flag disagrees with raw distance for "
                     f"{row['scene']}/{row['episode']}/{row['arm']}: "
                     f"flag={row['reached_B']} distance={final_dist_b}")
                fail(int(float(row["reached"])) == reached_from_distance,
                     "compact reached field disagrees with raw distance")
                fail(row["metric_depth_sensor_consumed_any"] == "False",
                     f"{row['scene']}/{row['episode']}/{row['arm']}: "
                     "consumed simulator metric depth")
                if row["arm"] == "raw_cec":
                    fail(int(row["certified_runtime_failure_count"]) == 0,
                         "raw_cec certificate runtime failure survived to "
                         "summary -- audit invalid")
                row["_raw_final_dist_B"] = final_dist_b
                row["_reached_from_distance"] = reached_from_distance
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    reported = json.loads(Path(args.report).read_text())
    success_distance_m = float(protocol["evaluation"]["success_distance_m"])
    rows = load(args.run_root, success_distance_m)

    expected_rows = 4 * int(protocol["evaluation"]["scenes"])
    fail(len(rows) == expected_rows,
         f"row count {len(rows)} != expected {expected_rows}")
    fail(len(rows) == reported["rows"], "row count disagrees with summary")

    by_unit = {}
    for row in rows:
        key = (row["scene"], row["episode"])
        by_unit.setdefault(key, {})[row["arm"]] = int(
            row["_reached_from_distance"])
    fail(all(set(arms) == set(ARMS) for arms in by_unit.values()),
         "incomplete arm coverage for at least one episode")

    gains = sum(1 for arms in by_unit.values()
                if arms["raw_cec"] == 1 and arms["raw_native"] == 0)
    losses = sum(1 for arms in by_unit.values()
                 if arms["raw_cec"] == 0 and arms["raw_native"] == 1)
    p = exact_mcnemar(gains, losses)

    itt = reported["intent_to_treat"]["contrast"]
    fail(itt["gains"] == gains and itt["losses"] == losses,
         f"gains/losses mismatch: recount {gains}/{losses} vs "
         f"reported {itt['gains']}/{itt['losses']}")
    fail(abs(itt["exact_mcnemar_two_sided_p"] - p) < 1e-9,
         "exact McNemar p-value mismatch")

    verification = {
        "verified": True,
        "authorized": True,
        "n": len(by_unit),
        "recount_gains": gains,
        "recount_losses": losses,
        "recount_exact_mcnemar_two_sided_p": p,
        "reported_gains": itt["gains"],
        "reported_losses": itt["losses"],
        "reported_exact_mcnemar_two_sided_p": itt["exact_mcnemar_two_sided_p"],
        "certified_runtime_failures_total": sum(
            int(r["certified_runtime_failure_count"])
            for r in rows if r["arm"] == "raw_cec"),
        "success_distance_m": success_distance_m,
        "success_recomputed_from_raw_final_distance": True,
        "raw_final_distance_records": len(rows),
        "known_gap": None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
