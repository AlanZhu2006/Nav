#!/usr/bin/env python3
"""Independently verify Table-III aggregates against raw arm metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def spl(success: int, geodesic: float, path: float) -> float:
    return float(success) * geodesic / max(geodesic, path, 1e-9)


def close(first: float, second: float, label: str) -> None:
    require(abs(float(first) - float(second)) <= 1e-12,
            f"{label} does not reproduce")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Table-III verification exists")
    summary = json.loads(args.summary.read_text())
    require(args.summary.with_name(args.summary.name + ".sha256").read_text().split()
            == [sha256(args.summary), args.summary.name],
            "Table-III summary receipt changed")
    require(summary["histories"] == 48 and summary["queries"] == 96,
            "Table-III summary is incomplete")
    manifest = json.loads((args.run_root /
        "query_population/role_pairs/manifest.json").read_text())
    require(len(manifest["episodes"]) == 48, "Table-III manifest is incomplete")
    records = []
    raw_rows = 0
    for index, episode in enumerate(manifest["episodes"]):
        root = (args.run_root / "evaluation/natural_direction" /
                f"{index:03d}_{episode['scene']}_{episode['episode']}")
        completion = json.loads((root / "completion.json").read_text())
        completion_path = root / "completion.json"
        require(completion_path.with_name("completion.json.sha256").read_text().split()
                == [sha256(completion_path), "completion.json"],
                f"completion receipt changed at {index}")
        raw_by_arm = {}
        for arm in ("mono_native", "mono_cec"):
            with (root / arm / "metric.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2
                    and {row["analysis_role"] for row in rows}
                    == {"novel", "revisit"},
                    f"raw paired metrics changed at {index}/{arm}")
            raw_by_arm[arm] = {row["analysis_role"]: row for row in rows}
            for row in rows:
                role = row["analysis_role"]
                reached = int(row["reached"])
                require(reached == int(completion["outcomes"][arm][role]),
                        f"completion/raw outcome mismatch {index}/{arm}/{role}")
                raw_rows += 1
        for role in ("novel", "revisit"):
            native = raw_by_arm["mono_native"][role]
            cec = raw_by_arm["mono_cec"][role]
            records.append({
                "scene": episode["scene"], "bin_name": episode["bin_name"],
                "role": role, "native": int(native["reached"]),
                "cec": int(cec["reached"]),
                "native_spl": spl(int(native["reached"]),
                                  float(native["geodesic_m"]),
                                  float(native["path_len_m"])),
                "cec_spl": spl(int(cec["reached"]),
                               float(cec["geodesic_m"]),
                               float(cec["path_len_m"])),
                "certificate_accept": int(cec["certificate_accept_plans"]) > 0,
            })
    require(raw_rows == 192, "expected 48 histories x 2 arms x 2 roles")
    for bin_name in ("0_to_20_m", "20_to_30_m", "30_to_50_m"):
        bin_rows = [row for row in records if row["bin_name"] == bin_name]
        require(len(bin_rows) == 32, f"{bin_name}: raw query count changed")
        for role in ("all", "novel", "revisit"):
            rows = bin_rows if role == "all" else [
                row for row in bin_rows if row["role"] == role]
            reported = summary["bins"][bin_name][role]
            require(reported["queries"] == len(rows),
                    f"{bin_name}/{role}: query count changed")
            require(reported["scene_clusters"]
                    == len({row["scene"] for row in rows}),
                    f"{bin_name}/{role}: scene count changed")
            close(reported["mono_native_SR"],
                  sum(row["native"] for row in rows) / len(rows),
                  f"{bin_name}/{role}/native SR")
            close(reported["mono_cec_SR"],
                  sum(row["cec"] for row in rows) / len(rows),
                  f"{bin_name}/{role}/CEC SR")
            close(reported["mono_native_SPL"],
                  sum(row["native_spl"] for row in rows) / len(rows),
                  f"{bin_name}/{role}/native SPL")
            close(reported["mono_cec_SPL"],
                  sum(row["cec_spl"] for row in rows) / len(rows),
                  f"{bin_name}/{role}/CEC SPL")
            gains = sum(row["cec"] == 1 and row["native"] == 0 for row in rows)
            losses = sum(row["cec"] == 0 and row["native"] == 1 for row in rows)
            require(reported["cec_vs_native_gains"] == gains
                    and reported["cec_vs_native_losses"] == losses,
                    f"{bin_name}/{role}: paired counts changed")
            close(reported["mcnemar_exact_p"], mcnemar(gains, losses),
                  f"{bin_name}/{role}/McNemar")
            require(reported["certificate_accept_queries"]
                    == sum(row["certificate_accept"] for row in rows),
                    f"{bin_name}/{role}: authorization count changed")
    result = {
        "schema_version": "hm3d_table3_actual_mono_result_verification_v1_20260830",
        "verified": True, "summary_sha256": sha256(args.summary),
        "raw_metric_rows": raw_rows, "histories": 48, "queries": 96,
        "partial_results_reported": False,
        "fallback_completion_used": False,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n")


if __name__ == "__main__":
    main()
