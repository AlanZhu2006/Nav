#!/usr/bin/env python3
"""Aggregate paired SR/SPL by frozen HM3D geodesic-length bin."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")


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
    return float(success) * float(geodesic) / max(float(geodesic), float(path), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Table-III analysis exists")
    population = args.run_root / "query_population"
    verification = json.loads((population / "independent_verification.json").read_text())
    require(verification["verified"] is True
            and verification["formal_policy_evaluation_authorized"] is True,
            "Table-III population was not independently authorized")
    manifest = json.loads((population / "role_pairs/manifest.json").read_text())
    require(len(manifest["episodes"]) == 48, "Table-III powered population changed")
    records = []
    for index, episode in enumerate(manifest["episodes"]):
        matches = list((args.run_root / "evaluation/natural_direction").glob(
            f"{index:03d}_{episode['scene']}_{episode['episode']}/completion.json"))
        require(len(matches) == 1, f"missing Table-III completion {index}")
        path = matches[0]
        require(path.with_name(path.name + ".sha256").read_text().split()
                == [sha256(path), path.name], f"completion receipt changed {index}")
        completion = json.loads(path.read_text())
        require(int(completion["history_index"]) == index
                and completion["arms"] == list(ARMS),
                f"paired completion identity changed {index}")
        for role in ROLES:
            records.append({
                "population_index": index, "scene": episode["scene"],
                "bin_name": episode["bin_name"], "role": role,
                "native": int(completion["outcomes"]["mono_native"][role]),
                "cec": int(completion["outcomes"]["mono_cec"][role]),
                "native_geo": float(completion["geodesic_m"]["mono_native"][role]),
                "cec_geo": float(completion["geodesic_m"]["mono_cec"][role]),
                "native_path": float(completion["path_len_m"]["mono_native"][role]),
                "cec_path": float(completion["path_len_m"]["mono_cec"][role]),
                "certificate_accept_plans": int(
                    completion["certificate_accept_plans"][role]),
            })
    by_bin = {}
    for spec in manifest["contract"]["bins_m"]:
        name = spec["name"]
        rows = [row for row in records if row["bin_name"] == name]
        require(len(rows) == 32, f"{name}: expected 16 histories / 32 queries")
        role_summary = {}
        for role in ("all", *ROLES):
            subset = rows if role == "all" else [row for row in rows if row["role"] == role]
            gains = sum(row["cec"] == 1 and row["native"] == 0 for row in subset)
            losses = sum(row["cec"] == 0 and row["native"] == 1 for row in subset)
            role_summary[role] = {
                "queries": len(subset),
                "scene_clusters": len({row["scene"] for row in subset}),
                "mono_native_SR": sum(row["native"] for row in subset) / len(subset),
                "mono_cec_SR": sum(row["cec"] for row in subset) / len(subset),
                "mono_native_SPL": sum(spl(row["native"], row["native_geo"],
                                             row["native_path"]) for row in subset) / len(subset),
                "mono_cec_SPL": sum(spl(row["cec"], row["cec_geo"],
                                          row["cec_path"]) for row in subset) / len(subset),
                "cec_vs_native_gains": gains,
                "cec_vs_native_losses": losses,
                "mcnemar_exact_p": mcnemar(gains, losses),
                "certificate_accept_queries": sum(
                    row["certificate_accept_plans"] > 0 for row in subset),
            }
        by_bin[name] = role_summary
    result = {
        "schema_version": "hm3d_table3_actual_mono_result_v1_20260830",
        "population_verification_sha256": sha256(
            population / "independent_verification.json"),
        "histories": 48, "queries": 96,
        "bins": by_bin, "records": records,
        "partial_results_reported": False,
        "fallback_completion_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n")


if __name__ == "__main__":
    main()
