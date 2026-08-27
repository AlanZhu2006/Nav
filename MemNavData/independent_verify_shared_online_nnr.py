#!/usr/bin/env python3
"""Independent raw-file recount for the actual-online NNR report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ARMS = ("native", "known_direct", "certified", "certified_budget", "certified_graph")
CONTRASTS = (
    ("known_direct", "native"),
    ("certified", "native"),
    ("certified_graph", "native"),
    ("certified_graph", "certified_budget"),
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    require(len(values) == 1, f"wrong metric row count: {path}")
    return values[0]


def mcnemar(gains: int, losses: int) -> float:
    count = gains + losses
    if count == 0:
        return 1.0
    small = min(gains, losses)
    return min(1.0, 2 * sum(math.comb(count, k) for k in range(small + 1)) / 2**count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), "verification output exists")
    manifest_path = args.benchmark_root / "manifest.json"
    require(sha(manifest_path) == args.expected_manifest_sha, "manifest changed")
    manifest = read(manifest_path)
    report = read(args.report)
    require(report["benchmark_manifest_sha256"] == args.expected_manifest_sha,
            "report used another manifest")
    records = []
    for index, source in enumerate(manifest["accepted"]):
        root = args.run_root / "scenes" / (
            f"{index:03d}_{source['scene']}_{source['episode']}"
        )
        require(root.is_dir(), f"missing episode {index}")
        metrics = {arm: row(root / arm / "metric.csv") for arm in ARMS}
        plans = {
            arm: read(root / arm / f"{source['episode']}_plans.json")
            for arm in ARMS
        }
        for arm in ARMS[1:]:
            for key in ("frozen_legA", "frozen_legB"):
                require(plans[arm][key] == plans["native"][key],
                        f"{index}/{arm}: shared plan mismatch")
            for kind in ("rollout_traces", "memory_traces"):
                for leg in ("legA", "legB"):
                    require(plans[arm][kind][leg] == plans["native"][kind][leg],
                            f"{index}/{arm}: shared {kind}/{leg} mismatch")
        outcomes = {arm: int(metrics[arm]["reached_C"]) for arm in ARMS}
        if outcomes["certified_graph"] > outcomes["certified_budget"]:
            require(any(
                item.get("certified_graph_rescue_active") is True
                and item.get("certified_graph_reason") == "historical_subgoal"
                for item in plans["certified_graph"]["legC"]
            ), f"{index}: graph gain without graph execution")
        records.append({"scene": source["scene"], "outcomes": outcomes})

    require(len(records) == report["constructible_population_size"], "report N differs")
    raw_arm_counts = {arm: sum(r["outcomes"][arm] for r in records) for arm in ARMS}
    for arm in ARMS:
        require(raw_arm_counts[arm] == report["arms"][arm]["successes"],
                f"{arm}: report count differs")
    raw_contrasts = {}
    for treatment, control in CONTRASTS:
        gains = sum(r["outcomes"][treatment] > r["outcomes"][control] for r in records)
        losses = sum(r["outcomes"][treatment] < r["outcomes"][control] for r in records)
        name = f"{treatment}_minus_{control}"
        reported = report["contrasts"][name]
        require((gains, losses) == (reported["gains"], reported["losses"]),
                f"{name}: discordance differs")
        p = mcnemar(gains, losses)
        require(abs(p - float(reported["exact_mcnemar_two_sided_p"])) <= 1e-15,
                f"{name}: McNemar differs")
        raw_contrasts[name] = {"gains": gains, "losses": losses, "p": p}

    verification = {
        "schema_version": "independent_shared_online_nnr_verification_v1_20260814",
        "benchmark_manifest_sha256": args.expected_manifest_sha,
        "report_sha256": sha(args.report),
        "episodes": len(records),
        "arm_successes": raw_arm_counts,
        "contrasts": raw_contrasts,
        "all_shared_A_B_records_equal": True,
        "all_graph_gains_have_historical_subgoals": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(verification, indent=2, sort_keys=True) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {args.out.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
