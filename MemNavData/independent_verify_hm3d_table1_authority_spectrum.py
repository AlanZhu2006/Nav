#!/usr/bin/env python3
"""Independent raw-receipt verifier for the HM3D authority spectrum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ARMS = (
    "mono_native",
    "mono_raw_fixed",
    "mono_unthresholded_witness",
    "mono_cec",
)
CONTRASTS = (
    ("mono_cec", "mono_native"),
    ("mono_raw_fixed", "mono_native"),
    ("mono_unthresholded_witness", "mono_native"),
    ("mono_cec", "mono_raw_fixed"),
    ("mono_cec", "mono_unthresholded_witness"),
)
SCHEMA = "hm3d_table1_authority_spectrum_independent_v1_20260904"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(gains, losses) + 1)) / 2 ** discordant
    return min(1.0, 2.0 * tail)


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
            "manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 28, "history denominator changed")
    summary = json.loads(args.summary.read_text())
    require(summary.get("status") == "complete", "summary is incomplete")
    require(summary.get("benchmark_manifest_sha256") ==
            args.expected_manifest_sha256, "summary manifest changed")

    raw: dict[tuple[str, str, str], int] = {}
    accept: dict[tuple[str, str, str], int] = {}
    source_files = [manifest_path, args.summary]
    proposal_pairs = 0
    for index, item in enumerate(manifest["episodes"]):
        label = f'{index:03d}_{item["scene"]}_{item["episode"]}'
        root = args.run_root / "evaluation" / "natural_direction" / label
        completion_path = root / "completion.json"
        completion_hash = root / "completion.json.sha256"
        source_files.extend([completion_path, completion_hash])
        require(sha256(completion_path) ==
                completion_hash.read_text().split()[0],
                f"{label}: completion hash mismatch")
        completion = json.loads(completion_path.read_text())
        expected_order = list(ARMS[index % 4:] + ARMS[:index % 4])
        require(completion.get("arm_order") == expected_order,
                f"{label}: arm order mismatch")
        require(completion.get("prefix_equality") is True and
                completion.get("initial_proposal_equality") is True,
                f"{label}: pairing audit failed")
        require(completion.get("runtime_role_visibility") == "none",
                f"{label}: role visibility changed")
        require(completion.get("fresh_confirmation") is False,
                f"{label}: scope mislabeled")
        for role in ("novel", "revisit"):
            require(completion["initial_proposal_audit"][role]
                    ["proposal_fields_equal"] is True,
                    f"{label}/{role}: proposal mismatch")
            proposal_pairs += 1
        for arm in ARMS:
            metric_path = root / arm / "metric.csv"
            source_files.append(metric_path)
            with metric_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2, f"{label}/{arm}: row count changed")
            for row in rows:
                role = row["analysis_role"]
                require(role in ("novel", "revisit"),
                        f"{label}/{arm}: invalid role")
                reached = int(row["reached"])
                distance = float(row["final_goal_dist_m"])
                require(reached == int(distance < 1.0),
                        f"{label}/{arm}/{role}: SR mismatch")
                require(row["navdp_depth_source"] == "monocular_sidecar" and
                        int(row["metric_depth_sensor_consumed_any"]) == 0 and
                        int(row["runtime_failure_plans"]) == 0,
                        f"{label}/{arm}/{role}: runtime contract changed")
                key = (str(item["scene"]), str(item["episode"]) + "/" + role, arm)
                raw[key] = reached
                accept[key] = int(int(row["certificate_accept_plans"]) > 0)

    recomputed = {}
    for role in ("novel", "revisit", "all"):
        keys = sorted({(scene, episode) for scene, episode, _ in raw
                       if role == "all" or episode.endswith("/" + role)})
        arm_stats = {}
        for arm in ARMS:
            values = [raw[(scene, episode, arm)] for scene, episode in keys]
            arm_stats[arm] = {
                "n": len(values),
                "successes": sum(values),
                "authorized_queries": sum(
                    accept[(scene, episode, arm)] for scene, episode in keys
                ),
            }
            reported = summary["results"][role]["arms"][arm]
            require(reported["n"] == len(values) and
                    reported["successes"] == sum(values),
                    f"summary {role}/{arm} mismatch")
        contrast_stats = {}
        for treatment, reference in CONTRASTS:
            gains = sum(raw[(scene, episode, treatment)] == 1 and
                        raw[(scene, episode, reference)] == 0
                        for scene, episode in keys)
            losses = sum(raw[(scene, episode, treatment)] == 0 and
                         raw[(scene, episode, reference)] == 1
                         for scene, episode in keys)
            result = {
                "gains": gains,
                "losses": losses,
                "exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses),
            }
            name = f"{treatment}_minus_{reference}"
            reported = summary["results"][role]["contrasts"][name]
            require(reported["gains"] == gains and reported["losses"] == losses,
                    f"summary {role}/{name} discordance mismatch")
            require(abs(float(reported["exact_mcnemar_two_sided_p"]) -
                        result["exact_mcnemar_two_sided_p"])
                    < 1e-12, f"summary {role}/{name} p-value mismatch")
            contrast_stats[name] = result
        recomputed[role] = {"arms": arm_stats, "contrasts": contrast_stats}

    payload = {
        "schema_version": SCHEMA,
        "verified": True,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "summary_sha256": sha256(args.summary),
        "histories": 28,
        "scene_clusters": len({item["scene"] for item in manifest["episodes"]}),
        "queries_per_arm": 56,
        "proposal_pairs_verified": proposal_pairs,
        "runtime_role_visibility": "none",
        "metric_depth_sensor_reads": 0,
        "recomputed": recomputed,
        "source_file_count": len(source_files),
        "source_digest_sha256": hashlib.sha256("".join(
            f"{sha256(path)}  {path}\n" for path in sorted(source_files)
        ).encode()).hexdigest(),
    }
    require(not args.out.exists(), "verification output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
