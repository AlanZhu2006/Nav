#!/usr/bin/env python3
"""Independent raw-CSV recount for the HM3D mixed-role summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ARMS = ("native", "raw_direct", "raw_fixed_bearing", "geometry_fixed", "certified")
ROLES = ("novel", "revisit")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    manifest = json.loads(
        (args.root / "benchmarks/natural_direction/manifest.json").read_text())
    expected = {(row["scene"], row["episode"]) for row in manifest["episodes"]}
    counts = {scope: defaultdict(int) for scope in (*ROLES, "all")}
    identities = set()
    intervention = {role: defaultdict(int) for role in ROLES}
    for root in sorted((args.root / "evaluation/natural_direction").iterdir()):
        if not root.is_dir() or root.name == "skipped":
            continue
        contract = json.loads((root / "episode_contract.json").read_text())
        identity = (contract["scene"], contract["episode"])
        if identity in identities or identity not in expected:
            raise RuntimeError(f"invalid identity {identity}")
        identities.add(identity)
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 2:
                raise RuntimeError("arm does not contain two queries")
            for row in rows:
                role = row["analysis_role"]
                if role not in ROLES:
                    raise RuntimeError("invalid role")
                value = int(row["reached"])
                counts[role][arm] += value
                counts["all"][arm] += value
                if arm == "certified":
                    intervention[role]["queries"] += 1
                    intervention[role]["certificate_accept_queries"] += int(
                        int(row["certificate_accept_plans"]) > 0)
                    intervention[role]["takeover_queries"] += int(
                        int(row["adapter_takeover_plans"]) > 0)
                    intervention[role]["runtime_failure_plans"] += int(
                        row["runtime_failure_plans"])
    if identities != expected:
        raise RuntimeError("raw evaluation population is incomplete")
    recounted = {scope: dict(values) for scope, values in counts.items()}
    if recounted != summary["arm_successes"]:
        raise RuntimeError("summary success counts do not match raw CSVs")
    recounted_intervention = {
        role: dict(values) for role, values in intervention.items()
    }
    if recounted_intervention != summary["certified_intervention"]:
        raise RuntimeError("summary intervention counts do not match raw CSVs")
    result = {
        "schema_version": "hm3d_mixed_role_independent_verification_v1_20260818",
        "verified": True,
        "summary_sha256": sha256_file(args.summary),
        "benchmark_manifest_sha256": sha256_file(
            args.root / "benchmarks/natural_direction/manifest.json"),
        "histories": len(expected),
        "scenes": len({scene for scene, _episode in expected}),
        "arm_successes": recounted,
        "certified_intervention": recounted_intervention,
    }
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
