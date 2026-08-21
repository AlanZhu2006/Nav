#!/usr/bin/env python3
"""Summarize the frozen disjoint GOAT certified-arrival confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "goat_certified_arrival_summary_v1_20260815"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(0, min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def summarize(manifest: Mapping[str, Any], tasks: list[Mapping[str, Any]]) \
        -> dict[str, Any]:
    expected = [
        (str(item["scene_id"]), str(item["episode_id"]))
        for item in manifest["episodes"]
    ]
    require(len(tasks) == len(expected), "task count differs from manifest")
    records = []
    for index, task in enumerate(tasks):
        require(task.get("complete") is True, f"task {index} incomplete")
        require(int(task.get("episode_index", -1)) == index,
                f"task {index} index changed")
        require(task.get("ground_truth_used_by_decision") is False,
                f"task {index} consumed GT in decision")
        require(task.get("arrival_contract") == manifest["arrival_contract"],
                f"task {index} arrival contract changed")
        record = task.get("record")
        require(isinstance(record, Mapping), f"task {index} has no record")
        require(record.get("status") == "complete",
                f"task {index} record incomplete")
        actual = (str(record.get("scene_id")), str(record.get("episode_id")))
        require(actual == expected[index], f"task {index} identity changed")
        require(record.get("first_task", [None, None])[1] == "image",
                f"task {index} is not ImageGoal-first")
        records.append(dict(record))

    certified = [bool(item["certified_success"]) for item in records]
    legacy = [bool(item["legacy_first_zero_success_counterfactual"])
              for item in records]
    certified_stops = [bool(item["certified_stop"]) for item in records]
    gains = sum(right and not left for left, right in zip(legacy, certified))
    losses = sum(left and not right for left, right in zip(legacy, certified))
    true_stops = sum(stop and success
                     for stop, success in zip(certified_stops, certified))
    false_stops = sum(stop and not success
                      for stop, success in zip(certified_stops, certified))
    true_stop_scenes = len({
        item["scene_id"]
        for item in records
        if item["certified_stop"] and item["certified_success"]
    })
    gate = manifest["primary_confirmation_gate"]
    gate_passed = bool(
        false_stops <= int(gate["maximum_false_certified_stops"])
        and true_stops >= int(gate["minimum_true_certified_stops"])
        and true_stop_scenes >= int(gate["minimum_true_stop_scenes"])
    )

    zero_events = []
    for item in records:
        for plan in item.get("plans", []):
            decision = plan.get("stop_decision")
            if decision is None:
                continue
            distance = decision.get("post_decision_official_distance_m")
            zero_events.append({
                "scene_id": item["scene_id"],
                "episode_id": item["episode_id"],
                "plan_index": int(plan["plan_index"]),
                "authorized": bool(decision["authorized_subtask_stop"]),
                "official_arrival_025": bool(
                    distance is not None and float(distance) < 0.25),
                "official_distance_m": distance,
                "reason": str(decision["reason"]),
            })
    event_false_positives = sum(
        item["authorized"] and not item["official_arrival_025"]
        for item in zero_events)
    event_true_positives = sum(
        item["authorized"] and item["official_arrival_025"]
        for item in zero_events)

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "disjoint GOAT first-ImageGoal arrival confirmation",
        "is_full_goat_benchmark_score": False,
        "ground_truth_used_by_decision": False,
        "episode_count": len(records),
        "scene_count": len({item["scene_id"] for item in records}),
        "certified": {
            "successes": int(sum(certified)),
            "stops": int(sum(certified_stops)),
            "true_stops": int(true_stops),
            "false_stops": int(false_stops),
            "true_stop_scenes": int(true_stop_scenes),
            "safe_stalls": int(sum(bool(item["safe_stall"])
                                   for item in records)),
            "forced_guard_stops": int(sum(bool(item["forced_guard_stop"])
                                          for item in records)),
        },
        "legacy_first_zero_counterfactual": {
            "successes": int(sum(legacy)),
        },
        "paired_certified_minus_legacy_first_zero": {
            "gains": int(gains),
            "losses": int(losses),
            "exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses),
        },
        "zero_proposal_events": {
            "count": len(zero_events),
            "authorized_true_positives": int(event_true_positives),
            "authorized_false_positives": int(event_false_positives),
            "records": zero_events,
        },
        "same_batch_fallback_count": int(sum(
            int(item["same_batch_fallback_count"]) for item in records)),
        "extra_resample_count": int(sum(
            int(item["extra_resample_count"]) for item in records)),
        "primary_confirmation_gate": dict(gate),
        "primary_gate_passed": gate_passed,
        "next_action": (
            "certified_semantic_stop_confirmed_for_imagegoal_subtasks"
            if gate_passed else
            "do_not_claim_deployable_goat_semantic_stop"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.manifest.is_file(), "manifest is missing")
    require(not args.out_dir.exists(), "output directory already exists")
    manifest = json.loads(args.manifest.read_text())
    paths = [args.episode_dir / f"episode_{index:02d}.json"
             for index in range(len(manifest["episodes"]))]
    require(all(path.is_file() for path in paths), "episode output is missing")
    tasks = [json.loads(path.read_text()) for path in paths]
    report = summarize(manifest, tasks)
    report.update({
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "episode_outputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in paths
        ],
    })
    args.out_dir.mkdir(parents=True)
    atomic_json(args.out_dir / "report.json", report)
    atomic_json(args.out_dir / "SHA256SUMS.json", {
        "report.json": sha256_file(args.out_dir / "report.json"),
    })
    (args.out_dir / "SEALED").touch(exist_ok=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

