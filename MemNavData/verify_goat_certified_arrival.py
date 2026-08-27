#!/usr/bin/env python3
"""Independent raw-record verifier for GOAT certified arrival."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mcnemar(gains: int, losses: int) -> float:
    total = gains + losses
    if not total:
        return 1.0
    lower = min(gains, losses)
    probability = 2.0 * sum(math.comb(total, k)
                            for k in range(lower + 1)) / (2 ** total)
    return min(1.0, probability)


def verify(manifest: dict, tasks: list[dict], report: dict) -> dict:
    require(len(tasks) == len(manifest["episodes"]), "task count changed")
    success = []
    legacy = []
    stops = []
    scenes = []
    false_events = 0
    true_events = 0
    for index, (expected, task) in enumerate(zip(manifest["episodes"], tasks)):
        require(task["complete"] is True, f"task {index} incomplete")
        require(task["episode_index"] == index, f"task {index} index changed")
        require(task["ground_truth_used_by_decision"] is False,
                f"task {index} decision read GT")
        record = task["record"]
        require((record["scene_id"], record["episode_id"])
                == (expected["scene_id"], expected["episode_id"]),
                f"task {index} identity changed")
        scenes.append(record["scene_id"])
        success.append(bool(record["certified_success"]))
        legacy.append(bool(
            record["legacy_first_zero_success_counterfactual"]))
        stops.append(bool(record["certified_stop"]))
        for plan in record["plans"]:
            decision = plan.get("stop_decision")
            if decision is None:
                continue
            distance = decision.get("post_decision_official_distance_m")
            arrived = distance is not None and float(distance) < 0.25
            authorized = bool(decision["authorized_subtask_stop"])
            true_events += int(authorized and arrived)
            false_events += int(authorized and not arrived)

    true_stops = sum(stop and ok for stop, ok in zip(stops, success))
    false_stops = sum(stop and not ok for stop, ok in zip(stops, success))
    true_scenes = len({scene for scene, stop, ok in zip(
        scenes, stops, success) if stop and ok})
    gains = sum(right and not left for left, right in zip(legacy, success))
    losses = sum(left and not right for left, right in zip(legacy, success))
    gate = manifest["primary_confirmation_gate"]
    passed = (
        false_stops <= int(gate["maximum_false_certified_stops"])
        and true_stops >= int(gate["minimum_true_certified_stops"])
        and true_scenes >= int(gate["minimum_true_stop_scenes"])
    )

    require(report["episode_count"] == len(tasks), "episode count differs")
    require(report["scene_count"] == len(set(scenes)), "scene count differs")
    require(report["certified"]["successes"] == sum(success),
            "certified successes differ")
    require(report["certified"]["stops"] == sum(stops),
            "certified stops differ")
    require(report["certified"]["true_stops"] == true_stops,
            "true stops differ")
    require(report["certified"]["false_stops"] == false_stops,
            "false stops differ")
    require(report["zero_proposal_events"]["authorized_true_positives"]
            == true_events, "event TP differs")
    require(report["zero_proposal_events"]["authorized_false_positives"]
            == false_events, "event FP differs")
    paired = report["paired_certified_minus_legacy_first_zero"]
    require(paired["gains"] == gains and paired["losses"] == losses,
            "paired counts differ")
    require(math.isclose(
        float(paired["exact_mcnemar_two_sided_p"]),
        mcnemar(gains, losses), abs_tol=1e-15), "McNemar differs")
    require(report["primary_gate_passed"] is bool(passed), "gate differs")
    return {
        "schema_version": "goat_certified_arrival_independent_verification_v1",
        "verified": True,
        "episodes": len(tasks),
        "scenes": len(set(scenes)),
        "certified_successes": int(sum(success)),
        "certified_stops": int(sum(stops)),
        "false_certified_stops": int(false_stops),
        "primary_gate_passed": bool(passed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), "verification output already exists")
    require((args.summary_dir / "SEALED").is_file(), "summary is not sealed")
    manifest = json.loads(args.manifest.read_text())
    tasks = [json.loads((args.episode_dir / f"episode_{index:02d}.json").read_text())
             for index in range(len(manifest["episodes"]))]
    report_path = args.summary_dir / "report.json"
    report = json.loads(report_path.read_text())
    result = verify(manifest, tasks, report)
    result.update({
        "manifest_sha256": sha256_file(args.manifest),
        "report_sha256": sha256_file(report_path),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

