#!/usr/bin/env python3
"""Independent arithmetic verifier for the forced-anchor Novel audit.

This file intentionally does not import the runner/summarizer implementation.
It recomputes proposal angles, per-query errors, aggregate coverage and the
scene-cluster bootstrap directly from the frozen manifest and unit JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-9) -> None:
    check(abs(float(left) - float(right)) <= tolerance, f"{left} != {right}")


def angle_error(left: float, right: float) -> float:
    # arccos(cos(delta)) is independent of the wrap implementation in the
    # production summarizer and returns a value in [0, 180].
    delta = math.radians(float(left) - float(right))
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(delta)))))


def aux_angle(aux: list[float]) -> float:
    check(len(aux) == 2, "bad aux pose")
    return math.degrees(math.atan2(float(aux[1]), float(aux[0])))


def bootstrap(rows: list[dict], seed: int, resamples: int) -> tuple[float, list[float]]:
    scenes = sorted({row["scene"] for row in rows})
    grouped = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.choice(scenes, size=len(scenes), replace=True)
        values = [
            row["advantage"]
            for scene in sampled
            for row in grouped[str(scene)]
        ]
        estimates[index] = np.mean(values)
    return float(np.mean(estimates <= 0.0)), np.quantile(
        estimates, [0.025, 0.975]
    ).tolist()


def recompute(manifest: dict, result_root: Path, target: str) -> dict:
    rows = []
    reproduction = []
    for index, spec in enumerate(manifest["records"]):
        path = result_root / f"{index:03d}_{spec['unit']}.json"
        check(path.is_file(), f"missing {path}")
        unit = json.loads(path.read_text())
        check(unit["identity"] == spec["identity"], "identity mismatch")
        check(unit["factual_anchor"] == spec["factual_anchor"], "factual changed")
        check(
            unit["counterfactual_anchors"] == spec["counterfactual_anchors"],
            "control anchors changed",
        )
        close(unit["factual"]["bearing_deg"], aux_angle(unit["factual"]["aux_pose"]))
        target_angle = float(spec[target])
        factual = angle_error(unit["factual"]["bearing_deg"], target_angle)
        controls = []
        for proposal in unit["counterfactuals"]:
            close(proposal["bearing_deg"], aux_angle(proposal["aux_pose"]))
            controls.append(angle_error(proposal["bearing_deg"], target_angle))
        mean_control = float(np.mean(controls))
        rows.append(
            {
                "scene": spec["scene"],
                "factual": factual,
                "control": mean_control,
                "advantage": mean_control - factual,
                "factual_hit": factual <= 30.0,
                "control_hit": float(np.mean(np.asarray(controls) <= 30.0)),
            }
        )
        reproduction.append(
            angle_error(unit["factual"]["bearing_deg"], spec["logged_bearing_deg"])
        )
    return {"rows": rows, "reproduction": reproduction}


def verify(args: argparse.Namespace) -> dict:
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = json.loads(args.report.read_text())
    check(len(manifest["records"]) == 19, "population changed")
    check(report["records"] == 19 and report["scene_clusters"] == 12, "bad report N")
    check(report["final14_accessed"] is False, "final14 flag changed")
    check(report["habitat_rollout"] is False, "rollout flag changed")
    check(
        report["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest hash mismatch",
    )

    outputs = {}
    for label, field, seed in (
        ("shortest_path", "shortest_path_target_deg", 20260816),
        ("direct_goal", "direct_goal_target_deg", 20260817),
    ):
        independent = recompute(manifest, args.result_root, field)
        rows = independent["rows"]
        expected = report[label]
        close(np.mean([row["factual"] for row in rows]), expected["factual_error_deg"]["mean"])
        close(np.mean([row["control"] for row in rows]), expected["counterfactual_expected_error_deg"]["mean"])
        close(np.mean([row["advantage"] for row in rows]), expected["factual_advantage_deg"]["mean"])
        close(sum(row["factual_hit"] for row in rows), expected["factual_count_le_30_deg"])
        close(sum(row["control_hit"] for row in rows), expected["counterfactual_expected_count_le_30_deg"])
        probability, interval = bootstrap(rows, seed, args.resamples)
        close(probability, expected["cluster_bootstrap"]["probability_bootstrap_mean_le_zero"])
        close(interval[0], expected["cluster_bootstrap"]["ci_95_deg"][0])
        close(interval[1], expected["cluster_bootstrap"]["ci_95_deg"][1])
        outputs[label] = {
            "factual_mean_error_deg": float(np.mean([row["factual"] for row in rows])),
            "control_mean_error_deg": float(np.mean([row["control"] for row in rows])),
            "mean_advantage_deg": float(np.mean([row["advantage"] for row in rows])),
            "ci_95_deg": interval,
            "factual_hits_le_30": int(sum(row["factual_hit"] for row in rows)),
            "control_expected_hits_le_30": float(sum(row["control_hit"] for row in rows)),
        }
    reproduction = recompute(
        manifest, args.result_root, "shortest_path_target_deg"
    )["reproduction"]
    close(
        np.mean(reproduction),
        report["local_vs_logged_factual_bearing_error_deg"]["mean"],
    )
    check(
        report["decision"]
        == "dino_visual_context_advantage_not_supported_stop_novel_dino_branch",
        "decision changed",
    )
    return {
        "schema_version": "raw_novel_forced_anchor_independent_verification_v1_20260816",
        "verified": True,
        "records": 19,
        "scene_clusters": 12,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "reproduction_mean_error_deg": float(np.mean(reproduction)),
        "targets": outputs,
        "decision": report["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
