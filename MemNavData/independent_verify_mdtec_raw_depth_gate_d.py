#!/usr/bin/env python3
"""Independent raw-file recount for the MDTEC Gate-D formal report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("metric_teacher", "zero_depth", "raw_first40")


def fail(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(run_root):
    rows = []
    for path in sorted((Path(run_root) / "scenes").glob("*_*")):
        with (path / "depth_arms.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                plans = json.loads((path / row["plans_file"]).read_text())
                fail(plans["arm"] == row["arm"], "raw plan identity mismatch")
                fail(int(row["reached"]) == (float(row["final_dist_m"]) < 1.0),
                     "independent success recount mismatch")
                if row["arm"] == "raw_first40":
                    fail(row["metric_depth_sensor_consumed_any"] == "False",
                         "raw consumed metric sensor")
                    for plan in plans["plans"]:
                        receipt = plan["monocular_depth_receipt"]
                        fail(receipt["image_sha256"] and
                             receipt["depth_png_sha256"], "missing wire hashes")
                        if int(receipt["frame_index"]) < 40:
                            fail(receipt["depth_nonzero_fraction"] == 0.0,
                                 "nonzero bootstrap depth")
                        else:
                            fail(receipt["scale_active"] is True,
                                 "inactive post-40 scale")
                            scale = receipt["scale_receipt"]
                            fail(scale["whole_episode_ground_cache_consumed"]
                                 is False, "whole episode cache consumed")
                row["reached"] = int(row["reached"])
                row["spl"] = float(row["spl"])
                rows.append(row)
    fail(len(rows) == 120, "independent row count mismatch")
    fail(len({(r["scene"], r["episode"]) for r in rows}) == 40,
         "independent pair count mismatch")
    return rows


def contrast(rows, a, b, seed, resamples):
    units = {}
    for row in rows:
        units.setdefault((row["scene"], row["episode"]), {})[row["arm"]] = row["reached"]
    delta = [v[a] - v[b] for v in units.values()]
    gains = sum(v[a] == 1 and v[b] == 0 for v in units.values())
    losses = sum(v[a] == 0 and v[b] == 1 for v in units.values())
    by_scene = {}
    for (scene, _), values in units.items():
        by_scene.setdefault(scene, []).append(values[a] - values[b])
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for i in range(resamples):
        picked = rng.integers(0, len(scenes), len(scenes))
        values = []
        for j in picked:
            values.extend(by_scene[scenes[int(j)]])
        boot[i] = np.mean(values)
    return {
        "n": len(units), "gains": gains, "losses": losses,
        "ties": len(units) - gains - losses,
        "risk_difference": float(np.mean(delta)),
        "exact_mcnemar_two_sided_p": exact(gains, losses),
        "scene_cluster_bootstrap_risk_difference_95": [
            float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--expected-analysis-sha", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fail(sha(a.analysis) == a.expected_analysis_sha, "analysis SHA mismatch")
    analysis = json.loads(Path(a.analysis).read_text())
    report = json.loads(Path(a.report).read_text())
    rows = load(a.run_root)
    spec = analysis["cluster_bootstrap"]
    independent = {}
    pairs = {
        "raw_first40_vs_metric_teacher": ("raw_first40", "metric_teacher"),
        "zero_depth_vs_metric_teacher": ("zero_depth", "metric_teacher"),
        "raw_first40_vs_zero_depth": ("raw_first40", "zero_depth"),
    }
    for name, (x, y) in pairs.items():
        value = contrast(rows, x, y, int(spec["seed"]), int(spec["resamples"]))
        expected = report["contrasts"][name]
        for key in ("n", "gains", "losses", "ties"):
            fail(value[key] == expected[key], f"{name}/{key} mismatch")
        for key in ("risk_difference", "exact_mcnemar_two_sided_p"):
            fail(close(value[key], expected[key]), f"{name}/{key} mismatch")
        fail(all(close(xv, yv) for xv, yv in zip(
            value["scene_cluster_bootstrap_risk_difference_95"],
            expected["scene_cluster_bootstrap_risk_difference_95"])),
            f"{name}/cluster interval mismatch")
        independent[name] = value
    arms = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {"successes": sum(r["reached"] for r in selected),
                     "n": len(selected),
                     "sr": float(np.mean([r["reached"] for r in selected])),
                     "mean_spl": float(np.mean([r["spl"] for r in selected]))}
        fail(arms[arm]["successes"] == report["arms"][arm]["successes"],
             f"{arm} success count mismatch")
        fail(close(arms[arm]["sr"], report["arms"][arm]["sr"]),
             f"{arm} SR mismatch")
        fail(close(arms[arm]["mean_spl"], report["arms"][arm]["mean_spl"]),
             f"{arm} SPL mismatch")
    result = {"verified": True, "rows": len(rows), "pairs": 40,
              "arms": arms, "contrasts": independent,
              "formal_decision": report["decision"]}
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
