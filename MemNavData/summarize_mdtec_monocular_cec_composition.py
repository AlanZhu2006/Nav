#!/usr/bin/env python3
"""Fail-closed formal summary for the MDTEC monocular x CEC composition run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.mdtec_monocular_cec_composition import (
    ARMS,
    exact_mcnemar_two_sided,
    paired_contrast,
    require,
    rotated_arm_order,
    scene_cluster_interval,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(run_root: Path, manifest: dict[str, Any],
              protocol_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenes = manifest["scenes"]
    for scene_index, scene in enumerate(scenes):
        matches = sorted((run_root / "scenes").glob(f"{scene_index:02d}_{scene}"))
        require(len(matches) == 1, f"missing/duplicate output for {scene}")
        scene_root = matches[0]
        receipt = json.loads((scene_root / "server_receipt.json").read_text())
        require(receipt.get("same_process_all_arms") is True,
                f"{scene}: same-process receipt failed")
        csv_path = scene_root / "depth_arms.csv"
        meta_path = scene_root / "run_meta.json"
        require(csv_path.is_file() and meta_path.is_file(),
                f"{scene}: incomplete evaluator output")
        meta = json.loads(meta_path.read_text())
        require(meta.get("formal") is True and meta.get("records") == 4,
                f"{scene}: invalid formal run metadata")
        require(meta.get("protocol_sha256") == protocol_sha,
                f"{scene}: protocol SHA mismatch")
        with csv_path.open(newline="") as handle:
            scene_rows = list(csv.DictReader(handle))
        require(len(scene_rows) == 4, f"{scene}: expected four arm rows")
        expected_episodes = [row["episode"] for row in manifest["episodes"][scene][:2]]
        for row in scene_rows:
            episode = row["episode"]
            arm = row["arm"]
            require(episode in expected_episodes and arm in ARMS,
                    f"{scene}: unknown episode/arm")
            episode_index = expected_episodes.index(episode)
            require(json.loads(row["arm_order"]) == list(
                rotated_arm_order(scene_index, episode_index)),
                f"{scene}/{episode}: arm order changed")
            require(row["metric_depth_sensor_consumed_any"] in ("True", "False"),
                    f"{scene}/{episode}/{arm}: malformed metric-sensor flag")
            require(row["metric_depth_sensor_consumed_any"] == "False",
                    f"{scene}/{episode}/{arm}: consumed simulator metric depth")
            if arm == "raw_cec":
                require(int(row["certified_runtime_failure_count"]) == 0,
                        f"{scene}/{episode}: raw_cec certificate runtime failure "
                        "-- audit invalid, fail closed")
            rows.append(row)
    require(len(rows) == 4 * len(scenes), "row count does not match population")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha256(protocol_path)
    manifest_path = Path(protocol["manifest"]["path"])
    require(sha256(manifest_path) == protocol["manifest"]["sha256"],
            "Fresh160 manifest SHA changed since protocol was frozen")
    manifest = json.loads(manifest_path.read_text())

    rows = load_rows(run_root, manifest, protocol_sha)

    # intent-to-treat: all 40 episodes x 2 arms, keyed on B success
    itt_rows = [{"scene": r["scene"], "episode": r["episode"], "arm": r["arm"],
                "reached": r["reached"]} for r in rows]
    itt_contrast = paired_contrast(itt_rows, "raw_cec", "raw_native")
    itt_ci = scene_cluster_interval(itt_rows, "raw_cec", "raw_native",
                                    seed=20260819, resamples=100000)

    # conditional: only rows where Goal-A succeeded
    cond_rows = [{"scene": r["scene"], "episode": r["episode"], "arm": r["arm"],
                 "reached": r["reached"]} for r in rows if int(r["reached_A"]) == 1]
    cond_contrast = (paired_contrast(cond_rows, "raw_cec", "raw_native")
                     if cond_rows else None)
    cond_ci = (scene_cluster_interval(cond_rows, "raw_cec", "raw_native",
                                      seed=20260819, resamples=100000)
              if cond_rows else None)

    def arm_summary(arm: str, pool: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [r for r in pool if r["arm"] == arm]
        n = len(vals)
        sr = sum(int(r["reached"]) for r in vals) / n if n else None
        return {"n": n, "sr": sr}

    summary = {
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "manifest_sha256": protocol["manifest"]["sha256"],
        "scenes": len(manifest["scenes"]),
        "rows": len(rows),
        "intent_to_treat": {
            "n": itt_contrast["n"],
            "raw_native": arm_summary("raw_native", itt_rows),
            "raw_cec": arm_summary("raw_cec", itt_rows),
            "contrast": itt_contrast,
            "scene_cluster_bootstrap_95": itt_ci,
        },
        "goal_a_conditional": {
            "n": cond_contrast["n"] if cond_contrast else 0,
            "raw_native": arm_summary("raw_native", cond_rows) if cond_rows else None,
            "raw_cec": arm_summary("raw_cec", cond_rows) if cond_rows else None,
            "contrast": cond_contrast,
            "scene_cluster_bootstrap_95": cond_ci,
        },
        "certified_totals": {
            "requests": sum(int(r["certified_request_count"])
                            for r in rows if r["arm"] == "raw_cec"),
            "accepts": sum(int(r["certified_accept_count"])
                          for r in rows if r["arm"] == "raw_cec"),
            "runtime_failures": sum(int(r["certified_runtime_failure_count"])
                                    for r in rows if r["arm"] == "raw_cec"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
