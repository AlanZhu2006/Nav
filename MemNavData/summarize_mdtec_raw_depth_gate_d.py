#!/usr/bin/env python3
"""Fail-closed formal summary for the MDTEC raw-depth Gate D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.mdtec_raw_depth_gate_d import (
    ARMS,
    audit_arm_contract,
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


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0", "", "none"}:
        return False
    raise RuntimeError(f"invalid boolean {value!r}")


def optional_boolean(value: Any) -> bool | None:
    if value in (None, "", "None"):
        return None
    return boolean(value)


def load_rows(run_root: Path, manifest: dict[str, Any],
              protocol_sha: str, analysis_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenes = manifest["selection"]["selected_scenes"]
    for scene_index, scene in enumerate(scenes):
        matches = sorted((run_root / "scenes").glob(
            f"{scene_index:02d}_{scene}"))
        require(len(matches) == 1, f"missing/duplicate output for {scene}")
        scene_root = matches[0]
        receipt_path = scene_root / "server_receipt.json"
        require(receipt_path.is_file(), f"{scene}: missing server receipt")
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("same_process_all_arms") is True,
                f"{scene}: same-process receipt failed")
        require(int(receipt.get("memnav_pid", 0)) > 0
                and int(receipt.get("navdp_pid", 0)) > 0,
                f"{scene}: invalid server PIDs")
        csv_path = scene_root / "depth_arms.csv"
        meta_path = scene_root / "run_meta.json"
        require(csv_path.is_file() and meta_path.is_file(),
                f"{scene}: incomplete evaluator output")
        meta = json.loads(meta_path.read_text())
        require(meta.get("formal") is True and meta.get("records") == 6,
                f"{scene}: invalid formal run metadata")
        require(meta.get("protocol_sha256") == protocol_sha,
                f"{scene}: protocol SHA mismatch")
        with csv_path.open(newline="") as handle:
            scene_rows = list(csv.DictReader(handle))
        require(len(scene_rows) == 6, f"{scene}: expected six arm rows")
        expected_episodes = [row["episode"]
                             for row in manifest["episodes"][scene]]
        for row in scene_rows:
            episode = row["episode"]
            arm = row["arm"]
            require(episode in expected_episodes and arm in ARMS,
                    f"{scene}: unknown episode/arm")
            episode_index = expected_episodes.index(episode)
            require(json.loads(row["arm_order"]) == list(
                rotated_arm_order(scene_index, episode_index)),
                f"{scene}/{episode}: arm order changed")
            require(int(row["arm_position"]) == json.loads(
                row["arm_order"]).index(arm), "arm position mismatch")
            require(row["protocol_sha256"] == protocol_sha,
                    "row protocol SHA mismatch")
            plans_path = scene_root / row["plans_file"]
            require(plans_path.is_file(), "missing raw plans file")
            plans_payload = json.loads(plans_path.read_text())
            require(plans_payload["scene"] == scene
                    and plans_payload["episode"] == episode
                    and plans_payload["arm"] == arm,
                    "plans identity mismatch")
            reached = int(row["reached"])
            final_dist = float(row["final_dist_m"])
            require(reached == int(final_dist < 1.0),
                    "success flag does not equal frozen distance criterion")
            normalized = {
                **row,
                "scene_index": scene_index,
                "episode_index": episode_index,
                "reached": reached,
                "spl": float(row["spl"]),
                "path_len_m": float(row["path_len_m"]),
                "final_dist_m": final_dist,
                "metric_depth_sensor_consumed_any": boolean(
                    row["metric_depth_sensor_consumed_any"]),
                "monocular_frame40_survived": boolean(
                    row["monocular_frame40_survived"]),
                "monocular_first40_scale_valid": optional_boolean(
                    row["monocular_first40_scale_valid"]),
                "monocular_first40_scale_clamped": optional_boolean(
                    row["monocular_first40_scale_clamped"]),
                "active_plan_count": int(row["active_plan_count"]),
                "bootstrap_plan_count": int(row["bootstrap_plan_count"]),
                "receipt_contract_valid": boolean(
                    row["receipt_contract_valid"]),
            }
            audit = audit_arm_contract(arm, {
                "plans": plans_payload["plans"],
                "navdp_depth_source": row["depth_source"],
                "metric_depth_sensor_consumed_any": normalized[
                    "metric_depth_sensor_consumed_any"],
                "monocular_frame40_survived": normalized[
                    "monocular_frame40_survived"],
            })
            require(int(audit["active_plan_count"]) ==
                    normalized["active_plan_count"],
                    "active plan recount mismatch")
            require(int(audit["bootstrap_plan_count"]) ==
                    normalized["bootstrap_plan_count"],
                    "bootstrap plan recount mismatch")
            rows.append(normalized)
    require(len(rows) == 120, "formal Gate D must have 120 arm rows")
    units = Counter((row["scene"], row["episode"]) for row in rows)
    require(len(units) == 40 and set(units.values()) == {3},
            "formal paired coverage changed")
    require(len({row["scene"] for row in rows}) == 20,
            "formal scene coverage changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--expected-protocol-sha", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()
    run_root = Path(cli.run_root).resolve()
    manifest_path = Path(cli.manifest).resolve()
    protocol_path = Path(cli.protocol).resolve()
    analysis_path = Path(cli.analysis).resolve()
    require(sha256(protocol_path) == cli.expected_protocol_sha,
            "formal protocol SHA changed")
    require(sha256(analysis_path) == cli.expected_analysis_sha,
            "analysis implementation SHA changed")
    protocol = json.loads(protocol_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(analysis["parent_protocol"]["sha256"] == cli.expected_protocol_sha,
            "analysis belongs to another protocol")
    rows = load_rows(run_root, manifest, cli.expected_protocol_sha,
                     cli.expected_analysis_sha)
    bootstrap = analysis["cluster_bootstrap"]
    contrasts = {}
    for name, treatment, reference in (
        ("raw_first40_vs_metric_teacher", "raw_first40", "metric_teacher"),
        ("zero_depth_vs_metric_teacher", "zero_depth", "metric_teacher"),
        ("raw_first40_vs_zero_depth", "raw_first40", "zero_depth"),
    ):
        value = paired_contrast(rows, treatment, reference)
        value["scene_cluster_bootstrap_risk_difference_95"] = (
            scene_cluster_interval(
                rows, treatment, reference,
                seed=int(bootstrap["seed"]),
                resamples=int(bootstrap["resamples"])))
        contrasts[name] = value

    arms = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "successes": sum(row["reached"] for row in arm_rows),
            "n": len(arm_rows),
            "sr": float(np.mean([row["reached"] for row in arm_rows])),
            "mean_spl": float(np.mean([row["spl"] for row in arm_rows])),
            "mean_path_len_m": float(np.mean(
                [row["path_len_m"] for row in arm_rows])),
        }
    raw_rows = [row for row in rows if row["arm"] == "raw_first40"]
    raw_metric_safe = all(
        not row["metric_depth_sensor_consumed_any"] for row in raw_rows)
    contract_failures = sum(
        not row["receipt_contract_valid"] for row in raw_rows)
    reached_frame40 = [row for row in raw_rows
                       if row["monocular_frame40_survived"]]
    valid_scale = [row for row in reached_frame40
                   if row["monocular_first40_scale_valid"] is True]
    active_contract = all(row["active_plan_count"] > 0 for row in valid_scale)
    primary = contrasts["raw_first40_vs_metric_teacher"]
    raw_zero = contrasts["raw_first40_vs_zero_depth"]
    strict_zero_improvement = (
        arms["raw_first40"]["sr"] > arms["zero_depth"]["sr"]
        or (arms["raw_first40"]["sr"] == arms["zero_depth"]["sr"]
            and arms["raw_first40"]["mean_spl"] >
            arms["zero_depth"]["mean_spl"]))
    engineering_checks = {
        "all_raw_metric_sensor_consumed_false": raw_metric_safe,
        "receipt_or_scale_contract_failures_zero": contract_failures == 0,
        "raw_minus_metric_point_rd_at_least_minus_10pp":
            primary["risk_difference"] >= -0.10,
        "raw_losses_minus_gains_at_most_four":
            primary["losses"] - primary["gains"] <= 4,
        "raw_strictly_better_than_zero_sr_or_equal_sr_higher_spl":
            strict_zero_improvement,
        "all_frame40_valid_scale_rollouts_consumed_raw": active_contract,
    }
    engineering_authorized = all(engineering_checks.values())
    ci = primary["scene_cluster_bootstrap_risk_difference_95"]
    paper_noninferiority = float(ci[0]) > -0.10
    result = {
        "schema": "mdtec_raw_depth_gate_d_summary_v1_20260819",
        "verified_inputs": {
            "protocol_sha256": cli.expected_protocol_sha,
            "analysis_sha256": cli.expected_analysis_sha,
            "manifest_sha256": sha256(manifest_path),
        },
        "population": {"scenes": 20, "episodes": 40, "arm_rows": 120},
        "arms": arms,
        "contrasts": contrasts,
        "deployability": {
            "raw_metric_sensor_safe_count": sum(
                not row["metric_depth_sensor_consumed_any"] for row in raw_rows),
            "raw_frame40_survived_count": len(reached_frame40),
            "raw_scale_valid_count_given_frame40": len(valid_scale),
            "raw_scale_clamped_count_given_frame40": sum(
                row["monocular_first40_scale_clamped"] is True
                for row in reached_frame40),
            "raw_active_contract_count_given_valid_scale": sum(
                row["active_plan_count"] > 0 for row in valid_scale),
        },
        "decision": {
            "engineering_checks": engineering_checks,
            "engineering_authorized_for_cec_on_monocular":
                engineering_authorized,
            "paper_noninferiority_10pp": paper_noninferiority,
            "frozen_disposition": (
                "continue_to_cec_on_monocular"
                if engineering_authorized else
                "stop_full_monocular_upgrade_keep_cec_rgbd_controller"),
        },
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
