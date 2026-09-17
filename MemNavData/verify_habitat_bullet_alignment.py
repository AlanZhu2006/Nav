#!/usr/bin/env python3
"""Offline alignment/interface audit, retaining invalid physical prefixes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.certified_relocalization_runtime import _quat_xyzw_to_matrix
from MemNavData.habitat_bullet_alignment import wrap
from MemNavData.habitat_executor_audit import sha


def lines(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def read(path):
    return json.loads(path.read_text())


def audit(root):
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    assert manifest["alignment_pair"] and not summary["changed_sources"]
    poses_by_arm, current = [], []
    for row in lines(root / "lingbot_pose.jsonl"):
        if row["frame_idx"] == 0 and current:
            poses_by_arm.append(current)
            current = []
        current.append(row)
    if current:
        poses_by_arm.append(current)
    assert len(poses_by_arm) == 2
    first_plans, initial_states, results = [], [], []
    for index, (policy, tracker) in enumerate(manifest["variants"]):
        folder = root / "evaluation" / f"{policy}__{tracker}"
        plans, actions = lines(folder / "plans.jsonl"), lines(folder / "physical_actions.jsonl")
        initial_states.append(read(folder / "initialization.json")["settled"])
        first_plans.append(plans[0])
        zero_plans, masked, candidates = 0, 0, 0
        for plan in plans:
            receipt = plan["navdp_interface_diagnostic"]
            assert receipt is not None, "Missing pre-/post-mask receipt"
            before = np.asarray(receipt["pre_mask"], dtype=np.float32)
            after = np.asarray(receipt["post_mask"], dtype=np.float32)
            mask = np.linalg.norm(before[:, :, -1, :2], axis=-1) < .5
            expected = before.copy()
            expected[mask, :, :2] = 0
            assert np.array_equal(expected, after), "Not the frozen short-path mask"
            assert np.array_equal(after, np.asarray(plan["all_trajectory"], dtype=np.float32))
            if "pointgoal_before_clip" in receipt:
                point = np.asarray(receipt["pointgoal_before_clip"])
                clipped = np.clip(point, -10, 10)
                clipped[:, 0] = np.clip(clipped[:, 0], 0, 10)
                assert np.array_equal(clipped, receipt["pointgoal_after_clip"])
                assert np.allclose(point[0, :2], plan["memory_controller_pointgoal"], rtol=0, atol=1e-7)
            masked += int(mask.sum())
            candidates += mask.size
            zero_plans += int(np.max(np.abs(np.asarray(plan["selected_trajectory"])[:, :2])) < 1e-9)
        alignment = read(folder / "alignment_receipt.json")
        turns = [a for a in actions if a["command_kind"] == "physical_alignment"]
        assert len(turns) == alignment["actions"]
        assert all(a["command_speed_mps"] == 0 and a["mpc_solve_s"] is None for a in turns)
        assert all(a["navmesh_motion_queries"] == 0 for a in actions)
        assert len(actions) <= manifest["max_steps"]
        observed = lines(folder / "turn_observations.jsonl")
        if turns:
            indices = [a["action"] for a in turns]
            assert indices == list(range(indices[0], indices[-1]+1))
            assert len(observed) == len(turns)-1
            assert all(r["executor_motion_fields_sent"] == [] for r in observed)
            assert len({r["receipt"]["frame_idx"] for r in observed}) == len(observed)
            assert not any(indices[0] < p["next_action"] <= indices[-1] for p in plans)
        terminal_file = folder / "terminal_measurements.json"
        terminal = read(terminal_file)[0] if terminal_file.exists() else None
        physics = read(folder / "physics_summary.json")
        if terminal is None:
            assert physics["max_tilt_deg"] >= 5
        else:
            path = sum(a["actual_translation_m"] for a in actions)
            assert len(actions) == terminal["steps"]
            assert math.isclose(path, terminal["actual_path_len_m"], abs_tol=1e-7)
            endpoint = np.asarray(actions[-1]["after"]["position"])-[0, .75, 0]
            assert np.allclose(endpoint, terminal["end_position"], rtol=0, atol=1e-7)
            distance = np.linalg.norm(endpoint[[0, 2]]-terminal["goal_xz_evaluator_only"])
            assert math.isclose(distance, terminal["final_goal_dist_m"], abs_tol=1e-7)
            assert int(distance < 1) == terminal["reached"]
            runtime = read(folder / "summary.json")
            assert runtime["runtime_role_visibility"] == "none"
            assert runtime["metric_depth_sensor_consumed_episodes"] == 0
        effect = None
        post = next((p for p in plans if turns and p["next_action"] > turns[-1]["action"]), None)
        if post is not None:
            first = plans[0]
            by_sha = {r["image_sha256"]: r for r in poses_by_arm[index]}
            before_pose = np.asarray(by_sha[first["navdp_interface_diagnostic"]["image_sha256"]]["camera_pose9"])
            after_pose = np.asarray(by_sha[post["navdp_interface_diagnostic"]["image_sha256"]]["camera_pose9"])
            rotation = _quat_xyzw_to_matrix(before_pose[3:7]).T @ _quat_xyzw_to_matrix(after_pose[3:7])
            lingbot_turn = math.atan2(-rotation[0, 2], rotation[2, 2])
            actual_turn = wrap(post["yaw"]-first["yaw"])
            scale = first["monocular_depth_receipt"]["scale_receipt"]["scale_hat"]
            first_bearing = math.atan2(first["memory_controller_pointgoal"][1], first["memory_controller_pointgoal"][0])
            point_after = post["memory_controller_pointgoal"]
            effect = {
                "post_turn_plan": post["plan"], "fresh_plan_at_action": post["next_action"],
                "pointgoal_before": first["memory_controller_pointgoal"], "pointgoal_after": point_after,
                "actual_turn_deg": math.degrees(actual_turn), "lingbot_relative_yaw_deg": math.degrees(lingbot_turn),
                "lingbot_yaw_disagreement_deg": math.degrees(abs(wrap(lingbot_turn-actual_turn))),
                "actual_planar_displacement_m": float(np.linalg.norm((np.asarray(post["position"])-first["position"])[[0, 2]])),
                "lingbot_translation_norm_times_frozen_scale_m": float(np.linalg.norm(after_pose[:3]-before_pose[:3])*scale),
                "bearing_after_under_rotation_only_deg": math.degrees(wrap(first_bearing-actual_turn)),
                "bearing_after_relocalization_deg": None if point_after is None else math.degrees(math.atan2(point_after[1], point_after[0])),
                "anchor_before": first["anchor"], "anchor_after": post["anchor"],
                "post_turn_selected_xy_zero": bool(np.max(np.abs(np.asarray(post["selected_trajectory"])[:, :2])) < 1e-9),
                "translation_caveat": "height-derived scale is not ground truth; compare as diagnostic only",
            }
        results.append({"policy": policy, "status": "navigation_complete" if terminal else "invalid_physics",
                        "reached": None if terminal is None else terminal["reached"],
                        "actions": len(actions), "plans": len(plans), "zero_selected_plans": zero_plans,
                        "masked_candidates": masked, "candidate_count": candidates,
                        "turn_actions": len(turns), "max_tilt_deg": physics["max_tilt_deg"],
                        "terminal": terminal, "turn_effect": effect})
    assert initial_states[0] == initial_states[1]
    for key in ("selected_trajectory", "all_trajectory", "all_values", "memory_controller_pointgoal"):
        assert first_plans[0][key] == first_plans[1][key], f"First input/plan mismatch: {key}"
    return {"verified": True, "scene": manifest["scene"], "results": results,
            "initial_state_and_first_plan_paired": True,
            "scope": "posthoc consumed mechanism diagnostic; not formal SR"}


def audit_batch(root):
    complete = read(root / "summary.json")
    assert complete["all_histories_attempted"] and len(complete["results"]) == 4
    prior_root = ROOT / ".diagnostics/habitat_physics_executor_20260908"
    prior = [prior_root / "query_four_arm_v3",
             prior_root / "complete_arms_v2/history_01_pLe4wQe7qrG",
             prior_root / "remaining_three_v1/history_02_yqstnuAEVhm",
             prior_root / "complete_arms_v2/history_03_mJXqzFtmKg4"]
    reports, repeats, videos = [], [], []
    counts = {"cec": Counter(), "cec_aligned": Counter()}
    for index, item in enumerate(complete["results"]):
        run = Path(item["run"])
        report = audit(run)
        reports.append(report)
        manifest = read(run / "manifest.json")
        for name, digest in manifest["sources"].items():
            assert sha(run / "sources" / Path(name).relative_to(ROOT)) == digest
        repeat = {"scene": report["scene"]}
        for filename, keys in (("plans.jsonl", ("position", "yaw", "selected_trajectory", "all_trajectory", "all_values")),
                               ("physical_actions.jsonl", ("before", "after", "command_speed_mps", "command_yaw_rps", "substeps"))):
            old = lines(prior[index] / "evaluation/cec__mpc" / filename)
            new = lines(run / "evaluation/cec__mpc" / filename)
            repeat[filename] = (len(old) == len(new)
                                and all(all(a[k] == b[k] for k in keys) for a, b in zip(old, new)))
        repeats.append(repeat)
        for row in report["results"]:
            status = ("invalid_physics" if row["terminal"] is None
                      else "reached" if row["reached"] else "not_reached")
            counts[row["policy"]][status] += 1
        rendering = read(run / "first_person_v1/render_receipt.json")
        for filename, digest in rendering["outputs"].items():
            assert sha(run / "first_person_v1" / filename) == digest
        for video in rendering["results"]:
            assert video.get("initial_RGB_hash_matches", video.get("first_RGB_exactly_matches")) is True
            folder = run / "evaluation" / video["variant"]
            assert video["frames"] == len(lines(folder / "physical_actions.jsonl"))+1
        videos.append({"scene": report["scene"], "videos_verified": len(rendering["results"])})
    return {"verified": True, "histories": 4, "arms_attempted": 8,
            "outcome_counts": {k: dict(v) for k, v in counts.items()},
            "reports": reports, "baseline_replication_core_fields": repeats,
            "video_audit": videos,
            "scope": "consumed Bullet cylinder query-stage diagnostic; no formal SR inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root) if (args.root / "evaluation").exists() else audit_batch(args.root)
    with (args.root / "alignment_audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
