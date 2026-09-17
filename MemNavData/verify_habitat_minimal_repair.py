#!/usr/bin/env python3
"""Independent reconstruction of bridge commands, trajectories and metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import habitat_sim
import cv2
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_depth_readout(row, contract):
    """Recompute pixel conversion from recorded producer tensor, independently."""
    receipt = row["monocular_depth_receipt"]
    raster = receipt["navdp_depth_raster"]
    assert row["audit_depth_raster_contract"] == raster["contract"] == contract
    path = Path(raster["artifact"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == raster["artifact_sha256"]
    with np.load(path, allow_pickle=False) as data:
        source, output = data["producer_depth"], data["navdp_readout_depth"]
        source_hw = list(data["source_rgb_hw"])
    assert hashlib.sha256(source.tobytes()).hexdigest() == raster["input_tensor_sha256"]
    assert hashlib.sha256(output.tobytes()).hexdigest() == raster["output_tensor_sha256"]
    assert list(source.shape) == raster["input_shape"]
    assert list(output.shape) == raster["output_shape"]
    assert source_hw == raster["source_rgb_hw"] == row["audit"]["image_calls"][0]["input_shape"][1:3]
    assert receipt["metric_depth_sensor_consumed"] is False
    height, width = map(int, source_hw)
    if contract == "source_rgb" and receipt["scale_state"] == "raw_lingbot_metric_depth":
        assert source.shape == (1, 518, 518, 1)
        scale = 518 / max(height, width)
        resized_h = 518 if height > width else round(height * scale / 14) * 14
        resized_w = 518 if width >= height else round(width * scale / 14) * 14
        y, x = (518-resized_h)//2, (518-resized_w)//2
        expected = cv2.resize(source[0, y:y+resized_h, x:x+resized_w, 0],
                              (width, height), interpolation=cv2.INTER_LINEAR)[None, :, :, None]
        assert raster["operation"] == "inverse_lingbot_pad_to_source_rgb"
        np.testing.assert_array_equal(expected, output)
    else:
        np.testing.assert_array_equal(source, output)
    if contract == "source_rgb":
        assert output.shape == (1, height, width, 1)
    return {"artifact": str(path), "contract": contract, "producer_shape": list(source.shape),
            "navdp_readout_shape": list(output.shape), "source_and_conversion_verified": True}


def verify_front_goal(folder, payload, actions, plans, http, enabled, *, memory_trace=None):
    """Check executed turns and causal observations without using adapter code."""
    receipt = json.loads((folder / "front_goal_receipt.json").read_text())
    assert receipt["enabled"] is enabled
    turn_actions = [a for a in actions if a["action_kind"] == "heading"]
    assert len(turn_actions) == receipt["actions"]
    assert enabled or not turn_actions
    events = receipt["events"]
    starts = [e for e in events if e["event"] == "rearward_point_request"]
    completions = {e["action_index"]: e for e in events if e["event"] == "heading_complete"}
    pose_rows = json.loads((folder / "lingbot_frame_poses.json").read_text())
    assert [r["frame_idx"] for r in pose_rows] == list(range(len(pose_rows)))
    assert all(r["motion_receipt_recorded"] is None for r in pose_rows)
    memory = payload["memory_traces"]["query"] if memory_trace is None else memory_trace
    assert len(memory) == len(actions)
    assert [m["step"] for m in memory] == list(range(len(actions)))
    indices = [m["frame_idx"] for m in memory]
    assert indices == list(range(indices[0], indices[0] + len(actions)))
    assert len(pose_rows) == indices[-1] + 1
    boundary = read_rows(folder / "memory_http_boundary.jsonl")
    forbidden = {"executed_translation_m", "executed_yaw_rad", "executed_forward_m",
                 "executed_left_m", "executor_local_se2_source", "executor_local_se2_contract"}
    for row in boundary:
        assert not (forbidden & set(row["sent_fields"]))
        assert set(row["original_fields"]) - set(row["sent_fields"]) == set(row["removed_fields"])
        assert not (set(row["removed_fields"]) - forbidden)
    query_http = [r for r in http if r["query_active"]]
    decisions = {p["next_action_index"]: p for p in plans}
    assert len(decisions) == len(plans)
    for p in plans:
        t = p["next_action_index"]
        assert p["receipt"]["memory_frame_idx"] == indices[t]
        actual_image = p["receipt"]["execution_input_audit"]["image_jpeg_sha256"]
        assert actual_image == pose_rows[indices[t]]["image_sha256"]
    # FIFO accepts each nominal decision observation once. A completed turn
    # adds a fresh post-turn decision, even when it is between nominal ticks.
    replay_ticks = [r["next_action_index"] for r in query_http if r["path"] == "/memory_replay_step"]
    expected_replay = [a["action_index"] for a in turn_actions
                       if a["action_index"] % 8 == 0 and a["action_index"] not in decisions]
    assert replay_ticks == expected_replay, (replay_ticks, expected_replay)
    for row in query_http:
        if row["path"] == "/memory_replay_step":
            t = row["next_action_index"]
            assert row["audit"]["image_jpeg_sha256"] == pose_rows[indices[t]]["image_sha256"]
    accepted = [e for e in starts if e["activated"]]
    expected_turn_indices, effects = [], []
    for e in starts:
        start = e["action_index"]
        plan = decisions[start]
        point = np.asarray(plan["receipt"]["memory_controller_pointgoal"])
        assert plan["receipt"]["revisit_adapter_takeover"] is True and point[0] < 0
        np.testing.assert_array_equal(point, e["pointgoal"])
        assert bool(e["activated"]) is enabled
        assert plan["pointgoal_alignment_started"] is enabled
        if not enabled:
            continue
        delta = math.atan2(point[1], point[0])
        target = plan["yaw"] + delta
        assert abs(target - e["target_yaw_rad"]) < 1e-10
        count = int(math.ceil((abs(delta) - 1e-10) / math.radians(4.5)))
        end = min(len(actions), start + count)
        expected_turn_indices.extend(range(start, end))
        for t in range(start, end):
            assert actions[t]["action_kind"] == "heading"
            assert abs(actions[t]["heading_target_yaw"] - target) < 1e-10
            assert t == start or t not in decisions  # no diffusion during turn
        if start + count > len(actions):
            assert receipt["active"]
            continue
        assert end - 1 in completions
        if end == len(actions):
            continue
        assert end in decisions, "the final new view must be replanned immediately"
        post = decisions[end]
        before_pose = np.asarray(pose_rows[indices[start]]["camera_pose9"])
        after_pose = np.asarray(pose_rows[indices[end]]["camera_pose9"])
        def rotation(q):
            x, y, z, w = q / np.linalg.norm(q)
            return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        relative = rotation(before_pose[3:7]).T @ rotation(after_pose[3:7])
        estimated_turn = math.atan2(-relative[0, 2], relative[2, 2])
        actual_turn = post["yaw"] - plan["yaw"]
        scale = plan["receipt"]["monocular_depth_receipt"]["scale_receipt"]["scale_hat"]
        disagreement = math.atan2(math.sin(estimated_turn-actual_turn), math.cos(estimated_turn-actual_turn))
        effects.append({"start_action": start, "post_turn_action": end,
                        "executed_turn_deg": math.degrees(actual_turn),
                        "lingbot_relative_yaw_deg": math.degrees(estimated_turn),
                        "lingbot_yaw_disagreement_deg": abs(math.degrees(disagreement)),
                        "lingbot_translation_times_frozen_scale_m": float(np.linalg.norm(after_pose[:3]-before_pose[:3])*scale),
                        "actual_translation_m": float(np.linalg.norm(np.asarray(post["position"])-plan["position"])),
                        "pointgoal_before": point.tolist(),
                        "pointgoal_after": post["receipt"]["memory_controller_pointgoal"],
                        "anchor_before": plan["receipt"]["anchor"], "anchor_after": post["receipt"]["anchor"],
                        "post_turn_selected_xy_zero": bool(np.max(np.abs(np.asarray(post["selected_trajectory"])[:, :2])) < 1e-9)})
    assert expected_turn_indices == [a["action_index"] for a in turn_actions]
    mandatory = set(range(0, len(actions), 8)) - set(expected_replay)
    mandatory |= {t + 1 for t in completions if t + 1 < len(actions)}
    # A nominal tick inside an active turn uses replay, not a plan.
    assert set(decisions) == mandatory
    return {"enabled": enabled, "rearward_requests": len(starts), "turn_sequences": len(accepted),
            "turn_actions": len(turn_actions), "causal_appends": len(memory),
            "replay_actions": replay_ticks, "turn_effects": effects,
            "pose_scale_caveat": "estimated height scale, not metric ground truth"}


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed"] and not summary["changed_source_files"]
    assert len(summary["results"]) == len(manifest["histories"]) * len(manifest["variants"])
    # Check archived program bytes, not a moving worktree after the run.
    project = Path(__file__).resolve().parents[1]
    for filename, digest in manifest["code_and_weights"].items():
        path = Path(filename)
        if path.suffix == ".py":
            saved = root / "source_snapshot" / path.relative_to(project)
            assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest, str(saved)
    details, indexed, depth_readouts = [], {}, []
    for result in summary["results"]:
        key = (result["scene"], result["policy"], result["executor"], result["color"])
        if result.get("depth_raster"):
            key += (result["depth_raster"],)
        if result.get("front_goal"):
            key += (result["front_goal"],)
        assert key not in indexed
        folder = root / "evaluation" / key[0] / "__".join(key[1:])
        payload_files = list(folder.glob("*_plans.json"))
        assert len(payload_files) == 1
        payload = json.loads(payload_files[0].read_text())
        actions = read_rows(folder / "executor_actions.jsonl")
        plans = read_rows(folder / "full_plan_outputs.jsonl")
        trace = payload["rollout_traces"]["query"]
        end = payload["query_result"]
        assert len(actions) == len(trace) == result["steps"]
        assert len(actions) <= manifest["max_steps"]
        assert [r["step"] for r in trace] == list(range(result["steps"]))
        points = np.array([[p[k] for k in ("x", "y", "z")] for p in trace] + [end["end_position"]])
        np.testing.assert_allclose(points[1:], [a["actual_position"] for a in actions], atol=1e-9, rtol=0)
        np.testing.assert_allclose(points[:-1], [a["position_before"] for a in actions], atol=1e-9, rtol=0)
        path_m = math.fsum(math.hypot(b[0]-a[0], b[2]-a[2]) for a, b in zip(points, points[1:]))
        final_dist = float(np.linalg.norm(points[-1, [0, 2]] - result["goal_xz_evaluator_only"]))
        success = int(final_dist < 1.)
        geo = result["geodesic_m"]
        spl = success * geo / max(geo, path_m) if max(geo, path_m) else float(success)
        assert abs(path_m - result["actual_path_len_m"]) < 1e-8
        assert abs(final_dist - result["final_goal_dist_m"]) < 1e-8
        assert success == result["reached"] == int(end["reached"])
        assert abs(spl - result["spl"]) < 1e-9
        pf = habitat_sim.PathFinder()
        assert pf.load_nav_mesh(str(folder / "execution.navmesh"))
        for index, action in enumerate(actions):
            assert action["action_index"] == index
            plan_index = action["plan_index"]
            if result.get("front_goal"):
                assert plan_index == max(p["plan_index"] for p in plans if p["next_action_index"] <= index)
            else:
                assert plan_index == index // manifest["exec_horizon"]
            plan = plans[plan_index]
            if not result.get("front_goal"):
                assert plan["next_action_index"] == plan_index * manifest["exec_horizon"]
            if action.get("action_kind") == "heading":
                assert result["front_goal"] == "heading_on" and result["executor"] == "bounded_standard"
                assert action["world_path"] is None and action["reference_executed"] is False
                delta = action["heading_target_yaw"] - action["yaw_before"]
                delta = math.atan2(math.sin(delta), math.cos(delta))
                expected_yaw = action["yaw_before"] + max(-math.radians(4.5), min(math.radians(4.5), delta))
                np.testing.assert_array_equal(action["position_before"], action["actual_position"])
                np.testing.assert_array_equal(action["position_before"], action["bounded_request"]["position"])
                assert action["actual_translation_m"] == action["bounded_request"]["displacement_m"] == 0
                assert abs(expected_yaw - action["actual_yaw"]) < 1e-10
                continue
            local = np.asarray(plan["selected_trajectory"])
            origin, yaw = np.asarray(plan["position"]), plan["yaw"]
            world = np.column_stack((origin[0] - local[:, 0]*math.sin(yaw) - local[:, 1]*math.cos(yaw),
                                     origin[2] - local[:, 0]*math.cos(yaw) + local[:, 1]*math.sin(yaw)))
            np.testing.assert_allclose(world, action["world_path"], atol=1e-9, rtol=0)
            p, yaw = np.asarray(action["position_before"]), action["yaw_before"]
            distances = np.linalg.norm(world - p[[0, 2]], axis=1)
            indices = [i for i, d in enumerate(distances) if d >= .7]
            target_index = indices[0] if indices else len(world) - 1
            d = float(distances[target_index])
            if result["executor"] == "bounded_standard" and d <= 1e-8:
                expected, expected_yaw = p, yaw
                assert action["bounded_request"]["displacement_m"] == 0
            else:
                delta = world[target_index] - p[[0, 2]]
                alpha = (math.atan2(-delta[0], -delta[1]) - yaw + math.pi) % (2*math.pi) - math.pi
                distance = .0376 * (.48 + .52 * (1 + math.cos(alpha)) / 2)
                if result["executor"] == "bounded_standard":
                    distance = min(distance, d)
                turn = max(-math.radians(4.5), min(math.radians(4.5),
                           max(-2.5, min(2.5, 2*alpha/.7))*distance))
                expected_yaw = yaw + turn
                forward = np.array([-math.sin(expected_yaw), 0, -math.cos(expected_yaw)])
                requested = p + distance*forward
                if result["executor"] == "bounded_standard":
                    np.testing.assert_allclose(requested, action["bounded_request"]["position"], atol=1e-10, rtol=0)
                    expected = np.asarray(pf.try_step(p, requested))
                else:
                    expected = np.asarray(pf.snap_point(requested))
                    if not np.isfinite(expected).all() or np.linalg.norm((expected-requested)[[0, 2]]) > .06:
                        short = p + .3*distance*forward
                        expected = np.asarray(pf.snap_point(short))
                        if not np.isfinite(expected).all() or np.linalg.norm((expected-short)[[0, 2]]) > .06:
                            expected = p
            np.testing.assert_allclose(expected, action["actual_position"], atol=1e-7, rtol=0)
            assert abs(expected_yaw - action["actual_yaw"]) < 1e-10
            actual_dl = float(np.linalg.norm((expected-p)[[0, 2]]))
            assert abs(actual_dl - action["actual_translation_m"]) < 1e-7
        http = read_rows(folder / "navdp_http_receipts.jsonl")
        for row in http:
            assert row["audit"]["contract"] == result["color"]
            assert row["audit"]["status_code"] == 200
            assert row["metric_depth_sensor_consumed"] is not True
        step_http = [r for r in http if r["path"] in ("/imagegoal_step", "/ipgoal_step")]
        # The exact active mixed endpoint is also identified by encoded image calls.
        predictions = [r for r in http if r["path"] not in ("/navigator_reset", "/memory_replay_step", "/navigator_reset_env")
                       and r["audit"]["image_calls"]]
        assert len(predictions) == len(plans), (folder, [r["path"] for r in predictions])
        if not result.get("front_goal"):
            assert len(plans) == math.ceil(len(actions)/8)
        if result.get("depth_raster"):
            for row in http:
                assert row["audit_depth_raster_contract"] == result["depth_raster"]
            depth_readouts.extend(verify_depth_readout(row, result["depth_raster"]) for row in predictions)
        indexed[key] = (payload, plans, actions, result)
        heading_audit = (verify_front_goal(folder, payload, actions, plans, http, result["front_goal"] == "heading_on")
                         if result.get("front_goal") else None)
        details.append({"scene":key[0], "policy":key[1], "executor":key[2], "color":key[3],
                        "reached":success, "actual_path_m":path_m, "spl":spl, "steps":len(actions),
                        "depth_raster":result.get("depth_raster"),
                        "front_goal":result.get("front_goal"), "heading_audit": heading_audit,
                        **result["execution"]})
    pairs = []
    for h in manifest["histories"]:
        if manifest.get("study") == "native_request":
            native = indexed[(h["scene"], "native", "bounded_standard", "rgb_v1", "source_rgb", "heading_on")]
            cec = indexed[(h["scene"], "cec", "bounded_standard", "rgb_v1", "source_rgb", "heading_on")]
            assert native[0]["replay"] == cec[0]["replay"]
            assert native[0]["rollout_traces"]["legA"] == cec[0]["rollout_traces"]["legA"]
            assert native[3]["first_query_rgb_sha256"] == cec[3]["first_query_rgb_sha256"]
            accepted = sum(p["receipt"].get("revisit_adapter_takeover") is True for p in cec[1])
            if accepted == 0:
                assert len(native[1]) == len(cec[1]) and len(native[2]) == len(cec[2])
                for a, b in zip(native[1], cec[1]):
                    for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                        np.testing.assert_array_equal(a[field], b[field])
                    for field in ("image_jpeg_sha256", "image_calls"):
                        assert a["receipt"]["execution_input_audit"][field] == b["receipt"]["execution_input_audit"][field]
                for a, b in zip(native[2], cec[2]):
                    for field in ("actual_position", "actual_yaw", "actual_translation_m", "action_kind"):
                        assert a[field] == b[field]
            pairs.append({"scene": h["scene"], "query_role_evaluator_only": "novel",
                          "cec_takeover_plans": accepted,
                          "exact_native_prefix_verified": accepted == 0,
                          "steps": len(cec[2]), "scope": "prefix equivalence, not complete Novel SR"})
            continue
        for policy in (("raw_fixed", "cec") if manifest.get("study") == "front_goal" else ("native", "cec")):
            if manifest.get("study") == "front_goal":
                old = indexed[(h["scene"], policy, "bounded_standard", "rgb_v1", "source_rgb", "heading_off")]
                new = indexed[(h["scene"], policy, "bounded_standard", "rgb_v1", "source_rgb", "heading_on")]
                assert old[0]["replay"] == new[0]["replay"]
                assert old[0]["rollout_traces"]["legA"] == new[0]["rollout_traces"]["legA"]
                assert old[3]["first_query_rgb_sha256"] == new[3]["first_query_rgb_sha256"]
                for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                    np.testing.assert_allclose(old[1][0][field], new[1][0][field], rtol=0, atol=1e-6)
                pairs.append({"scene": h["scene"], "policy": policy, "first_state_input_output_paired": True,
                              "heading_off_reached": old[3]["reached"], "heading_on_reached": new[3]["reached"]})
                continue
            if manifest.get("study") == "depth_raster":
                old = indexed[(h["scene"], policy, "bounded_standard", "rgb_v1", "legacy_square")]
                new = indexed[(h["scene"], policy, "bounded_standard", "rgb_v1", "source_rgb")]
                assert old[0]["replay"] == new[0]["replay"]
                assert old[0]["rollout_traces"]["legA"] == new[0]["rollout_traces"]["legA"]
                assert old[3]["first_query_rgb_sha256"] == new[3]["first_query_rgb_sha256"]
                pairs.append({"scene":h["scene"], "policy":policy, "shared_history_and_first_view":True,
                              "legacy_square_reached":old[3]["reached"], "source_rgb_reached":new[3]["reached"],
                              "first_plan_equality_required":False})
                continue
            old = indexed[(h["scene"], policy, "legacy_snap", "legacy_bgr")]
            new = indexed[(h["scene"], policy, "bounded_standard", "legacy_bgr")]
            rgb = indexed[(h["scene"], policy, "bounded_standard", "rgb_v1")]
            for other in (new, rgb):
                assert old[0]["replay"] == other[0]["replay"]
                assert old[0]["rollout_traces"]["legA"] == other[0]["rollout_traces"]["legA"]
                assert old[3]["first_query_rgb_sha256"] == other[3]["first_query_rgb_sha256"]
            first_error = float(np.max(np.abs(np.asarray(old[1][0]["selected_trajectory"]) - new[1][0]["selected_trajectory"])))
            assert first_error < 1e-6, (h["scene"], policy, first_error)
            pairs.append({"scene":h["scene"], "policy":policy, "shared_history_and_first_view":True,
                          "same_color_first_plan_max_error":first_error,
                          "legacy_reached":old[3]["reached"], "bounded_reached":new[3]["reached"],
                          "bounded_rgb_reached":rgb[3]["reached"]})
    report = dict(verified=True, records=details, pairs=pairs, depth_readouts=depth_readouts,
                  scope="consumed diagnostic; no population equivalence or autonomous STOP claim",
                  source_policy="archived .py hashes verified; current worktree need not be unchanged later")
    (root / "independent_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root.resolve())
