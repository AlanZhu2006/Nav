#!/usr/bin/env python3
"""Independently measure actual mono-A and every resulting mixed-role query."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np

from MemNavData.verify_habitat_minimal_repair import read_rows, verify_depth_readout, verify_front_goal


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion(position, yaw, world, pathfinder):
    """Reconstruct the frozen bounded steering law, without importing tracker."""
    p = np.asarray(position, dtype=float)
    distance = np.linalg.norm(world - p[[0, 2]], axis=1)
    eligible = np.flatnonzero(distance >= .7)
    k = int(eligible[0]) if len(eligible) else len(distance) - 1
    d = float(distance[k])
    if d <= 1e-8:
        return p.copy(), float(yaw), 0.
    delta = world[k] - p[[0, 2]]
    alpha = (math.atan2(-delta[0], -delta[1]) - yaw + math.pi) % (2*math.pi) - math.pi
    travel = min(.0376 * (.48 + .52 * (1 + math.cos(alpha)) / 2), d)
    turn = max(-math.radians(4.5), min(math.radians(4.5),
               max(-2.5, min(2.5, 2 * alpha / .7)) * travel))
    yaw2 = yaw + turn
    requested = p + travel * np.array([-math.sin(yaw2), 0, -math.cos(yaw2)])
    return np.asarray(pathfinder.try_step(p, requested)), yaw2, travel


def verify_rollout(row, *, execution_pathfinder=None, success_radius_m=1.0):
    folder = Path(row["directory"])
    evidence = json.loads((folder / "rollout_evidence.json").read_text())
    actions = read_rows(folder / "executor_actions.jsonl")
    plans = read_rows(folder / "full_plan_outputs.jsonl")
    http = read_rows(folder / "navdp_http_receipts.jsonl")
    trace = evidence["rollout_trace"]
    assert len(actions) == len(trace) == row["steps"] <= 600
    assert [p["step"] for p in trace] == list(range(len(actions)))
    points = np.array([[p[k] for k in ("x", "y", "z")] for p in trace] + [row["end_position"]])
    np.testing.assert_allclose(points[:-1], [a["position_before"] for a in actions], rtol=0, atol=1e-9)
    np.testing.assert_allclose(points[1:], [a["actual_position"] for a in actions], rtol=0, atol=1e-9)
    path = math.fsum(math.hypot(b[0]-a[0], b[2]-a[2]) for a, b in zip(points, points[1:]))
    final_dist = float(np.linalg.norm(points[-1, [0, 2]] - row["goal_xz_evaluator_only"]))
    assert math.isfinite(success_radius_m) and success_radius_m > 0
    reached = int(final_dist < success_radius_m)
    value = reached * row["geodesic_m"] / max(row["geodesic_m"], path)
    assert reached == row["reached"] == evidence["terminal"]["reached"]
    assert abs(final_dist-row["final_goal_dist_m"]) < 1e-8
    assert abs(path-row["actual_path_len_m"]) < 1e-8 and abs(value-row["spl"]) < 1e-9
    # A freshly rebuilt Habitat mesh and its serialized/reloaded copy can
    # disagree on connected islands after zero-area polygons are disabled.
    # An audit may supply the original construction context after proving
    # its serialized mesh is byte-identical. The controller is not changed.
    pf = execution_pathfinder
    if pf is None:
        pf = habitat_sim.PathFinder()
        assert pf.load_nav_mesh(str(folder / "execution.navmesh"))
    for index, action in enumerate(actions):
        assert action["selected_mode"] == "bounded_standard" and action["action_index"] == index
        assert action["plan_index"] == max(p["plan_index"] for p in plans if p["next_action_index"] <= index)
        p, yaw = np.asarray(action["position_before"]), action["yaw_before"]
        if action["action_kind"] == "heading":
            assert action["reference_executed"] is False and action["world_path"] is None
            delta = action["heading_target_yaw"] - yaw
            delta = math.atan2(math.sin(delta), math.cos(delta))
            expected = p
            yaw2 = yaw + max(-math.radians(4.5), min(math.radians(4.5), delta))
            travel = 0.
        else:
            plan = plans[action["plan_index"]]
            local = np.asarray(plan["selected_trajectory"])
            origin, a = np.asarray(plan["position"]), plan["yaw"]
            world = np.column_stack((origin[0]-local[:, 0]*math.sin(a)-local[:, 1]*math.cos(a),
                                     origin[2]-local[:, 0]*math.cos(a)+local[:, 1]*math.sin(a)))
            np.testing.assert_allclose(world, action["world_path"], rtol=0, atol=1e-9)
            expected, yaw2, travel = motion(p, yaw, world, pf)
        np.testing.assert_allclose(expected, action["actual_position"], rtol=0, atol=1e-7)
        assert abs(yaw2-action["actual_yaw"]) < 1e-10
        assert abs(travel-action["bounded_request"]["displacement_m"]) < 1e-9
        assert abs(np.linalg.norm((expected-p)[[0, 2]])-action["actual_translation_m"]) < 1e-7
    predictions = [r for r in http if r["query_active"] and r["path"] != "/memory_replay_step"
                   and r["audit"]["image_calls"]]
    assert len(predictions) == len(plans)
    depths = [verify_depth_readout(r, "source_rgb") for r in predictions]
    assert all(r["audit"]["contract"] == "rgb_v1" and r["audit"]["status_code"] == 200 for r in http)
    assert all(r["metric_depth_sensor_consumed"] is not True for r in http)
    turn = verify_front_goal(folder, {}, actions, plans, http, True, memory_trace=evidence["memory_trace"])
    if row["role"] == "goal_a" or row["arm"] == "native":
        assert turn["turn_actions"] == 0
        assert not any(p["receipt"]["revisit_adapter_takeover"] is True for p in plans)
    return {"scene": row["scene"], "role": row["role"], "arm": row["arm"],
            "steps": len(actions), "reached": reached, "actual_path_m": path, "spl": value,
            "depth_arrays_verified": len(depths), "heading": turn}, evidence, plans, actions


def constructed_histories(report, trace_path, trace):
    """Check collection attrition before invoking the nonempty query auditor."""
    from MemNavData.audit_shared_online_role_pairs import audit as audit_benchmark
    benchmark = Path(report["benchmark"])
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert (benchmark / "manifest.json.sha256").read_text().split()[0] == digest(manifest_path)
    materialization, construction = report["materialization"], report["construction"]
    assert materialization["source_traces"] == 1
    assert materialization["goal_a_successes"] == int(trace["reached"])
    assert manifest["source_online_manifest_sha256"] == materialization["manifest_sha256"]
    assert digest(Path(manifest["source_online_root"]) / "manifest.json") == materialization["manifest_sha256"]
    histories = manifest["episodes"]
    assert len(histories) == construction["retained_standard_natural_histories"]
    if not trace["reached"]:
        assert not histories and materialization["materialized"] == 0
        assert any(r["reason"] == "mono_a_failed" and r["trace_sha256"] == digest(trace_path)
                   for r in materialization["attrition"])
    if histories:
        assert audit_benchmark(benchmark)["ok"]
    else:
        assert not any(a["retained"] for a in construction["attempts"])
        if materialization["materialized"] == 0:
            assert materialization["attrition"]
        else:
            assert len(construction["attempts"]) == materialization["materialized"]
    return histories


def verify(root):
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    from MemNavData.hm3d_fullmono_mixed_role import audit_goal_a_plans
    from MemNavData.materialize_online_a_traces import native_control_audit
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    construction = json.loads((root / "construction_summary.json").read_text())
    assert summary["completed"] and not summary["changed_source_files"]
    assert len(summary["goal_a"]) == len(manifest["sources"]) == 2
    project = Path(__file__).resolve().parents[1]
    central = manifest.get("source_snapshot_mode") == "immutable_bundle_reference"
    if central:
        receipt = manifest["immutable_bundle_receipt"]
        assert digest(receipt["path"]) == receipt["sha256"]
        import subprocess
        subprocess.run(["sha256sum", "-c", "--quiet", receipt["path"]], cwd=project, check=True)
    for filename, expected in manifest["code_and_weights"].items():
        p = Path(filename)
        if p.suffix in (".py", ".md"):
            assert digest(p if central else root / "source_snapshot" / p.relative_to(project)) == expected
    if "source_manifest" in manifest:
        receipt = manifest["source_manifest"]
        assert digest(receipt["path"]) == receipt["sha256"]
    verified, a_traces, indexed = [], {}, {}
    for source, row in zip(manifest["sources"], summary["goal_a"]):
        assert row["scene"] == source["scene"] and row["episode"] == source["episode"]
        for path, expected in source["source_files"].items():
            assert digest(path) == expected
        trace_path = Path(row["directory"]) / f"{row['episode']}_leg1_trace.json"
        trace = json.loads(trace_path.read_text())
        validate_leg1_trace(trace)
        assert trace["source_hybrid_route"] == "native_sidecar" and trace["episode_seed"] == source["seed"]
        assert native_control_audit(trace)["ok"]
        audit_goal_a_plans(trace["plans"])
        result, evidence, plans, actions = verify_rollout(row)
        assert trace["poses"] == evidence["rollout_trace"]
        assert trace["end_position"] == row["end_position"]
        a_traces[row["scene"]] = (trace_path, trace)
        verified.append(result)
    expected_keys = set()
    for report in construction["reports"]:
        source_index = next(i for i, source in enumerate(manifest["sources"])
                            if source["scene"] == report["scene"])
        source = manifest["sources"][source_index]
        assert report["construction"]["scene_rank"] == int(source.get("source_scene_rank", source_index))
        trace_path, trace = a_traces[report["scene"]]
        history = constructed_histories(report, trace_path, trace)
        for h in history:
            path, trace = a_traces[h["scene"]]
            assert h["online_a_trace_sha256"] == digest(path)
            assert trace["reached"]
            assert digest(Path(h["online_a_episode"]) / "online_a_trace.json") == digest(path)
            expected_keys.update((h["scene"], role, arm) for role in ("novel", "revisit") for arm in manifest["arms"])
    for row in summary["queries"]:
        key = (row["scene"], row["role"], row["arm"])
        assert key in expected_keys and key not in indexed
        result, evidence, plans, actions = verify_rollout(row)
        payload_files = list(Path(row["directory"]).glob("*_plans.json"))
        assert len(payload_files) == 1
        payload = json.loads(payload_files[0].read_text())
        trace_path, trace = a_traces[row["scene"]]
        assert payload["rollout_traces"]["legA"] == trace["poses"]
        assert payload["rollout_traces"]["query"] == evidence["rollout_trace"]
        assert evidence["rollout_trace"][0]["x"] == trace["end_position"][0]
        assert evidence["rollout_trace"][0]["z"] == trace["end_position"][2]
        assert evidence["rollout_trace"][0]["yaw"] == trace["end_yaw"]
        indexed[key] = (row, payload, evidence, plans, actions)
        verified.append(result)
    assert set(indexed) == expected_keys
    pairs = []
    for scene, role in sorted({key[:2] for key in expected_keys}):
        arms = {arm: indexed[(scene, role, arm)] for arm in manifest["arms"]}
        native, cec = arms["native"], arms["cec"]
        for data in arms.values():
            assert data[0]["first_query_rgb_sha256"] == native[0]["first_query_rgb_sha256"]
            assert data[1]["replay"] == native[1]["replay"]
            assert data[0]["goal_xz_evaluator_only"] == native[0]["goal_xz_evaluator_only"]
        takeover = sum(p["receipt"]["revisit_adapter_takeover"] is True for p in cec[3])
        if not takeover:
            assert len(native[3]) == len(cec[3]) and len(native[4]) == len(cec[4])
            for a, b in zip(native[3], cec[3]):
                for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                    np.testing.assert_array_equal(a[field], b[field])
                for field in ("image_jpeg_sha256", "image_calls"):
                    assert a["receipt"]["execution_input_audit"][field] == b["receipt"]["execution_input_audit"][field]
            for a, b in zip(native[4], cec[4]):
                for field in ("actual_position", "actual_yaw", "action_kind"):
                    assert a[field] == b[field]
            depth_rows = []
            for data in (native, cec):
                receipts = read_rows(Path(data[0]["directory"]) / "navdp_http_receipts.jsonl")
                depth_rows.append([r["monocular_depth_receipt"] for r in receipts
                                   if r["query_active"] and r["path"] != "/memory_replay_step"
                                   and r["audit"]["image_calls"]])
            assert len(depth_rows[0]) == len(depth_rows[1]) == len(native[3])
            for a, b in zip(*depth_rows):
                for field in ("depth_png_sha256", "image_sha256", "scale_receipt_sha256", "frame_index"):
                    assert a[field] == b[field]
                for field in ("input_tensor_sha256", "output_tensor_sha256"):
                    assert a["navdp_depth_raster"][field] == b["navdp_depth_raster"][field]
        pairs.append({"scene": scene, "role": role, "shared_new_mono_a": True,
                      "success": {arm: data[0]["reached"] for arm, data in arms.items()},
                      "cec_takeover_plans": takeover, "no_takeover_exact_native": takeover == 0,
                      "no_takeover_depth_pairs_verified": len(native[3]) if not takeover else 0})
    result = {"verified": True, "scope": manifest["scope"],
              "goal_a_count": len(a_traces), "query_arm_count": len(indexed), "records": verified, "pairs": pairs,
              "verifier_sha256": digest(__file__),
              "postprocessing_note": "Empty collection populations checked separately from nonempty query manifests; navigation unchanged."}
    (root / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))


def render(root):
    from MemNavData.generate_twoleg import make_sim, render as render_frame
    from MemNavData.render_recorded_pose_first_person import Encoder, camera_position
    assert json.loads((root / "independent_verification.json").read_text())["verified"]
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    out = root / "first_person"
    out.mkdir(exist_ok=False)
    files = []
    for source in manifest["sources"]:
        sim = make_sim(source["asset"], "", agent_radius=.30)
        try:
            for row in summary["goal_a"] + summary["queries"]:
                if row["scene"] != source["scene"]:
                    continue
                evidence = json.loads((Path(row["directory"]) / "rollout_evidence.json").read_text())
                states = evidence["rollout_trace"] + [dict(zip(("x", "y", "z"), row["end_position"]), yaw=row["end_yaw_rad"])]
                first, _ = render_frame(sim, camera_position(states[0], .5), states[0]["yaw"])
                assert hashlib.sha256(first.tobytes()).hexdigest() == row["first_query_rgb_sha256"]
                target = out / f"{row['scene']}__{row['role']}__{row['arm']}.mp4"
                writer = Encoder(target, first.shape[1], first.shape[0], 10, "rgb24")
                try:
                    writer.write(first)
                    for state in states[1:]:
                        rgb, _ = render_frame(sim, camera_position(state, .5), state["yaw"])
                        writer.write(rgb)
                finally:
                    writer.close()
                files.append({"file": target.name, "sha256": digest(target), "frames": len(states),
                              "first_rgb_matches": True, "reached": row["reached"]})
        finally:
            sim.close()
    (out / "render_receipt.json").write_text(json.dumps({"outputs": files, "policy_rerun": False,
        "fps": 10, "playback": "control ticks; excludes GPU/HTTP wall time"}, indent=2) + "\n")
    print(f"Rendered {len(files)} complete success/failure rollouts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    (render if args.render else verify)(args.root.resolve())
