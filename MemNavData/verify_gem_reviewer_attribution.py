"""Independent motion/input/authority checks for the reviewer control pilot."""
import argparse
import math
from pathlib import Path

import habitat_sim
import numpy as np

from MemNavData.gem_bearing_attribution import ROOT, dump, load, sha
from MemNavData.verify_repaired_fullmono_local import motion, verify_rollout
from MemNavData.verify_habitat_minimal_repair import read_rows, verify_depth_readout


def fixed_scan(row):
    """Reconstruct a constant positive scan without importing its implementation."""
    folder = Path(row["directory"])
    evidence = load(folder / "rollout_evidence.json")
    actions = read_rows(folder / "executor_actions.jsonl")
    plans = read_rows(folder / "full_plan_outputs.jsonl")
    http = read_rows(folder / "navdp_http_receipts.jsonl")
    trace = evidence["rollout_trace"]
    assert len(actions) == len(trace) == row["steps"] <= 600
    assert len(actions) > 40
    points = np.array([[p[k] for k in ("x", "y", "z")] for p in trace] + [row["end_position"]])
    np.testing.assert_allclose(points[:-1], [a["position_before"] for a in actions], rtol=0, atol=1e-9)
    np.testing.assert_allclose(points[1:], [a["actual_position"] for a in actions], rtol=0, atol=1e-9)
    path = math.fsum(math.hypot(b[0]-a[0], b[2]-a[2]) for a, b in zip(points, points[1:]))
    distance = float(np.linalg.norm(points[-1, [0, 2]] - row["goal_xz_evaluator_only"]))
    reached = int(distance < 1.)
    spl = reached * row["geodesic_m"] / max(row["geodesic_m"], path)
    assert reached == row["reached"] == evidence["terminal"]["reached"]
    assert abs(path-row["actual_path_len_m"]) < 1e-8
    assert abs(distance-row["final_goal_dist_m"]) < 1e-8 and abs(spl-row["spl"]) < 1e-9
    pf = habitat_sim.PathFinder()
    assert pf.load_nav_mesh(str(folder / "execution.navmesh"))
    for index, action in enumerate(actions):
        assert action["action_index"] == index and action["selected_mode"] == "bounded_standard"
        assert action["plan_index"] == max(p["plan_index"] for p in plans if p["next_action_index"] <= index)
        p, yaw = np.asarray(action["position_before"]), action["yaw_before"]
        if index < 40:
            assert action["action_kind"] == "heading" and not action["reference_executed"]
            assert action["world_path"] is None
            expected, yaw2, travel = p, yaw + math.pi/40, 0.
            assert abs(action["heading_target_yaw"] - plans[0]["yaw"] - math.pi) < 1e-9
        else:
            assert action["action_kind"] == "trajectory_tracking"
            plan = plans[action["plan_index"]]
            local = np.asarray(plan["selected_trajectory"])
            origin, a = np.asarray(plan["position"]), plan["yaw"]
            world = np.column_stack((origin[0]-local[:, 0]*math.sin(a)-local[:, 1]*math.cos(a),
                                     origin[2]-local[:, 0]*math.cos(a)+local[:, 1]*math.sin(a)))
            np.testing.assert_allclose(world, action["world_path"], rtol=0, atol=1e-9)
            expected, yaw2, travel = motion(p, yaw, world, pf)
        np.testing.assert_allclose(expected, action["actual_position"], rtol=0, atol=1e-7)
        assert abs(yaw2-action["actual_yaw"]) < 1e-9
        assert abs(travel-action["bounded_request"]["displacement_m"]) < 1e-9
        assert abs(np.linalg.norm((expected-p)[[0, 2]])-action["actual_translation_m"]) < 1e-7
    receipt = load(folder / "front_goal_receipt.json")
    assert receipt["enabled"] and not receipt["active"] and receipt["actions"] == 40
    assert [r["event"] for r in receipt["events"]] == ["fixed_initial_half_turn", "heading_complete"]
    assert receipt["events"][0]["goal_or_memory_information_used"] is False
    assert receipt["events"][1]["action_index"] == 39
    pose_rows = load(folder / "lingbot_frame_poses.json")
    assert [p["frame_idx"] for p in pose_rows] == list(range(len(pose_rows)))
    assert all(p["motion_receipt_recorded"] is None for p in pose_rows)
    memory = evidence["memory_trace"]
    assert len(memory) == len(actions)
    assert [m["step"] for m in memory] == list(range(len(actions)))
    ids = [m["frame_idx"] for m in memory]
    assert ids == list(range(ids[0], ids[0] + len(actions))) and len(pose_rows) == ids[-1]+1
    assert [p["next_action_index"] for p in plans] == [0] + list(range(40, len(actions), 8))
    query_http = [r for r in http if r["query_active"]]
    replay = [r for r in query_http if r["path"] == "/memory_replay_step"]
    assert [r["next_action_index"] for r in replay] == [8, 16, 24, 32]
    for p in plans:
        t = p["next_action_index"]
        assert p["receipt"]["memory_frame_idx"] == ids[t]
        assert p["receipt"]["execution_input_audit"]["image_jpeg_sha256"] == pose_rows[ids[t]]["image_sha256"]
    for r in replay:
        assert r["audit"]["image_jpeg_sha256"] == pose_rows[ids[r["next_action_index"]]]["image_sha256"]
    predictions = [r for r in query_http if r["path"] == "/imagegoal_step"]
    assert len(predictions) == len(plans)
    for r in predictions:
        verify_depth_readout(r, "source_rgb")
    return dict(scene=row["scene"], role=row["role"], arm=row["arm"], steps=len(actions),
                reached=reached, actual_path_m=path, spl=spl, depth_arrays_verified=len(predictions),
                heading=dict(turn_actions=40, causal_appends=len(memory))), evidence, plans, actions


def verify(out, partial=False):
    plan, summary = load(out / "plan.json"), load(out / "summary.json")
    assert summary["plan_sha256"] == sha(out / "plan.json")
    assert sha(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    if not partial:
        assert summary["completed"] and len(summary["records"]) == plan["total_rollouts"] == 20
    for path, digest in plan["source_sha256"].items():
        assert sha(out / "source_snapshot" / Path(path).relative_to(ROOT)) == digest
    checked, indexed = [], {}
    for row in summary["records"]:
        audit, evidence, plans, actions = fixed_scan(row) if row["arm"] == "fixed_half_turn" else verify_rollout(row)
        folder = Path(row["directory"])
        decisions = read_rows(folder / "intervention_decisions.jsonl")
        boundary = read_rows(folder / "memory_http_boundary.jsonl")
        spec = next(s for s in plan["runs"] if (s["scene"],s["role"],s["arm"]) == (row["scene"],row["role"],row["arm"]))
        assert len(decisions) == len(plans)
        for decision, p in zip(decisions, plans):
            assert decision["goal_sha256"] == spec["goal_sha256"]
            assert decision["diffusion_seed_requested"] == decision["diffusion_seed_returned"]
            assert decision["image_sha256"] == p["receipt"]["execution_input_audit"]["image_jpeg_sha256"]
        forbidden = {"executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m",
                     "executor_local_se2_source", "executor_local_se2_contract"}
        assert not any(forbidden & set(r["sent_fields"]) for r in boundary)
        if row["arm"] in ("native", "fixed_half_turn"):
            assert all(not p["receipt"]["revisit_adapter_takeover"] and p["receipt"]["memory_controller_pointgoal"] is None for p in plans)
            assert not any(r["path"] in ("/retrieval_probe_step", "/certified_relocalize") for r in boundary)
            http = read_rows(folder / "navdp_http_receipts.jsonl")
            assert all(r["path"] in ("/imagegoal_step", "/memory_replay_step") for r in http if r["query_active"])
        else:
            assert audit["heading"]["turn_actions"] == 0
        key = row["scene"], row["role"], row["arm"]
        assert key not in indexed
        indexed[key] = row, plans, actions
        checked.append(audit)
    pairs = []
    for cell in plan["cells"]:
        for role in ("revisit", "novel"):
            keys = [(cell["scene"], role, a) for a in ("native", "fixed_half_turn")]
            if not all(k in indexed for k in keys):
                continue
            native, scan = [indexed[k] for k in keys]
            for field in ("first_query_rgb_sha256", "goal_xz_evaluator_only", "geodesic_m"):
                assert native[0][field] == scan[0][field]
            for field in ("position", "yaw", "selected_trajectory", "all_trajectory", "all_values"):
                np.testing.assert_array_equal(native[1][0][field], scan[1][0][field])
            for field in ("memory_frame_idx", "diffusion_seed"):
                assert native[1][0]["receipt"][field] == scan[1][0]["receipt"][field]
            pairs.append(dict(scene=cell["scene"], role=role, native=native[0]["reached"],
                              fixed_half_turn=scan[0]["reached"], initial_sample_identical=True))
    parent = Path(plan["parent_results"])
    assert sha(parent / "summary.json") == plan["parent_summary_sha256"]
    from MemNavData.verify_gem_bearing_attribution import non_timing
    for row in summary["records"]:
        if row["arm"] != "gem_no_turn":
            continue
        old = load(parent / "evaluation" / row["scene"] / "full_gem" / "query_plans.json")["query_leg"][0]
        new = load(Path(row["directory"]) / "query_plans.json")["query_leg"][0]
        for field in ("router_selected_anchor", "memory_controller_pointgoal", "certified_relocalization_accepted",
                      "certified_relocalization_pnp", "certified_relocalization_certificate"):
            assert non_timing(old[field]) == non_timing(new[field]), (row["scene"], field)
    result = dict(verified=True, complete=summary["completed"], rollouts_checked=len(checked),
                  pairs=pairs, checks=checked, verifier_sha256=sha(__file__))
    dump(out / ("partial_verification.json" if partial else "independent_verification.json"), result)
    print({k:v for k,v in result.items() if k != "checks"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    verify(args.out.resolve(), args.partial)
