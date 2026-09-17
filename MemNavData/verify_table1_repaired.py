"""Independent SR/SPL, bounded-motion and selected-policy interface checks."""
import json
import math
from pathlib import Path
import sys

import habitat_sim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.verify_repaired_fullmono_local import motion, verify_rollout
from MemNavData.verify_habitat_minimal_repair import read_rows


def image_rollout(row):
    folder = Path(row["directory"])
    evidence = json.loads((folder / "rollout_evidence.json").read_text())
    actions, plans = read_rows(folder / "executor_actions.jsonl"), read_rows(folder / "full_plan_outputs.jsonl")
    http = read_rows(folder / "navdp_http_receipts.jsonl")
    requests = read_rows(folder / "image_controller_requests.jsonl")
    trace = evidence["rollout_trace"]
    assert len(actions) == len(trace) == row["steps"] <= 600
    points = np.array([[p[k] for k in ("x", "y", "z")] for p in trace] + [row["end_position"]])
    np.testing.assert_allclose(points[:-1], [a["position_before"] for a in actions], rtol=0, atol=1e-9)
    np.testing.assert_allclose(points[1:], [a["actual_position"] for a in actions], rtol=0, atol=1e-9)
    distance = float(np.linalg.norm(points[-1,[0,2]]-row["goal_xz_evaluator_only"]))
    path = math.fsum(math.hypot(b[0]-a[0], b[2]-a[2]) for a,b in zip(points, points[1:]))
    reached = int(distance < 1.)
    spl = reached * row["geodesic_m"] / max(row["geodesic_m"], path)
    assert reached == row["reached"] and abs(distance-row["final_goal_dist_m"]) < 1e-8
    assert abs(path-row["actual_path_len_m"]) < 1e-8 and abs(spl-row["spl"]) < 1e-9
    pf = habitat_sim.PathFinder()
    assert pf.load_nav_mesh(str(folder / "execution.navmesh"))
    for i, a in enumerate(actions):
        assert a["action_index"] == i and a["selected_mode"] == "bounded_standard"
        assert a["plan_index"] == max(p["plan_index"] for p in plans if p["next_action_index"] <= i)
        origin, yaw = np.asarray(a["position_before"]), a["yaw_before"]
        if a["action_kind"] == "heading":
            assert a["world_path"] is None and not a["reference_executed"]
            delta = math.atan2(math.sin(a["heading_target_yaw"]-yaw), math.cos(a["heading_target_yaw"]-yaw))
            expected, angle, travel = origin, yaw+max(-math.pi/40,min(math.pi/40,delta)), 0.
        else:
            p = plans[a["plan_index"]]
            local, ref, psi = np.asarray(p["selected_trajectory"]), p["position"], p["yaw"]
            world = np.column_stack((ref[0]-local[:,0]*math.sin(psi)-local[:,1]*math.cos(psi),
                                     ref[2]-local[:,0]*math.cos(psi)+local[:,1]*math.sin(psi)))
            np.testing.assert_allclose(world, a["world_path"], rtol=0, atol=1e-9)
            expected, angle, travel = motion(origin,yaw,world,pf)
        np.testing.assert_allclose(expected,a["actual_position"],rtol=0,atol=1e-7)
        assert abs(angle-a["actual_yaw"]) < 1e-10
        assert abs(travel-a["bounded_request"]["displacement_m"]) < 1e-9
    memory = evidence["memory_trace"]
    assert len(memory) == len(actions)
    ids = [m["frame_idx"] for m in memory]
    assert ids == list(range(ids[0],ids[0]+len(actions)))
    pose = json.loads((folder / "lingbot_frame_poses.json").read_text())
    assert [p["frame_idx"] for p in pose] == list(range(len(pose)))
    assert len(pose) == ids[-1]+1 and all(p["motion_receipt_recorded"] is None for p in pose)
    decisions = {p["next_action_index"]:p for p in plans}
    assert len(requests) == len(plans) == len(decisions)
    turns = [a["action_index"] for a in actions if a["action_kind"] == "heading"]
    replay = [r["next_action_index"] for r in http if r["query_active"] and r["path"] == "/memory_replay_step"]
    assert replay == [i for i in turns if i%8 == 0 and i not in decisions]
    assert set(range(0,len(actions),8)) - set(replay) <= set(decisions)
    heading = json.loads((folder / "front_goal_receipt.json").read_text())
    assert heading["enabled"] and heading["actions"] == len(turns)
    expected_turns = []
    for event in (e for e in heading["events"] if e["event"] == "rearward_point_request"):
        i = event["action_index"]
        p = decisions[i]
        point = np.asarray(p["receipt"]["memory_controller_pointgoal"])
        assert p["receipt"]["revisit_adapter_takeover"] is True and point[0] < 0
        np.testing.assert_array_equal(point,event["pointgoal"])
        count = int(math.ceil((abs(math.atan2(point[1],point[0]))-1e-10)/(math.pi/40)))
        end = min(len(actions),i+count)
        expected_turns.extend(range(i,end))
        assert not (set(range(i+1,end)) & set(decisions))
        if end < len(actions):
            assert end in decisions
    assert expected_turns == turns
    for p,r in zip(plans,requests):
        t = p["next_action_index"]
        assert r["controller"] == row["controller"] and r["controller_depth_source"] == "none"
        assert r["policy_pointgoal_consumed"] is False and r["distance_used_to_suppress_trajectory"] is False
        a = r["execution_input_audit"]
        assert a["received_file_fields"] == ["goal","image"] and a["received_form_fields"] == ["diffusion_seed"]
        assert a["contract"] == "rgb_v1" and a["image_jpeg_sha256"] == pose[ids[t]]["image_sha256"]
        assert a["goal_jpeg_sha256"] == r["controller_goal_sha256"]
        assert r["accepted"] == (p["receipt"]["revisit_adapter_takeover"] is True)
        if r["accepted"]:
            assert r["controller_goal_sha256"] == r["anchor_sha256"]
        else:
            assert r["controller_goal_sha256"] == r["original_goal_sha256"]
        if row["controller"] == "nomad":
            assert r["goal_mask"] == [0] and r["controller_seed_consumed"]
            assert r["diffusion_seed"] == p["receipt"]["diffusion_seed"]
    if row["arm"] == "native":
        assert not turns and not any(r["accepted"] for r in requests)
    return dict(scene=row["scene"],role=row["role"],arm=row["arm"],steps=len(actions),reached=reached,
                actual_path_m=path,spl=spl,controller=row["controller"],heading_actions=len(turns)), evidence,plans,actions


def verify(root):
    manifest=json.loads((root/"manifest.json").read_text())
    summary=json.loads((root/"summary.json").read_text())
    assert summary["completed"] and len(summary["records"]) == 4
    assert sha(manifest["plan"]) == manifest["plan_sha256"]
    assert all(sha(p)==digest for p,digest in manifest["source_hashes"].items())
    from MemNavData.table1_repaired_eval import history
    _,frozen,_=history(summary["cell"])
    verified,indexed=[],{}
    for row in summary["records"]:
        checked,evidence,plans,actions=(verify_rollout if row["controller"]=="navdp" else image_rollout)(row)
        folder=Path(row["directory"])
        data=json.loads((folder/"query_plans.json").read_text())
        assert data["replay"]["all_rgb_hashes_verified"] and data["replay"]["diffusion_samples_during_replay"]==0
        assert data["rollout_traces"]["legA"]==frozen["trace"]["poses"]
        np.testing.assert_array_equal(actions[0]["position_before"],frozen["trace"]["end_position"])
        assert actions[0]["yaw_before"]==frozen["trace"]["end_yaw"]
        for r in read_rows(folder/"memory_http_boundary.jsonl"):
            assert not set(r["sent_fields"]) & {"role","analysis_role","floor_position","gt_pose",
                "executed_translation_m","executed_yaw_rad","executed_forward_m","executed_left_m"}
        key=(row["role"],row["arm"])
        assert key not in indexed
        indexed[key]=(row,data,plans,actions)
        verified.append(checked)
    assert set(indexed)=={(r,a) for r in ("novel","revisit") for a in ("native","cec")}
    pairs=[]
    for role in ("novel","revisit"):
        native,gem=indexed[(role,"native")],indexed[(role,"cec")]
        for field in ("first_query_rgb_sha256","goal_xz_evaluator_only","geodesic_m"):
            assert native[0][field]==gem[0][field]
        assert native[1]["replay"]==gem[1]["replay"]
        assert native[1]["original_goal_sha256"]==gem[1]["original_goal_sha256"]
        takeover=any(p["receipt"]["revisit_adapter_takeover"] is True for p in gem[2])
        if not takeover:
            assert len(native[2])==len(gem[2]) and len(native[3])==len(gem[3])
            for a,b in zip(native[2],gem[2]):
                for field in ("selected_trajectory","all_trajectory","all_values","position","yaw"):
                    np.testing.assert_array_equal(a[field],b[field])
            for a,b in zip(native[3],gem[3]):
                np.testing.assert_array_equal(a["actual_position"],b["actual_position"])
                assert a["actual_yaw"]==b["actual_yaw"]
        pairs.append(dict(role=role,native=native[0]["reached"],cec=gem[0]["reached"],
                          cec_takeover=takeover,no_takeover_exact_native=not takeover))
    dump(root/"independent_verification.json",dict(verified=True,cell=summary["cell"],records=verified,pairs=pairs))
    print("VERIFIED Table-I controller-native pairing, exact executed SR/SPL",flush=True)


if __name__=="__main__":
    verify(Path(sys.argv[1]).resolve())
