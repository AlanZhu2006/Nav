"""Matched-support goal versus historical-camera navigation, evaluated to 0.3 m."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MemNavData")]
from MemNavData.gem_bearing_attribution import dump, load, sha

HERE = Path(__file__).resolve()
ARMS = ("pnp_goal", "historical_camera")


def first_passage(distances, radius):
    return next((index for index, value in enumerate(distances) if value < radius), None)


def check_plan(path):
    plan = load(path)
    assert plan["schema"] == "gem_position_readout_v1"
    assert plan["arms"] == list(ARMS) and len(plan["cells"]) == 7
    assert plan["total_rollouts"] == 14 and plan["success_radius_m"] == 0.3
    assert plan["memory_mode"] == "legacy" and plan["dense_window"] == 32
    assert plan["authority_policy"] == "strict_certificate" and plan["max_steps"] == 600
    assert [c["index"] for c in plan["cells"]] == list(range(7))
    assert len({(c["dataset"], c["scene"], c["episode"]) for c in plan["cells"]}) == 7
    assert all(c["controller"] == "navdp" and c["selection_offset_m"] >= 1.0 for c in plan["cells"])
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return plan


def command(cell, target, mem, nav):
    from MemNavData.table1_repaired_eval import command as original
    cmd = original(cell, "revisit", "cec", target, mem, nav)
    cmd[2] = str(HERE)
    assert cmd.count("--success_dist") == 1
    cmd[cmd.index("--success_dist") + 1] = "0.3"
    return cmd


def run(args):
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_gem_support_navigation import gpu_binding
    from MemNavData.run_repaired_fullmono_local import execution_environment, run_child
    from MemNavData.table1_repaired_eval import history
    plan = check_plan(args.plan)
    cell = plan["cells"][args.index]
    payload, frozen, folder = history(cell)
    query = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")
    assert query["goal_rgb_sha256"] == cell["goal_rgb_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    inputs = [folder / "role_pairs.json", folder / query["goal_rgb"],
              frozen["source"] / "receipt.json", frozen["source"] / "online_a_trace.json"]
    order = list(ARMS if args.index % 2 == 0 else reversed(ARMS))
    dump(args.out / "manifest.json", dict(cell=cell, arms_in_order=order, plan_sha256=sha(args.plan),
         input_sha256={str(p): sha(p) for p in inputs}, source_sha256=plan["source_sha256"]))
    env = dict(execution_environment(), TABLE1_PLAN=str(args.plan), TABLE1_INDEX=str(args.index))
    key = hashlib.md5((folder / query["goal_rgb"]).read_bytes()).hexdigest()
    summary = dict(completed=False, cell=cell, records=[], plan_sha256=sha(args.plan))
    dump(args.out / "summary.json", summary)
    start = time.monotonic()
    try:
        for arm in order:
            run_child(command(cell, args.out / "dry" / arm, args.mem_port, args.nav_port) + ["--contract_dry_run"],
                      args.out / "logs" / (arm + "_cli.log"), environment=env)
        with servers(args.out, "legacy", args.mem_port, args.nav_port, dense_window=32,
                     memory_server_entrypoint=ROOT / "MemNavData/gem_position_readout_server.py"):
            for arm in order:
                dump(args.out / "active_position_readout.json", dict(arm=arm, goal_key=key))
                target = args.out / "evaluation" / arm
                logs = [args.out / "lingbot_pose_readout.jsonl", args.out / "position_readout_audit.jsonl"]
                offsets = [p.stat().st_size if p.exists() else 0 for p in logs]
                dump(args.out / "progress.json", dict(arm=arm, completed=len(summary["records"]), total=2))
                print("START", args.index, cell["scene"], arm, flush=True)
                elapsed = run_child(command(cell, target, args.mem_port, args.nav_port),
                                    args.out / "logs" / (arm + ".log"), environment=env)
                for path, offset, name in zip(logs, offsets, ["lingbot_frame_poses.json", "position_audit.json"]):
                    with path.open() as stream:
                        stream.seek(offset)
                        dump(target / name, [json.loads(line) for line in stream if line.strip()])
                terminal = load(target / "terminal_measurements.json")
                assert len(terminal) == 1
                row = dict(terminal[0], directory=str(target), scene=cell["scene"], episode=cell["episode"],
                           role="revisit", arm=arm, wall_seconds=elapsed)
                summary["records"].append(row)
                dump(args.out / "summary.json", summary)
                print("DONE", args.index, arm, row["reached"], row["steps"], flush=True)
            dump(args.out / "paired_gpu_binding.json", gpu_binding(args.out))
        check_plan(args.plan)
        summary.update(completed=True, elapsed_seconds=time.monotonic() - start)
        dump(args.out / "summary.json", summary)
        run_child([os.environ["REPAIRED_HAB_PY"], "-u", str(HERE), "verify", "--plan", str(args.plan),
                   "--index", str(args.index), "--out", str(args.out)], args.out / "logs/independent_verify.log", environment=env)
        assert load(args.out / "independent_verification.json")["verified"]
        dump(args.out / "completion.json", dict(completed=True, index=args.index,
             elapsed_seconds=time.monotonic() - start, plan_sha256=sha(args.plan),
             summary_sha256=sha(args.out / "summary.json"),
             independent_verification_sha256=sha(args.out / "independent_verification.json"),
             gpu=load(args.out / "paired_gpu_binding.json")))
    except BaseException as error:
        dump(args.out / "failure.json", dict(type=type(error).__name__, error=str(error),
             elapsed_seconds=time.monotonic() - start))
        raise


def execution_metrics(row, goal, frozen):
    """Use the byte-verified construction context for collision and geodesics."""
    import habitat_sim
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_rebuilt_collision_context import rebuilt_execution_context
    folder = Path(row["directory"])
    receipt = frozen["receipt"]
    with rebuilt_execution_context(receipt["source_asset"], receipt["source_asset_sha256"],
            folder / "execution.navmesh", folder / "verification.navmesh") as (pf, mesh):
        audit, evidence, plans, actions = verify_rollout(
            row, execution_pathfinder=pf, success_radius_m=0.3)
        locations = [a["position_before"] for a in actions] + [row["end_position"]]
        distances = [math.hypot(p[0]-goal[0], p[2]-goal[2]) for p in locations]
        geodesics = []
        for location in locations:
            request = habitat_sim.ShortestPath()
            request.requested_start, request.requested_end = np.asarray(location), np.asarray(goal)
            geodesics.append(float(request.geodesic_distance) if pf.find_path(request) else None)
        assert geodesics[0] is not None and abs(geodesics[0] - row["geodesic_m"]) < 1e-5
        assert all(d is None or math.isfinite(d) and d >= 0 for d in geodesics)
    return audit, evidence, plans, actions, distances, geodesics, mesh


def verify(args):
    import numpy as np
    from MemNavData.verify_habitat_minimal_repair import read_rows
    from MemNavData.verify_gem_bearing_attribution import non_timing
    from MemNavData.certified_relocalization_runtime import scale_free_relative_xy
    from MemNavData.table1_repaired_eval import history
    plan = check_plan(args.plan)
    cell = plan["cells"][args.index]
    manifest, summary = load(args.out / "manifest.json"), load(args.out / "summary.json")
    assert summary["completed"] and len(summary["records"]) == 2
    assert summary["cell"] == manifest["cell"] == cell
    assert summary["plan_sha256"] == manifest["plan_sha256"] == sha(args.plan)
    for path, expected in manifest["input_sha256"].items():
        assert sha(path) == expected, path
    payload, frozen, _ = history(cell)
    goal = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")["floor_position"]
    indexed, checks, metrics = {}, [], []
    for row in summary["records"]:
        arm, folder = row["arm"], Path(row["directory"])
        audit, evidence, plans, actions, distances, geodesics, mesh = execution_metrics(row, goal, frozen)
        reads = load(folder / "position_audit.json")
        poses = load(folder / "lingbot_frame_poses.json")
        query = load(folder / "query_plans.json")["query_leg"]
        assert len(reads) == len(query) == len(plans) > 0
        for read, p in zip(reads, query):
            assert read["arm"] == arm and read["accepted"] == p["certified_relocalization_accepted"]
            assert read["selected_anchor"] == p["router_selected_anchor"]
            assert p["certified_relocalization_reason"] != "certificate_endpoint_failure"
            if read["accepted"]:
                index = p["frame_idx"]
                expected = (read["pnp_bearing"] if arm == "pnp_goal" else
                            scale_free_relative_xy(poses[index]["camera_pose9"], poses[read["selected_anchor"]]["camera_pose9"]))
                np.testing.assert_allclose(read["consumed_bearing"], expected, rtol=0, atol=1e-12)
                norm = np.linalg.norm(read["consumed_bearing"])
                if norm <= 1e-12:
                    assert p["memory_bearing_unit"] is None and not p["revisit_adapter_takeover"]
                else:
                    unit = np.asarray(read["consumed_bearing"]) / norm
                    np.testing.assert_allclose(p["memory_bearing_unit"], unit, rtol=0, atol=1e-7)
        boundary = read_rows(folder / "memory_http_boundary.jsonl")
        forbidden = {"executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m",
                     "executor_local_se2_source", "executor_local_se2_contract"}
        assert not any(forbidden & set(p["sent_fields"]) for p in boundary)
        passages = [{"radius_m": radius, "euclidean_first_action": first_passage(distances, radius),
                     "geodesic_first_action": next((i for i, d in enumerate(geodesics) if d is not None and d < radius), None)}
                    for radius in (1., .5, .3)]
        assert (passages[-1]["euclidean_first_action"] is not None) == bool(row["reached"])
        if row["reached"]:
            assert passages[-1]["euclidean_first_action"] == row["steps"]
        metrics.append(dict(arm=arm, reached=row["reached"], steps=row["steps"], passages=passages,
                            endpoint_error_m=distances[-1], minimum_error_m=min(distances),
                            final_geodesic_m=geodesics[-1], turn_actions=audit["heading"]["turn_actions"],
                            euclidean_distances=distances, geodesic_distances=geodesics,
                            execution_context=mesh))
        indexed[arm] = dict(reads=reads, poses=poses, plans=plans)
        checks.append(audit)
    assert set(indexed) == set(ARMS)
    pnp, historical = indexed["pnp_goal"], indexed["historical_camera"]
    for field in ("accepted", "selected_anchor", "candidates", "pnp_bearing", "certificate", "pnp"):
        assert non_timing(pnp["reads"][0][field]) == non_timing(historical["reads"][0][field]), field
    if pnp["reads"][0]["accepted"]:
        assert pnp["reads"][0]["selected_anchor"] == cell["selection_anchor"]
    h = cell["history_frames"]
    assert len(pnp["poses"]) > h and len(historical["poses"]) > h
    for a, b in zip(pnp["poses"][:h+1], historical["poses"][:h+1]):
        for field in ("frame_idx", "image_sha256", "camera_pose9", "pose_count", "motion_receipt_recorded"):
            assert a[field] == b[field], field
    for field in ("input_tensor_sha256", "output_tensor_sha256", "contract"):
        assert pnp["plans"][0]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field] == historical["plans"][0]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field]
    dump(args.out / "independent_verification.json", dict(verified=True, checks=checks, metrics=metrics,
         first_certificate_and_pnp_identical=True, predicted_historical_bearing_verified=True,
         full_causal_prefix_identical=True, success_radius_m=0.3, thresholds_measured_on_new_complete_trajectories=True,
         plan_sha256=sha(args.plan), summary_sha256=sha(args.out / "summary.json")))
    print("VERIFIED", args.index, "matched position readouts", flush=True)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "eval":
        from MemNavData.run_gem_memory_navigation import evaluate
        evaluate()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--index", type=int, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--mem-port", type=int, default=21930)
        parser.add_argument("--nav-port", type=int, default=21931)
        args = parser.parse_args()
        args.plan, args.out = args.plan.resolve(), args.out.resolve()
        if action == "run":
            run(args)
        elif action == "verify":
            verify(args)
        else:
            parser.error("Expected run, verify, or eval")
