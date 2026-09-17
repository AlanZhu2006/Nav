"""Standalone use of two already selected Revisit queries, never fake role pairs.

The sealed model/server/evaluator stack is reused. This file only reconstructs
saved construction identities, replays their factual histories and measures
six query rollouts. It does not search for replacement goals or collect A.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ARMS = ("native", "raw_fixed", "cec")
SCHEMA = "repaired_existing_revisit_integration_20260909_v1"
SCOPE = "two existing supported Revisits; integration only, not mixed-role confirmation"
RESULTS = Path("/scratch/yz11502/Research/Nav-axis-uturn-results")
SOURCES = (
    ("rJhMRvNn4DS", RESULTS / "repaired_hm3d_gate_20260908/gate_ee865620b4540675/integration",
     "3a8eb572734e0144242b3eac3c5c30f4dddd08acbbe8f8826bb3a8dc0aab5bfc",
     "5d0d104de37440cc2bb86789be2ef742495367d0948a35eac7c92bec0395f348"),
    ("jgPBycuV1Jq", RESULTS / "repaired_hm3d_extension_20260908/extension_c8cf8c60e7efd55f/shard_0/integration",
     "2362a5f3d485910d9ccdd29c50a3f0450df3f92306a040b1895f02f530c456a4",
     "0265f87e7281a78a416a49e820a26cf855417b0f7bff64295ad560c211e142f8"),
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def arm_order(index):
    require(index in (0, 1), "exactly two frozen histories")
    return ARMS if index == 0 else tuple(reversed(ARMS))


def reconstruct_pose(pathfinder, history, selected, scene, episode):
    import numpy as np
    import build_final14_role_pair_scene as b
    frame = selected["source_frame"]
    radius, direction, offset = b.deterministic_pose_grid(
        f"{scene}/{episode}/{frame}")[selected["render_attempt"] - 1]
    original = history["floor_positions"][frame]
    raw = original + np.array([radius * math.cos(direction), 0., radius * math.sin(direction)])
    position = np.asarray(pathfinder.snap_point(raw), dtype=float)
    yaw = b.history_tools.wrap_radians(history["poses"][frame]["yaw"] + math.radians(offset))
    require(pathfinder.is_navigable(position), "saved query pose became invalid")
    require(abs(position[1] - original[1]) <= .20
            and np.linalg.norm((position-raw)[[0, 2]]) <= .20, "saved pose snap changed")
    return position, yaw


def check_selected(selected, actual):
    """Tolerance is numerical replay only; scientific thresholds do not change."""
    for key in ("source_frame", "render_attempt", "max_online_a_covis_frame", "support_band"):
        require(selected[key] == actual[key], f"saved selected identity changed: {key}")
    for key in ("query_geodesic_m", "initial_path_bearing_rad", "translation_m",
                "yaw_delta_deg", "source_anchor_covis", "max_online_a_covis", "pixel_mae"):
        require(math.isclose(selected[key], actual[key], abs_tol=1e-5, rel_tol=0),
                f"saved query measurement changed: {key}: {selected[key]} vs {actual[key]}")


def build(out):
    import numpy as np
    import build_final14_role_pair_scene as b
    from shared_online_role_pair_contract import validate_query
    out.mkdir(parents=True, exist_ok=False)
    episodes, sources = [], []
    for index, (scene, original, construction_sha, verification_sha) in enumerate(SOURCES):
        require(sha(original / "construction_summary.json") == construction_sha, "construction changed")
        require(sha(original / "independent_verification.json") == verification_sha, "A verification changed")
        require(load(original / "independent_verification.json")["verified"], "A not verified")
        source = next(s for s in load(original / "manifest.json")["sources"] if s["scene"] == scene)
        for path, expected in source["source_files"].items():
            require(sha(path) == expected, f"source changed: {path}")
        report = next(r for r in load(original / "construction_summary.json")["reports"] if r["scene"] == scene)
        attempt = report["construction"]["attempts"][0]
        selected = attempt["selected"]["standard"]
        require(selected is not None, "no preselected standard Revisit")
        online = original / "construction" / scene / "online_a" / scene / source["episode"]
        receipt = load(online / "receipt.json")
        history = b.history_tools.load_online_history(online, receipt)
        require(history["trace"]["reached"] is True, "A was unsuccessful")
        a_trace = original / "goal_a" / scene / source["episode"] / f"{source['episode']}_leg1_trace.json"
        require(sha(a_trace) == sha(online / "online_a_trace.json"), "online A trace mismatch")
        sim = b.make_sim(source["asset"], "", agent_radius=.30)
        destination = out / scene / source["episode"]
        destination.mkdir(parents=True)
        try:
            position, yaw = reconstruct_pose(sim.pathfinder, history, selected, scene, source["episode"])
            camera = position + np.array([0., receipt["camera_height_m"], 0.])
            rgb, depth = b.render(sim, camera, yaw)
            points = b.history_tools.goal_world_points(depth, camera, yaw)
            curve = b.covis_curve(points, history["transforms"], history["depths"], tol=b.DEPTH_TOLERANCE_M)
            frame = selected["source_frame"]
            best = b.ELIGIBLE_FRAME_FLOOR + int(np.argmax(curve[b.ELIGIBLE_FRAME_FLOOR:]))
            distance, bearing = b.pair_tools.query_geometry(sim.pathfinder, b.online_endpoint(history)[0], position)
            actual = dict(selected, query_geodesic_m=distance, initial_path_bearing_rad=bearing,
                translation_m=float(np.linalg.norm((position-history["floor_positions"][frame])[[0, 2]])),
                yaw_delta_deg=b.history_tools.angle_delta_degrees(yaw, history["poses"][frame]["yaw"]),
                source_anchor_covis=float(b.covis_frac(points, history["transforms"][frame], history["depths"][frame])),
                max_online_a_covis=float(curve[best]), max_online_a_covis_frame=best,
                pixel_mae=b.history_tools.pixel_mae(rgb, history["rgbs"][frame]),
                support_band=b.support_band(float(curve[best]), abs(best-frame)),
                _position=position, _yaw=yaw, _rgb=rgb, _depth=depth, _covis_curve=curve.tolist())
            check_selected(selected, actual)
            require(2 <= distance <= 9 and .20 <= actual["translation_m"] <= .80
                    and 12 <= actual["yaw_delta_deg"] <= 45 and actual["support_band"] == "standard",
                    "original standard support conditions failed")
            rgb_path, depth_path, rgb_sha, depth_sha = b._write_goal(destination, "query", actual)
            query = b._query_record(query_id="query_00", role="revisit", candidate=actual,
                episode_root=destination, rgb_path=rgb_path, depth_path=depth_path,
                rgb_sha=rgb_sha, depth_sha=depth_sha)
            validate_query(query)
            payload = {"schema_version": SCHEMA, "scope": SCOPE, "scene": scene,
                "episode": source["episode"], "source": source, "online_a_episode": str(online),
                "online_a_receipt_sha256": sha(online / "receipt.json"),
                "online_a_trace_sha256": sha(online / "online_a_trace.json"),
                "online_a_steps": len(history["poses"]), "query": query,
                "selected_record": selected, "original_construction": str(original / "construction_summary.json"),
                "original_construction_sha256": construction_sha, "original_verification_sha256": verification_sha,
                "arm_order": arm_order(index), "new_a_rollouts": 0, "mixed_role_pairs": 0}
            dump(destination / "revisit_query.json", payload)
            episodes.append({"scene": scene, "path": str(destination / "revisit_query.json"),
                             "sha256": sha(destination / "revisit_query.json")})
            sources.append(source)
            print(f"RESTORED {scene}: frame={frame} geo={distance:.6f} covis={curve[best]:.6f}", flush=True)
        finally:
            sim.close()
    manifest = {"schema": SCHEMA, "scope": SCOPE, "episodes": episodes, "sources": sources,
                "arms": ARMS, "query_count": 2, "query_arm_count": 6, "new_a_rollouts": 0,
                "mixed_role_pairs": 0, "protocol_sha256": sha(HERE.with_name("REPAIRED_HM3D_REVISIT_INTEGRATION_PROTOCOL_20260909.md"))}
    dump(out / "manifest.json", manifest)
    return manifest


def load_query(path, expected_scene):
    p = load(path)
    require(p["schema_version"] == SCHEMA and p["scene"] == expected_scene, "standalone scene/schema mismatch")
    require("pairs" not in p and p["mixed_role_pairs"] == 0, "not a role-pair population")
    source = Path(p["online_a_episode"])
    for name, field in (("receipt.json", "online_a_receipt_sha256"), ("online_a_trace.json", "online_a_trace_sha256")):
        require(sha(source / name) == p[field], f"historical {name} changed")
    trace = load(source / "online_a_trace.json")
    require(trace["reached"] is True and len(trace["poses"]) == p["online_a_steps"], "historical A changed")
    return {"benchmark": p, "source": source, "receipt": load(source / "receipt.json"), "trace": trace}


def evaluate_query():
    import numpy as np
    import pandas as pd
    import eval_shared_online_role_pairs as shared
    from shared_online_role_pair_contract import runtime_query
    base, args = shared.base, shared.args
    arm, backend = shared.validate_cli()
    require(args.role_pair_scope == "consumed_integration" and args.role_pair_query_role == "revisit",
            "standalone Revisit scope must be explicit")
    if args.contract_dry_run:
        print(f"STANDALONE REVISIT CLI OK: {arm}; no pair schema alteration")
        return
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "query output must be empty")
    episode_dir = Path(args.episode_root) / args.episode_ids
    path = episode_dir / "revisit_query.json"
    manifest = load(episode_dir.parent.parent / "manifest.json")
    expected = next(e for e in manifest["episodes"] if e["scene"] == base.SCENE_IDENTITY)
    require(sha(path) == expected["sha256"], "frozen query manifest changed")
    frozen = load_query(path, base.SCENE_IDENTITY)
    receipt, payload, trace = frozen["receipt"], frozen["benchmark"], frozen["trace"]
    require(sha(args.scene) == receipt["source_asset_sha256"], "scene asset mismatch")
    parquet = Path(receipt["source_episode"]) / "data/chunk-000/episode_000000.parquet"
    require(sha(parquet) == receipt["source_parquet_sha256"], "camera intrinsic carrier changed")
    intrinsic = np.stack([np.asarray(row, dtype=float) for row in
                         pd.read_parquet(parquet).iloc[0]["observation.camera_intrinsic"]])
    height = float(receipt["camera_height_m"])
    require(math.isclose(height, base.CAM_H, abs_tol=1e-12), "camera height mismatch")
    query = runtime_query(payload["query"])
    for field in ("goal_rgb", "goal_depth"):
        require(sha(episode_dir / query[field]) == query[field + "_sha256"], "query asset changed")
    # The depth file and GT position are evaluator-side only. run_policy_leg
    # receives GT solely for scoring; its HTTP boundary strips pose/role data.
    goal_jpg = (episode_dir / query["goal_rgb"]).read_bytes()
    goal = np.asarray(query["floor_position"], float)
    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    try:
        base.srv_reset(camera_height=height, seed=int(trace["episode_seed"]),
            episode_len=payload["online_a_steps"] + args.max_steps, camera_intrinsic=intrinsic,
            causal_history_sha256=payload["online_a_trace_sha256"])
        a, replay = shared.replay_prefix(frozen)
        ok, geo, _ = base.geodesic(sim.pathfinder, a["end_pos"], goal)
        require(ok and abs(geo-payload["query"]["geodesic_from_a_end_m"]) <= .05, "query geodesic mismatch")
        result = base.run_policy_leg(sim, sim.pathfinder, a["end_pos"], a["end_psi"],
            goal_jpg, goal[[0, 2]], float(geo), None, terminal_mode="off", goal_yaw=query["yaw_rad"],
            camera_intrinsic=intrinsic, policy_backend=backend, episode_seed=int(trace["episode_seed"]), leg_index=1)
        dump(output / "query_plans.json", {"schema": SCHEMA, "arm": arm, "query_runtime_fields": sorted(query),
            "analysis_role_not_forwarded": True, "replay": replay, "query_leg": result["plans"],
            "rollout_traces": {"legA": trace["poses"], "query": result["rollout_trace"]},
            "counts": shared.router_counts(result["plans"]), "depth": shared.depth_counts(result["plans"]),
            "query_result": {"reached": bool(result["reached"]), "end_position": result["end_pos"].tolist()}})
    finally:
        sim.close()


def evaluator_command(source, out, mem, nav, *, arm, benchmark):
    from MemNavData.run_repaired_fullmono_local import evaluator_command as original
    cmd = original(source, out, mem, nav, arm=arm, role="revisit", benchmark=benchmark)
    return cmd[:2] + [str(HERE), "eval"] + cmd[4:]


def run(out, mem, nav):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, execution_environment, private_servers, run_child)
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    benchmark = out / "benchmark"
    run_child([HAB_PY, str(HERE), "build", "--out", str(benchmark)],
              out / "logs/build.log", environment=execution_environment())
    manifest = load(benchmark / "manifest.json")
    manifest["runtime_source_receipt"] = {"path": os.environ["REPAIRED_SOURCE_RECEIPT"],
                                          "sha256": sha(os.environ["REPAIRED_SOURCE_RECEIPT"])}
    manifest["integration_source_receipt"] = {"path": str(HERE.parent / "SOURCE_BUNDLE.sha256"),
                                              "sha256": sha(HERE.parent / "SOURCE_BUNDLE.sha256")}
    dump(out / "manifest.json", manifest)
    summary = {"schema": SCHEMA, "scope": SCOPE, "completed": False,
               "goal_a": [], "reused_a_count": 2, "queries": []}
    dump(out / "summary.json", summary)
    pose_log = out / "lingbot_pose_readout.jsonl"
    with private_servers(out, mem, nav):
        for index, source in enumerate(manifest["sources"]):
            for arm in arm_order(index):
                folder = out / "evaluation" / source["scene"] / arm
                command = evaluator_command(source, folder, mem, nav, arm=arm, benchmark=benchmark)
                dump(out / "progress.json", {"scene": source["scene"], "arm": arm,
                     "completed_query_arms": len(summary["queries"]), "command": command})
                print(f"QUERY {source['scene']} {arm}", flush=True)
                pose_offset = pose_log.stat().st_size if pose_log.exists() else 0
                elapsed = run_child(command, out / "logs" / f"{source['scene']}__{arm}.log",
                                    environment=execution_environment())
                with pose_log.open() as stream:
                    stream.seek(pose_offset)
                    dump(folder / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
                terminal = load(folder / "terminal_measurements.json")
                require(len(terminal) == 1, "expected one standalone rollout")
                row = dict(terminal[0], scene=source["scene"], episode=source["episode"], role="revisit",
                           arm=arm, directory=str(folder), wall_seconds=elapsed)
                summary["queries"].append(row)
                dump(out / "summary.json", summary)
                print(f"DONE {source['scene']} {arm}: reached={row['reached']} steps={row['steps']}", flush=True)
    summary["completed"] = True
    dump(out / "summary.json", summary)
    run_child([HAB_PY, str(HERE), "verify", "--out", str(out)], out / "logs/verification.log",
              environment=execution_environment())
    run_child([HAB_PY, str(HERE), "render", "--out", str(out)], out / "logs/render.log",
              environment=execution_environment())


def verify(out):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out / "manifest.json"), load(out / "summary.json")
    for field in ("runtime_source_receipt", "integration_source_receipt"):
        require(sha(manifest[field]["path"]) == manifest[field]["sha256"], "source receipt changed")
    require(summary["completed"] and len(summary["queries"]) == 6 and summary["goal_a"] == [], "incomplete integration")
    indexed, records, pairs = {}, [], []
    for item in manifest["episodes"]:
        require(sha(item["path"]) == item["sha256"], "query metadata changed")
        frozen = load_query(item["path"], item["scene"])
        require(sha(frozen["benchmark"]["original_construction"]) == frozen["benchmark"]["original_construction_sha256"],
                "source construction changed")
    for row in summary["queries"]:
        key = (row["scene"], row["arm"])
        require(key not in indexed, "duplicate query arm")
        checked, evidence, plans, actions = verify_rollout(row)
        folder = Path(row["directory"])
        payload = load(folder / "query_plans.json")
        item = next(e for e in manifest["episodes"] if e["scene"] == row["scene"])
        frozen = load_query(item["path"], row["scene"])
        trace = frozen["trace"]
        require(payload["rollout_traces"]["legA"] == trace["poses"], "wrong history replay")
        require(payload["replay"]["all_rgb_hashes_verified"]
                and payload["replay"]["diffusion_samples_during_replay"] == 0, "history resampled")
        np.testing.assert_array_equal(actions[0]["position_before"], trace["end_position"])
        require(actions[0]["yaw_before"] == trace["end_yaw"], "start yaw changed")
        require(payload["analysis_role_not_forwarded"] and "analysis_role" not in payload["query_runtime_fields"], "role leak")
        forbidden = {"analysis_role", "max_online_a_covis", "executed_translation_m", "executed_yaw_rad",
                     "executed_forward_m", "executed_left_m"}
        for boundary in read_rows(folder / "memory_http_boundary.jsonl"):
            require(not forbidden.intersection(boundary["sent_fields"]), "privileged memory request")
        indexed[key] = (row, payload, plans, actions)
        records.append(checked)
    require(set(indexed) == {(s[0], arm) for s in SOURCES for arm in ARMS}, "wrong population")
    for scene, *_ in SOURCES:
        native = indexed[(scene, "native")]
        cec = indexed[(scene, "cec")]
        for arm in ARMS:
            row, payload, _, _ = indexed[(scene, arm)]
            require(row["first_query_rgb_sha256"] == native[0]["first_query_rgb_sha256"]
                    and row["goal_xz_evaluator_only"] == native[0]["goal_xz_evaluator_only"]
                    and payload["replay"] == native[1]["replay"], "arms not paired")
        takeover = sum(p["receipt"]["revisit_adapter_takeover"] is True for p in cec[2])
        if not takeover:
            require(len(native[2]) == len(cec[2]) and len(native[3]) == len(cec[3]), "no-takeover length differs")
            for a, b in zip(native[2], cec[2]):
                for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                    np.testing.assert_array_equal(a[field], b[field])
            for a, b in zip(native[3], cec[3]):
                for field in ("actual_position", "actual_yaw", "action_kind"):
                    require(a[field] == b[field], "no-takeover action differs")
            depth = []
            for item in (native, cec):
                receipts = read_rows(Path(item[0]["directory"]) / "navdp_http_receipts.jsonl")
                depth.append([r["monocular_depth_receipt"] for r in receipts
                    if r["query_active"] and r["path"] != "/memory_replay_step" and r["audit"]["image_calls"]])
            require(len(depth[0]) == len(depth[1]) == len(native[2]), "depth receipt count differs")
            for a, b in zip(*depth):
                for field in ("depth_png_sha256", "image_sha256", "scale_receipt_sha256", "frame_index"):
                    require(a[field] == b[field], "no-takeover depth differs")
                require(a["navdp_depth_raster"]["input_tensor_sha256"] == b["navdp_depth_raster"]["input_tensor_sha256"]
                        and a["navdp_depth_raster"]["output_tensor_sha256"] == b["navdp_depth_raster"]["output_tensor_sha256"],
                        "no-takeover depth tensor differs")
        pairs.append({"scene": scene, "success": {a: indexed[(scene, a)][0]["reached"] for a in ARMS},
                      "cec_takeover_plans": takeover, "no_takeover_exact_native": takeover == 0})
    dump(out / "independent_verification.json", {"verified": True, "scope": SCOPE,
        "records": records, "pairs": pairs, "query_arm_count": 6,
        "new_a_rollouts": 0, "mixed_role_pairs": 0, "verifier_sha256": sha(HERE)})
    print(json.dumps(pairs, indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sys.argv.pop(1)
        # Import the small callback-enabled wrapper beside this script, while
        # all actual runtime modules continue to resolve to the sealed bundle.
        spec = importlib.util.spec_from_file_location("revisit_execution_instrumentation",
            HERE.with_name("run_habitat_minimal_repair_local.py"))
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        wrapper.evaluate(query_main=evaluate_query)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "run", "verify", "render", "dry-run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memnav-port", type=int, default=21750)
    parser.add_argument("--navdp-port", type=int, default=21751)
    args = parser.parse_args()
    if args.mode == "build":
        build(args.out.resolve())
    elif args.mode == "run":
        run(args.out.resolve(), args.memnav_port, args.navdp_port)
    elif args.mode == "verify":
        verify(args.out.resolve())
    elif args.mode == "render":
        from MemNavData.verify_repaired_fullmono_local import render
        render(args.out.resolve())
    else:
        import subprocess
        from MemNavData.run_repaired_fullmono_local import execution_environment
        source = {"scene": SOURCES[0][0], "episode": "episode_0000", "asset": "/unused/scene.glb",
                  "source_episode": "/unused/scene/episode_0000", "seed": 2026082200}
        for arm in ARMS:
            cmd = evaluator_command(source, args.out / arm, args.memnav_port, args.navdp_port,
                                    arm=arm, benchmark=Path("/unused/benchmark")) + ["--contract_dry_run"]
            subprocess.run(cmd, env=execution_environment(), check=True)
        require(not args.out.exists(), "dry run wrote an output tree")


if __name__ == "__main__":
    main()
