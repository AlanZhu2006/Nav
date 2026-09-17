"""Local Table-II A -> {B-N, B-R} -> {C-N, C-R} integration.

The existing repaired executor and frozen servers own navigation. This file
owns new goal construction, actual-prefix materialization, branch selection
and measurement only. No formal population is frozen by this local pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from MemNavData.table2_novel_sampling import (
    balanced_c_sources, geometry_pool, initial_a_yaw,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve()
SCHEMA = "table2_mixed_actual_prefix_local_20260911_v1"
ROLES = ("novel", "revisit")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    from MemNavData.habitat_executor_audit import sha as digest
    return digest(path)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compose_prefix(a, b):
    """Join ONLY the factual A and this native-B branch; check the join."""
    if not a["reached"] or not b["reached"]:
        raise ValueError("C requires successful actual A and native B")
    for trace in (a, b):
        if trace["source_hybrid_route"] != "native_sidecar":
            raise ValueError("A GEM rollout cannot supply the shared C prefix")
        if [p["step"] for p in trace["poses"]] != list(range(trace["steps"])):
            raise ValueError("Prefix observations must be contiguous")
    if any(a[k] != b[k] for k in ("episode", "episode_seed", "source_scene")):
        raise ValueError("A and B source identities differ")
    first = b["poses"][0]
    if not np.allclose([first[k] for k in "xyz"], a["end_position"], atol=1e-8, rtol=0):
        raise ValueError("B must start at the actual final A position")
    if abs(math.atan2(math.sin(first["yaw"] - a["end_yaw"]),
                      math.cos(first["yaw"] - a["end_yaw"]))) > 1e-8:
        raise ValueError("B must inherit the actual A heading")
    from MemNavData.hm3d_table2_leg3_mixed_role import compose_actual_ab_trace
    result = compose_actual_ab_trace(a, b, episode=a["episode"])
    # The original function is role-agnostic; its old Novel-B-only caller is
    # deliberately NOT used for this experiment.
    result["prefix_semantics"] = "actual_A_then_one_native_B_branch"
    return result


def support_sources(curve, a_steps):
    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= a_steps <= len(values):
        raise ValueError("Invalid complete A/B support curve")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Co-visibility lies outside [0,1]")
    a = float(values[:a_steps].max()) if a_steps else 0.
    b = float(values[a_steps:].max()) if a_steps < len(values) else 0.
    if a >= .55 and b < .10:
        label = "A_only"
    elif b >= .55 and a < .10:
        label = "B_only"
    elif a >= .55 and b >= .55:
        label = "both"
    else:
        label = "intermediate"
    return dict(max_A_covis=a, max_B_covis=b, support_source=label)


def runtime_spec(query):
    """Evaluator-only pose/score fields remain local; no role enters a server."""
    keys = ("schema", "scene", "episode", "asset", "seed", "stage_number",
            "start_position", "start_yaw", "camera_height_m", "camera_intrinsic",
            "goal_rgb", "goal_rgb_sha256", "floor_position", "yaw_rad",
            "geodesic_m", "prefix_root", "prefix_trace_sha256")
    return {key: query[key] for key in keys}


def base_source(source, rank):
    if "initial_state" in source:
        spec = deepcopy(source["initial_state"])
        if (spec["scene"], spec["episode"], spec["seed"], spec["asset"]) != (
                source["scene"], source["episode"], source["seed"], source["asset"]):
            raise ValueError("Declared initial state differs from the source identity")
        if spec["prefix_root"] is not None or spec["prefix_trace_sha256"] is not None:
            raise ValueError("New Goal-A must start without memory")
        return spec
    import pandas as pd
    from MemNavData.habitat_rollout_primitives import DATA_TO_HABITAT_ROTATION
    rows = pd.read_parquet(Path(source["source_episode"]) / "data/chunk-000/episode_000000.parquet")
    meta = load(Path(source["source_episode"]) / "meta/gen_meta.json")
    camera_height = float(meta.get("camera_height_m", .5))
    action = np.stack(rows.iloc[0]["action"]).reshape(4, 4)
    position = DATA_TO_HABITAT_ROTATION @ action[:3, 3] - [0., camera_height, 0.]
    key = f"{source['scene']}/{source['episode']}"
    return dict(schema=SCHEMA, scene=source["scene"], episode=source["episode"],
                asset=source["asset"], seed=2026091100 + rank, stage_number=0,
                start_position=position.tolist(), start_yaw=initial_a_yaw(key),
                camera_height_m=camera_height,
                camera_intrinsic=np.stack(rows.iloc[0]["observation.camera_intrinsic"]).tolist(),
                prefix_root=None, prefix_trace_sha256=None)


def history_at(prefix):
    from MemNavData.build_shared_online_double_revisit import load_online_history
    return load_online_history(Path(prefix), load(Path(prefix) / "receipt.json"))


def revisit_source_frames(history_length):
    """Sample addresses before perturbation; distance bounds apply to goals."""
    return list(range(39, history_length - 16, 8))


def construct_query(source, *, stage, prefix, out):
    """Render and select before outcomes: no DINO, proof, or policy call."""
    from MemNavData.generate_twoleg import make_sim, render, covis_curve, covis_frac, cam_to_world_hab
    from MemNavData.build_shared_online_double_revisit import (
        goal_world_points, jpeg_bytes, write_depth_png, pixel_mae,
    )
    from MemNavData.build_final14_role_pair_scene import deterministic_pose_grid
    from MemNavData.build_shared_online_role_pairs import query_geometry
    from MemNavData.final14_role_pair_contract import (
        DEPTH_TOLERANCE_M, relative_direction_degrees, STRATA, direction_in_stratum,
    )

    spec = deepcopy(source)
    history = history_at(prefix) if prefix is not None else None
    if history is not None:
        trace = history["trace"]
        if not trace["reached"]:
            raise ValueError("Cannot construct from an unsuccessful prefix")
        spec.update(start_position=trace["end_position"], start_yaw=trace["end_yaw"],
                    prefix_root=str(Path(prefix).resolve()),
                    prefix_trace_sha256=sha(Path(prefix) / "online_a_trace.json"))
    spec["stage_number"] = 0 if stage == "A" else 1 if stage == "B" else 2
    out.mkdir(parents=True, exist_ok=False)
    sim = make_sim(spec["asset"], "", agent_radius=.30)
    pf, height = sim.pathfinder, spec["camera_height_m"]
    start = np.asarray(spec["start_position"])
    key = f"{spec['scene']}/{spec['episode']}/{stage}/{Path(prefix).name if prefix else 'empty'}"
    reports, queries = {}, {}
    try:
        sim.pathfinder.save_nav_mesh(str(out / "construction.navmesh"))
        _, current_depth = render(sim, start + [0., height, 0.], spec["start_yaw"])
        current_transform = cam_to_world_hab(start + [0., height, 0.], spec["start_yaw"])

        def measure(position, yaw):
            rgb, depth = render(sim, np.asarray(position) + [0., height, 0.], yaw)
            points = goal_world_points(depth, np.asarray(position) + [0., height, 0.], yaw)
            curve = covis_curve(points, history["transforms"], history["depths"], tol=DEPTH_TOLERANCE_M) if history else np.empty(0)
            visible = float(covis_frac(points, current_transform, current_depth, tol=DEPTH_TOLERANCE_M))
            return rgb, depth, curve, visible

        def save(role, candidate, observed):
            rgb, depth, curve, visible = observed
            goal_dir = out / role
            goal_dir.mkdir()
            (goal_dir / "goal.jpg").write_bytes(jpeg_bytes(rgb))
            write_depth_png(goal_dir / "goal_depth_evaluator_only.png", depth)
            n = len(curve)
            a_steps = history["trace"].get("prefix_A_steps", n) if history else 0
            row = dict(spec, **candidate, analysis_role=role,
                       goal_rgb=str((goal_dir / "goal.jpg").resolve()),
                       goal_rgb_sha256=sha(goal_dir / "goal.jpg"),
                       covis_curve=curve.tolist(), max_history_covis=float(curve.max()) if n else 0.,
                       max_runtime_eligible_covis=float(curve[8:].max()) if n > 8 else 0.,
                       runtime_eligible_frame_floor=8, construction_source_frame_floor=39,
                       history_frames=n, current_view_covis=visible,
                       **support_sources(curve, a_steps))
            dump(goal_dir / "query.json", row)
            dump(goal_dir / "runtime.json", runtime_spec(row))
            queries[role] = str((goal_dir / "query.json").resolve())

        pool = geometry_pool(pf, start, spec["start_yaw"], key, per_direction=16, max_attempts=5000)
        checked = []
        # Pilot supply audit: chronological proposal order, no direction quota.
        # Formal cross-stage quotas remain undecided; no SR-driven replacement.
        for candidate in sorted(pool["candidates"], key=lambda r: r["proposal_index"]):
            observed = measure(candidate["floor_position"], candidate["yaw_rad"])
            curve = observed[2]
            maximum = float(curve.max()) if len(curve) else 0.
            checked.append(dict(proposal_index=candidate["proposal_index"], max_covis=maximum))
            if maximum < .10:
                row = {k: v for k, v in candidate.items() if k not in ("visual_support_measured", "geodesic_m")}
                row["geodesic_m"] = candidate["geodesic_m"]
                save("novel", row, observed)
                break
        reports["novel"] = dict(spatial_supply=pool, visual_checked=checked,
                                constructed="novel" in queries)
        if history is not None:
            counts = Counter()
            selected = False
            sources = revisit_source_frames(len(history["poses"]))
            # A source closer than 2 m can produce a legal perturbed goal.
            # The exact 2--9 m test is therefore applied only to the final
            # candidate pose below, not to its historical source address.
            # Fixed temporal sampling covers both A and B, not just early A.
            for frame in sources:
                origin = history["floor_positions"][frame]
                origin_yaw = history["poses"][frame]["yaw"]
                for attempt, (radius, angle, dyaw) in enumerate(deterministic_pose_grid(key + f"/{frame}")):
                    if radius > .8 or not 12. <= abs(dyaw) <= 45.:
                        continue
                    raw = origin + radius * np.asarray([math.cos(angle), 0., math.sin(angle)])
                    position = np.asarray(pf.snap_point(raw))
                    counts["pose_proposals"] += 1
                    if (not pf.is_navigable(position) or abs(position[1] - origin[1]) > .20
                            or np.linalg.norm((position - raw)[[0, 2]]) > .20
                            or not .20 <= np.linalg.norm((position - origin)[[0, 2]]) <= .80):
                        continue
                    geometry = query_geometry(pf, start, position)
                    if geometry is None or not 2. <= geometry[0] <= 9.:
                        continue
                    yaw = (origin_yaw + math.radians(dyaw) + math.pi) % (2*math.pi) - math.pi
                    observed = measure(position, yaw)
                    counts["full_history_scored"] += 1
                    mae = pixel_mae(observed[0], history["rgbs"][frame])
                    curve = observed[2]
                    maximum = float(curve[8:].max()) if len(curve) > 8 else 0.
                    if mae < 5. or not .55 <= maximum <= .90:
                        continue
                    relative = relative_direction_degrees(geometry[1], spec["start_yaw"])
                    row = dict(floor_position=position.tolist(), yaw_rad=yaw, geodesic_m=geometry[0],
                               source_online_frame=frame, source_pose_attempt=attempt,
                               translation_from_source_m=float(np.linalg.norm((position-origin)[[0, 2]])),
                               yaw_delta_from_source_deg=abs(dyaw), pixel_mae_from_source=mae,
                               initial_relative_route_angle_deg=relative,
                               direction_stratum=next(s for s in STRATA if direction_in_stratum(relative, s)),
                               straight_distance_m=float(np.linalg.norm((position-start)[[0, 2]])))
                    row["route_ratio"] = row["geodesic_m"] / row["straight_distance_m"]
                    save("revisit", row, observed)
                    selected = True
                    break
                if selected:
                    break
            reports["revisit"] = dict(counts=counts, source_frames=sources, constructed=selected,
                                     distance_filter_applied_to="final_perturbed_goal")
        result = dict(schema=SCHEMA, stage=stage, queries=queries, diagnostics=reports,
                      both_constructed=all(r in queries for r in ROLES),
                      navigation_outcomes_read=False)
        dump(out / "construction.json", result)
        return result
    finally:
        sim.close()


def materialize_prefix(rollout, parent, out, source):
    """Render ONLY recorded physical observations and verify their JPEG bytes."""
    from MemNavData.generate_twoleg import make_sim, render
    from MemNavData.build_shared_online_double_revisit import jpeg_bytes, write_depth_png
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    b = load(Path(rollout) / "actual_trace.json")
    validate_leg1_trace(b)
    if b["source_hybrid_route"] != "native_sidecar" or not b["reached"]:
        raise ValueError("Only successful native rollouts may supply a prefix")
    a = load(Path(parent) / "online_a_trace.json") if parent else None
    trace = compose_prefix(a, b) if a else b
    out.mkdir(parents=True, exist_ok=False)
    for kind in ("rgb", "depth"):
        (out / kind).mkdir()
        if parent:
            for p in sorted((Path(parent) / kind).iterdir()):
                os.link(p, out / kind / p.name)
    offset = len(a["poses"]) if a else 0
    sim = make_sim(source["asset"], "", agent_radius=.30)
    try:
        for pose in b["poses"]:
            floor = np.asarray([pose[k] for k in "xyz"])
            rgb, depth = render(sim, floor + [0., source["camera_height_m"], 0.], pose["yaw"])
            encoded = jpeg_bytes(rgb)
            if hashlib.sha256(encoded).hexdigest() != pose["jpg_sha256"]:
                raise RuntimeError(f"Actual RGB rerender mismatch at {pose['step']}")
            index = offset + pose["step"]
            (out / "rgb" / f"{index:06d}.jpg").write_bytes(encoded)
            write_depth_png(out / "depth" / f"{index:06d}.png", depth)
    finally:
        sim.close()
    dump(out / "online_a_trace.json", trace)
    dump(out / "receipt.json", dict(schema_version="shared_online_a_materialized_v1",
         history_source="actual_frozen_mono_navdp", online_a_steps=len(trace["poses"]),
         camera_height_m=source["camera_height_m"], camera_intrinsic=source["camera_intrinsic"],
         online_a_trace_sha256=sha(out / "online_a_trace.json"),
         parent_prefix=str(Path(parent).resolve()) if parent else None,
         source_rollout=str(Path(rollout).resolve()), source_trace_sha256=sha(Path(rollout)/"actual_trace.json"),
         source_asset=source["asset"], source_asset_sha256=sha(source["asset"])))


def evaluate_one():
    """Called inside the unchanged repaired evaluation hooks."""
    import eval_2leg_habitat as base
    from MemNavData.shared_online_double_revisit_runtime import replay_online_a
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    query_path = Path(os.environ["TABLE2_QUERY"])
    query = load(query_path)
    if set(query) != set(runtime_spec(query)):
        raise ValueError("Runtime specification contains analysis-only fields")
    if sha(query["goal_rgb"]) != query["goal_rgb_sha256"]:
        raise ValueError("Goal JPEG changed")
    if base.args.contract_dry_run:
        print("TABLE2_DRY_RUN: role-free repaired native/GEM query entrypoint", flush=True)
        return
    output = Path(base.args.out)
    simulator = base.make_sim(query["asset"], "", agent_radius=.30)
    prefix = query["prefix_root"]
    trace = load(Path(prefix)/"online_a_trace.json") if prefix else None
    if trace and sha(Path(prefix)/"online_a_trace.json") != query["prefix_trace_sha256"]:
        raise ValueError("Actual branch history changed")
    if trace:
        validate_leg1_trace(trace)
    try:
        base.srv_reset(camera_height=query["camera_height_m"], seed=query["seed"],
            episode_len=(len(trace["poses"]) if trace else 0) + base.args.max_steps,
            camera_intrinsic=np.asarray(query["camera_intrinsic"]),
            causal_history_sha256=query["prefix_trace_sha256"])
        replay = replay_online_a(dict(source=prefix, trace=trace),
            memory_step=base.srv_memory, navdp_replay_step=base.srv_navdp_memory_replay) if trace else None
        start = np.asarray(query["start_position"])
        if trace:
            np.testing.assert_allclose(start, trace["end_position"], atol=1e-8, rtol=0)
            assert query["start_yaw"] == trace["end_yaw"]
        target = np.asarray(query["floor_position"])
        ok, distance, _ = base.geodesic(simulator.pathfinder, start, target)
        if not ok or abs(distance-query["geodesic_m"]) > .05:
            raise ValueError("Construction/runtime geodesic differs")
        # The arm, NEVER the Novel/Revisit label, selects memory authority.
        backend = "navdp" if base.args.hybrid_route == "native_sidecar" else "navdp_auto"
        if base.args.hybrid_route not in ("native_sidecar", "certified_relocalization"):
            raise ValueError("Only the frozen native and GEM arms are in scope")
        leg = base.run_policy_leg(simulator, simulator.pathfinder, start, query["start_yaw"],
            Path(query["goal_rgb"]).read_bytes(), target[[0, 2]], distance, None,
            terminal_mode="off", goal_yaw=query["yaw_rad"],
            camera_intrinsic=np.asarray(query["camera_intrinsic"]), policy_backend=backend,
            success_dist=1., episode_seed=query["seed"], leg_index=query["stage_number"])
        actual = base.leg1_trace_payload(episode=query["episode"], episode_seed=query["seed"],
            goal_jpg=Path(query["goal_rgb"]).read_bytes(), goal_source_episode=query["episode"],
            source_scene=query["scene"], leg=leg)
        validate_leg1_trace(actual)
        dump(output / "actual_trace.json", actual)
        dump(output / "prefix_replay.json", replay)
        dump(output / "query_receipt.json", dict(runtime_query_sha256=sha(query_path), query=query))
    finally:
        simulator.close()


def query_command(source, query, output, mem_port, nav_port, arm):
    from MemNavData.run_repaired_fullmono_local import evaluator_command
    command = evaluator_command(source, output, mem_port, nav_port, arm=arm,
                                role="novel", benchmark=output.parent)
    command[2:4] = [str(HERE), "eval"]
    command[command.index("--seed")+1] = str(query["seed"])
    index = command.index("--role_pair_query_role")
    del command[index:index+2]
    return command


def pilot(args):
    from MemNavData.run_repaired_fullmono_local import (
        sources, private_servers, run_child, execution_environment, source_files, HAB_PY, hab_env,
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    source_rows = sources()
    manifest = dict(schema=SCHEMA, sources=source_rows, scope="two consumed scenes; new actual A; local interface pilot",
        stage_roles={"A": ["novel"], "B": list(ROLES), "C": list(ROLES)},
        goal_selection="first construction-valid proposal in predeclared order, no direction quota in local pilot",
        C_prefix_selection="native B successes with both C roles constructible; equal B-role sources before C evaluation",
        runtime="unchanged bounded_standard/rgb_v1/source_rgb/heading_on, canonical GEM",
        max_steps=600, execution_horizon=8, radius_m=1.,
        formal_population_frozen=False,
        local_protocol_sha256=sha(HERE.with_name("TABLE2_MIXED_LOCAL_PROTOCOL_20260911.md")),
        code_and_weights={str(p): sha(p) for p in source_files()})
    reused_a = {}
    if args.resume_a_root:
        original = args.resume_a_root.resolve()
        old_manifest = load(original / "manifest.json")
        old_summary = load(original / "summary.json")
        old_verifier = args.resume_a_verification.resolve()
        if not load(old_verifier)["verified"] or old_manifest["sources"] != source_rows:
            raise ValueError("Reused A must come from the verified, identical source population")
        if (not old_summary["completed"] or old_summary["changed_source_files"]
                or any(r["phase"] != "A" for r in old_summary["records"])):
            raise ValueError("Construction repair must precede all B/C navigation outcomes")
        permitted_edits = {str(HERE), str(HERE.with_name("test_table2_mixed_local.py"))}
        differences = [p for p, h in old_manifest["code_and_weights"].items()
                       if manifest["code_and_weights"].get(p) != h]
        if set(differences) - permitted_edits:
            raise ValueError(f"Runtime changed since A collection: {differences}")
        reused_a = {r["scene"]: r for r in old_summary["records"]}
        if set(reused_a) != {s["scene"] for s in source_rows}:
            raise ValueError("Retain every attempted A, including failures")
        manifest["reused_actual_A"] = dict(root=str(original), new_A_rollouts=0,
            manifest_sha256=sha(original/"manifest.json"), summary_sha256=sha(original/"summary.json"),
            verification=str(old_verifier), verification_sha256=sha(old_verifier),
            construction_and_logging_edits=differences)
    dump(out / "manifest.json", manifest)
    for p in source_files():
        if p.suffix in (".py", ".md"):
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    runs, attrition, c_sources = [], [], []

    def child_mode(mode, destination, extra):
        run_child([HAB_PY, "-u", str(HERE), mode, "--out", str(destination), *extra],
                  out / "logs" / f"{mode}_{len(list((out/'logs').iterdir())):03d}.log", environment=hab_env())

    # All A proposals are constructed before starting any navigation network.
    for rank, source in enumerate(source_rows):
        spec_path = out / source["scene"] / "source.json"
        if reused_a:
            dump(spec_path, load(args.resume_a_root / source["scene"] / "source.json"))
            continue
        # base_source needs pandas only, available in the supervisor environment.
        spec = base_source(source, rank)
        dump(spec_path, spec)
        child_mode("construct", spec_path.parent / "A_goals", ["--source", str(spec_path), "--stage", "A"])

    for source in source_rows:
        if reused_a:
            continue
        built = load(out / source["scene"] / "A_goals/construction.json")
        if "novel" in built["queries"]:
            query_path = Path(built["queries"]["novel"])
            query = load(query_path)
            command = query_command(source, query, out / "unused_dry_run", args.memnav_port, args.navdp_port, "native")
            run_child(command + ["--contract_dry_run"], out / "logs" / f"dry_run_{source['scene']}.log",
                      environment=dict(execution_environment(), TABLE2_QUERY=str(query_path.with_name("runtime.json"))))

    def run(query_file, source, phase, arm):
        q = load(query_file)
        destination = out / q["scene"] / "rollouts" / phase / q["analysis_role"] / arm
        print(f"START {q['scene']} {phase}/{q['analysis_role']}/{arm}", flush=True)
        pose_log = out / "lingbot_pose_readout.jsonl"
        pose_offset = pose_log.stat().st_size if pose_log.exists() else 0
        env = dict(execution_environment(), TABLE2_QUERY=str(Path(query_file).with_name("runtime.json")))
        wall = run_child(query_command(source, q, destination, args.memnav_port, args.navdp_port, arm),
                         out / "logs" / f"eval_{q['scene']}_{phase}_{q['analysis_role']}_{arm}.log", environment=env)
        with pose_log.open() as stream:
            stream.seek(pose_offset)
            dump(destination / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
        terminal, = load(destination / "terminal_measurements.json")
        row = dict(terminal, scene=q["scene"], episode=q["episode"], phase=phase,
                   role="goal_a" if phase == "A" else q["analysis_role"], arm=arm,
                   query_file=str(query_file), directory=str(destination), wall_seconds=wall)
        dump(destination / "result.json", row)
        runs.append(row)
        dump(out / "progress" / f"{len(runs):03d}.json", row)
        print(f"DONE {q['scene']} {phase}/{q['analysis_role']}/{arm}: SR={row['reached']} SPL={row['spl']:.3f} steps={row['steps']}", flush=True)
        return row

    with private_servers(out, args.memnav_port, args.navdp_port):
        for rank, source in enumerate(source_rows):
            scene_root = out / source["scene"]
            source_path = scene_root / "source.json"
            if reused_a:
                a_row = reused_a[source["scene"]]
                runs.append(a_row)
                dump(out / "progress" / f"{len(runs):03d}.json", a_row)
                print(f"REUSE actual A {source['scene']}: SR={a_row['reached']}", flush=True)
            else:
                a_construct = load(scene_root / "A_goals/construction.json")
                if "novel" not in a_construct["queries"]:
                    attrition.append(dict(scene=source["scene"], stage="A", reason="A_not_constructible"))
                    continue
                a_row = run(a_construct["queries"]["novel"], source, "A", "native")
            if not a_row["reached"]:
                attrition.append(dict(scene=source["scene"], stage="B", reason="actual_A_failed"))
                continue
            if reused_a:
                a_prefix = args.resume_a_root.resolve() / source["scene"] / "prefix_A"
            else:
                a_prefix = scene_root / "prefix_A"
                child_mode("materialize", a_prefix, ["--source", str(source_path), "--rollout", a_row["directory"]])
            b_goals = scene_root / "B_goals"
            child_mode("construct", b_goals, ["--source", str(source_path), "--stage", "B", "--prefix", str(a_prefix)])
            b_construct = load(b_goals / "construction.json")
            if reused_a:
                old_goals = load(args.resume_a_root / source["scene"] / "B_goals/construction.json")
                # This repair must not silently replace the already selected Novel.
                for role in ("novel",):
                    if role in old_goals["queries"]:
                        old_goal = load(old_goals["queries"][role])
                        new_goal = load(b_construct["queries"][role])
                        for key in ("goal_rgb_sha256", "floor_position", "yaw_rad", "prefix_trace_sha256"):
                            assert new_goal[key] == old_goal[key], key
            if not b_construct["both_constructed"]:
                attrition.append(dict(scene=source["scene"], stage="B", reason="B_role_pair_not_constructible"))
                continue
            for role_index, role in enumerate(ROLES):
                order = ("native", "cec") if (rank+role_index)%2 == 0 else ("cec", "native")
                results = {arm: run(b_construct["queries"][role], source, "B", arm) for arm in order}
                native = results["native"]
                c_source = dict(scene=source["scene"], collector="native", b_role=role,
                                b_reached=bool(native["reached"]), both_c_constructed=False)
                if native["reached"]:
                    prefix = scene_root / f"prefix_AB_{role}"
                    child_mode("materialize", prefix, ["--source", str(source_path), "--rollout", native["directory"], "--prefix", str(a_prefix)])
                    goals = scene_root / f"C_after_{role}_goals"
                    child_mode("construct", goals, ["--source", str(source_path), "--stage", "C", "--prefix", str(prefix)])
                    built = load(goals / "construction.json")
                    c_source.update(both_c_constructed=built["both_constructed"], queries=built["queries"], prefix=str(prefix))
                else:
                    attrition.append(dict(scene=source["scene"], stage="C", b_role=role, reason="native_B_failed"))
                c_sources.append(c_source)
        selected = balanced_c_sources(c_sources)
        dump(out / "c_population_before_eval.json", dict(all_sources=c_sources, selected=selected,
              C_navigation_outcomes_read=False, equal_B_role_sources=True))
        for rank, c in enumerate(selected):
            source = next(s for s in source_rows if s["scene"] == c["scene"])
            for j, role in enumerate(ROLES):
                for arm in (("native", "cec") if (rank+j)%2 == 0 else ("cec", "native")):
                    run(c["queries"][role], source, "C_after_"+c["b_role"], arm)
    changed = [p for p, h in manifest["code_and_weights"].items() if sha(p) != h]
    dump(out / "summary.json", dict(schema=SCHEMA, completed=True, records=runs,
          attrition=attrition, C_sources=c_sources, selected_C_sources=selected,
          changed_source_files=changed, formal_result=False))
    child_mode("verify", out / "verification", ["--run-root", str(out)])


def verify_local(root, out):
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(root / "manifest.json"), load(root / "summary.json")
    assert summary["completed"] and not summary["changed_source_files"]
    records, pairs = [], []
    grouped = {}
    for row in summary["records"]:
        result, evidence, plans, actions = verify_rollout(row)
        query = load(row["query_file"])
        folder = Path(row["directory"])
        assert load(folder/"query_receipt.json")["query"] == runtime_spec(query)
        if query["prefix_root"]:
            trace = load(Path(query["prefix_root"])/"online_a_trace.json")
            assert sha(Path(query["prefix_root"])/"online_a_trace.json") == query["prefix_trace_sha256"]
            assert evidence["rollout_trace"][0]["yaw"] == trace["end_yaw"]
            np.testing.assert_allclose([evidence["rollout_trace"][0][k] for k in "xyz"], trace["end_position"], atol=1e-8, rtol=0)
            replay = load(folder/"prefix_replay.json")
            assert replay["online_frames"] == len(trace["poses"]) and replay["diffusion_samples_during_replay"] == 0
        if row["phase"] != "A":
            grouped.setdefault((row["scene"], row["phase"], row["role"]), {})[row["arm"]] = (row, plans, actions)
        records.append(result)
    for key, arms in grouped.items():
        assert set(arms) == {"native", "cec"}
        n, g = arms["native"], arms["cec"]
        assert n[0]["query_file"] == g[0]["query_file"]
        for filename in ("prefix_replay.json", "query_receipt.json"):
            assert load(Path(n[0]["directory"])/filename) == load(Path(g[0]["directory"])/filename)
        assert n[0]["first_query_rgb_sha256"] == g[0]["first_query_rgb_sha256"]
        takeover = any(p["receipt"]["revisit_adapter_takeover"] is True for p in g[1])
        if not takeover:
            assert len(n[1]) == len(g[1]) and len(n[2]) == len(g[2])
            for x, y in zip(n[1], g[1]):
                for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                    np.testing.assert_array_equal(x[field], y[field])
            for x, y in zip(n[2], g[2]):
                for field in ("actual_position", "actual_yaw", "action_kind"):
                    assert x[field] == y[field]
            depth = []
            for item in (n, g):
                http = read_rows(Path(item[0]["directory"])/"navdp_http_receipts.jsonl")
                depth.append([r["monocular_depth_receipt"] for r in http if r["query_active"] and r["audit"]["image_calls"] and r["path"]!="/memory_replay_step"])
            assert len(depth[0]) == len(depth[1]) == len(n[1])
            for x, y in zip(*depth):
                for field in ("depth_png_sha256", "image_sha256", "scale_receipt_sha256", "frame_index"):
                    assert x[field] == y[field]
        pairs.append(dict(scene=key[0], phase=key[1], role=key[2], native=n[0]["reached"],
                          cec=g[0]["reached"], takeover=takeover, reject_exact_native=not takeover))
    c = load(root/"c_population_before_eval.json")
    assert c["selected"] == balanced_c_sources(c["all_sources"])
    for row in c["all_sources"]:
        if not row["b_reached"]:
            assert not row["both_c_constructed"]
            continue
        prefix = Path(row["prefix"])
        receipt = load(prefix/"receipt.json")
        a = load(Path(receipt["parent_prefix"])/"online_a_trace.json")
        b = load(Path(receipt["source_rollout"])/"actual_trace.json")
        assert load(prefix/"online_a_trace.json") == compose_prefix(a, b)
        assert Path(receipt["source_rollout"]).parts[-2:] == (row["b_role"], "native")
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"independent_verification.json", dict(verified=True, records=records, pairs=pairs,
         C_reference_sources=len(c["selected"]), exact_branch_composition_checked=True,
         verifier_sha256=sha(HERE),
         formal_population_frozen=manifest["formal_population_frozen"]))


def main():
    if len(sys.argv)>1 and sys.argv[1] == "eval":
        del sys.argv[1]
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate("role_pair", query_main=evaluate_one)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("pilot", "construct", "materialize", "verify"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--stage", choices=("A", "B", "C"))
    p.add_argument("--prefix", type=Path)
    p.add_argument("--rollout", type=Path)
    p.add_argument("--run-root", type=Path)
    p.add_argument("--resume-a-root", type=Path)
    p.add_argument("--resume-a-verification", type=Path)
    p.add_argument("--memnav-port", type=int, default=19781)
    p.add_argument("--navdp-port", type=int, default=19782)
    a = p.parse_args()
    if a.resume_a_root is not None and a.resume_a_verification is None:
        p.error("--resume-a-root requires --resume-a-verification")
    if a.mode == "pilot":
        pilot(a)
    elif a.mode == "construct":
        construct_query(load(a.source), stage=a.stage, prefix=a.prefix, out=a.out)
    elif a.mode == "materialize":
        materialize_prefix(a.rollout, a.prefix, a.out, load(a.source))
    else:
        verify_local(a.run_root, a.out)


if __name__ == "__main__":
    main()
