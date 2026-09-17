"""Local continuous A/B/C lifecycle smoke with the unchanged repaired runtime.

This is NOT the formal paired Table II. A is selected before navigation;
subsequent targets are constructed from this one arm's actual continuous
history using the existing rules. Its purpose is to test state continuity,
goal switching and task supply, not compare differently generated tasks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from MemNavData.table2_mixed_local import ROOT, load, dump, sha
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA, role_cells
from MemNavData.table2_continuous_chain import run_chain

HERE = Path(__file__).resolve()
SCHEMA = "table2_continuous_lifecycle_local_20260911_v1"
CONSUMED_SOURCE = ROOT / ".diagnostics/table2_mixed_actual_local_20260911_v3/pLe4wQe7qrG/source.json"
DECLARATION = ROOT / ".diagnostics/table2_forward_only_audit_20260911_v1/source_declaration.json"


class NoConstructibleGoal(RuntimeError):
    pass


def choose(menu, role, key):
    from MemNavData.table2_balanced_sampling import balanced_assignment
    selected = balanced_assignment([dict(source_id="local_lifecycle", candidates=menu["menus"][role])],
                                   key=key, cells=role_cells(FORWARD_SCHEMA, role))["selected"]
    if not selected:
        raise NoConstructibleGoal(f"No {role} goal within the unchanged construction budget")
    return load(selected[0]["query"])


def materialize_own_prefix(base, sim, source, traces, out):
    """Offline task-construction copy only. Never replay it into either server."""
    from MemNavData.build_shared_online_double_revisit import write_depth_png
    from MemNavData.hm3d_table2_leg3_mixed_role import compose_actual_ab_trace
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    if len(traces) not in (1, 2):
        raise ValueError("Only own A or own A+B is needed before the next goal")
    if len({t["source_hybrid_route"] for t in traces}) != 1:
        raise ValueError("Cannot splice histories from different methods")
    trace = deepcopy(traces[0]) if len(traces) == 1 else compose_actual_ab_trace(
        traces[0], traces[1], episode=source["episode"])
    trace["prefix_semantics"] = "own_continuous_observations_offline_construction_only"
    validate_leg1_trace(trace)
    out.mkdir(parents=True, exist_ok=False)
    (out / "rgb").mkdir()
    (out / "depth").mkdir()
    for pose in trace["poses"]:
        camera = np.asarray([pose[k] for k in "xyz"]) + [0., source["camera_height_m"], 0.]
        rgb, depth = base.render(sim, camera, pose["yaw"])
        encoded = base.jpg_bytes(rgb)
        if hashlib.sha256(encoded).hexdigest() != pose["jpg_sha256"]:
            raise ValueError("Actual RGB could not be reproduced for offline task construction")
        (out / "rgb" / f"{pose['step']:06d}.jpg").write_bytes(encoded)
        write_depth_png(out / "depth" / f"{pose['step']:06d}.png", depth)
    dump(out / "online_a_trace.json", trace)
    dump(out / "receipt.json", dict(online_a_steps=len(trace["poses"]),
        camera_height_m=source["camera_height_m"], camera_intrinsic=source["camera_intrinsic"],
        online_a_trace_sha256=sha(out / "online_a_trace.json"),
        history_source="own_actual_continuous_policy", construction_only=True,
        replayed_into_runtime=False, source_asset=source["asset"]))
    return out


def evaluate_chain(*, goal_provider=None, after_leg=None):
    import eval_2leg_habitat as base
    from MemNavData.table2_balanced_sampling import construct
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    from MemNavData.final14_spl_replay import measurement

    manifest = load(os.environ["TABLE2_CONTINUOUS_MANIFEST"])
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported continuous lifecycle")
    source = manifest["source"]
    common = goal_provider is not None
    if common != bool(manifest.get("common_online_goals", False)):
        raise ValueError("The goal provider must match the declared episode protocol")
    formal = bool(manifest["formal_population"])
    if formal and (not common or not manifest.get("population_sha256") or "task_index" not in manifest):
        raise ValueError("Formal execution requires the frozen common-goal coordinator")
    if base.args.contract_dry_run:
        print("CONTINUOUS_DRY_RUN: one reset, own-state A/B/C, no prefix replay", flush=True)
        return
    out = Path(base.args.out)
    sim = base.make_sim(source["asset"], "", agent_radius=.30)
    traces, selected, observed = [], [], []
    base.CAM_H = float(source["camera_height_m"])
    backend = "navdp" if base.args.hybrid_route == "native_sidecar" else "navdp_auto"
    if base.args.hybrid_route not in ("native_sidecar", "certified_relocalization"):
        raise ValueError("Only the unchanged native/GEM routes are supported")
    try:
        # Exactly one reset for the whole chain. No reset or replay at B/C.
        base.srv_reset(camera_height=source["camera_height_m"], seed=source["seed"],
            episode_len=3 * base.args.max_steps, camera_intrinsic=np.asarray(source["camera_intrinsic"]))

        def execute(index, stage, position, yaw):
            role = "novel" if manifest["sequence"][index] == "N" else "revisit"
            if index == 0:
                query = load(manifest["a_query"])
                if sha(manifest["a_query"]) != manifest["a_query_sha256"]:
                    raise ValueError("Preselected A changed")
                if common:
                    query = goal_provider(index, stage, None, query, position, yaw)
            else:
                prefix = materialize_own_prefix(base, sim, source, traces, out / f"history_before_{stage}")
                if common:
                    query = goal_provider(index, stage, prefix, None, position, yaw)
                else:
                    menu = construct(source, stage=stage, prefix=prefix, out=out / f"goals_{stage}")
                    query = choose(menu, role, f"continuous_local/{stage}/{role}")
                # A pending alignment belongs to the previous goal, not to
                # the geometry/history state. With distance-only arrival it
                # should already be inactive; do not hide a lifecycle fault.
                adapter = getattr(base, "_front_goal_adapter", None)
                if adapter is not None and adapter.active:
                    raise RuntimeError("Previous goal ended during an unfinished heading action")
            np.testing.assert_allclose(query["start_position"], position, atol=1e-8, rtol=0)
            if abs(float(query["start_yaw"]) - yaw) > 1e-8:
                raise ValueError("Task construction changed the actual yaw")
            goal = Path(query["goal_rgb"]).read_bytes()
            if hashlib.sha256(goal).hexdigest() != query["goal_rgb_sha256"]:
                raise ValueError("Goal image changed")
            target = np.asarray(query["floor_position"])
            ok, geo, _ = base.geodesic(sim.pathfinder, position, target)
            if not ok or abs(geo - query["geodesic_m"]) > .05:
                raise ValueError("Actual task geometry differs from construction")
            selected.append(query)
            dump(out / f"selected_{stage}.json", query)
            print(f"START {stage}: role hidden; distance={geo:.3f}m angle={query['initial_relative_route_angle_deg']:.1f}deg", flush=True)
            return base.run_policy_leg(sim, sim.pathfinder, position, yaw, goal,
                target[[0, 2]], geo, None, terminal_mode="off", goal_yaw=query["yaw_rad"],
                camera_intrinsic=np.asarray(source["camera_intrinsic"]), policy_backend=backend,
                success_dist=1., episode_seed=source["seed"], leg_index=index)

        def observe(index, stage, leg, continuity):
            query = selected[index]
            trace = base.leg1_trace_payload(episode=source["episode"], episode_seed=source["seed"],
                goal_jpg=Path(query["goal_rgb"]).read_bytes(), goal_source_episode=source["episode"],
                source_scene=source["scene"], leg=leg)
            validate_leg1_trace(trace)
            measured = measurement(leg, np.asarray(query["floor_position"])[[0, 2]], query["geodesic_m"])
            dump(out / f"leg_{stage}/actual_trace.json", trace)
            dump(out / f"leg_{stage}/measurement.json", measured)
            dump(out / f"leg_{stage}/continuity.json", continuity)
            traces.append(trace)
            observed.append(dict(stage=stage, continuity=continuity, measurement=measured))
            print(f"DONE {stage}: reached={leg['reached']} steps={leg['steps']} memory={continuity['first_memory_index']}..{continuity['last_memory_index']}", flush=True)
            if after_leg is not None:
                after_leg(index, stage, continuity, measured)

        try:
            result = run_chain(source["start_position"], source["start_yaw"], "ABC", execute, observe)
            result.update(status="lifecycle_finished", formal_population=formal,
                          paired_navigation_comparison=common, measurements=observed)
        except NoConstructibleGoal as exc:
            result = dict(status="task_construction_blocked", reason=str(exc),
                          formal_population=formal, paired_navigation_comparison=common,
                          measurements=observed, no_navigation_failure_imputed=True)
        dump(out / "chain_summary.json", result)
    finally:
        sim.close()


def run_local(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, private_servers, run_child, hab_env, evaluator_command, execution_environment,
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    source_path = CONSUMED_SOURCE if args.source_index is None else DECLARATION
    source = (load(source_path) if args.source_index is None else
              deepcopy(load(source_path)["sources"][args.source_index]["initial_state"]))
    source["schema"] = FORWARD_SCHEMA
    dump(out / "source.json", source)
    run_child([HAB_PY, "-u", "-m", "MemNavData.table2_balanced_sampling", "construct",
               "--source", str(out / "source.json"), "--stage", "A", "--out", str(out / "A_menu")],
              out / "logs/construct_A.log", environment=hab_env())
    query = choose(load(out / "A_menu/construction.json"), "novel", "continuous_local/A/novel")
    dump(out / "A_query.json", query)
    manifest = dict(schema=SCHEMA, scope="single-arm local lifecycle and supply smoke, NOT formal SR",
        formal_population=False, paired_navigation_comparison=False, source=source,
        sequence=args.sequence, arm=args.arm, a_query=str(out / "A_query.json"),
        a_query_sha256=sha(out / "A_query.json"),
        successor_selection="same frozen rules on own actual history; no cross-arm comparison",
        runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on",
        max_steps_per_leg=600, execution_horizon=8, success_radius_m=1.,
        source_path=str(source_path), declared_source_index=args.source_index,
        source_sha256=sha(source_path), code_sha256={str(p): sha(p) for p in (
            HERE, HERE.with_name("table2_continuous_chain.py"), HERE.with_name("table2_balanced_sampling.py"),
            HERE.with_name("run_habitat_minimal_repair_local.py"), HERE.with_name("navdp_front_goal_adapter.py"))})
    dump(out / "manifest.json", manifest)
    command = evaluator_command(source, out / "evaluation", args.memnav_port, args.navdp_port,
                                arm=args.arm, role="novel", benchmark=out)
    command[2:4] = [str(HERE), "eval"]
    at = command.index("--role_pair_query_role")
    del command[at:at+2]
    command[command.index("--leg1_mode")+1] = "policy"
    env = dict(execution_environment(), TABLE2_CONTINUOUS_MANIFEST=str(out / "manifest.json"))
    run_child(command + ["--contract_dry_run"], out / "logs/cli_preflight.log", environment=env)
    if args.prepare_only:
        print(f"PREPARED {out}; no model servers or navigation started", flush=True)
        return
    with private_servers(out, args.memnav_port, args.navdp_port):
        run_child(command, out / "logs/evaluation.log", environment=env)
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        del sys.argv[1]
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate("role_pair", query_main=evaluate_chain)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--sequence", choices=("NNN", "NNR", "NRN", "NRR"), default="NRR")
        parser.add_argument("--arm", choices=("native", "cec"), default="native")
        parser.add_argument("--memnav-port", type=int, default=21981)
        parser.add_argument("--navdp-port", type=int, default=21982)
        parser.add_argument("--prepare-only", action="store_true")
        parser.add_argument("--source-index", type=int, help="Local smoke from the existing declaration; no source-pool change")
        run_local(parser.parse_args())
