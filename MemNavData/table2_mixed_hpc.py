"""Staged Table-II integration: actual A, paired B, global C mix, paired C.

This orchestrator reuses the tested local constructor and repaired controller.
It does not choose targets by navigation outcomes or define a formal sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re

import numpy as np

from MemNavData.table2_mixed_local import (
    HERE as LOCAL, ROOT, ROLES, base_source, compose_prefix, dump, load,
    query_command, runtime_spec, sha,
)
from MemNavData.table2_novel_sampling import balanced_c_sources
from MemNavData.table2_sampling_profiles import (
    FORMAL_SCHEMAS, FORWARD_SCHEMA, protocol_path, verify_query_direction,
)

HERE = Path(__file__).resolve()
PROTOCOL = HERE.with_name("TABLE2_MIXED_HPC_PILOT_PROTOCOL_20260911.md")
SCHEMA = "table2_mixed_staged_pilot_20260911_v1"


def formal(plan):
    return plan["schema"] in FORMAL_SCHEMAS


def protocol_for(plan):
    return protocol_path(plan["schema"]) if formal(plan) else PROTOCOL


def source_id(source):
    parts = (source["scene"], source["episode"])
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in parts):
        raise ValueError("Source identity must be a safe scene/episode pair")
    return "/".join(parts)


def validate_plan(plan):
    if (plan["schema"] not in (SCHEMA, *FORMAL_SCHEMAS)
            or bool(plan["formal_result"]) != formal(plan)):
        raise ValueError("Unknown or incorrectly labelled Table-II protocol")
    ids = [source_id(s) for s in plan["sources"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene/episode source")
    for i, source in enumerate(plan["sources"]):
        if source["index"] != i or source["source_id"] != ids[i]:
            raise ValueError("Plan identity/order mismatch")
        if plan["schema"] == FORWARD_SCHEMA and source.get("initial_state", {}).get("schema") != FORWARD_SCHEMA:
            raise ValueError("Forward-Novel source must carry the new construction profile")
    if not re.fullmatch(r"/tmp/memnav_table2_[A-Za-z0-9_-]+", plan["work_root"]):
        raise ValueError("Use an experiment-specific virtual node-local root")
    if plan["arms"] != ["native", "cec"] or plan["c_selection"] != "global_equal_native_B_roles":
        raise ValueError("Method or reference-prefix contract changed")
    return plan


def prepare_plan(parent_path, out, runtime_sha, *, scenes=4, per_scene=2,
                 work_root="/tmp/memnav_table2_pilot_20260911_v1"):
    parent = load(parent_path)
    selected_scenes = list(dict.fromkeys(s["scene"] for s in parent["sources"]))[:scenes]
    sources = []
    for scene in selected_scenes:
        group = [s for s in parent["sources"] if s["scene"] == scene][:per_scene]
        if len(group) != per_scene:
            raise ValueError("Insufficient predeclared source carriers")
        for s in group:
            # Reuse ONLY first-pose/calibration carriers and scene assets.
            # The old goal photo, old A outcome and expert route are not used.
            files = {s["task_files"][k]["path"]: s["task_files"][k]["sha256"]
                     for k in ("metadata", "parquet")}
            files.update({s["asset"]["glb_path"]: s["asset"]["glb_sha256"],
                          s["asset"]["navmesh_path"]: s["asset"]["navmesh_sha256"]})
            sources.append(dict(index=len(sources), source_id=source_id(s),
                scene=s["scene"], episode=s["episode"], seed=2026091100 + s["source_index"],
                seed_rank=s["source_index"], asset=s["asset"]["glb_path"],
                source_episode=s["source_episode"], source_files=files))
    if len(selected_scenes) != scenes:
        raise ValueError("Insufficient predeclared scenes")
    plan = dict(schema=SCHEMA, formal_result=False, sources=sources, arms=["native", "cec"],
        source_selection="first declared scenes; first declared carriers per scene; no old SR read",
        parent_source_plan_sha256=sha(parent_path), runtime_sha256=runtime_sha,
        protocol_sha256=sha(PROTOCOL), work_root=work_root,
        c_selection="global_equal_native_B_roles", c_navigation_before_selection=False,
        source_budget=len(sources), runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on",
        goal_rules="unchanged local constructor: 2--9 m; N<0.10; R in [0.55,0.90]",
        matching="supply diagnostic only; no formal distance/direction balance claimed",
        success="evaluator planar distance <1 m; final action position recorded; not autonomous STOP")
    validate_plan(plan)
    dump(out, plan)
    return plan


def runtime_provenance(plan):
    """Keep the scientific plan immutable while recording a sealed transport repair."""
    actual = sha(ROOT / "SOURCE_BUNDLE.sha256")
    planned = plan["runtime_sha256"]
    result = dict(runtime_sha256=actual, planned_runtime_sha256=planned)
    if actual != planned:
        path = ROOT / "multipart_repair_manifest.json"
        repair = load(path)
        if (repair["original_bundle_sha256"] != planned
                or repair["model_parameters_changed"] is not False
                or repair["controller_changed"] is not False
                or repair["jpeg_validation_removed"] is not False):
            raise ValueError("Runtime differs without an authorized transport repair")
        result["transport_repair_manifest_sha256"] = sha(path)
    return result


def task_root(plan, stage, index):
    if stage not in ("A", "B", "C") or index < 0:
        raise ValueError("Invalid stage/index")
    return Path(plan["work_root"]) / stage / f"task_{index:03d}" / "task"


def durable_root(run, stage, index):
    return Path(run) / stage / f"task_{index:03d}"


def task_source(plan, stage, index, population=None):
    if index < 0:
        raise ValueError("Negative task index")
    if stage == "A":
        source_index, role = index, "novel"
    elif stage == "B":
        source_index, j = divmod(index, 2)
        role = ROLES[j]
    elif stage == "C":
        c_index, j = divmod(index, 2)
        entry = population["selected"][c_index]
        source_index, role = entry["source_index"], ROLES[j]
    else:
        raise ValueError(stage)
    source = plan["sources"][source_index]
    return source, role


def arm_order(index):
    # Alternate within EACH query role, not only over the interleaved array.
    parent_rank, role_rank = divmod(index, 2)
    return ("native", "cec") if (parent_rank + role_rank) % 2 == 0 else ("cec", "native")


def read_completed(run, stage, index, plan_sha):
    folder = durable_root(run, stage, index)
    receipt = load(folder / "archive_receipt.json")
    summary = load(folder / "summary.json")
    verifier = load(folder / "independent_verification.json")
    if not (receipt["completed"] and receipt["all_member_hashes_readback_verified"]
            and summary["completed"] and verifier["verified"]):
        raise ValueError(f"Incomplete predecessor {stage}/{index}")
    if (summary["plan_sha256"] != plan_sha or summary["stage"] != stage
            or summary["index"] != index or verifier["summary_sha256"] != sha(folder/"summary.json")):
        raise ValueError("Predecessor identity or verification mismatch")
    return summary


def select_c(plan_path, run, out):
    plan = validate_plan(load(plan_path))
    plan_sha = sha(plan_path)
    candidates, inputs = [], []
    for index in range(2 * len(plan["sources"])):
        summary = read_completed(run, "B", index, plan_sha)
        candidate = summary["c_source"]
        expected, role = task_source(plan, "B", index)
        if candidate["source_id"] != expected["source_id"] or candidate["b_role"] != role:
            raise ValueError("Global C source has the wrong parent identity")
        candidates.append(candidate)
        path = durable_root(run, "B", index) / "summary.json"
        inputs.append(dict(path=str(path), sha256=sha(path)))
    # This runs ONCE across the entire cohort, not once per source/array cell.
    if formal(plan):
        from MemNavData.table2_formal_population import assign_c_queries
        selected, assignments = assign_c_queries(candidates, schema=plan["schema"])
    else:
        selected = balanced_c_sources(candidates)
        assignments = None
    result = dict(schema=plan["schema"], plan_sha256=plan_sha, completed=True, formal_result=formal(plan),
        all_sources=candidates, selected=selected, input_summaries=inputs,
        query_assignments=assignments,
        C_navigation_outcomes_read=False, equal_B_role_sources=True,
        counts=dict(Counter(s["b_role"] for s in selected)), c_query_tasks=2*len(selected))
    dump(out, result)
    return result


def population_at(plan_path, run):
    population = load(Path(run) / "c_population.json")
    if population["plan_sha256"] != sha(plan_path) or population["C_navigation_outcomes_read"]:
        raise ValueError("C population must precede C navigation")
    if formal(load(plan_path)):
        from MemNavData.table2_formal_population import assign_c_queries
        expected, assignments = assign_c_queries(population["all_sources"], schema=load(plan_path)["schema"])
        if assignments != population["query_assignments"]:
            raise ValueError("C query assignment changed")
    else:
        expected = balanced_c_sources(population["all_sources"])
    if population["selected"] != expected:
        raise ValueError("C source mix/order changed")
    for item in population["input_summaries"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("B result changed after global C selection")
    return population


def run_task(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, execution_environment, hab_env, private_servers, run_child,
    )
    from MemNavData.table2_task_storage import restore_task

    plan = validate_plan(load(args.plan))
    if plan["protocol_sha256"] != sha(protocol_for(plan)):
        raise ValueError("Protocol changed")
    population = population_at(args.plan, args.run) if args.stage == "C" else None
    source, role = task_source(plan, args.stage, args.index, population)
    out = task_root(plan, args.stage, args.index)
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    os.environ.update(REPAIRED_BUFFER_ROOT=str(out/"buffer"), REPAIRED_RUNTIME_ROOT=str(out/"runtime"))
    manifest = dict(schema=plan["schema"], source=source, stage=args.stage, index=args.index,
                    plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
                    run_root=str(args.run.resolve()),
                    **runtime_provenance(plan), formal_result=formal(plan))
    dump(out / "manifest.json", manifest)
    records, attrition = [], []
    summary = dict(schema=plan["schema"], stage=args.stage, index=args.index,
        source_id=source["source_id"], plan_sha256=sha(args.plan), completed=True,
        formal_result=formal(plan), records=records, attrition=attrition)

    def child(mode, destination, extra):
        entry = HERE.with_name("table2_balanced_sampling.py") if formal(plan) and mode == "construct" else LOCAL
        run_child([HAB_PY, "-u", str(entry), mode, "--out", str(destination), *map(str, extra)],
                  out / "logs" / f"{mode}_{len(list((out/'logs').iterdir())):03d}.log", environment=hab_env())

    def restore(stage, index):
        completed = read_completed(args.run, stage, index, sha(args.plan))
        target = task_root(plan, stage, index)
        restore_task(durable_root(args.run, stage, index)/"archive_receipt.json", target)
        if load(target/"summary.json") != completed:
            raise ValueError("Restored predecessor summary differs")
        return completed, target

    def run(query_file, phase, arm):
        q = load(query_file)
        if (q["scene"], q["episode"], q["seed"]) != (source["scene"], source["episode"], source["seed"]):
            raise ValueError("Query/source identity mismatch")
        destination = out / "rollouts" / source["source_id"] / phase / q["analysis_role"] / arm
        poses = out / "lingbot_pose_readout.jsonl"
        offset = poses.stat().st_size if poses.exists() else 0
        env = dict(execution_environment(), TABLE2_QUERY=str(Path(query_file).with_name("runtime.json")))
        cmd = query_command(source, q, destination, args.mem_port, args.nav_port, arm)
        print(f"START {source['source_id']} {phase}/{q['analysis_role']}/{arm}", flush=True)
        wall = run_child(cmd, out / "logs" / f"{phase}_{q['analysis_role']}_{arm}.log", environment=env)
        with poses.open() as stream:
            stream.seek(offset)
            import json
            dump(destination/"lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
        terminal, = load(destination / "terminal_measurements.json")
        row = dict(terminal, scene=q["scene"], episode=q["episode"], source_id=source["source_id"],
            phase=phase, role="goal_a" if phase == "A" else q["analysis_role"], arm=arm,
            query_file=str(query_file), directory=str(destination), wall_seconds=wall,
            geometry={k: q.get(k) for k in ("geodesic_m", "straight_distance_m", "route_ratio",
                "direction_stratum", "initial_relative_route_angle_deg", "current_view_covis",
                "max_history_covis", "max_runtime_eligible_covis", "history_frames",
                "max_A_covis", "max_B_covis", "support_source")})
        dump(destination / "result.json", row)
        records.append(row)
        dump(out / "progress" / f"{len(records):03d}.json", row)
        print(f"DONE {source['source_id']} {phase}/{arm}: reached={row['reached']} steps={row['steps']}", flush=True)
        return row

    def paired(query_file, phase):
        with private_servers(out, args.mem_port, args.nav_port):
            return {arm: run(query_file, phase, arm) for arm in arm_order(args.index)}

    if args.stage == "A":
        spec = base_source(source, source["seed_rank"])
        spec_path = out / "source.json"
        dump(spec_path, spec)
        if formal(plan):
            selected = source["a_selection"]
            if selected and sha(selected["query"]) != selected["query_sha256"]:
                raise ValueError("Frozen A query changed")
            dump(out/"A_goals/construction.json", dict(
                queries={"novel": selected["query"]} if selected else {},
                selected_before_any_A_rollout=True, selection=selected))
        else:
            child("construct", out/"A_goals", ["--source", spec_path, "--stage", "A"])
        goals = load(out/"A_goals/construction.json")
        summary["b_construction"] = None
        if "novel" not in goals["queries"]:
            attrition.append("A_not_constructible")
        else:
            with private_servers(out, args.mem_port, args.nav_port):
                a = run(goals["queries"]["novel"], "A", "native")
            if a["reached"]:
                child("materialize", out/"prefix_A", ["--source", spec_path, "--rollout", a["directory"]])
                child("construct", out/"B_goals", ["--source", spec_path, "--stage", "B", "--prefix", out/"prefix_A"])
                summary["b_construction"] = load(out/"B_goals/construction.json")
            else:
                attrition.append("actual_A_failed")
    elif args.stage == "B":
        a, a_root = restore("A", source["index"])
        candidate = dict(source_id=source["source_id"], source_index=source["index"],
            b_task_index=args.index, scene=source["scene"], episode=source["episode"],
            collector="native", b_role=role, b_reached=False, both_c_constructed=False)
        summary["c_source"] = candidate
        built = a["b_construction"]
        if formal(plan):
            from MemNavData.table2_formal_population import b_population_at
            bpop = b_population_at(args.plan, args.run)
            selections = bpop["selected"].get(source["source_id"])
            summary["b_population_sha256"] = sha(Path(args.run)/"b_population.json")
            if selections:
                selected = selections[role]
                menu_entry = {k: v for k, v in selected.items() if k != "source_id"}
                if menu_entry not in built["menus"][role] or sha(selected["query"]) != selected["query_sha256"]:
                    raise ValueError("B selection is not a frozen A-produced candidate")
                query_file = selected["query"]
            else:
                query_file = None
        else:
            query_file = built["queries"][role] if built and built["both_constructed"] else None
        if query_file is None:
            attrition.append("A_failed_or_B_role_pair_not_constructible")
        else:
            results = paired(query_file, "B")
            native = results["native"]
            candidate["b_reached"] = bool(native["reached"])
            if native["reached"]:
                prefix = out / f"prefix_AB_{role}"
                child("materialize", prefix, ["--source", a_root/"source.json", "--rollout", native["directory"], "--prefix", a_root/"prefix_A"])
                child("construct", out/"C_goals", ["--source", a_root/"source.json", "--stage", "C", "--prefix", prefix])
                built = load(out/"C_goals/construction.json")
                candidate.update(both_c_constructed=built["both_constructed"], prefix=str(prefix))
                if formal(plan):
                    candidate["menus"] = built["menus"]
                else:
                    candidate["queries"] = built["queries"]
            else:
                attrition.append("native_B_failed")
    else:
        chosen = population["selected"][args.index//2]
        restore("A", source["index"])
        b, _ = restore("B", chosen["b_task_index"])
        matched = (all(chosen.get(k) == v for k, v in b["c_source"].items())
                   if formal(plan) else b["c_source"] == chosen)
        if not matched:
            raise ValueError("C does not reference the sealed actual B branch")
        summary["c_source"] = chosen
        summary["c_population_sha256"] = sha(Path(args.run)/"c_population.json")
        paired(chosen["queries"][role], "C_after_"+chosen["b_role"])
    dump(out / "summary.json", summary)
    run_child([HAB_PY, str(HERE), "verify", "--out", str(out)], out/"logs/verify.log", environment=hab_env())


def verify_task(out):
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out/"manifest.json"), load(out/"summary.json")
    assert summary["completed"] and summary["plan_sha256"] == sha(manifest["plan"])
    source = manifest["source"]
    assert summary["source_id"] == source_id(source)
    plan = validate_plan(load(manifest["plan"]))
    assert source == plan["sources"][source["index"]]
    assert summary["stage"] == manifest["stage"] and summary["index"] == manifest["index"]
    if manifest["stage"] == "A":
        goals = load(out/"A_goals/construction.json")
        assert len(summary["records"]) == int("novel" in goals["queries"])
    elif manifest["stage"] == "B":
        parent = load(task_root(plan, "A", source["index"])/"summary.json")
        built = parent["b_construction"]
        assert len(summary["records"]) == (2 if built and built["both_constructed"] else 0)
        if formal(plan):
            from MemNavData.table2_formal_population import b_population_at
            selected = b_population_at(Path(manifest["plan"]), Path(manifest["run_root"]))["selected"].get(source["source_id"])
            assert bool(selected) == bool(summary["records"])
            assert summary["b_population_sha256"] == sha(Path(manifest["run_root"])/"b_population.json")
    else:
        assert len(summary["records"]) == 2
        population = population_at(Path(manifest["plan"]), Path(manifest["run_root"]))
        assert population["selected"][manifest["index"]//2] == summary["c_source"]
        assert summary["c_population_sha256"] == sha(Path(manifest["run_root"])/"c_population.json")
    if manifest["stage"] != "A" and summary["records"]:
        assert [r["arm"] for r in summary["records"]] == list(arm_order(manifest["index"]))
        assert {r["role"] for r in summary["records"]} == {ROLES[manifest["index"] % 2]}
    checked, groups = [], {}
    for row in summary["records"]:
        assert row["source_id"] == source["source_id"]
        assert (row["scene"], row["episode"]) == (source["scene"], source["episode"])
        result, evidence, plans, actions = verify_rollout(row)
        q = load(row["query_file"])
        if formal(plan):
            from MemNavData.table2_balanced_sampling import novel_supported_as_task, cell_of
            verify_query_direction(q, plan["schema"])
            assert q["cell"] == cell_of(q) and q["current_view_covis"] < .10
            assert q["goal_surface_points"] > 0 and len(q["covis_curve"]) == q["history_frames"]
            if q["analysis_role"] == "novel":
                assert novel_supported_as_task(q["goal_surface_points"], q["covis_curve"], q["current_view_covis"])
            else:
                assert .55 <= q["max_history_covis"] <= .90 and q["max_runtime_eligible_covis"] >= .55
            if manifest["stage"] == "A":
                selection = source["a_selection"]
            elif manifest["stage"] == "B":
                selection = selected[q["analysis_role"]]
            else:
                selection = summary["c_source"]["query_selections"][q["analysis_role"]]
            assert str(row["query_file"]) == selection["query"]
            assert sha(row["query_file"]) == selection["query_sha256"]
        folder = Path(row["directory"])
        assert load(folder/"query_receipt.json")["query"] == runtime_spec(q)
        assert sha(q["goal_rgb"]) == q["goal_rgb_sha256"]
        actual = load(folder/"actual_trace.json")
        assert actual["poses"] == evidence["rollout_trace"]
        assert actual["end_position"] == row["end_position"]
        assert actual["episode_seed"] == source["seed"]
        if q["prefix_root"]:
            prefix = Path(q["prefix_root"])
            trace = load(prefix/"online_a_trace.json")
            assert sha(prefix/"online_a_trace.json") == q["prefix_trace_sha256"]
            np.testing.assert_allclose([evidence["rollout_trace"][0][k] for k in "xyz"], trace["end_position"], rtol=0, atol=1e-8)
            assert evidence["rollout_trace"][0]["yaw"] == trace["end_yaw"]
            replay = load(folder/"prefix_replay.json")
            assert replay["online_frames"] == len(trace["poses"]) and replay["diffusion_samples_during_replay"] == 0
        key = (row["source_id"], row["phase"], row["role"])
        assert row["arm"] not in groups.setdefault(key, {})
        groups[key][row["arm"]] = (row, plans, actions)
        checked.append(result)
    pairs = []
    for key, arms in groups.items():
        if key[1] == "A":
            assert set(arms) == {"native"}
            continue
        assert set(arms) == {"native", "cec"}
        n, g = arms["native"], arms["cec"]
        assert n[0]["query_file"] == g[0]["query_file"]
        assert n[0]["first_query_rgb_sha256"] == g[0]["first_query_rgb_sha256"]
        for filename in ("query_receipt.json", "prefix_replay.json"):
            assert load(Path(n[0]["directory"])/filename) == load(Path(g[0]["directory"])/filename)
        takeover = any(p["receipt"]["revisit_adapter_takeover"] is True for p in g[1])
        if not takeover:
            assert len(n[1]) == len(g[1]) and len(n[2]) == len(g[2])
            for x, y in zip(n[1], g[1]):
                for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                    np.testing.assert_array_equal(x[field], y[field])
            for x, y in zip(n[2], g[2]):
                for field in ("actual_position", "actual_yaw", "action_kind"):
                    assert x[field] == y[field]
            depths = []
            for arm in (n, g):
                http = read_rows(Path(arm[0]["directory"])/"navdp_http_receipts.jsonl")
                depths.append([h["monocular_depth_receipt"] for h in http if h["query_active"]
                               and h["audit"]["image_calls"] and h["path"] != "/memory_replay_step"])
            assert len(depths[0]) == len(depths[1]) == len(n[1])
            for x, y in zip(*depths):
                for field in ("depth_png_sha256", "image_sha256", "scale_receipt_sha256", "frame_index"):
                    assert x[field] == y[field]
        pairs.append(dict(source_id=key[0], phase=key[1], role=key[2],
            cec_takeover=takeover, no_takeover_exact_native=not takeover))
    candidate = summary.get("c_source")
    if candidate and candidate["b_reached"]:
        prefix = Path(candidate["prefix"])
        receipt = load(prefix/"receipt.json")
        a = load(Path(receipt["parent_prefix"])/"online_a_trace.json")
        b = load(Path(receipt["source_rollout"])/"actual_trace.json")
        assert load(prefix/"online_a_trace.json") == compose_prefix(a, b)
        assert Path(receipt["source_rollout"]).parts[-2:] == (candidate["b_role"], "native")
    dump(out/"independent_verification.json", dict(verified=True, formal_result=formal(plan),
        summary_sha256=sha(out/"summary.json"), records=checked, pairs=pairs,
        source_id=source["source_id"], stage=manifest["stage"], index=manifest["index"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    prepare = subs.add_parser("prepare-plan")
    prepare.add_argument("--parent", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--runtime-sha", required=True)
    run = subs.add_parser("run")
    run.add_argument("--stage", choices=("A", "B", "C"), required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--mem-port", type=int, required=True)
    run.add_argument("--nav-port", type=int, required=True)
    select = subs.add_parser("select-c")
    select.add_argument("--out", type=Path, required=True)
    for sub in (run, select):
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--run", type=Path, required=True)
    verify = subs.add_parser("verify")
    verify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare-plan":
        prepare_plan(args.parent, args.out, args.runtime_sha)
    elif args.mode == "run":
        run_task(args)
    elif args.mode == "select-c":
        select_c(args.plan, args.run, args.out)
    else:
        verify_task(args.out)


if __name__ == "__main__":
    main()
