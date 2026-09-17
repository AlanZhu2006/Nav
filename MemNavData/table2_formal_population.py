"""Immutable prospective sources and stage-wise query selection for Table II."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from MemNavData.table2_mixed_local import dump, load, sha, runtime_spec
from MemNavData.table2_novel_sampling import initial_a_yaw, balanced_c_sources
from MemNavData.table2_balanced_sampling import SCHEMA, CELLS, balanced_assignment, construct
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA, protocol_path, role_cells

PROTOCOL = Path(__file__).with_name("TABLE2_FULL_RERUN_PROTOCOL_20260911.md")


def prepare_sources(inventory_path, capacity_path, out, *, schema=SCHEMA):
    """Use all capacity-screened scenes, never rank scenes by previous SR."""
    import pandas as pd
    inventory = load(inventory_path)
    capacity = load(capacity_path)
    if not capacity["verified"] or capacity["navigation_rollouts"] != 0:
        raise ValueError("Source capacity must be an independently verified offline audit")
    selected_scenes = {r["source_id"].split("/")[0]
                       for r in capacity["spatially_screened_visual_followup_sources"]}
    groups = defaultdict(list)
    for row in inventory["histories"]:
        if row["scene"] in selected_scenes:
            groups[row["scene"]].append(row)
    sources = []
    for scene in sorted(groups):
        carriers = sorted(groups[scene], key=lambda r: r["episode"])
        # Eight *new* independent yaw/goal/seed cases per scene; the original
        # physical start is a scenario carrier, never a recorded memory.
        for replicate in range(8):
            carrier = carriers[replicate % len(carriers)]
            first = pd.read_parquet(carrier["parquet"], columns=["observation.camera_intrinsic"]).iloc[0]
            episode = f"episode_{replicate:04d}"
            index = len(sources)
            spec = dict(schema=schema, scene=scene, episode=episode, asset=carrier["asset"],
                seed=2026091100 + index, stage_number=0,
                start_position=carrier["positions"][0],
                start_yaw=initial_a_yaw(f"{SCHEMA}/{scene}/{episode}"),
                camera_height_m=carrier["camera_height_m"],
                camera_intrinsic=np.stack(first["observation.camera_intrinsic"]).tolist(),
                prefix_root=None, prefix_trace_sha256=None)
            sources.append(dict(index=index, source_id=f"{scene}/{episode}", scene=scene,
                episode=episode, seed=spec["seed"], seed_rank=index, asset=carrier["asset"],
                initial_state=spec, carrier_source_id=carrier["source_id"],
                carrier_metadata_sha256=carrier["metadata_sha256"],
                carrier_parquet_sha256=carrier["parquet_sha256"],
                reuse_index=replicate // len(carriers), expert_history_used=False))
    if len(groups) != 18 or len(sources) != 144:
        raise ValueError("This protocol is the capacity-screened 18-scene, 144-source population")
    dump(out, dict(schema=schema, protocol_sha256=sha(protocol_path(schema)),
        inventory_sha256=sha(inventory_path), source_scenes=len(groups), source_budget=len(sources),
        capacity_verification_sha256=sha(capacity_path),
        source_selection="all 18 scenes with B/C unseen-view supply and diagnostic A/B endpoints 2--9 m apart",
        prior_navigation_outcomes_read=False, sources=sources,
        population="MP3D assets from the previously consumed PT1/audited-gapfill inventory; not held-out confirmation"))


def prepare_a(sources_path, out, indices=None):
    data = load(sources_path)
    if data["protocol_sha256"] != sha(protocol_path(data["schema"])):
        raise ValueError("Protocol changed after source declaration")
    out.mkdir(parents=True, exist_ok=True)
    selected = data["sources"] if indices is None else [data["sources"][i] for i in indices]
    for source in selected:
        folder = out / f"source_{source['index']:03d}"
        if folder.exists():
            raise FileExistsError(folder)
        print(f"CONSTRUCT A {source['index']:03d} {source['source_id']}", flush=True)
        result = construct(source["initial_state"], stage="A", prefix=None, out=folder)
        dump(folder / "source_binding.json", dict(source_id=source["source_id"],
             sources_sha256=sha(sources_path), constructor_sha256=sha(Path(__file__).with_name("table2_balanced_sampling.py"))))
        print(f"DONE A {source['index']:03d}: {len(result['menus']['novel'])} cells", flush=True)


def publish(sources_path, menus, out, remote_payload, asset_root):
    """Package generated menus at their final immutable HPC addresses."""
    declared = load(sources_path)
    if declared["protocol_sha256"] != sha(protocol_path(declared["schema"])):
        raise ValueError("Protocol differs from prospective source declaration")
    out.mkdir(parents=True, exist_ok=False)
    assignment_input, sources, assets = [], [], {}
    for source in declared["sources"]:
        folder = menus / f"source_{source['index']:03d}"
        binding = load(folder / "source_binding.json")
        if binding["sources_sha256"] != sha(sources_path):
            raise ValueError("A menu does not belong to this prospective population")
        source = deepcopy(source)
        local_asset = Path(source["asset"])
        remote_asset = str(asset_root / source["scene"] / f"{source['scene']}.glb")
        assets.setdefault(remote_asset, sha(local_asset))
        source["asset"] = source["initial_state"]["asset"] = remote_asset
        source["source_files"] = {remote_asset: assets[remote_asset]}
        dst = out / folder.name
        shutil.copytree(folder, dst)
        original_prefix = str(folder.resolve())
        remote_prefix = str(remote_payload / folder.name)

        def rewrite(value):
            if isinstance(value, str):
                if value == str(local_asset):
                    return remote_asset
                return remote_prefix + value[len(original_prefix):] if value.startswith(original_prefix + "/") else value
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            return value

        # These are generated unpublished artifacts, never existing results.
        for path in dst.rglob("*.json"):
            path.write_text(json.dumps(rewrite(load(path)), indent=2, allow_nan=False) + "\n")
        built = load(dst / "construction.json")
        for candidates in built["menus"].values():
            for row in candidates:
                local_query = out / Path(row["query"]).relative_to(remote_payload)
                row["query_sha256"] = sha(local_query)
        (dst / "construction.json").write_text(json.dumps(built, indent=2, allow_nan=False) + "\n")
        assignment_input.append(dict(source_id=source["source_id"], candidates=built["menus"]["novel"]))
        sources.append(source)
    assignment = balanced_assignment(assignment_input, key="A/novel",
                                     cells=role_cells(declared["schema"], "novel"))
    selected = {r["source_id"]: r for r in assignment["selected"]}
    for source in sources:
        source["a_selection"] = selected.get(source["source_id"])
    dump(out / "source_declaration.json", declared)
    dump(out / "published_sources.json", dict(schema=declared["schema"], sources=sources,
        source_declaration_sha256=sha(sources_path), protocol_sha256=sha(protocol_path(declared["schema"])),
        remote_payload=str(remote_payload), source_budget=len(sources), source_scenes=declared["source_scenes"],
        A_assignment=assignment, assets=assets))
    # Final receipt hashes bind both images and their published JSON addresses.
    paths = sorted(p for p in out.rglob("*") if p.is_file())
    with (out / "PAYLOAD.sha256").open("x") as stream:
        for path in paths:
            stream.write(f"{sha(path)}  {path.relative_to(out).as_posix()}\n")


def make_plan(published_path, out, runtime_sha, payload_receipt_sha, work_root):
    published = load(published_path)
    if published["protocol_sha256"] != sha(protocol_path(published["schema"])):
        raise ValueError("Published protocol mismatch")
    dump(out, dict(schema=published["schema"], formal_result=True, sources=published["sources"],
        source_budget=published["source_budget"], source_scenes=published["source_scenes"],
        runtime_sha256=runtime_sha, protocol_sha256=published["protocol_sha256"], work_root=work_root,
        payload_root=published["remote_payload"], payload_receipt_sha256=payload_receipt_sha,
        published_sources_sha256=sha(published_path), arms=["native", "cec"],
        c_selection="global_equal_native_B_roles", A_assignment=published["A_assignment"],
        runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on",
        B_pair_requires_both_roles=True, no_adaptive_expansion=True))


def select_b(plan_path, run, out):
    from MemNavData.table2_mixed_hpc import validate_plan, read_completed, durable_root
    plan = validate_plan(load(plan_path))
    inputs, eligible, excluded = [], [], []
    for source in plan["sources"]:
        summary = read_completed(run, "A", source["index"], sha(plan_path))
        inputs.append(dict(index=source["index"], sha256=sha(durable_root(run, "A", source["index"])/"summary.json")))
        built = summary["b_construction"]
        if built and built["both_constructed"]:
            eligible.append((source, built))
        else:
            excluded.append(source["source_id"])
    assignments, selected = {}, {}
    for role in ("novel", "revisit"):
        menus = [dict(source_id=s["source_id"], candidates=b["menus"][role]) for s, b in eligible]
        assignments[role] = balanced_assignment(menus, key="B/"+role,
                                              cells=role_cells(plan["schema"], role))
        for row in assignments[role]["selected"]:
            selected.setdefault(row["source_id"], {})[role] = row
    result = dict(schema=plan["schema"], plan_sha256=sha(plan_path), input_summaries=inputs,
        selected=selected, assignments=assignments, excluded_sources=excluded,
        B_navigation_outcomes_read=False, queries=2*len(selected))
    dump(out, result)
    return result


def b_population_at(plan_path, run):
    from MemNavData.table2_mixed_hpc import durable_root
    result = load(Path(run)/"b_population.json")
    if result["plan_sha256"] != sha(plan_path) or result["B_navigation_outcomes_read"]:
        raise ValueError("Invalid prospective B population")
    for row in result["input_summaries"]:
        if sha(durable_root(run, "A", row["index"])/"summary.json") != row["sha256"]:
            raise ValueError("An A parent changed after B selection")
    return result


def archive_unconstructed_b(plan_path, run):
    """Record absent B queries on CPU; do not request a GPU for empty tasks."""
    from MemNavData.table2_mixed_hpc import read_completed, durable_root, task_source
    from MemNavData.covisibility_task_archive import archive
    plan = load(plan_path)
    population = b_population_at(plan_path, run)
    runnable = []
    for index in range(2 * len(plan["sources"])):
        source, role = task_source(plan, "B", index)
        if source["source_id"] in population["selected"]:
            runnable.append(index)
            continue
        parent = read_completed(run, "A", source["index"], sha(plan_path))
        built = parent["b_construction"]
        if built and built["both_constructed"]:
            raise ValueError("A constructible B role-pair cannot be omitted")
        durable = durable_root(run, "B", index)
        if durable.exists():
            existing = read_completed(run, "B", index, sha(plan_path))
            if existing["records"]:
                raise ValueError("Cannot replace a navigation result by a construction record")
            continue
        folder = Path(run)/"construction_only"/f"B_{index:03d}"/"task"
        summary = dict(schema=plan["schema"], stage="B", index=index, source_id=source["source_id"],
            plan_sha256=sha(plan_path), formal_result=True, completed=True, records=[],
            attrition=["A_failed_or_B_role_pair_not_constructible"],
            c_source=dict(source_id=source["source_id"], source_index=source["index"],
                b_task_index=index, scene=source["scene"], episode=source["episode"], collector="native",
                b_role=role, b_reached=False, both_c_constructed=False),
            b_population_sha256=sha(Path(run)/"b_population.json"),
            parent_A_summary_sha256=sha(durable_root(run,"A",source["index"])/"summary.json"))
        dump(folder/"summary.json",summary)
        dump(folder/"independent_verification.json",dict(verified=True, formal_result=True,
            summary_sha256=sha(folder/"summary.json"), records=[], pairs=[],
            verification_mode="no query exists in the frozen B population; no navigation performed",
            parent_A_summary_sha256=summary["parent_A_summary_sha256"]))
        durable.mkdir(parents=True)
        archive(folder,durable,0)
    return runnable


def assign_c_queries(candidates, *, schema=SCHEMA):
    """Keep the original 50/50 prefix rule; select C goals only afterwards."""
    chosen = deepcopy(balanced_c_sources(candidates))
    assignments = {}
    for role in ("novel", "revisit"):
        menus = [dict(source_id=str(c["b_task_index"]), candidates=c["menus"][role]) for c in chosen]
        assignments[role] = balanced_assignment(menus, key="C/"+role,
                                              cells=role_cells(schema, role))
        mapping = {r["source_id"]: r for r in assignments[role]["selected"]}
        for c in chosen:
            selected = mapping[str(c["b_task_index"])]
            c.setdefault("queries", {})[role] = selected["query"]
            c.setdefault("query_selections", {})[role] = selected
    return chosen, assignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("sources")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--capacity", type=Path, required=True)
    p.add_argument("--novel-front-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("construct-a")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--indices", help="comma-separated declared indices; omission constructs all")
    p = sub.add_parser("publish")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--menus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-payload", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p = sub.add_parser("plan")
    p.add_argument("--published", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runtime-sha", required=True)
    p.add_argument("--payload-receipt-sha", required=True)
    p.add_argument("--work-root", required=True)
    p = sub.add_parser("select-b")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("empty-b")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "sources":
        prepare_sources(args.inventory, args.capacity, args.out,
                        schema=FORWARD_SCHEMA if args.novel_front_only else SCHEMA)
    elif args.mode == "construct-a":
        prepare_a(args.sources, args.out, None if args.indices is None else list(map(int, args.indices.split(","))))
    elif args.mode == "publish":
        publish(args.sources, args.menus, args.out, args.remote_payload, args.asset_root)
    elif args.mode == "plan":
        make_plan(args.published, args.out, args.runtime_sha, args.payload_receipt_sha, args.work_root)
    elif args.mode == "select-b":
        select_b(args.plan, args.run, args.out)
    else:
        indices = archive_unconstructed_b(args.plan,args.run)
        dump(args.out,dict(indices=indices,plan_sha256=sha(args.plan)))
