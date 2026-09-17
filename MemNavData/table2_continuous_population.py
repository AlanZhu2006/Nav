"""Freeze continuous Table II sources using construction menus, never rollout scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import shutil

from MemNavData.table2_mixed_local import load, dump, sha
from MemNavData.table2_balanced_sampling import BINS, balanced_assignment
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA, role_cells
from MemNavData.final14_role_pair_contract import stable_u32

SCHEMA = "table2_continuous_population_20260912_v1"
SEQUENCES = ("NNN", "NNR", "NRN", "NRR")


def order_key(*parts):
    return stable_u32(SCHEMA, *map(str, parts))


def allocate_sequences(records):
    """Balance sequences within A-distance bands, interleaving scene sources."""
    records = deepcopy(records)
    for band in BINS:
        scenes = defaultdict(list)
        for row in records:
            if row["a_distance_band"] == band:
                scenes[row["scene"]].append(row)
        for rows in scenes.values():
            rows.sort(key=lambda r: (order_key("source", r["source_id"]), r["source_id"]))
        scene_order = sorted(scenes, key=lambda s: (order_key("scene", band, s), s))
        interleaved = [scenes[s][i] for i in range(max(map(len, scenes.values()), default=0))
                       for s in scene_order if i < len(scenes[s])]
        for i, row in enumerate(interleaved):
            row["sequence"] = SEQUENCES[i % 4]
            row["successor_distance_priority"] = [band] + sorted(
                (b for b in BINS if b != band), key=lambda b: order_key(row["source_id"], "distance", b))
    return records


def first_batch_order(records):
    """One source per sequence × A-distance cell; maximize distinct scenes first."""
    groups = [(s, b) for s in SEQUENCES for b in BINS]
    candidates = {g: [r for r in records if (r["sequence"], r["a_distance_band"]) == g]
                  for g in groups}
    if any(not rows for rows in candidates.values()):
        raise ValueError("The first twelve require every sequence × distance cell")
    scene_owner = {}

    def match(group, seen):
        scenes = sorted({r["scene"] for r in candidates[group]},
                        key=lambda s: (order_key("batch-scene", *group, s), s))
        for scene in scenes:
            if scene in seen:
                continue
            seen.add(scene)
            if scene not in scene_owner or match(scene_owner[scene], seen):
                scene_owner[scene] = group
                return True
        return False

    for group in sorted(groups, key=lambda g: (len({r["scene"] for r in candidates[g]}), g)):
        match(group, set())
    assigned = {g: s for s, g in scene_owner.items()}
    selected, counts = [], Counter()
    for group in groups:
        rows = candidates[group]
        if group in assigned:
            rows = [r for r in rows if r["scene"] == assigned[group]]
        row = min(rows, key=lambda r: (counts[r["scene"]], order_key("batch-source", r["source_id"])))
        selected.append(row)
        counts[row["scene"]] += 1
    used = {r["source_id"] for r in selected}
    remaining = sorted((r for r in records if r["source_id"] not in used),
                       key=lambda r: (order_key("remaining", r["source_id"]), r["source_id"]))
    return [dict(r, task_index=i, first_batch=i < 12) for i, r in enumerate(selected + remaining)]


def build(declaration, menus, portable_plan, out):
    declared = load(declaration)
    if declared["schema"] != FORWARD_SCHEMA:
        raise ValueError("Expected the already audited forward-only declaration")
    portable = {r["source_id"]: r for r in load(portable_plan)["sources"]}
    menu_rows, bindings = [], {}
    for row in declared["sources"]:
        folder = menus / f"source_{row['index']:03d}"
        binding = load(folder / "source_binding.json")
        if binding["sources_sha256"] != sha(declaration) or binding["source_id"] != row["source_id"]:
            raise ValueError("Construction belongs to a different declaration")
        built = load(folder / "construction.json")
        if built["stage"] != "A" or built["schema"] != FORWARD_SCHEMA:
            raise ValueError("Only pre-navigation A construction is allowed")
        menu_rows.append(dict(source_id=row["source_id"], candidates=built["menus"]["novel"]))
        bindings[row["source_id"]] = dict(menu_sha256=sha(folder / "construction.json"), **binding)
    assignment = balanced_assignment(menu_rows, key="A/novel", cells=role_cells(FORWARD_SCHEMA, "novel"))
    choices = {r["source_id"]: r for r in assignment["selected"]}
    records = [dict(source_id=r["source_id"], source_index=r["index"], scene=r["scene"],
                    a_distance_band=choices[r["source_id"]]["cell"].split("/")[0])
               for r in declared["sources"] if r["source_id"] in choices]
    records = first_batch_order(allocate_sequences(records))
    out.mkdir(parents=True, exist_ok=False)
    by_id = {r["source_id"]: r for r in declared["sources"]}
    for record in records:
        original = by_id[record["source_id"]]
        remote = portable[record["source_id"]]
        source = deepcopy(original["initial_state"])
        for key in ("scene", "episode", "seed", "start_position", "start_yaw", "camera_height_m", "camera_intrinsic"):
            if source[key] != remote["initial_state"][key]:
                raise ValueError(f"Portable asset address changed initial state: {key}")
        source["asset"] = remote["initial_state"]["asset"]
        source["asset_sha256"] = remote["source_files"][source["asset"]]
        if source["prefix_root"] is not None or source["prefix_trace_sha256"] is not None:
            raise ValueError("A must start with empty actual memory")
        choice = choices[record["source_id"]]
        if sha(choice["query"]) != choice["query_sha256"]:
            raise ValueError("Precomputed A query changed")
        query = load(choice["query"])
        if sha(query["goal_rgb"]) != query["goal_rgb_sha256"]:
            raise ValueError("A goal image changed")
        relative = Path("goals") / f"source_{record['source_index']:03d}.jpg"
        (out / relative).parent.mkdir(exist_ok=True)
        shutil.copyfile(query["goal_rgb"], out / relative)
        query["goal_rgb"] = relative.as_posix()
        query["asset"] = source["asset"]
        record.update(source=source, a_query=query, construction_binding=bindings[record["source_id"]])
    population = dict(schema=SCHEMA, formal_population=True,
        scope="continuous common-online tasks on a previously consumed MP3D source pool",
        source_declaration_sha256=sha(declaration), portable_plan_sha256=sha(portable_plan),
        declared_sources=len(declared["sources"]), declared_scenes=declared["source_scenes"],
        included_sources=len(records), included_scenes=len({r["scene"] for r in records}),
        excluded_sources=[dict(source_index=r["index"], source_id=r["source_id"],
            reason="No legal forward A within the unchanged construction budget")
            for r in declared["sources"] if r["source_id"] not in choices],
        a_distance_counts=assignment["counts"], sequence_counts=dict(Counter(r["sequence"] for r in records)),
        first_batch_tasks=12, first_batch_scenes=len({r["scene"] for r in records[:12]}),
        navigation_outcomes_read=False, expert_memory_used=False,
        successor_selection="Prefer A distance band; otherwise next predeclared band with a legal common goal. Never relax eligibility.",
        successor_distance_exactly_matched=False,
        release_remaining="Manual technical review of first twelve; no SR threshold; no replacement of failed sources",
        tasks=records)
    shutil.copyfile(declaration, out / "source_declaration.json")
    dump(out / "population.json", population)
    return population


def task_at(path, index):
    path = Path(path).resolve()
    population = load(path)
    if population["schema"] != SCHEMA or population["formal_population"] is not True:
        raise ValueError("Unsupported continuous population")
    row = deepcopy(population["tasks"][index])
    if row["task_index"] != index or index < 0 or row["sequence"] not in SEQUENCES:
        raise ValueError("Wrong frozen task address")
    goal = (path.parent / row["a_query"]["goal_rgb"]).resolve()
    if not goal.is_relative_to(path.parent) or sha(goal) != row["a_query"]["goal_rgb_sha256"]:
        raise ValueError("Frozen A goal payload mismatch")
    row["a_query"]["goal_rgb"] = str(goal)
    row["population_file"] = str(path)
    row["population_sha256"] = sha(path)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declaration", type=Path, required=True)
    p.add_argument("--menus", type=Path, required=True)
    p.add_argument("--portable-plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(a.declaration, a.menus, a.portable_plan, a.out.resolve())
    print({k: v for k, v in result.items() if k not in ("tasks", "excluded_sources")})
