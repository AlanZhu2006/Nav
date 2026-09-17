"""Differential audit: the v3 constructor is the v2 front subset, not a new runtime.

Run static checks with the existing MemNav environment and render checks with
the existing Habitat environment. No policy inference or navigation is run.
Previously consumed A/B histories are diagnostic fixtures, never v3 memory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path

from MemNavData.table2_mixed_local import dump, load, sha, runtime_spec
from MemNavData.table2_sampling_profiles import BALANCED_SCHEMA, FORWARD_SCHEMA, FORWARD_CELLS


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".diagnostics/table2_full_rerun_20260911_v2"
BASE = OLD / "bundle_r2"


def static_check(out):
    from MemNavData.prepare_table2_mixed_bundle import FORWARD_OVERLAY
    from MemNavData.table2_balanced_sampling import balanced_assignment
    from MemNavData.table2_formal_population import prepare_sources

    original = load(OLD / "source_declaration.json")
    inventory = ROOT / ".diagnostics/pt1_multinovel_capacity_20260911_v1"
    prepare_sources(inventory / "trajectory_inventory.json",
                    inventory / "independent_verification.json",
                    out / "source_declaration.json", schema=FORWARD_SCHEMA)
    declared = load(out / "source_declaration.json")
    expected = deepcopy(original)
    expected.update(schema=FORWARD_SCHEMA, protocol_sha256=declared["protocol_sha256"])
    for source in expected["sources"]:
        source["initial_state"]["schema"] = FORWARD_SCHEMA
    if expected != declared:
        raise ValueError("A non-schema source declaration field changed")

    changed, unchanged = [], []
    for line in (BASE / "SOURCE_BUNDLE.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if sha(BASE / relative) != digest:
            raise ValueError(f"Sealed baseline changed: {relative}")
        current = ROOT / relative
        (unchanged if current.is_file() and sha(current) == digest else changed).append(relative)
    allowed = {"MemNavData/" + name for name in FORWARD_OVERLAY}
    # The frozen-baseline packager does NOT import these unrelated changes.
    excluded = sorted(set(changed) - allowed)
    protected = (
        "MemNavData/bounded_pursuit.py", "MemNavData/navdp_front_goal_adapter.py",
        "MemNavData/certified_relocalization_contract.py",
        "MemNavData/certified_relocalization_runtime.py",
        "MemNavData/monocular_depth_runtime.py", "MemNavData/lingbot_pose_diagnostic_server.py",
        "MemNavData/navdp_depth_raster_audit_server.py", "MemNavData/lingbot_depth_raster.py",
        "MemNavData/run_repaired_fullmono_local.py", "MemNavData/table2_mixed_local.py",
        "MemNavData/table2_novel_sampling.py", "MemNavData/repaired_hm3d_hpc_env.sh",
        "NavDP/baselines/navdp/policy_agent.py", "NavDP/baselines/navdp/policy_network.py",
        "NavDP/baselines/memnav/memnav_server.py", "NavDP/baselines/memnav/policy_agent.py",
    )
    if not set(protected).issubset(unchanged):
        raise ValueError("Local construction audit would mix changed runtime dependencies")
    spec = importlib.util.spec_from_file_location(
        "sealed_table2_balanced", BASE / "MemNavData/table2_balanced_sampling.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    menus = [dict(source_id=s["source_id"], candidates=load(
        OLD / "A_menus" / f"source_{s['index']:03d}" / "construction.json")["menus"]["novel"])
        for s in original["sources"]]
    # Refactoring to support front-only cells must preserve v2 exactly.
    if baseline.balanced_assignment(menus, key="A/novel") != balanced_assignment(menus, key="A/novel"):
        raise ValueError("The existing all-direction selection changed")
    forward = deepcopy(menus)
    for menu in forward:
        menu["candidates"] = [c for c in menu["candidates"] if c["cell"] in FORWARD_CELLS]
    assignment = balanced_assignment(forward, key="A/novel", cells=FORWARD_CELLS)
    result = dict(verified=True, source_count=len(declared["sources"]), source_scenes=declared["source_scenes"],
        source_fields_changed=["schema", "protocol_sha256", "sources[*].initial_state.schema"],
        identical_existing_bundle_files=len(unchanged), protected_runtime_files=list(protected),
        changed_experiment_files=sorted(set(changed) & allowed),
        unrelated_workspace_changes_excluded_from_bundle=excluded,
        legacy_assignment_identical=True, preview_forward_assignment=assignment,
        preview_is_not_a_formal_payload=True, no_navigation_performed=True,
        baseline_receipt_sha256=sha(BASE / "SOURCE_BUNDLE.sha256"))
    dump(out / "static_verification.json", result)
    print({k: result[k] for k in ("verified", "source_count", "source_scenes",
        "identical_existing_bundle_files", "unrelated_workspace_changes_excluded_from_bundle")}, flush=True)


def semantic_query(query):
    return {k: v for k, v in query.items() if k not in ("schema", "goal_rgb")}


def render_check(out):
    from MemNavData.table2_balanced_sampling import construct

    cases = [("A_source_000", "A", OLD / "A_menus/source_000")]
    cases += [(name, "B" if name == "B" else "C", OLD / "consumed_prefix_checks" / name)
              for name in ("B", "C_after_novel", "C_after_revisit")]
    results = []
    source_keys = ("schema", "scene", "episode", "asset", "seed", "stage_number", "start_position",
                   "start_yaw", "camera_height_m", "camera_intrinsic", "prefix_root", "prefix_trace_sha256")
    for name, stage, folder in cases:
        print("CONSTRUCTION_DIFFERENTIAL", name, flush=True)
        old = load(folder / "construction.json")
        first = next(row for rows in old["menus"].values() for row in rows)
        old_query = load(first["query"])
        source = {key: old_query[key] for key in source_keys}
        source["schema"] = FORWARD_SCHEMA
        prefix = Path(source["prefix_root"]) if source["prefix_root"] else None
        new = construct(source, stage=stage, prefix=prefix, out=out / name)
        expected_pool = [c for c in old["diagnostics"]["novel"]["spatial"]["candidates"]
                         if c["cell"] in FORWARD_CELLS]
        if expected_pool != new["diagnostics"]["novel"]["spatial"]["candidates"]:
            raise ValueError(f"{name}: front proposals changed their seed/position/photo yaw")
        expected_checks = [c for c in old["diagnostics"]["novel"]["checked"]
                           if c["cell"] in FORWARD_CELLS]
        if expected_checks != new["diagnostics"]["novel"]["checked"]:
            raise ValueError(f"{name}: front visual evidence changed")
        if old["diagnostics"].get("revisit") != new["diagnostics"].get("revisit"):
            raise ValueError(f"{name}: Revisit candidate generation/evidence changed")
        checked = []
        for role in ("novel", "revisit"):
            old_rows = {c["cell"]: c for c in old["menus"][role]
                        if role == "revisit" or c["cell"] in FORWARD_CELLS}
            new_rows = {c["cell"]: c for c in new["menus"][role]}
            if old_rows.keys() != new_rows.keys():
                raise ValueError(f"{name}/{role}: menu gained or lost a permitted candidate")
            for cell, row in new_rows.items():
                a, b = load(old_rows[cell]["query"]), load(row["query"])
                if semantic_query(a) != semantic_query(b):
                    raise ValueError(f"{name}/{role}/{cell}: a non-schema/path query field changed")
                if sha(a["goal_rgb"]) != sha(b["goal_rgb"]):
                    raise ValueError(f"{name}/{role}/{cell}: goal pixels changed")
                if load(Path(row["query"]).with_name("runtime.json")) != runtime_spec(b):
                    raise ValueError("Runtime spec includes undeclared fields")
                if prefix:
                    trace = load(prefix / "online_a_trace.json")
                    if b["start_position"] != trace["end_position"] or b["start_yaw"] != trace["end_yaw"]:
                        raise ValueError("Constructor changed the actual prefix endpoint or yaw")
                checked.append(dict(role=role, cell=cell, image_sha256=b["goal_rgb_sha256"],
                                    initial_route_angle_deg=b["initial_relative_route_angle_deg"]))
        record = dict(case=name, stage=stage, verified=True,
                      novel_cells=len(new["menus"]["novel"]), revisit_cells=len(new["menus"]["revisit"]),
                      checked_queries=checked, prefix=str(prefix) if prefix else None)
        dump(out / f"{name}_verification.json", record)
        results.append(record)
        print("VERIFIED", name, record["novel_cells"], record["revisit_cells"], flush=True)
    dump(out / "render_verification.json", dict(verified=True, no_navigation_performed=True,
        consumed_prefixes_are_diagnostics_only=True, records=results))


def verify_all_a(out):
    """Compare every newly rendered A menu against the sealed front subset."""
    from MemNavData.table2_balanced_sampling import balanced_assignment
    sources_path = out / "source_declaration.json"
    declared = load(sources_path)
    menus, records = [], []
    for source in declared["sources"]:
        folder = out / "A_menus" / f"source_{source['index']:03d}"
        old_folder = OLD / "A_menus" / folder.name
        a, b = load(old_folder / "construction.json"), load(folder / "construction.json")
        binding = load(folder / "source_binding.json")
        if binding["sources_sha256"] != sha(sources_path) or binding["source_id"] != source["source_id"]:
            raise ValueError("New A menu has the wrong source declaration")
        expected = {r["cell"]: r for r in a["menus"]["novel"] if r["cell"] in FORWARD_CELLS}
        actual = {r["cell"]: r for r in b["menus"]["novel"]}
        if expected.keys() != actual.keys() or b["menus"]["revisit"]:
            raise ValueError(f"Unexpected A candidate set: {source['source_id']}")
        for cell, row in actual.items():
            before, after = load(expected[cell]["query"]), load(row["query"])
            if semantic_query(before) != semantic_query(after) or sha(before["goal_rgb"]) != sha(after["goal_rgb"]):
                raise ValueError(f"Non-direction A change: {source['source_id']}/{cell}")
            if after["schema"] != FORWARD_SCHEMA or abs(after["initial_relative_route_angle_deg"]) > 60.:
                raise ValueError("New A query is not forward-Novel")
        menus.append(dict(source_id=source["source_id"], candidates=b["menus"]["novel"]))
        records.append(dict(source_id=source["source_id"], forward_cells=list(actual),
                            construction_sha256=sha(folder / "construction.json")))
    assignment = balanced_assignment(menus, key="A/novel", cells=FORWARD_CELLS)
    result = dict(verified=True, declared_sources=len(records),
        constructible_sources=len(assignment["selected"]), empty_sources=assignment["empty_sources"],
        distance_cell_counts=assignment["counts"], goal_images_and_evidence_match_v2_front_subset=True,
        sources_sha256=sha(sources_path), no_navigation_performed=True, records=records)
    dump(out / "full_A_differential_verification.json", result)
    print({k: v for k, v in result.items() if k not in ("records", "empty_sources")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "render", "verify-a"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"static": static_check, "render": render_check, "verify-a": verify_all_a}[args.mode](args.out.resolve())
