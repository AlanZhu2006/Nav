#!/usr/bin/env python3
"""Compare direct and reverse-graph control on causal conditional-C."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from MemNavData.summarize_conditional_c_eval import (
        compare,
        load_arm,
        require,
        summarize_rows,
    )
except ModuleNotFoundError:  # direct script invocation
    from summarize_conditional_c_eval import (  # type: ignore
        compare,
        load_arm,
        require,
        summarize_rows,
    )


ARM_SPECS = {
    "native": ("direct", "navdp_native"),
    "direct_gap16": ("direct", "geometry_router"),
    "graph_gap16": ("graph", "geometry_router"),
    "oracle_anchor_direct": ("direct", "oracle_anchor"),
    "oracle_anchor_graph": ("graph", "oracle_anchor"),
    "oracle_point": ("direct", "oracle_point"),
}


def validate_configuration(
        name: str, rows: list[dict], *, expected_gap: int | None,
        expected_spacing_m: float) -> None:
    for row in rows:
        if expected_gap is not None:
            require(row["candidate_gap"] == expected_gap,
                    f"{name} candidate gap mismatch")
        require(math.isclose(
            row["graph_spacing_m"], expected_spacing_m,
            rel_tol=0.0, abs_tol=1e-9),
            f"{name} graph spacing mismatch")


def load_run(manifest: dict, root: Path, physical_arm: str) -> tuple[dict, set]:
    scenes = manifest["selection"]["selected_scenes"]
    expected = {
        (scene, record["episode"])
        for scene in scenes for record in manifest["episodes"][scene]
    }
    output = {}
    for index, scene in enumerate(scenes):
        scene_root = root / "scenes" / f"{index:02d}_{scene}"
        output.update(load_arm(scene_root, scene, physical_arm))
    require(set(output) == expected,
            f"{physical_arm} result keys differ from manifest: {root}")
    return output, expected


def summarize_graph_conditional(
        manifest: dict, direct_root: Path, graph_root: Path) -> dict:
    roots = {"direct": direct_root, "graph": graph_root}
    rows = {}
    expected = None
    for name, (root_name, physical_arm) in ARM_SPECS.items():
        loaded, run_expected = load_run(
            manifest, roots[root_name], physical_arm)
        if expected is None:
            expected = run_expected
        require(run_expected == expected, f"manifest mismatch: {name}")
        rows[name] = loaded
    assert expected is not None

    for name in ("direct_gap16", "oracle_anchor_direct"):
        validate_configuration(
            name, list(rows[name].values()),
            expected_gap=16, expected_spacing_m=0.0)
    for name in ("graph_gap16", "oracle_anchor_graph"):
        validate_configuration(
            name, list(rows[name].values()),
            expected_gap=16, expected_spacing_m=1.25)
    for key in expected:
        direct = rows["direct_gap16"][key]
        graph = rows["graph_gap16"][key]
        require(direct["deterministic_plan_seeds"] is True
                and graph["deterministic_plan_seeds"] is True,
                f"direct/graph plan seeding disabled: {key}")
        require(direct["memory_prefix_frames"] > 0,
                f"direct memory prefix is empty: {key}")
        require(direct["memory_prefix_frames"] == graph["memory_prefix_frames"],
                f"direct/graph LingBot prefix mismatch: {key}")
        require(direct["navdp_prefix_decision_frames"] > 0,
                f"direct NavDP prefix is empty: {key}")
        require(
            direct["navdp_prefix_decision_frames"]
            == graph["navdp_prefix_decision_frames"],
            f"direct/graph NavDP prefix mismatch: {key}")

    comparisons = {}
    for left, right in (
        ("native", "direct_gap16"),
        ("direct_gap16", "graph_gap16"),
        ("graph_gap16", "oracle_anchor_graph"),
        ("oracle_anchor_direct", "oracle_anchor_graph"),
        ("oracle_anchor_graph", "oracle_point"),
    ):
        comparisons[f"{left}_vs_{right}"] = compare(
            left, right, rows[left], rows[right], expected)

    return {
        "audit": {
            "status": "ok",
            "protocol": "conditional_C_after_causal_source_AB_replay",
            "diagnostic_not_end_to_end_sr": True,
            "direct_graph_same_prefix_seed_geodesic": True,
            "scenes": len(manifest["selection"]["selected_scenes"]),
            "episodes": len(expected),
        },
        "arms": {
            name: summarize_rows([run[key] for key in sorted(expected)])
            for name, run in rows.items()
        },
        "pairwise": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(summarize_graph_conditional(
        manifest, args.direct_root, args.graph_root),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
