#!/usr/bin/env python3
"""Fixed synthetic heading probes at recorded states; never a navigation SR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from habitat_executor_audit import compare_step, dump, sha, summarize_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import habitat_sim
    mesh = args.inputs / "eF36g7L6Z9M.basis.navmesh"
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(mesh)):
        raise ValueError(mesh)
    args.out.mkdir(parents=True, exist_ok=False)
    angles = (-180, -90, -60, -30, 0, 30, 60, 90, 180)
    contract = {
        "kind": "synthetic command probes at consumed recorded states",
        "heading_offsets_deg": angles, "target_radius_m": 2.5,
        "navmesh_sha256": sha(mesh), "policy_rerun": False, "SR_measured": False,
        "note": "These headings are not claimed to be the missing original NavDP commands.",
    }
    dump(args.out / "manifest.json", contract)
    results = []
    for arm in ("mono_native", "mono_cec_endpoint", "mono_cec_route_tangent"):
        path = args.inputs / f"{arm}.json"
        states = json.loads(path.read_text())["rollout_traces"]["query"]
        rows = []
        with (args.out / f"{arm}_probes.jsonl").open("x") as stream:
            for state in states:
                p = np.array([state[k] for k in ("x", "y", "z")])
                for angle in angles:
                    yaw = state["yaw"] + np.deg2rad(angle)
                    target = p[[0, 2]] + 2.5*np.array([-np.sin(yaw), -np.cos(yaw)])
                    endings, _, record = compare_step(p, state["yaw"], [target], pf)
                    record.update(state_step=state["step"], synthetic_heading_offset_deg=angle)
                    actual_legacy_reachable = np.asarray(pf.try_step(p, endings["legacy_snap"]))
                    record["legacy_endpoint_unreachable_error_m"] = float(np.linalg.norm(
                        actual_legacy_reachable-endings["legacy_snap"]))
                    rows.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        result = dict(arm=arm, source_sha256=sha(path), states=len(states), **summarize_steps(rows),
                      legacy_endpoint_unreachable_gt_1mm=sum(
                          r["legacy_endpoint_unreachable_error_m"] > .001 for r in rows))
        results.append(result)
        print(json.dumps(result), flush=True)
    dump(args.out / "summary.json", dict(contract=contract, results=results))


if __name__ == "__main__":
    main()
