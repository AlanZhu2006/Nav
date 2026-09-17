#!/usr/bin/env python3
"""Offline and in-process diagnostics of the Habitat kinematic executor.

All geometry here belongs to the simulator/execution audit, not the policy.
try_step is still a NavMesh collision approximation, not Go2 physics or a
GT-free controller. Recorded-transition checks do not reconstruct old commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DEFAULTS = dict(v_max=0.0376, lookahead=0.7, r_min=0.40, max_turn_deg=4.5)
MODES = ("legacy_snap", "try_step", "try_step_no_sliding")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def commanded_step(pos, psi, path_xz, cfg):
    """Same pre-collision pursuit law as eval_2leg_habitat.pursuit_step."""
    pos, path_xz = np.asarray(pos, float), np.asarray(path_xz, float)
    xz = pos[[0, 2]]
    distance = np.linalg.norm(path_xz - xz[None, :], axis=1)
    ahead = np.where(distance >= cfg.lookahead)[0]
    target = path_xz[ahead[0]] if len(ahead) else path_xz[-1]
    delta = target - xz
    desired_yaw = np.arctan2(-delta[0], -delta[1])
    alpha = (desired_yaw - psi + np.pi) % (2 * np.pi) - np.pi
    v = cfg.v_max * (0.48 + 0.52 * (1 + np.cos(alpha)) / 2)
    curvature = np.clip(2 * alpha / cfg.lookahead, -1 / cfg.r_min, 1 / cfg.r_min)
    max_turn = np.deg2rad(cfg.max_turn_deg)
    new_yaw = psi + float(np.clip(curvature * v, -max_turn, max_turn))
    forward = np.array([-np.sin(new_yaw), 0., -np.cos(new_yaw)])
    return pos + v * forward, new_yaw, float(v), forward


def compare_step(pos, psi, path_xz, pf, cfg=None):
    cfg = cfg or SimpleNamespace(**DEFAULTS)
    pos = np.asarray(pos, float)
    command, yaw, v, forward = commanded_step(pos, psi, path_xz, cfg)
    first = np.asarray(pf.snap_point(command), float)
    branch, legacy = "full_snap", first
    if not np.isfinite(first).all() or np.linalg.norm((first-command)[[0, 2]]) > .06:
        short_command = pos + .3*v*forward
        second = np.asarray(pf.snap_point(short_command), float)
        if np.isfinite(second).all() and np.linalg.norm((second-short_command)[[0, 2]]) <= .06:
            branch, legacy = "short_snap", second
        else:
            branch, legacy = "blocked", pos.copy()
    endings = {"legacy_snap": legacy,
               "try_step": np.asarray(pf.try_step(pos, command), float),
               "try_step_no_sliding": np.asarray(pf.try_step_no_sliding(pos, command), float)}
    if not all(np.isfinite(p).all() for p in endings.values()):
        raise ValueError("Non-finite collision-interface output; do not silently substitute")
    record = {
        "position_before": pos.tolist(), "yaw_before": float(psi),
        "commanded_position": command.tolist(), "yaw_after": yaw,
        "commanded_translation_m": v, "legacy_branch": branch,
        "end_positions": {k: p.tolist() for k, p in endings.items()},
        "legacy_command_correction_m": float(np.linalg.norm((legacy-command)[[0, 2]])),
        "legacy_vs_try_step_m": float(np.linalg.norm(legacy-endings["try_step"])),
        "legacy_vs_no_sliding_m": float(np.linalg.norm(legacy-endings["try_step_no_sliding"])),
    }
    return endings, yaw, record


def summarize_steps(rows):
    return {
        "actions": len(rows),
        "legacy_branches": {k: sum(r["legacy_branch"] == k for r in rows)
                            for k in ("full_snap", "short_snap", "blocked")},
        "legacy_command_correction_gt_1mm": sum(r["legacy_command_correction_m"] > .001 for r in rows),
        "legacy_vs_try_step_gt_1mm": sum(r["legacy_vs_try_step_m"] > .001 for r in rows),
        "legacy_vs_no_sliding_gt_1mm": sum(r["legacy_vs_no_sliding_m"] > .001 for r in rows),
        "max_legacy_vs_try_step_m": max((r["legacy_vs_try_step_m"] for r in rows), default=0.),
    }


def audit_trace(path, pf):
    data = json.loads(Path(path).read_text())
    rows = list(data["rollout_traces"]["query"])
    end = data.get("query_result")
    terminal_included = False
    if end and int(end["steps"]) > int(rows[-1]["step"]):
        rows.append(dict(zip(("x", "y", "z"), end["end_position"]),
                         step=end["steps"], yaw=end["end_yaw_rad"]))
        terminal_included = True
    records = []
    for a, b in zip(rows, rows[1:]):
        if int(b["step"]) != int(a["step"]) + 1:
            raise ValueError("Cannot treat a multi-action gap as one action")
        p, q = [np.array([r[k] for k in ("x", "y", "z")]) for r in (a, b)]
        displacement = q-p
        direction = np.array([-np.sin(b["yaw"]), -np.cos(b["yaw"])])
        sliding = np.asarray(pf.try_step(p, q))
        no_sliding = np.asarray(pf.try_step_no_sliding(p, q))
        if not np.isfinite(sliding).all() or not np.isfinite(no_sliding).all():
            raise ValueError("Non-finite recorded-transition check")
        records.append({
            "step": int(a["step"]), "position_before": p.tolist(), "position_after": q.tolist(),
            "realized_planar_distance_m": float(np.linalg.norm(displacement[[0, 2]])),
            "lateral_relative_to_new_heading_m": float(displacement[0]*direction[1]-displacement[2]*direction[0]),
            "forward_relative_to_new_heading_m": float(displacement[[0, 2]] @ direction),
            "observed_endpoint_vs_try_step_m": float(np.linalg.norm(sliding-q)),
            "observed_endpoint_vs_no_sliding_m": float(np.linalg.norm(no_sliding-q)),
        })
    return {"path": str(Path(path).resolve()), "sha256": sha(path),
            "terminal_included": terminal_included,
            "recorded_reached": None if end is None else end["reached"],
            "transitions": len(records),
            "lateral_gt_1mm": sum(abs(r["lateral_relative_to_new_heading_m"]) > .001 for r in records),
            "backwards_gt_1mm": sum(r["forward_relative_to_new_heading_m"] < -.001 for r in records),
            "step_gt_vmax_plus_1mm": sum(r["realized_planar_distance_m"] > .0386 for r in records),
            "observed_endpoint_vs_try_step_gt_1mm": sum(r["observed_endpoint_vs_try_step_m"] > .001 for r in records),
            "observed_endpoint_vs_no_sliding_gt_1mm": sum(r["observed_endpoint_vs_no_sliding_m"] > .001 for r in records),
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import habitat_sim
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(args.navmesh)):
        raise ValueError("Cannot load pinned NavMesh")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "habitat_recorded_transition_audit_v1", "policy_rerun": False,
              "scope": "observed endpoint reachability, not replay of original control commands",
              "navmesh": str(args.navmesh.resolve()), "navmesh_sha256": sha(args.navmesh),
              "habitat_version": habitat_sim.__version__, "results": []}
    for path in args.trace:
        result = audit_trace(path, pf)
        dump(args.out / (path.stem + "_transitions.json"), result.pop("records"))
        report["results"].append(result)
    dump(args.out / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
