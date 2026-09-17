#!/usr/bin/env python3
"""Independently verify complete Bullet query receipts; no model or simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def close(a, b, tolerance=1e-8):
    if not np.allclose(a, b, rtol=0, atol=tolerance):
        raise AssertionError(f"mismatch {a} != {b}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "independent_verification.json"
    if target.exists():
        raise FileExistsError(target)
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    assert summary["completed"] and not summary["changed_sources"]
    repo = Path(__file__).resolve().parents[1]
    for name, expected in manifest["sources"].items():
        snapshot = root / "sources" / Path(name).relative_to(repo)
        assert sha(snapshot) == expected, name
    assert len(summary["results"]) == len(manifest["variants"])
    results, initializations, image_hashes, initial_plans = [], [], [], {}
    for policy, tracker in manifest["variants"]:
        folder = root / "evaluation" / f"{policy}__{tracker}"
        init = read(folder / "initialization.json")
        terminal = read(folder / "terminal_measurements.json")[0]
        runtime = read(folder / "summary.json")
        assert runtime["runtime_role_visibility"] == "none"
        assert runtime["metric_depth_sensor_consumed_episodes"] == 0
        assert runtime["shared_A_all_hashes_ok"] and runtime["shared_A_total_diffusion_samples"] == 0
        assert runtime["runtime_failure_plans"] == 0
        with (folder / "plans.jsonl").open() as stream:
            first_plan = json.loads(next(stream))
        initial_plans.setdefault(policy, []).append(first_plan["selected_trajectory"])
        initializations.append(init["settled"])
        picture_hash = hashlib.sha256(np.asarray(Image.open(folder / "first_query_rgb.png")).tobytes()).hexdigest()
        assert picture_hash == terminal["first_query_rgb_sha256"]
        image_hashes.append(picture_hash)
        count, path, micro_path, tilt = 0, 0., 0., 0.
        previous = np.array(init["settled"]["position"])
        solves = []
        for line in (folder / "physical_actions.jsonl").open():
            action = json.loads(line)
            assert action["action"] == count and action["navmesh_motion_queries"] == 0
            close(action["before"]["position"], previous)
            assert len(action["substeps"]) == 24
            assert -1e-6 <= action["command_speed_mps"] <= .376+1e-6
            assert abs(action["command_yaw_rps"]) <= math.pi/4+1e-6
            after = np.array(action["after"]["position"])
            distance = float(np.linalg.norm((after-previous)[[0,2]]))
            close(distance, action["actual_translation_m"])
            path += distance
            for substep in action["substeps"]:
                current = np.array(substep["position"])
                micro_path += float(np.linalg.norm((current-previous)[[0,2]]))
                previous = current
                tilt = max(tilt, substep["tilt_deg"])
            close(action["after"]["position"], previous)
            if action["mpc_solve_s"] is not None:
                solves.append(action["mpc_solve_s"])
            count += 1
        assert count == terminal["steps"] and tilt < 5
        close(terminal["end_position"], previous-[0,.75,0])
        close(terminal["actual_path_len_m"], path)
        goal = np.array(terminal["goal_xz_evaluator_only"])
        distance = float(np.linalg.norm(previous[[0,2]]-goal))
        close(distance, terminal["final_goal_dist_m"])
        success = int(distance < 1)
        assert success == terminal["reached"]
        result = {"policy":policy, "tracker":tracker, "reached":success, "actions":count,
                  "final_distance_m":distance, "command_boundary_path_m":path,
                  "physics_substep_path_m":micro_path, "max_tilt_deg":tilt,
                  "mpc_solve_median_ms":None if not solves else float(np.median(solves)*1000)}
        results.append(result)
    assert all(state == initializations[0] for state in initializations)
    assert len(set(image_hashes)) == 1
    planner_pairing = {policy:all(np.array_equal(plan, plans[0]) for plan in plans)
                       for policy, plans in initial_plans.items()}
    report = {"verified":True, "independent_histories":1, "results":results,
              "initial_body_state_exactly_paired":True, "first_RGB_exactly_paired":True,
              "first_selected_plan_equal_within_policy":planner_pairing,
              "runtime_role_hidden":True, "metric_depth_sensor_consumed":False,
              "scope":"consumed query-stage cylinder proxy diagnostic; not full physics A or formal SR",
              "inferential_statistical_claim":False,
              "source_note":"motion-interface source and trace checked; no claim of Dingo dynamics"}
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
