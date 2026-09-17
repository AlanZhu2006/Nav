#!/usr/bin/env python3
"""Describe every attempted Bullet arm, including invalid and unrun arms.

Offline only. Never turn a proxy failure into navigation SR or pool only the
surviving histories as an unbiased estimate of performance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm(run, policy, tracker):
    folder = run / "evaluation" / f"{policy}__{tracker}"
    row = dict(policy=policy, tracker=tracker, folder=str(folder))
    actions_file = folder / "physical_actions.jsonl"
    if not actions_file.exists():
        return dict(row, status="not_executed", reached=None)
    actions = [json.loads(line) for line in actions_file.read_text().splitlines()]
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_text().splitlines()]
    zero_plans = {plan["plan"] for plan in plans
                  if np.all(np.asarray(plan["selected_trajectory"])[:, :2] == 0)}
    zero_commands = [action for action in actions if action["plan"] in zero_plans]
    init = read(folder / "initialization.json")["settled"]
    previous = np.asarray(init["position"])
    path = commanded = 0.0
    maximum_tilt = float(init["tilt_deg"])
    blocked = high_contact_commands = 0
    last_high_contacts = []
    positions = [previous[[0, 2]]]
    for index, action in enumerate(actions):
        assert action["action"] == index
        assert action["navmesh_motion_queries"] == 0
        assert len(action["substeps"]) == 24
        np.testing.assert_allclose(action["before"]["position"], previous, rtol=0, atol=1e-8)
        after = np.asarray(action["after"]["position"])
        distance = float(np.linalg.norm((after - previous)[[0, 2]]))
        np.testing.assert_allclose(distance, action["actual_translation_m"], rtol=0, atol=1e-8)
        path += distance
        commanded += float(action["command_speed_mps"]) * 0.1
        blocked += int(action["command_speed_mps"] > 0.1 and distance < 0.005)
        high_contacts = []
        for substep in action["substeps"]:
            maximum_tilt = max(maximum_tilt, substep["tilt_deg"])
            # The stage contains both floor and furniture. This is explicitly a
            # descriptive height/force filter, NOT an obstacle collision label.
            nominal_base_y = substep["position"][1] - 0.75
            high_contacts.extend(c for c in substep["contacts"]
                                 if c["normal_force_n"] > 1.0
                                 and min(c["point_a"][1], c["point_b"][1]) > nominal_base_y + 0.30)
        high_contact_commands += bool(high_contacts)
        if high_contacts:
            last_high_contacts = high_contacts[-4:]
        previous = after
        positions.append(after[[0, 2]])
    terminal_file = folder / "terminal_measurements.json"
    terminal = read(terminal_file)[0] if terminal_file.exists() else None
    if terminal is not None:
        assert maximum_tilt < 5.0
        assert len(actions) == terminal["steps"]
        np.testing.assert_allclose(path, terminal["actual_path_len_m"], rtol=0, atol=1e-8)
        goal = np.asarray(terminal["goal_xz_evaluator_only"])
        distances = np.linalg.norm(np.asarray(positions) - goal, axis=1)
        assert int(distances[-1] < 1.0) == terminal["reached"]
        row.update(status="navigation_complete", reached=terminal["reached"],
                   final_distance_m=float(distances[-1]), min_distance_m=float(distances.min()))
        runtime = read(folder / "summary.json")
        assert runtime["runtime_role_visibility"] == "none"
        assert runtime["metric_depth_sensor_consumed_episodes"] == 0
        assert runtime["shared_A_all_hashes_ok"]
        assert runtime["shared_A_total_diffusion_samples"] == 0
        assert runtime["runtime_failure_plans"] == 0
        row["completed_runtime_receipts_checked"] = True
    else:
        row.update(status="invalid_physics" if maximum_tilt >= 5.0 else "incomplete_other",
                   reached=None)
    row.update(actions=len(actions), actual_path_m=path, integrated_speed_command_m=commanded,
               commanded_forward_but_less_than_5mm=blocked, max_tilt_deg=maximum_tilt,
               exact_zero_XY_plans=len(zero_plans), selected_plans=len(plans),
               commands_on_zero_XY_plans=len(zero_commands),
               actual_motion_on_zero_XY_plans_m=sum(a["actual_translation_m"] for a in zero_commands),
               first_zero_XY_command=None if not zero_commands else {
                   key: zero_commands[0][key] for key in
                   ("action", "plan", "command_speed_mps", "command_yaw_rps", "actual_translation_m")},
               commands_with_forced_contact_above_nominal_base_30cm=high_contact_commands,
               last_high_contact_examples=last_high_contacts,
               action_receipt_sha256=sha(actions_file), navmesh_motion_queries=0)
    return row


def history_report(run):
    manifest = read(run / "manifest.json")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in manifest["sources"].items():
        assert sha(run / "sources" / Path(name).relative_to(repo)) == digest
    verification_file = run / "independent_verification.json"
    verification = read(verification_file) if verification_file.exists() else None
    rows = [arm(run, *variant) for variant in manifest["variants"]]
    initializations, images, plans = [], [], {}
    for row in rows:
        folder = Path(row["folder"])
        if row["status"] == "not_executed":
            continue
        initializations.append(read(folder / "initialization.json")["settled"])
        physics = read(folder / "physics_summary.json")
        images.append(physics["first_query_rgb_sha256"])
        with (folder / "plans.jsonl").open() as stream:
            plans.setdefault(row["policy"], []).append(json.loads(next(stream))["selected_trajectory"])
    assert all(state == initializations[0] for state in initializations)
    assert len(set(images)) == 1
    pairing = {policy: all(plan == candidates[0] for plan in candidates)
               for policy, candidates in plans.items()}
    assert all(pairing.values())
    return dict(scene=manifest["scene"], run=str(run),
                four_arm_verification_passed=bool(verification and verification["verified"]),
                attempted_arm_initial_state_and_RGB_equal=True,
                first_selected_plan_equal_within_policy=pairing, arms=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path,
                        help="Separate post-batch attempts that completed previously skipped arms")
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = read(batch / "manifest.json")
    if not (batch / "summary.json").exists():
        raise RuntimeError("Wait for the batch to finish before freezing this report")
    runs = [Path(manifest["reused_history_zero"])] + [
        batch / f"history_{h['index']:02d}_{h['scene']}" for h in manifest["histories"]]
    histories = [history_report(run) for run in runs]
    completions = []
    if args.completion_root:
        import ast
        for index in (1, 3):
            original = runs[index]
            current = args.completion_root.resolve() / original.name
            summary = read(current / "summary.json")
            assert summary["all_arms_attempted"] and not summary["changed_sources"]
            old_source = original / "sources/MemNavData/run_habitat_bullet_query_local.py"
            new_source = current / "sources/MemNavData/run_habitat_bullet_query_local.py"
            implementations = [ast.dump(next(n for n in ast.parse(p.read_text()).body
                                            if isinstance(n, ast.FunctionDef) and n.name == "evaluate"),
                                        include_attributes=False) for p in (old_source, new_source)]
            assert implementations[0] == implementations[1]
            row = history_report(current)
            repeated = {}
            for name in ("physical_actions.jsonl", "plans.jsonl"):
                old = original / "evaluation/native__pure_pursuit" / name
                new = current / "evaluation/native__pure_pursuit" / name
                repeated[name] = sha(old) == sha(new)
            row.update(original_attempt=str(original), evaluate_function_unchanged=True,
                       repeated_native_PP_files_identical=repeated)
            completions.append(row)
    counts = {}
    for history in histories:
        for row in history["arms"]:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = dict(
        histories=histories, completion_attempts=completions, arm_status_counts=counts,
        complete_four_arm_histories=sum(h["four_arm_verification_passed"] for h in histories),
        planned_histories=len(histories),
        batch_summary_sha256=sha(batch / "summary.json"),
        no_unbiased_population_SR_claim=True,
        notes=[
            "Invalid and unrun arms are retained, not assigned reached=0 and not silently omitted.",
            "Complete-case SR would condition on policy-dependent physical validity.",
            "High-contact filter is post hoc: force>1 N and contact height>nominal base+0.30 m.",
            "Contact height/force examples do not identify an object or measure collision rate.",
            "Exact zero XY is a logged trajectory property, not an arrival or semantic STOP label.",
            "GT pose and arrival scoring remain; only NavMesh motion filtering is removed.",
            "Query-only test using shared old A histories, not full physical A recollection.",
        ])
    if completions:
        coverage = {h["scene"]: h for h in histories}
        coverage.update({h["scene"]: h for h in completions})
        coverage_counts = {}
        for history in coverage.values():
            for row in history["arms"]:
                coverage_counts[row["status"]] = coverage_counts.get(row["status"], 0) + 1
        report["coverage_status_counts_after_completion"] = coverage_counts
        report["completion_note"] = "Original attempts retained above; completion runs are NOT new independent histories."
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"arm_status_counts": counts,
                      "complete_four_arm_histories": report["complete_four_arm_histories"],
                      "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
