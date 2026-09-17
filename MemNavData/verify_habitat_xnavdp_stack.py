#!/usr/bin/env python3
"""Read-only measurement/pairing audit of completed X-stack diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".diagnostics/habitat_physics_executor_20260908/alignment_pair_v2"
OLD_X = ROOT / ".diagnostics/habitat_physics_executor_20260908/xnavdp_stack_v2"


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)


def audit(run):
    manifest, summary = read(run/"manifest.json"), read(run/"summary.json")
    assert summary["all_arms_attempted"] and not summary["changed_sources"]
    for name, digest in manifest["sources"].items():
        assert sha(run/"sources"/Path(name).relative_to(ROOT)) == digest
    results, first_images, inits, first_proofs, xplans = [], [], [], [], []
    for policy, tracker in manifest["variants"]:
        name = f"{policy}__{tracker}"
        folder = run/"evaluation"/name
        init = read(folder/"initialization.json")["settled"]
        physics = read(folder/"physics_summary.json")
        plans, actions = rows(folder/"plans.jsonl"), rows(folder/"physical_actions.jsonl")
        digest = hashlib.sha256(np.asarray(Image.open(folder/"first_query_rgb.png")).tobytes()).hexdigest()
        assert digest == physics["first_query_rgb_sha256"]
        first_images.append(digest)
        inits.append(init)
        first = plans[0]
        assert first["metric_depth_sensor_consumed"] is False
        first_proofs.append([first["anchor"], first["memory_controller_pointgoal"],
                             first["monocular_depth_receipt"]["depth_png_sha256"]])
        if policy.startswith("cec_x"):
            xplans.extend(plans)
            if "xnavdp_rgb_contract" in manifest:
                expected_contract = manifest.get("xnavdp_rgb_contracts", {}).get(
                    policy, manifest["xnavdp_rgb_contract"])
                for plan in plans:
                    if plan["pose_controller"] == "xnavdp_point_posttrain":
                        assert plan["xnavdp_rgb_receipt"]["contract"] == expected_contract
                # First observations are identical across arms. Check actor bytes,
                # not merely identical JPEG transport bytes (the old blind spot).
                original_image = np.asarray(Image.open(folder/"first_query_rgb.png").convert("RGB"))
                import io
                wire = io.BytesIO()
                Image.fromarray(original_image).save(wire, format="JPEG", quality=95)
                assert sha_bytes(wire.getvalue()) == first["xnavdp_rgb_receipt"]["wire_image_sha256"]
                decoded = np.asarray(Image.open(io.BytesIO(wire.getvalue())).convert("RGB"))
                actor = decoded if expected_contract == "rgb_v1" else decoded[..., ::-1]
                assert sha_bytes(actor.tobytes()) == first["xnavdp_rgb_receipt"]["actor_image_sha256"]
        previous, path, max_tilt, reverses, xcommands = np.array(init["position"]), 0., 0., 0, 0
        solve_times = []
        for index, action in enumerate(actions):
            assert action["action"] == index and action["navmesh_motion_queries"] == 0
            close(action["before"]["position"], previous)
            assert len(action["substeps"]) == 24
            assert abs(action["command_speed_mps"]) <= .376+1e-6
            assert abs(action["command_yaw_rps"]) <= math.pi/4+1e-6
            current = np.array(action["after"]["position"])
            distance = float(np.linalg.norm((current-previous)[[0, 2]]))
            close(distance, action["actual_translation_m"])
            path += distance
            previous = current
            max_tilt = max(max_tilt, *(s["tilt_deg"] for s in action["substeps"]))
            reverses += action["command_speed_mps"] < -1e-4
            x = action.get("xmpc_receipt")
            if x:
                assert x["solver_status"] == 0
                assert 0 <= x["consumed_index"] < 8
                close(x["controls"][x["consumed_index"]],
                      [action["command_speed_mps"], action["command_yaw_rps"]])
                xcommands += 1
            if action["mpc_solve_s"] is not None:
                solve_times.append(action["mpc_solve_s"])
        assert len(actions) == physics["actions"] and len(plans) == physics["plans"]
        terminal = read(folder/"terminal_measurements.json")[0] if (folder/"terminal_measurements.json").exists() else None
        if terminal:
            assert max_tilt < 5
            close(terminal["end_position"], previous-[0, .75, 0])
            close(terminal["actual_path_len_m"], path)
            distance = float(np.linalg.norm(previous[[0, 2]]-terminal["goal_xz_evaluator_only"]))
            close(distance, terminal["final_goal_dist_m"])
            assert terminal["reached"] == int(distance < 1)
            runtime = read(folder/"summary.json")
            assert runtime["runtime_role_visibility"] == "none"
            assert runtime["metric_depth_sensor_consumed_episodes"] == 0
            assert runtime["runtime_failure_plans"] == 0
            assert runtime["shared_A_all_hashes_ok"] and runtime["shared_A_total_diffusion_samples"] == 0
        else:
            assert max_tilt >= 5 and read(run/"failure.json")["type"] == "InvalidPhysicalArms"
        assert all(p["monocular_depth_receipt"]["metric_depth_sensor_consumed"] is False for p in plans)
        original_equal = None
        if policy == "cec_aligned":
            old_actions = rows(OLD/run.name/"evaluation"/name/"physical_actions.jsonl")
            keys = ("before", "after", "command_speed_mps", "command_yaw_rps", "substeps")
            original_equal = len(old_actions) == len(actions) and all(
                all(a[k] == b[k] for k in keys) for a, b in zip(actions, old_actions))
        legacy_equal = None
        if policy == "cec_x_legacy":
            old_actions = rows(OLD_X/run.name/"evaluation/cec_x__xmpc/physical_actions.jsonl")
            keys = ("before", "after", "command_speed_mps", "command_yaw_rps", "substeps")
            legacy_equal = len(old_actions) == len(actions) and all(
                all(a[k] == b[k] for k in keys) for a, b in zip(actions, old_actions))
        results.append(dict(arm=name, reached=None if terminal is None else terminal["reached"],
            status="invalid_physics" if terminal is None else "navigation_complete",
            commands=len(actions), plans=len(plans), actual_path_m=path, max_tilt_deg=max_tilt,
            final_distance_m=None if terminal is None else distance, reverse_commands=int(reverses),
            xmpc_commands=xcommands, original_aligned_baseline_exact=original_equal,
            old_legacy_X_exact=legacy_equal,
            zero_xy_plans=sum(bool(np.max(np.abs(np.array(p["selected_trajectory"])[:, :2])) < 1e-8) for p in plans),
            mpc_solve_median_ms=None if not solve_times else float(np.median(solve_times)*1000)))
    assert len(set(first_images)) == 1 and all(i == inits[0] for i in inits)
    # Same first certificate, direction and depth, even though policy outputs differ.
    assert all(p == first_proofs[0] for p in first_proofs)
    xcalls = rows(run/"xnavdp_requests.jsonl")
    for call in xcalls:
        assert call["metric_depth_sensor_consumed"] is False
        assert call["xnavdp_input_diagnostic"]["sensor_depth_uploaded"] is False
        close(call["xnavdp_input_diagnostic"]["goal_before_processing"],
              call["xnavdp_input_diagnostic"]["goal_after_processing"])
        assert call["rtc_robot_state_used"] is True
        assert call["checkpoint_load_audit"]["audited"] is True
    assert len(xcalls) == sum(p["pose_controller"] == "xnavdp_point_posttrain" for p in xplans)
    if "xnavdp_rgb_contract" in manifest:
        rgb = rows(run/"xnavdp_rgb_observations.jsonl")
        for policy, _ in manifest["variants"]:
            if not policy.startswith("cec_x"):
                continue
            contract = manifest.get("xnavdp_rgb_contracts", {}).get(policy, manifest["xnavdp_rgb_contract"])
            selected = [r for r in rgb if r["contract"] == contract]
            replay = [r for r in selected if r["endpoint"] == "/memory_replay_step"]
            assert replay and [r["history_frame_count"] for r in selected] == [[i] for i in range(1, len(selected)+1)]
        assert len([r for r in rgb if r["endpoint"] == "/pointgoal_step"]) == len(xcalls)
    return dict(scene=manifest["scene"], verified=True, same_initial_RGB_depth_certificate=True,
                x_queries=len(xcalls), results=results)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root/"xnavdp_independent_verification.json"
    if target.exists():
        raise FileExistsError(target)
    if (root/"evaluation").exists():
        result = audit(root)
    else:
        assert read(root/"summary.json")["all_histories_attempted"]
        histories = sorted(root.glob("history_*"))
        assert len(histories) == 4
        result = dict(verified=True, histories=[audit(h) for h in histories],
                      formal_SR_claim=False, interpretation="stack comparison, not an MPC-only or real-robot proof")
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
