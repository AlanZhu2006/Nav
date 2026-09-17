"""Paired retrieval-image interface pilot on four already consumed queries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData import run_low_covisibility_navigation_local as coverage

HERE = Path(__file__).resolve()
ARMS = ("native", "dino_imagegoal", "raw_fixed", "cec")
SCOPE = "consumed two-scene retrieval-interface pilot; not a RANa reproduction or confirmation"
PROTOCOL = ROOT / "MemNavData/DINO_IMAGEGOAL_LOCAL_PROTOCOL_20260910.md"


def load(path):
    return json.loads(Path(path).read_text())


def command(inputs, index, arm, out, mem, nav):
    if arm not in ARMS:
        raise ValueError(arm)
    cmd = coverage.command(inputs, index, "native" if arm == "dino_imagegoal" else arm,
                           out, mem, nav)
    cmd[2:4] = [str(HERE), "eval"]
    return cmd


def evaluate_query():
    import eval_shared_online_role_pairs as shared
    arm = os.environ["DINO_IMAGEGOAL_ARM"]
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "dino_imagegoal" and not shared.args.contract_dry_run:
        from MemNavData.dino_imagegoal_substitution import ImageGoalSubstitution
        inputs = Path(os.environ["LOW_COVIS_INPUTS"])
        index = int(os.environ["LOW_COVIS_QUERY_INDEX"])
        *_, frozen = coverage.task(inputs, index)
        hashes = frozen["receipt"]["rgb_frame_hashes"]
        images = [frozen["source"] / "rgb" / f"{i:06d}.jpg" for i in range(len(hashes))]
        # No pose, role, support or target-location metadata enters this hook.
        ImageGoalSubstitution(shared.base, images, hashes, shared.args.out).install()
    coverage.evaluate_query()


def run(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, execution_environment, private_servers, run_child)
    inputs, out = args.inputs.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    for item in load(inputs / "manifest.json")["files"]:
        assert sha(inputs / item["path"]) == item["sha256"], item["path"]
    schedule = [dict(query_index=i, arms=list(ARMS[i:] + ARMS[:i])) for i in range(4)]
    paths = [HERE, PROTOCOL, ROOT / "MemNavData/dino_imagegoal_substitution.py",
        Path(coverage.__file__), ROOT / "MemNavData/run_repaired_fullmono_local.py",
        ROOT / "MemNavData/run_habitat_minimal_repair_local.py",
        ROOT / "MemNavData/eval_2leg_habitat.py",
        ROOT / "MemNavData/eval_shared_online_role_pairs.py",
        ROOT / "MemNavData/certified_relocalization_runtime.py",
        ROOT / "MemNavData/certified_relocalization_contract.py",
        ROOT / "MemNavData/bounded_pursuit.py", ROOT / "MemNavData/navdp_front_goal_adapter.py",
        ROOT / "NavDP/baselines/memnav/memnav_server.py",
        ROOT / "NavDP/baselines/memnav/policy_agent.py",
        ROOT / "MemNavData/navdp_depth_raster_audit_server.py",
        ROOT / "MemNavData/navdp_execution_audit_server.py"]
    hashes = {str(p): sha(p) for p in paths}
    dump(out / "manifest.json", dict(scope=SCOPE, inputs=str(inputs),
        input_manifest_sha256=sha(inputs / "manifest.json"), source_hashes=hashes,
        schedule=schedule, max_steps=600, exec_horizon=8, success_radius_m=1.0,
        original_goal_used_for_scoring=True, history_source="old actual mono A",
        runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on",
        candidate_domain="DINO image and CEC: frame>=8 to first-query boundary",
        raw_fixed_domain="unchanged legacy amargin=39/exclude_recent=32",
        formal_full_rana_reproduction=False))
    for arm in ARMS:
        env = dict(execution_environment(), DINO_IMAGEGOAL_ARM=arm)
        run_child(command(inputs, 0, arm, out / "dry", args.mem_port, args.nav_port)
                  + ["--contract_dry_run"], out / "logs" / f"dry_{arm}.log", environment=env)
    summary = dict(scope=SCOPE, completed=False, queries=[])
    dump(out / "summary.json", summary)
    pose_log = out / "lingbot_pose_readout.jsonl"
    with private_servers(out, args.mem_port, args.nav_port):
        for scheduled in schedule:
            index = scheduled["query_index"]
            item, _, query, _, _, _ = coverage.task(inputs, index)
            for arm in scheduled["arms"]:
                if shutil.disk_usage(out).free < 1_000_000_000:
                    raise RuntimeError("less than 1 GB free; preserve results before next rollout")
                dest = out / "evaluation" / f"q{index:02d}" / arm
                env = dict(execution_environment(), DINO_IMAGEGOAL_ARM=arm,
                    LOW_COVIS_INPUTS=str(inputs), LOW_COVIS_QUERY_INDEX=str(index))
                cmd = command(inputs, index, arm, dest, args.mem_port, args.nav_port)
                dump(out / "progress.json", dict(query_index=index, arm=arm,
                    completed_rollouts=len(summary["queries"]), command=cmd))
                print(f"START query={index} scene={item['scene']} arm={arm}", flush=True)
                offset = pose_log.stat().st_size if pose_log.exists() else 0
                seconds = run_child(cmd, out / "logs" / f"q{index:02d}_{arm}.log", environment=env)
                with pose_log.open() as stream:
                    stream.seek(offset)
                    dump(dest / "lingbot_frame_poses.json", [json.loads(l) for l in stream if l.strip()])
                terminal = load(dest / "terminal_measurements.json")
                assert len(terminal) == 1
                # The reused evaluator retains its prior provenance string;
                # identify this experiment in a separate immutable receipt.
                dump(dest / "experiment_identity.json", dict(scope=SCOPE, arm=arm,
                    source_manifest_sha256=sha(out / "manifest.json")))
                row = dict(terminal[0], query_index=index, history=item["history"],
                    scene=item["scene"], role=query["analysis_role"], arm=arm,
                    directory=str(dest), wall_seconds=seconds,
                    counts=load(dest / "query_plans.json")["counts"])
                summary["queries"].append(row)
                dump(out / "summary.json", summary)
                print(f"DONE query={index} arm={arm} SR={row['reached']} steps={row['steps']}", flush=True)
    summary["completed"] = True
    dump(out / "summary.json", summary)
    run_child([HAB_PY, str(HERE), "verify", "--out", str(out)],
              out / "logs/verification.log", environment=execution_environment())


def verify(out):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out / "manifest.json"), load(out / "summary.json")
    assert summary["completed"] and len(summary["queries"]) == 16
    assert all(sha(Path(p)) == h for p, h in manifest["source_hashes"].items())
    assert sha(Path(manifest["inputs"]) / "manifest.json") == manifest["input_manifest_sha256"]
    results, pairs = [], {}
    for row in summary["queries"]:
        checked, evidence, plans, actions = verify_rollout(row)
        folder = Path(row["directory"])
        data = load(folder / "query_plans.json")
        *_, frozen = coverage.task(Path(manifest["inputs"]), row["query_index"])
        np.testing.assert_array_equal(actions[0]["position_before"], frozen["trace"]["end_position"])
        assert actions[0]["yaw_before"] == frozen["trace"]["end_yaw"]
        assert data["replay"]["all_rgb_hashes_verified"]
        assert data["replay"]["diffusion_samples_during_replay"] == 0
        assert data["rollout_traces"]["legA"] == frozen["trace"]["poses"]
        requests = read_rows(folder / "memory_http_boundary.jsonl")
        forbidden = {"analysis_role", "q_eligible", "floor_position", "covis_curve",
            "executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m"}
        assert all(not forbidden.intersection(r["sent_fields"]) for r in requests)
        if row["arm"] == "dino_imagegoal":
            selection = load(folder / "retrieval_selection.json")
            substituted = read_rows(folder / "imagegoal_substitution.jsonl")
            anchor = selection["anchor"]
            assert 8 <= anchor <= selection["candidate_ceiling"] == len(frozen["receipt"]["rgb_frame_hashes"]) - 1
            assert selection["anchor_rgb_sha256"] == frozen["receipt"]["rgb_frame_hashes"][anchor]
            assert sha(folder / "retrieved_imagegoal.jpg") == selection["anchor_rgb_sha256"]
            assert len(substituted) == len(plans)
            assert sum(r["path"] == "/retrieval_probe_step" for r in requests) == 1
            assert not any(r["path"] in ("/certified_relocalize", "/posegoal_query", "/posegoal_step") for r in requests)
            assert checked["heading"]["turn_actions"] == 0
            for sub, plan in zip(substituted, plans):
                assert sub["controller_goal_sha256"] == selection["anchor_rgb_sha256"]
                assert sub["original_goal_sha256"] == selection["original_goal_sha256"]
                audit = plan["receipt"]["execution_input_audit"]
                assert audit["goal_jpeg_sha256"] == selection["anchor_rgb_sha256"]
                assert audit["endpoint"] == "/imagegoal_step"
                assert plan["receipt"]["memory_controller_pointgoal"] is None
                assert plan["receipt"]["pose_controller"] == "navdp_retrieved_imagegoal"
        results.append(dict(checked, query_index=row["query_index"]))
        pairs.setdefault(row["query_index"], {})[row["arm"]] = (row, data, plans, actions)
    for arms in pairs.values():
        assert set(arms) == set(ARMS)
        native = arms["native"]
        for arm, (row, data, plans, actions) in arms.items():
            assert row["first_query_rgb_sha256"] == native[0]["first_query_rgb_sha256"]
            assert row["goal_xz_evaluator_only"] == native[0]["goal_xz_evaluator_only"]
            assert data["replay"] == native[1]["replay"]
            if arm == "cec" and not any(p["receipt"]["revisit_adapter_takeover"] is True for p in plans):
                assert len(plans) == len(native[2]) and len(actions) == len(native[3])
                for p, n in zip(plans, native[2]):
                    for key in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                        np.testing.assert_array_equal(p[key], n[key])
                for a, n in zip(actions, native[3]):
                    np.testing.assert_array_equal(a["actual_position"], n["actual_position"])
                    assert a["actual_yaw"] == n["actual_yaw"]
    dump(out / "independent_verification.json", dict(verified=True, scope=SCOPE, results=results))
    print("VERIFIED 16 paired rollouts; image-only goal replacement audited", flush=True)


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "eval":
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate(query_main=evaluate_query)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--inputs", type=Path)
        parser.add_argument("--mem-port", type=int, default=21910)
        parser.add_argument("--nav-port", type=int, default=21911)
        args = parser.parse_args()
        if mode == "run":
            run(args)
        elif mode == "verify":
            verify(args.out)
        else:
            parser.error("mode must be run, eval, or verify")
