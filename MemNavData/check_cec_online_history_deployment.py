#!/usr/bin/env python3
"""Recorded-RGB integration check of the actual deployment model interfaces.

Starts only two private GPU services. No ROS, Jetson client, actuator, new
dataset collection or navigation rollout. Existing field services are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REAL = Path("/home/asus/Research/MemNav-RealWorld")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def listening(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--survey-checkpoint", action="store_true",
                        help="Also save/reset/restore an isolated recorded-RGB Survey")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ports = (21560, 21561)
    if any(listening(port) for port in ports):
        raise RuntimeError("isolated check ports already occupied")
    system = json.loads((REAL / "deployment/config/system.json").read_text())
    gpu = system["sites"]["gpu"]
    models = gpu["models"]
    py = gpu["python"]
    # The current resident wrapper requires an explicit resolved run config.
    # Derive it from current configuration into this check's private namespace.
    isolated_system = json.loads(json.dumps(system))
    isolated_system["sites"]["gpu"]["runtime_root"] = str(out / "resident_runtime")
    isolated_system["sites"]["gpu"]["ports"].update(memnav=ports[0], navdp=ports[1])
    isolated_system["stack"]["cec"].update(
        historical_depth_source="online_history", eager_depth_cache=False)
    system_path = out / "system.json"
    dump(system_path, isolated_system)
    experiment = json.loads((REAL / "deployment/config/experiments/fullmono_imagegoal.json").read_text())
    experiment["system_config"] = str(system_path)
    fixture = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/gxdoqLR6rwA/episode_0000"
    fixture_meta = json.loads((fixture / "role_pairs.json").read_text())
    fixture_goal = next(q for q in fixture_meta["pairs"][0]["queries"] if q["analysis_role"] == "revisit")
    experiment["experiment"]["id"] = "gem-module-recorded-rgb-check"
    experiment["experiment"]["navigation"]["image_goal"] = str(fixture / fixture_goal["goal_rgb"])
    experiment["experiment"]["arrival"]["image_goal"] = str(fixture / fixture_goal["goal_rgb"])
    experiment_path = out / "experiment.json"
    dump(experiment_path, experiment)
    resolved = out / "resolved.json"
    subprocess.run([py, str(REAL / "deployment/runtime_config.py"), "resolve",
                    "--config", str(experiment_path), "--output", str(resolved)],
                   cwd=REAL, env=dict(os.environ, PYTHONPATH=str(REAL)), check=True)
    common = dict(os.environ, PYTHONUNBUFFERED="1", NAVDP_DISABLE_VIDEO="1",
                  MEMNAV_RESIDENT_CONFIG=str(resolved),
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  LINGBOT_REPO=models["lingbot_repository"], LINGBOT_WEIGHTS=models["lingbot_weights"],
                  MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
                  MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
                  MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
                  MEMNAV_REPORT_TO="none")
    common["PYTHONPATH"] = ":".join([str(ROOT), str(REAL), models["dependency_root"],
        models["lightglue_repository"], models["internnav_root"] + "/src/diffusion-policy"])
    commands = [
        [py, "-u", str(REAL / "deployment/gpu/resident_memnav_server.py"),
         str(ROOT / "NavDP/baselines/memnav/memnav_server.py"),
         "--host", "127.0.0.1", "--port", str(ports[0]),
         "--checkpoint", models["memnav_checkpoint"], "--internnav_root", models["internnav_root"],
         "--num_samples", "16", "--exclude_recent", "32", "--retrieval", "raw",
         "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
         "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60",
         "--flow_gate", "auto", "--buffer_root", str(out / "buffer"),
         "--certified_relocalization", "--certified_reference_depth_source", "online_history",
         "--lightglue_repo", models["lightglue_repository"],
         "--lightglue_dependency_root", models["dependency_root"], "--lightglue_max_keypoints", "2048"],
        [py, "-u", str(REAL / "baselines/navdp/navdp_server.py"), "--port", str(ports[1]),
         "--checkpoint", models["navdp_checkpoint"], "--depth_source", "monocular_sidecar",
         "--require_monocular_depth_transaction", "--monocular_depth_url",
         f"http://127.0.0.1:{ports[0]}/monocular_depth_query"],
    ]
    processes, logs = [], []
    try:
        for label, port, command in zip(("memnav", "navdp"), ports, commands):
            log = (out / f"{label}.log").open("x")
            logs.append(log)
            env = dict(common)
            if label == "navdp":
                env["PYTHONPATH"] = str(REAL / "baselines/navdp") + ":" + env["PYTHONPATH"]
            process = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            dump(out / "owned_processes.json", [p.pid for p in processes])
            deadline = time.monotonic() + 600
            while not listening(port):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"{label} startup failed; see its log")
                time.sleep(1)
            print(f"private {label} ready pid={process.pid}", flush=True)
        sys.path.insert(0, str(REAL))
        from deployment.gpu.realworld_cec_hub import CecHybridRouter, UpstreamConfig
        from deployment.gpu.episodic_dataset import EpisodicDatasetStore
        import requests
        import numpy as np
        bench = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/gxdoqLR6rwA/episode_0000"
        meta = json.loads((bench / "role_pairs.json").read_text())
        history = Path(meta["online_a_episode"])
        receipt = json.loads((history / "receipt.json").read_text())
        trace = json.loads((history / "online_a_trace.json").read_text())
        intrinsic = json.loads(subprocess.check_output([
            "/home/asus/miniconda3/envs/habitat/bin/python", "-c",
            "import pandas as pd,json,sys,numpy as np; r=pd.read_parquet(sys.argv[1]).iloc[0]; print(json.dumps(np.asarray(list(r['observation.camera_intrinsic']),dtype=float).tolist()))",
            str(Path(receipt["source_episode"]) / "data/chunk-000/episode_000000.parquet")], text=True))
        store = None
        if args.survey_checkpoint:
            store = EpisodicDatasetStore(out / "resident_runtime/episodic_datasets")
            store.start("gem_recorded_rgb_check", metadata={
                "collection_mode": "local_raw_survey_v1",
                "goal_selection_contract": "survey_goal_after_capture_v1",
                "goal_candidates_required": False, "engineering_only": True,
                "raw_manifest_sha256": hashlib.sha256((history / "online_a_trace.json").read_bytes()).hexdigest(),
                "input_provenance": "existing_Habitat_RGB_integration_fixture",
                "physical_robot_experiment": False,
            })
            for i in range(int(meta["online_a_steps"])):
                image = (history / "rgb" / f"{i:06d}.jpg").read_bytes()
                store.append_memory(frame_index=i, image=image, upstream_sha256=hashlib.sha256(image).hexdigest())
            dump(out / "fixture.json", store.seal(protocol={"purpose": "module checkpoint regression"}))
        router = CecHybridRouter(UpstreamConfig(
            memnav_url=f"http://127.0.0.1:{ports[0]}", navdp_url=f"http://127.0.0.1:{ports[1]}",
            camera_height_m=receipt["camera_height_m"], historical_depth_source="online_history"), dataset_store=store)
        reset_payload = {"intrinsic": intrinsic, "seed": trace["episode_seed"],
                              "episode_len": int(meta["online_a_steps"]) + 600,
                              "stop_threshold": -0.5, "batch_size": 1}
        reset = router.reset(reset_payload)
        dump(out / "reset.json", reset)
        started = time.perf_counter()
        if args.survey_checkpoint:
            first_load = router.load_dataset("gem_recorded_rgb_check")
            dump(out / "survey_first_load.json", first_load)
            assert first_load["frames_replayed"] == int(meta["online_a_steps"])
            image = (history / "rgb" / f"{int(meta['online_a_steps']) - 1:06d}.jpg").read_bytes()
            def current_evidence():
                depth = requests.post(f"http://127.0.0.1:{ports[0]}/monocular_depth_query",
                    data={"expected_image_sha256": hashlib.sha256(image).hexdigest()}, timeout=60)
                depth.raise_for_status()
                poses = requests.post(f"http://127.0.0.1:{ports[0]}/causal_pose_trace_query", timeout=60)
                poses.raise_for_status()
                return depth.json(), poses.json()
            before_depth, before_poses = current_evidence()
            router.reset(reset_payload)
            restored = router.load_dataset("gem_recorded_rgb_check")
            dump(out / "survey_restored.json", restored)
            assert restored["frames_replayed"] == 0
            assert restored["frames_restored"] == int(meta["online_a_steps"])
            after_depth, after_poses = current_evidence()
            assert before_poses == after_poses
            for field in ("frame_index", "image_sha256", "depth_png_sha256", "scale_receipt_sha256"):
                assert before_depth[field] == after_depth[field]
            dump(out / "survey_evidence_comparison.json", {"passed": True,
                "all_poses_equal": True, "current_depth_equal": True, "replayed_frames_on_restore": 0})
        else:
            for i in range(int(meta["online_a_steps"])):
                image = (history / "rgb" / f"{i:06d}.jpg").read_bytes()
                router.memory_step(image)
        print(f"recorded RGB replay complete: {router.frames_recorded} frames", flush=True)
        dump(out / "begin_revisit.json", router.begin_revisit(
            query_start_image=image if args.survey_checkpoint else None))
        query = next(q for q in meta["pairs"][0]["queries"] if q["analysis_role"] == "revisit")
        goal = (bench / query["goal_rgb"]).read_bytes()
        plan_start = time.perf_counter()
        plan = router.plan_imagegoal(image=image, goal=goal, form={"seed": "20260803"})
        plan_elapsed = time.perf_counter() - plan_start
        dump(out / "unexecuted_plan.json", plan)
        trajectory = np.asarray(plan["trajectory"], dtype=float)
        if trajectory.shape != (1, 24, 3) or not np.isfinite(trajectory).all():
            raise RuntimeError("NavDP returned an invalid trajectory payload")
        proof = plan["cec_relocalization_trace"]
        if proof["reference_depth_source"] != "online_history" or not plan["cec_takeover"]:
            raise RuntimeError("recorded supported query did not use online-history CEC")
        if proof["reference_depth_cache"]["replayed_frames"] != 0:
            raise RuntimeError("online-history query unexpectedly replayed geometry")
        status = requests.get(f"http://127.0.0.1:{ports[0]}/resident/status", timeout=10).json()
        dump(out / "populated_status.json", status)
        released = requests.post(f"http://127.0.0.1:{ports[0]}/resident/release", timeout=60).json()
        if released["memory_frames"] or released["historical_depth_cached_frames"] or released["historical_depth_cache_bytes"]:
            raise RuntimeError("private episode release left history depth behind")
        dump(out / "released_status.json", released)
        result = {"verified": True, "scope": "recorded RGB, unexecuted GPU plan; no robot navigation",
                  "survey_checkpoint_verified": bool(args.survey_checkpoint),
                  "history_frames": meta["online_a_steps"], "reference_depth_source": proof["reference_depth_source"],
                  "replayed_geometry_frames": 0, "cec_takeover": plan["cec_takeover"],
                  "trajectory_shape": list(trajectory.shape), "trajectory_points": trajectory.shape[-2],
                  "plan_wall_s": plan_elapsed,
                  "elapsed_after_reset_s": time.perf_counter() - started,
                  "historical_depth_cache_bytes_before_release": status["historical_depth_cache_bytes"],
                  "historical_depth_cache_bytes_after_release": released["historical_depth_cache_bytes"],
                  "motion_commands_sent": 0}
        dump(out / "verification.json", result)
        print(json.dumps(result), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
