#!/usr/bin/env python3
"""Local consumed-history CEC depth-source experiment; no production changes.

The server/evaluator hooks exist only in child processes launched by this file.
They select an existing depth provider and observe results, not new control.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814"
HERE = Path(__file__).resolve()
SOURCES = ("canonical", "route_sparse")
MEM_PY = os.environ.get("REPAIRED_MEM_PY", "/home/asus/miniconda3/envs/memnav/bin/python")
HAB_PY = os.environ.get("REPAIRED_HAB_PY", "/home/asus/miniconda3/envs/habitat/bin/python")
MEM_CKPT = Path(os.environ.get("REPAIRED_MEM_CKPT", "/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt"))
NAV_CKPT = Path(os.environ.get("REPAIRED_NAV_CKPT", "/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt"))
LINGBOT = Path(os.environ.get("REPAIRED_LINGBOT_REPO", "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map"))


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def install_server_hooks(agent_class):
    """Experimental process only: reset-bound source, unchanged default source."""
    from flask import has_request_context, request
    import torch
    original_reset = agent_class.reset
    original_certificate = agent_class.certified_relocalize

    def reset(self, *a, **kw):
        source = "canonical"
        if has_request_context():
            source = (request.get_json(silent=True) or {}).get(
                "cec_depth_experiment_source", "canonical")
        if source not in SOURCES:
            raise ValueError("unknown experimental depth source")
        self._cec_experiment_source = source
        self._cec_experiment_first = True
        self.certified_route_depth_cache_stride = int(source == "route_sparse")
        return original_reset(self, *a, **kw)

    def certificate(self, *a, **kw):
        source = self._cec_experiment_source
        kw["reference_depth_source"] = source
        first = self._cec_experiment_first
        state = None
        if first:
            from MemNavData.benchmark_cec_dense_cache_equivalence import online_state_digest
            state = online_state_digest(self)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_certificate(self, *a, **kw)
        torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        cache = self._certified_route_reference_depth_cache
        result["depth_experiment"] = {
            "source": source, "first_query": first,
            "certificate_wall_ms": elapsed_ms,
            "writer_stride": self.certified_route_depth_cache_stride,
            "online_cache_frames": len(cache),
            "online_cache_bytes": sum(
                int(d.nbytes + c.nbytes)
                for d, c in cache.values()),
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
        }
        if first:
            result["depth_experiment"]["initial_online_state"] = state
            result["depth_experiment"]["monocular_status"] = self.monocular_depth_status()
        self._cec_experiment_first = False
        return result

    agent_class.reset = reset
    agent_class.certified_relocalize = certificate


def server():
    from policy_agent import MemNavAgent
    install_server_hooks(MemNavAgent)
    runpy.run_path(str(ROOT / "NavDP/baselines/memnav/memnav_server.py"),
                   run_name="__main__")


def evaluate():
    import eval_2leg_habitat as base
    from MemNavData.final14_spl_replay import measurement
    source = os.environ["CEC_EXPERIMENT_SOURCE"]
    assert source in SOURCES
    output = Path(base.args.out)
    original_post = base.requests.post
    original_leg = base.run_policy_leg
    records, certificate_calls = [], []

    def post(url, *a, **kw):
        if url == f"{base.BASE}/navigator_reset":
            kw["json"] = dict(kw["json"], cec_depth_experiment_source=source)
        started = time.perf_counter()
        response = original_post(url, *a, **kw)
        if url == f"{base.BASE}/certified_relocalize":
            response.raise_for_status()
            value = response.json()
            if value.get("reference_depth_source") != source:
                raise RuntimeError("experimental source was not honored")
            certificate_calls.append({
                "query_index": len(records), "response": value,
                "request_wall_ms_including_audit": 1000 * (time.perf_counter() - started),
            })
            with (output / "certificate_calls.jsonl").open("a") as f:
                f.write(json.dumps(certificate_calls[-1], allow_nan=False) + "\n")
        return response

    def observed_leg(*a, **kw):
        started = time.perf_counter()
        leg = original_leg(*a, **kw)
        record = measurement(leg, a[5], a[6])
        record["rollout_wall_s_including_audit"] = time.perf_counter() - started
        record["source"] = source
        records.append(record)
        dump(output / "terminal_measurements.json", records)
        return leg

    base.requests.post = post
    base.run_policy_leg = observed_leg
    runpy.run_path(str(ROOT / "MemNavData/eval_shared_online_role_pairs.py"),
                   run_name="__main__")
    with (output / "metric.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(records) == 2
    for row, record in zip(rows, records):
        assert int(row["reached"]) == record["reached"]
        assert int(row["steps"]) == record["steps"]
        row.update(reference_depth_source=source,
                   actual_path_len_m=record["actual_path_len_m"],
                   spl=record["spl"])
    with (output / "metric_actual.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def preflight():
    from MemNavData.audit_shared_online_role_pairs import audit
    from MemNavData.generate_twoleg import make_sim, render, geodesic
    import numpy as np
    assert audit(BENCH)["ok"]
    manifest = json.loads((BENCH / "manifest.json").read_text())
    for h in manifest["episodes"]:
        receipt = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
        sim = make_sim(receipt["source_asset"], "", agent_radius=0.30)
        try:
            endpoint = np.array(h["online_a_endpoint"]["floor_position"])
            rgb, depth = render(sim, endpoint + [0, 0.5, 0],
                                h["online_a_endpoint"]["yaw_rad"])
            assert rgb.shape[:2] == depth.shape[:2] and np.isfinite(rgb).all()
            for pair in h["pairs"]:
                for q in pair["queries"]:
                    ok, d, _ = geodesic(sim.pathfinder, endpoint, np.array(q["floor_position"]))
                    assert ok and abs(d - q["geodesic_from_a_end_m"]) <= 0.05
            print(json.dumps({"scene": h["scene"], "render": list(rgb.shape),
                              "geodesics_verified": True}), flush=True)
        finally:
            sim.close()


def hab_env():
    env = os.environ.copy()
    vendor = os.environ.get("REPAIRED_HAB_VENDOR", "/home/asus/miniconda3/envs/habitat/lib/python3.9/site-packages/pip/_vendor")
    extra = os.environ.get("REPAIRED_HAB_EXTRA", "")
    pythonpath = f"{ROOT}:{ROOT}/MemNavData:{vendor}"
    if extra:
        pythonpath += f":{extra}"
    env.update(PYTHONPATH=pythonpath,
               PYTHONUNBUFFERED="1", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet")
    return env


def open_port(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_local():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--memnav-port", type=int, default=21560)
    parser.add_argument("--navdp-port", type=int, default=21561)
    parser.add_argument("--max-steps", type=int, default=600)
    cfg = parser.parse_args()
    output = cfg.out.resolve()
    assert not output.exists(), "use a fresh experiment directory"
    assert cfg.memnav_port != cfg.navdp_port
    assert not open_port(cfg.memnav_port) and not open_port(cfg.navdp_port)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    manifest = json.loads((BENCH / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 4
    files = [HERE, ROOT / "MemNavData/eval_2leg_habitat.py",
             ROOT / "MemNavData/eval_shared_online_role_pairs.py",
             ROOT / "MemNavData/generate_twoleg.py",
             ROOT / "MemNavData/final14_spl_replay.py",
             ROOT / "MemNavData/executed_path_metrics.py",
             ROOT / "MemNavData/revisit_bearing_adapter.py",
             ROOT / "MemNavData/certified_relocalization_contract.py",
             ROOT / "MemNavData/certified_relocalization_runtime.py",
             ROOT / "MemNavData/lingbot_pnp_localization.py",
             ROOT / "NavDP/baselines/memnav/memnav_server.py",
             ROOT / "NavDP/baselines/memnav/policy_agent.py",
             ROOT / "NavDP/baselines/navdp/navdp_server.py",
             ROOT / "NavDP/baselines/navdp/policy_agent.py",
             MEM_CKPT, NAV_CKPT, LINGBOT / "weights/lingbot-map-long.pt"]
    contract = {
        "schema": "cec_stream_depth_local_closed_loop_v1_20260907",
        "scope": "consumed internal mechanism; not a formal paper replication",
        "benchmark": str(BENCH), "benchmark_sha256": sha(BENCH / "manifest.json"),
        "histories": [{"scene": h["scene"], "episode": h["episode"],
                       "trace_sha256": h["online_a_trace_sha256"],
                       "arm_order": list(SOURCES[::1 if i % 2 == 0 else -1])}
                      for i, h in enumerate(manifest["episodes"])],
        "queries": 8, "rollouts": 16, "max_steps": cfg.max_steps,
        "exec_horizon": 8, "success_distance_m": 1.0,
        "residual_m": 2.5, "observation_depth": "monocular_sidecar",
        "history_source": "existing actual metric-NavDP-A RGB replay",
        "query_initial_state": "actual recorded A endpoint plus freshly rendered RGB",
        "runtime_role_visible": False, "models_in_one_resident_process_per_model": True,
        "writer_stride": {"canonical": 0, "route_sparse": 1},
        "code_and_weight_sha256": {str(p): sha(p) for p in files},
    }
    dump(output / "manifest.json", contract)
    processes, handles = [], []
    try:
        with (output / "logs/preflight.log").open("x") as log:
            subprocess.run([HAB_PY, "-u", str(HERE), "preflight"],
                           cwd=ROOT, env=hab_env(), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print("Four-scene rendering and query geodesic preflight passed", flush=True)
        common = os.environ.copy()
        common.update(PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                      LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
                      MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
                      MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
                      MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
                      MEMNAV_REPORT_TO="none", NAVDP_DISABLE_VIDEO="1")
        shared_path = f"{ROOT}:{ROOT}/.diagnostics/dependencies/python:{ROOT}/.diagnostics/dependencies/LightGlue:{ROOT}/InternNav/src/diffusion-policy"
        memenv = dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared_path}")
        navenv = dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/navdp:{shared_path}")
        servers = [
            ("memnav", cfg.memnav_port, memenv, [MEM_PY, "-u", str(HERE), "server",
             "--host", "127.0.0.1", "--port", str(cfg.memnav_port),
             "--checkpoint", str(MEM_CKPT), "--internnav_root", str(ROOT / "InternNav"),
             "--num_samples", "16", "--exclude_recent", "32", "--retrieval", "raw",
             "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
             "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60",
             "--flow_gate", "auto", "--buffer_root", str(output / "buffer"),
             "--certified_relocalization", "--lightglue_repo", str(ROOT / ".diagnostics/dependencies/LightGlue"),
             "--lightglue_dependency_root", str(ROOT / ".diagnostics/dependencies/python"),
             "--lightglue_max_keypoints", "2048"]),
            ("navdp", cfg.navdp_port, navenv, [MEM_PY, "-u", str(ROOT / "NavDP/baselines/navdp/navdp_server.py"),
             "--port", str(cfg.navdp_port), "--checkpoint", str(NAV_CKPT),
             "--depth_source", "monocular_sidecar", "--require_monocular_depth_transaction",
             "--monocular_depth_url", f"http://127.0.0.1:{cfg.memnav_port}/monocular_depth_query"]),
        ]
        for name, port, env, command in servers:
            log = (output / f"logs/{name}.log").open("x")
            handles.append(log)
            cwd = output / "runtime" / name
            cwd.mkdir(parents=True)
            p = subprocess.Popen(command, env=env, cwd=cwd, stdout=log,
                                 stderr=subprocess.STDOUT)
            processes.append(p)
            dump(output / "owned_processes.json", [{"pid": x.pid} for x in processes])
            deadline = time.monotonic() + 600
            while not open_port(port):
                if p.poll() is not None:
                    raise RuntimeError(f"{name} startup failed; see its own log")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} startup timeout")
                time.sleep(1)
            print(f"Own {name} ready pid={p.pid} port={port}", flush=True)
        for i, h in enumerate(manifest["episodes"]):
            receipt = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
            for source in SOURCES[::1 if i % 2 == 0 else -1]:
                dest = output / "evaluation" / h["scene"] / source
                dest.parent.mkdir(parents=True, exist_ok=True)
                command = [HAB_PY, "-u", str(HERE), "eval",
                    "--episode_root", str(BENCH / h["scene"]), "--episode_ids", h["episode"],
                    "--scene", receipt["source_asset"], "--scene_identity", h["scene"],
                    "--host", "127.0.0.1", "--port", str(cfg.memnav_port),
                    "--novel_port", str(cfg.navdp_port), "--out", str(dest),
                    "--server_backend", "hybrid_pose", "--hybrid_route", "certified_relocalization",
                    "--revisit_adapter", "verified_bearing_v1", "--navdp_depth_source", "monocular_sidecar",
                    "--success_dist", "1.0", "--max_steps", str(cfg.max_steps), "--exec_horizon", "8",
                    "--trajectory_selector", "server", "--trajectory_selector_scope", "all",
                    "--leg1_mode", "shared_trace", "--leg1_goal_source", "own", "--seed", "0",
                    "--terminal_uturn", "off", "--terminal_visual_refine", "off",
                    "--deterministic_plan_seeds", "--retrieval_override", "off",
                    "--certified_cdec_rescue", "off", "--certified_stagnation_graph", "off",
                    "--revisit_controller", "navdp_mixed", "--role_pair_scope", "consumed_integration"]
                env = dict(hab_env(), CEC_EXPERIMENT_SOURCE=source)
                print(f"START {i + 1}/4 {h['scene']} {source}: Novel + Revisit", flush=True)
                logpath = output / f"logs/{i}_{source}.log"
                with logpath.open("x") as log:
                    subprocess.run(command, env=env, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                print(f"DONE {h['scene']} {source}", flush=True)
        summarize(output)
    except BaseException as e:
        dump(output / "failure.json", {"type": type(e).__name__, "error": str(e)})
        raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for h in handles:
            h.close()


def summarize(output):
    """Recount endpoints and physical paths without using the measurement helper."""
    manifest = json.loads((output / "manifest.json").read_text())
    all_rows, pairs = [], []
    for history in manifest["histories"]:
        collected = {}
        for source in SOURCES:
            folder = output / "evaluation" / history["scene"] / source
            with (folder / "metric_actual.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            terminals = json.loads((folder / "terminal_measurements.json").read_text())
            calls = [json.loads(x) for x in (folder / "certificate_calls.jsonl").read_text().splitlines()]
            assert len(rows) == len(terminals) == 2
            for index, (row, terminal) in enumerate(zip(rows, terminals)):
                payload = json.loads((folder / f"{row['episode']}_{row['query_id']}_plans.json").read_text())
                trace = payload["rollout_traces"]["query"]
                count = int(row["steps"])
                points = [p for p in trace if int(p["step"]) <= count]
                assert [p["step"] for p in points] in (list(range(count)), list(range(count + 1)))
                xz = [(p["x"], p["z"]) for p in points]
                end = terminal["end_position"]
                if len(points) == count:
                    xz.append((end[0], end[2]))
                else:
                    assert math.dist(xz[-1], (end[0], end[2])) < 1e-8
                length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xz, xz[1:]))
                distance = math.dist((end[0], end[2]), terminal["goal_xz_evaluator_only"])
                assert int(distance < 1.0) == int(row["reached"]) == terminal["reached"]
                assert abs(length - float(row["actual_path_len_m"])) < 1e-7
                geo = float(row["geodesic_m"])
                spl = int(row["reached"]) * geo / max(geo, length)
                assert abs(spl - float(row["spl"])) < 1e-9
                qc = [c["response"] for c in calls if c["query_index"] == index]
                assert qc and all(c["reference_depth_source"] == source for c in qc)
                first = qc[0]["depth_experiment"]
                assert first["writer_stride"] == int(source == "route_sparse")
                row.update(first_certificate_ms=first["certificate_wall_ms"],
                           initial_online_cache_bytes=first["online_cache_bytes"],
                           initial_state_sha256=first["initial_online_state"]["sha256"])
                collected.setdefault(row["analysis_role"], {})[source] = (row, payload)
                all_rows.append(row)
        for role, values in collected.items():
            left, lp = values["canonical"]
            right, rp = values["route_sparse"]
            for field in ("seed", "query_id", "geodesic_m", "shared_A_frames", "initial_state_sha256"):
                assert left[field] == right[field], f"{history['scene']} {role}: {field} mismatch"
            assert lp["rollout_traces"]["legA"] == rp["rollout_traces"]["legA"]
            assert lp["replay"] == rp["replay"]
            both_rejected = not int(left["certificate_accept_plans"]) and not int(right["certificate_accept_plans"])
            if both_rejected:
                assert lp["rollout_traces"]["query"] == rp["rollout_traces"]["query"]
                assert lp["query_result"]["end_position"] == rp["query_result"]["end_position"]
            pairs.append({"scene": history["scene"], "role": role,
                          "canonical": int(left["reached"]), "route_sparse": int(right["reached"]),
                          "both_rejected_exact_trajectory": both_rejected})
    assert len(all_rows) == 16 and len(pairs) == 8
    result = {"verified": True, "histories": 4, "rollouts": 16, "paired_queries": 8,
              "scope": manifest["scope"], "pairs": pairs, "records": all_rows,
              "sr": {s: {r: sum(p[s] for p in pairs if p["role"] == r)
                          for r in ("novel", "revisit")} for s in SOURCES},
              "gain": sum(p["route_sparse"] > p["canonical"] for p in pairs),
              "loss": sum(p["route_sparse"] < p["canonical"] for p in pairs)}
    dump(output / "independent_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "server":
        server()
    elif mode == "eval":
        evaluate()
    elif mode == "preflight":
        preflight()
    elif mode == "local":
        run_local()
    elif mode == "summarize":
        summarize(Path(sys.argv[1]).resolve())
    else:
        raise SystemExit(f"unknown mode {mode}")
