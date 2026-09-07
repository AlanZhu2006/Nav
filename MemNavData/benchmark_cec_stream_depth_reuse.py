#!/usr/bin/env python3
"""Paired local diagnostic of live-stream versus replayed historical depth.

No controller is executed and no production default is changed. Fixed archived
RGB prefixes, anchors, and identical LightGlue correspondences isolate the
historical-depth source. Query images are earlier RGBs: this is a component
diagnostic, not an independent Revisit or Novel navigation evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_path(rgb, index):
    plain = Path(rgb) / f"{index}.jpg"
    return plain if plain.is_file() else Path(rgb) / f"{index:06d}.jpg"


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / ".diagnostics/cec_latency_optimization_20260818"
    specifications = [
        ("e2e_eager_seed17_anchor80_goal79_v2.json", 80),
        ("pLe_anchor120_normal.json", 120),
        ("yq_anchor160_normal.json", 160),
    ]
    cases = []
    for name, anchor in specifications:
        old = json.loads((source / name).read_text())
        rgb = Path(old["rgb_dir"])
        current = anchor + 40
        frames = [frame_path(rgb, i) for i in range(current + 1)]
        trace = rgb.parent / "online_a_trace.json"
        cases.append({
            "id": f"{rgb.parent.parent.name}_{rgb.parent.name}",
            "rgb_dir": str(rgb), "trace": str(trace),
            "trace_sha256": sha(trace),
            "episode_len": old["episode_len"], "seed": old["seed"],
            "anchor": anchor, "current": current,
            "goal_indices": [anchor - 1, anchor - 5, anchor - 20],
            "rgb_sha256": [sha(p) for p in frames],
        })
    manifest = {
        "schema": "cec_stream_depth_reuse_diagnostic_v1",
        "created_unix": time.time(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha(__file__),
        "agent_sha256": sha(ROOT / "NavDP/baselines/memnav/policy_agent.py"),
        "scope": "consumed fixed-RGB component diagnostic; not navigation SR",
        "selection": "same three archived latency fixtures; fixed anchor offsets",
        "camera_height_m": 0.5, "flow_gate": "auto",
        "cache_schedule": "materialize only the preselected historical anchor",
        "query_history_overlap": "query is an exact earlier RGB; no Novel negatives",
        "cases": cases,
    }
    save(output / "manifest.json", manifest)
    print(json.dumps({"prepared": str(output), "cases": len(cases),
                      "query_pairs": sum(len(c["goal_indices"]) for c in cases)}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/"
        "checkpoints/gatecurr600.memnav.ckpt"))
    parser.add_argument("--internnav-root", type=Path, default=ROOT / "InternNav")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        prepare(output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    if (output / "summary.json").exists():
        raise FileExistsError("completed output must not be overwritten")
    assert manifest["script_sha256"] == sha(__file__)
    assert manifest["agent_sha256"] == sha(
        ROOT / "NavDP/baselines/memnav/policy_agent.py")
    for case in manifest["cases"]:
        assert sha(case["trace"]) == case["trace_sha256"]
        for i, expected in enumerate(case["rgb_sha256"]):
            assert sha(frame_path(case["rgb_dir"], i)) == expected

    import cv2
    import numpy as np
    import torch
    from MemNavData.benchmark_cec_dense_cache_equivalence import (
        compare, online_state_digest, snapshot_storage_bytes, timed)
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
    from NavDP.baselines.memnav.policy_agent import MemNavAgent

    class FrozenMatches:
        def __init__(self, expected, payload):
            self.expected = expected
            self.payload = payload
            self.calls = 0

        def match_paths(self, reference, query, **kwargs):
            assert (sha(reference), sha(query)) == self.expected
            self.calls += 1
            return copy.deepcopy(self.payload)

    def angle(a, b):
        if a is None or b is None:
            return None
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return (None if norm < 1e-10 else
                float(np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)))))

    def array_digest(payload):
        h = hashlib.sha256()
        for key in sorted(payload):
            value = np.asarray(payload[key])
            h.update(key.encode())
            h.update(str(value.shape).encode())
            h.update(str(value.dtype).encode())
            h.update(value.tobytes())
        return h.hexdigest()

    print("Loading one isolated MemNavAgent; no HTTP servers or controllers.", flush=True)
    model_started = time.monotonic()
    agent = MemNavAgent(
        checkpoint=str(args.checkpoint), internnav_root=str(args.internnav_root),
        device=args.device, buffer_root=str(output / "runtime_buffer"),
        flow_gate=manifest["flow_gate"], certified_eager_depth_cache=False,
        certified_route_depth_cache_stride=0)
    matcher = LightGluePointMatcher(
        ROOT / ".diagnostics/dependencies/LightGlue",
        dependency_root=ROOT / ".diagnostics/dependencies/python",
        device=args.device, max_keypoints=2048)
    save(output / "runtime.json", {
        "python": sys.executable, "torch": torch.__version__,
        "opencv": cv2.__version__, "device": torch.cuda.get_device_name(),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
        "model_load_s": time.monotonic() - model_started,
        "manifest_sha256": sha(output / "manifest.json"),
    })
    all_cases = []
    for case_index, case in enumerate(manifest["cases"]):
        case_dir = output / case["id"]
        case_dir.mkdir(exist_ok=False)
        rgb = Path(case["rgb_dir"])
        anchor = case["anchor"]
        phases = {}
        # Alternate ingestion order across histories. reset() restores seeds.
        order = (["baseline", "writer"] if case_index % 2 == 0 else
                 ["writer", "baseline"])
        sparse_depth = sparse_conf = None
        for phase in order:
            agent.reset(camera_height=manifest["camera_height_m"],
                        episode_len=case["episode_len"], seed=case["seed"])
            ingest_s = 0.0
            materialization = None
            for index in range(case["current"] + 1):
                data = frame_path(rgb, index).read_bytes()
                _, elapsed = timed(lambda: agent.add_frame(data))
                ingest_s += elapsed
                if phase == "writer" and index == anchor:
                    before = online_state_digest(agent)
                    (sparse_depth, sparse_conf, source), elapsed = timed(
                        agent._materialize_current_route_depth)
                    after = online_state_digest(agent)
                    materialization = {
                        "s": elapsed, "source": source,
                        "state_preserved": before == after,
                        "cache_bytes": sparse_depth.nbytes + sparse_conf.nbytes,
                    }
                    assert materialization["state_preserved"]
                if index % 40 == 0:
                    print(f"{case['id']} {phase} {index}/{case['current']}", flush=True)
            phases[phase] = {
                "ingest_s_excluding_materialization_and_audit": ingest_s,
                "materialization": materialization,
                "online_state": online_state_digest(agent),
                "scale_receipt": copy.deepcopy(agent._first40_scale_receipt),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "second_dense_state_bytes": snapshot_storage_bytes(
                    agent._certified_dense_stream_snapshot),
            }
        stream_same = phases["writer"]["online_state"] == phases["baseline"]["online_state"]
        scale_same = phases["writer"]["scale_receipt"] == phases["baseline"]["scale_receipt"]
        save(case_dir / "ingestion.json", {"order": order, "phases": phases,
             "same_stream_state": stream_same, "same_scale_receipt": scale_same})
        if not stream_same or not scale_same:
            raise RuntimeError("writer changed paired stream or immutable scale receipt")
        # If baseline was last, attach the already-saved CPU payload; no replay.
        agent._certified_route_reference_depth_cache[anchor] = (
            sparse_depth.copy(), sparse_conf.copy())
        initial_state = online_state_digest(agent)
        trace = json.loads(Path(case["trace"]).read_text())
        poses = {int(p["step"]): p for p in trace["poses"]}
        for frame in [case["current"], *case["goal_indices"]]:
            assert poses[frame]["jpg_sha256"] == case["rgb_sha256"][frame]
        current = poses[case["current"]]
        pairs = []
        for goal_number, goal_index in enumerate(case["goal_indices"]):
            goal_path = frame_path(rgb, goal_index)
            goal_bytes = goal_path.read_bytes()
            goal_key = hashlib.md5(goal_bytes).hexdigest()
            agent._goal_start_frame[goal_key] = agent.n
            matched, match_s = timed(lambda: matcher.match_paths(
                frame_path(rgb, anchor), goal_path,
                target_height=518, target_width=518, patch_size=agent.lb.patch_size))
            frozen = FrozenMatches((sha(frame_path(rgb, anchor)), sha(goal_path)), matched)
            agent.certified_relocalization_matcher = frozen
            np.savez_compressed(case_dir / f"matches_{goal_index}.npz", **matched)
            digest = array_digest(matched)
            arms = {}
            arm_order = (["canonical", "route_sparse"]
                         if (case_index + goal_number) % 2 == 0 else
                         ["route_sparse", "canonical"])
            for arm in arm_order:
                agent._certified_relocalization_cache.clear()
                cv2.setRNGSeed(case["seed"])
                receipt, elapsed = timed(lambda: agent.certified_relocalize(
                    goal_bytes, [{"anchor": anchor, "score": 1.0}],
                    reference_depth_source=arm))
                arms[arm] = {"receipt": receipt, "query_s_excluding_matching": elapsed}
                assert receipt["ok"], receipt
            assert frozen.calls == 2 and array_digest(matched) == digest
            p = poses[goal_index]
            dx, dz = p["x"] - current["x"], p["z"] - current["z"]
            yaw = current["yaw"]
            gt = [-dx * np.sin(yaw) - dz * np.cos(yaw),
                  -dx * np.cos(yaw) + dz * np.sin(yaw)]
            distance = float(np.hypot(dx, dz))
            a, b = [arms[x]["receipt"] for x in ["canonical", "route_sparse"]]
            pair = {
                "goal_index": goal_index, "arm_order": arm_order,
                "correspondences_sha256": digest, "matches": len(matched["scores"]),
                "matching_s_shared": match_s, "arms": arms,
                "same_acceptance": a["accepted"] == b["accepted"],
                "same_reason": a["reason"] == b["reason"],
                "same_selected_anchor": a["selected_anchor"] == b["selected_anchor"],
                "bearing_difference_deg": angle(a.get("aux_pose"), b.get("aux_pose")),
                "evaluation_only": {
                    "gt_planar_distance_m": distance,
                    "gt_bearing_forward_left": gt,
                    "bearing_well_conditioned": distance >= 0.5,
                    "canonical_bearing_error_deg": (
                        angle(a.get("aux_pose"), gt) if distance >= 0.5 else None),
                    "route_sparse_bearing_error_deg": (
                        angle(b.get("aux_pose"), gt) if distance >= 0.5 else None),
                },
            }
            pairs.append(pair)
            save(case_dir / f"query_{goal_index}.json", pair)
            print(json.dumps({"case": case["id"], "goal": goal_index,
                              "accept": [a["accepted"], b["accepted"]],
                              "bearing_difference_deg": pair["bearing_difference_deg"]}),
                  flush=True)
        # A rejected precheck may never have requested canonical depth.
        (dense_depth, dense_conf), final_lookup_s = timed(
            lambda: agent._certified_reference_depth(anchor))
        np.savez_compressed(case_dir / "depths_and_poses.npz",
                            canonical_depth=dense_depth, canonical_conf=dense_conf,
                            sparse_depth=sparse_depth, sparse_conf=sparse_conf,
                            poses=np.asarray([p.numpy() for p in agent.cam_pose]))
        state_preserved = initial_state == online_state_digest(agent)
        assert state_preserved
        valid = (dense_depth > 0) & (sparse_depth > 0)
        summary = {
            "id": case["id"], "anchor": anchor, "current": case["current"],
            "stream_and_scale_identical": stream_same and scale_same,
            "query_state_preserved": state_preserved,
            "depth": compare(dense_depth, sparse_depth),
            "confidence": compare(dense_conf, sparse_conf),
            "median_sparse_to_dense_depth_ratio": float(np.median(
                sparse_depth[valid] / dense_depth[valid])),
            "depth_mean_absolute_relative_difference": float(np.mean(
                np.abs(sparse_depth[valid] - dense_depth[valid]) / dense_depth[valid])),
            "final_canonical_lookup_s": final_lookup_s,
            "ingestion": phases, "pairs": pairs,
        }
        save(case_dir / "result.json", summary)
        all_cases.append(summary)
        save(output / "progress.json", {"completed_cases": len(all_cases),
                                        "planned_cases": len(manifest["cases"])})
    pairs = [p for c in all_cases for p in c["pairs"]]
    result = {
        "schema": manifest["schema"], "completed": True,
        "manifest_sha256": sha(output / "manifest.json"),
        "navigation_SR": None, "production_defaults_changed": False,
        "cases": all_cases,
        "counts": {"histories": len(all_cases), "query_pairs": len(pairs),
                   "canonical_accepts": sum(p["arms"]["canonical"]["receipt"]["accepted"] for p in pairs),
                   "sparse_accepts": sum(p["arms"]["route_sparse"]["receipt"]["accepted"] for p in pairs),
                   "acceptance_disagreements": sum(not p["same_acceptance"] for p in pairs)},
        "interpretation_limits": [
            "Exact historical JPEG queries and preselected single anchors; not the full retrieval population.",
            "No Novel false-activation or navigation SR claim.",
            "One materialized anchor per history; not the amortized cost of an all-history writer.",
            "Reported query times exclude shared feature matching; first canonical lookup is cold, later queries reuse depth.",
            "This pilot cannot authorize replacing canonical depth in the paper baseline.",
        ],
    }
    save(output / "summary.json", result)
    print(json.dumps({"completed": True, "counts": result["counts"],
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
