#!/usr/bin/env python3
"""Consumed cross-view/open-set CEC depth-source diagnostic; no navigation."""

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
from MemNavData.benchmark_cec_stream_depth_reuse import frame_path, save, sha


def prepare(output):
    roots = [
        ("natural", ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814"),
        ("heading30", ROOT / ".diagnostics/shared_online_role_pair_heading30_v3_smoke_20260814"),
        ("hard_support", ROOT / ".diagnostics/final14_population_v3_consumed_20260817/finalizer_consumed2_run/benchmarks/hard_support"),
    ]
    history_root = ROOT / ".diagnostics/shared_online_a_v0v1_pilot_native_20260812"
    cases = {}
    for group, source in roots:
        for path in sorted(source.glob("*/*/role_pairs.json")):
            meta = json.loads(path.read_text())
            scene, episode = path.parent.parent.name, path.parent.name
            key = f"{scene}_{episode}"
            history = history_root / scene / episode
            trace = history / "online_a_trace.json"
            assert sha(trace) == meta["online_a_trace_sha256"]
            raw = json.loads(trace.read_text())
            n = int(meta["online_a_steps"])
            if key not in cases:
                files = [frame_path(history / "rgb", i) for i in range(n)]
                digest = [sha(p) for p in files]
                pose_map = {int(p["step"]): p for p in raw["poses"]}
                assert len(files) == raw["steps"]
                assert all(pose_map[i]["jpg_sha256"] == digest[i] for i in range(n))
                cases[key] = {"id": key, "scene": scene, "episode": episode,
                              "rgb_dir": str(history / "rgb"), "trace": str(trace),
                              "trace_sha256": sha(trace), "frame_count": n,
                              "seed": raw["episode_seed"], "rgb_sha256": digest,
                              "queries": []}
            case = cases[key]
            assert case["frame_count"] == n
            for pair in meta["pairs"]:
                for query in pair["queries"]:
                    image = path.parent / query["goal_rgb"]
                    image_sha = sha(image)
                    assert image_sha == query["goal_rgb_sha256"]
                    assert image_sha not in case["rgb_sha256"], "query duplicates history JPEG"
                    alias = {"group": group, "role_pairs": str(path),
                             "role_pairs_sha256": sha(path)}
                    previous = next((q for q in case["queries"] if q["goal_sha256"] == image_sha), None)
                    if previous is not None:
                        assert previous["analysis_role"] == query["analysis_role"]
                        previous["aliases"].append(alias)
                        continue
                    case["queries"].append({
                        "id": f"q{len(case['queries']):02d}_{image_sha[:10]}",
                        "goal_path": str(image), "goal_sha256": image_sha,
                        "analysis_role": query["analysis_role"], "group": group,
                        "max_covis_reporting_only": query["max_online_a_covis"],
                        "covis_curve_reporting_only": query["covis_curve"],
                        "goal_position_reporting_only": query["floor_position"],
                        "goal_yaw_reporting_only": query["yaw_rad"],
                        "geodesic_from_source_end_reporting_only": query["geodesic_from_a_end_m"],
                        "viewpoint_translation_reporting_only": query.get("translation_from_source_m"),
                        "viewpoint_yaw_delta_reporting_only": query.get("yaw_delta_from_source_deg"),
                        "aliases": [alias],
                    })
    assert len(cases) == 4
    manifest = {"schema": "cec_stream_depth_consumed_role_queries_v1",
                "scope": "fixed-RGB component comparison on consumed construction smokes; no SR",
                "created_unix": time.time(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_sha256": {p: sha(ROOT / p) for p in [
                    "MemNavData/benchmark_cec_stream_depth_role_queries.py",
                    "NavDP/baselines/memnav/policy_agent.py",
                    "MemNavData/certified_relocalization_contract.py",
                    "MemNavData/certified_relocalization_runtime.py"]},
                "camera_height_m": 0.5, "flow_gate": "auto",
                "writer_stride": 1, "cases": list(cases.values()),
                "selection": "all unique query images in the three named consumed roots; no outcome selection",
                "current_state": "last recorded RGB, not the unobserved final-action endpoint",
                "runtime_role_visible": False}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", manifest)
    queries = [q for c in cases.values() for q in c["queries"]]
    print(json.dumps({"prepared": str(output), "histories": len(cases),
                      "unique_queries": len(queries),
                      "roles": {role: sum(q["analysis_role"] == role for q in queries)
                                for role in ["novel", "revisit"]}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/"
        "checkpoints/gatecurr600.memnav.ckpt"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        prepare(output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    assert not (output / "summary.json").exists()
    for path, digest in manifest["code_sha256"].items():
        assert sha(ROOT / path) == digest
    for case in manifest["cases"]:
        assert sha(case["trace"]) == case["trace_sha256"]
        for i, digest in enumerate(case["rgb_sha256"]):
            assert sha(frame_path(case["rgb_dir"], i)) == digest
        for q in case["queries"]:
            assert sha(q["goal_path"]) == q["goal_sha256"]

    import cv2
    import numpy as np
    import torch
    from MemNavData.benchmark_cec_dense_cache_equivalence import (
        compare, online_state_digest, snapshot_storage_bytes, timed)
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
    from NavDP.baselines.memnav.policy_agent import MemNavAgent

    class FixedMatcher:
        def __init__(self, reference_hashes, goal_sha, matches):
            self.reference_hashes = reference_hashes
            self.goal_sha = goal_sha
            self.matches = matches
            self.calls = []

        def match_paths(self, reference, query, **kwargs):
            anchor = int(Path(reference).stem)
            assert sha(reference) == self.reference_hashes[anchor]
            assert sha(query) == self.goal_sha
            self.calls.append(anchor)
            return copy.deepcopy(self.matches[anchor])

    def angle(a, b):
        if a is None or b is None:
            return None
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return None if norm < 1e-10 else float(np.degrees(
            np.arccos(np.clip(np.dot(a, b) / norm, -1, 1))))

    print("Loading isolated model for consumed role queries; no controller.", flush=True)
    agent = MemNavAgent(checkpoint=str(args.checkpoint), internnav_root=str(ROOT / "InternNav"),
                        device="cuda:0", buffer_root=str(output / "runtime_buffer"),
                        flow_gate=manifest["flow_gate"], certified_eager_depth_cache=False)
    matcher = LightGluePointMatcher(ROOT / ".diagnostics/dependencies/LightGlue",
                                    dependency_root=ROOT / ".diagnostics/dependencies/python",
                                    device="cuda:0", max_keypoints=2048)
    save(output / "runtime.json", {"python": sys.executable, "torch": torch.__version__,
                                   "opencv": cv2.__version__, "gpu": torch.cuda.get_device_name(),
                                   "checkpoint_sha256": sha(args.checkpoint)})
    case_results = []
    for case_number, case in enumerate(manifest["cases"]):
        directory = output / case["id"]
        directory.mkdir(exist_ok=False)
        n = case["frame_count"]
        phase_order = ["baseline", "writer"] if case_number % 2 == 0 else ["writer", "baseline"]
        phases = {}
        sparse = {}
        for phase in phase_order:
            agent.certified_route_depth_cache_stride = 1 if phase == "writer" else 0
            agent.reset(camera_height=manifest["camera_height_m"], episode_len=n, seed=case["seed"])
            durations = []
            for index in range(n):
                data = frame_path(case["rgb_dir"], index).read_bytes()
                _, elapsed = timed(lambda: agent.add_frame(data))
                durations.append(elapsed)
                if index % 80 == 0 or index == n - 1:
                    print(f"{case['id']} {phase} {index + 1}/{n}", flush=True)
            phases[phase] = {"ingest_s": sum(durations), "frame_s": durations,
                             "online_state": online_state_digest(agent),
                             "scale_receipt": copy.deepcopy(agent._first40_scale_receipt),
                             "cached_frames": len(agent._certified_route_reference_depth_cache),
                             "cpu_depth_cache_bytes": sum(a.nbytes for pair in agent._certified_route_reference_depth_cache.values() for a in pair),
                             "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                             "second_dense_state_bytes": snapshot_storage_bytes(agent._certified_dense_stream_snapshot)}
            if phase == "writer":
                sparse = dict(agent._certified_route_reference_depth_cache)
        same_state = phases["baseline"]["online_state"] == phases["writer"]["online_state"]
        same_scale = phases["baseline"]["scale_receipt"] == phases["writer"]["scale_receipt"]
        save(directory / "ingestion.json", {"phase_order": phase_order, "phases": phases,
                                            "same_state": same_state, "same_scale": same_scale})
        if not (same_state and same_scale):
            raise RuntimeError("all-frame writer changed paired stream/scale")
        agent._certified_route_reference_depth_cache = sparse
        agent.certified_relocalization_matcher = matcher
        before = online_state_digest(agent)
        poses = np.asarray([p.numpy() for p in agent.cam_pose])
        np.save(directory / "online_poses.npy", poses)
        trace = json.loads(Path(case["trace"]).read_text())
        current = next(p for p in trace["poses"] if p["step"] == n - 1)
        query_results = []
        for query_number, query in enumerate(case["queries"]):
            query_dir = directory / query["id"]
            query_dir.mkdir()
            goal = Path(query["goal_path"]).read_bytes()
            gkey = hashlib.md5(goal).hexdigest()
            agent._begin_goal_session(gkey)
            agent._goal_start_frame[gkey] = n - 1
            agent.certified_relocalization_matcher = matcher
            (candidates, cosine), retrieval_s = timed(lambda: agent._certified_shortlist_before_decoder_warmup(
                goal, gkey, n - 1, n - 2))
            assert len(candidates) == 8
            # Verify this helper produces exactly the production plan() shortlist.
            plan, plan_s = timed(lambda: agent.plan(goal, retrieval_only=True))
            assert plan["certified_visual_candidates"] == candidates
            matches = {}
            match_s = 0.0
            match_shas = {}
            for candidate in candidates:
                anchor = candidate["anchor"]
                value, elapsed = timed(lambda: matcher.match_paths(
                    frame_path(case["rgb_dir"], anchor), Path(query["goal_path"]),
                    target_height=518, target_width=518, patch_size=agent.lb.patch_size))
                matches[anchor] = value
                match_s += elapsed
                p = query_dir / f"matches_{anchor}.npz"
                np.savez_compressed(p, **value)
                match_shas[str(anchor)] = sha(p)
            fixed = FixedMatcher({i: case["rgb_sha256"][i] for i in matches}, query["goal_sha256"], matches)
            agent.certified_relocalization_matcher = fixed
            arm_order = ["canonical", "route_sparse"] if (case_number + query_number) % 2 == 0 else ["route_sparse", "canonical"]
            arms = {}
            for arm in arm_order:
                agent._certified_relocalization_cache.clear()
                cv2.setRNGSeed(case["seed"])
                receipt, elapsed = timed(lambda: agent.certified_relocalize(
                    goal, candidates, reference_depth_source=arm))
                arms[arm] = {"receipt": receipt, "query_s_excluding_retrieval_and_matching": elapsed}
            assert fixed.calls == [c["anchor"] for c in candidates] * 2
            a, b = [arms[k]["receipt"] for k in ["canonical", "route_sparse"]]
            anchor = a["selected_anchor"]
            assert anchor == b["selected_anchor"]
            depth_file = None
            depth_comparison = None
            if anchor in agent._certified_reference_depth_cache:
                dense_depth, dense_conf = agent._certified_reference_depth_cache[anchor]
                sparse_depth, sparse_conf = sparse[anchor]
                depth_file = directory / f"anchor_{anchor}.npz"
                if not depth_file.exists():
                    np.savez_compressed(depth_file, canonical_depth=dense_depth, canonical_conf=dense_conf,
                                        sparse_depth=sparse_depth, sparse_conf=sparse_conf)
                depth_comparison = compare(dense_depth, sparse_depth)
            goal_position = query["goal_position_reporting_only"]
            dx, dz = goal_position[0] - current["x"], goal_position[2] - current["z"]
            yaw = current["yaw"]
            gt = [-dx * np.sin(yaw) - dz * np.cos(yaw), -dx * np.cos(yaw) + dz * np.sin(yaw)]
            distance = float(np.hypot(dx, dz))
            pair = {"query_id": query["id"], "analysis_role": query["analysis_role"],
                    "group": query["group"], "max_covis_reporting_only": query["max_covis_reporting_only"],
                    "goal_sha256": query["goal_sha256"], "candidates": candidates,
                    "retrieval_s": retrieval_s, "shortlist_plan_parity_s": plan_s,
                    "shortlist_equals_production_plan": True,
                    "matching_s_shared": match_s, "matches_file_sha256": match_shas,
                    "arm_order": arm_order, "arms": arms,
                    "same_acceptance": a["accepted"] == b["accepted"],
                    "same_reason": a["reason"] == b["reason"],
                    "bearing_difference_deg": angle(a.get("aux_pose"), b.get("aux_pose")),
                    "depth_file": None if depth_file is None else depth_file.name,
                    "depth_sha256": None if depth_file is None else sha(depth_file),
                    "depth_comparison": depth_comparison,
                    "evaluation_only": {"gt_planar_distance_m": distance,
                                        "canonical_bearing_error_deg": angle(a.get("aux_pose"), gt) if distance >= 0.5 else None,
                                        "route_sparse_bearing_error_deg": angle(b.get("aux_pose"), gt) if distance >= 0.5 else None,
                                        "selected_anchor_covis": query["covis_curve_reporting_only"][anchor]}}
            save(query_dir / "result.json", pair)
            for arm in arms:
                r = arms[arm]["receipt"]
                assert r["ok"] and r["pnp"]["status"] != "runtime_exception", r
            query_results.append(pair)
            print(json.dumps({"case": case["id"], "query": query["id"],
                              "reporting_role": query["analysis_role"], "group": query["group"],
                              "accept": [a["accepted"], b["accepted"]],
                              "reasons": [a["reason"], b["reason"]],
                              "bearing_difference_deg": pair["bearing_difference_deg"]}), flush=True)
        preserved = before == online_state_digest(agent)
        assert preserved
        result = {"case": case["id"], "frames": n, "ingestion": phases,
                  "same_stream_and_scale": same_state and same_scale, "query_state_preserved": preserved,
                  "queries": query_results}
        save(directory / "result.json", result)
        case_results.append(result)
        save(output / "progress.json", {"completed_histories": len(case_results), "planned_histories": len(manifest["cases"])})
    pairs = [p for c in case_results for p in c["queries"]]
    counts = {role: {"queries": sum(p["analysis_role"] == role for p in pairs),
                    **{arm: sum(p["analysis_role"] == role and p["arms"][arm]["receipt"]["accepted"] for p in pairs)
                       for arm in ["canonical", "route_sparse"]}}
              for role in ["novel", "revisit"]}
    result = {"schema": manifest["schema"], "completed": True, "navigation_SR": None,
              "manifest_sha256": sha(output / "manifest.json"), "counts": counts,
              "acceptance_disagreements": sum(not p["same_acceptance"] for p in pairs),
              "cases": case_results,
              "scope": "consumed four-scene component diagnostic; not a new held-out SR or safety guarantee"}
    save(output / "summary.json", result)
    print(json.dumps({"completed": True, "counts": counts,
                      "acceptance_disagreements": result["acceptance_disagreements"]}), flush=True)


if __name__ == "__main__":
    main()
