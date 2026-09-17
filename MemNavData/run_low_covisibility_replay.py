#!/usr/bin/env python3
"""Fixed-proposal, real-model coverage ablation on consumed RGB histories.

No simulator, controller, scene asset, or navigation SR is involved. Ground
truth is loaded only after inference to score the predicted direction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.certified_relocalization_runtime import (
    COVERAGE_ABLATION_AUTHORITY_POLICY, STRICT_AUTHORITY_POLICY,
    UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY, scale_free_relative_xy,
)
from MemNavData.score_low_covisibility_witnesses import angle, endpoint_bearing

ARCHIVE_SHA = "ae7ac43646498102798bb5e515bc0671d1df811b9a74f2ae50dfffc5028128ff"
POLICIES = [STRICT_AUTHORITY_POLICY, COVERAGE_ABLATION_AUTHORITY_POLICY,
            UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def unpack(archive, target, expected_sha):
    if sha(archive) != expected_sha:
        raise ValueError("Unexpected fixed RGB input archive")
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        for member in source:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(source.extractfile(member).read())
    for entry in json.loads((target / "input_files.json").read_text()):
        path = target / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Input verification failed: {path}")
    return json.loads((target / "runtime_inputs.json").read_text())


def model_environment():
    lingbot = Path("/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map")
    settings = {
        "LINGBOT_REPO": str(lingbot),
        "LINGBOT_WEIGHTS": str(lingbot / "weights/lingbot-map-long.pt"),
        "MEMNAV_WINDOW": "32", "MEMNAV_NUM_SCALE": "8", "MEMNAV_MAX_FRAME_NUM": "2048",
        "MEMNAV_GROUND_SCALE_MAX": "6.0", "MEMNAV_GATE_FUSION": "complementary",
        "MEMNAV_AUX_POSE_CALIBRATION": "empirical", "MEMNAV_COLLISION_SELECT": "1",
        "MEMNAV_REPORT_TO": "none", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    }
    os.environ.update(settings)
    for path in (ROOT / ".diagnostics/dependencies/python",
                 ROOT / ".diagnostics/dependencies/LightGlue",
                 ROOT / "InternNav/src/diffusion-policy"):
        sys.path.insert(0, str(path))
    return settings


class FixedMatcher:
    """Compute actual SuperPoint/LightGlue once; share matches across policies."""
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def match_paths(self, reference, query, **kwargs):
        anchor = int(Path(reference).stem)
        self.calls.append(anchor)
        return copy.deepcopy(self.matches[anchor])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/"
        "checkpoints/gatecurr600.memnav.ckpt"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    manifest = unpack(args.archive, inputs, args.archive_sha256)
    environment = model_environment()

    import cv2
    import numpy as np
    import torch
    from MemNavData.benchmark_cec_dense_cache_equivalence import online_state_digest, timed
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
    from NavDP.baselines.memnav.policy_agent import MemNavAgent

    torch.set_num_threads(4)
    sources = [Path(__file__), ROOT / "NavDP/baselines/memnav/policy_agent.py",
               ROOT / "MemNavData/certified_relocalization_runtime.py",
               ROOT / "MemNavData/certified_relocalization_contract.py"]
    save(out / "runtime.json", {"python": sys.executable, "torch": torch.__version__,
        "opencv": cv2.__version__, "gpu": torch.cuda.get_device_name(),
        "environment": environment, "checkpoint_sha256": sha(args.checkpoint),
        "input_archive_sha256": args.archive_sha256,
        "selection": manifest["selection"], "sources": {str(p): sha(p) for p in sources},
        "scope": "fixed archived DINO shortlist; matched and localized locally; no SR"})
    print("Loading isolated LingBot + CEC model; existing servers are not used.", flush=True)
    agent = MemNavAgent(checkpoint=str(args.checkpoint), internnav_root=str(ROOT / "InternNav"),
        device="cuda:0", buffer_root=str(out / "runtime_buffer"), exclude_recent=32,
        retrieval_mode="raw", flow_gate="auto", certified_eager_depth_cache=False,
        certified_reference_depth_source="canonical")
    matcher = LightGluePointMatcher(ROOT / ".diagnostics/dependencies/LightGlue",
        dependency_root=ROOT / ".diagnostics/dependencies/python", device="cuda:0", max_keypoints=2048)
    completed = []
    started = time.monotonic()
    for case_number, case in enumerate(manifest["histories"]):
        directory = out / f"history_{case['history']:02d}"
        directory.mkdir()
        agent.reset(camera_height=case["camera_height_m"], seed=case["seed"],
                    episode_len=case["episode_len"], camera_intrinsic=case["camera_intrinsic"])
        durations = []
        for i in range(case["frames"]):
            rgb = (inputs / case["rgb"] / f"{i}.jpg").read_bytes()
            _, elapsed = timed(lambda: agent.add_frame(rgb))
            durations.append(elapsed)
            if i % 40 == 0 or i + 1 == case["frames"]:
                print(f"history={case['history']} RGB {i+1}/{case['frames']}", flush=True)
        state_before = online_state_digest(agent)
        current_pose = agent.cam_pose[-1].float().numpy().tolist()
        save(directory / "ingestion.json", {"frame_seconds": durations,
            "flow_threshold": agent.flow_threshold, "current_pose9": current_pose,
            "online_state": state_before, "scale_receipt": agent._first40_scale_receipt})
        for query_number, query in enumerate(case["queries"]):
            query_dir = directory / f"task_{query['task']:03d}"
            query_dir.mkdir()
            goal_path = inputs / query["goal"]
            goal = goal_path.read_bytes()
            key = hashlib.md5(goal).hexdigest()
            agent._begin_goal_session(key)
            agent._goal_start_frame[key] = case["frames"] - 1
            matches, matching_s = {}, 0.0
            for candidate in query["candidates"]:
                anchor = candidate["anchor"]
                data, elapsed = timed(lambda: matcher.match_paths(
                    inputs / case["rgb"] / f"{anchor}.jpg", goal_path,
                    target_height=518, target_width=518, patch_size=agent.lb.patch_size))
                matches[anchor] = data
                matching_s += elapsed
                np.savez_compressed(query_dir / f"matches_{anchor}.npz", **data)
            fixed = FixedMatcher(matches)
            agent.certified_relocalization_matcher = fixed
            offset = (case_number + query_number) % len(POLICIES)
            order = POLICIES[offset:] + POLICIES[:offset]
            arms = {}
            for policy in order:
                agent._certified_relocalization_cache.clear()
                cv2.setRNGSeed(0)
                receipt, elapsed = timed(lambda: agent.certified_relocalize(
                    goal, query["candidates"], authority_policy=policy))
                save(query_dir / f"{policy}.json", receipt)
                if not receipt["ok"] or receipt["pnp"]["status"] == "runtime_exception":
                    raise RuntimeError(f"Local query failed: {query['task']}, {policy}: {receipt}")
                arms[policy] = {"receipt": receipt, "localization_s_excluding_matching": elapsed}
            if len({a["receipt"]["selected_anchor"] for a in arms.values()}) != 1:
                raise ValueError("Authority-only ablation changed selected anchor")
            solved = [a["receipt"]["pnp"] for a in arms.values() if "pose9" in a["receipt"]["pnp"]]
            if any(p != solved[0] for p in solved[1:]):
                raise ValueError("Repeated PnP on the same local evidence changed")
            if fixed.calls != [c["anchor"] for c in query["candidates"]] * len(POLICIES):
                raise ValueError("Policies did not consume identical correspondences")
            result = {"history": case["history"], "task": query["task"],
                "candidates": query["candidates"], "arm_order": order, "arms": arms,
                "current_pose9": current_pose, "goal_sha256": sha(goal_path),
                "matching_s_shared": matching_s, "same_candidate_and_solved_pnp": True}
            save(query_dir / "result.json", result)
            completed.append(result)
            print(json.dumps({"history": case["history"], "task": query["task"],
                "accept": {p: arms[p]["receipt"]["accepted"] for p in POLICIES},
                "reasons": {p: arms[p]["receipt"]["reason"] for p in POLICIES}}), flush=True)
        preserved = state_before == online_state_digest(agent)
        save(directory / "state_check.json", {"query_preserved_online_state": preserved})
        if not preserved:
            raise ValueError("Query changed causal online geometry state")
    # Evaluation labels and Habitat state never enter the agent or its matcher.
    scoring = json.loads((inputs / "evaluation_only.json").read_text())
    for result in completed:
        truth = scoring[str(result["task"])]
        expected = endpoint_bearing(truth["state"], truth["goal_floor_position"])
        errors = {}
        for policy in POLICIES:
            receipt = result["arms"][policy]["receipt"]
            pose = receipt["pnp"].get("pose9")
            bearing = None if pose is None else scale_free_relative_xy(result["current_pose9"], pose)
            errors[policy] = angle(bearing, expected)
        result["evaluation_only"] = {"group": truth["group"], "q_eligible": truth["q_eligible"],
                                      "bearing_errors_deg": errors}
    summary = {"completed": True, "queries": completed, "navigation_SR": None,
               "same_process_pairing": True, "causal_state_preserved": True,
               "wall_s_excluding_model_load": time.monotonic()-started}
    save(out / "summary.json", summary)
    print(json.dumps({"completed": True, "queries": len(completed), "navigation_SR": None}), flush=True)


if __name__ == "__main__":
    main()
