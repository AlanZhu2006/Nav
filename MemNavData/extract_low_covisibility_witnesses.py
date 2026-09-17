#!/usr/bin/env python3
"""Read first-decision evidence from sealed archives; no inference or actions.

Runs with the Python standard library on the archive host. Outputs JSONL to
stdout so the caller can retain the small evidence slice without copying RGB
buffers or altering the original experiment. Outcomes never select rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time


DEFAULT_ROOT = Path(
    "/scratch/yz11502/Research/Nav-axis-uturn-results/"
    "hm3d_covisibility_repaired_20260909/run_bfb8fc3d5a1bb081/"
    "evaluation_f284c23d7a98b0c2")
SUMMARY_SHA = "0d242ef3dabc35017e4f4bb0b4d7d66c80c1a7b3e8dc23d049d2e433f673bfca"
GROUPS = ("c10_30", "c30_50", "natural_novel")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected=None):
    data = Path(path).read_bytes()
    if expected is not None and sha(data) != expected:
        raise ValueError(f"Changed source: {path}")
    return json.loads(data)


def archive_members(folder, targets):
    receipt = checked_json(folder / "archive_receipt.json")
    if not (receipt["completed"] and receipt["all_member_hashes_readback_verified"]):
        raise ValueError("Incomplete source archive")
    found, provenance, index = {}, {}, None
    with tarfile.open(receipt["archive"], "r|gz") as stream:
        for member in stream:
            if member.name == "task/artifact_index.json":
                index = {r["path"]: r for r in json.load(stream.extractfile(member))["files"]}
            relative = member.name.removeprefix("task/")
            if relative not in targets:
                continue
            if index is None or relative not in index:
                raise ValueError(f"Missing archived provenance for {relative}")
            data = stream.extractfile(member).read()
            expected = index[relative]
            if len(data) != expected["bytes"] or sha(data) != expected["sha256"]:
                raise ValueError(f"Archive member mismatch: {relative}")
            found[relative] = json.loads(data)
            provenance[relative] = {"sha256": sha(data), "bytes": len(data)}
            if set(found) == set(targets):
                break
    if set(found) != set(targets):
        raise ValueError(f"Missing members: {set(targets) - set(found)}")
    return found, provenance


def first_decision(plans, poses):
    plan = plans["query_leg"][0]
    frame = int(plan["frame_idx"])
    pose = next(p for p in poses if p["frame_idx"] == frame)
    state = next(p for p in plans["rollout_traces"]["query"]
                 if p["step"] == plan["step"])
    if state["jpg_sha256"] != pose["image_sha256"]:
        raise ValueError("Pose/decision/observed RGB are not frame-bound")
    keys = (
        "step", "frame_idx", "goal_start_frame", "candidate_ceiling",
        "anchor", "retrieved_anchor", "raw_score", "current_goal_cos",
        "memory_bearing_unit", "memory_unbounded_pointgoal",
        "memory_controller_pointgoal", "revisit_adapter_takeover",
        "certified_relocalization_accepted", "certified_relocalization_reason",
        "certified_relocalization_pnp", "certified_relocalization_certificate",
        "certified_relocalization_authority", "router_selected_anchor",
        "router_candidate_trials", "evaluation_gt_goal_distance_m",
    )
    return {"plan": {k: plan.get(k) for k in keys}, "state": state,
            "camera_pose9": pose["camera_pose9"],
            "image_sha256": pose["image_sha256"],
            "pose_count": pose["pose_count"]}


def extract_record(root, record):
    folder = root / f"task_{record['task_index']:03d}"
    manifest = checked_json(folder / "manifest.json")
    task = manifest["task"]
    construction = checked_json(task["construction"], task["construction_sha256"])
    query = next(q for q in construction["queries"] if q["query_id"] == record["query_id"])
    targets = [f"evaluation/{arm}/{name}" for arm in ("cec", "raw_fixed")
               for name in ("query_plans.json", "lingbot_frame_poses.json")]
    members, provenance = archive_members(folder, targets)
    arms = {arm: first_decision(members[f"evaluation/{arm}/query_plans.json"],
                                members[f"evaluation/{arm}/lingbot_frame_poses.json"])
            for arm in ("cec", "raw_fixed")}
    state_a, state_b = arms["cec"]["state"], arms["raw_fixed"]["state"]
    if any(state_a[k] != state_b[k] for k in ("step", "x", "y", "z", "yaw", "jpg_sha256")):
        raise ValueError("First decisions are at different physical states")
    return {
        "task": record["task_index"], "history": record["history_index"],
        "scene": record["scene"], "group": record["query_id"],
        "q_eligible": query["q_eligible"], "goal_floor_position": query["floor_position"],
        "goal_yaw_rad": query["yaw_rad"], "arms": arms,
        "original_reached": {k: v["reached"] for k, v in record["arms"].items()},
        "original_cec_takeover_plans": record["cec_takeover_plans"],
        "provenance": {"members": provenance, "construction": task["construction"],
            "construction_sha256": task["construction_sha256"],
            "online_a_episode": construction["online_a_episode"],
            "online_a_trace_sha256": construction["online_a_trace_sha256"],
            "goal_rgb": str(Path(task["construction"]).parent / query["goal_rgb"]),
            "goal_rgb_sha256": query["goal_rgb_sha256"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--out", type=Path,
                        help="New JSONL evidence file; stdout then contains progress only")
    args = parser.parse_args(argv)
    summary = checked_json(args.root / "paired_summary.json", SUMMARY_SHA)
    records = sorted((r for r in summary["paired_records"] if r["query_id"] in args.groups),
                     key=lambda r: r["task_index"])
    destination = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        destination = args.out.open("x")

    def emit(prefix, payload):
        line = prefix + " " + json.dumps(payload, allow_nan=False)
        if destination is not None:
            destination.write(line + "\n")
            destination.flush()
        if destination is None or prefix != "EVIDENCE_ROW":
            print(line, flush=True)

    emit("EVIDENCE_HEADER", {"schema": "low_covis_first_witness_v1",
          "root": str(args.root), "summary_sha256": SUMMARY_SHA, "groups": args.groups,
          "expected_rows": len(records), "selection": "all queries in selected groups; no outcome selection",
          "new_rollouts": 0, "inference_calls": 0})
    started = time.monotonic()
    for i, record in enumerate(records):
        emit("EVIDENCE_ROW", extract_record(args.root, record))
        print(f"EVIDENCE_PROGRESS {i+1}/{len(records)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    emit("EVIDENCE_FINAL", {"completed": True, "rows": len(records),
          "wall_seconds": time.monotonic()-started})
    if destination is not None:
        destination.close()
        data = args.out.read_bytes()
        print("EVIDENCE_FILE " + json.dumps({"path": str(args.out),
              "sha256": sha(data), "bytes": len(data)}), flush=True)


if __name__ == "__main__":
    main()
