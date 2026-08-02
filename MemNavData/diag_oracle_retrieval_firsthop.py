"""Paired raw-vs-oracle retrieval diagnostic at the first leg-B plan.

For each generated 2-leg episode this script replays leg A twice with the same
seed.  The two arms differ only in whether the server may select its normal
anchor or is forced to use ``goal.covis_argmax``.  It compares the chosen
diffusion waypoint direction with the recorded GT leg-B direction.

This is an evaluator diagnostic, not a deployable policy: the oracle anchor and
recorded first-hop direction are used only for attribution.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import requests

# Habitat Y-up -> stored data Z-up rotation.  Keep this small diagnostic
# importable outside the Habitat environment (generate_twoleg imports quaternion).
M_W = np.asarray([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def parquet_pose_hab(row_action):
    transform = np.stack([
        np.asarray(row, dtype=np.float64) for row in row_action
    ]).reshape(4, 4)
    position = M_W.T @ transform[:3, 3]
    rotation = M_W.T @ transform[:3, :3]
    yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
    return position, yaw


def world_delta_to_local(delta_xz, yaw):
    dx, dz = np.asarray(delta_xz, dtype=np.float64)
    sine, cosine = math.sin(yaw), math.cos(yaw)
    return np.asarray([
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    ])


def lookahead_point(path_xy, distance):
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.ndim != 2 or path_xy.shape[0] == 0 or path_xy.shape[1] != 2:
        return None
    norm = np.linalg.norm(path_xy, axis=1)
    indices = np.flatnonzero(norm >= distance)
    return path_xy[indices[0] if len(indices) else -1]


def angle_error_deg(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if np.linalg.norm(prediction) < 1e-8 or np.linalg.norm(target) < 1e-8:
        return None
    pred_angle = math.atan2(prediction[1], prediction[0])
    target_angle = math.atan2(target[1], target[0])
    return math.degrees(abs(wrap_angle(pred_angle - target_angle)))


def recorded_first_hop(rows, switch, lookahead_m):
    start_position, start_yaw = parquet_pose_hab(rows.iloc[switch - 1]["action"])
    target_position = start_position
    travelled = 0.0
    previous = start_position
    for index in range(switch, len(rows)):
        candidate, _ = parquet_pose_hab(rows.iloc[index]["action"])
        travelled += float(np.linalg.norm(candidate[[0, 2]] - previous[[0, 2]]))
        target_position = candidate
        previous = candidate
        if travelled >= lookahead_m:
            break
    local = world_delta_to_local(
        target_position[[0, 2]] - start_position[[0, 2]], start_yaw)
    return local, travelled


def reset(base_url, camera_height, seed):
    response = requests.post(
        f"{base_url}/navigator_reset",
        json={"camera_height": float(camera_height), "seed": int(seed)},
    )
    response.raise_for_status()


def replay_and_plan(base_url, episode_dir, switch, goal_jpg, seed,
                    camera_height, forced_anchor=None, forced_gate=None):
    reset(base_url, camera_height, seed)
    rgb_dir = episode_dir / "videos/chunk-000/observation.images.rgb"
    for index in range(switch):
        response = requests.post(
            f"{base_url}/memory_step",
            files={"image": ("image.jpg", (rgb_dir / f"{index}.jpg").read_bytes())},
        )
        response.raise_for_status()
    data = {}
    if forced_anchor is not None:
        data["forced_anchor"] = str(int(forced_anchor))
    if forced_gate is not None:
        data["forced_gate"] = str(float(forced_gate))
    response = requests.post(
        f"{base_url}/imagegoal_step",
        files={
            "image": ("image.jpg", (rgb_dir / f"{switch - 1}.jpg").read_bytes()),
            "goal": ("goal.jpg", goal_jpg),
        },
        data=data,
    )
    response.raise_for_status()
    return response.json()


def summarize_response(response, gt_hop, lookahead_m):
    if "trajectory" not in response:
        return {"error": response.get("error", "trajectory unavailable")}
    trajectory = np.asarray(response["trajectory"], dtype=np.float64)
    hop = lookahead_point(trajectory[:, :2], lookahead_m)
    return {
        "anchor": response.get("anchor"),
        "retrieved_anchor": response.get("retrieved_anchor"),
        "forced_anchor": response.get("forced_anchor"),
        "raw_score": response.get("raw_score"),
        "forced_anchor_score": response.get("forced_anchor_score"),
        "gate": response.get("gate"),
        "predicted_gate": response.get("predicted_gate"),
        "forced_gate": response.get("forced_gate"),
        "goal_rel_yaw_deg": (None if response.get("goal_rel_yaw") is None
                             else math.degrees(response["goal_rel_yaw"])),
        "hop_xy": hop.tolist() if hop is not None else None,
        "hop_norm_m": float(np.linalg.norm(hop)) if hop is not None else None,
        "direction_error_deg": angle_error_deg(hop, gt_hop),
        "endpoint_xy": trajectory[-1, :2].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--scenes", required=True,
                        help="comma-separated scene directory names")
    parser.add_argument("--episode_ids", default="",
                        help="optional comma-separated episode directory names")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--base_seed", type=int, default=20260802)
    parser.add_argument("--lookahead_m", type=float, default=0.70)
    parser.add_argument("--include_gate_oracle", action="store_true",
                        help="also force GT anchor with decoder gate=1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    base_url = f"http://{args.host}:{args.port}"
    wanted = {value.strip() for value in args.episode_ids.split(",")
              if value.strip()}
    report_rows = []
    for scene in [value.strip() for value in args.scenes.split(",")
                  if value.strip()]:
        for episode_dir in sorted((args.dataset_root / scene).glob("episode_*")):
            if wanted and episode_dir.name not in wanted:
                continue
            meta = json.loads((episode_dir / "meta/gen_meta.json").read_text())
            if meta.get("n_legs", 2) != 2:
                continue
            goal = meta["goals"][0]
            switch = int(meta["switch_idx"])
            parquet = episode_dir / "data/chunk-000/episode_000000.parquet"
            rows = pd.read_parquet(parquet)
            gt_hop, gt_travelled = recorded_first_hop(
                rows, switch, args.lookahead_m)
            episode_number = int(episode_dir.name.rsplit("_", 1)[1])
            seed = args.base_seed + episode_number
            camera_height = float(meta.get("camera_height_m", 0.5))
            goal_jpg = (episode_dir / "goal_1.jpg").read_bytes()
            raw_response = replay_and_plan(
                base_url, episode_dir, switch, goal_jpg, seed, camera_height)
            oracle_response = replay_and_plan(
                base_url, episode_dir, switch, goal_jpg, seed, camera_height,
                forced_anchor=int(goal["covis_argmax"]))
            raw = summarize_response(raw_response, gt_hop, args.lookahead_m)
            oracle = summarize_response(oracle_response, gt_hop, args.lookahead_m)
            gate_oracle = None
            if args.include_gate_oracle:
                gate_response = replay_and_plan(
                    base_url, episode_dir, switch, goal_jpg, seed, camera_height,
                    forced_anchor=int(goal["covis_argmax"]), forced_gate=1.0)
                gate_oracle = summarize_response(
                    gate_response, gt_hop, args.lookahead_m)
            paired_hop_error = (None if raw.get("hop_xy") is None
                                or oracle.get("hop_xy") is None
                                else angle_error_deg(
                                    oracle["hop_xy"], raw["hop_xy"]))
            row = {
                "scene": scene,
                "episode": episode_dir.name,
                "seed": seed,
                "switch_idx": switch,
                "gt_anchor": int(goal["covis_argmax"]),
                "gt_hop_xy": gt_hop.tolist(),
                "gt_hop_recorded_m": gt_travelled,
                "raw": raw,
                "oracle": oracle,
                "oracle_gate1": gate_oracle,
                "oracle_vs_raw_hop_angle_deg": paired_hop_error,
                "oracle_gate1_vs_raw_hop_angle_deg": (
                    None if gate_oracle is None
                    or raw.get("hop_xy") is None
                    or gate_oracle.get("hop_xy") is None
                    else angle_error_deg(gate_oracle["hop_xy"], raw["hop_xy"])),
            }
            report_rows.append(row)
            print(json.dumps({
                "scene": scene,
                "episode": episode_dir.name,
                "anchor": [raw.get("anchor"), oracle.get("anchor")],
                "direction_error_deg": [raw.get("direction_error_deg"),
                                        oracle.get("direction_error_deg")],
                "oracle_vs_raw_hop_angle_deg": paired_hop_error,
                "gate1_direction_error_deg": (
                    gate_oracle.get("direction_error_deg")
                    if gate_oracle is not None else None),
            }))

    valid = [row for row in report_rows
             if row["raw"].get("direction_error_deg") is not None
             and row["oracle"].get("direction_error_deg") is not None]
    summary = {
        "episodes": len(report_rows),
        "valid_pairs": len(valid),
        "raw_direction_error_mean_deg": (float(np.mean([
            row["raw"]["direction_error_deg"] for row in valid]))
            if valid else None),
        "oracle_direction_error_mean_deg": (float(np.mean([
            row["oracle"]["direction_error_deg"] for row in valid]))
            if valid else None),
        "oracle_improved_count": sum(
            row["oracle"]["direction_error_deg"]
            < row["raw"]["direction_error_deg"] for row in valid),
        "oracle_worsened_count": sum(
            row["oracle"]["direction_error_deg"]
            > row["raw"]["direction_error_deg"] for row in valid),
    }
    report = {"summary": summary, "rows": report_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
