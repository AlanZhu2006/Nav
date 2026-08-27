#!/usr/bin/env python3
"""Causal live-policy probe for MemNav's Novel image-goal conditioning.

For each requested current frame, reset and replay an identical RGB stream,
then compare the full DDPM candidate tensor under:

1. the episode's correct Goal-A image and seed S;
2. a different episode's Goal-A image and the same seed S;
3. the correct Goal-A image and seed S+1.

The same-seed goal swap isolates goal-image influence.  The second-seed arm is
only a scale reference for ordinary diffusion-sample variation.  Gate is forced
to zero, so this probes the current-state + Novel path without revisit tokens.
The policy server is external and must already be running.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests


def goal_a_path(episode: Path) -> Path:
    meta = json.loads((episode / "meta/gen_meta.json").read_text())
    switch = int(meta["switch_idx"])
    return (episode / "videos/chunk-000/observation.images.rgb"
            / f"{switch - 1}.jpg")


def selected_index(response: dict) -> int:
    paths = np.asarray(response["all_trajectory"], dtype=np.float64)
    selected = np.asarray(response["trajectory"], dtype=np.float64)
    errors = np.max(np.abs(paths - selected[None]), axis=(1, 2))
    return int(np.argmin(errors))


def rms(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--swapped_episode", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--steps", default="40,122")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    meta = json.loads((args.episode / "meta/gen_meta.json").read_text())
    rgb_dir = args.episode / "videos/chunk-000/observation.images.rgb"
    correct_goal = goal_a_path(args.episode).read_bytes()
    swapped_goal = goal_a_path(args.swapped_episode).read_bytes()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]

    def reset_and_plan(k: int, goal: bytes, seed: int) -> dict:
        payload = {
            "camera_height": float(meta.get("camera_height_m", 0.5)),
            "seed": int(seed),
            "episode_len": int(meta["n_frames"]),
        }
        response = requests.post(f"{base}/navigator_reset", json=payload)
        response.raise_for_status()
        for frame_idx in range(k):
            frame = (rgb_dir / f"{frame_idx}.jpg").read_bytes()
            response = requests.post(
                f"{base}/memory_step",
                files={"image": ("image.jpg", frame)},
            )
            response.raise_for_status()
        current = (rgb_dir / f"{k}.jpg").read_bytes()
        response = requests.post(
            f"{base}/imagegoal_step",
            files={
                "image": ("image.jpg", current),
                "goal": ("goal.jpg", goal),
            },
            data={"forced_gate": "0"},
        )
        response.raise_for_status()
        result = response.json()
        if "all_trajectory" not in result:
            raise RuntimeError(f"planning failed at k={k}: {result}")
        return result

    rows = []
    for k in steps:
        correct = reset_and_plan(k, correct_goal, args.seed)
        swapped = reset_and_plan(k, swapped_goal, args.seed)
        resampled = reset_and_plan(k, correct_goal, args.seed + 1)
        c = np.asarray(correct["all_trajectory"], dtype=np.float64)
        w = np.asarray(swapped["all_trajectory"], dtype=np.float64)
        n = np.asarray(resampled["all_trajectory"], dtype=np.float64)
        if c.shape != w.shape or c.shape != n.shape:
            raise RuntimeError(f"candidate shape mismatch: {c.shape}, {w.shape}, {n.shape}")
        goal_rms = rms(c, w)
        noise_rms = rms(c, n)
        spread = float(np.sqrt(np.mean((c - c.mean(axis=0, keepdims=True)) ** 2)))
        rows.append({
            "k": k,
            "num_candidates": int(c.shape[0]),
            "candidate_shape": list(c.shape),
            "all_candidate_goal_swap_rms": goal_rms,
            "all_candidate_seed_change_rms": noise_rms,
            "goal_to_seed_rms_ratio": goal_rms / max(noise_rms, 1e-12),
            "correct_candidate_spread_rms": spread,
            "goal_to_spread_rms_ratio": goal_rms / max(spread, 1e-12),
            "mean_path_goal_swap_rms": rms(c.mean(axis=0), w.mean(axis=0)),
            "selected_path_goal_swap_rms": rms(
                correct["trajectory"], swapped["trajectory"]),
            "correct_selected_idx": selected_index(correct),
            "swapped_selected_idx": selected_index(swapped),
            "correct_predicted_gate": correct.get("predicted_gate"),
            "swapped_predicted_gate": swapped.get("predicted_gate"),
        })

    result = {
        "episode": str(args.episode),
        "swapped_episode": str(args.swapped_episode),
        "seed": args.seed,
        "forced_gate": 0.0,
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
