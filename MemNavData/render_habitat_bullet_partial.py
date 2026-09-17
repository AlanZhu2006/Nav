#!/usr/bin/env python3
"""Render saved poses from aborted Bullet diagnostics, without rerunning policy.

The final pose may have crossed the physical-validity bound. Such a frame is
reconstructed from the recorded yaw-only camera contract, not claimed to have
been consumed by the policy. No navigation success is assigned to aborted arms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_physics_scene_probe import make_scene, rgb
from MemNavData.habitat_physics_probe import save, sha
from MemNavData.render_recorded_pose_first_person import Encoder


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / "failure.json").exists():
        raise RuntimeError("This renderer is for a closed, aborted diagnostic only")
    manifest = read(args.run / "manifest.json")
    benchmark = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/manifest.json"
    assert sha(benchmark) == manifest["source_manifest_sha256"]
    history = next(h for h in read(benchmark)["episodes"] if h["scene"] == manifest["scene"])
    source = read(Path(history["online_a_episode"]) / "receipt.json")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for policy, tracker in manifest["variants"]:
        name = f"{policy}__{tracker}"
        folder = args.run / "evaluation" / name
        poses = folder / "physical_actions.jsonl"
        if not poses.exists():
            results.append(dict(variant=name, status="not_executed", video=None))
            continue
        physics = read(folder / "physics_summary.json")
        init = read(folder / "initialization.json")["settled"]
        terminal_path = folder / "terminal_measurements.json"
        terminal = read(terminal_path)[0] if terminal_path.exists() else None
        status = "completed_arm_in_aborted_history" if terminal else "aborted_partial"
        video = args.out / f"{name}__{status}.mp4"
        sim = make_scene(source["source_asset"], folder.with_name(name + "_physics_config.json"))
        encoder = None
        try:
            first = rgb(sim, np.asarray(init["position"]) + [0, -.25, 0], init["yaw_rad"])
            assert hashlib.sha256(first.tobytes()).hexdigest() == physics["first_query_rgb_sha256"]
            encoder = Encoder(video, first.shape[1], first.shape[0], 10, "rgb24")
            encoder.write(first)
            for line in poses.read_text().splitlines():
                state = json.loads(line)["after"]
                encoder.write(rgb(sim, np.asarray(state["position"]) + [0, -.25, 0], state["yaw_rad"]))
            results.append(dict(variant=name, status=status, video=video.name, frames=encoder.frames,
                                first_RGB_exactly_matches=True, max_tilt_deg=physics["max_tilt_deg"],
                                pose_source_sha256=sha(poses),
                                reached=None if terminal is None else terminal["reached"]))
            print(name, status, encoder.frames, "frames", flush=True)
        finally:
            if encoder is not None:
                encoder.close()
            sim.close()
    save(args.out / "render_receipt.json", dict(
        policy_rerun=False, pose_interpolation=False, fps=10, results=results,
        input_failure_sha256=sha(args.run / "failure.json"),
        playback="0.1 simulated seconds per action boundary; inference/network wait excluded",
        limitation="Aborted arm has no SR label; final rendered pose may violate the proxy tilt bound.",
        outputs={p.name: sha(p) for p in args.out.glob("*.mp4")}))


if __name__ == "__main__":
    main()
