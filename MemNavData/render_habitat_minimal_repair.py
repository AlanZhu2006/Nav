#!/usr/bin/env python3
"""Render all saved bridge outcomes, including failures, without running policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from MemNavData.generate_twoleg import make_sim, render
from MemNavData.render_recorded_pose_first_person import Encoder, recorded_states, camera_position


def run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    verification = json.loads((root / "independent_verification.json").read_text())
    assert summary["completed"] and verification["verified"]
    bench = Path(manifest["benchmark"])
    histories = {h["scene"]:h for h in json.loads((bench / "manifest.json").read_text())["episodes"]}
    output = root / "first_person"
    output.mkdir(exist_ok=False)
    results = []
    for scene in dict.fromkeys(r["scene"] for r in summary["results"]):
        source = json.loads((Path(histories[scene]["online_a_episode"]) / "receipt.json").read_text())
        sim = make_sim(source["source_asset"], "", agent_radius=.30)
        try:
            for row in [r for r in summary["results"] if r["scene"] == scene]:
                name = "__".join(row[k] for k in ("policy", "executor", "color"))
                if row.get("depth_raster"):
                    name += "__" + row["depth_raster"]
                if row.get("front_goal"):
                    name += "__" + row["front_goal"]
                folder = root / "evaluation" / scene / name
                files = list(folder.glob("*_plans.json"))
                assert len(files) == 1
                payload = json.loads(files[0].read_text())
                states = recorded_states(payload)
                first, _ = render(sim, camera_position(states[0], .5), states[0]["yaw"])
                assert hashlib.sha256(first.tobytes()).hexdigest() == row["first_query_rgb_sha256"]
                encoder = Encoder(output / f"{scene}__{name}.mp4", first.shape[1], first.shape[0], 10, "rgb24")
                try:
                    encoder.write(first)
                    for state in states[1:]:
                        rgb, _ = render(sim, camera_position(state, .5), state["yaw"])
                        encoder.write(rgb)
                    results.append({"scene":scene, "arm":name, "frames":encoder.frames,
                                    "reached":row["reached"], "first_RGB_matches":True})
                finally:
                    encoder.close()
                print(f"RENDERED {scene} {name}: {len(states)} frames", flush=True)
        finally:
            sim.close()
    receipt = dict(policy_rerun=False, pose_interpolation=False, fps=10, records=results,
                   playback="simulated command ticks, excluding inference/network waiting",
                   outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.mp4")})
    (output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root.resolve())
