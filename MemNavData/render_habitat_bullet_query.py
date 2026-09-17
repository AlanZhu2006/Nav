#!/usr/bin/env python3
"""First-person replay of measured Bullet query poses, never a policy rerun."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    if not summary["completed"]:
        raise RuntimeError("Only completed runs are replayed")
    manifest = json.loads((args.run / "manifest.json").read_text())
    benchmark = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/manifest.json"
    if sha(benchmark) != manifest["source_manifest_sha256"]:
        raise RuntimeError("Source benchmark changed")
    h = next(h for h in json.loads(benchmark.read_text())["episodes"] if h["scene"] == manifest["scene"])
    source = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for policy, tracker in manifest["variants"]:
        name = f"{policy}__{tracker}"
        folder = args.run / "evaluation" / name
        init = json.loads((folder / "initialization.json").read_text())["settled"]
        terminal = json.loads((folder / "terminal_measurements.json").read_text())[0]
        sim = make_scene(source["source_asset"], folder.with_name(name+"_physics_config.json"))
        encoder = None
        try:
            first = rgb(sim, np.array(init["position"])+[0,-.25,0], init["yaw_rad"])
            matched = hashlib.sha256(first.tobytes()).hexdigest() == terminal["first_query_rgb_sha256"]
            if not matched:
                raise RuntimeError("First rendered frame differs from recorded camera input")
            encoder = Encoder(args.out / (name+".mp4"), first.shape[1], first.shape[0], 10, "rgb24")
            encoder.write(first)
            with (folder / "physical_actions.jsonl").open() as stream:
                for line in stream:
                    state = json.loads(line)["after"]
                    encoder.write(rgb(sim, np.array(state["position"])+[0,-.25,0], state["yaw_rad"]))
            results.append({"variant":name, "frames":encoder.frames, "initial_RGB_hash_matches":True,
                            "pose_source_sha256":sha(folder / "physical_actions.jsonl"),
                            "recorded_reached":terminal["reached"]})
            print(name, encoder.frames, "frames", flush=True)
        finally:
            if encoder is not None:
                encoder.close()
            sim.close()
    save(args.out / "render_receipt.json", {
        "policy_rerun":False, "pose_interpolation":False, "fps":10,
        "playback":"0.1 simulated seconds per action boundary; model/network latency excluded",
        "camera":"measured cylinder position minus 0.25m; recorded yaw, no goal alignment",
        "input_summary_sha256":sha(args.run / "summary.json"),
        "results":results, "outputs":{p.name:sha(p) for p in args.out.glob("*.mp4")},
    })


if __name__ == "__main__":
    main()
