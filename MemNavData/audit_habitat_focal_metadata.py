#!/usr/bin/env python3
"""Read-only covisibility sensitivity to the measured Habitat focal length.

No model inference, navigation, dataset mutation or runtime threshold selection.
The focal replacement is local to this separate diagnostic process.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import generate_twoleg as geometry
from build_shared_online_double_revisit import (
    goal_world_points, jpeg_bytes, load_online_history, read_depth_png,
)


def run(benchmark, output, render_goal=False):
    manifest = json.loads((benchmark / "manifest.json").read_text())
    old_fy = geometry.FY
    # Projection matrix measurement on the same 480x270 camera; square pixels.
    actual_fy = 355.8146059513092
    results = []
    for h in manifest["episodes"][:4]:
        episode = Path(h["online_a_episode"])
        receipt = json.loads((episode / "receipt.json").read_text())
        history = load_online_history(episode, receipt)
        query_root = benchmark / h["scene"] / h["episode"]
        roles = json.loads((query_root / "role_pairs.json").read_text())
        sim = geometry.make_sim(receipt["source_asset"], "", agent_radius=.30) if render_goal else None
        for pair in roles["pairs"]:
            for q in pair["queries"]:
                depth_path = query_root / q["goal_depth"]
                depth = read_depth_png(depth_path)
                position = np.asarray(q["floor_position"]) + [0., receipt["camera_height_m"], 0.]
                raw_rgb_matches = None
                if sim is not None:
                    rgb, depth = geometry.render(sim, position, q["yaw_rad"])
                    raw_rgb_matches = hashlib.sha256(jpeg_bytes(rgb)).hexdigest() == q["goal_rgb_sha256"]
                    assert raw_rgb_matches, "Goal re-render must reproduce the actual stored RGB"
                curves = []
                for fy in (old_fy, actual_fy):
                    geometry.FY = fy
                    points = goal_world_points(depth, position, q["yaw_rad"])
                    curves.append(geometry.covis_curve(points, history["transforms"], history["depths"]))
                old, new = curves
                stored = np.asarray(q["covis_curve"])
                difference = np.abs(new-old)
                results.append({"scene": h["scene"], "role": q["analysis_role"],
                                "goal_depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                                "old_max": float(old.max()), "actual_fy_max": float(new.max()),
                                "old_argmax": int(old.argmax()), "actual_fy_argmax": int(new.argmax()),
                                "max_curve_absolute_change": float(difference.max()),
                                "mean_curve_absolute_change": float(difference.mean()),
                                "nonidentical_entries": int(np.count_nonzero(difference)),
                                "rendered_goal_RGB_matches": raw_rgb_matches,
                                "goal_depth_pixels_beyond_png_range": int(np.count_nonzero(depth > 6.5535)),
                                "stored_curve_max_disagreement": float(np.abs(old-stored).max()),
                                "old_curve": old.tolist(), "actual_fy_curve": new.tolist()})
        if sim is not None:
            sim.close()
    geometry.FY = old_fy
    report = {"navigation_rerun": False, "data_modified": False,
              "scope": "4 consumed histories / saved quantized depth; not population relabeling",
              "goal_depth_source": "rerendered_float" if render_goal else "saved_uint16_png",
              "old_fy": old_fy, "measured_fy": actual_fy, "records": results,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(geometry.__file__), benchmark/'manifest.json')}}
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k:v for k,v in row.items() if not k.endswith("_curve")} for row in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render-goal", action="store_true")
    args = parser.parse_args()
    run(args.benchmark.resolve(), args.output.resolve(), args.render_goal)
