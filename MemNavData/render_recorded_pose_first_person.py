#!/usr/bin/env python3
"""First-person Habitat re-render of recorded query poses, without a policy.

Uses the original camera specification and scene asset. No interpolation,
navigation, collision simulation, success recomputation, or camera alignment
is applied. Comparison playback holds an arm's final recorded state after it
ends. Individual videos contain just that arm's RGB sequence.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image


ARMS = ("mono_native", "mono_cec_endpoint", "mono_cec_route_tangent")
LABELS = ("Native (mono)", "CEC: endpoint bearing", "CEC: route tangent (diagnostic)")
W, H = 480, 270
CW, CH = 1440, 504


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_states(payload):
    """Keep pre-action observations and append the explicitly logged endpoint."""
    rows = [dict(row) for row in payload["rollout_traces"]["query"]]
    if not rows:
        raise ValueError("Empty query trace")
    steps = [int(row["step"]) for row in rows]
    if steps[0] != 0 or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Query steps must start at zero and strictly increase")
    end = payload["query_result"]
    terminal = dict(step=int(end["steps"]), x=float(end["end_position"][0]),
                    y=float(end["end_position"][1]), z=float(end["end_position"][2]),
                    yaw=float(end["end_yaw_rad"]), terminal=True)
    if terminal["step"] < steps[-1]:
        raise ValueError("Terminal step precedes observations")
    if terminal["step"] == steps[-1]:
        if not np.allclose([rows[-1][k] for k in ("x", "y", "z", "yaw")],
                           [terminal[k] for k in ("x", "y", "z", "yaw")],
                           rtol=0, atol=1e-8):
            raise ValueError("Conflicting states at the same step")
        rows[-1]["terminal"] = True
    else:
        rows.append(terminal)
    for row in rows:
        if not np.isfinite([row[k] for k in ("x", "y", "z", "yaw")]).all():
            raise ValueError("Non-finite recorded camera state")
    return rows


def state_index(steps, step):
    return max(0, min(len(steps) - 1, bisect.bisect_right(steps, step) - 1))


def camera_position(row, camera_height):
    return np.array([row["x"], row["y"] + camera_height, row["z"]])


def jpeg_digest(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=95)
    return hashlib.sha256(stream.getvalue()).hexdigest()


class Encoder:
    def __init__(self, path, width, height, fps, pixel_format):
        self.frames = 0
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", pixel_format,
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("Video encoding failed")


def text(canvas, line, x, y, scale=0.56, color=(40, 40, 40), thickness=1):
    cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def comparison_frame(images, rows, payloads, goal_rgb, goal_xyz, meta, step, maximum, fps):
    canvas = np.full((CH, CW, 3), 246, dtype=np.uint8)
    text(canvas, "HM3D long-range Revisit | First-person pose replay", 16, 31, 0.82, thickness=2)
    text(canvas, f"{meta['scene']} / {meta['episode']}  |  action index {step}/{maximum}", 16, 58)
    text(canvas, "Same goal and initial state; recorded outcomes retained. No policy rerun.", 16, 83)
    text(canvas, f"Playback: {fps:g} recorded states/s, not original wall-clock speed.", 16, 108)
    text(canvas, "Goal image", 1096, 27, 0.49)
    canvas[8:116, 1232:1424] = cv2.resize(cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2BGR), (192, 108))
    for index, (arm, label, image, row) in enumerate(zip(ARMS, LABELS, images, rows)):
        x = index * W
        text(canvas, label, x + 12, 151, 0.56, thickness=2)
        canvas[165:435, x:x+W] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        end = payloads[arm]["query_result"]
        distance = float(np.linalg.norm(np.array([row[k] for k in ("x", "y", "z")]) - goal_xyz))
        finished = step >= int(end["steps"])
        status = ("SUCCESS" if end["reached"] else str(end["termination_reason"]).upper()) if finished else "RUNNING"
        text(canvas, f"step {row['step']} | goal distance {distance:.2f} m | {status}", x+12, 458, 0.47)
        text(canvas, "Terminal view held" if finished else "Recorded pose; camera height 0.50 m", x+12, 480, 0.45)
    return canvas


def run(args):
    import habitat_sim
    from generate_twoleg import HFOV_DEG, make_sim, render

    started = time.monotonic()
    root, out = args.inputs, args.out_dir
    receipt = json.loads((root / "source_receipt.json").read_text())
    for name, expected in receipt["sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    if (receipt["camera_width"], receipt["camera_height_pixels"]) != (W, H):
        raise ValueError("Camera resolution differs from the recorded experiment")
    expected_hfov = np.degrees(2 * np.arctan(0.5*W/receipt["camera_fx"]))
    if abs(HFOV_DEG - expected_hfov) > 1e-9:
        raise ValueError("Camera HFOV differs from the recorded experiment")
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((root / "role_pairs.json").read_text())
    query = next(q for p in meta["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")
    goal_xyz = np.array(query["floor_position"])
    goal_rgb = np.array(Image.open(root / "goal.jpg").convert("RGB"))
    payloads = {a: json.loads((root / f"{a}.json").read_text()) for a in ARMS}
    states = {a: recorded_states(payloads[a]) for a in ARMS}
    indices = {a: [row["step"] for row in states[a]] for a in ARMS}
    maximum = max(s[-1] for s in indices.values())
    cam_h = float(receipt["camera_height_m"])
    sim = make_sim(str(root / f"{meta['scene']}.basis.glb"),
                   str(root / f"{meta['scene']}.basis.navmesh"), recompute_navmesh=False)
    encoders, counts = {}, {}
    sample_matches = {a: [] for a in ARMS}
    try:
        goal_render, _ = render(sim, goal_xyz + np.array([0, cam_h, 0]), float(query["yaw_rad"]))
        cv2.imwrite(str(out / "goal_rerender.png"), cv2.cvtColor(goal_render, cv2.COLOR_RGB2BGR))
        goal_mae = float(np.abs(goal_render.astype(float) - goal_rgb.astype(float)).mean())
        timeline = [0, maximum] if args.smoke else range(maximum + 1)
        if not args.smoke:
            encoders["comparison"] = Encoder(out / "first_person_comparison.mp4", CW, CH, args.fps, "bgr24")
            for arm in ARMS:
                encoders[arm] = Encoder(out / f"{arm}_first_person.mp4", W, H, args.fps, "rgb24")
        cached_rows, cached_images = {}, {}
        for step in timeline:
            images, rows = [], []
            for arm in ARMS:
                index = state_index(indices[arm], step)
                row = states[arm][index]
                if cached_rows.get(arm) != index:
                    rgb, _ = render(sim, camera_position(row, cam_h), float(row["yaw"]))
                    cached_rows[arm], cached_images[arm] = index, rgb
                    if "jpg_sha256" in row and (row["step"] % 100 == 0 or index == len(states[arm])-2):
                        sample_matches[arm].append(dict(step=row["step"],
                            original_jpeg_hash_matches=jpeg_digest(rgb) == row["jpg_sha256"]))
                images.append(cached_images[arm])
                rows.append(row)
                if not args.smoke and step <= indices[arm][-1]:
                    encoders[arm].write(cached_images[arm])
            canvas = comparison_frame(images, rows, payloads, goal_rgb, goal_xyz, meta, step, maximum, args.fps)
            if step in (0, maximum):
                cv2.imwrite(str(out / ("start.png" if step == 0 else "final.png")), canvas)
            if not args.smoke:
                encoders["comparison"].write(canvas)
            if step % 100 == 0 or step == maximum:
                print(f"rendered action {step}/{maximum}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    finally:
        sim.close()
        for name, encoder in encoders.items():
            encoder.close()
            counts[name] = encoder.frames
    files = sorted(out.glob("*.mp4"))
    report = {
        "schema": "recorded_pose_first_person_replay_v1", "smoke": args.smoke,
        "scene": meta["scene"], "episode": meta["episode"],
        "source_receipt_sha256": sha256_file(root / "source_receipt.json"),
        "source_hashes": receipt["sha256"], "policy_rerun": False,
        "success_recomputed": False, "interpolated_poses": False,
        "camera_height_m": cam_h, "hfov_degrees": float(HFOV_DEG),
        "habitat_sim_version": habitat_sim.__version__,
        "playback_states_per_second": args.fps,
        "playback_is_wall_clock_time": False,
        "comparison_after_termination": "hold explicitly recorded terminal position and yaw",
        "terminal_camera_orientation": "recorded end_yaw_rad; never turn to goal for presentation",
        "goal_render_vs_original_jpeg_rgb_mae": goal_mae,
        "sampled_original_jpeg_hash_checks": sample_matches,
        "frame_counts": counts,
        "recorded_results": {a: payloads[a]["query_result"] for a in ARMS},
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha256_file(p) for p in files},
    }
    for name, expected in receipt["sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Source changed during rendering: {name}")
    (out / "render_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "goal_rgb_mae": goal_mae,
                      "frame_counts": counts}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be positive")
    run(args)


if __name__ == "__main__":
    main()
