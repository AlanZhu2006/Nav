#!/usr/bin/env python3
"""Render every fresh endpoint/route-tangent discordance without selection.

The renderer consumes only independently verified, sealed rollout traces.  It
does not rerun a policy or alter the formal decision.  Each video uses the
same layout and shows actual top-down motion, actionwise 3-D distance, and the
route-tangent progress/interface receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


ARMS = (
    "mono_native", "mono_cec_endpoint", "mono_cec_route_tangent",
)
COLORS = {
    "mono_native": (120, 120, 120),
    "mono_cec_endpoint": (43, 130, 217),
    "mono_cec_route_tangent": (163, 111, 53),
}
LABELS = {
    "mono_native": "Mono native",
    "mono_cec_endpoint": "CEC endpoint bearing",
    "mono_cec_route_tangent": "CEC route tangent",
}
WIDTH, HEIGHT = 1600, 900
MAP_RECT = (35, 95, 900, 830)
GRAPH_RECT = (970, 345, 1560, 650)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing sidecar: {path}")
    fields = sidecar.read_text().split()
    require(len(fields) >= 2 and fields[0] == digest,
            f"invalid sidecar: {path}")
    return digest


def revisit_goal(item: dict[str, Any]) -> np.ndarray:
    queries = item["pairs"][0]["queries"]
    targets = [row for row in queries if row["analysis_role"] == "revisit"]
    require(len(targets) == 1, "manifest item has no unique Revisit target")
    goal = np.asarray(targets[0]["floor_position"], dtype=np.float64)
    require(goal.shape == (3,) and np.isfinite(goal).all(),
            "Revisit goal is malformed")
    return goal


def load_payload(episode_root: Path, arm: str) -> dict[str, Any]:
    paths = list((episode_root / arm).glob("*_plans.json"))
    require(len(paths) == 1, f"{episode_root.name}/{arm}: plan count changed")
    return json.loads(paths[0].read_text())


def trace_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trace = payload["rollout_traces"]["query"]
    require(bool(trace), "query trace is empty")
    steps = np.asarray([int(row["step"]) for row in trace], dtype=np.int64)
    xyz = np.asarray([
        [row["x"], row["y"], row["z"]] for row in trace
    ], dtype=np.float64)
    end = np.asarray(payload["query_result"]["end_position"], dtype=np.float64)
    end_step = int(payload["query_result"]["steps"])
    if not np.allclose(xyz[-1], end, atol=1e-9):
        xyz = np.concatenate([xyz, end[None, :]], axis=0)
        steps = np.concatenate([steps, np.asarray([end_step])])
    require(np.isfinite(xyz).all(), "query trace is non-finite")
    return steps, xyz


def put_text(
    canvas: np.ndarray, text: str, x: int, y: int, *, scale: float = 0.55,
    color: tuple[int, int, int] = (35, 35, 35), thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def last_index_at_or_before(steps: np.ndarray, current: int) -> int:
    return int(np.clip(np.searchsorted(steps, current, side="right") - 1,
                       0, len(steps) - 1))


def plan_at_or_before(plans: list[dict[str, Any]], current: int) -> dict[str, Any]:
    eligible = [row for row in plans if int(row.get("step", 0)) <= current]
    return eligible[-1] if eligible else plans[0]


def map_transform(all_xyz: np.ndarray):
    x0, y0, x1, y1 = MAP_RECT
    planar = all_xyz[:, [0, 2]]
    low = np.min(planar, axis=0)
    high = np.max(planar, axis=0)
    span = np.maximum(high - low, 1.0)
    low -= 0.08 * span + 0.25
    high += 0.08 * span + 0.25
    span = high - low
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    center = 0.5 * (low + high)
    pixel_center = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])

    def project(xyz: np.ndarray) -> tuple[int, int]:
        point = np.asarray([xyz[0], xyz[2]], dtype=np.float64)
        pixel = pixel_center + np.asarray([1.0, -1.0]) * (point - center) * scale
        return int(round(pixel[0])), int(round(pixel[1]))

    return project


def draw_distance_graph(
    canvas: np.ndarray,
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    goal: np.ndarray,
    current_step: int,
    max_step: int,
) -> None:
    x0, y0, x1, y1 = GRAPH_RECT
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (225, 225, 225), 1)
    maxima = []
    distances = {}
    for arm, (steps, xyz) in traces.items():
        values = np.linalg.norm(xyz - goal[None, :], axis=1)
        distances[arm] = (steps, values)
        maxima.append(float(np.max(values)))
    max_distance = max(1.0, max(maxima))

    def graph_point(step: float, distance: float) -> tuple[int, int]:
        gx = x0 + int(round((x1 - x0) * step / max(1, max_step)))
        gy = y1 - int(round((y1 - y0) * distance / max_distance))
        return gx, gy

    success_y = graph_point(0, 1.0)[1]
    cv2.line(canvas, (x0, success_y), (x1, success_y), (95, 145, 95), 1,
             cv2.LINE_AA)
    put_text(canvas, "1 m success threshold", x0 + 8, success_y - 7,
             scale=0.42, color=(75, 120, 75))
    for arm in ARMS:
        steps, values = distances[arm]
        visible = steps <= current_step
        points = np.asarray([
            graph_point(int(step), float(value))
            for step, value in zip(steps[visible], values[visible])
        ], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, COLORS[arm], 2, cv2.LINE_AA)
    put_text(canvas, "Actual 3-D distance to target", x0, y0 - 12,
             scale=0.56, thickness=1)
    put_text(canvas, "step", x1 - 38, y1 + 22, scale=0.43)


def render_one(
    *, item: dict[str, Any], discordant: dict[str, Any], run_root: Path,
    out_path: Path, fps: int, maximum_video_frames: int,
) -> dict[str, Any]:
    index = int(discordant["history_index"])
    label = f"{index:03d}_{item['scene']}_{item['episode']}"
    episode_root = run_root / "evaluation" / label
    payloads = {arm: load_payload(episode_root, arm) for arm in ARMS}
    traces = {arm: trace_arrays(payloads[arm]) for arm in ARMS}
    goal = revisit_goal(item)
    outcomes = {
        arm: int(payloads[arm]["query_result"]["reached"]) for arm in ARMS
    }
    require(outcomes["mono_cec_endpoint"] != outcomes[
        "mono_cec_route_tangent"], "requested pair is not discordant")
    require(outcomes["mono_cec_endpoint"] == int(discordant["endpoint"])
            and outcomes["mono_cec_route_tangent"]
            == int(discordant["route_tangent"]),
            "failure-audit discordance differs from sealed payloads")

    max_step = max(int(steps[-1]) for steps, _ in traces.values())
    timeline = np.unique(np.linspace(
        0, max_step, min(maximum_video_frames, max_step + 1),
        dtype=np.int64))
    all_xyz = np.concatenate([
        goal[None, :], *[xyz for _, xyz in traces.values()]
    ], axis=0)
    project = map_transform(all_xyz)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
        (WIDTH, HEIGHT))
    require(writer.isOpened(), f"could not open video writer: {out_path}")

    tangent_plans = payloads["mono_cec_route_tangent"]["query_leg"]
    for current_step in timeline:
        canvas = np.full((HEIGHT, WIDTH, 3), 250, dtype=np.uint8)
        put_text(canvas, "Fresh long-range paired diagnostic", 35, 38,
                 scale=0.85, thickness=2)
        put_text(canvas,
                 f"history {index:02d} | {item['scene']} | {item['episode']} | "
                 f"step {int(current_step)}/{max_step}",
                 35, 70, scale=0.54, color=(80, 80, 80))
        cv2.rectangle(canvas, MAP_RECT[:2], MAP_RECT[2:], (215, 215, 215), 1)

        for arm in ARMS:
            steps, xyz = traces[arm]
            cursor = last_index_at_or_before(steps, int(current_step))
            visible = xyz[:cursor + 1]
            points = np.asarray([project(point) for point in visible],
                                dtype=np.int32)
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, COLORS[arm], 3,
                              cv2.LINE_AA)
            cv2.circle(canvas, project(visible[-1]), 6, COLORS[arm], -1,
                       cv2.LINE_AA)
        cv2.drawMarker(canvas, project(goal), (65, 65, 180),
                       cv2.MARKER_DIAMOND, 18, 3, cv2.LINE_AA)
        start = next(iter(traces.values()))[1][0]
        cv2.drawMarker(canvas, project(start), (35, 35, 35),
                       cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

        y = 110
        for arm in ARMS:
            steps, xyz = traces[arm]
            cursor = last_index_at_or_before(steps, int(current_step))
            distance = float(np.linalg.norm(xyz[cursor] - goal))
            final = float(payloads[arm]["query_result"][
                "final_goal_3d_dist_m"])
            cv2.line(canvas, (980, y - 7), (1012, y - 7), COLORS[arm], 4,
                     cv2.LINE_AA)
            put_text(canvas, LABELS[arm], 1025, y, scale=0.55,
                     color=(35, 35, 35), thickness=1)
            put_text(canvas,
                     f"current {distance:5.2f} m | final {final:5.2f} m | "
                     f"{'SUCCESS' if outcomes[arm] else 'FAIL'}",
                     1025, y + 25, scale=0.46, color=COLORS[arm])
            y += 70

        draw_distance_graph(
            canvas, traces, goal, int(current_step), max_step)
        plan = plan_at_or_before(tangent_plans, int(current_step))
        progress = plan.get("local_tangent_progress_fraction")
        bearing = plan.get("local_tangent_unit_bearing")
        heading = None
        if isinstance(bearing, list) and len(bearing) == 2:
            heading = math.degrees(math.atan2(float(bearing[1]),
                                              float(bearing[0])))
        critic = plan.get("navdp_critic_max")
        status = plan.get("local_tangent_status", "not active")
        put_text(canvas, "Route-tangent receipts", 970, 700,
                 scale=0.60, thickness=2)
        put_text(canvas,
                 f"status: {status}", 970, 735, scale=0.50)
        put_text(canvas,
                 "progress: " + ("n/a" if progress is None
                                  else f"{100.0 * float(progress):.1f}%"),
                 970, 765, scale=0.50)
        put_text(canvas,
                 "requested heading: " + ("n/a" if heading is None
                                           else f"{heading:+.1f} deg"),
                 970, 795, scale=0.50)
        put_text(canvas,
                 "critic max: " + ("n/a" if critic is None
                                    else f"{float(critic):+.3f}"),
                 970, 825, scale=0.50)
        put_text(canvas,
                 "Sealed recorded poses only; no policy rerun.",
                 970, 868, scale=0.46, color=(90, 90, 90))
        writer.write(canvas)

    for _ in range(max(1, int(round(1.5 * fps)))):
        writer.write(canvas)
    writer.release()
    return {
        "history_index": index,
        "scene": item["scene"],
        "episode": item["episode"],
        "outcomes": outcomes,
        "video": out_path.name,
        "video_sha256": sha256_file(out_path),
        "frames": int(len(timeline) + max(1, int(round(1.5 * fps)))),
        "fps": int(fps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--maximum-video-frames", type=int, default=360)
    args = parser.parse_args()
    require(args.fps > 0 and args.maximum_video_frames > 0,
            "fps and maximum-video-frames must be positive")
    require(not args.out_dir.exists(), f"refusing to overwrite {args.out_dir}")
    audit_sha = verify_sidecar(args.failure_audit)
    audit = json.loads(args.failure_audit.read_text())
    require(audit.get("formal_result_independently_verified") is True,
            "failure audit is not bound to a verified formal result")
    require(audit.get("render_every_primary_discordant_pair") is True,
            "failure audit did not authorize complete discordant rendering")
    manifest_path = args.population / "role_pairs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(sha256_file(manifest_path) == audit["benchmark_manifest_sha256"],
            "visualization population differs from failure audit")
    discordants = audit["primary_discordant_pairs"]
    args.out_dir.mkdir(parents=True)
    videos = []
    for row in discordants:
        index = int(row["history_index"])
        item = manifest["episodes"][index]
        safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["scene"]))
        out = args.out_dir / f"{index:03d}_{safe_scene}_endpoint_vs_tangent.mp4"
        videos.append(render_one(
            item=item, discordant=row, run_root=args.run_root,
            out_path=out, fps=args.fps,
            maximum_video_frames=args.maximum_video_frames,
        ))
    receipt = {
        "schema_version": (
            "hm3d_longrange_route_tangent_discordant_videos_v1_20260903"),
        "claim_scope": "illustrative sealed-trace visualization only",
        "failure_audit_sha256": audit_sha,
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "rendered_all_primary_discordants": True,
        "discordant_count": len(discordants),
        "videos": videos,
    }
    receipt_path = args.out_dir / "manifest.json"
    receipt_path.write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt_path.with_name(receipt_path.name + ".sha256").write_text(
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n")
    print(json.dumps({
        "status": "complete",
        "discordant_count": len(discordants),
        "out_dir": str(args.out_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
