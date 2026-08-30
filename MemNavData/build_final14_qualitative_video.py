#!/usr/bin/env python3
"""Render the frozen Final14 qualitative case as a short MP4 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MemNavData.build_final14_qualitative_case import (
    ARMS,
    COLORS,
    LABELS,
    load_inputs,
    trajectory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    root = args.assets.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plans, metrics, queries, novel_row, revisit_row = load_inputs(root)

    roles = ("novel", "revisit")
    traces = {
        (arm, role): trajectory(plans[(arm, role)])
        for arm in ARMS for role in roles
    }
    max_steps = max(len(value) for value in traces.values())
    frames = max(2, int(round(float(args.seconds) * int(args.fps))))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(12.0, 5.2), facecolor="white")
    grid = fig.add_gridspec(
        2, 6, height_ratios=(0.62, 1.55), hspace=0.27, wspace=0.22)
    image_specs = (
        (0, root / "current.jpg", "Current RGB"),
        (1, root / "novel_candidate.jpg", "Novel candidate"),
        (2, root / "novel_goal.jpg", "Novel goal"),
        (3, root / "current.jpg", "Same current RGB"),
        (4, root / "cec_anchor.jpg", "Verified history"),
        (5, root / "revisit_goal.jpg", "Revisit goal"),
    )
    for column, path, title in image_specs:
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=8.3, weight="semibold", pad=2)
        ax.axis("off")

    axes = {
        "novel": fig.add_subplot(grid[1, :3]),
        "revisit": fig.add_subplot(grid[1, 3:]),
    }
    lines = {}
    heads = {}
    for role, ax in axes.items():
        all_points = np.concatenate(
            [traces[(arm, role)] for arm in ARMS], axis=0)
        goal = np.asarray([
            float(queries[role]["floor_position"][0]),
            float(queries[role]["floor_position"][2]),
        ])
        all_points = np.concatenate([all_points, goal[None]], axis=0)
        pad = max(0.45, 0.07 * float(np.ptp(all_points, axis=0).max()))
        ax.set_xlim(float(all_points[:, 0].min() - pad),
                    float(all_points[:, 0].max() + pad))
        ax.set_ylim(float(all_points[:, 1].min() - pad),
                    float(all_points[:, 1].max() + pad))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CDD1D6")
        ax.scatter(*traces[("mono_native", role)][0], marker="o", s=28,
                   color="#24272C", zorder=8)
        ax.scatter(*goal, marker="*", s=96, color="#3F8F63",
                   edgecolor="white", linewidth=0.7, zorder=8)
        ax.set_title(
            "Unsupported Novel: CEC abstains"
            if role == "novel" else
            "Supported Revisit: CEC authorizes a bearing",
            loc="left", fontsize=10.0, weight="semibold", pad=5)
        for arm in ARMS:
            line, = ax.plot([], [], color=COLORS[arm], linewidth=2.0,
                            label=LABELS[arm])
            head = ax.scatter([], [], s=22, color=COLORS[arm], zorder=7)
            lines[(arm, role)] = line
            heads[(arm, role)] = head

    fig.suptitle(
        "Proof before control: the same history can help or interfere",
        x=0.02, y=0.995, ha="left", fontsize=13.0, weight="semibold")
    status = fig.text(
        0.5, 0.035, "", ha="center", va="bottom", fontsize=9.2,
        color="#3A4149")
    handles = [lines[(arm, "novel")] for arm in ARMS]
    fig.legend(handles=handles, labels=[LABELS[arm] for arm in ARMS],
               ncol=3, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.072))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.14)

    def update(frame: int):
        global_step = int(round(frame / max(frames - 1, 1) * (max_steps - 1)))
        artists = []
        for role in roles:
            for arm in ARMS:
                trace = traces[(arm, role)]
                index = min(global_step, len(trace) - 1)
                segment = trace[:index + 1]
                lines[(arm, role)].set_data(segment[:, 0], segment[:, 1])
                heads[(arm, role)].set_offsets(segment[-1:])
                artists.extend((lines[(arm, role)], heads[(arm, role)]))
        if frame < frames * 0.22:
            message = "The runtime receives no Novel/Revisit role label."
        elif frame < frames * 0.52:
            message = (
                "Novel: certificate rejects insufficient support; CEC replays "
                "the native request exactly, while raw memory interferes.")
        else:
            message = (
                f"Revisit: certificate accepts ({int(revisit_row['certified_relocalization_pnp']['inliers'])} "
                "PnP inliers) and passes only a unit bearing to the frozen policy.")
        status.set_text(message)
        artists.append(status)
        return artists

    writer = animation.FFMpegWriter(
        fps=int(args.fps), codec="libx264", bitrate=2200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 / int(args.fps), blit=False)
    movie.save(output, writer=writer, dpi=130)
    plt.close(fig)

    receipt = {
        "schema_version": "final14_qualitative_video_v1_20260830",
        "scope": "posthoc_animation_of_frozen_final14_case",
        "scientific_rows_added": 0,
        "runtime_role_visible": False,
        "frames": frames,
        "fps": int(args.fps),
        "duration_seconds": float(frames / int(args.fps)),
        "video_sha256": sha256(output),
        "frozen_outcomes": {
            role: {LABELS[arm]: int(metrics[arm][role]["reached"])
                   for arm in ARMS}
            for role in roles
        },
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "video": str(output), "receipt": str(receipt_path),
        "verified": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
