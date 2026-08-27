"""Reproduce one frozen top-8 session through the online Pi3X runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from MemNavData.pi3x_online_relocalizer import Pi3XOnlineRelocalizer


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return math.inf
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _load_session(rows_csv: Path, session_id: str) -> list[dict[str, str]]:
    with rows_csv.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["session_id"] == session_id
        ]
    rows.sort(key=lambda row: int(row["dino_rank"]))
    if len(rows) != 8 or [int(row["dino_rank"]) for row in rows] != list(
            range(1, 9)):
        raise ValueError("smoke session must contain the frozen DINO top-8")
    if len({row["decision_frame"] for row in rows}) != 1:
        raise ValueError("smoke session has inconsistent decision frames")
    return rows


def _load_shadow(
    path: Path,
    *,
    scene: str,
    episode: str,
    current_frame: int,
) -> dict[int, dict]:
    matches: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("scene") == scene
                    and row.get("episode") == episode
                    and int(row.get("current_frame", -1)) == current_frame):
                matches[int(row["anchor_frame"])] = row
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_csv", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--pi3_root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected_model_sha256", required=True)
    parser.add_argument("--proof_manifest", type=Path, required=True)
    parser.add_argument("--offline_shadow", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--inference_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = _load_session(args.rows_csv, args.session_id)
    first = rows[0]
    current_frame = int(first["decision_frame"])
    scene = first["scene"]
    episode = first["source_episode"]
    goal_path = args.data_root / first["query_relative_path"]
    source_rgb = (
        args.data_root / first["candidate_relative_path"]
    ).parent
    if not goal_path.is_file() or not source_rgb.is_dir():
        raise FileNotFoundError("frozen smoke images are incomplete")
    candidates = [
        {
            "anchor": int(row["candidate_frame"]),
            "score": float(row["dino_cosine"]),
            "dino_rank": int(row["dino_rank"]),
        }
        for row in rows
    ]

    runtime = Pi3XOnlineRelocalizer(
        pi3_root=args.pi3_root,
        snapshot=args.snapshot,
        expected_model_sha256=args.expected_model_sha256,
        proof_manifest=args.proof_manifest,
        device=args.device,
        inference_dtype=args.inference_dtype,
    )
    with tempfile.TemporaryDirectory(prefix="pi3x_online_smoke_") as temporary:
        flat = Path(temporary)
        for source in source_rgb.glob("*.jpg"):
            (flat / source.name).symlink_to(source)
        result = runtime.relocalize(
            rgb_dir=flat,
            current_frame=current_frame,
            candidates=candidates,
            goal_path=goal_path,
        )

    comparison = None
    if args.offline_shadow is not None:
        shadow = _load_shadow(
            args.offline_shadow,
            scene=scene,
            episode=episode,
            current_frame=current_frame,
        )
        overlap_errors = []
        bearing_errors = []
        for candidate in result["ranked_candidates"]:
            offline = shadow.get(int(candidate["anchor"]))
            if offline is None:
                raise ValueError("offline shadow lacks an online candidate")
            overlap_errors.append(abs(
                float(candidate["pi3x_overlap"])
                - float(offline["best_view_f1_20cm"])
            ))
            bearing_errors.append(_angle_degrees(
                np.asarray(
                    candidate["pi3x_scale_free_bearing_forward_left"],
                    dtype=np.float64,
                ),
                np.asarray(
                    offline["predicted_scale_free_bearing"],
                    dtype=np.float64,
                ),
            ))
        comparison = {
            "offline_candidate_count": len(shadow),
            "maximum_overlap_absolute_error": max(overlap_errors),
            "maximum_bearing_angle_error_deg": max(bearing_errors),
        }

    receipt = {
        "schema_version": 1,
        "session_id": args.session_id,
        "scene": scene,
        "episode": episode,
        "current_frame": current_frame,
        "runtime_status": runtime.status(),
        "result": result,
        "offline_parity": comparison,
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else None
        ),
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
