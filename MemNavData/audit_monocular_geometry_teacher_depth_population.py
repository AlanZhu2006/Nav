#!/usr/bin/env python3
"""Audit privileged teacher-depth availability before MDTEC shard extraction.

This is an input-quality audit only.  It reproduces the builder's frozen scene,
episode and frame selection, reads no model output, and never changes the
population.  A repair may use its receipt to define deterministic attrition,
but must not silently reinterpret an all-zero depth image as a valid teacher.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def depth_receipt(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"valid": False, "reason": "missing_file", "path": str(path)}
    raw = np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    nonzero = raw[raw > 0]
    result: dict[str, object] = {
        "path": str(path),
        "shape": list(raw.shape),
        "nonzero_pixels": int(nonzero.size),
        "raw_min": float(raw.min()) if raw.size else None,
        "raw_max": float(raw.max()) if raw.size else None,
        "median_m": None if not nonzero.size else float(np.median(nonzero) / 10000.0),
    }
    if raw.ndim != 2 or raw.size == 0:
        result.update(valid=False, reason="invalid_shape_or_empty_array")
    elif not nonzero.size:
        result.update(valid=False, reason="all_zero_depth")
    elif float(np.median(nonzero) / 10000.0) > 20.0:
        result.update(valid=False, reason="decoded_unit_guard")
    else:
        result.update(valid=True, reason="valid_metric_depth")
    return result


def run(args) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from MemNavData.build_monocular_geometry_shards import (
        _load_scene_selection,
        balanced_episode_subset,
        discover_episode_pairs,
        frame_schedule,
    )

    selected = _load_scene_selection(args.scene_split, args.scene_field)
    rows = discover_episode_pairs(
        args.data_root,
        args.feature_root,
        None if selected is None else set(selected),
    )
    rows = balanced_episode_subset(rows, args.max_episodes_per_scene)
    invalid = []
    valid_states = 0
    selected_states = 0
    selected_episodes = []
    for row in rows:
        episode = Path(row["episode"])
        feature_episode = Path(row["feature_episode"])
        meta = json.loads((episode / "meta/gen_meta.json").read_text())
        with np.load(
            feature_episode / "videos/chunk-000/lingbot_cache.npz",
            allow_pickle=False,
        ) as cache:
            cache_frames = int(cache["num_frames"].item())
        frames = frame_schedule(meta, cache_frames, args.states_per_episode)
        selected_episodes.append(
            {
                "group": row.get("group", ""),
                "scene": row["scene"],
                "episode_name": row["episode_name"],
                "frames": frames,
            }
        )
        depth_root = episode / "videos/chunk-000/observation.images.depth"
        for frame in frames:
            selected_states += 1
            receipt = depth_receipt(depth_root / f"{frame}.png")
            if receipt["valid"]:
                valid_states += 1
            else:
                invalid.append(
                    {
                        "group": row.get("group", ""),
                        "scene": row["scene"],
                        "episode_name": row["episode_name"],
                        "frame": frame,
                        "depth": receipt,
                    }
                )
    scene_counts = Counter(str(row["scene"]) for row in rows)
    invalid_scene_counts = Counter(str(row["scene"]) for row in invalid)
    invalid_episode_counts = Counter(
        f"{row['group']}/{row['scene']}/{row['episode_name']}" for row in invalid
    )
    result = {
        "schema": "monocular_geometry_teacher_depth_population_audit_v1_20260818",
        "status": "complete",
        "input_quality_only_not_model_selection": True,
        "population_unchanged": True,
        "scene_count": len(scene_counts),
        "episode_count": len(rows),
        "selected_state_count": selected_states,
        "valid_state_count": valid_states,
        "invalid_state_count": len(invalid),
        "invalid_scene_count": len(invalid_scene_counts),
        "invalid_episode_count": len(invalid_episode_counts),
        "invalid_reason_counts": dict(
            sorted(Counter(row["depth"]["reason"] for row in invalid).items())
        ),
        "episodes_per_scene": dict(sorted(scene_counts.items())),
        "invalid_states_per_scene": dict(sorted(invalid_scene_counts.items())),
        "invalid_states_per_episode": dict(sorted(invalid_episode_counts.items())),
        "invalid_states": invalid,
        "selected_episodes": selected_episodes,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--scene-field", default="train")
    parser.add_argument("--max-episodes-per-scene", type=int, default=4)
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
