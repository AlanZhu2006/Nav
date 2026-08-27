#!/usr/bin/env python3
"""Build scene-addressed RGB-only geometry-distillation shards.

The expensive frozen LingBot and NavDP teachers run exactly once per sampled
state.  Output shards contain only compact LingBot evidence, teacher latents,
and fixed functional probes; they never contain model weights.  Scene identity
is explicit so the trainer can fail closed on any train/validation overlap.

This is a development-data builder.  It must not be pointed at development,
blind, held-out, or final-confirmation scenes while architecture choices remain
open.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNNAV_ROOT = REPO_ROOT / "InternNav"
SCHEMA = "monocular_geometry_distillation_shards_v1_20260818"
CAUSAL_SCALE_CONTRACT = "causal_first_prefix_rgb_only_v1"
TEACHER_DEPTH_AUDIT_SCHEMA = (
    "monocular_geometry_teacher_depth_population_audit_v1_20260818"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_depth_audit(path: Path | None):
    """Load a frozen input-invalidity receipt without changing selection.

    The audit is allowed to remove only explicitly enumerated all-zero teacher
    states from the otherwise unchanged population.  Missing/corrupt/unit-
    ambiguous inputs remain fatal and require a new audited repair.
    """

    if path is None:
        return set(), None
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != TEACHER_DEPTH_AUDIT_SCHEMA:
        raise RuntimeError("teacher-depth audit schema drift")
    if payload.get("status") != "complete":
        raise RuntimeError("teacher-depth audit is incomplete")
    if payload.get("input_quality_only_not_model_selection") is not True:
        raise RuntimeError("teacher-depth audit scope drift")
    if payload.get("population_unchanged") is not True:
        raise RuntimeError("teacher-depth audit changed the population")
    invalid_rows = payload.get("invalid_states", [])
    if int(payload.get("invalid_state_count", -1)) != len(invalid_rows):
        raise RuntimeError("teacher-depth invalid-state count drift")
    if int(payload.get("selected_state_count", -1)) != (
        int(payload.get("valid_state_count", -1)) + len(invalid_rows)
    ):
        raise RuntimeError("teacher-depth population arithmetic drift")
    keys = set()
    for row in invalid_rows:
        depth = row.get("depth", {})
        if depth.get("valid") is not False or depth.get("reason") != "all_zero_depth":
            raise RuntimeError("teacher-depth audit contains an unauthorized reason")
        key = (
            str(row.get("group", "")),
            str(row["scene"]),
            str(row["episode_name"]),
            int(row["frame"]),
        )
        if key in keys:
            raise RuntimeError("duplicate teacher-depth invalid state")
        keys.add(key)
    return keys, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": payload["schema"],
        "selected_state_count": int(payload["selected_state_count"]),
        "valid_state_count": int(payload["valid_state_count"]),
        "invalid_state_count": len(invalid_rows),
        "invalid_reason_counts": payload.get("invalid_reason_counts", {}),
    }


def _load_scene_selection(path: Path | None, field: str) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    value = payload
    for part in field.split("."):
        value = value[part]
    scenes = sorted({str(item) for item in value})
    if not scenes:
        raise ValueError("scene selection is empty")
    return scenes


def discover_episode_pairs(
    data_root: Path,
    feature_root: Path,
    selected_scenes: set[str] | None,
) -> list[dict[str, Path | str]]:
    """Join raw episodes and sparse caches by ``(group, scene, episode)``.

    PT1 contains both ``mp3d_2leg/<scene>/episode_0000`` and
    ``mp3d_3leg/<scene>/episode_0000``.  Dropping the group silently aliases
    distinct trajectories.  Local extracted roots may omit the group on one
    side; a unique ``(scene, episode)`` fallback is allowed only when it is
    unambiguous on both sides.
    """

    def identity(episode: Path, root: Path):
        relative = episode.relative_to(root)
        parts = relative.parts
        if len(parts) < 2:
            raise RuntimeError(f"episode path is too shallow: {episode}")
        group = "/".join(parts[:-2])
        return group, episode.parent.name, episode.name

    raw = {}
    for meta in data_root.rglob("meta/gen_meta.json"):
        episode = meta.parents[1]
        scene = episode.parent.name
        if selected_scenes is not None and scene not in selected_scenes:
            continue
        key = identity(episode, data_root)
        if key in raw:
            raise RuntimeError(f"duplicate raw episode key {key}")
        raw[key] = episode

    features = {}
    for cache in feature_root.rglob("lingbot_cache.npz"):
        chunk = cache.parent
        episode = chunk.parents[1]
        scene = episode.parent.name
        if selected_scenes is not None and scene not in selected_scenes:
            continue
        key = identity(episode, feature_root)
        if key in features:
            raise RuntimeError(f"duplicate feature episode key {key}")
        features[key] = episode

    pairs = [(key, raw[key], features[key]) for key in sorted(set(raw) & set(features))]
    raw_remaining = {key: value for key, value in raw.items() if key not in features}
    feature_remaining = {key: value for key, value in features.items() if key not in raw}
    raw_simple: dict[tuple[str, str], list[tuple[tuple[str, str, str], Path]]] = {}
    feature_simple: dict[tuple[str, str], list[tuple[tuple[str, str, str], Path]]] = {}
    for key, value in raw_remaining.items():
        raw_simple.setdefault(key[1:], []).append((key, value))
    for key, value in feature_remaining.items():
        feature_simple.setdefault(key[1:], []).append((key, value))
    for simple in sorted(set(raw_simple) & set(feature_simple)):
        if len(raw_simple[simple]) == len(feature_simple[simple]) == 1:
            raw_key, episode = raw_simple[simple][0]
            _, feature_episode = feature_simple[simple][0]
            pairs.append((raw_key, episode, feature_episode))

    rows = []
    for key, episode, feature_episode in sorted(pairs, key=lambda row: row[0]):
        required = [
            episode / "videos/chunk-000/observation.images.rgb",
            episode / "videos/chunk-000/observation.images.depth",
            feature_episode / "videos/chunk-000/lingbot_cache.npz",
            feature_episode / "videos/chunk-000/lingbot_cam_cache.npz",
        ]
        if all(path.exists() for path in required):
            rows.append(
                {
                    "group": key[0],
                    "scene": key[1],
                    "episode_name": key[2],
                    "episode": episode,
                    "feature_episode": feature_episode,
                }
            )
    return rows


def balanced_episode_subset(
    rows: list[dict[str, Path | str]], max_episodes_per_scene: int
) -> list[dict[str, Path | str]]:
    grouped: dict[str, list[dict[str, Path | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene"]), []).append(row)
    selected = []
    for scene in sorted(grouped):
        ordered = sorted(
            grouped[scene],
            key=lambda row: _sha256_text(
                f"mono-geometry-v1:{row.get('group','')}:{scene}:{row['episode_name']}"
            ),
        )
        selected.extend(ordered[:max_episodes_per_scene])
    return selected


def frame_schedule(meta: dict, cache_frames: int, states: int) -> list[int]:
    lower = 40
    upper = min(int(meta.get("n_frames", cache_frames)), cache_frames) - 2
    if upper < lower:
        return []
    count = min(states, upper - lower + 1)
    return sorted(set(np.linspace(lower, upper, count, dtype=np.int64).tolist()))


def validate_causal_scale_receipt(receipt: dict, prefix_frames: int) -> None:
    """Fail closed if a shard retained whole-episode scale evidence."""

    if receipt.get("scale_evidence_contract") != CAUSAL_SCALE_CONTRACT:
        raise RuntimeError("scale receipt is not causal-first-prefix")
    if receipt.get("whole_episode_ground_cache_consumed") is not False:
        raise RuntimeError("scale receipt consumed whole-episode ground cache")
    if int(receipt.get("scale_prefix_frames", -1)) != int(prefix_frames):
        raise RuntimeError("scale prefix length drift")
    if int(receipt.get("scale_prefix_first_frame", -1)) != 0:
        raise RuntimeError("scale prefix does not start at frame zero")
    if int(receipt.get("scale_prefix_last_frame", -1)) != int(prefix_frames) - 1:
        raise RuntimeError("scale prefix boundary drift")


def goal_path_for_frame(episode: Path, meta: dict, frame: int) -> tuple[Path, str]:
    switches = [int(value) for value in meta.get("switches", [])]
    leg = sum(frame >= switch for switch in switches)
    if leg == 0:
        path, role = episode / "goal_image.jpg", "goal_a"
    else:
        path, role = episode / f"goal_{leg}.jpg", f"goal_{leg}"
    if not path.is_file():
        raise FileNotFoundError(f"missing active goal image {path}")
    return path, role


def _pool(tokens: torch.Tensor, side: int = 16) -> torch.Tensor:
    count, dim = tokens.shape
    source_side = math.isqrt(count)
    if source_side * source_side != count:
        raise ValueError(f"non-square token grid {tuple(tokens.shape)}")
    return (
        F.adaptive_avg_pool2d(
            tokens.T.reshape(1, dim, source_side, source_side), (side, side)
        )
        .flatten(2)
        .transpose(1, 2)[0]
    )


def _episode_cache(feature_episode: Path, lingbot, rgb_dir: Path, device: str):
    cache_path = feature_episode / "videos/chunk-000/lingbot_cache.npz"
    cam_path = feature_episode / "videos/chunk-000/lingbot_cam_cache.npz"
    with np.load(cache_path, allow_pickle=False) as raw:
        cache_np = {key: raw[key] for key in raw.files}
    with np.load(cam_path, allow_pickle=False) as raw:
        cam_np = {key: raw[key] for key in raw.files}
    if int(cache_np["kv_cache_sliding_window"].item()) != 32:
        raise ValueError(f"cache window is not 32: {cache_path}")
    scale_k, scale_v = lingbot.get_scale_kv(str(rgb_dir))
    cache = {
        "scale_k": scale_k,
        "scale_v": scale_v,
        "anchor_k": torch.as_tensor(
            cache_np["anchor_k"], device=device, dtype=torch.bfloat16
        ).permute(1, 2, 0, 3, 4).contiguous(),
        "anchor_v": torch.as_tensor(
            cache_np["anchor_v"], device=device, dtype=torch.bfloat16
        ).permute(1, 2, 0, 3, 4).contiguous(),
        "anchor_frame_indices": torch.as_tensor(
            cache_np["anchor_frame_indices"], dtype=torch.long
        ),
    }
    return cache, cache_np, cam_np


def _functional_probe(navdp, teacher, goal_embed, seed: int, candidates: int):
    generator = torch.Generator(device=teacher.device).manual_seed(seed)
    clean = torch.randn(1, navdp.predict_size, 3, generator=generator, device=teacher.device)
    noise = torch.randn(clean.shape, generator=generator, device=teacher.device)
    timestep_value = int(seed % navdp.noise_scheduler.config.num_train_timesteps)
    timestep_batch = torch.tensor([timestep_value], dtype=torch.long, device=teacher.device)
    noisy = navdp.noise_scheduler.add_noise(clean, noise, timestep_batch)
    model_timestep = torch.tensor([timestep_value], dtype=torch.long, device=teacher.device)
    trajectories = torch.randn(
        candidates, navdp.predict_size, 3, generator=generator, device=teacher.device
    )
    with torch.no_grad():
        epsilon = navdp.predict_noise(noisy, model_timestep, goal_embed, teacher)
        critic = navdp.predict_critic(
            trajectories, teacher.expand(candidates, -1, -1)
        )[None]
    return {
        "noisy": noisy[0].half().cpu(),
        "timestep": torch.tensor(timestep_value, dtype=torch.int64),
        "goal_embed": goal_embed[0].half().cpu(),
        "teacher_epsilon": epsilon[0].half().cpu(),
        "candidates": trajectories.half().cpu(),
        "teacher_critic": critic[0].half().cpu(),
    }


def _encode_depth_baselines(navdp_agent, navdp, teacher_rgb, raw_metric_depth):
    zeros = np.zeros((1, teacher_rgb.shape[2], teacher_rgb.shape[3], 1), np.float32)
    zero_input = navdp_agent.process_depth(zeros)
    raw_input = navdp_agent.process_depth(raw_metric_depth[None, :, :, None].copy())
    with torch.no_grad():
        zero = navdp.rgbd_encoder(teacher_rgb, zero_input)
        raw = navdp.rgbd_encoder(teacher_rgb, raw_input)
    return zero[0].half().cpu(), raw[0].half().cpu()


def _extract_episode(
    row, args, lingbot, navdp_agent, navdp, invalid_teacher_states=frozenset()
):
    from MemNavData.preflight_monocular_geometry_adapter import (
        _causal_lingbot_scale_features,
        _load_bgr,
        _teacher_inputs,
    )

    episode = Path(row["episode"])
    feature_episode = Path(row["feature_episode"])
    meta = json.loads((episode / "meta/gen_meta.json").read_text())
    rgb_dir = episode / "videos/chunk-000/observation.images.rgb"
    depth_dir = episode / "videos/chunk-000/observation.images.depth"
    cache, cache_np, cam_np = _episode_cache(
        feature_episode, lingbot, rgb_dir, args.device
    )
    frames = frame_schedule(meta, int(cache_np["num_frames"].item()), args.states_per_episode)
    scale_features, scale_receipt = _causal_lingbot_scale_features(
        lingbot,
        rgb_dir,
        np.asarray(cam_np["cam_pose_enc"]),
        camera_height_m=float(meta.get("camera_height_m", 0.5)),
        prefix_frames=args.scale_prefix_frames,
    )
    validate_causal_scale_receipt(scale_receipt, args.scale_prefix_frames)
    scale_hat = scale_receipt["scale_hat"]

    tensors: dict[str, list[torch.Tensor]] = {}
    records = []
    skipped_records = []
    for frame in frames:
        teacher_key = (
            str(row.get("group", "")),
            str(row["scene"]),
            str(row["episode_name"]),
            int(frame),
        )
        if teacher_key in invalid_teacher_states:
            skipped_records.append(
                {
                    "frame": int(frame),
                    "reason": "frozen_teacher_depth_audit_all_zero",
                    "teacher_key": list(teacher_key),
                }
            )
            continue
        paths = [rgb_dir / f"{index}.jpg" for index in range(frame - 31, frame + 1)]
        images = lingbot.load_images([str(path) for path in paths]).to(args.device)
        teacher_rgb, teacher_depth = _teacher_inputs(
            navdp_agent, rgb_dir, depth_dir, frame
        )
        goal_path, goal_role = goal_path_for_frame(episode, meta, frame)
        goal_processed = navdp_agent.process_image(
            np.stack([_load_bgr(goal_path)])
        )
        with torch.no_grad():
            window, current_agg, patch_start = lingbot.window_forward(
                cache, images, frame, return_multilayer=True
            )
            depth_feature = lingbot.depth_feature(
                current_agg, images[-1:][None], patch_start
            )
            depth_prediction = lingbot.model._predict_depth(
                current_agg, images[-1:][None], patch_start
            )
            teacher = navdp.rgbd_encoder(teacher_rgb, teacher_depth)
            goal_embed = navdp.image_encoder(
                np.concatenate((goal_processed, teacher_rgb[:, -1]), axis=-1)
            ).unsqueeze(1)

        relative_depth = depth_prediction["depth"][0, 0, ..., 0].float().cpu().numpy()
        raw_metric_depth = (
            relative_depth * float(scale_hat)
            if scale_hat is not None
            else np.zeros_like(relative_depth)
        )
        zero_tokens, raw_tokens = _encode_depth_baselines(
            navdp_agent, navdp, teacher_rgb, raw_metric_depth
        )
        sample_seed = int(
            _sha256_text(
                f"{row.get('group','')}:{row['scene']}:{row['episode_name']}:{frame}"
            )[:8], 16
        )
        values = {
            "recent_specials": window[-8:, :6].half().cpu(),
            "pooled_current_patches": _pool(window[-1, 6:].float()).half().cpu(),
            "pooled_depth_features": _pool(depth_feature.float()).half().cpu(),
            "scale_features": scale_features[0].float().cpu(),
            "teacher_tokens": teacher[0].half().cpu(),
            "raw_depth_tokens": raw_tokens,
            "zero_depth_tokens": zero_tokens,
            **_functional_probe(navdp, teacher, goal_embed, sample_seed, args.candidates),
        }
        for name, value in values.items():
            tensors.setdefault(name, []).append(value)
        records.append(
            {
                "frame": frame,
                "goal_role": goal_role,
                "goal_path": str(goal_path),
                "probe_seed": sample_seed,
            }
        )

    if not records:
        raise RuntimeError(
            f"all selected states lack a valid teacher for {row['scene']}/"
            f"{row['episode_name']}"
        )
    stacked = {name: torch.stack(values) for name, values in tensors.items()}
    metadata = {
        "schema": SCHEMA,
        "scene": row["scene"],
        "group": row.get("group", ""),
        "episode_name": row["episode_name"],
        "episode": str(episode),
        "feature_episode": str(feature_episode),
        "samples": records,
        "skipped_samples": skipped_records,
        "scale": scale_receipt,
    }
    return {"metadata": metadata, "tensors": stacked}


def run(args):
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(INTERNNAV_ROOT))
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    from MemNavData.preflight_monocular_geometry_adapter import _load_navdp_agent

    selected = _load_scene_selection(args.scene_split, args.scene_field)
    invalid_teacher_states, teacher_depth_audit = load_teacher_depth_audit(
        args.teacher_depth_audit
    )
    rows = discover_episode_pairs(
        args.data_root,
        args.feature_root,
        None if selected is None else set(selected),
    )
    rows = balanced_episode_subset(rows, args.max_episodes_per_scene)
    present = sorted({str(row["scene"]) for row in rows})
    if selected is not None and set(present) != set(selected):
        missing = sorted(set(selected) - set(present))
        raise RuntimeError(f"selected scenes lack paired episodes: {missing}")
    if len(present) < args.min_scenes:
        raise RuntimeError(f"only {len(present)} scenes discovered; need {args.min_scenes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    navdp_agent = _load_navdp_agent(args.navdp_checkpoint, args.device)
    navdp = navdp_agent.navi_former
    lingbot = LingBotStream(
        lingbot_repo=str(args.lingbot_repo),
        weights=str(args.lingbot_weights),
        num_scale=8,
        window=32,
        max_frame_num=2048,
        device=args.device,
        use_sdpa=True,
    )

    manifest_rows = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        group_slug = str(row.get("group", "root")).replace("/", "_") or "root"
        name = f"{group_slug}__{row['scene']}__{row['episode_name']}.pt"
        output = args.output_dir / name
        try:
            if output.exists() and args.resume:
                payload = torch.load(output, map_location="cpu", weights_only=False)
            else:
                payload = _extract_episode(
                    row,
                    args,
                    lingbot,
                    navdp_agent,
                    navdp,
                    invalid_teacher_states=invalid_teacher_states,
                )
                torch.save(payload, output)
        finally:
            # window_forward leaves injected causal K/V resident, while the scale
            # LRU intentionally retains per-episode blocks.  A corpus builder must
            # release both at episode boundaries or GPU use grows with the number
            # of trajectories.
            lingbot.model.clean_kv_cache()
            lingbot._scale_lru.clear()
            gc.collect()
            torch.cuda.empty_cache()
        count = len(payload["metadata"]["samples"])
        skipped_samples = payload["metadata"].get("skipped_samples", [])
        scale_receipt = payload["metadata"]["scale"]
        validate_causal_scale_receipt(scale_receipt, args.scale_prefix_frames)
        manifest_rows.append(
            {
                "scene": row["scene"],
                "group": row.get("group", ""),
                "episode_name": row["episode_name"],
                "shard": name,
                "samples": count,
                "selected_samples": count + len(skipped_samples),
                "skipped_samples": skipped_samples,
                "scale": scale_receipt,
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['scene']}/{row['episode_name']} "
            f"states={count} elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )

    observed_invalid_states = {
        tuple(sample["teacher_key"])
        for row in manifest_rows
        for sample in row.get("skipped_samples", [])
    }
    if observed_invalid_states != invalid_teacher_states:
        missing = sorted(invalid_teacher_states - observed_invalid_states)
        unexpected = sorted(observed_invalid_states - invalid_teacher_states)
        raise RuntimeError(
            f"teacher-depth audit binding drift missing={missing} unexpected={unexpected}"
        )
    selected_sample_count = sum(row["selected_samples"] for row in manifest_rows)
    sample_count = sum(row["samples"] for row in manifest_rows)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "development_data_only": True,
        "data_root": str(args.data_root),
        "feature_root": str(args.feature_root),
        "scene_split": None if args.scene_split is None else str(args.scene_split),
        "scene_field": args.scene_field,
        "scenes": present,
        "scene_count": len(present),
        "episode_count": len(manifest_rows),
        "selected_sample_count": selected_sample_count,
        "sample_count": sample_count,
        "invalid_teacher_state_count": len(observed_invalid_states),
        "teacher_depth_audit": teacher_depth_audit,
        "states_per_episode": args.states_per_episode,
        "rows": manifest_rows,
        "elapsed_seconds": time.time() - started,
        "teacher": "official_navdp_rgbd_frozen",
        "student_input": "frozen_lingbot_causal_rgb_only",
        "baselines": ["zero_depth", "raw_lingbot_metricized_depth", "latent_adapter"],
        "scale_evidence_contract": CAUSAL_SCALE_CONTRACT,
        "scale_prefix_frames": args.scale_prefix_frames,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: manifest[key] for key in ("scene_count", "episode_count", "sample_count", "elapsed_seconds")}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path)
    parser.add_argument("--scene-field", default="train")
    parser.add_argument("--max-episodes-per-scene", type=int, default=4)
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--scale-prefix-frames", type=int, default=40)
    parser.add_argument("--teacher-depth-audit", type=Path)
    parser.add_argument("--min-scenes", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--lingbot-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
