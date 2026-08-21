#!/usr/bin/env python3
"""M2P S-1: candidate-free, read-only LingBot/GCT goal-query diagnostic.

This is a zero-training feasibility gate.  It streams the generated expert
Goal-A prefix of a two-leg episode through the frozen LingBot GCT, then asks the
cached map to localize ``goal_1.jpg`` without supplying a retrieved history
anchor.  The resulting current-to-goal direction is compared with the same
NavDP-frame ground truth used by the certified relocalization audit.  True
online NavDP histories are handled by ``diag_m2p_s1_online_query.py``; keeping
the two scopes explicit prevents an expert-history probe from being reported as
a deployment-history result.

For attribution, the script also evaluates an oracle-locality control: reset the
stream, stop at the metadata ``covis_argmax`` anchor, query the same goal, and
combine that goal pose with the current pose from the full stream.  Thus:

* oracle-locality good, candidate-free bad -> global goal-query/readout failure;
* both bad -> frozen GCT pose/map representation is insufficient for this task;
* both good -> proceed to the small learned support/readout probe (not long
  closed-loop evaluation yet).

The query is genuinely non-writing.  LingBot's ``_set_skip_append(True)`` keeps
KV tensors unchanged, but its camera head still increments ``frame_idx``.  This
diagnostic therefore rolls that counter back explicitly and verifies every
persistent KV tensor (identity, version, shape, and sampled contents) before and
after every query.  No source dataset, checkpoint, or cache is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


DEFAULT_EPISODE_ROOT = Path(
    "/home/asus/Research/datasets/mp3d_20scene/episodes")
DEFAULT_LINGBOT_REPO = Path(
    "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map")
DEFAULT_OUT = Path(
    ".diagnostics/m2p_s1_gct_query_smoke_20260813")

_HABITAT_TO_DATA_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(
        value.tolist() if hasattr(value, "tolist") else value,
        dtype=np.float64)
    if array.size != 16:
        raise ValueError(f"{name} must contain 16 values, got {array.shape}")
    return array.reshape(4, 4)


def _resolve_generated_mount(extrinsic: np.ndarray,
                             frame_convention: str) -> np.ndarray:
    """Mirror the audited generated-data compatibility rule."""
    result = np.asarray(extrinsic, dtype=np.float64).copy()
    if not str(frame_convention or "").startswith(
            "positions+parquet in data(Zup,M_W)"):
        return result
    rotation = result[:3, :3]
    if np.allclose(rotation, _HABITAT_TO_DATA_ROTATION, atol=1e-6):
        return result
    if not np.allclose(rotation, np.eye(3), atol=1e-6):
        raise ValueError("generated Z-up episode has an unsupported camera mount")
    result[:3, :3] = _HABITAT_TO_DATA_ROTATION
    return result


def _yaw_habitat_to_data_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    habitat = np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)
    return _HABITAT_TO_DATA_ROTATION @ habitat


def _goal_camera_to_world(goal: dict[str, Any]) -> np.ndarray:
    position = np.asarray(goal.get("pos"), dtype=np.float64)
    require(position.shape == (3,) and bool(np.isfinite(position).all()),
            "invalid goal position")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _yaw_habitat_to_data_rotation(
        float(goal.get("yaw_habitat", 0.0)))
    result[:3, 3] = position
    return result


def _navdp_ground_truth_relative(
        current_camera_to_world: np.ndarray,
        goal_camera_to_world: np.ndarray,
        base_extrinsic: np.ndarray) -> np.ndarray:
    """Return audited NavDP [forward, lateral] current-to-goal translation."""
    current = np.asarray(current_camera_to_world, dtype=np.float64)
    goal = np.asarray(goal_camera_to_world, dtype=np.float64)
    mount = np.asarray(base_extrinsic, dtype=np.float64)
    base_rotation = current[:3, :3] @ np.linalg.inv(mount[:3, :3])
    local = base_rotation.T @ (goal[:3, 3] - current[:3, 3])
    return np.array([local[1], -local[0]], dtype=np.float64)


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    require(quaternion.shape == (4,), "quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    require(np.isfinite(norm) and norm > 1e-12,
            "quaternion is non-finite or degenerate")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _lingbot_relative_direction(current_pose9: np.ndarray,
                                goal_pose9: np.ndarray) -> np.ndarray:
    """Decode scale-free LingBot direction in NavDP [forward, lateral]."""
    current = np.asarray(current_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    require(current.shape == (9,) and goal.shape == (9,),
            "LingBot poses must have shape (9,)")
    rotation = _quaternion_xyzw_to_matrix(current[3:7])
    translation = rotation.T @ (goal[:3] - current[:3])
    return np.array([translation[2], -translation[0]], dtype=np.float64)


def direction_error_degrees(predicted: np.ndarray,
                            target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pn, tn = float(np.linalg.norm(predicted)), float(np.linalg.norm(target))
    if pn <= 1e-9 or tn <= 1e-9:
        return float("nan")
    cosine = float(np.clip(predicted @ target / (pn * tn), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signed_bearing_degrees(pointgoal: np.ndarray) -> float:
    pointgoal = np.asarray(pointgoal, dtype=np.float64)
    return float(np.degrees(np.arctan2(pointgoal[1], pointgoal[0])))


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Cheap mutation-sensitive signature without cloning a multi-GB cache."""
    flat = tensor.detach().reshape(-1)
    if flat.numel():
        indices = sorted({0, flat.numel() // 3,
                          2 * flat.numel() // 3, flat.numel() - 1})
        sample = flat[torch.as_tensor(
            indices, device=flat.device, dtype=torch.long)]
        sample_bytes = sample.float().cpu().numpy().tobytes()
        sample_sha256 = hashlib.sha256(sample_bytes).hexdigest()
    else:
        sample_sha256 = hashlib.sha256(b"").hexdigest()
    # Tensors allocated inside ``torch.inference_mode`` intentionally do not
    # expose a version counter.  Pointer + sampled contents still catch both
    # replacement and mutation for those tensors; normal tensors additionally
    # carry their PyTorch mutation version.
    try:
        version: int | None = int(
            tensor._version)  # pylint: disable=protected-access
    except RuntimeError:
        version = None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
        "version": version,
        "sample_sha256": sample_sha256,
    }


def _value_signature(value: object) -> object:
    if torch.is_tensor(value):
        return _tensor_signature(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def persistent_state_signature(model: torch.nn.Module) -> dict[str, Any]:
    """Describe all persistent streaming KVs and temporal counters."""
    agg = model.aggregator
    camera = model.camera_head
    agg_cache = {
        str(key): _value_signature(value)
        for key, value in sorted(agg.kv_cache.items(), key=lambda item: str(item[0]))
    }
    camera_cache = [
        {
            str(key): _value_signature(value)
            for key, value in sorted(cache.items(), key=lambda item: str(item[0]))
        }
        for cache in camera.kv_cache
    ]
    return {
        "aggregator_total_frames_processed": int(
            agg.total_frames_processed),
        "camera_frame_idx": int(camera.frame_idx),
        "aggregator_kv": agg_cache,
        "camera_kv": camera_cache,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    sys.path.insert(0, os.fspath(args.lingbot_repo))
    from lingbot_map.layers.rope import (  # pylint: disable=import-error
        WanRotaryPosEmbed, get_1d_rotary_pos_embed)
    from lingbot_map.models.gct_stream import (  # pylint: disable=import-error
        GCTStream)

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.window,
        kv_cache_scale_frames=args.num_scale,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=args.camera_iterations,
    )
    checkpoint = torch.load(
        args.weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict
    require(not missing and not unexpected,
            f"LingBot checkpoint mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}")

    extended = 0
    for module in model.modules():
        if (isinstance(module, WanRotaryPosEmbed)
                and module.max_seq_len < args.max_frame_num):
            old = module.freqs
            dimensions = module.fhw_dim
            new = torch.cat([
                get_1d_rotary_pos_embed(
                    dimension, args.max_frame_num, 10000.0,
                    use_real=False, repeat_interleave_real=False,
                    freqs_dtype=torch.float64)
                for dimension in dimensions
            ], dim=1)
            require(torch.allclose(
                new[:old.shape[0]].to(old.dtype), old,
                atol=1e-6), "RoPE extension changed the overlap region")
            module.freqs = new.to(old.device)
            module.max_seq_len = args.max_frame_num
            extended += 1
    print(f"[model] exact checkpoint load; extended {extended} RoPE table(s)",
          flush=True)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _stream_prefix(model: torch.nn.Module, images: torch.Tensor,
                   *, num_scale: int, device: str,
                   progress_label: str,
                   return_all: bool = False
                   ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Reset and causally stream one prefix; return its last (or all) pose9."""
    require(len(images) >= num_scale,
            f"prefix needs at least {num_scale} frames")
    model.clean_kv_cache()
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16):
        scale = images[:num_scale][None].to(device)
        aggregated, _ = model._aggregate_features(
            scale, num_frame_for_scale=num_scale,
            num_frame_per_block=num_scale)
        poses = model.camera_head(
            aggregated, causal_inference=True,
            num_frame_per_block=num_scale,
            num_frame_for_scale=num_scale)
        scale_poses = poses[-1][0].float().cpu().numpy()
        last = scale_poses[-1]
        all_poses = [scale_poses] if return_all else None
        del scale, aggregated, poses
        for frame_index in range(num_scale, len(images)):
            frame = images[frame_index:frame_index + 1][None].to(device)
            aggregated, _ = model._aggregate_features(
                frame, num_frame_for_scale=num_scale,
                num_frame_per_block=1)
            poses = model.camera_head(
                aggregated, causal_inference=True,
                num_frame_per_block=1,
                num_frame_for_scale=num_scale)
            last = poses[-1][0, -1].float().cpu().numpy()
            if all_poses is not None:
                all_poses.append(last[None])
            del frame, aggregated, poses
            if ((frame_index + 1) % 20 == 0
                    or frame_index + 1 == len(images)):
                elapsed = time.monotonic() - started
                print(f"[{progress_label}] {frame_index + 1}/{len(images)} "
                      f"frames, {elapsed:.1f}s", flush=True)
    if all_poses is not None:
        return last, np.concatenate(all_poses, axis=0)
    return last


def _read_only_query(model: torch.nn.Module, goal_image: torch.Tensor,
                     *, num_scale: int, device: str,
                     label: str) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Query a goal at the next virtual time and restore all persistent state."""
    before = persistent_state_signature(model)
    saved_agg_time = int(model.aggregator.total_frames_processed)
    saved_camera_time = int(model.camera_head.frame_idx)
    agg_cache = model.aggregator.kv_cache
    agg_had_skip = "_skip_append" in agg_cache
    agg_skip_value = agg_cache.get("_skip_append")
    camera_skip_state = [
        ("_skip_append" in cache, cache.get("_skip_append"))
        for cache in model.camera_head.kv_cache
    ]
    manager = getattr(model.aggregator, "kv_cache_manager", None)
    manager_had_skip = (
        manager is not None and hasattr(manager, "_skip_append"))
    manager_skip_value = (
        getattr(manager, "_skip_append") if manager_had_skip else None)
    started = time.monotonic()
    model._set_skip_append(True)
    try:
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            frame = goal_image[None, None].to(device)
            aggregated, _ = model._aggregate_features(
                frame, num_frame_for_scale=num_scale,
                num_frame_per_block=1)
            poses = model.camera_head(
                aggregated, causal_inference=True,
                num_frame_per_block=1,
                num_frame_for_scale=num_scale)
            result = poses[-1][0, -1].float().cpu().numpy()
            del frame, aggregated, poses
    finally:
        model._set_skip_append(False)
        # `_set_skip_append` protects K/Vs and aggregator time, but the upstream
        # camera head always advances frame_idx.  Restore it to make this API
        # actually read-only and repeatable.
        model.aggregator.total_frames_processed = saved_agg_time
        model.camera_head.frame_idx = saved_camera_time
        # Restore flag *presence* as well as value.  Upstream's first call to
        # `_set_skip_append` may create an aggregator control key that was not
        # present in the clean-stream state; leaving a new false-valued key is
        # behaviourally harmless but fails an exact read-only identity audit.
        if agg_had_skip:
            agg_cache["_skip_append"] = agg_skip_value
        else:
            agg_cache.pop("_skip_append", None)
        for cache, (had_skip, skip_value) in zip(
                model.camera_head.kv_cache, camera_skip_state):
            if had_skip:
                cache["_skip_append"] = skip_value
            else:
                cache.pop("_skip_append", None)
        if manager is not None:
            if manager_had_skip:
                manager._skip_append = manager_skip_value
            elif hasattr(manager, "_skip_append"):
                delattr(manager, "_skip_append")
    after = persistent_state_signature(model)
    identity = before == after
    audit = {
        "label": label,
        "elapsed_s": time.monotonic() - started,
        "state_identity": identity,
        "aggregator_time_before": saved_agg_time,
        "camera_time_before": saved_camera_time,
        "aggregator_time_after": int(model.aggregator.total_frames_processed),
        "camera_time_after": int(model.camera_head.frame_idx),
    }
    return result, identity, audit


def _discover_episodes(args: argparse.Namespace) -> list[Path]:
    candidates: list[tuple[int, str, Path]] = []
    for metadata_path in args.episode_root.glob(
            "*/episode_*/meta/gen_meta.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        goals = metadata.get("goals", [])
        if (int(metadata.get("n_legs", 2)) != 2 or not goals
                or goals[0].get("kind") != "revisit"):
            continue
        episode = metadata_path.parent.parent
        relative = episode.relative_to(args.episode_root).as_posix()
        candidates.append((int(metadata["switch_idx"]), relative, episode))
    if args.episode:
        wanted = set(args.episode)
        selected = [item for item in candidates if item[1] in wanted]
        found = {item[1] for item in selected}
        require(found == wanted,
                f"unknown episode(s): {sorted(wanted - found)}")
        selected.sort(key=lambda item: args.episode.index(item[1]))
    else:
        candidates.sort(key=(
            (lambda item: (item[0], item[1]))
            if args.selection == "shortest"
            else (lambda item: item[1])))
        selected = candidates[:args.episodes]
    require(bool(selected), "no eligible 2-leg Revisit episodes found")
    return [item[2] for item in selected]


def _load_episode(episode: Path) -> dict[str, Any]:
    metadata = json.loads(
        (episode / "meta/gen_meta.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(
        episode / "data/chunk-000/episode_000000.parquet",
        columns=["action", "observation.camera_extrinsic"])
    actions = np.stack([_matrix(value, "action") for value in frame["action"]])
    require(len(actions) == int(metadata["n_frames"]),
            "metadata/parquet frame count mismatch")
    mount = _resolve_generated_mount(
        _matrix(frame.iloc[0]["observation.camera_extrinsic"],
                "camera extrinsic"),
        str(metadata.get("frame_convention", "")))
    switch = int(metadata["switch_idx"])
    goal = metadata["goals"][0]
    anchor = int(goal["covis_argmax"])
    require(0 <= anchor < switch <= len(actions),
            "invalid anchor/switch ordering")
    rgb = episode / "videos/chunk-000/observation.images.rgb"
    paths = [rgb / f"{index}.jpg" for index in range(switch)]
    require(all(path.is_file() for path in paths),
            "online Goal-A history has missing frames")
    goal_path = episode / "goal_1.jpg"
    require(goal_path.is_file(), "goal_1.jpg is missing")
    return {
        "metadata": metadata,
        "actions": actions,
        "mount": mount,
        "switch": switch,
        "goal": goal,
        "anchor": anchor,
        "rgb_paths": paths,
        "goal_path": goal_path,
    }


def _finite_json(value: object) -> object:
    """Replace non-finite floats so strict JSON remains valid."""
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (bool, str)) or value is None:
        return value
    return str(value)


def _cdf(errors: Iterable[float], threshold: float) -> dict[str, Any]:
    values = [float(value) for value in errors if math.isfinite(float(value))]
    hits = sum(value <= threshold for value in values)
    return {"hits": hits, "total": len(values),
            "rate": hits / len(values) if values else None}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row["candidate_free_direction_error_deg"] for row in rows]
    oracle = [row["oracle_locality_direction_error_deg"] for row in rows]
    return {
        "episodes": len(rows),
        "candidate_free": {
            "median_direction_error_deg": float(np.nanmedian(full)),
            "cdf_le_15": _cdf(full, 15.0),
            "cdf_le_30": _cdf(full, 30.0),
            "cdf_le_45": _cdf(full, 45.0),
        },
        "oracle_locality": {
            "median_direction_error_deg": float(np.nanmedian(oracle)),
            "cdf_le_15": _cdf(oracle, 15.0),
            "cdf_le_30": _cdf(oracle, 30.0),
            "cdf_le_45": _cdf(oracle, 45.0),
        },
        "all_query_state_identity": all(
            bool(row["all_query_state_identity"]) for row in rows),
        "max_repeat_pose_l2": max(
            float(row["candidate_free_repeat_pose_l2"]) for row in rows),
        "interpretation": (
            "mechanism_smoke_only_not_a_pass_fail_for_training"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path,
                        default=DEFAULT_EPISODE_ROOT)
    parser.add_argument(
        "--episode", action="append", default=[],
        help="Explicit scene/episode_XXXX relative path; repeatable.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--selection", choices=("shortest", "lexical"),
                        default="shortest")
    parser.add_argument("--lingbot-repo", type=Path,
                        default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-iterations", type=int, default=4)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = args.lingbot_repo / "weights/lingbot-map-long.pt"
    require(args.episodes >= 1, "--episodes must be positive")
    require(args.lingbot_repo.is_dir(), "LingBot repository is missing")
    require(args.weights.is_file(), "LingBot weights are missing")
    require(args.episode_root.is_dir(), "episode root is missing")
    require(args.device.startswith("cuda") and torch.cuda.is_available(),
            "this diagnostic requires CUDA")
    return args


def main() -> None:
    args = parse_args()
    episode_paths = _discover_episodes(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[selection]", *(path.relative_to(args.episode_root)
                           for path in episode_paths), flush=True)

    model = _build_model(args)
    from lingbot_map.utils.load_fn import (  # pylint: disable=import-error
        load_and_preprocess_images)

    rows: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episode_paths, start=1):
        loaded = _load_episode(episode)
        switch, anchor = loaded["switch"], loaded["anchor"]
        print(f"[episode {episode_index}/{len(episode_paths)}] "
              f"{episode.relative_to(args.episode_root)} "
              f"switch={switch} anchor={anchor} "
              f"gap={loaded['goal'].get('recall_gap')}", flush=True)
        images = load_and_preprocess_images(
            [os.fspath(path) for path in loaded["rgb_paths"]],
            mode="pad", image_size=args.image_size,
            patch_size=args.patch_size)
        goal_image = load_and_preprocess_images(
            [os.fspath(loaded["goal_path"])], mode="pad",
            image_size=args.image_size, patch_size=args.patch_size)[0]

        current_pose = _stream_prefix(
            model, images, num_scale=args.num_scale,
            device=args.device, progress_label="full-history")
        candidate_goal_1, identity_1, audit_1 = _read_only_query(
            model, goal_image, num_scale=args.num_scale,
            device=args.device, label="candidate_free_1")
        candidate_goal_2, identity_2, audit_2 = _read_only_query(
            model, goal_image, num_scale=args.num_scale,
            device=args.device, label="candidate_free_repeat")
        self_goal, identity_self, audit_self = _read_only_query(
            model, images[-1], num_scale=args.num_scale,
            device=args.device, label="current_frame_goal_swap_control")

        # Oracle-locality only changes where the query is attached.  It does not
        # provide GT coordinates to the frozen model.
        _stream_prefix(
            model, images[:anchor + 1], num_scale=args.num_scale,
            device=args.device, progress_label="oracle-anchor-prefix")
        oracle_goal, identity_oracle, audit_oracle = _read_only_query(
            model, goal_image, num_scale=args.num_scale,
            device=args.device, label="oracle_locality")

        gt = _navdp_ground_truth_relative(
            loaded["actions"][switch - 1],
            _goal_camera_to_world(loaded["goal"]), loaded["mount"])
        candidate_direction = _lingbot_relative_direction(
            current_pose, candidate_goal_1)
        oracle_direction = _lingbot_relative_direction(
            current_pose, oracle_goal)
        self_direction = _lingbot_relative_direction(current_pose, self_goal)
        row = {
            "episode": episode.relative_to(args.episode_root).as_posix(),
            "scene": episode.parent.name,
            "switch_idx": switch,
            "anchor_idx": anchor,
            "recall_gap": int(loaded["goal"].get(
                "recall_gap", switch - 1 - anchor)),
            "teacher_covis": float(loaded["goal"]["covis"]),
            "gt_pointgoal": gt.tolist(),
            "gt_distance_m": float(np.linalg.norm(gt)),
            "gt_bearing_deg": signed_bearing_degrees(gt),
            "candidate_free_direction": candidate_direction.tolist(),
            "candidate_free_raw_norm": float(np.linalg.norm(
                candidate_direction)),
            "candidate_free_bearing_deg": signed_bearing_degrees(
                candidate_direction),
            "candidate_free_direction_error_deg": direction_error_degrees(
                candidate_direction, gt),
            "oracle_locality_direction": oracle_direction.tolist(),
            "oracle_locality_raw_norm": float(np.linalg.norm(
                oracle_direction)),
            "oracle_locality_bearing_deg": signed_bearing_degrees(
                oracle_direction),
            "oracle_locality_direction_error_deg": direction_error_degrees(
                oracle_direction, gt),
            "self_query_raw_norm": float(np.linalg.norm(self_direction)),
            "goal_swap_pose_l2": float(np.linalg.norm(
                candidate_goal_1[:7] - self_goal[:7])),
            "candidate_free_repeat_pose_l2": float(np.linalg.norm(
                candidate_goal_1 - candidate_goal_2)),
            "all_query_state_identity": bool(
                identity_1 and identity_2 and identity_self
                and identity_oracle),
            "query_audits": [audit_1, audit_2, audit_self, audit_oracle],
        }
        rows.append(row)
        print(
            f"[result] GT={row['gt_bearing_deg']:+.1f}deg; "
            f"candidate-free={row['candidate_free_bearing_deg']:+.1f}deg "
            f"(err {row['candidate_free_direction_error_deg']:.1f}); "
            f"oracle-locality={row['oracle_locality_bearing_deg']:+.1f}deg "
            f"(err {row['oracle_locality_direction_error_deg']:.1f}); "
            f"cache_identity={row['all_query_state_identity']}", flush=True)
        del images, goal_image
        model.clean_kv_cache()
        torch.cuda.empty_cache()

    report = {
        "schema": "m2p_s1_gct_query_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "episode_root": os.fspath(args.episode_root.resolve()),
            "lingbot_repo": os.fspath(args.lingbot_repo.resolve()),
            "lingbot_revision": _git_revision(args.lingbot_repo),
            "weights": os.fspath(args.weights.resolve()),
            "weights_size_bytes": args.weights.stat().st_size,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "num_scale": args.num_scale,
            "window": args.window,
            "max_frame_num": args.max_frame_num,
            "camera_iterations": args.camera_iterations,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "summary": _summary(rows),
        "episodes": rows,
    }
    strict_report = _finite_json(report)
    _atomic_json(args.out / "report.json", strict_report)
    table_rows = [{key: value for key, value in row.items()
                   if key != "query_audits"} for row in rows]
    pd.DataFrame(table_rows).to_csv(args.out / "episodes.csv", index=False)
    print(json.dumps(strict_report["summary"], indent=2,
                     sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
