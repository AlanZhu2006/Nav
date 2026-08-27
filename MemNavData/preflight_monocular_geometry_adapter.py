#!/usr/bin/env python3
"""One-state real-model preflight for the monocular geometry adapter.

The script is intentionally a diagnostic, not a trainer.  It loads one already
consumed local MP3D trajectory, reconstructs a causal LingBot window from the
versioned sparse cache, computes the official NavDP RGB-D teacher latent, and
runs one adapter-only backward pass through the frozen NavDP decoder/critic.

No navigation outcome or held-out population is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNNAV_ROOT = REPO_ROOT / "InternNav"
NAVDP_ROOT = REPO_ROOT / "NavDP" / "baselines" / "navdp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_navdp_agent(checkpoint: Path, device: str):
    # NavDP's inference files use sibling absolute imports (``policy_network``),
    # so import them with their directory first on sys.path.
    sys.path.insert(0, str(NAVDP_ROOT))
    try:
        module = importlib.import_module("policy_agent")
        agent = module.NavDP_Agent(
            np.eye(3, dtype=np.float64),
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=str(checkpoint),
            device=device,
        )
    finally:
        sys.path.pop(0)
    for parameter in agent.navi_former.parameters():
        parameter.requires_grad_(False)
    agent.navi_former.eval()
    return agent


def _load_bgr(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _teacher_inputs(agent, rgb_dir: Path, depth_dir: Path, frame: int):
    indices = list(range(frame - 7, frame + 1))
    rgb = np.stack([_load_bgr(rgb_dir / f"{index}.jpg") for index in indices])
    processed_rgb = agent.process_image(rgb)[None]
    depth = np.asarray(
        Image.open(depth_dir / f"{frame}.png").convert("I"), dtype=np.float32
    )[None, :, :, None] / 10000.0
    # generate_twoleg.py persists Habitat metric depth as uint16 metres*10000.
    # Omitting this conversion makes process_depth clip almost every non-zero
    # pixel at >5 and silently turns the privileged teacher into zero-depth.
    nonzero = depth[depth > 0]
    if nonzero.size == 0 or float(np.median(nonzero)) > 20.0:
        raise ValueError("decoded teacher depth is empty or not expressed in metres")
    processed_depth = agent.process_depth(depth)
    return processed_rgb, processed_depth


def _lingbot_scale_features(
    camera_height_m: float, ground_h_est: float, ground_dbg: np.ndarray
) -> tuple[torch.Tensor, dict[str, object]]:
    from internnav.model.basemodel.memnav.lingbot_stream import (
        GROUND_BIAS_CORRECTION,
        GROUND_SCALE_RANGE,
        ground_scale_from_h_est,
    )

    valid = math.isfinite(ground_h_est) and ground_h_est > 0.0
    if valid:
        scale = ground_scale_from_h_est(ground_h_est, camera_height_m)
        raw_scale = GROUND_BIAS_CORRECTION * camera_height_m / ground_h_est
        valid = scale is not None and math.isfinite(scale) and scale > 0.0
    else:
        scale = None
        raw_scale = float("nan")
    if valid:
        n_frames = max(float(ground_dbg[1]), 1.0)
        valid_ratio = float(ground_dbg[2]) / n_frames
        relative_iqr = float(ground_dbg[3]) / ground_h_est
        clamped = float(not math.isclose(float(scale), raw_scale, rel_tol=1e-6))
        log_scale = math.log(float(scale))
    else:
        valid_ratio = relative_iqr = clamped = log_scale = 0.0
    features = torch.tensor(
        [[
            math.log(camera_height_m),
            log_scale,
            float(valid),
            valid_ratio,
            relative_iqr,
            clamped,
        ]],
        dtype=torch.float32,
    )
    receipt = {
        "camera_height_m": camera_height_m,
        "ground_h_est_raw": ground_h_est,
        "scale_valid": bool(valid),
        "scale_hat": None if not valid else float(scale),
        "valid_frame_ratio": valid_ratio,
        "relative_floor_iqr": relative_iqr,
        "scale_clamped": bool(clamped),
    }
    return features, receipt


def _causal_scale_prefix(
    rgb_dir: Path,
    cam_pose_enc: np.ndarray,
    prefix_frames: int = 40,
) -> tuple[list[Path], np.ndarray]:
    """Return only the deployment-visible prefix used for metric scale.

    The ``ground_h_est`` stored in legacy camera caches pools an entire episode.
    It is valid for offline localization audits, but not for a causal controller
    state.  Gate C therefore freezes one early RGB prefix and never consumes the
    cached whole-episode aggregate.
    """

    prefix_frames = int(prefix_frames)
    if prefix_frames < 8:
        raise ValueError("causal scale prefix must include the 8-frame scale block")
    poses = np.asarray(cam_pose_enc)
    if poses.ndim != 2 or poses.shape[1] != 9:
        raise ValueError(f"cam_pose_enc must have shape [T,9], got {poses.shape}")
    if len(poses) < prefix_frames:
        raise ValueError(
            f"only {len(poses)} camera poses for {prefix_frames}-frame scale prefix"
        )
    paths = [rgb_dir / f"{index}.jpg" for index in range(prefix_frames)]
    if not all(path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(f"causal scale RGB prefix is incomplete: {missing}")
    return paths, poses[:prefix_frames].copy()


def _causal_lingbot_scale_features(
    lingbot,
    rgb_dir: Path,
    cam_pose_enc: np.ndarray,
    camera_height_m: float,
    prefix_frames: int = 40,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Estimate one immutable RGB-only scale from the first causal prefix."""

    paths, poses = _causal_scale_prefix(rgb_dir, cam_pose_enc, prefix_frames)
    scale, debug = lingbot.compute_metric_scale(
        [str(path) for path in paths],
        poses,
        camera_height_m=float(camera_height_m),
        n_frames=int(prefix_frames),
        return_debug=True,
    )
    debug = dict(debug)
    h_est = debug.get("h_est")
    ground_dbg = np.asarray(
        [
            debug.get("n_points", 0),
            debug.get("n_frames", 0),
            debug.get("n_valid", 0),
            np.nan if debug.get("h_iqr") is None else debug["h_iqr"],
        ],
        dtype=np.float64,
    )
    features, receipt = _lingbot_scale_features(
        float(camera_height_m),
        float("nan") if h_est is None else float(h_est),
        ground_dbg,
    )
    if receipt["scale_valid"]:
        if scale is None or not math.isclose(
            float(receipt["scale_hat"]), float(scale), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise RuntimeError("causal scale helper and feature receipt disagree")
    elif scale is not None:
        raise RuntimeError("invalid scale receipt retained a metric scale")
    receipt.update(
        {
            "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
            "scale_prefix_frames": int(prefix_frames),
            "scale_prefix_first_frame": 0,
            "scale_prefix_last_frame": int(prefix_frames) - 1,
            "whole_episode_ground_cache_consumed": False,
        }
    )
    return features, receipt


def _load_lingbot_state(
    episode: Path,
    feature_episode: Path,
    frame: int,
    camera_height_m: float,
    lingbot_repo: Path,
    lingbot_weights: Path,
    device: str,
):
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream

    rgb_dir = episode / "videos/chunk-000/observation.images.rgb"
    cache_path = feature_episode / "videos/chunk-000/lingbot_cache.npz"
    cam_path = feature_episode / "videos/chunk-000/lingbot_cam_cache.npz"
    if not cache_path.is_file() or not cam_path.is_file():
        raise FileNotFoundError(f"missing LingBot cache pair under {feature_episode}")

    lingbot = LingBotStream(
        lingbot_repo=str(lingbot_repo),
        weights=str(lingbot_weights),
        num_scale=8,
        window=32,
        max_frame_num=2048,
        device=device,
        use_sdpa=True,
    )
    with np.load(cache_path, allow_pickle=False) as raw:
        cache_np = {key: raw[key] for key in raw.files}
    with np.load(cam_path, allow_pickle=False) as raw:
        cam_np = {key: raw[key] for key in raw.files}
    if int(cache_np["num_frames"].item()) <= frame:
        raise ValueError("requested frame lies outside the LingBot cache")
    if int(cache_np["kv_cache_sliding_window"].item()) != 32:
        raise ValueError("preflight requires a cache generated with window=32")

    scale_k, scale_v = lingbot.get_scale_kv(str(rgb_dir))
    anchor_k = torch.as_tensor(
        cache_np["anchor_k"], device=device, dtype=torch.bfloat16
    ).permute(1, 2, 0, 3, 4).contiguous()
    anchor_v = torch.as_tensor(
        cache_np["anchor_v"], device=device, dtype=torch.bfloat16
    ).permute(1, 2, 0, 3, 4).contiguous()
    cache = {
        "scale_k": scale_k,
        "scale_v": scale_v,
        "anchor_k": anchor_k,
        "anchor_v": anchor_v,
        "anchor_frame_indices": torch.as_tensor(
            cache_np["anchor_frame_indices"], dtype=torch.long
        ),
    }
    paths = [rgb_dir / f"{index}.jpg" for index in range(frame - 31, frame + 1)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("causal 32-frame RGB window is incomplete")
    images = lingbot.load_images([str(path) for path in paths]).to(device)
    with torch.no_grad():
        window, current_agg, patch_start = lingbot.window_forward(
            cache, images, frame, return_multilayer=True
        )
        depth_features = lingbot.depth_feature(
            current_agg, images[-1:][None], patch_start
        )
    scale_features, scale_receipt = _causal_lingbot_scale_features(
        lingbot,
        rgb_dir,
        np.asarray(cam_np["cam_pose_enc"]),
        camera_height_m=float(camera_height_m),
        prefix_frames=40,
    )
    return (
        lingbot,
        window[-8:][None].float(),
        depth_features[None].float(),
        scale_features,
        scale_receipt,
        {
            "aggregator_cache": str(cache_path),
            "camera_cache": str(cam_path),
            "window_shape": list(window.shape),
            "adapter_window_shape": list(window[-8:][None].shape),
            "depth_feature_shape": list(depth_features[None].shape),
            "patch_start": int(patch_start),
            "scale_evidence_contract": scale_receipt["scale_evidence_contract"],
        },
    )


def _freeze_audit(navdp, adapter) -> dict[str, object]:
    frozen_trainable = [
        name for name, parameter in navdp.named_parameters() if parameter.requires_grad
    ]
    adapter_trainable = [
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    ]
    if frozen_trainable:
        raise RuntimeError(f"NavDP unexpectedly trainable: {frozen_trainable[:8]}")
    if not adapter_trainable:
        raise RuntimeError("adapter has no trainable parameters")
    return {
        "navdp_trainable_tensors": frozen_trainable,
        "adapter_trainable_tensor_count": len(adapter_trainable),
    }


def run(args) -> dict[str, object]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(INTERNNAV_ROOT))
    from MemNavData.monocular_geometry_adapter import (
        GeometryTokenAdapter,
        adapter_parameter_receipt,
        geometry_distillation_losses,
    )

    device = args.device
    episode = args.episode.resolve()
    feature_episode = args.feature_episode.resolve()
    meta = json.loads((episode / "meta/gen_meta.json").read_text())
    camera_height_m = float(meta.get("camera_height_m", 0.5))
    rgb_dir = episode / "videos/chunk-000/observation.images.rgb"
    depth_dir = episode / "videos/chunk-000/observation.images.depth"

    started = time.perf_counter()
    navdp_agent = _load_navdp_agent(args.navdp_checkpoint, device)
    navdp = navdp_agent.navi_former
    checkpoint_state = torch.load(
        args.navdp_checkpoint, map_location="cpu", weights_only=False
    )
    teacher_rgb, teacher_depth = _teacher_inputs(
        navdp_agent, rgb_dir, depth_dir, args.frame
    )
    with torch.no_grad():
        teacher_tokens = navdp.rgbd_encoder(teacher_rgb, teacher_depth).detach()

    (
        lingbot,
        window_tokens,
        depth_features,
        scale_features,
        scale_receipt,
        lingbot_receipt,
    ) = _load_lingbot_state(
        episode,
        feature_episode,
        args.frame,
        camera_height_m,
        args.lingbot_repo,
        args.lingbot_weights,
        device,
    )
    scale_features = scale_features.to(device)

    adapter = GeometryTokenAdapter().to(device)
    teacher_query_key = adapter.initialize_queries_from_navdp(checkpoint_state)
    freeze_receipt = _freeze_audit(navdp, adapter)
    student_tokens = adapter(window_tokens, depth_features, scale_features)
    if tuple(student_tokens.shape) != tuple(teacher_tokens.shape):
        raise RuntimeError(
            f"student shape {tuple(student_tokens.shape)} != teacher "
            f"{tuple(teacher_tokens.shape)}"
        )

    batch = student_tokens.shape[0]
    generator = torch.Generator(device=torch.device(device)).manual_seed(args.seed)
    clean = torch.randn(
        batch, navdp.predict_size, 3, generator=generator, device=device
    )
    noise = torch.randn(
        clean.shape, generator=generator, device=device
    )
    timestep_value = int(args.seed % navdp.noise_scheduler.config.num_train_timesteps)
    timesteps = torch.full(
        (batch,), timestep_value, dtype=torch.long, device=device
    )
    noisy = navdp.noise_scheduler.add_noise(clean, noise, timesteps)
    goal_embed = torch.randn(
        batch, 1, navdp.token_dim, generator=generator, device=device
    )
    model_timestep = torch.tensor([timestep_value], dtype=torch.long, device=device)
    with torch.no_grad():
        teacher_epsilon = navdp.predict_noise(
            noisy, model_timestep, goal_embed, teacher_tokens
        )
    student_epsilon = navdp.predict_noise(
        noisy, model_timestep, goal_embed, student_tokens
    )

    candidate_count = 4
    candidates = torch.randn(
        candidate_count,
        navdp.predict_size,
        3,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        teacher_critic = navdp.predict_critic(
            candidates, teacher_tokens.expand(candidate_count, -1, -1)
        )[None]
    student_critic = navdp.predict_critic(
        candidates, student_tokens.expand(candidate_count, -1, -1)
    )[None]
    losses = geometry_distillation_losses(
        student_tokens,
        teacher_tokens,
        student_epsilon=student_epsilon,
        teacher_epsilon=teacher_epsilon,
        student_critic=student_critic,
        teacher_critic=teacher_critic,
    )
    losses["loss"].backward()

    adapter_grad_norm_sq = 0.0
    adapter_missing_grad = []
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None:
            adapter_missing_grad.append(name)
        else:
            adapter_grad_norm_sq += float(parameter.grad.float().square().sum())
    navdp_grad_tensors = [
        name for name, parameter in navdp.named_parameters() if parameter.grad is not None
    ]
    lingbot_grad_tensors = [
        name for name, parameter in lingbot.named_parameters() if parameter.grad is not None
    ]
    if navdp_grad_tensors or lingbot_grad_tensors:
        raise RuntimeError("a frozen expert accumulated parameter gradients")
    if not math.isfinite(adapter_grad_norm_sq) or adapter_grad_norm_sq <= 0.0:
        raise RuntimeError("adapter gradient norm is invalid or zero")

    token_cosine = torch.nn.functional.cosine_similarity(
        student_tokens.detach(), teacher_tokens, dim=-1
    ).mean()
    elapsed = time.perf_counter() - started
    return {
        "status": "passed",
        "diagnostic_only": True,
        "episode": str(episode),
        "frame": int(args.frame),
        "seed": int(args.seed),
        "device": device,
        "elapsed_seconds": elapsed,
        "navdp_checkpoint": str(args.navdp_checkpoint),
        "navdp_checkpoint_sha256": _sha256(args.navdp_checkpoint),
        "lingbot_weights": str(args.lingbot_weights),
        "lingbot_weights_sha256": _sha256(args.lingbot_weights),
        "teacher_query_key": teacher_query_key,
        "teacher_shape": list(teacher_tokens.shape),
        "student_shape": list(student_tokens.shape),
        "untrained_token_cosine": float(token_cosine),
        "losses_before_training": {
            name: float(value.detach()) for name, value in losses.items()
        },
        "adapter_gradient_norm": math.sqrt(adapter_grad_norm_sq),
        "adapter_missing_gradient_tensors": adapter_missing_grad,
        "navdp_gradient_tensors": navdp_grad_tensors,
        "lingbot_gradient_tensors": lingbot_grad_tensors,
        "adapter": adapter_parameter_receipt(adapter),
        "freeze_audit": freeze_receipt,
        "lingbot_state": lingbot_receipt,
        "scale": scale_receipt,
        "scientific_interpretation": (
            "shape/gradient/runtime contract only; untrained losses are not an "
            "effectiveness result"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=Path(
            "/home/asus/Research/datasets/memnav_eval2leg_v1/"
            "17DRP5sb8fy/episode_0002"
        ),
    )
    parser.add_argument(
        "--feature-episode",
        type=Path,
        default=Path(
            "/home/asus/Research/datasets/memnav_eval2leg_v1_feat_flowgate/"
            "mp3d_eval2leg/17DRP5sb8fy/episode_0002"
        ),
    )
    parser.add_argument("--frame", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--navdp-checkpoint",
        type=Path,
        default=Path(
            "/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/"
            "navdp_checkpoint.ckpt"
        ),
    )
    parser.add_argument(
        "--lingbot-repo",
        type=Path,
        default=Path("/home/asus/Research/lingbot-map"),
    )
    parser.add_argument(
        "--lingbot-weights",
        type=Path,
        default=Path("/home/asus/Research/lingbot-map/weights/lingbot-map-long.pt"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame < 31:
        raise ValueError("real preflight frame must be >=31 for the frozen window")
    receipt = run(args)
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
