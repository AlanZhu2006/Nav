#!/usr/bin/env python3
"""Gate-B real-data overfit for the monocular Geometry Token Adapter.

This script uses 16 fixed states from one *already consumed* local trajectory.
It is an optimization/contract test only.  Passing it authorizes building the
scene-grouped PT1 training bundle; it is not evidence of generalization or SR.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNNAV_ROOT = REPO_ROOT / "InternNav"


def _extract_states(args, navdp_agent, navdp):
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    from MemNavData.preflight_monocular_geometry_adapter import (
        _causal_lingbot_scale_features,
        _teacher_inputs,
    )

    episode = args.episode.resolve()
    feature_episode = args.feature_episode.resolve()
    meta = json.loads((episode / "meta/gen_meta.json").read_text())
    camera_height_m = float(meta.get("camera_height_m", 0.5))
    switch = int(meta["switches"][0])
    maximum = min(switch - 1, args.max_frame)
    if maximum < args.min_frame:
        raise ValueError("episode does not contain the requested overfit frame range")
    frames = np.linspace(
        args.min_frame, maximum, args.states, dtype=np.int64
    ).tolist()
    if len(set(frames)) != args.states:
        raise ValueError("overfit frame range is too short for unique states")

    rgb_dir = episode / "videos/chunk-000/observation.images.rgb"
    depth_dir = episode / "videos/chunk-000/observation.images.depth"
    cache_path = feature_episode / "videos/chunk-000/lingbot_cache.npz"
    cam_path = feature_episode / "videos/chunk-000/lingbot_cam_cache.npz"
    with np.load(cache_path, allow_pickle=False) as raw:
        cache_np = {key: raw[key] for key in raw.files}
    with np.load(cam_path, allow_pickle=False) as raw:
        cam_np = {key: raw[key] for key in raw.files}

    lingbot = LingBotStream(
        lingbot_repo=str(args.lingbot_repo),
        weights=str(args.lingbot_weights),
        num_scale=8,
        window=32,
        max_frame_num=2048,
        device=args.device,
        use_sdpa=True,
    )
    scale_k, scale_v = lingbot.get_scale_kv(str(rgb_dir))
    anchor_k = torch.as_tensor(
        cache_np["anchor_k"], device=args.device, dtype=torch.bfloat16
    ).permute(1, 2, 0, 3, 4).contiguous()
    anchor_v = torch.as_tensor(
        cache_np["anchor_v"], device=args.device, dtype=torch.bfloat16
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
    scale_features, scale_receipt = _causal_lingbot_scale_features(
        lingbot,
        rgb_dir,
        np.asarray(cam_np["cam_pose_enc"]),
        camera_height_m=camera_height_m,
        prefix_frames=args.scale_prefix_frames,
    )

    extracted = {
        "recent_specials": [],
        "current_patches": [],
        "depth_features": [],
        "scale_features": [],
        "teacher_tokens": [],
    }
    frame_times = []
    for frame in frames:
        started = time.perf_counter()
        paths = [rgb_dir / f"{index}.jpg" for index in range(frame - 31, frame + 1)]
        images = lingbot.load_images([str(path) for path in paths]).to(args.device)
        with torch.no_grad():
            window, current_agg, patch_start = lingbot.window_forward(
                cache, images, frame, return_multilayer=True
            )
            depth_feature = lingbot.depth_feature(
                current_agg, images[-1:][None], patch_start
            )
            teacher_rgb, teacher_depth = _teacher_inputs(
                navdp_agent, rgb_dir, depth_dir, frame
            )
            teacher = navdp.rgbd_encoder(teacher_rgb, teacher_depth)
        extracted["recent_specials"].append(
            window[-8:, :6].to(dtype=torch.float16, device="cpu")
        )
        extracted["current_patches"].append(
            window[-1, 6:].to(dtype=torch.float16, device="cpu")
        )
        extracted["depth_features"].append(
            depth_feature.to(dtype=torch.float16, device="cpu")
        )
        extracted["scale_features"].append(scale_features[0])
        extracted["teacher_tokens"].append(teacher[0].float().cpu())
        frame_times.append(time.perf_counter() - started)

    states = {name: torch.stack(values) for name, values in extracted.items()}
    source_bytes = int(sum(value.numel() * value.element_size() for value in states.values()))
    receipt = {
        "episode": str(episode),
        "feature_episode": str(feature_episode),
        "frames": frames,
        "state_count": len(frames),
        "source_bytes": source_bytes,
        "per_frame_extract_seconds": frame_times,
        "scale": scale_receipt,
        "shapes": {name: list(value.shape) for name, value in states.items()},
    }
    del lingbot, cache, scale_k, scale_v, anchor_k, anchor_v
    gc.collect()
    torch.cuda.empty_cache()
    return states, receipt


def _functional_targets(states, navdp, args):
    generator = torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
    count = states["teacher_tokens"].shape[0]
    noisy = torch.randn(
        count, navdp.predict_size, 3, generator=generator, device=args.device
    )
    goal = torch.randn(
        count, 1, navdp.token_dim, generator=generator, device=args.device
    )
    candidates = torch.randn(
        count,
        args.candidates,
        navdp.predict_size,
        3,
        generator=generator,
        device=args.device,
    )
    teacher = states["teacher_tokens"].to(args.device)
    timestep = torch.tensor([args.timestep], dtype=torch.long, device=args.device)
    with torch.no_grad():
        teacher_epsilon = navdp.predict_noise(noisy, timestep, goal, teacher)
        teacher_critic = navdp.predict_critic(
            candidates.flatten(0, 1),
            teacher.repeat_interleave(args.candidates, dim=0),
        ).reshape(count, args.candidates)
    return {
        "noisy": noisy.cpu(),
        "goal": goal.cpu(),
        "candidates": candidates.cpu(),
        "teacher_epsilon": teacher_epsilon.cpu(),
        "teacher_critic": teacher_critic.cpu(),
    }


def _batch_loss(adapter, navdp, states, functional, indices, args):
    from MemNavData.monocular_geometry_adapter import geometry_distillation_losses

    index = torch.as_tensor(indices, dtype=torch.long)
    student = adapter.forward_compact(
        states["recent_specials"][index].to(args.device, dtype=torch.float32),
        states["current_patches"][index].to(args.device, dtype=torch.float32),
        states["depth_features"][index].to(args.device, dtype=torch.float32),
        states["scale_features"][index].to(args.device, dtype=torch.float32),
    )
    teacher = states["teacher_tokens"][index].to(args.device)
    noisy = functional["noisy"][index].to(args.device)
    goal = functional["goal"][index].to(args.device)
    timestep = torch.tensor([args.timestep], dtype=torch.long, device=args.device)
    student_epsilon = navdp.predict_noise(noisy, timestep, goal, student)
    candidates = functional["candidates"][index].to(args.device)
    student_critic = navdp.predict_critic(
        candidates.flatten(0, 1),
        student.repeat_interleave(args.candidates, dim=0),
    ).reshape(len(indices), args.candidates)
    return geometry_distillation_losses(
        student,
        teacher,
        student_epsilon=student_epsilon,
        teacher_epsilon=functional["teacher_epsilon"][index].to(args.device),
        student_critic=student_critic,
        teacher_critic=functional["teacher_critic"][index].to(args.device),
    )


@torch.no_grad()
def _evaluate(adapter, navdp, states, functional, args):
    adapter.eval()
    totals = {}
    for start in range(0, args.states, args.batch_size):
        indices = list(range(start, min(args.states, start + args.batch_size)))
        losses = _batch_loss(adapter, navdp, states, functional, indices, args)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(indices)
    adapter.train()
    return {name: value / args.states for name, value in totals.items()}


def run(args):
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(INTERNNAV_ROOT))
    from MemNavData.monocular_geometry_adapter import GeometryTokenAdapter
    from MemNavData.preflight_monocular_geometry_adapter import _load_navdp_agent

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    navdp_agent = _load_navdp_agent(args.navdp_checkpoint, args.device)
    navdp = navdp_agent.navi_former
    checkpoint = torch.load(
        args.navdp_checkpoint, map_location="cpu", weights_only=False
    )
    states, extraction = _extract_states(args, navdp_agent, navdp)
    functional = _functional_targets(states, navdp, args)

    adapter = GeometryTokenAdapter().to(args.device)
    query_key = adapter.initialize_queries_from_navdp(checkpoint)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = _evaluate(adapter, navdp, states, functional, args)
    history = [{"step": 0, **initial}]
    generator = torch.Generator().manual_seed(args.seed + 1)
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            args.states, (args.batch_size,), generator=generator
        ).tolist()
        optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(adapter, navdp, states, functional, indices, args)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            current = _evaluate(adapter, navdp, states, functional, args)
            history.append({"step": step, **current})
            print(
                f"step={step} total={current['loss']:.6f} "
                f"token={current['token']:.6f} "
                f"denoise={current['denoise']:.6f} "
                f"critic={current['critic']:.6f} rank={current['rank']:.6f}",
                flush=True,
            )

    final = history[-1]
    ratios = {
        name: final[name] / max(initial[name], 1e-12)
        for name in ("loss", "token", "denoise", "critic", "rank")
    }
    passed = (
        ratios["loss"] <= args.max_total_ratio
        and ratios["token"] <= args.max_token_ratio
        and ratios["denoise"] <= args.max_denoise_ratio
        and math.isfinite(final["loss"])
    )
    navdp_gradients = [
        name for name, parameter in navdp.named_parameters() if parameter.grad is not None
    ]
    if navdp_gradients:
        raise RuntimeError("frozen NavDP accumulated parameter gradients")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "overfit_adapter_only.pt"
    torch.save(
        {
            "architecture": "geometry_token_adapter_v1",
            "state_dict": adapter.state_dict(),
            "config": adapter.config.to_dict(),
            "teacher_query_key": query_key,
            "diagnostic_only": True,
        },
        checkpoint_path,
    )
    receipt = {
        "status": "passed" if passed else "failed",
        "gate": "B_real_16_state_overfit",
        "diagnostic_only": True,
        "generalization_claim": False,
        "teacher_query_key": query_key,
        "initial": initial,
        "final": final,
        "final_over_initial": ratios,
        "thresholds": {
            "max_total_ratio": args.max_total_ratio,
            "max_token_ratio": args.max_token_ratio,
            "max_denoise_ratio": args.max_denoise_ratio,
        },
        "history": history,
        "extraction": extraction,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint": str(checkpoint_path),
        "navdp_gradient_tensors": navdp_gradients,
        "decision": (
            "authorize_scene_grouped_pt1_bundle"
            if passed
            else "stop_do_not_submit_hpc_long_train"
        ),
    }
    (args.output_dir / "overfit_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


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
    parser.add_argument("--min-frame", type=int, default=40)
    parser.add_argument("--max-frame", type=int, default=160)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--scale-prefix-frames", type=int, default=40)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--timestep", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-total-ratio", type=float, default=0.35)
    parser.add_argument("--max-token-ratio", type=float, default=0.35)
    parser.add_argument("--max-denoise-ratio", type=float, default=0.50)
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
        "--lingbot-repo", type=Path, default=Path("/home/asus/Research/lingbot-map")
    )
    parser.add_argument(
        "--lingbot-weights",
        type=Path,
        default=Path("/home/asus/Research/lingbot-map/weights/lingbot-map-long.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".diagnostics/monocular_geometry_adapter_20260818",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
