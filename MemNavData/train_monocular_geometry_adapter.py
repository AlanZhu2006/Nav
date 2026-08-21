#!/usr/bin/env python3
"""Scene-grouped training and Gate-C audit for Geometry Token Adapter."""

from __future__ import annotations

import argparse
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
SCHEMA = "monocular_geometry_scene_grouped_gate_c_v1_20260818"


def fixed_scene_split(scenes: list[str], validation_fraction: float, salt: str):
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{salt}:{scene}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = sorted(ordered[:validation_count])
    train = sorted(set(ordered) - set(validation))
    if set(train) & set(validation):
        raise RuntimeError("scene split overlap")
    return train, validation


def load_shards(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("input shard manifest is incomplete")
    tensors: dict[str, list[torch.Tensor]] = {}
    scenes = []
    sample_metadata = []
    for row in manifest["rows"]:
        payload = torch.load(root / row["shard"], map_location="cpu", weights_only=False)
        count = len(payload["metadata"]["samples"])
        if count != row["samples"]:
            raise RuntimeError(f"shard count mismatch: {row['shard']}")
        for name, value in payload["tensors"].items():
            tensors.setdefault(name, []).append(value)
        scenes.extend([row["scene"]] * count)
        for record in payload["metadata"]["samples"]:
            sample_metadata.append(
                {
                    "scene": row["scene"],
                    "episode_name": row["episode_name"],
                    **record,
                    "scale": payload["metadata"]["scale"],
                }
            )
    joined = {name: torch.cat(values) for name, values in tensors.items()}
    count = len(scenes)
    if any(value.shape[0] != count for value in joined.values()):
        raise RuntimeError("tensor sample axes disagree")
    return joined, np.asarray(scenes), sample_metadata, manifest


def _to_device(value: torch.Tensor, indices, device: str):
    return value[indices].to(device=device, dtype=torch.float32)


def _predict_epsilon(navdp, noisy, timesteps, goals, representation):
    output = torch.empty_like(noisy)
    for timestep in torch.unique(timesteps).tolist():
        mask = timesteps == int(timestep)
        output[mask] = navdp.predict_noise(
            noisy[mask],
            torch.tensor([int(timestep)], device=noisy.device, dtype=torch.long),
            goals[mask],
            representation[mask],
        )
    return output


def _student_outputs(adapter, navdp, tensors, indices, device):
    representation = adapter.forward_pooled_compact(
        _to_device(tensors["recent_specials"], indices, device),
        _to_device(tensors["pooled_current_patches"], indices, device),
        _to_device(tensors["pooled_depth_features"], indices, device),
        _to_device(tensors["scale_features"], indices, device),
    )
    noisy = _to_device(tensors["noisy"], indices, device)
    timesteps = tensors["timestep"][indices].to(device=device, dtype=torch.long)
    goals = _to_device(tensors["goal_embed"], indices, device)
    epsilon = _predict_epsilon(navdp, noisy, timesteps, goals, representation)
    candidates = _to_device(tensors["candidates"], indices, device)
    batch, count = candidates.shape[:2]
    critic = navdp.predict_critic(
        candidates.flatten(0, 1),
        representation.repeat_interleave(count, dim=0),
    ).reshape(batch, count)
    return representation, epsilon, critic


def _rankdata(value: torch.Tensor) -> torch.Tensor:
    order = value.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ordinal = torch.arange(value.shape[-1], device=value.device, dtype=torch.float32)
    ranks.scatter_(1, order, ordinal[None].expand_as(ranks))
    return ranks


def _metrics(representation, epsilon, critic, teacher, teacher_epsilon, teacher_critic):
    token_cosine_error = 1.0 - F.cosine_similarity(representation, teacher, dim=-1).mean()
    epsilon_cosine_error = 1.0 - F.cosine_similarity(
        epsilon.flatten(1), teacher_epsilon.flatten(1), dim=-1
    ).mean()
    student_rank = _rankdata(critic)
    teacher_rank = _rankdata(teacher_critic)
    student_centered = student_rank - student_rank.mean(dim=1, keepdim=True)
    teacher_centered = teacher_rank - teacher_rank.mean(dim=1, keepdim=True)
    spearman = (
        (student_centered * teacher_centered).sum(dim=1)
        / (
            student_centered.square().sum(dim=1).sqrt()
            * teacher_centered.square().sum(dim=1).sqrt()
        ).clamp_min(1e-8)
    ).mean()
    top1 = (critic.argmax(dim=1) == teacher_critic.argmax(dim=1)).float().mean()
    student_top2 = critic.topk(min(2, critic.shape[1]), dim=1).indices
    teacher_top2 = teacher_critic.topk(min(2, critic.shape[1]), dim=1).indices
    top2 = (
        (student_top2[:, :, None] == teacher_top2[:, None, :]).any(dim=-1).float().mean()
    )
    return {
        "token_cosine_error": token_cosine_error,
        "token_smooth_l1": F.smooth_l1_loss(representation, teacher),
        "epsilon_mse": F.mse_loss(epsilon, teacher_epsilon),
        "epsilon_cosine_error": epsilon_cosine_error,
        "critic_mse": F.mse_loss(critic, teacher_critic),
        "critic_spearman": spearman,
        "critic_top1_agreement": top1,
        "critic_top2_overlap": top2,
    }


def _representation_outputs(navdp, tensors, indices, representation_name, device):
    representation = _to_device(tensors[representation_name], indices, device)
    noisy = _to_device(tensors["noisy"], indices, device)
    timesteps = tensors["timestep"][indices].to(device=device, dtype=torch.long)
    goals = _to_device(tensors["goal_embed"], indices, device)
    epsilon = _predict_epsilon(navdp, noisy, timesteps, goals, representation)
    candidates = _to_device(tensors["candidates"], indices, device)
    batch, count = candidates.shape[:2]
    critic = navdp.predict_critic(
        candidates.flatten(0, 1), representation.repeat_interleave(count, dim=0)
    ).reshape(batch, count)
    return representation, epsilon, critic


@torch.no_grad()
def evaluate(name, adapter, navdp, tensors, indices, args):
    totals = {}
    count = 0
    if adapter is not None:
        adapter.eval()
    for start in range(0, len(indices), args.eval_batch_size):
        batch_indices = indices[start : start + args.eval_batch_size]
        if name == "adapter":
            outputs = _student_outputs(adapter, navdp, tensors, batch_indices, args.device)
        else:
            outputs = _representation_outputs(
                navdp, tensors, batch_indices, name, args.device
            )
        teacher = _to_device(tensors["teacher_tokens"], batch_indices, args.device)
        teacher_epsilon = _to_device(
            tensors["teacher_epsilon"], batch_indices, args.device
        )
        teacher_critic = _to_device(
            tensors["teacher_critic"], batch_indices, args.device
        )
        metrics = _metrics(*outputs, teacher, teacher_epsilon, teacher_critic)
        weight = len(batch_indices)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * weight
        count += weight
    if adapter is not None:
        adapter.train()
    return {key: value / count for key, value in totals.items()}


def gate_decision(metrics, validation_samples, validation_scenes, args):
    if validation_samples < args.min_gate_samples or validation_scenes < args.min_gate_scenes:
        return {
            "authorized": False,
            "reason": "underpowered_diagnostic_no_gate_decision",
        }
    zero = metrics["zero_depth_tokens"]

    def qualifies(candidate):
        return (
            candidate["token_cosine_error"] <= 0.80 * zero["token_cosine_error"]
            and candidate["epsilon_mse"] <= 0.90 * zero["epsilon_mse"]
            and candidate["critic_spearman"] >= zero["critic_spearman"] - 0.05
            and candidate["critic_top1_agreement"]
            >= zero["critic_top1_agreement"] - 0.05
            and (
                candidate["critic_spearman"] >= zero["critic_spearman"] + 0.02
                or candidate["critic_top1_agreement"]
                >= zero["critic_top1_agreement"] + 0.02
            )
        )

    raw_ok = qualifies(metrics["raw_depth_tokens"])
    adapter_ok = qualifies(metrics["adapter"])
    if not raw_ok and not adapter_ok:
        choice = "stop_no_rgb_only_geometry_path_qualifies"
    elif raw_ok and not adapter_ok:
        choice = "raw_lingbot_depth"
    elif adapter_ok and not raw_ok:
        choice = "latent_adapter"
    else:
        raw = metrics["raw_depth_tokens"]
        adapter = metrics["adapter"]
        choice = (
            "latent_adapter"
            if adapter["epsilon_mse"] <= 0.90 * raw["epsilon_mse"]
            and adapter["critic_spearman"] >= raw["critic_spearman"]
            else "raw_lingbot_depth_simpler_tie_break"
        )
    return {
        "authorized": raw_ok or adapter_ok,
        "raw_qualifies": raw_ok,
        "adapter_qualifies": adapter_ok,
        "choice": choice,
        "thresholds_frozen_before_gate_c": {
            "token_cosine_error_vs_zero_max_ratio": 0.80,
            "epsilon_mse_vs_zero_max_ratio": 0.90,
            "critic_noninferiority_margin": 0.05,
            "critic_minimum_improvement": 0.02,
            "adapter_vs_raw_epsilon_winner_ratio": 0.90,
        },
    }


def run(args):
    sys.path.insert(0, str(REPO_ROOT))
    from MemNavData.monocular_geometry_adapter import (
        GeometryTokenAdapter,
        geometry_distillation_losses,
    )
    from MemNavData.preflight_monocular_geometry_adapter import _load_navdp_agent

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tensors, scenes, sample_metadata, source_manifest = load_shards(args.shard_root)
    unique_scenes = sorted(set(scenes.tolist()))
    train_scenes, validation_scenes = fixed_scene_split(
        unique_scenes, args.validation_fraction, args.split_salt
    )
    train_indices = np.flatnonzero(np.isin(scenes, train_scenes)).tolist()
    validation_indices = np.flatnonzero(np.isin(scenes, validation_scenes)).tolist()
    if not train_indices or not validation_indices:
        raise RuntimeError("empty scene-grouped partition")

    navdp_agent = _load_navdp_agent(args.navdp_checkpoint, args.device)
    navdp = navdp_agent.navi_former
    checkpoint = torch.load(args.navdp_checkpoint, map_location="cpu", weights_only=False)
    adapter = GeometryTokenAdapter().to(args.device)
    query_key = adapter.initialize_queries_from_navdp(checkpoint)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    baseline = {
        name: evaluate(name, None, navdp, tensors, validation_indices, args)
        for name in ("zero_depth_tokens", "raw_depth_tokens")
    }
    initial = evaluate("adapter", adapter, navdp, tensors, validation_indices, args)
    history = [{"epoch": 0, "validation": initial}]
    best_score = float("inf")
    best_state = None
    generator = torch.Generator().manual_seed(args.seed + 1)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        epoch_totals = {}
        seen = 0
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            indices = [train_indices[position] for position in positions]
            optimizer.zero_grad(set_to_none=True)
            representation, epsilon, critic = _student_outputs(
                adapter, navdp, tensors, indices, args.device
            )
            losses = geometry_distillation_losses(
                representation,
                _to_device(tensors["teacher_tokens"], indices, args.device),
                student_epsilon=epsilon,
                teacher_epsilon=_to_device(
                    tensors["teacher_epsilon"], indices, args.device
                ),
                student_critic=critic,
                teacher_critic=_to_device(
                    tensors["teacher_critic"], indices, args.device
                ),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in losses.items():
                epoch_totals[key] = (
                    epoch_totals.get(key, 0.0)
                    + float(value.detach()) * len(indices)
                )
            seen += len(indices)
        validation = evaluate("adapter", adapter, navdp, tensors, validation_indices, args)
        score = (
            validation["token_cosine_error"]
            / max(baseline["zero_depth_tokens"]["token_cosine_error"], 1e-12)
            + validation["epsilon_mse"]
            / max(baseline["zero_depth_tokens"]["epsilon_mse"], 1e-12)
            + 0.25
            * validation["critic_mse"]
            / max(baseline["zero_depth_tokens"]["critic_mse"], 1e-12)
        )
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / seen for key, value in epoch_totals.items()},
                "validation": validation,
                "selection_score": score,
            }
        )
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
        print(
            f"epoch={epoch} train={epoch_totals['loss']/seen:.5f} "
            f"val_token={validation['token_cosine_error']:.5f} "
            f"val_eps={validation['epsilon_mse']:.7f} "
            f"val_rho={validation['critic_spearman']:.3f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    adapter.load_state_dict(best_state)
    final_metrics = {
        **baseline,
        "adapter": evaluate("adapter", adapter, navdp, tensors, validation_indices, args),
    }
    gate = gate_decision(
        final_metrics, len(validation_indices), len(validation_scenes), args
    )
    navdp_gradients = [
        name for name, parameter in navdp.named_parameters() if parameter.grad is not None
    ]
    if navdp_gradients:
        raise RuntimeError("frozen NavDP accumulated gradients")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "geometry_adapter_gate_c.pt"
    torch.save(
        {
            "architecture": "geometry_token_adapter_v1",
            "state_dict": best_state,
            "config": adapter.config.to_dict(),
            "teacher_query_key": query_key,
            "scene_grouped_gate_c_only": True,
        },
        checkpoint_path,
    )
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "not_closed_loop_or_sr": True,
        "source_manifest": str(args.shard_root / "manifest.json"),
        "source_scene_count": len(unique_scenes),
        "source_sample_count": len(scenes),
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "scene_overlap": sorted(set(train_scenes) & set(validation_scenes)),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "metrics": final_metrics,
        "initial_adapter_validation": initial,
        "gate_c": gate,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "teacher_query_key": query_key,
        "navdp_gradient_tensors": navdp_gradients,
        "elapsed_seconds": time.time() - started,
        "sample_strata_available": sorted(sample_metadata[0].keys()),
        "source_baselines": source_manifest.get("baselines"),
    }
    (args.output_dir / "gate_c_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"metrics": final_metrics, "gate_c": gate}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-salt", default="mdtec-gate-c-v1-20260818")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--min-gate-samples", type=int, default=32)
    parser.add_argument("--min-gate-scenes", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
