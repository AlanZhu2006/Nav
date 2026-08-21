#!/usr/bin/env python3
"""Read-only stratified audit for a completed MDTEC Gate C run.

The training receipt intentionally decides Gate C from frozen aggregate
metrics.  This script is a post-hoc diagnostic: it reloads that selected
adapter and the exact scene-disjoint validation samples, then reports the same
functional metrics by causal-scale quality together with selected-candidate
endpoint/heading disagreement.  Its output cannot authorize Gate C and must
not be used to reselect a checkpoint or tune a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]


def scale_stratum(scale: dict) -> str:
    if not bool(scale.get("scale_valid", False)):
        return "scale_invalid"
    if bool(scale.get("scale_clamped", False)):
        return "scale_valid_clamped"
    return "scale_valid_unclamped"


def selected_candidate_disagreement(
    candidates: torch.Tensor,
    student_critic: torch.Tensor,
    teacher_critic: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare the student/teacher selected action sequences.

    NavDP critic inputs are action increments.  The deployed trajectory is the
    cumulative sum divided by four, matching ``policy_network.py``.  Heading is
    the bearing of the final planar endpoint, not the third action channel.
    """

    if candidates.ndim != 4 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape [B,K,T,3]")
    if student_critic.shape != teacher_critic.shape:
        raise ValueError("student/teacher critic shapes disagree")
    if tuple(student_critic.shape) != tuple(candidates.shape[:2]):
        raise ValueError("critic and candidate axes disagree")
    batch = torch.arange(candidates.shape[0], device=candidates.device)
    student = candidates[batch, student_critic.argmax(dim=1)]
    teacher = candidates[batch, teacher_critic.argmax(dim=1)]
    student_endpoint = torch.cumsum(student / 4.0, dim=1)[:, -1, :2]
    teacher_endpoint = torch.cumsum(teacher / 4.0, dim=1)[:, -1, :2]
    student_heading = torch.atan2(student_endpoint[:, 1], student_endpoint[:, 0])
    teacher_heading = torch.atan2(teacher_endpoint[:, 1], teacher_endpoint[:, 0])
    heading_delta = torch.atan2(
        torch.sin(student_heading - teacher_heading),
        torch.cos(student_heading - teacher_heading),
    ).abs()
    return {
        "critic_top1_agreement": (
            student_critic.argmax(dim=1) == teacher_critic.argmax(dim=1)
        ).float(),
        "selected_endpoint_l2_m": (student_endpoint - teacher_endpoint).norm(dim=1),
        "selected_heading_abs_error_deg": torch.rad2deg(heading_delta),
    }


def _rankdata(value: torch.Tensor) -> torch.Tensor:
    order = value.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ordinal = torch.arange(value.shape[-1], device=value.device, dtype=torch.float32)
    ranks.scatter_(1, order, ordinal[None].expand_as(ranks))
    return ranks


def _sample_metrics(
    representation: torch.Tensor,
    epsilon: torch.Tensor,
    critic: torch.Tensor,
    teacher: torch.Tensor,
    teacher_epsilon: torch.Tensor,
    teacher_critic: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, torch.Tensor]:
    student_rank = _rankdata(critic)
    teacher_rank = _rankdata(teacher_critic)
    student_centered = student_rank - student_rank.mean(dim=1, keepdim=True)
    teacher_centered = teacher_rank - teacher_rank.mean(dim=1, keepdim=True)
    spearman = (student_centered * teacher_centered).sum(dim=1) / (
        student_centered.square().sum(dim=1).sqrt()
        * teacher_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    result = {
        "token_cosine_error": 1.0 - F.cosine_similarity(representation, teacher, dim=-1).mean(dim=1),
        "epsilon_mse": (epsilon - teacher_epsilon).square().flatten(1).mean(dim=1),
        "epsilon_cosine_error": 1.0 - F.cosine_similarity(
            epsilon.flatten(1), teacher_epsilon.flatten(1), dim=-1
        ),
        "critic_mse": (critic - teacher_critic).square().mean(dim=1),
        "critic_spearman": spearman,
    }
    result.update(
        selected_candidate_disagreement(candidates, critic, teacher_critic)
    )
    return result


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


def _outputs(navdp, adapter, tensors, indices, path: str, device: str):
    def take(name, dtype=torch.float32):
        return tensors[name][indices].to(device=device, dtype=dtype)

    if path == "adapter":
        representation = adapter.forward_pooled_compact(
            take("recent_specials"),
            take("pooled_current_patches"),
            take("pooled_depth_features"),
            take("scale_features"),
        )
    else:
        representation = take(path)
    noisy = take("noisy")
    timesteps = take("timestep", torch.long)
    goals = take("goal_embed")
    epsilon = _predict_epsilon(navdp, noisy, timesteps, goals, representation)
    candidates = take("candidates")
    batch, count = candidates.shape[:2]
    critic = navdp.predict_critic(
        candidates.flatten(0, 1), representation.repeat_interleave(count, dim=0)
    ).reshape(batch, count)
    return representation, epsilon, critic, candidates


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def run(args) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from MemNavData.monocular_geometry_adapter import (
        GeometryAdapterConfig,
        GeometryTokenAdapter,
    )
    from MemNavData.preflight_monocular_geometry_adapter import _load_navdp_agent
    from MemNavData.train_monocular_geometry_adapter import fixed_scene_split, load_shards

    receipt = json.loads(args.receipt.read_text())
    tensors, scenes, metadata, manifest = load_shards(args.shard_root)
    train_scenes, validation_scenes = fixed_scene_split(
        sorted(set(scenes.tolist())), args.validation_fraction, args.split_salt
    )
    if receipt["train_scenes"] != train_scenes or receipt["validation_scenes"] != validation_scenes:
        raise RuntimeError("receipt and independently reconstructed scene split disagree")
    validation_indices = np.flatnonzero(np.isin(scenes, validation_scenes)).tolist()

    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("scene_grouped_gate_c_only") is not True:
        raise RuntimeError("adapter checkpoint is not a Gate C artifact")
    adapter = GeometryTokenAdapter(GeometryAdapterConfig(**checkpoint["config"])).to(args.device)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval()
    navdp_agent = _load_navdp_agent(args.navdp_checkpoint, args.device)
    navdp = navdp_agent.navi_former

    per_path: dict[str, dict[str, list[float]]] = {}
    strata = {index: scale_stratum(metadata[index]["scale"]) for index in validation_indices}
    with torch.no_grad():
        for start in range(0, len(validation_indices), args.batch_size):
            batch_indices = validation_indices[start : start + args.batch_size]
            teacher = tensors["teacher_tokens"][batch_indices].to(args.device, torch.float32)
            teacher_epsilon = tensors["teacher_epsilon"][batch_indices].to(args.device, torch.float32)
            teacher_critic = tensors["teacher_critic"][batch_indices].to(args.device, torch.float32)
            for path in ("zero_depth_tokens", "raw_depth_tokens", "adapter"):
                representation, epsilon, critic, candidates = _outputs(
                    navdp, adapter, tensors, batch_indices, path, args.device
                )
                metrics = _sample_metrics(
                    representation,
                    epsilon,
                    critic,
                    teacher,
                    teacher_epsilon,
                    teacher_critic,
                    candidates,
                )
                for local, global_index in enumerate(batch_indices):
                    stratum = strata[global_index]
                    for group in ("all", stratum):
                        bucket = per_path.setdefault(path, {}).setdefault(group, {})
                        bucket.setdefault("sample_count", []).append(1.0)
                        for name, value in metrics.items():
                            bucket.setdefault(name, []).append(float(value[local].cpu()))

    summary = {}
    for path, groups in per_path.items():
        summary[path] = {}
        for group, values in groups.items():
            summary[path][group] = {
                "sample_count": len(values["sample_count"]),
                **{name: _mean(series) for name, series in values.items() if name != "sample_count"},
            }
    receipt_names = {
        "zero_depth_tokens": "zero_depth_tokens",
        "raw_depth_tokens": "raw_depth_tokens",
        "adapter": "adapter",
    }
    comparable = (
        "token_cosine_error",
        "epsilon_mse",
        "critic_mse",
        "critic_spearman",
        "critic_top1_agreement",
    )
    aggregate_reproduction = {}
    for path, receipt_name in receipt_names.items():
        expected = receipt["metrics"][receipt_name]
        observed = summary[path]["all"]
        differences = {
            name: abs(float(observed[name]) - float(expected[name]))
            for name in comparable
        }
        aggregate_reproduction[path] = {
            "absolute_differences": differences,
            "within_cross_process_tolerance": all(
                value <= args.aggregate_tolerance for value in differences.values()
            ),
        }
    result = {
        "schema": "monocular_geometry_gate_c_stratified_audit_v1_20260818",
        "status": "complete",
        "posthoc_diagnostic_not_authorization": True,
        "checkpoint_reselection_permitted": False,
        "threshold_tuning_permitted": False,
        "source_manifest": str(args.shard_root / "manifest.json"),
        "source_gate_c_receipt": str(args.receipt),
        "validation_scenes": validation_scenes,
        "validation_sample_count": len(validation_indices),
        "scale_evidence_contract": manifest.get("scale_evidence_contract"),
        "aggregate_reproduction_tolerance": args.aggregate_tolerance,
        "aggregate_reproduction": aggregate_reproduction,
        "aggregate_reproduction_all_within_tolerance": all(
            row["within_cross_process_tolerance"]
            for row in aggregate_reproduction.values()
        ),
        "paths": summary,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate-tolerance", type=float, default=5e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-salt", default="mdtec-gate-c-v1-20260818")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
