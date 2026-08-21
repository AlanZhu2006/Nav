#!/usr/bin/env python3
"""Fit the frozen deployment ensemble after scene-OOF model selection.

Each member is trained on 3/4 of the train40 scenes and calibrated on the
remaining 1/4 that the same member never sees during fitting.  The resulting
threshold remains attached to that exact checkpoint.  This script produces no
held-out performance estimate; the next valid measurement is a fresh
scene-disjoint closed-loop comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from MemNavData.summarize_pi3x_multiview_shadow import choose_threshold
from MemNavData.train_pi3x_spatial_reliability_crossfit_oof import (
    Pi3XSpatialReliabilityHead,
    _fit,
    _fixed_proposal_picks,
    _load_spatial,
    _predict,
)
from MemNavData.train_pi3x_viewtoken_reliability_oof import (
    _atomic_json,
    _load,
    _sha256,
)


def deployment_scene_splits(scenes: list[str], members: int) -> list[tuple[set[str], set[str]]]:
    ordered = np.asarray(sorted(scenes))
    splitter = KFold(n_splits=members, shuffle=True, random_state=101)
    output = []
    for fit_indices, calibration_indices in splitter.split(ordered):
        fit = set(ordered[fit_indices].tolist())
        calibration = set(ordered[calibration_indices].tolist())
        if fit & calibration or fit | calibration != set(ordered):
            raise RuntimeError("deployment fit/calibration split is invalid")
        output.append((fit, calibration))
    return output


def run(args: argparse.Namespace) -> dict:
    global_arrays, rows = _load(args)
    spatial_arrays = _load_spatial(
        args.spatial_npz, args.expected_spatial_sha256, len(rows)
    )
    scenes = sorted({row["scene"] for row in rows})
    if args.members != args.consensus_denominator:
        raise ValueError("members must equal the frozen consensus denominator")
    if not 1 <= args.consensus_numerator <= args.members:
        raise ValueError("invalid consensus numerator")
    if args.output_manifest.exists() or args.checkpoint_dir.exists():
        raise ValueError("deployment output already exists")
    args.checkpoint_dir.mkdir(parents=True)
    model_reports = []
    for member, (fit_scenes, calibration_scenes) in enumerate(
        deployment_scene_splits(scenes, args.members)
    ):
        model = _fit(
            global_arrays,
            spatial_arrays,
            rows,
            fit_scenes,
            args,
            seed=args.seed + member,
        )
        calibration_scores = _predict(
            model,
            global_arrays,
            spatial_arrays,
            rows,
            calibration_scenes,
            args,
        )
        calibration_picks = _fixed_proposal_picks(
            rows, calibration_scores, calibration_scenes
        )
        threshold, calibration = choose_threshold(
            calibration_picks,
            minimum_precision=args.minimum_precision,
            maximum_fpr=args.maximum_fpr,
            correctness_key="selected_navigation_action_label",
        )
        checkpoint = args.checkpoint_dir / f"member_{member}.pt"
        torch.save({
            "schema_version": 1,
            "model_name": "pi3x_spatial_reliability_head_v1",
            "member": member,
            "fit_scenes": sorted(fit_scenes),
            "calibration_scenes": sorted(calibration_scenes),
            "threshold": float(threshold),
            "model_config": {
                "descriptor_dim": int(global_arrays["view_descriptors"].shape[-1]),
                "model_dim": args.model_dim,
                "layers": args.layers,
                "heads": args.heads,
            },
            "state_dict": model.cpu().state_dict(),
        }, checkpoint)
        report = {
            "member": member,
            "fit_scenes": sorted(fit_scenes),
            "calibration_scenes": sorted(calibration_scenes),
            "threshold": float(threshold),
            "calibration_reporting_only": calibration,
            "checkpoint": str(Path("checkpoints") / checkpoint.name),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        model_reports.append(report)
        print(json.dumps({
            "member": member,
            "threshold": threshold,
            "calibration": calibration,
        }, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probe = Pi3XSpatialReliabilityHead(
        int(global_arrays["view_descriptors"].shape[-1]),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
    )
    manifest = {
        "schema_version": 1,
        "status": "frozen_for_fresh_scene_disjoint_closed_loop_not_evaluated",
        "method": (
            "dino_top8_pi3x_b16_overlap_proposal_spatial_learned_proof_"
            "scale_free_bearing_native_fallback"
        ),
        "training_population": {
            "rows": len(rows),
            "sessions": len({row["session_id"] for row in rows}),
            "scenes": len(scenes),
            "scene_ids": scenes,
        },
        "model": {
            "name": "pi3x_spatial_reliability_head_v1",
            "parameters_per_member": sum(
                parameter.numel() for parameter in probe.parameters()
            ),
            "descriptor_dim": int(global_arrays["view_descriptors"].shape[-1]),
            "model_dim": args.model_dim,
            "layers": args.layers,
            "heads": args.heads,
            "epochs": args.epochs,
            "support_weight": args.support_weight,
            "pi3x_frozen": True,
        },
        "proposal": "raw_pi3x_overlap_top1",
        "authorization": {
            "member_thresholds_bound_to_checkpoints": True,
            "consensus_numerator": args.consensus_numerator,
            "consensus_denominator": args.consensus_denominator,
            "minimum_calibration_precision": args.minimum_precision,
            "maximum_calibration_fpr": args.maximum_fpr,
            "rejection_behavior": "exact_native_navdp_fallback",
        },
        "members": model_reports,
        "inputs": {
            "rows_csv_sha256": _sha256(args.rows_csv),
            "shadow_jsonl_sha256": _sha256(args.shadow_jsonl),
            "descriptors_npz_sha256": _sha256(args.descriptors_npz),
            "spatial_npz_sha256": _sha256(args.spatial_npz),
        },
        "prohibitions": [
            "no development blind Fresh160 Attempt7 or external held-out fitting",
            "no post-freeze threshold tuning",
            "offline result is not navigation SR",
        ],
    }
    _atomic_json(args.output_manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--shadow-jsonl", type=Path, required=True)
    parser.add_argument("--descriptors-npz", type=Path, required=True)
    parser.add_argument("--spatial-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=3840)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--expected-spatial-sha256")
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--consensus-numerator", type=int, default=2)
    parser.add_argument("--consensus-denominator", type=int, default=4)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--maximum-fpr", type=float, default=0.0275)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-sessions", type=int, default=12)
    parser.add_argument("--inference-batch-rows", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "members": len(result["members"]),
        "consensus": result["authorization"],
    }, sort_keys=True))
