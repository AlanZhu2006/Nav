#!/usr/bin/env python3
"""Bounded, cached-only known-anchor POSITION probe, not a deployed localizer.

Selection uses existing train40 support labels once. Runtime model inputs are
only cached goal and historical-anchor patch features. Metrics here are NOT SR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from MemNavData.anchor_relation_decoder import (
    AnchorRelationDecoder, position_direction_loss,
)


SOURCE_TABLE_SHA = "85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127"
GEOMETRY_TABLE_SHA = "8e1b22901a7520e5bec5c6cb753eac9fab1342a19652d980f39264ecaa4bb24f"
PATCH_SHA = "f5561b23bf5a42e3f4e203f3ac2dd222acddc9d3cebba93e30caa3cb57d34a21"
SPLIT_SALT = "anchor-relation-position-probe-v0-20260906"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def split_scenes(scenes: list[str], validation_count: int = 8) -> tuple[set, set]:
    order = sorted(set(scenes), key=lambda s: hashlib.sha256(
        f"{SPLIT_SALT}:{s}".encode()).hexdigest())
    if not 0 < validation_count < len(order):
        raise ValueError("validation requires separate, nonempty scenes")
    return set(order[validation_count:]), set(order[:validation_count])


def select_pairs(geometry_rows: list[dict], source_rows: list[dict],
                 allowed_scenes: set[str]) -> tuple[list[dict], int]:
    """Select supported fixed anchors, retaining CEC-rejected supported pairs.

    Duplicate RGB pairs across sessions are collapsed before splitting or
    training. Teacher PnP/GCT results are reference predictions, never inputs.
    """
    by_key = {(r["session_id"], int(r["candidate_frame"])): r for r in source_rows}
    pairs = {}
    duplicates = 0
    for row in geometry_rows:
        if row["scene"] not in allowed_scenes:
            raise ValueError("geometry table contains a non-training scene")
        if int(row["label"]) != 1:
            continue
        source = by_key[(row["session_id"], int(row["candidate_frame"]))]
        if source["split_role"] != "train":
            raise ValueError("not a train-only source")
        if source["no_future"].lower() not in ("true", "1"):
            raise ValueError("noncausal source")
        if int(source["candidate_frame"]) >= int(source["decision_frame"]):
            raise ValueError("anchor must precede the decision")
        if (source["query_path"] != row["query_path"]
                or source["candidate_path"] != row["candidate_path"]):
            raise ValueError("teacher and image identities do not match")
        target = np.asarray(json.loads(row["target_relative_xy_m_center_json"]),
                            dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("finite anchor-local ground truth is required")
        hypothesis = json.loads(row["hypotheses_json"])[0]
        if int(hypothesis["anchor"]) != int(row["candidate_frame"]):
            raise ValueError("target coordinate anchor changed")
        if not np.allclose(target, hypothesis["target_relative_xy_m"], atol=1e-9):
            raise ValueError("nested target coordinate mismatch")
        pnp = hypothesis.get("pnp_lightglue", {})
        pnp_xy = pnp.get("predicted_relative_xy_m") if pnp.get("status") == "ok" else None
        pair_id = source["query_content_sha256"] + ":" + source["candidate_rgb_content_sha256"]
        record = {
            "pair_id": pair_id, "scene": row["scene"],
            "session_id": row["session_id"], "source_sessions": [row["session_id"]],
            "query_relative_path": source["query_relative_path"],
            "candidate_relative_path": source["candidate_relative_path"],
            "query_sha256": source["query_content_sha256"],
            "candidate_sha256": source["candidate_rgb_content_sha256"],
            "candidate_frame": int(row["candidate_frame"]),
            "decision_frame": int(source["decision_frame"]),
            "target_xy_m": target.tolist(),
            "gct_xy_m": json.loads(row["predicted_relative_xy_m_center_json"]),
            "pnp_xy_m": pnp_xy,
            "support_reporting_only": float(row["teacher_covis"]),
        }
        if pair_id in pairs:
            previous = pairs[pair_id]
            if (previous["scene"] != record["scene"]
                    or not np.allclose(previous["target_xy_m"], target, atol=1e-7)):
                raise ValueError("duplicate visual pair has inconsistent geometry")
            previous["source_sessions"].append(row["session_id"])
            duplicates += 1
        else:
            pairs[pair_id] = record
    return sorted(pairs.values(), key=lambda p: p["pair_id"]), duplicates


def finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def prediction_metrics(prediction: np.ndarray, target: np.ndarray,
                       scenes: list[str]) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction).all(-1)
    error = np.linalg.norm(prediction - target, axis=-1)
    target_range = np.linalg.norm(target, axis=-1)
    predicted_range = np.linalg.norm(prediction, axis=-1)
    direction_population = target_range >= .25
    direction_valid = direction_population & valid & (predicted_range > 1e-6)
    angles = np.full(len(target), np.nan)
    angles[direction_valid] = np.degrees(np.arccos(np.clip(
        (prediction[direction_valid] * target[direction_valid]).sum(-1)
        / (predicted_range[direction_valid] * target_range[direction_valid]), -1, 1)))
    scene_errors = [np.mean(error[(np.asarray(scenes) == s) & valid])
                    for s in sorted(set(scenes)) if ((np.asarray(scenes) == s) & valid).any()]
    return {
        "pairs": len(target), "valid_predictions": int(valid.sum()),
        "position_error_m": finite_summary(error),
        "scene_macro_position_error_m": float(np.mean(scene_errors)) if scene_errors else None,
        "position_within_025m": int((valid & (error <= .25)).sum()),
        "position_within_05m": int((valid & (error <= .5)).sum()),
        "anchor_local_direction_eligible": int(direction_population.sum()),
        "anchor_local_direction_valid": int(direction_valid.sum()),
        "anchor_local_direction_error_deg": finite_summary(angles),
        "anchor_local_direction_within_30deg": int((angles <= 30).sum()),
        "anchor_local_direction_over_90deg": int((angles > 90).sum()),
        "navigation_SR": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--overfit-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.overfit_pairs < 0:
        raise ValueError("invalid training dimensions")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base = Path(".diagnostics/certificate_distilled_compass_20260813")
    geometry_path = Path(".diagnostics/train40_certificate_reuse_20260814/geometry_rows.csv")
    source_path = base / "static_top8_480_lightglue_open_set_rows.csv"
    patch_path = base / "cdec_patch_cache_fixedbatch_v2.npz"
    sources = [(source_path, SOURCE_TABLE_SHA), (geometry_path, GEOMETRY_TABLE_SHA),
               (patch_path, PATCH_SHA)]
    for path, expected in sources:
        if sha256(path) != expected:
            raise ValueError(f"source identity changed: {path}")
    split = json.loads(Path("MemNavData/router_multiscene_split_20260805.json").read_text())
    train_scenes, validation_scenes = split_scenes(split["train"])
    pairs, duplicate_count = select_pairs(read_csv(geometry_path), read_csv(source_path),
                                          set(split["train"]))
    for p in pairs:
        p["split"] = "train" if p["scene"] in train_scenes else "validation"
        for kind in ("query", "candidate"):
            path = base / "certificate_images" / p[f"{kind}_relative_path"]
            if sha256(path) != p[f"{kind}_sha256"]:
                raise ValueError(f"RGB hash mismatch: {path}")
    train_indices = [i for i, p in enumerate(pairs) if p["split"] == "train"]
    validation_indices = [i for i, p in enumerate(pairs) if p["split"] == "validation"]
    if args.overfit_pairs:
        train_indices = train_indices[:args.overfit_pairs]
    manifest = {
        "schema_version": "known_anchor_visual_position_probe_v0",
        "deployment_approved": False,
        "scope": "train40 cached-only positive-anchor diagnostic; no geometry input; not SR",
        "source_sha256": {str(p): s for p, s in sources},
        "deduplicated_pairs": len(pairs), "duplicates_removed": duplicate_count,
        "train_scene_ids": sorted(train_scenes), "validation_scene_ids": sorted(validation_scenes),
        "train_indices": train_indices, "validation_indices": validation_indices,
        "pairs": pairs,
        "model_inputs": ["frozen_goal_patch_tokens", "frozen_anchor_patch_tokens"],
        "target": "audited anchor-base planar [forward, lateral] metres",
        "forbidden_inputs": ["role", "scene", "teacher_errors", "proof_statistics",
                             "depth_scale_raw", "GT_pose", "DINO_similarity"],
        "parameters": vars(args) | {"out_dir": str(args.out_dir)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with np.load(patch_path) as cache:
        if cache["rows_csv_sha256"].item() != SOURCE_TABLE_SHA:
            raise ValueError("patch cache image universe changed")
        path_index = {p: i for i, p in enumerate(cache["relative_paths"].tolist())}
        tokens = cache["tokens"]
        goal = np.stack([tokens[path_index[p["query_relative_path"]]] for p in pairs])
        anchor = np.stack([tokens[path_index[p["candidate_relative_path"]]] for p in pairs])
    device = torch.device(args.device)
    goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
    anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device)
    target = torch.tensor([p["target_xy_m"] for p in pairs], dtype=torch.float32, device=device)
    model = AnchorRelationDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        indices = rng.choice(train_indices, size=min(args.batch_size, len(train_indices)), replace=False)
        indices = torch.as_tensor(indices, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(goal[indices], anchor[indices])
        loss = position_direction_loss(output, target[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.detach()),
                      "gradient_norm": float(nnorm), "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            print(json.dumps(record), flush=True)
    model.eval()

    def predict(indices, *, shuffle_goal=False):
        outputs = []
        ordered_goals = np.asarray(indices)
        if shuffle_goal and len(indices) > 1:
            ordered_goals = np.roll(ordered_goals, 1)
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                idx = indices[start:start + args.batch_size]
                qidx = ordered_goals[start:start + args.batch_size]
                outputs.append(model(goal[qidx], anchor[idx]).cpu().numpy())
        return np.concatenate(outputs)

    target_numpy = target.cpu().numpy()
    mean = target_numpy[train_indices].mean(0)
    reports = {}
    prediction_rows = []
    for name, indices in (("train", train_indices), ("validation", validation_indices)):
        y = target_numpy[indices]
        scene_ids = [pairs[i]["scene"] for i in indices]
        predictions = {
            "learned_visual_position": predict(indices),
            "anchor_copy": np.zeros_like(y),
            "training_mean_position": np.broadcast_to(mean, y.shape),
            "existing_frozen_gct_query": np.asarray([pairs[i]["gct_xy_m"] for i in indices]),
            "existing_finite_pnp": np.asarray([
                pairs[i]["pnp_xy_m"] if pairs[i]["pnp_xy_m"] is not None else [np.nan, np.nan]
                for i in indices]),
            "learned_shuffled_goal_diagnostic": predict(indices, shuffle_goal=True),
        }
        reports[name] = {key: prediction_metrics(value, y, scene_ids)
                         for key, value in predictions.items()}
        for local, index in enumerate(indices):
            for arm, value in predictions.items():
                prediction_rows.append({
                    "pair_id": pairs[index]["pair_id"], "scene": pairs[index]["scene"],
                    "split": name, "arm": arm,
                    "target_forward_m": float(y[local, 0]), "target_lateral_m": float(y[local, 1]),
                    "predicted_forward_m": float(value[local, 0]) if np.isfinite(value[local, 0]) else "",
                    "predicted_lateral_m": float(value[local, 1]) if np.isfinite(value[local, 1]) else "",
                })
    report = {
        "manifest_sha256": sha256(args.out_dir / "manifest.json"),
        "deployment_approved": False, "navigation_SR": None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": reports,
        "interpretation": "input-ablation probe; not full geometry-conditioned decoder, open-set evidence, or closed-loop SR",
    }
    torch.save({"model": model.state_dict(), "deployment_approved": False,
                "manifest_sha256": report["manifest_sha256"]}, args.out_dir / "probe.pt")
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for filename, rows in (("training_curve.csv", history), ("predictions.csv", prediction_rows)):
        with (args.out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"completed": True, "report": str(args.out_dir / "report.json"),
                      "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
