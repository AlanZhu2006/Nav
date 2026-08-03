#!/usr/bin/env python
"""Paired, fixed-sample evaluation for MemNav checkpoints.

This is deliberately different from reading training loss curves:

* every checkpoint sees the same unique episodes, current frame ``k``, images,
  labels, diffusion noise, and diffusion timesteps;
* the policy is in ``eval()`` mode, so retrieval selects the live anchor and the
  decoder uses the predicted gate (no positive-anchor or gate teacher forcing);
* action loss is averaged over repeated, paired diffusion trials after the
  expensive frozen front-end is encoded once;
* results are written per sample so leg depth, recall depth, and long tails can
  be inspected without averaging unrelated cases together.

The manifest contains six balanced groups: novel Goal-A, novel covis goals,
and shallow/deep revisit cases for both leg 2 and leg 3.  At most one sample is
selected from each trajectory, so paired uncertainty is not inflated by many
correlated goals from one episode.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy
from scripts.train.configs.memnav import memnav_exp_cfg


GROUPS = (
    "novel_goala",
    "novel_covis",
    "leg2_shallow",
    "leg2_deep",
    "leg3_shallow",
    "leg3_deep",
)
DEEP_GAP = 200
LOWER_IS_BETTER = {
    "action_loss_pred",
    "action_loss_mid",
    "action_loss_oracle",
    "gate_bce",
    "retrieval_loss",
    "aux_loss",
    "pos_err_m",
    "pos_dir_err_deg",
    "rot_err_deg",
    "positive_frame_distance",
}
UPPER_IS_BETTER = {"gate_correct", "retrieval_hit"}
DESCRIPTIVE = {"gate_prob", "oracle_better_frac", "pred_oracle_rms"}


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/absolute/path/memnav.ckpt")
    label, path = value.split("=", 1)
    if not label or not os.path.isabs(path):
        raise argparse.ArgumentTypeError("checkpoint label must be nonempty and path absolute")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label, path


def _group_for(sample: dict, k: int, pos: np.ndarray, null_pos: bool) -> tuple[str, int | None]:
    if null_pos:
        return ("novel_covis" if sample["has_covis"] else "novel_goala"), None
    recent_positive = int(np.flatnonzero(pos)[-1])
    gap = int(k - recent_positive)
    leg = "leg2" if int(sample["goal_j"]) == 0 else "leg3"
    depth = "deep" if gap >= DEEP_GAP else "shallow"
    return f"{leg}_{depth}", gap


def build_manifest(dataset: MemNav_Dataset, per_group: int, seed: int) -> list[dict]:
    """Select a deterministic, episode-unique balanced manifest without image I/O."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[dict]] = {name: [] for name in GROUPS}
    used_trajectories: set[int] = set()

    for sample_index in rng.permutation(len(dataset.samples)).tolist():
        sample = dataset.samples[sample_index]
        traj_idx = int(sample["traj_idx"])
        if traj_idx in used_trajectories:
            continue
        k_lo, k_hi = int(sample["k_lo"]), int(sample["k_hi"])
        if k_hi < k_lo:
            continue
        candidates = np.arange(k_lo, k_hi + 1, dtype=np.int64)
        if candidates.size > 12:
            candidates = rng.choice(candidates, size=12, replace=False)
        else:
            candidates = rng.permutation(candidates)

        chosen = None
        for k_raw in candidates.tolist():
            k = int(k_raw)
            pos, neg, _cand, null_pos = dataset._build_label(sample, k)
            group, gt_gap = _group_for(sample, k, pos, null_pos)
            if len(buckets[group]) >= per_group:
                continue
            # Ranking metrics require actual positive/negative contrast.
            if not null_pos and (not pos.any() or not neg.any()):
                continue
            chosen = {
                "record_id": f"{group}:{traj_idx}:{sample_index}:{k}",
                "sample_index": int(sample_index),
                "traj_idx": traj_idx,
                "trajectory": os.path.relpath(dataset.trajectory_dirs[traj_idx], dataset.trajectory_dirs[0]),
                "group": group,
                "goal_j": int(sample["goal_j"]),
                "k": k,
                "goal_step": int(sample["goal_step"]),
                "gt_recent_positive_gap": gt_gap,
                "n_pos": int(pos.sum()),
                "n_neg": int(neg.sum()),
            }
            break
        if chosen is None:
            continue
        buckets[chosen["group"]].append(chosen)
        used_trajectories.add(traj_idx)
        if all(len(buckets[name]) >= per_group for name in GROUPS):
            break

    counts = {name: len(rows) for name, rows in buckets.items()}
    if any(counts[name] != per_group for name in GROUPS):
        raise RuntimeError(f"could not fill balanced manifest: requested={per_group}, found={counts}")

    # Interleave groups so a partial run remains balanced.
    return [buckets[name][i] for i in range(per_group) for name in GROUPS]


def materialize_batch(dataset: MemNav_Dataset, records: list[dict]) -> dict:
    items = []
    for record in records:
        # random_digit=False in the production config, so k is the only randint.
        with patch("numpy.random.randint", return_value=int(record["k"])):
            item = dataset[int(record["sample_index"])]
        if int(item["cur_step"]) != int(record["k"]):
            raise RuntimeError(f"fixed k did not hold for {record['record_id']}")
        actual_revisit = bool(item["is_revisit"].item())
        expected_revisit = not record["group"].startswith("novel")
        if actual_revisit != expected_revisit:
            raise RuntimeError(f"revisit label changed for {record['record_id']}")
        items.append(item)
    return memnav_collate_fn(items)


def load_heads(model: MemNavPolicy, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    fusion_key = "core.gate_fusion_residual"
    # Checkpoints predating residual fusion have complementary semantics. Reset
    # explicitly so evaluation order cannot leak a newer checkpoint's mode into
    # a subsequent legacy load on the reused frozen backbone.
    if fusion_key not in state:
        model.core.gate_fusion_residual.zero_()
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint tensors: {incompatible.unexpected_keys[:10]}")
    probes = (
        "core.retrieval.gate_a",
        "core.retrieval.gate_b",
        "core.revisit_merge.revisit_head.weight",
        "core.decoder.layers.0.self_attn.in_proj_weight",
    )
    current = model.state_dict()
    mismatched = [key for key in probes if key not in state or not torch.equal(current[key].cpu(), state[key])]
    if mismatched:
        raise RuntimeError(f"checkpoint probes were not loaded exactly: {mismatched}")
    print(
        f"[fixed-eval] loaded {path}: tensors={len(state)} "
        f"missing_frozen={len(incompatible.missing_keys)} "
        f"fusion={'residual' if model.core.gate_fusion_residual.item() else 'complementary'}",
        flush=True,
    )


def _finite_float(value: torch.Tensor | float | int) -> float:
    out = float(value.item() if torch.is_tensor(value) else value)
    if not math.isfinite(out):
        raise RuntimeError(f"non-finite metric: {out}")
    return out


@torch.inference_mode()
def evaluate_batch(
    model: MemNavPolicy,
    batch: dict,
    records: list[dict],
    *,
    trials: int,
    seed: int,
    batch_index: int,
) -> list[dict]:
    core = model.core
    dev = next(model.parameters()).device
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with amp:
        encoded = core.encode_memory(batch)
        current = core.build_current_state(encoded["current"], encoded["depth_feat"])
        revisit, aux_pose, r_rel = core.build_revisit(
            encoded["cur_pose"], encoded["goal_pose"], encoded["metric_scale"]
        )
        novel = core.novel(
            batch["batch_window_images"][:, -1].to(dev),
            batch["batch_goal_image"].to(dev),
        )

    labels = batch["batch_labels"].to(dev)
    target_gate = batch["batch_is_revisit"].to(dev)
    predicted_gate = encoded["revisit_gate"].float()
    mid_gate = predicted_gate + 0.5 * (target_gate - predicted_gate)
    oracle_gate = target_gate
    bsz = labels.shape[0]
    sums = {name: torch.zeros(bsz, device=dev) for name in (
        "pred", "mid", "oracle", "rms", "oracle_better"
    )}

    for trial in range(trials):
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed + 1_000_003 * batch_index + trial))
        noise = torch.randn(labels.shape, generator=generator, device=dev, dtype=labels.dtype)
        timesteps = torch.randint(
            0,
            core.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            generator=generator,
            device=dev,
        )
        noisy = core.noise_scheduler.add_noise(labels, noise, timesteps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = core.predict_noise(noisy, timesteps, current, revisit, novel, predicted_gate)
            mid = core.predict_noise(noisy, timesteps, current, revisit, novel, mid_gate)
            oracle = core.predict_noise(noisy, timesteps, current, revisit, novel, oracle_gate)
        lp = (pred.float() - noise).square().mean(dim=(-2, -1))
        lm = (mid.float() - noise).square().mean(dim=(-2, -1))
        lo = (oracle.float() - noise).square().mean(dim=(-2, -1))
        sums["pred"] += lp
        sums["mid"] += lm
        sums["oracle"] += lo
        sums["rms"] += (pred.float() - oracle.float()).square().mean(dim=(-2, -1)).sqrt()
        sums["oracle_better"] += (lo < lp).float()
    for key in sums:
        sums[key] /= trials

    logits = encoded["ret_logits"].float()
    gate_logit = encoded["gate_logit"].float()
    pos = batch["batch_pos_mask"].to(dev).bool()
    neg = batch["batch_neg_mask"].to(dev).bool()
    floor = torch.finfo(logits.dtype).min
    valid_rank = pos.any(-1) & neg.any(-1)
    per_rank = torch.zeros(bsz, device=dev)
    per_rank[valid_rank] = (
        logits.masked_fill(~(pos | neg), floor).logsumexp(-1)
        - logits.masked_fill(~pos, floor).logsumexp(-1)
    )[valid_rank]
    per_gate_bce = F.binary_cross_entropy_with_logits(gate_logit, target_gate, reduction="none")
    match_idx = encoded["match_idx"].long()
    hit = pos.gather(1, match_idx[:, None]).squeeze(1).float()

    gt_xy = batch["batch_goal_rel_pose"][:, :2].to(dev).float()
    pred_xy = aux_pose.float()
    pos_err = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    pos_cos = (
        (pred_xy * gt_xy).sum(-1)
        / (torch.linalg.norm(pred_xy, dim=-1) * torch.linalg.norm(gt_xy, dim=-1) + 1e-9)
    ).clamp(-1, 1)
    pos_dir = torch.rad2deg(torch.arccos(pos_cos))
    aux_mse = (pred_xy - gt_xy).square().mean(-1)

    c_rot = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=dev,
        dtype=r_rel.dtype,
    )
    r_conv = c_rot @ r_rel @ c_rot.transpose(-1, -2)
    gt_rot = batch["batch_goal_rel_rotation"].to(dev, r_rel.dtype)
    rot_cos = (((r_conv * gt_rot).sum(dim=(-2, -1)) - 1) / 2).clamp(-1, 1)
    rot_err = torch.rad2deg(torch.arccos(rot_cos)).float()

    rows = []
    for i, record in enumerate(records):
        is_revisit = bool(target_gate[i].item() > 0.5)
        match = int(match_idx[i].item())
        goal_anchor = int(encoded["goal_anchor_idx"][i].item())
        positive_distance = None
        if is_revisit:
            positive_indices = torch.where(pos[i])[0]
            positive_distance = int((positive_indices - match).abs().min().item())
        row = dict(record)
        row.update(
            is_revisit=is_revisit,
            match_idx=match,
            goal_anchor_idx=goal_anchor,
            recall_gap=int(record["k"] - goal_anchor),
            predicted_depth=("deep" if int(record["k"] - goal_anchor) >= DEEP_GAP else "shallow")
            if is_revisit else None,
            metric_scale=_finite_float(encoded["metric_scale"][i]),
            gate_prob=_finite_float(predicted_gate[i]),
            gate_bce=_finite_float(per_gate_bce[i]),
            gate_correct=float((predicted_gate[i] > 0.5).item() == is_revisit),
            retrieval_loss=_finite_float(per_rank[i]) if bool(valid_rank[i]) else None,
            retrieval_hit=_finite_float(hit[i]) if is_revisit else None,
            positive_frame_distance=positive_distance,
            aux_loss=_finite_float(aux_mse[i]) if is_revisit else None,
            pos_err_m=_finite_float(pos_err[i]) if is_revisit else None,
            pos_dir_err_deg=_finite_float(pos_dir[i]) if is_revisit else None,
            rot_err_deg=_finite_float(rot_err[i]) if is_revisit else None,
            action_loss_pred=_finite_float(sums["pred"][i]),
            action_loss_mid=_finite_float(sums["mid"][i]),
            action_loss_oracle=_finite_float(sums["oracle"][i]),
            oracle_better_frac=_finite_float(sums["oracle_better"][i]),
            pred_oracle_rms=_finite_float(sums["rms"][i]),
        )
        rows.append(row)
    return rows


def _scope_rows(rows: list[dict]) -> dict[str, list[dict]]:
    scopes = {
        "all": rows,
        "novel": [r for r in rows if r["group"].startswith("novel")],
        "revisit": [r for r in rows if r["is_revisit"]],
        "leg2": [r for r in rows if r["group"].startswith("leg2")],
        "leg3": [r for r in rows if r["group"].startswith("leg3")],
        "gt_shallow": [r for r in rows if r["group"].endswith("shallow")],
        "gt_deep": [r for r in rows if r["group"].endswith("deep")],
        "pred_shallow": [r for r in rows if r.get("predicted_depth") == "shallow"],
        "pred_deep": [r for r in rows if r.get("predicted_depth") == "deep"],
    }
    scopes.update({group: [r for r in rows if r["group"] == group] for group in GROUPS})
    return scopes


def summarize(rows: list[dict]) -> dict:
    metrics = sorted(LOWER_IS_BETTER | UPPER_IS_BETTER | DESCRIPTIVE)
    result = {}
    for scope, scoped in _scope_rows(rows).items():
        block = {"count": len(scoped)}
        for metric in metrics:
            values = np.asarray(
                [r[metric] for r in scoped if r.get(metric) is not None], dtype=np.float64
            )
            if values.size:
                block[metric] = {
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                }
        result[scope] = block
    return result


def paired_comparison(baseline: list[dict], candidate: list[dict], seed: int) -> dict:
    base_by_id = {r["record_id"]: r for r in baseline}
    cand_by_id = {r["record_id"]: r for r in candidate}
    if set(base_by_id) != set(cand_by_id):
        raise RuntimeError("checkpoint result rows do not share an identical manifest")
    ordered = [record_id for record_id in base_by_id]
    scopes = _scope_rows([cand_by_id[record_id] for record_id in ordered])
    rng = np.random.default_rng(seed)
    out = {}
    metrics = sorted(LOWER_IS_BETTER | UPPER_IS_BETTER | DESCRIPTIVE)
    for scope, scope_rows in scopes.items():
        ids = [r["record_id"] for r in scope_rows]
        block = {"count": len(ids)}
        for metric in metrics:
            pairs = [
                (base_by_id[rid].get(metric), cand_by_id[rid].get(metric))
                for rid in ids
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                continue
            a = np.asarray([x[0] for x in pairs], dtype=np.float64)
            b = np.asarray([x[1] for x in pairs], dtype=np.float64)
            delta = b - a
            if len(delta) == 1:
                lo = hi = float(delta[0])
            else:
                indices = rng.integers(0, len(delta), size=(2000, len(delta)))
                boot = delta[indices].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5]).tolist()
            if metric in LOWER_IS_BETTER:
                better = float((b < a).mean())
            elif metric in UPPER_IS_BETTER:
                better = float((b > a).mean())
            else:
                better = None
            block[metric] = {
                "n": len(delta),
                "baseline_mean": float(a.mean()),
                "candidate_mean": float(b.mean()),
                "paired_delta": float(delta.mean()),
                "ci95": [float(lo), float(hi)],
                "candidate_better_frac": better,
            }
        out[scope] = block
    return out


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-group", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        parser.error("exactly two --checkpoint arguments are required")
    if args.samples_per_group < 1 or args.batch_size < 1 or args.trials < 1:
        parser.error("sample, batch, and trial counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    il = memnav_exp_cfg.il
    dataset = MemNav_Dataset(
        il.root_dir,
        predict_size=il.predict_size,
        image_size=il.image_size,
        lingbot_repo=il.lingbot_repo,
        feature_root=getattr(il, "feature_root", None),
        window_size=il.window_size,
        num_scale=il.num_scale,
        max_legs=getattr(il, "max_legs", None),
    )
    manifest = build_manifest(dataset, args.samples_per_group, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "manifest.json",
        {
            "seed": args.seed,
            "samples_per_group": args.samples_per_group,
            "batch_size": args.batch_size,
            "trials": args.trials,
            "groups": GROUPS,
            "records": manifest,
        },
    )
    group_counts = {group: sum(record["group"] == group for record in manifest) for group in GROUPS}
    print(
        f"[fixed-eval] manifest={len(manifest)} groups={group_counts}",
        flush=True,
    )

    config = MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump())
    _first_label, first_path = args.checkpoint[0]
    model = MemNavPolicy.from_pretrained(first_path, config=config).cuda().eval()
    load_heads(model, first_path)
    if model.training or model.core.training:
        raise RuntimeError("policy did not enter eval mode")

    all_results: dict[str, list[dict]] = {}
    for checkpoint_index, (label, path) in enumerate(args.checkpoint):
        if checkpoint_index:
            load_heads(model, path)
            model.eval()
        rows = []
        row_path = args.out / f"{label}.rows.jsonl"
        with row_path.open("w") as handle:
            for start in range(0, len(manifest), args.batch_size):
                records = manifest[start : start + args.batch_size]
                batch = materialize_batch(dataset, records)
                batch_rows = evaluate_batch(
                    model,
                    batch,
                    records,
                    trials=args.trials,
                    seed=args.seed,
                    batch_index=start // args.batch_size,
                )
                for row in batch_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                rows.extend(batch_rows)
                print(
                    f"[fixed-eval] {label}: {len(rows)}/{len(manifest)} "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                    flush=True,
                )
                del batch, batch_rows
                gc.collect()
        all_results[label] = rows
        summary = summarize(rows)
        write_json(args.out / f"{label}.summary.json", summary)
        primary = summary["all"]
        revisit_summary = summary["revisit"]
        print(
            f"[fixed-eval] SUMMARY {label}: "
            f"action={primary['action_loss_pred']['mean']:.6f} "
            f"gate_acc={primary['gate_correct']['mean']:.4f} "
            f"retrieval={revisit_summary['retrieval_loss']['mean']:.6f} "
            f"hit={revisit_summary['retrieval_hit']['mean']:.4f} "
            f"pos={revisit_summary['pos_err_m']['mean']:.4f}m "
            f"rot={revisit_summary['rot_err_deg']['mean']:.3f}deg",
            flush=True,
        )

    baseline_label, candidate_label = args.checkpoint[0][0], args.checkpoint[1][0]
    comparison = paired_comparison(
        all_results[baseline_label], all_results[candidate_label], args.seed + 91
    )
    write_json(
        args.out / "comparison.json",
        {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "scopes": comparison,
        },
    )
    for scope in ("all", "revisit", "leg2", "leg3", "gt_shallow", "gt_deep"):
        block = comparison[scope]
        fields = []
        for metric in ("action_loss_pred", "retrieval_loss", "gate_bce", "pos_err_m", "rot_err_deg"):
            if metric in block:
                fields.append(f"{metric}={block[metric]['paired_delta']:+.5f}")
        print(f"[fixed-eval] DELTA candidate-baseline {scope}: {' '.join(fields)}", flush=True)
    print(f"[fixed-eval] PASS outputs={args.out}", flush=True)


if __name__ == "__main__":
    main()
