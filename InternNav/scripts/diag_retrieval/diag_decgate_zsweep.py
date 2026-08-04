#!/usr/bin/env python
"""Is the action loss flat in the decoder gate logit z?

The decgate run (W&B l3bhs8i8) left dec_gate_a/b at a random walk around init
(net +0.009/+0.165 over 5.5k steps) while the BCE-trained gate_a/gate_b in the
SAME 10x-lr param group moved monotonically (+1.47/+1.08).  That pattern says
the action-loss gradient in z is ~zero-mean — i.e. the decoder is indifferent
to the revisit/novel attention split.  This script measures that directly:

  * fixed samples (every goal of every local episode, several fixed k each),
  * fixed paired diffusion noise/timesteps across all arms,
  * one encode per batch, then predict_noise at dec_gate_logit + dz for a grid
    of offsets dz.

If epsilon-MSE(dz) is flat on GT-revisit rows, there is nothing for
dec_gate_a/b to descend — retraining with routing scaffolding (logit-space
curriculum / counterfactual anchor supervision) is needed, not more steps.

Run (memnav env, after precomputing flowgate caches for the eval episodes):

  python scripts/diag_retrieval/diag_decgate_zsweep.py \
      --checkpoint /path/to/checkpoint-5570/memnav.ckpt \
      --root_dirs /home/asus/Research/datasets/_zsweep_root \
      --feature_root /home/asus/Research/datasets/memnav_eval2leg_v1_feat_flowgate \
      --out /path/to/out_dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# geometry must match the flowgate precompute + decgate training (W32/S8, RoPE 2048)
os.environ.setdefault("MEMNAV_MAX_FRAME_NUM", "2048")

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn  # noqa: E402
from internnav.model.basemodel.memnav.memnav_policy import (                            # noqa: E402
    MemNavModelConfig, MemNavPolicy)
from scripts.train.configs.memnav import memnav_exp_cfg                                  # noqa: E402


def pick_ks(sample, dataset, per_goal):
    """Deterministic spread of current-step choices across the sample's k range."""
    k_lo, k_hi = int(sample["k_lo"]), int(sample["k_hi"])
    if k_hi < k_lo:
        return []
    ks = np.unique(np.linspace(k_lo, k_hi, num=min(per_goal, k_hi - k_lo + 1)).astype(int))
    return [int(k) for k in ks]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root_dirs", required=True)
    ap.add_argument("--feature_root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--offsets", default="-4,-2,0,2,4,6",
                    help="additive offsets applied to the predicted dec_gate_logit")
    ap.add_argument("--ks-per-goal", type=int, default=3)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()
    offsets = [float(x) for x in args.offsets.split(",")]
    if 0.0 not in offsets:
        offsets.append(0.0)
    offsets = sorted(offsets)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    il = memnav_exp_cfg.il
    dataset = MemNav_Dataset(
        args.root_dirs,
        predict_size=il.predict_size,
        image_size=il.image_size,
        lingbot_repo=il.lingbot_repo,
        feature_root=args.feature_root,
        window_size=il.window_size,
        num_scale=il.num_scale,
    )
    print(f"[zsweep] dataset: {len(dataset.trajectory_dirs)} trajectories, "
          f"{len(dataset.samples)} goal samples", flush=True)

    # ---- fixed manifest: every goal, ks-per-goal fixed current steps ---- #
    records = []
    for sample_index, sample in enumerate(dataset.samples):
        for k in pick_ks(sample, dataset, args.ks_per_goal):
            pos, neg, cand, null_pos = dataset._build_label(sample, k)
            records.append(dict(
                sample_index=sample_index,
                trajectory=os.path.relpath(
                    dataset.trajectory_dirs[sample["traj_idx"]], args.root_dirs),
                goal_j=int(sample["goal_j"]),
                k=int(k),
                gt_revisit=bool(pos.any()),
                n_pos=int(pos.sum()), n_neg=int(neg.sum()),
            ))
    n_rev = sum(r["gt_revisit"] for r in records)
    print(f"[zsweep] manifest: {len(records)} rows ({n_rev} GT-revisit, "
          f"{len(records) - n_rev} GT-novel)", flush=True)

    policy = MemNavPolicy.from_pretrained(
        args.checkpoint, config=MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump()))
    policy.eval()
    core = policy.core
    dev = core.device
    print(f"[zsweep] dec_gate_a={core.dec_gate_a.item():.4f} "
          f"dec_gate_b={core.dec_gate_b.item():.4f}", flush=True)

    rows_out = []
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        items = []
        for rec in chunk:
            with patch("numpy.random.randint", return_value=int(rec["k"])):
                item = dataset[int(rec["sample_index"])]
            if int(item["cur_step"]) != int(rec["k"]):
                raise RuntimeError(f"fixed k did not hold for {rec}")
            items.append(item)
        batch = memnav_collate_fn(items)

        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with amp:
            enc = core.encode_memory(batch)
            current = core.build_current_state(enc["current"], enc["depth_feat"])
            revisit, aux_pose, _ = core.build_revisit(
                enc["cur_pose"], enc["goal_pose"], enc["metric_scale"])
            novel = core.novel(
                batch["batch_window_images"][:, -1].to(dev),
                batch["batch_goal_image"].to(dev))

        labels = batch["batch_labels"].to(dev)
        z_pred = enc["dec_gate_logit"].float()
        bsz = labels.shape[0]
        sums = {dz: torch.zeros(bsz, device=dev) for dz in offsets}

        for trial in range(args.trials):
            g = torch.Generator(device=dev)
            g.manual_seed(int(args.seed + 1_000_003 * (start // args.batch_size) + trial))
            noise = torch.randn(labels.shape, generator=g, device=dev, dtype=labels.dtype)
            timesteps = torch.randint(
                0, core.noise_scheduler.config.num_train_timesteps, (bsz,),
                generator=g, device=dev)
            noisy = core.noise_scheduler.add_noise(labels, noise, timesteps)
            with amp:
                for dz in offsets:
                    pred = core.predict_noise(
                        noisy, timesteps, current, revisit, novel, z_pred + dz)
                    sums[dz] += (pred.float() - noise).square().mean(dim=(-2, -1))

        for i, rec in enumerate(chunk):
            row = dict(rec)
            row.update(
                z_pred=float(z_pred[i].item()),
                gate_prob=float(enc["revisit_gate"][i].item()),
                match_idx=int(enc["match_idx"][i].item()),
                anchor_idx=int(enc["anchor_idx"][i].item()),
                loss_by_dz={str(dz): float((sums[dz][i] / args.trials).item())
                            for dz in offsets},
            )
            rows_out.append(row)
        done = start + len(chunk)
        print(f"[zsweep] {done}/{len(records)} rows", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "rows.jsonl", "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    # ---- summary: paired deltas vs dz=0 ---- #
    rng = np.random.default_rng(args.seed)
    summary = {"offsets": offsets, "n_rows": len(rows_out),
               "checkpoint": args.checkpoint, "groups": {}}
    for gname, rows in (("revisit", [r for r in rows_out if r["gt_revisit"]]),
                        ("novel", [r for r in rows_out if not r["gt_revisit"]])):
        if not rows:
            continue
        base = np.array([r["loss_by_dz"]["0.0"] for r in rows])
        entry = {"n": len(rows), "loss_at_0": float(base.mean()),
                 "z_pred_mean": float(np.mean([r["z_pred"] for r in rows])),
                 "delta_vs_0": {}}
        for dz in offsets:
            d = np.array([r["loss_by_dz"][str(dz)] for r in rows]) - base
            boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            entry["delta_vs_0"][str(dz)] = dict(
                mean=float(d.mean()),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                frac_improved=float((d < 0).mean()),
            )
        summary["groups"][gname] = entry
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for gname, entry in summary["groups"].items():
        print(f"\n[zsweep] {gname} rows (n={entry['n']}, "
              f"mean z_pred={entry['z_pred_mean']:+.3f}, "
              f"loss@dz=0 {entry['loss_at_0']:.6f}):")
        for dz in offsets:
            d = entry["delta_vs_0"][str(dz)]
            print(f"  dz={dz:+4.1f}  dloss={d['mean']:+.6f}  "
                  f"CI[{d['ci_lo']:+.6f},{d['ci_hi']:+.6f}]  "
                  f"improved={100 * d['frac_improved']:.0f}%")


if __name__ == "__main__":
    main()
