#!/usr/bin/env python
"""Which decoder branches are USED, and which are USEFUL?

Follow-up to diag_decgate_zsweep.py, which showed the action loss is locally
optimal in the decoder gate logit z.  That sweep moved the revisit/novel
attention *balance*; it cannot say which branch's CONTENT the decoder reads.
This script ablates content at the fixed operating point (z = z_pred):

  arms (same fixed rows / paired noise / timesteps as the z-sweep):
    base          intact revisit + novel tokens
    zero_novel    novel tokens zeroed
    zero_revisit  revisit tokens zeroed (the eval gate-skip precedent)
    zero_both     both zeroed
    swap_novel    novel tokens from a different row in the batch
    swap_revisit  revisit tokens from a different row in the batch

  per row / arm:
    loss   epsilon-MSE (usefulness: does removing/corrupting content hurt?)
    rms    RMS(pred_arm - pred_base) (usage: does the output depend on it?)

Reading: a branch is USED iff its zero/swap arms move the output (rms >> 0).
It is USEFUL iff corrupting it raises the loss.  Used-but-not-useful = the
decoder attends but the content doesn't steer toward better waypoints.
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

os.environ.setdefault("MEMNAV_MAX_FRAME_NUM", "2048")

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn  # noqa: E402
from internnav.model.basemodel.memnav.memnav_policy import (                            # noqa: E402
    MemNavModelConfig, MemNavPolicy)
from scripts.train.configs.memnav import memnav_exp_cfg                                  # noqa: E402

ARMS = ("base", "zero_novel", "zero_revisit", "zero_both", "swap_novel", "swap_revisit")


def pick_ks(sample, per_goal):
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
    ap.add_argument("--ks-per-goal", type=int, default=3)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

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

    records = []
    for sample_index, sample in enumerate(dataset.samples):
        for k in pick_ks(sample, args.ks_per_goal):
            pos, _neg, _cand, _null = dataset._build_label(sample, k)
            records.append(dict(
                sample_index=sample_index,
                trajectory=os.path.relpath(
                    dataset.trajectory_dirs[sample["traj_idx"]], args.root_dirs),
                goal_j=int(sample["goal_j"]), k=int(k),
                gt_revisit=bool(pos.any()),
            ))
    # permute so swap arms pair rows from different episodes, not adjacent ks
    rng = np.random.default_rng(args.seed)
    records = [records[i] for i in rng.permutation(len(records))]
    n_rev = sum(r["gt_revisit"] for r in records)
    print(f"[ablate] manifest: {len(records)} rows ({n_rev} GT-revisit)", flush=True)

    policy = MemNavPolicy.from_pretrained(
        args.checkpoint, config=MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump()))
    policy.eval()
    core = policy.core
    dev = core.device

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
            revisit, _aux, _r = core.build_revisit(
                enc["cur_pose"], enc["goal_pose"], enc["metric_scale"])
            novel = core.novel(
                batch["batch_window_images"][:, -1].to(dev),
                batch["batch_goal_image"].to(dev))

        labels = batch["batch_labels"].to(dev)
        z = enc["dec_gate_logit"].float()
        bsz = labels.shape[0]
        shift = 1 if bsz > 1 else 0
        variants = {
            "base": (novel, revisit),
            "zero_novel": (torch.zeros_like(novel), revisit),
            "zero_revisit": (novel, torch.zeros_like(revisit)),
            "zero_both": (torch.zeros_like(novel), torch.zeros_like(revisit)),
            "swap_novel": (novel.roll(shift, dims=0), revisit),
            "swap_revisit": (novel, revisit.roll(shift, dims=0)),
        }
        loss_sums = {a: torch.zeros(bsz, device=dev) for a in ARMS}
        rms_sums = {a: torch.zeros(bsz, device=dev) for a in ARMS}

        for trial in range(args.trials):
            g = torch.Generator(device=dev)
            g.manual_seed(int(args.seed + 1_000_003 * (start // args.batch_size) + trial))
            noise = torch.randn(labels.shape, generator=g, device=dev, dtype=labels.dtype)
            timesteps = torch.randint(
                0, core.noise_scheduler.config.num_train_timesteps, (bsz,),
                generator=g, device=dev)
            noisy = core.noise_scheduler.add_noise(labels, noise, timesteps)
            preds = {}
            with amp:
                for arm in ARMS:
                    nv, rv = variants[arm]
                    preds[arm] = core.predict_noise(noisy, timesteps, current, rv, nv, z)
            for arm in ARMS:
                loss_sums[arm] += (preds[arm].float() - noise).square().mean(dim=(-2, -1))
                rms_sums[arm] += (preds[arm].float() - preds["base"].float()) \
                    .square().mean(dim=(-2, -1)).sqrt()

        for i, rec in enumerate(chunk):
            row = dict(rec)
            row.update(
                z_pred=float(z[i].item()),
                gate_prob=float(enc["revisit_gate"][i].item()),
                loss={a: float((loss_sums[a][i] / args.trials).item()) for a in ARMS},
                rms_vs_base={a: float((rms_sums[a][i] / args.trials).item()) for a in ARMS},
            )
            rows_out.append(row)
        print(f"[ablate] {start + len(chunk)}/{len(records)} rows", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "rows.jsonl", "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    rng = np.random.default_rng(args.seed)
    summary = {"n_rows": len(rows_out), "checkpoint": args.checkpoint, "groups": {}}
    for gname, rows in (("revisit", [r for r in rows_out if r["gt_revisit"]]),
                        ("novel", [r for r in rows_out if not r["gt_revisit"]])):
        if not rows:
            continue
        base = np.array([r["loss"]["base"] for r in rows])
        entry = {"n": len(rows), "loss_base": float(base.mean()), "arms": {}}
        for arm in ARMS[1:]:
            d = np.array([r["loss"][arm] for r in rows]) - base
            boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            entry["arms"][arm] = dict(
                dloss=float(d.mean()),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                frac_worse=float((d > 0).mean()),
                rms=float(np.mean([r["rms_vs_base"][arm] for r in rows])),
            )
        summary["groups"][gname] = entry
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for gname, entry in summary["groups"].items():
        print(f"\n[ablate] {gname} rows (n={entry['n']}, base loss {entry['loss_base']:.6f}):")
        for arm in ARMS[1:]:
            a = entry["arms"][arm]
            print(f"  {arm:13s} dloss={a['dloss']:+.6f} CI[{a['ci_lo']:+.6f},{a['ci_hi']:+.6f}]"
                  f"  worse={100 * a['frac_worse']:.0f}%  output_rms={a['rms']:.4f}")


if __name__ == "__main__":
    main()
