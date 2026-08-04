#!/usr/bin/env python
"""Build a MemNav warm-start checkpoint from a trained NavDP checkpoint.

MemNav's NovelBranch wraps the SAME image-goal encoder architecture NavDP
trains (NavDP_ImageGoal_Backbone: DINOv2-S/14, 6-channel early-fusion
current||goal patch embed) — verified 174/174 key/shape match against
checkpoint-5570. NavDP's encoder is goal-sensitive because its training data +
point-goal co-supervision made ignoring the goal expensive; MemNav's two-leg
data does not (measured collapse: wrong-goal swap moves the output by 0.13-3.16%
of seed-level variation vs NavDP's 176.8%). Importing the weights imports that
sensitivity as an INITIAL CONDITION; keep it protected during training with
MEMNAV_FREEZE_NOVEL_BACKBONE=1 (hard guarantee) and/or the goal-swap
counterfactual loss (usage-level pressure) — initialization alone does not
change which solutions are low-loss.

Usage:
  python scripts/train_memnav/warmstart_navdp.py \
      --navdp_ckpt /path/to/navdp_checkpoint.ckpt \
      --out checkpoints/navdp_warmstart/memnav_novel_init.ckpt
  # then: export MEMNAV_CKPT_TO_LOAD=<out>   (loaded strict=False; every other
  # MemNav head stays freshly initialized)

Optionally pass --reference <memnav.ckpt> to verify key/shape coverage against
a real MemNav checkpoint before writing.
"""

from __future__ import annotations

import argparse
import os

import torch

# NavDP prefix -> MemNav prefix. project_layer is deliberately NOT mapped
# (NovelBranch replaces it with nn.Identity — it skips NavDP's mean-pool
# readout); mask_token is a DINOv2 MIM artifact absent from MemNav instances.
PREFIX_MAP = {
    "image_encoder.": "core.novel.backbone.",
}
SKIP_SUBSTRINGS = ("project_layer", "mask_token")


def remap(navdp_sd: dict) -> dict:
    out = {}
    for key, value in navdp_sd.items():
        for src, dst in PREFIX_MAP.items():
            if key.startswith(src) and not any(s in key for s in SKIP_SUBSTRINGS):
                out[dst + key[len(src):]] = value
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navdp_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None,
                    help="optional MemNav checkpoint to verify coverage/shapes against")
    args = ap.parse_args()

    sd = torch.load(args.navdp_ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    mapped = remap(sd)
    if not mapped:
        raise SystemExit(f"no keys matched {list(PREFIX_MAP)} in {args.navdp_ckpt}")

    # structural sanity: the 6-channel early-fusion patch embed is the signature
    # of the image-goal encoder — refuse to write a checkpoint without it.
    pe = mapped.get("core.novel.backbone.imagegoal_encoder.patch_embed.proj.weight")
    if pe is None or pe.shape[1] != 6:
        raise SystemExit(f"unexpected patch_embed in source checkpoint: "
                         f"{None if pe is None else tuple(pe.shape)} (want [*, 6, *, *])")

    if args.reference:
        ref = torch.load(args.reference, map_location="cpu", weights_only=True)
        ref = ref.get("state_dict", ref) if isinstance(ref, dict) else ref
        ref_novel = {k for k in ref if k.startswith("core.novel.backbone.")}
        missing = sorted(ref_novel - set(mapped))
        mismatched = sorted(k for k in mapped
                            if k in ref and tuple(ref[k].shape) != tuple(mapped[k].shape))
        extra = sorted(set(mapped) - ref_novel)
        print(f"[warmstart] reference check: covered {len(ref_novel) - len(missing)}"
              f"/{len(ref_novel)} novel-backbone keys; "
              f"missing={len(missing)} mismatched={len(mismatched)} extra={len(extra)}")
        if missing or mismatched or extra:
            for k in (missing + mismatched + extra)[:10]:
                print("  !!", k)
            raise SystemExit("reference verification failed — refusing to write")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(mapped, args.out)
    n_params = sum(v.numel() for v in mapped.values())
    print(f"[warmstart] wrote {len(mapped)} tensors ({n_params / 1e6:.1f}M params) -> {args.out}")
    print("[warmstart] use with: export MEMNAV_CKPT_TO_LOAD=" + os.path.abspath(args.out))
    print("[warmstart] recommended: export MEMNAV_FREEZE_NOVEL_BACKBONE=1")


if __name__ == "__main__":
    main()
