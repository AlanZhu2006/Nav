#!/usr/bin/env python
"""One-real-batch MemNav forward/backward preflight for long Slurm runs.

Run only in a short prerequisite job (``MEMNAV_PREFLIGHT_ONLY=1``), inside the
same container/overlay/Conda environment as training.  This proves more than an
import smoke: it loads the configured cache, frozen LingBot weights, optional
warm-start heads, applies the gate curriculum, and checks finite gradients.
"""

from __future__ import annotations

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


def deterministic_revisit(dataset: MemNav_Dataset):
    """Load one revisit row at a deterministic k, without random retry I/O."""
    for idx, sample in enumerate(dataset.samples):
        if not sample["has_covis"]:
            continue
        valid = []
        for k in range(int(sample["k_lo"]), int(sample["k_hi"]) + 1):
            pos, neg, _cand, _null = dataset._build_label(sample, k)
            if pos.any() and neg.any():
                valid.append(k)
        if not valid:
            continue
        k = valid[len(valid) // 2]
        # random_digit is false in the train config, so this is the sole randint.
        with patch("numpy.random.randint", return_value=k):
            item = dataset[idx]
        if int(item["cur_step"]) != k or not bool(item["is_revisit"].item()):
            raise RuntimeError("deterministic revisit selection did not hold")
        return idx, item
    raise RuntimeError("dataset contains no revisit row with positive/negative contrast")


def main() -> None:
    torch.manual_seed(17)
    np.random.seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(0)

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
    idx, item = deterministic_revisit(dataset)
    batch = memnav_collate_fn([item])
    print(
        f"[full-preflight] sample={idx} k={batch['cur_steps'][0]} "
        f"goal={batch['goal_steps'][0]} pos={int(batch['batch_pos_mask'].sum())} "
        f"neg={int(batch['batch_neg_mask'].sum())}",
        flush=True,
    )

    config = MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump())
    model = MemNavPolicy.from_pretrained(il.ckpt_to_load, config=config).cuda().train()
    if il.ckpt_to_load:
        state = torch.load(il.ckpt_to_load, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        current = model.state_dict()
        probe_keys = (
            "core.retrieval.gate_a",
            "core.retrieval.gate_b",
            "core.revisit_merge.revisit_head.weight",
            "core.decoder.layers.0.self_attn.in_proj_weight",
        )
        mismatched = [key for key in probe_keys
                      if key not in state or not torch.equal(current[key].cpu(), state[key].cpu())]
        if mismatched:
            raise RuntimeError(f"warm-start tensors were not loaded exactly: {mismatched}")
        print(
            f"[full-preflight] warm start tensors match: {il.ckpt_to_load} "
            f"({len(probe_keys)} probes)",
            flush=True,
        )
        del state, current
    # The wrapper follows train mode; its frozen feature producer must not.
    if model.core.lingbot.model.training or model.core.lingbot.depth_feat_head.training:
        raise RuntimeError("frozen LingBot child escaped into train mode")

    teacher_ratio = 0.5
    batch["decoder_gate_teacher_ratio"] = teacher_ratio
    torch.cuda.reset_peak_memory_stats()
    forward = model(batch)
    target = batch["batch_is_revisit"].cuda()
    predicted_gate = forward["predicted_revisit_gate"]
    expected_gate = predicted_gate + teacher_ratio * (target - predicted_gate)
    if not torch.allclose(forward["revisit_gate"], expected_gate):
        raise RuntimeError("decoder gate did not apply teacher blend")

    action_loss = (forward["noise_pred"] - forward["noise"]).square().mean()
    action_to_gate = torch.autograd.grad(
        action_loss, forward["gate_logit"], retain_graph=True, allow_unused=False
    )[0]
    if not torch.isfinite(action_to_gate).all() or not bool(action_to_gate.abs().max() > 0):
        raise RuntimeError(f"invalid action-to-gate gradient: {action_to_gate}")

    logits = forward["ret_logits"]
    pos = batch["batch_pos_mask"].cuda().bool()
    neg = batch["batch_neg_mask"].cuda().bool()
    floor = torch.finfo(logits.dtype).min
    rank_loss = (
        logits.masked_fill(~(pos | neg), floor).logsumexp(-1)
        - logits.masked_fill(~pos, floor).logsumexp(-1)
    ).mean()
    gate_loss = F.binary_cross_entropy_with_logits(forward["gate_logit"], target)
    gt_pose = batch["batch_goal_rel_pose"][:, :2].cuda()
    aux_loss = (forward["aux_pose"] - gt_pose).square().mean()
    loss = action_loss + il.w_retrieval * rank_loss + il.w_gate * gate_loss + il.w_aux_pose * aux_loss

    model.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    bad = [name for name, param in trainable
           if param.grad is not None and not torch.isfinite(param.grad).all()]
    if bad:
        raise RuntimeError(f"non-finite trainable gradients: {bad[:10]}")
    critical = (
        "core.retrieval.gate_a",
        "core.revisit_merge.revisit_head.weight",
        "core.decoder.layers.0.self_attn.in_proj_weight",
        "core.novel.proj.weight",
    )
    named = dict(model.named_parameters())
    missing_grad = [name for name in critical if named[name].grad is None]
    if missing_grad:
        raise RuntimeError(f"critical parameters lack gradients: {missing_grad}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[full-preflight] PASS loss={loss.item():.6f} "
        f"pred_gate={predicted_gate.item():.6f} decoder_gate={forward['revisit_gate'].item():.6f} "
        f"action_dgate={action_to_gate.item():+.3e} peak_alloc={peak:.2f}GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
