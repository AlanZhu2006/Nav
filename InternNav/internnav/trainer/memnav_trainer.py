import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.trainer.base import BaseTrainer


class MemNavTrainer(BaseTrainer):
    """memnav: frozen LingBot front-end + trainable retrieval / novel / current_state /
    revisit / DDPM decoder. Loss = 0.5·ng + 0.5·mg (ε-MSE) + retrieval-CE + aux-pose.
    No critic — collision is checked geometrically from the point map at eval."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.w_retr = getattr(config.il, "w_retrieval", 1.0)
        self.w_aux = getattr(config.il, "w_aux_pose", 0.5)
        self.model_device = (self.model.module if hasattr(self.model, "module") else self.model).device
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        fwd = model(inputs)                                       # forward(batch) moves tensors internally

        # --- diffusion action loss (classifier-free ng + mg) ---
        noise = fwd["noise"]
        ng_loss = (fwd["noise_ng"] - noise).square().mean()
        mg_loss = (fwd["noise_mg"] - noise).square().mean()
        action_loss = 0.5 * ng_loss + 0.5 * mg_loss

        # --- retrieval: multi-positive InfoNCE over real frames + a null slot ---
        # A goal view co-observes a CONTIGUOUS band of history frames, so retrieval
        # has MANY positives (thresholded from covis_curve), not a single index.
        # pos/neg masks are over real frames [0..L-1]; the extra null column is a
        # positive for NOVEL goals (no real match) and a negative for REVISIT goals.
        logits = fwd["ret_logits"]                               # [B, L+1] (last = null)
        pos_real = inputs["batch_pos_mask"].to(dev).bool()       # [B, L]
        neg_real = inputs["batch_neg_mask"].to(dev).bool()       # [B, L]
        null_pos = inputs["batch_null_pos"].to(dev).bool()       # [B]
        pos_full = torch.cat([pos_real, null_pos[:, None]], 1)   # [B, L+1]
        neg_full = torch.cat([neg_real, (~null_pos)[:, None]], 1)
        valid = pos_full | neg_full                              # ignore-band frames excluded from both
        NEG_INF = torch.finfo(logits.dtype).min
        lse_all = logits.masked_fill(~valid, NEG_INF).logsumexp(-1)     # denom: over pos ∪ neg
        lse_pos = logits.masked_fill(~pos_full, NEG_INF).logsumexp(-1)  # numer: over positives
        # -log( Σ_pos e^s / Σ_{pos∪neg} e^s ). Every sample has ≥1 positive by
        # construction (revisit → real positive; novel → null positive).
        retrieval_loss = (lse_all - lse_pos).mean()

        # --- aux pose (x,y,θ): MSE on REVISIT samples only (relocalization branch) ---
        gt_pose = inputs["batch_goal_rel_pose"].to(dev)          # [B,3]
        revisit = (~null_pos).float()                            # 1 = goal is in memory
        per = (fwd["aux_pose"] - gt_pose).square().mean(-1)      # [B]
        aux_loss = (per * revisit).sum() / revisit.sum().clamp(min=1.0)

        loss = action_loss + self.w_retr * retrieval_loss + self.w_aux * aux_loss

        with torch.no_grad():
            pred = logits.argmax(-1)                              # [B] over [0..L] (incl null)
            correct = pos_full.gather(1, pred[:, None]).squeeze(1).float()  # predicted a positive slot
            ret_acc = correct.mean()
            # --- gate revisit/novel separation + per-mode match acc (key diagnostics) ---
            gate = fwd["revisit_gate"]                            # [B] P(some real match): HIGH revisit / LOW novel
            ns = revisit.sum().clamp(min=1.0)
            nu = (1.0 - revisit).sum().clamp(min=1.0)
            gate_seen = (gate * revisit).sum() / ns               # → 1 (visited)
            gate_unseen = (gate * (1.0 - revisit)).sum() / nu     # → 0 (novel)
            gate_sep = gate_seen - gate_unseen                    # → large +  (the separation)
            seen_match = (correct * revisit).sum() / ns           # found a real match (revisit)
            unseen_null = (correct * (1.0 - revisit)).sum() / nu  # correctly chose null (novel)
        outputs = dict(loss=loss, action_loss=action_loss, ng_loss=ng_loss, mg_loss=mg_loss,
                       retrieval_loss=retrieval_loss, aux_loss=aux_loss, ret_acc=ret_acc,
                       gate_seen=gate_seen, gate_unseen=gate_unseen, gate_sep=gate_sep,
                       seen_match_acc=seen_match, unseen_null_acc=unseen_null)
        if (dist.get_rank() if dist.is_initialized() else 0) == 0:
            print(f"[Step {self.state.global_step}] loss={loss.item():.4f} act={action_loss.item():.4f} "
                  f"retr={retrieval_loss.item():.4f}(acc {ret_acc.item():.2f}) aux={aux_loss.item():.4f} | "
                  f"gate seen={gate_seen.item():.2f} unseen={gate_unseen.item():.2f} sep={gate_sep.item():+.2f} | "
                  f"match seen={seen_match.item():.2f} unseen_null={unseen_null.item():.2f}")

        # Per-component metrics → wandb/tb. self.log is rank-0-only inside HF Trainer;
        # gate by logging_steps to match train/loss cadence and avoid extra .item() syncs.
        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {
                'train/action_loss': action_loss.item(),
                'train/ng_loss': ng_loss.item(),
                'train/mg_loss': mg_loss.item(),
                'train/retrieval_loss': retrieval_loss.item(),
                'train/aux_loss': aux_loss.item(),
                'train/ret_acc': ret_acc.item(),
                'train/gate_seen': gate_seen.item(),
                'train/gate_unseen': gate_unseen.item(),
                'train/gate_sep': gate_sep.item(),
                'train/seen_match_acc': seen_match.item(),
                'train/unseen_null_acc': unseen_null.item(),
            }
            if dev.type == 'cuda':
                # Peak since previous logging_step, in GiB. Reset right after so the
                # next window measures its own peak — otherwise max_ stays monotone.
                alloc = torch.cuda.max_memory_allocated(dev) / 2**30
                reserved = torch.cuda.max_memory_reserved(dev) / 2**30
                log_payload['train/mem_alloc_gb'] = alloc
                log_payload['train/mem_reserved_gb'] = reserved
                if (dist.get_rank() if dist.is_initialized() else 0) == 0:
                    print(f"[Step {self.state.global_step}] mem peak "
                          f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB")
                torch.cuda.reset_peak_memory_stats(dev)
            self.log(log_payload)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------ #
    def create_optimizer(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        lr = getattr(self.config.il, "lr", 1e-4)
        m = self.model.module if hasattr(self.model, "module") else self.model
        params = [p for p in m.parameters() if p.requires_grad]       # frozen LingBot excluded
        self.optimizer = torch.optim.Adam(params, lr=lr)
        if rank == 0:
            n = sum(p.numel() for p in params)
            print(f"[Rank 0] Adam lr={lr}; trainable params: {n:,} ({len(params)} tensors)")
        return self.optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator or memnav_collate_fn,
        )

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """Save only the trainable heads (skip the frozen LingBot — reloaded separately at eval)."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        sd = {k: v for k, v in m.state_dict().items() if "lingbot." not in k}
        os.makedirs(output_dir, exist_ok=True)
        torch.save(sd, os.path.join(output_dir, "memnav.ckpt"))
        print(f"Saved {len(sd)} trainable tensors to {output_dir}/memnav.ckpt")
