"""Retrieval and revisit-gate head for MemNav.

Kept independent of the heavy visual backbones so its masking, initialization,
and gate semantics can be unit-tested without LingBot/LongCLIP installed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalHead(nn.Module):
    """Rank candidate frames and independently classify revisit vs novel.

    Ranking uses trainable projected cosine. The gate uses the raw frozen-DINO
    maximum cosine because that is the statistic whose class separation and
    threshold were measured by the offline probe.
    """

    def __init__(self, dino_dim=1024, proj_dim=256, temp_init=0.07):
        super().__init__()
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        # Preserve the frozen representation's cosine ordering at step zero.
        # The projections remain independent parameters and can diverge later.
        with torch.no_grad():
            self.proj_mem.weight.copy_(self.proj_goal.weight)
            self.proj_mem.bias.copy_(self.proj_goal.bias)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))
        self.gate_a = nn.Parameter(torch.tensor(10.0))
        self.gate_b = nn.Parameter(torch.tensor(-8.0))

    def forward(self, goal_cls, mem_cls, cand_mask):
        """Return match index, gate logit, ranking logits, and raw gate feature."""
        raw_goal = F.normalize(goal_cls.float(), dim=-1)
        raw_mem = F.normalize(mem_cls.float(), dim=-1)
        raw_cos = (raw_goal.unsqueeze(1) * raw_mem).sum(-1)

        gq = F.normalize(self.proj_goal(goal_cls), dim=-1)
        mk = F.normalize(self.proj_mem(mem_cls), dim=-1)
        temp = self.log_temp.exp().clamp(0.01, 1.0)
        rank_cos = (gq.unsqueeze(1) * mk).sum(-1)

        neg_inf = torch.finfo(rank_cos.dtype).min
        # Mask after division. Dividing a literal -inf by trainable temperature
        # can produce a NaN temperature gradient even when upstream is masked.
        ret_logits = (rank_cos / temp).masked_fill(~cand_mask, neg_inf)

        has_candidate = cand_mask.any(-1)
        max_raw_cos = raw_cos.masked_fill(~cand_mask, -1.0).max(-1).values
        max_raw_cos = torch.where(
            has_candidate, max_raw_cos, max_raw_cos.new_full((), -1.0)
        )
        gate_logit = self.gate_a * max_raw_cos + self.gate_b
        match_idx = ret_logits.argmax(-1)
        return match_idx, gate_logit, ret_logits, max_raw_cos
