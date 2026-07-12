import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalHead(nn.Module):
    """Retrieve history frames for a goal, with a learnable novel-goal slot."""

    def __init__(self, dino_dim=1024, proj_dim=256, temp_init=0.07):
        super().__init__()
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        self.null_key = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))

    def forward(self, goal_cls, mem_cls, mem_mask):
        """Score ``mem_cls`` and null using a structural candidate mask."""
        if not mem_mask.any(-1).all():
            raise ValueError("each sample needs at least one reconstructable retrieval candidate")

        goal_query = F.normalize(self.proj_goal(goal_cls), dim=-1)
        memory_keys = F.normalize(self.proj_mem(mem_cls), dim=-1)
        null_key = F.normalize(self.null_key, dim=-1)
        temperature = self.log_temp.exp().clamp(0.01, 1.0)

        scores = (goal_query.unsqueeze(1) * memory_keys).sum(-1) / temperature
        scores = scores.masked_fill(~mem_mask, float("-inf"))
        null = (goal_query * null_key).sum(-1, keepdim=True) / temperature
        logits = torch.cat([scores, null], dim=1)

        probabilities = logits.softmax(-1)
        revisit_gate = 1.0 - probabilities[:, -1]
        match_idx = scores.argmax(-1)
        return match_idx, revisit_gate, logits
