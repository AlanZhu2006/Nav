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

    def __init__(
        self,
        dino_dim=1024,
        proj_dim=256,
        temp_init=0.07,
        gate_center=0.94,
        gate_width=0.04,
        gate_slope_init=1.6,
        gate_bias_init=0.0,
    ):
        super().__init__()
        if gate_width <= 0:
            raise ValueError(f'gate_width must be positive, got {gate_width}')
        if gate_slope_init <= 0:
            raise ValueError(
                f'gate_slope_init must be positive, got {gate_slope_init}'
            )
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        # Preserve the frozen representation's cosine ordering at step zero.
        # The projections remain independent parameters and can diverge later.
        with torch.no_grad():
            self.proj_mem.weight.copy_(self.proj_goal.weight)
            self.proj_mem.bias.copy_(self.proj_goal.bias)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))
        # Raw DINO cosine lives in a narrow, high-similarity interval.  Optimizing
        # ``a * cosine + b`` at the policy LR made its decision threshold
        # ``-b/a`` almost immovable: both parameters had to coordinate while the
        # useful feature variation was only a few hundredths.  Work in calibrated
        # feature units instead, so the learnable slope/bias stay O(1).
        #
        # The defaults come from the *training* split probe (not the fixed eval
        # subset): revisit/novel separation occurs around 0.94 and a 0.04 width
        # covers the overlapping band.  Persistent buffers make the calibration
        # part of the checkpoint rather than an implicit runtime assumption.
        self.register_buffer(
            'gate_center', torch.tensor(float(gate_center), dtype=torch.float32)
        )
        self.register_buffer(
            'gate_width', torch.tensor(float(gate_width), dtype=torch.float32)
        )
        self.gate_log_slope = nn.Parameter(
            torch.tensor(float(np.log(gate_slope_init)), dtype=torch.float32)
        )
        self.gate_bias = nn.Parameter(
            torch.tensor(float(gate_bias_init), dtype=torch.float32)
        )

    @property
    def gate_slope(self):
        """Positive slope in normalized-feature units."""
        return self.gate_log_slope.exp()

    @property
    def effective_gate_threshold(self):
        """Raw-cosine value whose revisit probability is exactly 0.5."""
        return self.gate_center - self.gate_width * self.gate_bias / self.gate_slope

    def upgrade_legacy_state_dict(self, state_dict, prefix='', copy=True):
        """Convert the legacy ``gate_a * cosine + gate_b`` parameterization.

        The conversion is algebraically exact:

        ``slope = gate_a * width`` and
        ``bias = gate_a * center + gate_b``.

        It is intentionally a model-state migration only.  Old optimizer moments
        have different units and must not be resumed under the new training setup.
        """
        upgraded = state_dict.copy() if copy else state_dict
        old_a_key = f'{prefix}gate_a'
        old_b_key = f'{prefix}gate_b'
        log_slope_key = f'{prefix}gate_log_slope'
        bias_key = f'{prefix}gate_bias'
        center_key = f'{prefix}gate_center'
        width_key = f'{prefix}gate_width'
        has_old = old_a_key in upgraded or old_b_key in upgraded
        if not has_old:
            return upgraded
        if old_a_key not in upgraded or old_b_key not in upgraded:
            raise ValueError('legacy gate checkpoint must contain both gate_a and gate_b')
        if log_slope_key in upgraded or bias_key in upgraded:
            raise ValueError('checkpoint mixes legacy and normalized gate parameters')

        old_a = upgraded.pop(old_a_key)
        old_b = upgraded.pop(old_b_key)
        width = self.gate_width.detach().to(device=old_a.device, dtype=old_a.dtype)
        center = self.gate_center.detach().to(device=old_a.device, dtype=old_a.dtype)
        normalized_slope = old_a * width
        if not bool(torch.all(normalized_slope > 0)):
            raise ValueError(
                'legacy gate_a must be positive to migrate to a monotonic gate'
            )
        upgraded[log_slope_key] = normalized_slope.log()
        upgraded[bias_key] = old_a * center + old_b
        upgraded[center_key] = center
        upgraded[width_key] = width
        return upgraded

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        # Also support direct RetrievalHead.load_state_dict(...) outside the policy
        # checkpoint helpers.
        self.upgrade_legacy_state_dict(state_dict, prefix=prefix, copy=False)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

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
        gate_feature = (max_raw_cos - self.gate_center) / self.gate_width
        gate_logit = self.gate_slope * gate_feature + self.gate_bias
        match_idx = ret_logits.argmax(-1)
        return match_idx, gate_logit, ret_logits, max_raw_cos
