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

    Ranking can use the legacy trainable projected cosine or a bounded learned
    residual on top of frozen-DINO cosine.  The residual path starts exactly at
    the raw-DINO ordering instead of destroying that useful prior with a random
    low-dimensional projection.  The gate always uses the raw frozen-DINO
    maximum cosine because that is the statistic whose class separation and
    threshold were measured by the offline probe.
    """

    RANK_MODE_TO_CODE = {
        'projected': 0,
        'raw_residual': 1,
        'raw_temporal': 2,
    }
    CODE_TO_RANK_MODE = {value: key for key, value in RANK_MODE_TO_CODE.items()}
    # Keep the first temporal experiment deliberately small and interpretable.
    # Both sides of a historical candidate are already observed at inference.
    TEMPORAL_OFFSETS = (-2, -1, 1, 2)
    TEMPORAL_FEATURE_DIM = 13

    def __init__(
        self,
        dino_dim=1024,
        proj_dim=256,
        temp_init=0.07,
        gate_center=0.94,
        gate_width=0.04,
        gate_slope_init=1.6,
        gate_bias_init=0.0,
        rank_mode='projected',
        raw_temp_init=0.01,
        residual_max=0.25,
        temporal_topk=10,
        temporal_residual_max=0.02,
    ):
        super().__init__()
        if gate_width <= 0:
            raise ValueError(f'gate_width must be positive, got {gate_width}')
        if gate_slope_init <= 0:
            raise ValueError(
                f'gate_slope_init must be positive, got {gate_slope_init}'
            )
        if rank_mode not in self.RANK_MODE_TO_CODE:
            raise ValueError(
                'rank_mode must be projected/raw_residual/raw_temporal, got '
                f'{rank_mode!r}'
            )
        if not 0.005 <= float(raw_temp_init) <= 1.0:
            raise ValueError(
                'raw_temp_init must be in [0.005, 1.0], got '
                f'{raw_temp_init}'
            )
        if residual_max <= 0:
            raise ValueError(
                f'residual_max must be positive, got {residual_max}'
            )
        if int(temporal_topk) < 1:
            raise ValueError(
                f'temporal_topk must be positive, got {temporal_topk}'
            )
        if temporal_residual_max <= 0:
            raise ValueError(
                'temporal_residual_max must be positive, got '
                f'{temporal_residual_max}'
            )
        self.register_buffer(
            'rank_mode_code',
            torch.tensor(self.RANK_MODE_TO_CODE[rank_mode], dtype=torch.int64),
        )
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        # Preserve the frozen representation's cosine ordering at step zero.
        # The projections remain independent parameters and can diverge later.
        with torch.no_grad():
            self.proj_mem.weight.copy_(self.proj_goal.weight)
            self.proj_mem.bias.copy_(self.proj_goal.bias)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))
        # Raw retrieval is already substantially stronger than the learned
        # projection on the fixed held-out diagnostic.  Give it its own calibrated
        # temperature, then learn a per-projection-dimension residual.  Zero
        # residual weights make checkpoint migration behaviorally exact: the first
        # forward is pure raw-DINO retrieval, while every residual weight receives a
        # gradient immediately.
        self.raw_log_temp = nn.Parameter(
            torch.tensor(float(np.log(raw_temp_init)), dtype=torch.float32)
        )
        self.residual_weights = nn.Parameter(torch.zeros(proj_dim))
        self.register_buffer(
            'residual_max', torch.tensor(float(residual_max), dtype=torch.float32)
        )
        # The temporal reranker consumes only inference-safe score-curve context.
        # A zero linear form makes its initial output exactly zero while giving
        # every coefficient an immediate gradient.  The tanh bound prevents a
        # short fine-tune from destroying raw-DINO's strong shortlist ordering.
        self.temporal_weights = nn.Parameter(
            torch.zeros(self.TEMPORAL_FEATURE_DIM, dtype=torch.float32)
        )
        self.temporal_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            'temporal_topk', torch.tensor(int(temporal_topk), dtype=torch.int64)
        )
        self.register_buffer(
            'temporal_residual_max',
            torch.tensor(float(temporal_residual_max), dtype=torch.float32),
        )
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
    def rank_mode(self):
        """Checkpoint-persistent active ranking implementation."""
        code = int(self.rank_mode_code.detach().cpu())
        if code not in self.CODE_TO_RANK_MODE:
            raise ValueError(f'unsupported retrieval rank mode code {code}')
        return self.CODE_TO_RANK_MODE[code]

    @property
    def effective_gate_threshold(self):
        """Raw-cosine value whose revisit probability is exactly 0.5."""
        return self.gate_center - self.gate_width * self.gate_bias / self.gate_slope

    @property
    def rank_temperature(self):
        """Temperature used by the active ranking path."""
        if self.rank_mode in {'raw_residual', 'raw_temporal'}:
            return self.raw_log_temp.exp().clamp(0.005, 1.0)
        return self.log_temp.exp().clamp(0.01, 1.0)

    @property
    def residual_weight_abs_mean(self):
        """Bounded residual magnitude diagnostic in pre-cosine units."""
        if self.rank_mode == 'raw_temporal':
            return (
                self.temporal_weights.tanh().abs().mean()
                * self.temporal_residual_max
            )
        return self.residual_weights.tanh().abs().mean() * self.residual_max

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
        if has_old:
            if old_a_key not in upgraded or old_b_key not in upgraded:
                raise ValueError(
                    'legacy gate checkpoint must contain both gate_a and gate_b'
                )
            if log_slope_key in upgraded or bias_key in upgraded:
                raise ValueError(
                    'checkpoint mixes legacy and normalized gate parameters'
                )

            old_a = upgraded.pop(old_a_key)
            old_b = upgraded.pop(old_b_key)
            width = self.gate_width.detach().to(
                device=old_a.device, dtype=old_a.dtype
            )
            center = self.gate_center.detach().to(
                device=old_a.device, dtype=old_a.dtype
            )
            normalized_slope = old_a * width
            if not bool(torch.all(normalized_slope > 0)):
                raise ValueError(
                    'legacy gate_a must be positive to migrate to a monotonic gate'
                )
            upgraded[log_slope_key] = normalized_slope.log()
            upgraded[bias_key] = old_a * center + old_b
            upgraded[center_key] = center
            upgraded[width_key] = width

        # Checkpoints written before raw-residual ranking have none of these
        # tensors.  Add the complete zero-residual state, but never repair a
        # partially written new checkpoint: strict loading must reject that.
        ranking_names = (
            'rank_mode_code', 'raw_log_temp',
            'residual_weights', 'residual_max',
        )
        ranking_keys = tuple(f'{prefix}{name}' for name in ranking_names)
        present = [key for key in ranking_keys if key in upgraded]
        had_ranking_state = bool(present)
        if not present:
            current = self.state_dict()
            for name, key in zip(ranking_names, ranking_keys):
                upgraded[key] = current[name].detach().cpu().clone()

        temporal_names = (
            'temporal_weights', 'temporal_bias',
            'temporal_topk', 'temporal_residual_max',
        )
        temporal_keys = tuple(f'{prefix}{name}' for name in temporal_names)
        temporal_present = [key for key in temporal_keys if key in upgraded]
        if not temporal_present:
            # Version-1 ranking checkpoints persisted mode 0/1 but predate the
            # complete temporal namespace.  They can be upgraded unambiguously.
            # A mode-2 checkpoint missing this whole namespace is corrupt and is
            # intentionally left incomplete for strict loading to reject.
            mode_key = f'{prefix}rank_mode_code'
            saved_mode = upgraded.get(mode_key)
            saved_mode_code = (
                int(saved_mode.detach().cpu())
                if isinstance(saved_mode, torch.Tensor) else None
            )
            if not had_ranking_state or saved_mode_code in {0, 1}:
                current = self.state_dict()
                for name, key in zip(temporal_names, temporal_keys):
                    upgraded[key] = current[name].detach().cpu().clone()
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

    @staticmethod
    def raw_cosine(goal_cls, mem_cls):
        """Frozen-DINO cosine used by the calibrated gate and diagnostics."""
        raw_goal = F.normalize(goal_cls.float(), dim=-1)
        raw_mem = F.normalize(mem_cls.float(), dim=-1)
        return (raw_goal.unsqueeze(1) * raw_mem).sum(-1)

    def raw_match(self, goal_cls, mem_cls, cand_mask):
        """Select the raw-DINO top-1 candidate without changing rank logits."""
        raw_cos = self.raw_cosine(goal_cls, mem_cls)
        floor = torch.finfo(raw_cos.dtype).min
        return raw_cos.masked_fill(~cand_mask, floor).argmax(-1), raw_cos

    def raw_topk_mask(self, raw_cos, cand_mask):
        """Return the raw-DINO shortlist used by the temporal reranker."""
        if raw_cos.shape != cand_mask.shape:
            raise ValueError('raw_cos and cand_mask must have the same shape')
        width = raw_cos.shape[-1]
        if width < 1:
            return cand_mask.clone()
        topk = min(int(self.temporal_topk.detach().cpu()), width)
        floor = torch.finfo(raw_cos.dtype).min
        indices = raw_cos.masked_fill(~cand_mask, floor).topk(topk, -1).indices
        return torch.zeros_like(cand_mask).scatter(1, indices, True) & cand_mask

    def temporal_features(self, raw_cos, cand_mask):
        """Build label-free local score-curve features for every memory frame.

        Features are relative to each centre score, so the reranker cannot win by
        learning a second temperature or an episode-specific absolute threshold.
        No absolute frame index is included.  Invalid neighbours are represented by
        zero deltas plus explicit validity flags.
        """
        if raw_cos.shape != cand_mask.shape:
            raise ValueError('raw_cos and cand_mask must have the same shape')
        if raw_cos.ndim != 2:
            raise ValueError('raw_cos must have shape [batch, time]')
        batch, width = raw_cos.shape
        if width < 1:
            return raw_cos.new_zeros(batch, 0, self.TEMPORAL_FEATURE_DIM)

        timeline = torch.arange(width, device=raw_cos.device)
        centre = raw_cos
        deltas = []
        validities = []
        for offset in self.TEMPORAL_OFFSETS:
            neighbour_index = timeline + offset
            in_bounds = (neighbour_index >= 0) & (neighbour_index < width)
            safe_index = neighbour_index.clamp(0, width - 1)
            neighbour = raw_cos[:, safe_index]
            valid = in_bounds.unsqueeze(0) & cand_mask[:, safe_index]
            deltas.append((neighbour - centre) * valid)
            validities.append(valid)

        delta = torch.stack(deltas, -1)
        valid = torch.stack(validities, -1)
        valid_float = valid.to(delta.dtype)
        count = valid_float.sum(-1).clamp_min(1.0)
        mean_delta = delta.sum(-1) / count
        rms_delta = (delta.square().sum(-1) / count).sqrt()
        positive_inf = torch.finfo(delta.dtype).max
        negative_inf = torch.finfo(delta.dtype).min
        min_delta = delta.masked_fill(~valid, positive_inf).amin(-1)
        max_delta = delta.masked_fill(~valid, negative_inf).amax(-1)
        any_valid = valid.any(-1)
        min_delta = torch.where(any_valid, min_delta, torch.zeros_like(min_delta))
        max_delta = torch.where(any_valid, max_delta, torch.zeros_like(max_delta))

        floor = torch.finfo(raw_cos.dtype).min
        row_max = raw_cos.masked_fill(~cand_mask, floor).max(-1).values
        row_max = torch.where(
            cand_mask.any(-1), row_max, torch.zeros_like(row_max)
        )
        rank_gap = centre - row_max.unsqueeze(-1)
        features = torch.cat((
            delta,
            valid_float,
            mean_delta.unsqueeze(-1),
            rms_delta.unsqueeze(-1),
            min_delta.unsqueeze(-1),
            max_delta.unsqueeze(-1),
            rank_gap.unsqueeze(-1),
        ), dim=-1)
        if features.shape[-1] != self.TEMPORAL_FEATURE_DIM:
            raise RuntimeError(
                f'temporal feature width {features.shape[-1]} != '
                f'{self.TEMPORAL_FEATURE_DIM}'
            )
        return features

    def temporal_residual(self, raw_cos, cand_mask):
        """Compute a bounded residual on raw-DINO's Top-K candidates only."""
        features = self.temporal_features(raw_cos, cand_mask)
        score = features @ self.temporal_weights + self.temporal_bias
        residual = score.tanh() * self.temporal_residual_max
        return residual * self.raw_topk_mask(raw_cos, cand_mask)

    def forward(self, goal_cls, mem_cls, cand_mask):
        """Return match index, gate logit, ranking logits, and raw gate feature."""
        raw_cos = self.raw_cosine(goal_cls, mem_cls)

        if self.rank_mode == 'raw_residual':
            gq = F.normalize(self.proj_goal(goal_cls), dim=-1)
            mk = F.normalize(self.proj_mem(mem_cls), dim=-1)
            projected_interaction = gq.unsqueeze(1) * mk
            residual = (
                projected_interaction * self.residual_weights.tanh()
            ).sum(-1) * self.residual_max
            rank_cos = raw_cos + residual
        elif self.rank_mode == 'raw_temporal':
            rank_cos = raw_cos + self.temporal_residual(raw_cos, cand_mask)
        else:
            gq = F.normalize(self.proj_goal(goal_cls), dim=-1)
            mk = F.normalize(self.proj_mem(mem_cls), dim=-1)
            rank_cos = (gq.unsqueeze(1) * mk).sum(-1)
        temp = self.rank_temperature

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
