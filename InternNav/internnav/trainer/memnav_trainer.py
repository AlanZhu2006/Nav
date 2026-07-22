import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.model.basemodel.memnav.route_sketch import route_direction_targets
from internnav.trainer.base import BaseTrainer


def project_auxiliary_gradients(
    action_gradients,
    auxiliary_gradients,
    max_norm_ratio,
    eps=1e-12,
):
    """Project conflicting auxiliary gradients and cap their global norm.

    The returned gradients have a non-negative dot product with the action
    gradients.  Their norm is at most ``max_norm_ratio`` times the action norm.
    Both operations use one norm/dot product over the complete parameter group,
    rather than projecting each tensor independently.
    """
    if len(action_gradients) != len(auxiliary_gradients):
        raise ValueError('action and auxiliary gradient lists must have equal length')
    if not action_gradients:
        raise ValueError('at least one gradient tensor is required')
    if max_norm_ratio < 0.0:
        raise ValueError('max_norm_ratio must be non-negative')
    for action_gradient, auxiliary_gradient in zip(
        action_gradients, auxiliary_gradients
    ):
        if action_gradient.shape != auxiliary_gradient.shape:
            raise ValueError('paired action/auxiliary gradient shapes must match')

    action_norm_sq = sum(
        gradient.float().square().sum() for gradient in action_gradients
    )
    auxiliary_norm_sq = sum(
        gradient.float().square().sum() for gradient in auxiliary_gradients
    )
    dot = sum(
        (action.float() * auxiliary.float()).sum()
        for action, auxiliary in zip(action_gradients, auxiliary_gradients)
    )
    action_norm = action_norm_sq.sqrt()
    auxiliary_norm = auxiliary_norm_sq.sqrt()
    cosine_denominator = action_norm * auxiliary_norm
    raw_cosine = torch.where(
        cosine_denominator > eps,
        dot / cosine_denominator.clamp_min(eps),
        dot.new_zeros(()),
    ).clamp(-1.0, 1.0)

    # Removing only the negative parallel component makes the auxiliary update
    # first-order non-adversarial to the action objective.
    projection_coefficient = torch.minimum(dot, dot.new_zeros(())) / (
        action_norm_sq.clamp_min(eps)
    )
    projected = [
        auxiliary - projection_coefficient.to(auxiliary.dtype) * action
        for action, auxiliary in zip(action_gradients, auxiliary_gradients)
    ]
    projected_norm = sum(
        gradient.float().square().sum() for gradient in projected
    ).sqrt()
    max_auxiliary_norm = float(max_norm_ratio) * action_norm
    cap_scale = torch.minimum(
        projected_norm.new_ones(()),
        max_auxiliary_norm / projected_norm.clamp_min(eps),
    )
    corrected = [
        gradient * cap_scale.to(gradient.dtype) for gradient in projected
    ]
    corrected_norm = projected_norm * cap_scale
    diagnostics = {
        'raw_cosine': raw_cosine,
        'raw_norm_ratio': auxiliary_norm / action_norm.clamp_min(eps),
        'corrected_norm_ratio': corrected_norm / action_norm.clamp_min(eps),
        'cap_scale': cap_scale,
        'conflict': (dot < 0.0).to(dot.dtype),
    }
    return corrected, diagnostics


class MemNavTrainer(BaseTrainer):
    """Trainer for the frozen LingBot front-end and trainable MemNav policy.

    The objective has diffusion action, retrieval ranking, semantic revisit gate,
    scale-invariant translation direction, and geometric pose-reliability terms.
    Semantic match existence and geometric trust are supervised separately.
    Metric x/y and camera-rotation errors are diagnostics only.
    """

    POSE_RELIABILITY_FEATURE_NAMES = (
        'range_code',
        'anchor_gap_code',
        'step_scale_drift',
        'goal_anchor_range_code',
        'vertical_ratio',
        'rotation_tilt',
        'semantic_score_z_scaled',
    )

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # HF otherwise classifies this custom dictionary batch as unlabeled in
        # prediction_step and bypasses compute_loss during scheduled evaluation.
        self.label_names = ['batch_labels']
        self.config = config
        self.w_retr = getattr(config.il, 'w_retrieval', 1.0)
        self.retrieval_denominator = getattr(
            config.il, 'retrieval_denominator', 'positive_negative'
        )
        self.retrieval_lr_multiplier = float(
            getattr(config.il, 'retrieval_lr_multiplier', 1.0)
        )
        self.retrieval_only = bool(
            getattr(config.il, 'retrieval_only', False)
        )
        self.retrieval_margin_cosine = float(
            getattr(config.il, 'retrieval_margin_cosine', 0.005)
        )
        self.retrieval_margin_weight = float(
            getattr(
                config.il,
                'retrieval_margin_weight',
                1.0 if self.retrieval_only else 0.0,
            )
        )
        self.w_gate = getattr(config.il, 'w_gate', 1.0)
        self.w_aux = getattr(config.il, 'w_aux_direction', 0.2)
        self.w_route = float(getattr(config.il, 'w_route_direction', 0.0))
        self.use_route_sketch = bool(
            getattr(config.il, 'use_route_sketch', False)
        )
        self.route_horizons = tuple(
            int(value) for value in getattr(
                config.il, 'route_horizons', (2, 8, 24)
            )
        )
        self.route_curvature_emphasis = float(
            getattr(config.il, 'route_curvature_emphasis', 0.0)
        )
        self.route_lr_multiplier = float(
            getattr(config.il, 'route_lr_multiplier', 10.0)
        )
        self.w_aux_range = float(getattr(config.il, 'w_aux_range', 0.0))
        self.aux_range_beta = float(getattr(config.il, 'aux_range_beta', 0.1))
        self.aux_range_grad_cap_ratio = float(
            getattr(config.il, 'aux_range_grad_cap_ratio', 0.0)
        )
        self.w_pose_reliability = getattr(config.il, 'w_pose_reliability', 0.2)
        self.anchor_tf_start = float(
            getattr(config.il, 'anchor_teacher_forcing_start', 1.0)
        )
        self.anchor_tf_end = float(
            getattr(config.il, 'anchor_teacher_forcing_end', 1.0)
        )
        self.anchor_tf_decay_steps = int(
            getattr(config.il, 'anchor_teacher_forcing_decay_steps', 0)
        )
        if self.w_aux_range < 0.0:
            raise ValueError('w_aux_range must be non-negative')
        if self.w_retr < 0.0:
            raise ValueError('w_retrieval must be non-negative')
        if self.retrieval_denominator not in {
            'positive_negative', 'all_candidates'
        }:
            raise ValueError(
                'retrieval_denominator must be positive_negative/all_candidates, '
                f'got {self.retrieval_denominator!r}'
            )
        if self.retrieval_lr_multiplier <= 0.0:
            raise ValueError('retrieval_lr_multiplier must be positive')
        if self.retrieval_margin_cosine < 0.0:
            raise ValueError('retrieval_margin_cosine must be non-negative')
        if self.retrieval_margin_weight < 0.0:
            raise ValueError('retrieval_margin_weight must be non-negative')
        if self.retrieval_only:
            rank_mode = getattr(
                config.il, 'retrieval_rank_mode', 'projected'
            )
            if rank_mode != 'raw_temporal':
                raise ValueError(
                    'retrieval_only requires retrieval_rank_mode=raw_temporal'
                )
            if self.retrieval_denominator != 'all_candidates':
                raise ValueError(
                    'retrieval_only requires retrieval_denominator=all_candidates'
                )
        if self.w_route < 0.0:
            raise ValueError('w_route_direction must be non-negative')
        if self.w_route > 0.0 and not self.use_route_sketch:
            raise ValueError(
                'positive w_route_direction requires use_route_sketch=True'
            )
        if self.route_curvature_emphasis < 0.0:
            raise ValueError('route_curvature_emphasis must be non-negative')
        if self.route_lr_multiplier <= 0.0:
            raise ValueError('route_lr_multiplier must be positive')
        if self.aux_range_beta <= 0.0:
            raise ValueError('aux_range_beta must be positive')
        if self.aux_range_grad_cap_ratio < 0.0:
            raise ValueError('aux_range_grad_cap_ratio must be non-negative')
        if self.aux_range_grad_cap_ratio > 0.0 and self.w_aux_range == 0.0:
            raise ValueError(
                'aux_range_grad_cap_ratio requires a positive w_aux_range'
            )
        if (self.aux_range_grad_cap_ratio > 0.0
                and dist.is_initialized() and dist.get_world_size() > 1):
            raise ValueError(
                'action-safe range gradients currently require single-process '
                'training; the inner gradient queries are not DDP-reducer safe'
            )
        if not (0.0 <= self.anchor_tf_start <= 1.0):
            raise ValueError('anchor_teacher_forcing_start must be in [0, 1]')
        if not (0.0 <= self.anchor_tf_end <= 1.0):
            raise ValueError('anchor_teacher_forcing_end must be in [0, 1]')
        if self.anchor_tf_decay_steps < 0:
            raise ValueError('anchor_teacher_forcing_decay_steps must be non-negative')
        if (self.anchor_tf_start != self.anchor_tf_end
                and self.anchor_tf_decay_steps == 0):
            raise ValueError(
                'anchor_teacher_forcing_decay_steps must be positive when the '
                'start and end probabilities differ'
            )
        self._metric_accumulators = {'train': {}, 'eval': {}}
        self.eval_seed = int(getattr(config.il, 'eval_seed', 0))
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if self.retrieval_only:
            trainable_suffixes = (
                'retrieval.temporal_weights',
                'retrieval.temporal_bias',
            )
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.endswith(trainable_suffixes))
            trainable = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            trainable_names = {name for name, _ in trainable}
            if len(trainable_names) != 2 or not all(
                any(name.endswith(suffix) for name in trainable_names)
                for suffix in trainable_suffixes
            ):
                raise RuntimeError(
                    'retrieval-only freeze contract expected exactly temporal '
                    f'weights+bias, got {sorted(trainable_names)}'
                )
            trainable_count = sum(parameter.numel() for _, parameter in trainable)
            if trainable_count != 14:
                raise RuntimeError(
                    'retrieval-only freeze contract expected 14 scalars, got '
                    f'{trainable_count}'
                )
        self.model_device = model.device
        # Known local-frame convention candidate. We report both raw and converted
        # camera-rotation errors so this correction can never hide a bad pose stream.
        self._C_rot = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rank = dist.get_rank() if dist.is_initialized() else 0
        fingerprint = self._dataset_fingerprint(self.train_dataset) or '<unavailable>'
        print(f'[Rank {rank}] Model device: {self.model_device}')
        print(f'[Rank {rank}] Dataset fingerprint: {fingerprint}')
        if self.retrieval_only:
            print(
                f'[Rank {rank}] Retrieval-only: 14 trainable scalars; '
                f'cosine margin={self.retrieval_margin_cosine}; '
                f'margin weight={self.retrieval_margin_weight}'
            )

    def create_optimizer(self):
        """Give small calibration heads their own learning-rate groups.

        The raw-cosine feature is frozen and already discriminative; only two
        scalar calibration parameters need to move.  A multiplier avoids tying
        their convergence to the much slower policy-wide cosine schedule, while
        zero weight decay prevents the slope from being pulled toward a flat gate.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        multiplier = float(getattr(self.config.il, 'gate_lr_multiplier', 10.0))
        pose_multiplier = float(
            getattr(self.config.il, 'pose_reliability_lr_multiplier', 5.0)
        )
        route_multiplier = self.route_lr_multiplier
        retrieval_multiplier = self.retrieval_lr_multiplier
        if multiplier <= 0:
            raise ValueError(
                f'gate_lr_multiplier must be positive, got {multiplier}'
            )
        if pose_multiplier <= 0:
            raise ValueError(
                'pose_reliability_lr_multiplier must be positive, got '
                f'{pose_multiplier}'
            )
        gate_suffixes = (
            'retrieval.gate_log_slope',
            'retrieval.gate_bias',
        )
        named_trainable = [
            (name, parameter)
            for name, parameter in opt_model.named_parameters()
            if parameter.requires_grad
        ]
        gate_names = {
            name for name, _ in named_trainable
            if name.endswith(gate_suffixes)
        }
        pose_reliability_names = {
            name for name, _ in named_trainable
            if '.pose_encoder.reliability_head.' in name
        }
        route_names = {
            name for name, _ in named_trainable if '.route_sketch.' in name
        }
        retrieval_names = {
            name for name, _ in named_trainable
            if '.retrieval.' in name and name not in gate_names
        }
        special_names = (
            gate_names | retrieval_names | pose_reliability_names | route_names
        )
        # Tiny test models and non-MemNav reuse should retain BaseTrainer's exact
        # optimizer behavior.
        if not special_names:
            return super().create_optimizer()

        decay_names = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
        decay_names = {name for name in decay_names if 'bias' not in name}
        regular_decay = [
            parameter for name, parameter in named_trainable
            if name not in special_names and name in decay_names
        ]
        regular_no_decay = [
            parameter for name, parameter in named_trainable
            if name not in special_names and name not in decay_names
        ]
        gate_parameters = [
            parameter for name, parameter in named_trainable if name in gate_names
        ]
        retrieval_decay_parameters = [
            parameter for name, parameter in named_trainable
            if name in retrieval_names and name in decay_names
        ]
        retrieval_no_decay_parameters = [
            parameter for name, parameter in named_trainable
            if name in retrieval_names and name not in decay_names
        ]
        pose_reliability_parameters = [
            parameter for name, parameter in named_trainable
            if name in pose_reliability_names
        ]
        route_decay_parameters = [
            parameter for name, parameter in named_trainable
            if name in route_names
            and name in decay_names
            and not name.endswith('residual_scale')
        ]
        route_no_decay_parameters = [
            parameter for name, parameter in named_trainable
            if name in route_names
            and (
                name not in decay_names or name.endswith('residual_scale')
            )
        ]
        optimizer_groups = [
            {'params': regular_decay, 'weight_decay': self.args.weight_decay},
            {'params': regular_no_decay, 'weight_decay': 0.0},
            {
                'params': gate_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate * multiplier,
                'memnav_group': 'gate_calibration',
                'lr_multiplier': multiplier,
            },
            {
                'params': retrieval_decay_parameters,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate * retrieval_multiplier,
                'memnav_group': 'retrieval_rank_decay',
                'lr_multiplier': retrieval_multiplier,
            },
            {
                'params': retrieval_no_decay_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate * retrieval_multiplier,
                'memnav_group': 'retrieval_rank_no_decay',
                'lr_multiplier': retrieval_multiplier,
            },
            {
                'params': pose_reliability_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate * pose_multiplier,
                'memnav_group': 'pose_reliability_calibration',
                'lr_multiplier': pose_multiplier,
            },
            {
                'params': route_decay_parameters,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate * route_multiplier,
                'memnav_group': 'route_sketch_decay',
                'lr_multiplier': route_multiplier,
            },
            {
                'params': route_no_decay_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate * route_multiplier,
                'memnav_group': 'route_sketch_no_decay',
                'lr_multiplier': route_multiplier,
            },
        ]
        optimizer_groups = [group for group in optimizer_groups if group['params']]
        optimizer_cls, optimizer_kwargs = (
            transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        )
        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

    def _accumulate(self, name, mean_value, weight=1.0, phase='train'):
        """Accumulate a weighted scalar mean on-device until HF's log boundary."""
        value = mean_value.detach().float()
        if torch.is_tensor(weight):
            weight = weight.detach().to(device=value.device, dtype=torch.float32)
        else:
            weight = value.new_tensor(float(weight))
        numerator = value * weight
        accumulator = self._metric_accumulators[phase]
        if name not in accumulator:
            accumulator[name] = [numerator, weight]
        else:
            old_num, old_den = accumulator[name]
            accumulator[name] = [old_num + numerator, old_den + weight]

    @staticmethod
    def _masked_mean(value, mask):
        weight = mask.float()
        return (value * weight).sum() / weight.sum().clamp(min=1.0)

    @staticmethod
    def _dataset_fingerprint(dataset):
        """Prefer a subset-specific fingerprint, then fall back to its parent."""
        if dataset is None:
            return None
        direct = getattr(dataset, 'dataset_fingerprint', None)
        if direct:
            return direct
        return getattr(getattr(dataset, 'dataset', None), 'dataset_fingerprint', None)

    def _training_objective_metadata(self):
        return {
            'w_retrieval': float(self.w_retr),
            'retrieval_denominator': self.retrieval_denominator,
            'retrieval_lr_multiplier': self.retrieval_lr_multiplier,
            'retrieval_only': self.retrieval_only,
            'retrieval_margin_cosine': self.retrieval_margin_cosine,
            'retrieval_margin_weight': self.retrieval_margin_weight,
            'retrieval_rank_mode': getattr(
                self.config.il, 'retrieval_rank_mode', 'projected'
            ),
            'retrieval_raw_temp_init': float(getattr(
                self.config.il, 'retrieval_raw_temp_init', 0.01
            )),
            'retrieval_residual_max': float(getattr(
                self.config.il, 'retrieval_residual_max', 0.25
            )),
            'retrieval_temporal_topk': int(getattr(
                self.config.il, 'retrieval_temporal_topk', 10
            )),
            'retrieval_temporal_residual_max': float(getattr(
                self.config.il, 'retrieval_temporal_residual_max', 0.02
            )),
            'retrieval_anchor_min_frame': getattr(
                self.config.il, 'retrieval_anchor_min_frame', None
            ),
            'w_gate': float(self.w_gate),
            'w_aux_direction': float(self.w_aux),
            'w_route_direction': float(self.w_route),
            'use_route_sketch': bool(self.use_route_sketch),
            'route_horizons': list(self.route_horizons),
            'route_curvature_emphasis': self.route_curvature_emphasis,
            'route_lr_multiplier': self.route_lr_multiplier,
            'w_aux_range': float(self.w_aux_range),
            'aux_range_beta': float(self.aux_range_beta),
            'aux_range_grad_cap_ratio': float(
                self.aux_range_grad_cap_ratio
            ),
            'w_pose_reliability': float(self.w_pose_reliability),
            'anchor_teacher_forcing_start': float(self.anchor_tf_start),
            'anchor_teacher_forcing_end': float(self.anchor_tf_end),
            'anchor_teacher_forcing_decay_steps': int(
                self.anchor_tf_decay_steps
            ),
        }

    def _anchor_teacher_forcing_probability(self):
        """Linear train-time schedule; defaults to the legacy constant 1.0."""
        if self.anchor_tf_start == self.anchor_tf_end:
            return self.anchor_tf_start
        progress = min(
            max(float(self.state.global_step), 0.0)
            / float(self.anchor_tf_decay_steps),
            1.0,
        )
        return self.anchor_tf_start + progress * (
            self.anchor_tf_end - self.anchor_tf_start
        )

    def _action_safe_range_correction(self, model, action_loss, range_loss):
        """Replace the range gradient on the shared adapter without changing loss value."""
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if '.revisit_merge.rel_adapter.' in name and parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(
                'range gradient control could not find revisit_merge.rel_adapter'
            )
        weighted_range_loss = self.w_aux_range * range_loss
        action_gradients = torch.autograd.grad(
            action_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        range_gradients = torch.autograd.grad(
            weighted_range_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        safe_action_gradients = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, action_gradients)
        ]
        safe_range_gradients = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, range_gradients)
        ]
        corrected, diagnostics = project_auxiliary_gradients(
            safe_action_gradients,
            safe_range_gradients,
            self.aux_range_grad_cap_ratio,
        )
        # This scalar is exactly zero in the forward pass.  Its backward replaces
        # the raw weighted-range gradient with the projected/capped gradient.
        correction = action_loss.new_zeros(())
        for parameter, raw_gradient, corrected_gradient in zip(
            parameters, safe_range_gradients, corrected
        ):
            gradient_delta = corrected_gradient - raw_gradient
            correction = correction + (
                (parameter - parameter.detach()) * gradient_delta
            ).sum()
        return correction, diagnostics

    @staticmethod
    def _sample_anchor_teacher_mask(pos_mask, probability, seed=None):
        """Bernoulli teacher exposure without perturbing diffusion's global RNG."""
        eligible = pos_mask.bool().any(-1)
        probability = float(probability)
        if probability <= 0.0:
            return torch.zeros_like(eligible)
        if probability >= 1.0:
            return eligible
        generator = None
        if seed is not None:
            generator = torch.Generator(device=eligible.device)
            generator.manual_seed(int(seed))
        draw = torch.rand(
            eligible.shape,
            device=eligible.device,
            dtype=torch.float32,
            generator=generator,
        )
        return eligible & (draw < probability)

    def _compute_retrieval_only_loss(
        self, model, inputs, return_outputs=False
    ):
        """Train the bounded temporal residual without policy-side gradients.

        Listwise multi-positive likelihood provides dense supervision.  A second
        hinge term is measured in pre-temperature cosine units and directly
        aligns optimization with strict inference Top-1: the best positive must
        beat the best gray-or-negative candidate by the configured margin.
        """
        phase = 'train' if model.training else 'eval'
        fwd = model(inputs)
        ret_logits = fwd['ret_logits']
        device = ret_logits.device
        positive = inputs['batch_pos_mask'].to(device).bool()
        negative = inputs['batch_neg_mask'].to(device).bool()
        candidate = inputs['batch_cand_mask'].to(device).bool()
        if ret_logits.shape != positive.shape:
            raise ValueError('retrieval logits/masks must have identical shapes')
        if bool((positive & negative).any()):
            raise ValueError('retrieval positive/negative masks must be disjoint')
        if bool(((positive | negative) & ~candidate).any()):
            raise ValueError(
                'retrieval positive/negative masks must be candidate subsets'
            )

        nonpositive = candidate & ~positive
        rows = positive.any(-1) & nonpositive.any(-1)
        row_count = rows.float().sum()
        floor = torch.finfo(ret_logits.dtype).min
        positive_lse = ret_logits.masked_fill(
            ~positive, floor
        ).logsumexp(-1)
        candidate_lse = ret_logits.masked_fill(
            ~candidate, floor
        ).logsumexp(-1)
        listwise_loss = (
            candidate_lse[rows] - positive_lse[rows]
        ).sum() / row_count.clamp_min(1.0)

        temperature = fwd['retrieval_rank_temperature'].to(
            device=device, dtype=ret_logits.dtype
        )
        rank_cosine = ret_logits * temperature
        best_positive = rank_cosine.masked_fill(
            ~positive, floor
        ).max(-1).values
        best_nonpositive = rank_cosine.masked_fill(
            ~nonpositive, floor
        ).max(-1).values
        strict_margin = best_positive - best_nonpositive
        margin_loss = F.relu(
            self.retrieval_margin_cosine - strict_margin[rows]
        ).sum() / row_count.clamp_min(1.0)
        loss = listwise_loss + self.retrieval_margin_weight * margin_loss

        match = fwd['match_idx'].to(device).long()
        selected_positive = positive.gather(1, match[:, None]).squeeze(1)
        selected_negative = negative.gather(1, match[:, None]).squeeze(1)
        selected_gray = (
            candidate.gather(1, match[:, None]).squeeze(1)
            & ~selected_positive
            & ~selected_negative
        )
        strict_top1 = self._masked_mean(selected_positive.float(), rows)
        negative_fraction = self._masked_mean(
            selected_negative.float(), rows
        )
        gray_fraction = self._masked_mean(selected_gray.float(), rows)
        strict_margin_mean = self._masked_mean(strict_margin, rows)

        def recall_at(k):
            width = ret_logits.shape[-1]
            count = min(int(k), width)
            top_indices = ret_logits.topk(count, -1).indices
            top_positive = positive.gather(1, top_indices).any(-1)
            return self._masked_mean(top_positive.float(), rows)

        recall_5 = recall_at(5)
        recall_10 = recall_at(10)
        residual_abs_mean = fwd['retrieval_residual_abs_mean']
        outputs = {
            'loss': loss,
            'retrieval_loss': listwise_loss,
            'retrieval_margin_loss': margin_loss,
            'retrieval_strict_margin_cosine': strict_margin_mean,
            'retrieval_strict_top1': strict_top1,
            'retrieval_recall_at_5': recall_5,
            'retrieval_recall_at_10': recall_10,
            'seen_match_negative_fraction': negative_fraction,
            'seen_match_gray_fraction': gray_fraction,
            'retrieval_rank_temperature': temperature,
            'retrieval_residual_abs_mean': residual_abs_mean,
        }
        for name, value in outputs.items():
            if name == 'loss':
                continue
            self._accumulate(name, value, row_count, phase)
        self._accumulate('rank_row_fraction', row_count / rows.numel(), rows.numel(), phase)
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.retrieval_only:
            return self._compute_retrieval_only_loss(
                model, inputs, return_outputs=return_outputs
            )
        dev = next(model.parameters()).device
        phase = 'train' if model.training else 'eval'
        anchor_tf_probability = (
            self._anchor_teacher_forcing_probability() if model.training else 0.0
        )
        model_inputs = inputs
        if model.training and inputs.get('batch_pos_mask') is not None:
            process_rank = dist.get_rank() if dist.is_initialized() else 0
            anchor_mask_seed = (
                int(self.args.seed)
                + 1_000_003 * int(self.state.global_step)
                + 97 * process_rank
            )
            model_inputs = dict(inputs)
            model_inputs['anchor_teacher_forcing_mask'] = (
                self._sample_anchor_teacher_mask(
                    inputs['batch_pos_mask'],
                    anchor_tf_probability,
                    seed=anchor_mask_seed,
                )
            )
        fwd = model(model_inputs)
        stratified_accumulations = []

        # Diffusion action objective. Keep per-coordinate errors so W&B can show
        # whether x, y, or wrapped theta is the coordinate that stopped learning.
        action_sq = (fwd['noise_pred'] - fwd['noise']).square()
        action_axis_mse = action_sq.mean(dim=(0, 1))
        action_loss = action_axis_mse.mean()
        diffusion_noise = fwd['noise']
        diffusion_noise_mean = diffusion_noise.mean()
        diffusion_noise_sq_mean = diffusion_noise.square().mean()
        diffusion_timesteps = fwd.get('timesteps')
        action_target = inputs['batch_labels'].to(dev)
        action_target_mean = action_target.mean(dim=(0, 1))
        action_target_sq_mean = action_target.square().mean(dim=(0, 1))

        # Multi-horizon route supervision uses only label-side future actions.
        # The model forward path that produced route_direction consumed solely
        # inference-available memory and goal inputs.
        route_direction_loss = action_loss.new_zeros(())
        route_direction_error_deg = None
        route_target_curvature_deg = None
        route_prediction_curvature_deg = None
        route_valid = None
        route_count = action_loss.new_zeros(())
        route_prediction = fwd.get('route_direction')
        if route_prediction is not None:
            route_target, route_valid = route_direction_targets(
                action_target, self.route_horizons
            )
            if route_prediction.shape != route_target.shape:
                raise ValueError(
                    'route prediction/target shape mismatch: '
                    f'{tuple(route_prediction.shape)} != '
                    f'{tuple(route_target.shape)}'
                )
            route_cosine = (
                route_prediction * route_target
            ).sum(dim=-1).clamp(-1.0, 1.0)
            target_curvature_cosine = (
                route_target[:, 0] * route_target[:, -1]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            prediction_curvature_cosine = (
                route_prediction[:, 0] * route_prediction[:, -1]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            curvature_valid = route_valid[:, 0] & route_valid[:, -1]
            target_curvature = torch.where(
                curvature_valid,
                1.0 - target_curvature_cosine,
                torch.zeros_like(target_curvature_cosine),
            )
            curvature_weight = 1.0 + self.route_curvature_emphasis * (
                target_curvature
            ).unsqueeze(-1)
            weighted_valid = route_valid.float() * curvature_weight
            route_count = weighted_valid.sum()
            route_direction_loss = (
                (1.0 - route_cosine) * weighted_valid
            ).sum() / weighted_valid.sum().clamp(min=1.0)
            route_direction_error_deg = torch.rad2deg(
                torch.arccos(route_cosine)
            )
            route_target_curvature_deg = torch.rad2deg(
                torch.arccos(target_curvature_cosine)
            )
            route_prediction_curvature_deg = torch.rad2deg(
                torch.arccos(prediction_curvature_cosine)
            )
        elif self.w_route > 0.0:
            raise RuntimeError(
                'route direction loss is enabled but model emitted no route sketch'
            )

        # Decoupled retrieval: rank a true co-visible frame on revisit rows, and
        # classify match existence on every row. This avoids an always-null shortcut.
        ret_logits = fwd['ret_logits']
        gate_logit = fwd['gate_logit']
        pos = inputs['batch_pos_mask'].to(dev).bool()
        neg = inputs['batch_neg_mask'].to(dev).bool()
        candidate_input = inputs.get('batch_cand_mask')
        if candidate_input is None:
            if self.retrieval_denominator == 'all_candidates':
                raise ValueError(
                    'all_candidates retrieval loss requires batch_cand_mask'
                )
            # Legacy smoke/test batches predate an explicit candidate mask; for
            # the legacy objective its exact structural support is pos | neg.
            cand = pos | neg
        else:
            cand = candidate_input.to(dev).bool()
        is_rev = inputs['batch_is_revisit'].to(dev).float()
        if bool((pos & neg).any()):
            raise ValueError('retrieval positive/negative masks must be disjoint')
        if bool(((pos | neg) & ~cand).any()):
            raise ValueError(
                'retrieval positive/negative masks must be candidate subsets'
            )
        neg_inf = torch.finfo(ret_logits.dtype).min
        lse_p = ret_logits.masked_fill(~pos, neg_inf).logsumexp(-1)

        def rank_loss_over(denominator):
            lse_denominator = ret_logits.masked_fill(
                ~denominator, neg_inf
            ).logsumexp(-1)
            rows = pos.any(-1) & (denominator & ~pos).any(-1)
            count = rows.float().sum()
            # Index first. On novel rows lse_p is the finite dtype floor;
            # subtracting it before masking can poison mixed-precision backward.
            value = (
                lse_denominator[rows] - lse_p[rows]
            ).sum() / count.clamp(min=1.0)
            return value, count, rows

        retrieval_hard_loss, hard_rank_count, _ = rank_loss_over(
            pos | neg
        )
        retrieval_all_candidate_loss, all_rank_count, _ = (
            rank_loss_over(cand)
        )
        if self.retrieval_denominator == 'all_candidates':
            rank_loss = retrieval_all_candidate_loss
            rank_count = all_rank_count
        else:
            rank_loss = retrieval_hard_loss
            rank_count = hard_rank_count
        retrieval_rank_temperature = fwd.get(
            'retrieval_rank_temperature', ret_logits.new_tensor(1.0)
        )
        retrieval_residual_abs_mean = fwd.get(
            'retrieval_residual_abs_mean', ret_logits.new_zeros(())
        )

        n_rev = is_rev.sum()
        n_novel = (1.0 - is_rev).sum()
        B = float(is_rev.numel())
        # Do not estimate class weights from a four-sample batch: that makes the
        # objective itself jump between batches (all-revisit previously used 0.1,
        # mixed batches used a different value). The dataset is approximately
        # balanced and fixed validation reports both class recalls explicitly.
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, is_rev)

        # LingBot translations have a per-sequence canonical scale. Supervise
        # direction plus a gauge-normalized, compressed range coordinate; raw
        # metric x/y remains diagnostic only.  When scheduled live retrieval picks
        # a known-negative anchor, do not ask the shared pose adapter to repair that
        # semantically wrong measurement.  The action and reliability objectives
        # still see it, which is precisely the desired train/eval exposure.
        pred_xy = fwd['aux_pose']
        gt_xy = inputs['batch_goal_rel_pose'][..., :2].to(dev)
        gt_norm = torch.linalg.norm(gt_xy, dim=-1)
        pred_norm = torch.linalg.norm(pred_xy, dim=-1)
        pred_unit = pred_xy / pred_norm.clamp(min=0.25).unsqueeze(-1)
        gt_unit = gt_xy / gt_norm.clamp(min=1e-4).unsqueeze(-1)
        direction_cos = (pred_unit * gt_unit).sum(-1).clamp(-1.0, 1.0)
        metric_pred_unit = pred_xy / pred_norm.clamp(min=1e-6).unsqueeze(-1)
        metric_direction_cos = (
            metric_pred_unit * gt_unit
        ).sum(-1).clamp(-1.0, 1.0)
        anchor_idx = fwd['anchor_idx'].to(dev).long()
        anchor_positive = pos.gather(1, anchor_idx.unsqueeze(1)).squeeze(1)
        revisit_pose_valid = (is_rev > 0.5) & (gt_norm > 1e-4)
        aux_valid = revisit_pose_valid & anchor_positive
        aux_count = aux_valid.float().sum()
        aux_direction_loss = (
            (1.0 - direction_cos) * aux_valid
        ).sum() / aux_count.clamp(min=1.0)

        pred_range_code = fwd['aux_range_code']
        target_range_code = inputs['batch_goal_range_code'].to(dev)
        finite_range = torch.isfinite(pred_range_code) & torch.isfinite(target_range_code)
        range_valid = aux_valid & finite_range
        range_count = range_valid.float().sum()
        safe_range_target = torch.where(
            finite_range, target_range_code, pred_range_code.detach()
        )
        range_per_row = F.smooth_l1_loss(
            pred_range_code,
            safe_range_target,
            reduction='none',
            beta=self.aux_range_beta,
        )
        aux_range_loss = (
            range_per_row * range_valid
        ).sum() / range_count.clamp(min=1.0)

        # Calibrate whether a semantically correct revisit pose is geometrically
        # useful.  The target is continuous raw-bearing quality: 1 for agreement,
        # 0 at/after 90 degrees.  It supervises only revisit rows and is detached so
        # the confidence head cannot improve its label by changing the pose code.
        raw_pose_direction = fwd['raw_pose_direction']
        raw_direction_cos = (raw_pose_direction * gt_unit).sum(-1).clamp(-1.0, 1.0)
        pose_quality_target = raw_direction_cos.clamp(0.0, 1.0).detach()
        pose_reliability = fwd['pose_reliability'].clamp(1e-5, 1.0 - 1e-5)
        pose_reliability_per_row = F.binary_cross_entropy(
            pose_reliability, pose_quality_target, reduction='none'
        )
        reliability_count = revisit_pose_valid.float().sum()
        pose_reliability_loss = (
            pose_reliability_per_row * revisit_pose_valid
        ).sum() / reliability_count.clamp(min=1.0)

        range_gradient_correction = action_loss.new_zeros(())
        range_gradient_diagnostics = {
            'raw_cosine': action_loss.new_zeros(()),
            'raw_norm_ratio': action_loss.new_zeros(()),
            'corrected_norm_ratio': action_loss.new_zeros(()),
            'cap_scale': action_loss.new_ones(()),
            'conflict': action_loss.new_zeros(()),
        }
        range_gradient_control_enabled = bool(
            model.training
            and self.w_aux_range > 0.0
            and self.aux_range_grad_cap_ratio > 0.0
            and bool(range_count.detach() > 0.0)
        )
        if range_gradient_control_enabled:
            (
                range_gradient_correction,
                range_gradient_diagnostics,
            ) = self._action_safe_range_correction(
                model, action_loss, aux_range_loss
            )

        loss = (
            action_loss
            + self.w_retr * rank_loss
            + self.w_gate * gate_loss
            + self.w_aux * aux_direction_loss
            + self.w_route * route_direction_loss
            + self.w_aux_range * aux_range_loss
            + self.w_pose_reliability * pose_reliability_loss
            + range_gradient_correction
        )

        with torch.no_grad():
            rev_mask = is_rev > 0.5
            novel_mask = ~rev_mask
            gate_prob = torch.sigmoid(gate_logit)
            effective_gate = fwd['effective_revisit_gate']
            anchor_teacher_forced = fwd['anchor_teacher_forced'].to(dev).bool()
            anchor_positive_fraction = self._masked_mean(anchor_positive, rev_mask)
            anchor_teacher_forced_fraction = self._masked_mean(
                anchor_teacher_forced, rev_mask
            )
            gate_seen = self._masked_mean(gate_prob, rev_mask)
            gate_unseen = self._masked_mean(gate_prob, novel_mask)
            effective_gate_seen = self._masked_mean(effective_gate, rev_mask)
            effective_gate_unseen = self._masked_mean(effective_gate, novel_mask)
            gate_sep = gate_seen - gate_unseen
            gate_acc = ((gate_prob > 0.5) == rev_mask).float().mean()
            gate_revisit_recall = self._masked_mean(gate_prob > 0.5, rev_mask)
            gate_novel_recall = self._masked_mean(gate_prob <= 0.5, novel_mask)

            pred_match = ret_logits.argmax(-1)
            hit = pos.gather(1, pred_match[:, None]).squeeze(1).float()
            seen_match = self._masked_mean(hit, rev_mask)
            selected_negative = neg.gather(
                1, pred_match[:, None]
            ).squeeze(1)
            selected_gray = ~(hit.bool() | selected_negative)
            seen_match_negative = self._masked_mean(
                selected_negative, rev_mask
            )
            seen_match_gray = self._masked_mean(selected_gray, rev_mask)

            xy_error = pred_xy - gt_xy
            xy_sq = xy_error.square()
            xy_l2 = torch.linalg.norm(xy_error, dim=-1)
            aux_mse_x = self._masked_mean(xy_sq[:, 0], rev_mask)
            aux_mse_y = self._masked_mean(xy_sq[:, 1], rev_mask)
            aux_xy_l2 = self._masked_mean(xy_l2, rev_mask)
            # Report the true bearing angle. The optimized cosine uses a larger
            # small-norm guard to avoid an unstable gradient near a zero vector.
            direction_err_deg = torch.rad2deg(torch.arccos(metric_direction_cos))
            direction_err = self._masked_mean(direction_err_deg, aux_valid)
            raw_direction_err_deg = torch.rad2deg(torch.arccos(raw_direction_cos))
            raw_direction_err = self._masked_mean(
                raw_direction_err_deg, revisit_pose_valid
            )
            pose_reliability_mean = self._masked_mean(
                pose_reliability, revisit_pose_valid
            )
            pose_quality_mean = self._masked_mean(
                pose_quality_target, revisit_pose_valid
            )
            pose_reliability_brier = self._masked_mean(
                (pose_reliability - pose_quality_target).square(), revisit_pose_valid
            )
            pose_reliability_features = fwd['pose_reliability_features']
            if pose_reliability_features.shape[-1] != len(
                self.POSE_RELIABILITY_FEATURE_NAMES
            ):
                raise ValueError('unexpected pose reliability feature width')
            raw_range_code = pose_reliability_features[:, 0]
            range_code_abs_error = (pred_range_code - safe_range_target).abs()
            raw_range_finite = torch.isfinite(raw_range_code) & torch.isfinite(
                target_range_code
            )
            raw_range_valid = aux_valid & raw_range_finite
            safe_raw_range_target = torch.where(
                raw_range_finite, target_range_code, raw_range_code
            )
            raw_range_code_abs_error = (
                raw_range_code - safe_raw_range_target
            ).abs()
            aux_range_code_mae = self._masked_mean(
                range_code_abs_error, range_valid
            )
            raw_range_code_mae = self._masked_mean(
                raw_range_code_abs_error, raw_range_valid
            )
            target_range_code_mean = self._masked_mean(
                safe_range_target, range_valid
            )
            target_range_code_sq_mean = self._masked_mean(
                safe_range_target.square(), range_valid
            )

            gate_feature = fwd['gate_feature']
            gate_feature_seen = self._masked_mean(gate_feature, rev_mask)
            gate_feature_unseen = self._masked_mean(gate_feature, novel_mask)
            gate_effective_threshold = fwd['gate_effective_threshold']
            gate_normalized_slope = fwd['gate_normalized_slope']

            # Compare actual relative camera rotation, separately from path theta.
            R_rel = fwd['R_rel']
            gt_rot = inputs['batch_goal_rel_rotation'].to(dev)
            raw_cos_ang = (((R_rel * gt_rot).sum(dim=(-2, -1)) - 1.0) / 2.0).clamp(-1, 1)
            rot_err_raw = self._masked_mean(
                torch.rad2deg(torch.arccos(raw_cos_ang)), rev_mask
            )
            C_rot = self._C_rot.to(dev, R_rel.dtype)
            R_rel_conv = C_rot @ R_rel @ C_rot.transpose(-1, -2)
            conv_cos_ang = (
                ((R_rel_conv * gt_rot).sum(dim=(-2, -1)) - 1.0) / 2.0
            ).clamp(-1, 1)
            rot_err_converted = self._masked_mean(
                torch.rad2deg(torch.arccos(conv_cos_ang)), rev_mask
            )

            # Window-level moments expose regression collapse without mistaking a
            # single batch's standard deviation for a stable statistic.
            pred_x = self._masked_mean(pred_xy[:, 0], rev_mask)
            pred_y = self._masked_mean(pred_xy[:, 1], rev_mask)
            pred_x2 = self._masked_mean(pred_xy[:, 0].square(), rev_mask)
            pred_y2 = self._masked_mean(pred_xy[:, 1].square(), rev_mask)
            gt_x = self._masked_mean(gt_xy[:, 0], rev_mask)
            gt_y = self._masked_mean(gt_xy[:, 1], rev_mask)
            gt_x2 = self._masked_mean(gt_xy[:, 0].square(), rev_mask)
            gt_y2 = self._masked_mean(gt_xy[:, 1].square(), rev_mask)

            # Preserve composition in W&B.  Aggregate aux y MSE was previously
            # dominated by two long C-leg rows and looked like a global axis bug.
            # These diagnostics never enter the loss; they expose goal type and
            # temporal support together with an explicit support fraction.
            per_sample_action = action_sq.mean(dim=(1, 2))
            decision_angle = inputs.get('batch_decision_route_angle_deg')
            decision_hard = inputs.get('batch_decision_curriculum_hard')
            if decision_angle is not None and decision_hard is not None:
                decision_angle = decision_angle.to(dev)
                decision_hard = decision_hard.to(dev).bool()
                hard_count = decision_hard.float().sum()
                easy_count = (~decision_hard).float().sum()
                stratified_accumulations.extend([
                    (
                        'decision_hard_fraction',
                        hard_count / B,
                        B,
                    ),
                    (
                        'decision_route_angle_deg',
                        decision_angle.mean(),
                        B,
                    ),
                    (
                        'action_loss_decision_hard',
                        self._masked_mean(per_sample_action, decision_hard),
                        hard_count,
                    ),
                    (
                        'action_loss_decision_easy',
                        self._masked_mean(per_sample_action, ~decision_hard),
                        easy_count,
                    ),
                ])
            goal_j = inputs.get('batch_goal_j')
            if goal_j is not None:
                goal_j = goal_j.to(dev)
                for goal_index, goal_label in ((-1, 'A'), (0, 'B'), (1, 'C')):
                    goal_mask = goal_j == goal_index
                    goal_count = goal_mask.float().sum()
                    stratified_accumulations.extend([
                        (
                            f'goal_{goal_label}_fraction',
                            goal_count / B,
                            B,
                        ),
                        (
                            f'action_loss_goal_{goal_label}',
                            self._masked_mean(per_sample_action, goal_mask),
                            goal_count,
                        ),
                        (
                            f'gate_goal_{goal_label}',
                            self._masked_mean(gate_prob, goal_mask),
                            goal_count,
                        ),
                        (
                            f'effective_gate_goal_{goal_label}',
                            self._masked_mean(effective_gate, goal_mask),
                            goal_count,
                        ),
                    ])
                    goal_aux_mask = goal_mask & aux_valid
                    goal_aux_count = goal_aux_mask.float().sum()
                    goal_range_mask = goal_mask & range_valid
                    goal_range_count = goal_range_mask.float().sum()
                    stratified_accumulations.extend([
                        (
                            f'aux_direction_loss_goal_{goal_label}_revisit',
                            self._masked_mean(1.0 - direction_cos, goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'aux_direction_err_deg_goal_{goal_label}_revisit',
                            self._masked_mean(direction_err_deg, goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'aux_range_loss_goal_{goal_label}_revisit',
                            self._masked_mean(range_per_row, goal_range_mask),
                            goal_range_count,
                        ),
                        (
                            f'aux_range_code_mae_goal_{goal_label}_revisit',
                            self._masked_mean(
                                range_code_abs_error, goal_range_mask
                            ),
                            goal_range_count,
                        ),
                        (
                            f'aux_mse_x_goal_{goal_label}_revisit',
                            self._masked_mean(xy_sq[:, 0], goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'aux_mse_y_goal_{goal_label}_revisit',
                            self._masked_mean(xy_sq[:, 1], goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'pose_reliability_goal_{goal_label}_revisit',
                            self._masked_mean(pose_reliability, goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'pose_quality_goal_{goal_label}_revisit',
                            self._masked_mean(pose_quality_target, goal_aux_mask),
                            goal_aux_count,
                        ),
                        (
                            f'raw_pose_direction_err_deg_goal_{goal_label}_revisit',
                            self._masked_mean(raw_direction_err_deg, goal_aux_mask),
                            goal_aux_count,
                        ),
                    ])

            def add_aux_bins(values, bins, prefix):
                for lower, upper, label in bins:
                    bin_mask = aux_valid & (values >= lower)
                    if upper is not None:
                        bin_mask = bin_mask & (values < upper)
                    count = bin_mask.float().sum()
                    range_bin_mask = bin_mask & range_valid
                    range_bin_count = range_bin_mask.float().sum()
                    stratified_accumulations.extend([
                        (
                            f'aux_{prefix}_{label}_fraction_revisit',
                            count / n_rev.clamp(min=1.0),
                            n_rev,
                        ),
                        (
                            f'aux_direction_err_deg_{prefix}_{label}',
                            self._masked_mean(direction_err_deg, bin_mask),
                            count,
                        ),
                        (
                            f'aux_range_code_mae_{prefix}_{label}',
                            self._masked_mean(
                                range_code_abs_error, range_bin_mask
                            ),
                            range_bin_count,
                        ),
                        (
                            f'aux_mse_x_{prefix}_{label}',
                            self._masked_mean(xy_sq[:, 0], bin_mask),
                            count,
                        ),
                        (
                            f'aux_mse_y_{prefix}_{label}',
                            self._masked_mean(xy_sq[:, 1], bin_mask),
                            count,
                        ),
                        (
                            f'raw_pose_direction_err_deg_{prefix}_{label}',
                            self._masked_mean(raw_direction_err_deg, bin_mask),
                            count,
                        ),
                        (
                            f'pose_reliability_{prefix}_{label}',
                            self._masked_mean(pose_reliability, bin_mask),
                            count,
                        ),
                        (
                            f'pose_quality_{prefix}_{label}',
                            self._masked_mean(pose_quality_target, bin_mask),
                            count,
                        ),
                        (
                            f'action_loss_{prefix}_{label}',
                            self._masked_mean(per_sample_action, bin_mask),
                            count,
                        ),
                    ])

            anchor_idx = fwd.get('anchor_idx')
            if anchor_idx is not None and inputs.get('cur_steps') is not None:
                cur_steps = torch.as_tensor(inputs['cur_steps'], device=dev)
                anchor_gap = cur_steps - anchor_idx.to(dev)
                add_aux_bins(
                    anchor_gap,
                    ((0, 256, 'gap_000_255'),
                     (256, 512, 'gap_256_511'),
                     (512, None, 'gap_512_plus')),
                    'anchor',
                )
            if (inputs.get('cur_steps') is not None
                    and inputs.get('goal_steps') is not None):
                cur_steps = torch.as_tensor(inputs['cur_steps'], device=dev)
                goal_steps = torch.as_tensor(inputs['goal_steps'], device=dev)
                remaining_span = goal_steps - cur_steps
                add_aux_bins(
                    remaining_span,
                    ((0, 128, 'span_000_127'),
                     (128, 256, 'span_128_255'),
                     (256, None, 'span_256_plus')),
                    'path',
                )

        outputs = {
            'loss': loss,
            'action_loss': action_loss,
            'action_noise_mse_x': action_axis_mse[0],
            'action_noise_mse_y': action_axis_mse[1],
            'action_noise_mse_theta': action_axis_mse[2],
            'retrieval_loss': rank_loss,
            'retrieval_hard_loss': retrieval_hard_loss,
            'retrieval_all_candidate_loss': retrieval_all_candidate_loss,
            'retrieval_rank_temperature': retrieval_rank_temperature,
            'retrieval_residual_abs_mean': retrieval_residual_abs_mean,
            'gate_loss': gate_loss,
            'aux_direction_loss': aux_direction_loss,
            'route_direction_loss': route_direction_loss,
            'aux_range_loss': aux_range_loss,
            'aux_range_code_mae': aux_range_code_mae,
            'raw_range_code_mae': raw_range_code_mae,
            'range_grad_action_cosine': range_gradient_diagnostics['raw_cosine'],
            'range_grad_raw_to_action_norm': range_gradient_diagnostics[
                'raw_norm_ratio'
            ],
            'range_grad_corrected_to_action_norm': range_gradient_diagnostics[
                'corrected_norm_ratio'
            ],
            'range_grad_cap_scale': range_gradient_diagnostics['cap_scale'],
            'range_grad_conflict': range_gradient_diagnostics['conflict'],
            'pose_reliability_loss': pose_reliability_loss,
            'anchor_tf_probability': loss.new_tensor(anchor_tf_probability),
            'anchor_teacher_forced_fraction': anchor_teacher_forced_fraction,
            'anchor_positive_fraction': anchor_positive_fraction,
            'gate_seen': gate_seen,
            'gate_unseen': gate_unseen,
            'effective_gate_seen': effective_gate_seen,
            'effective_gate_unseen': effective_gate_unseen,
            'gate_sep': gate_sep,
            'gate_acc': gate_acc,
            'gate_revisit_recall': gate_revisit_recall,
            'gate_novel_recall': gate_novel_recall,
            'gate_effective_threshold': gate_effective_threshold,
            'gate_normalized_slope': gate_normalized_slope,
            'seen_match_acc': seen_match,
            'seen_match_negative_fraction': seen_match_negative,
            'seen_match_gray_fraction': seen_match_gray,
            'rot_err_raw_deg': rot_err_raw,
            'rot_err_converted_deg': rot_err_converted,
            'aux_mse_x': aux_mse_x,
            'aux_mse_y': aux_mse_y,
            'aux_xy_l2': aux_xy_l2,
            'aux_direction_err_deg': direction_err,
            'raw_pose_direction_err_deg': raw_direction_err,
            'pose_reliability': pose_reliability_mean,
            'pose_quality': pose_quality_mean,
            'pose_reliability_brier': pose_reliability_brier,
        }

        # Accumulate over exactly the same interval as Trainer's own train/loss.
        action_value_count = float(action_target.shape[0] * action_target.shape[1])
        self._accumulate('action_loss', action_loss, action_value_count * 3.0, phase)
        self._accumulate(
            'diffusion_noise_mean', diffusion_noise_mean,
            float(diffusion_noise.numel()), phase,
        )
        self._accumulate(
            'diffusion_noise_sq_mean', diffusion_noise_sq_mean,
            float(diffusion_noise.numel()), phase,
        )
        if diffusion_timesteps is not None:
            self._accumulate(
                'diffusion_timestep_mean', diffusion_timesteps.float().mean(),
                action_target.shape[0], phase,
            )
        for axis, value in zip(('x', 'y', 'theta'), action_axis_mse):
            self._accumulate(
                f'action_noise_mse_{axis}', value, action_value_count, phase
            )
        for axis_index, axis in enumerate(('x', 'y', 'theta')):
            self._accumulate(
                f'action_target_{axis}_mean', action_target_mean[axis_index],
                action_value_count, phase,
            )
            self._accumulate(
                f'action_target_{axis}_sq_mean', action_target_sq_mean[axis_index],
                action_value_count, phase,
            )
        self._accumulate('retrieval_loss', rank_loss, rank_count, phase)
        self._accumulate(
            'retrieval_hard_loss', retrieval_hard_loss,
            hard_rank_count, phase,
        )
        self._accumulate(
            'retrieval_all_candidate_loss', retrieval_all_candidate_loss,
            all_rank_count, phase,
        )
        self._accumulate(
            'retrieval_rank_temperature', retrieval_rank_temperature,
            B, phase,
        )
        self._accumulate(
            'retrieval_residual_abs_mean',
            retrieval_residual_abs_mean, B, phase,
        )
        self._accumulate('gate_loss', gate_loss, B, phase)
        self._accumulate('aux_direction_loss', aux_direction_loss, aux_count, phase)
        self._accumulate(
            'route_direction_loss', route_direction_loss, route_count, phase
        )
        if route_prediction is not None:
            route_raw_norm = fwd['route_raw_direction_norm']
            route_scale = fwd['route_residual_scale']
            route_gate = fwd['route_curvature_gate']
            route_row_valid = route_valid[:, 0] & route_valid[:, -1]
            route_row_count = route_row_valid.float().sum()
            self._accumulate(
                'route_target_curvature_deg',
                self._masked_mean(
                    route_target_curvature_deg, route_row_valid
                ),
                route_row_count,
                phase,
            )
            self._accumulate(
                'route_prediction_curvature_deg',
                self._masked_mean(
                    route_prediction_curvature_deg, route_row_valid
                ),
                route_row_count,
                phase,
            )
            self._accumulate(
                'route_curvature_gate',
                self._masked_mean(route_gate, route_row_valid),
                route_row_count,
                phase,
            )
            for horizon_index, horizon in enumerate(self.route_horizons):
                horizon_valid = route_valid[:, horizon_index]
                horizon_count = horizon_valid.float().sum()
                horizon_error = self._masked_mean(
                    route_direction_error_deg[:, horizon_index], horizon_valid
                )
                self._accumulate(
                    f'route_direction_err_deg_h{horizon}',
                    horizon_error,
                    horizon_count,
                    phase,
                )
                self._accumulate(
                    f'route_raw_norm_h{horizon}',
                    self._masked_mean(
                        route_raw_norm[:, horizon_index], horizon_valid
                    ),
                    horizon_count,
                    phase,
                )
                self._accumulate(
                    f'route_residual_scale_h{horizon}',
                    route_scale[horizon_index],
                    B,
                    phase,
                )
        self._accumulate('aux_range_loss', aux_range_loss, range_count, phase)
        self._accumulate(
            'aux_range_code_mae', aux_range_code_mae, range_count, phase
        )
        self._accumulate(
            'raw_range_code_mae', raw_range_code_mae,
            raw_range_valid.float().sum(), phase,
        )
        if range_gradient_control_enabled:
            for name, value in (
                ('range_grad_action_cosine', range_gradient_diagnostics['raw_cosine']),
                (
                    'range_grad_raw_to_action_norm',
                    range_gradient_diagnostics['raw_norm_ratio'],
                ),
                (
                    'range_grad_corrected_to_action_norm',
                    range_gradient_diagnostics['corrected_norm_ratio'],
                ),
                ('range_grad_cap_scale', range_gradient_diagnostics['cap_scale']),
                ('range_grad_conflict', range_gradient_diagnostics['conflict']),
            ):
                self._accumulate(name, value, B, phase)
        self._accumulate(
            'aux_range_target_mean', target_range_code_mean, range_count, phase
        )
        self._accumulate(
            'aux_range_target_sq_mean', target_range_code_sq_mean,
            range_count, phase,
        )
        self._accumulate(
            'pose_reliability_loss', pose_reliability_loss,
            reliability_count, phase,
        )
        self._accumulate('revisit_fraction', n_rev / B, B, phase)
        self._accumulate(
            'anchor_tf_probability', loss.new_tensor(anchor_tf_probability), B, phase
        )
        self._accumulate(
            'anchor_teacher_forced_fraction', anchor_teacher_forced_fraction,
            n_rev, phase,
        )
        self._accumulate(
            'anchor_positive_fraction', anchor_positive_fraction, n_rev, phase
        )
        self._accumulate('rank_row_fraction', rank_count / B, B, phase)
        self._accumulate('gate_acc', gate_acc, B, phase)
        self._accumulate('gate_revisit_recall', gate_revisit_recall, n_rev, phase)
        self._accumulate('gate_novel_recall', gate_novel_recall, n_novel, phase)
        self._accumulate('gate_seen', gate_seen, n_rev, phase)
        self._accumulate('gate_unseen', gate_unseen, n_novel, phase)
        self._accumulate('effective_gate_seen', effective_gate_seen, n_rev, phase)
        self._accumulate('effective_gate_unseen', effective_gate_unseen, n_novel, phase)
        self._accumulate('seen_match_acc', seen_match, n_rev, phase)
        self._accumulate(
            'seen_match_negative_fraction', seen_match_negative, n_rev, phase
        )
        self._accumulate(
            'seen_match_gray_fraction', seen_match_gray, n_rev, phase
        )
        self._accumulate('gate_feature_seen', gate_feature_seen, n_rev, phase)
        self._accumulate('gate_feature_unseen', gate_feature_unseen, n_novel, phase)
        self._accumulate('gate_effective_threshold', gate_effective_threshold, B, phase)
        self._accumulate('gate_normalized_slope', gate_normalized_slope, B, phase)
        self._accumulate('aux_mse_x', aux_mse_x, n_rev, phase)
        self._accumulate('aux_mse_y', aux_mse_y, n_rev, phase)
        self._accumulate('aux_xy_l2', aux_xy_l2, n_rev, phase)
        self._accumulate('aux_direction_err_deg', direction_err, aux_count, phase)
        self._accumulate(
            'raw_pose_direction_err_deg', raw_direction_err,
            reliability_count, phase,
        )
        self._accumulate(
            'pose_reliability', pose_reliability_mean, reliability_count, phase
        )
        self._accumulate('pose_quality', pose_quality_mean, reliability_count, phase)
        self._accumulate(
            'pose_reliability_brier', pose_reliability_brier,
            reliability_count, phase,
        )
        for feature_index, feature_name in enumerate(
            self.POSE_RELIABILITY_FEATURE_NAMES
        ):
            self._accumulate(
                f'pose_cue_{feature_name}',
                self._masked_mean(
                    pose_reliability_features[:, feature_index],
                    revisit_pose_valid,
                ),
                reliability_count,
                phase,
            )
        self._accumulate('rot_err_raw_deg', rot_err_raw, n_rev, phase)
        self._accumulate('rot_err_converted_deg', rot_err_converted, n_rev, phase)
        for name, value in (
            ('aux_pred_x_mean', pred_x), ('aux_pred_y_mean', pred_y),
            ('aux_pred_x_sq_mean', pred_x2), ('aux_pred_y_sq_mean', pred_y2),
            ('aux_gt_x_mean', gt_x), ('aux_gt_y_mean', gt_y),
            ('aux_gt_x_sq_mean', gt_x2), ('aux_gt_y_sq_mean', gt_y2),
        ):
            self._accumulate(name, value, n_rev, phase)
        for name, value, weight in stratified_accumulations:
            self._accumulate(name, value, weight, phase)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Attach interval-averaged component metrics once per HF log event."""
        phase = 'eval' if any(key.startswith('eval_') for key in logs) else 'train'
        component_logs = {}
        accumulator = self._metric_accumulators[phase]
        had_components = bool(accumulator)
        for name, (numerator, denominator) in accumulator.items():
            den = float(denominator.item())
            if den > 0.0:
                component_logs[name] = float((numerator / denominator).item())
        accumulator.clear()
        # These are literal zero contributions to the objective when an interval
        # contains no supported revisit rows. Log the zero together with the
        # support fractions instead of leaving a sparse/stale W&B series.
        if had_components:
            component_logs.setdefault('retrieval_loss', 0.0)
            component_logs.setdefault('retrieval_hard_loss', 0.0)
            component_logs.setdefault('retrieval_all_candidate_loss', 0.0)
            if self.retrieval_only:
                component_logs.setdefault('retrieval_margin_loss', 0.0)
                component_logs.setdefault('retrieval_strict_top1', 0.0)
                component_logs.setdefault('retrieval_recall_at_5', 0.0)
                component_logs.setdefault('retrieval_recall_at_10', 0.0)
            component_logs.setdefault('aux_direction_loss', 0.0)
            component_logs.setdefault('route_direction_loss', 0.0)
            component_logs.setdefault('aux_range_loss', 0.0)
            component_logs.setdefault('pose_reliability_loss', 0.0)

        # Convert accumulated first/second moments into an interval-level std.
        for entity in (
            'action_target_x', 'action_target_y', 'action_target_theta',
            'diffusion_noise',
            'aux_pred_x', 'aux_pred_y', 'aux_gt_x', 'aux_gt_y',
            'aux_range_target',
        ):
            mean_key = f'{entity}_mean'
            sq_key = f'{entity}_sq_mean'
            if mean_key in component_logs and sq_key in component_logs:
                mean = component_logs[mean_key]
                variance = max(component_logs.pop(sq_key) - mean * mean, 0.0)
                component_logs[f'{entity}_std'] = variance ** 0.5

        # Compute separation from the whole logging window, rather than averaging
        # noisy per-batch differences whose class counts can be very different.
        if 'gate_seen' in component_logs and 'gate_unseen' in component_logs:
            component_logs['gate_sep'] = (
                component_logs['gate_seen'] - component_logs['gate_unseen']
            )

        dev = self.model_device
        if dev.type == 'cuda' and torch.cuda.is_available():
            component_logs['mem_alloc_gb'] = torch.cuda.max_memory_allocated(dev) / 2**30
            component_logs['mem_reserved_gb'] = torch.cuda.max_memory_reserved(dev) / 2**30
            torch.cuda.reset_peak_memory_stats(dev)

        if phase == 'eval':
            component_logs = {f'eval_{key}': value for key, value in component_logs.items()}
        logs.update(component_logs)
        rank = dist.get_rank() if dist.is_initialized() else 0
        action_key = 'eval_action_loss' if phase == 'eval' else 'action_loss'
        if rank == 0 and self.retrieval_only and (
            'eval_retrieval_loss' if phase == 'eval' else 'retrieval_loss'
        ) in component_logs:
            prefix = 'eval_' if phase == 'eval' else ''
            print(
                f"[Step {self.state.global_step}] phase={phase} "
                f"rank={component_logs[prefix + 'retrieval_loss']:.4f} "
                f"margin={component_logs.get(prefix + 'retrieval_margin_loss', 0.0):.4f} "
                f"top1={component_logs.get(prefix + 'retrieval_strict_top1', 0.0):.4f} "
                f"r10={component_logs.get(prefix + 'retrieval_recall_at_10', 0.0):.4f}"
            )
        elif rank == 0 and action_key in component_logs:
            def display(key):
                value = component_logs.get(key)
                return 'n/a' if value is None else f'{value:.4f}'

            print(
                f"[Step {self.state.global_step}] "
                f"phase={phase} "
                f"act={display(action_key)} "
                f"rank={display('eval_retrieval_loss' if phase == 'eval' else 'retrieval_loss')} "
                f"gate={display('eval_gate_loss' if phase == 'eval' else 'gate_loss')} "
                f"aux_dir={display('eval_aux_direction_loss' if phase == 'eval' else 'aux_direction_loss')} "
                f"route={display('eval_route_direction_loss' if phase == 'eval' else 'route_direction_loss')} "
                f"aux_range={display('eval_aux_range_loss' if phase == 'eval' else 'aux_range_loss')} "
                f"pose_rel={display('eval_pose_reliability_loss' if phase == 'eval' else 'pose_reliability_loss')}"
            )
        return super().log(logs, start_time)

    def evaluate(self, *args, **kwargs):
        """Use the same diffusion noise/timestep sequence at every validation."""
        devices = []
        if self.model_device.type == 'cuda':
            devices = [self.model_device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.eval_seed)
            if devices:
                torch.cuda.manual_seed_all(self.eval_seed)
            return super().evaluate(*args, **kwargs)

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=self.args.seed,
        )
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
        """Save trainable policy state without duplicating frozen LingBot weights."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        full_state = state_dict if state_dict is not None else model.state_dict()
        state = {key: value for key, value in full_state.items() if 'lingbot.' not in key}
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state, os.path.join(output_dir, 'memnav.ckpt'))
        retrieval = getattr(getattr(model, 'core', None), 'retrieval', None)
        revisit_merge = getattr(getattr(model, 'core', None), 'revisit_merge', None)
        pose_encoder = getattr(revisit_merge, 'pose_encoder', None)
        route_sketch = getattr(getattr(model, 'core', None), 'route_sketch', None)
        metadata = {
            'format_version': 4,
            'dataset_fingerprint': self._dataset_fingerprint(self.train_dataset),
            'eval_dataset_fingerprint': self._dataset_fingerprint(self.eval_dataset),
            'gate_parameterization': 'normalized_raw_cosine_v1',
            'retrieval_rank_mode': getattr(
                retrieval, 'rank_mode', None
            ),
            'gate_center': (
                float(retrieval.gate_center.detach().cpu()) if retrieval is not None else None
            ),
            'gate_width': (
                float(retrieval.gate_width.detach().cpu()) if retrieval is not None else None
            ),
            'revisit_pose_code': getattr(pose_encoder, 'CODE_VERSION', None),
            'route_sketch_code': getattr(route_sketch, 'CODE_VERSION', None),
            # These values do not change checkpoint architecture, so format 4
            # remains load-compatible. Persist them for exact objective/schedule
            # provenance across resumes and comparisons.
            'training_objective': self._training_objective_metadata(),
        }
        with open(os.path.join(output_dir, 'memnav_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f'Saved {len(state)} non-LingBot tensors to {output_dir}/memnav.ckpt')

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore our compact model file; HF restores optimizer/scheduler/RNG."""
        checkpoint_path = os.path.join(resume_from_checkpoint, 'memnav.ckpt')
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f'MemNav checkpoint is missing {checkpoint_path}; cannot resume safely.'
            )

        metadata_path = os.path.join(resume_from_checkpoint, 'memnav_metadata.json')
        if not os.path.isfile(metadata_path):
            raise ValueError(
                f'MemNav checkpoint is missing {metadata_path}; cannot verify gate '
                'optimizer units or dataset identity for a safe resume.'
            )
        with open(metadata_path, encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('format_version') != 4:
            raise ValueError(
                'Cannot resume this MemNav checkpoint safely: expected metadata '
                'format 4 (normalized gate + gauge-invariant pose code), got '
                f"{metadata.get('format_version')!r}. Legacy checkpoints remain "
                'available for offline evaluation, but training must start a new run.'
            )
        saved_objective = metadata.get('training_objective')
        current_objective = self._training_objective_metadata()
        if saved_objective is not None:
            # Checkpoints written before action-safe range gradients existed are
            # exactly equivalent to the new default-off value.
            saved_objective = dict(saved_objective)
            saved_objective.setdefault('aux_range_grad_cap_ratio', 0.0)
            saved_objective.setdefault(
                'retrieval_denominator', 'positive_negative'
            )
            saved_objective.setdefault('retrieval_lr_multiplier', 1.0)
            saved_objective.setdefault('retrieval_only', False)
            saved_objective.setdefault('retrieval_margin_cosine', 0.005)
            saved_objective.setdefault('retrieval_margin_weight', 0.0)
            saved_objective.setdefault('retrieval_rank_mode', 'projected')
            saved_objective.setdefault('retrieval_raw_temp_init', 0.01)
            saved_objective.setdefault('retrieval_residual_max', 0.25)
            saved_objective.setdefault('retrieval_temporal_topk', 10)
            saved_objective.setdefault(
                'retrieval_temporal_residual_max', 0.02
            )
            saved_objective.setdefault('retrieval_anchor_min_frame', None)
            saved_objective.setdefault('w_route_direction', 0.0)
            saved_objective.setdefault('use_route_sketch', False)
            saved_objective.setdefault('route_horizons', [2, 8, 24])
            saved_objective.setdefault('route_curvature_emphasis', 0.0)
            saved_objective.setdefault('route_lr_multiplier', 10.0)
        if saved_objective is not None and saved_objective != current_objective:
            changed = sorted(
                key for key in set(saved_objective) | set(current_objective)
                if saved_objective.get(key) != current_objective.get(key)
            )
            raise ValueError(
                'Training objective changed since the checkpoint was written: '
                f'{changed}. Start a new run with ckpt_to_load instead of resuming '
                'optimizer/scheduler state under a different objective.'
            )
        saved_fingerprint = metadata.get('dataset_fingerprint')
        current_fingerprint = self._dataset_fingerprint(self.train_dataset)
        if (saved_fingerprint and current_fingerprint
                and saved_fingerprint != current_fingerprint):
            raise ValueError(
                'Dataset fingerprint changed since the checkpoint was written: '
                f'{saved_fingerprint} != {current_fingerprint}. Refusing to resume '
                'with a different training population.'
            )
        saved_eval_fingerprint = metadata.get('eval_dataset_fingerprint')
        current_eval_fingerprint = self._dataset_fingerprint(self.eval_dataset)
        if (saved_eval_fingerprint and current_eval_fingerprint
                and saved_eval_fingerprint != current_eval_fingerprint):
            raise ValueError(
                'Evaluation fingerprint changed since the checkpoint was written: '
                f'{saved_eval_fingerprint} != {current_eval_fingerprint}. '
                'Refusing to resume with a different fixed validation population.'
            )

        target = model if model is not None else self.model
        target = target.module if hasattr(target, 'module') else target
        target_revisit = getattr(getattr(target, 'core', None), 'revisit_merge', None)
        target_pose_encoder = getattr(target_revisit, 'pose_encoder', None)
        target_pose_code = getattr(target_pose_encoder, 'CODE_VERSION', None)
        if metadata.get('revisit_pose_code') != target_pose_code:
            raise ValueError(
                'Revisit pose-code version changed since the checkpoint was written: '
                f"{metadata.get('revisit_pose_code')!r} != {target_pose_code!r}."
            )
        target_route = getattr(getattr(target, 'core', None), 'route_sketch', None)
        target_route_code = getattr(target_route, 'CODE_VERSION', None)
        if metadata.get('route_sketch_code') != target_route_code:
            raise ValueError(
                'Route-sketch version changed since the checkpoint was written: '
                f"{metadata.get('route_sketch_code')!r} != "
                f'{target_route_code!r}.'
            )
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location='cpu')
        if hasattr(target, 'upgrade_checkpoint_state_dict'):
            state = target.upgrade_checkpoint_state_dict(state)
        current = target.state_dict()
        unexpected = sorted(key for key in state if key not in current)
        mismatched = sorted(
            key for key, value in state.items()
            if key in current and tuple(value.shape) != tuple(current[key].shape)
        )
        missing = sorted(
            key for key in current
            if 'lingbot.' not in key and key not in state
        )
        if unexpected or mismatched or missing:
            raise ValueError(
                'MemNav checkpoint is not architecture-compatible: '
                f'missing={missing[:8]}, unexpected={unexpected[:8]}, '
                f'mismatched={mismatched[:8]}'
            )
        incompatible = target.load_state_dict(state, strict=False)
        unexpected_loaded = [
            key for key in incompatible.unexpected_keys if 'lingbot.' not in key
        ]
        if unexpected_loaded:
            raise ValueError(f'Unexpected non-LingBot state: {unexpected_loaded[:8]}')
        print(f'Restored MemNav model state from {checkpoint_path}')
