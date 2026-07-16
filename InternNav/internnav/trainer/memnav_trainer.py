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
from internnav.trainer.base import BaseTrainer


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
        self.w_gate = getattr(config.il, 'w_gate', 1.0)
        self.w_aux = getattr(config.il, 'w_aux_direction', 0.2)
        self.w_pose_reliability = getattr(config.il, 'w_pose_reliability', 0.2)
        self._metric_accumulators = {'train': {}, 'eval': {}}
        self.eval_seed = int(getattr(config.il, 'eval_seed', 0))
        model = self.model.module if hasattr(self.model, 'module') else self.model
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
        calibration_names = gate_names | pose_reliability_names
        # Tiny test models and non-MemNav reuse should retain BaseTrainer's exact
        # optimizer behavior.
        if not calibration_names:
            return super().create_optimizer()

        decay_names = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
        decay_names = {name for name in decay_names if 'bias' not in name}
        regular_decay = [
            parameter for name, parameter in named_trainable
            if name not in calibration_names and name in decay_names
        ]
        regular_no_decay = [
            parameter for name, parameter in named_trainable
            if name not in calibration_names and name not in decay_names
        ]
        gate_parameters = [
            parameter for name, parameter in named_trainable if name in gate_names
        ]
        pose_reliability_parameters = [
            parameter for name, parameter in named_trainable
            if name in pose_reliability_names
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
                'params': pose_reliability_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate * pose_multiplier,
                'memnav_group': 'pose_reliability_calibration',
                'lr_multiplier': pose_multiplier,
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        fwd = model(inputs)
        phase = 'train' if model.training else 'eval'
        stratified_accumulations = []

        # Diffusion action objective. Keep per-coordinate errors so W&B can show
        # whether x, y, or wrapped theta is the coordinate that stopped learning.
        action_sq = (fwd['noise_pred'] - fwd['noise']).square()
        action_axis_mse = action_sq.mean(dim=(0, 1))
        action_loss = action_axis_mse.mean()
        action_target = inputs['batch_labels'].to(dev)
        action_target_mean = action_target.mean(dim=(0, 1))
        action_target_sq_mean = action_target.square().mean(dim=(0, 1))

        # Decoupled retrieval: rank a true co-visible frame on revisit rows, and
        # classify match existence on every row. This avoids an always-null shortcut.
        ret_logits = fwd['ret_logits']
        gate_logit = fwd['gate_logit']
        pos = inputs['batch_pos_mask'].to(dev).bool()
        neg = inputs['batch_neg_mask'].to(dev).bool()
        is_rev = inputs['batch_is_revisit'].to(dev).float()
        neg_inf = torch.finfo(ret_logits.dtype).min
        lse_pn = ret_logits.masked_fill(~(pos | neg), neg_inf).logsumexp(-1)
        lse_p = ret_logits.masked_fill(~pos, neg_inf).logsumexp(-1)
        rank_rows = pos.any(-1) & neg.any(-1)
        rank_count = rank_rows.float().sum()
        # Index first. On novel rows lse_p is the finite dtype floor; forming
        # the difference and multiplying it by zero later needlessly creates a
        # value near float32 max and can poison mixed-precision backward.
        rank_loss = (
            lse_pn[rank_rows] - lse_p[rank_rows]
        ).sum() / rank_count.clamp(min=1.0)

        n_rev = is_rev.sum()
        n_novel = (1.0 - is_rev).sum()
        B = float(is_rev.numel())
        # Do not estimate class weights from a four-sample batch: that makes the
        # objective itself jump between batches (all-revisit previously used 0.1,
        # mixed batches used a different value). The dataset is approximately
        # balanced and fixed validation reports both class recalls explicitly.
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, is_rev)

        # LingBot translations have a per-sequence canonical scale. Supervise only
        # direction, the identifiable signal, and keep raw metric error diagnostic.
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
        aux_valid = (is_rev > 0.5) & (gt_norm > 1e-4)
        aux_count = aux_valid.float().sum()
        aux_direction_loss = (
            (1.0 - direction_cos) * aux_valid
        ).sum() / aux_count.clamp(min=1.0)

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
        pose_reliability_loss = (
            pose_reliability_per_row * aux_valid
        ).sum() / aux_count.clamp(min=1.0)

        loss = (
            action_loss
            + self.w_retr * rank_loss
            + self.w_gate * gate_loss
            + self.w_aux * aux_direction_loss
            + self.w_pose_reliability * pose_reliability_loss
        )

        with torch.no_grad():
            rev_mask = is_rev > 0.5
            novel_mask = ~rev_mask
            gate_prob = torch.sigmoid(gate_logit)
            effective_gate = fwd['effective_revisit_gate']
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
            raw_direction_err = self._masked_mean(raw_direction_err_deg, aux_valid)
            pose_reliability_mean = self._masked_mean(pose_reliability, aux_valid)
            pose_quality_mean = self._masked_mean(pose_quality_target, aux_valid)
            pose_reliability_brier = self._masked_mean(
                (pose_reliability - pose_quality_target).square(), aux_valid
            )
            pose_reliability_features = fwd['pose_reliability_features']
            if pose_reliability_features.shape[-1] != len(
                self.POSE_RELIABILITY_FEATURE_NAMES
            ):
                raise ValueError('unexpected pose reliability feature width')

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
            'gate_loss': gate_loss,
            'aux_direction_loss': aux_direction_loss,
            'pose_reliability_loss': pose_reliability_loss,
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
        self._accumulate('gate_loss', gate_loss, B, phase)
        self._accumulate('aux_direction_loss', aux_direction_loss, aux_count, phase)
        self._accumulate(
            'pose_reliability_loss', pose_reliability_loss, aux_count, phase
        )
        self._accumulate('revisit_fraction', n_rev / B, B, phase)
        self._accumulate('rank_row_fraction', rank_count / B, B, phase)
        self._accumulate('gate_acc', gate_acc, B, phase)
        self._accumulate('gate_revisit_recall', gate_revisit_recall, n_rev, phase)
        self._accumulate('gate_novel_recall', gate_novel_recall, n_novel, phase)
        self._accumulate('gate_seen', gate_seen, n_rev, phase)
        self._accumulate('gate_unseen', gate_unseen, n_novel, phase)
        self._accumulate('effective_gate_seen', effective_gate_seen, n_rev, phase)
        self._accumulate('effective_gate_unseen', effective_gate_unseen, n_novel, phase)
        self._accumulate('seen_match_acc', seen_match, n_rev, phase)
        self._accumulate('gate_feature_seen', gate_feature_seen, n_rev, phase)
        self._accumulate('gate_feature_unseen', gate_feature_unseen, n_novel, phase)
        self._accumulate('gate_effective_threshold', gate_effective_threshold, B, phase)
        self._accumulate('gate_normalized_slope', gate_normalized_slope, B, phase)
        self._accumulate('aux_mse_x', aux_mse_x, n_rev, phase)
        self._accumulate('aux_mse_y', aux_mse_y, n_rev, phase)
        self._accumulate('aux_xy_l2', aux_xy_l2, n_rev, phase)
        self._accumulate('aux_direction_err_deg', direction_err, aux_count, phase)
        self._accumulate(
            'raw_pose_direction_err_deg', raw_direction_err, aux_count, phase
        )
        self._accumulate('pose_reliability', pose_reliability_mean, aux_count, phase)
        self._accumulate('pose_quality', pose_quality_mean, aux_count, phase)
        self._accumulate(
            'pose_reliability_brier', pose_reliability_brier, aux_count, phase
        )
        for feature_index, feature_name in enumerate(
            self.POSE_RELIABILITY_FEATURE_NAMES
        ):
            self._accumulate(
                f'pose_cue_{feature_name}',
                self._masked_mean(
                    pose_reliability_features[:, feature_index], aux_valid
                ),
                aux_count,
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
            component_logs.setdefault('aux_direction_loss', 0.0)
            component_logs.setdefault('pose_reliability_loss', 0.0)

        # Convert accumulated first/second moments into an interval-level std.
        for entity in (
            'action_target_x', 'action_target_y', 'action_target_theta',
            'aux_pred_x', 'aux_pred_y', 'aux_gt_x', 'aux_gt_y',
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
        if rank == 0 and action_key in component_logs:
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
        metadata = {
            'format_version': 4,
            'dataset_fingerprint': self._dataset_fingerprint(self.train_dataset),
            'eval_dataset_fingerprint': self._dataset_fingerprint(self.eval_dataset),
            'gate_parameterization': 'normalized_raw_cosine_v1',
            'gate_center': (
                float(retrieval.gate_center.detach().cpu()) if retrieval is not None else None
            ),
            'gate_width': (
                float(retrieval.gate_width.detach().cpu()) if retrieval is not None else None
            ),
            'revisit_pose_code': getattr(pose_encoder, 'CODE_VERSION', None),
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
