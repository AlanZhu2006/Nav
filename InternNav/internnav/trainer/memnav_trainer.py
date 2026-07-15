import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.trainer.base import BaseTrainer


class MemNavTrainer(BaseTrainer):
    """Trainer for the frozen LingBot front-end and trainable MemNav policy.

    The objective has three policy terms (diffusion action, retrieval ranking,
    revisit gate) plus a bounded, scale-invariant translation-direction
    auxiliary that shares ``rel_adapter`` with the revisit policy tokens.
    Metric x/y and camera-rotation errors are diagnostics only.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # HF otherwise classifies this custom dictionary batch as unlabeled in
        # prediction_step and bypasses compute_loss during scheduled evaluation.
        self.label_names = ['batch_labels']
        self.config = config
        self.w_retr = getattr(config.il, 'w_retrieval', 1.0)
        self.w_gate = getattr(config.il, 'w_gate', 1.0)
        self.w_aux = getattr(config.il, 'w_aux_direction', 0.2)
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

        loss = (
            action_loss
            + self.w_retr * rank_loss
            + self.w_gate * gate_loss
            + self.w_aux * aux_direction_loss
        )

        with torch.no_grad():
            rev_mask = is_rev > 0.5
            novel_mask = ~rev_mask
            gate_prob = torch.sigmoid(gate_logit)
            gate_seen = self._masked_mean(gate_prob, rev_mask)
            gate_unseen = self._masked_mean(gate_prob, novel_mask)
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

            gate_feature = fwd['gate_feature']
            gate_feature_seen = self._masked_mean(gate_feature, rev_mask)
            gate_feature_unseen = self._masked_mean(gate_feature, novel_mask)

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

        outputs = {
            'loss': loss,
            'action_loss': action_loss,
            'action_noise_mse_x': action_axis_mse[0],
            'action_noise_mse_y': action_axis_mse[1],
            'action_noise_mse_theta': action_axis_mse[2],
            'retrieval_loss': rank_loss,
            'gate_loss': gate_loss,
            'aux_direction_loss': aux_direction_loss,
            'gate_seen': gate_seen,
            'gate_unseen': gate_unseen,
            'gate_sep': gate_sep,
            'gate_acc': gate_acc,
            'gate_revisit_recall': gate_revisit_recall,
            'gate_novel_recall': gate_novel_recall,
            'seen_match_acc': seen_match,
            'rot_err_raw_deg': rot_err_raw,
            'rot_err_converted_deg': rot_err_converted,
            'aux_mse_x': aux_mse_x,
            'aux_mse_y': aux_mse_y,
            'aux_xy_l2': aux_xy_l2,
            'aux_direction_err_deg': direction_err,
        }

        # Accumulate over exactly the same interval as Trainer's own train/loss.
        B = float(is_rev.numel())
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
        self._accumulate('revisit_fraction', n_rev / B, B, phase)
        self._accumulate('rank_row_fraction', rank_count / B, B, phase)
        self._accumulate('gate_acc', gate_acc, B, phase)
        self._accumulate('gate_revisit_recall', gate_revisit_recall, n_rev, phase)
        self._accumulate('gate_novel_recall', gate_novel_recall, n_novel, phase)
        self._accumulate('gate_seen', gate_seen, n_rev, phase)
        self._accumulate('gate_unseen', gate_unseen, n_novel, phase)
        self._accumulate('seen_match_acc', seen_match, n_rev, phase)
        self._accumulate('gate_feature_seen', gate_feature_seen, n_rev, phase)
        self._accumulate('gate_feature_unseen', gate_feature_unseen, n_novel, phase)
        self._accumulate('aux_mse_x', aux_mse_x, n_rev, phase)
        self._accumulate('aux_mse_y', aux_mse_y, n_rev, phase)
        self._accumulate('aux_xy_l2', aux_xy_l2, n_rev, phase)
        self._accumulate('aux_direction_err_deg', direction_err, aux_count, phase)
        self._accumulate('rot_err_raw_deg', rot_err_raw, n_rev, phase)
        self._accumulate('rot_err_converted_deg', rot_err_converted, n_rev, phase)
        for name, value in (
            ('aux_pred_x_mean', pred_x), ('aux_pred_y_mean', pred_y),
            ('aux_pred_x_sq_mean', pred_x2), ('aux_pred_y_sq_mean', pred_y2),
            ('aux_gt_x_mean', gt_x), ('aux_gt_y_mean', gt_y),
            ('aux_gt_x_sq_mean', gt_x2), ('aux_gt_y_sq_mean', gt_y2),
        ):
            self._accumulate(name, value, n_rev, phase)

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
                f"aux_dir={display('eval_aux_direction_loss' if phase == 'eval' else 'aux_direction_loss')}"
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
        metadata = {
            'format_version': 2,
            'dataset_fingerprint': self._dataset_fingerprint(self.train_dataset),
            'eval_dataset_fingerprint': self._dataset_fingerprint(self.eval_dataset),
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
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding='utf-8') as f:
                metadata = json.load(f)
            if metadata.get('format_version') != 2:
                raise ValueError(
                    'Unsupported MemNav checkpoint metadata format: '
                    f"{metadata.get('format_version')!r}"
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
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location='cpu')
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
