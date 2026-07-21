import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from transformers import TrainingArguments

from internnav.model.basemodel.memnav.memnav_policy import MemNavPolicy
from internnav.model.basemodel.memnav.retrieval_head import RetrievalHead
from internnav.model.basemodel.memnav.route_sketch import (
    ResidualRouteSketch,
    route_curvature_gate,
)
from internnav.trainer.memnav_trainer import (
    MemNavTrainer,
    project_auxiliary_gradients,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, batch):
        return {'loss': self.linear(batch['x']).square().mean()}


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    @property
    def device(self):
        return self.scale.device

    def forward(self, batch):
        batch_size, horizon, _ = batch['batch_labels'].shape
        noise = torch.zeros(batch_size, horizon, 3, device=self.device)
        noise_pred = self.scale * torch.ones_like(noise)
        gate_logit = self.scale * torch.tensor([2.0, -2.0], device=self.device)
        ret_logits = self.scale * torch.tensor(
            [[3.0, 0.0], [0.0, 2.0]], device=self.device
        )
        aux_pose = self.scale * torch.tensor(
            [[1.0, 2.0], [0.5, 0.5]], device=self.device
        )
        aux_range_code = self.scale * torch.tensor(
            [0.3, 0.4], device=self.device
        )
        raw_pose_direction = torch.tensor(
            [[1.0, 2.0], [1.0, 0.0]], device=self.device
        )
        raw_pose_direction = torch.nn.functional.normalize(
            raw_pose_direction, dim=-1
        )
        pose_reliability = torch.sigmoid(
            self.scale * torch.tensor([2.0, 1.0], device=self.device)
        )
        revisit_gate = torch.sigmoid(gate_logit)
        return {
            'noise': noise,
            'noise_pred': noise_pred,
            'gate_logit': gate_logit,
            'revisit_gate': revisit_gate,
            'effective_revisit_gate': revisit_gate * pose_reliability,
            'ret_logits': ret_logits,
            'aux_pose': aux_pose,
            'aux_range_code': aux_range_code,
            'raw_pose_direction': raw_pose_direction,
            'pose_reliability': pose_reliability,
            'pose_reliability_features': torch.zeros(
                batch_size, 7, device=self.device
            ),
            'gate_feature': torch.tensor([0.9, 0.4], device=self.device),
            'gate_effective_threshold': torch.tensor(0.94, device=self.device),
            'gate_normalized_slope': torch.tensor(1.6, device=self.device),
            'match_idx': torch.tensor([0, 1], device=self.device),
            'anchor_idx': torch.tensor([0, 1], device=self.device),
            'anchor_teacher_forced': torch.tensor(
                [True, False], device=self.device
            ),
            'R_rel': torch.eye(3, device=self.device).repeat(batch_size, 1, 1),
        }


class _OptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.core = nn.Module()
        self.core.retrieval = nn.Module()
        self.core.retrieval.gate_log_slope = nn.Parameter(torch.tensor(0.0))
        self.core.retrieval.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.core.retrieval.proj = nn.Linear(3, 2)
        self.core.retrieval.raw_log_temp = nn.Parameter(torch.tensor(-4.0))
        self.core.retrieval.residual_weights = nn.Parameter(torch.zeros(2))
        self.core.revisit_merge = nn.Module()
        self.core.revisit_merge.pose_encoder = nn.Module()
        self.core.revisit_merge.pose_encoder.reliability_head = nn.Sequential(
            nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 1)
        )

    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, batch):
        return {'loss': self.linear(batch['x']).square().mean()}


class _RetrievalOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unrelated = nn.Linear(3, 2)
        self.core = nn.Module()
        self.core.retrieval = RetrievalHead(
            dino_dim=2,
            proj_dim=2,
            rank_mode='raw_temporal',
            raw_temp_init=0.01,
            temporal_topk=4,
            temporal_residual_max=0.02,
        )

    @property
    def device(self):
        return self.core.retrieval.temporal_weights.device

    def forward(self, batch):
        self.assert_retrieval_contract(batch)
        match, gate_logit, ret_logits, gate_feature = self.core.retrieval(
            batch['batch_goal_cls'].to(self.device),
            batch['batch_mem_cls'].to(self.device),
            batch['batch_cand_mask'].to(self.device),
        )
        return {
            'match_idx': match,
            'gate_logit': gate_logit,
            'gate_feature': gate_feature,
            'ret_logits': ret_logits,
            'retrieval_rank_temperature': (
                self.core.retrieval.rank_temperature
            ),
            'retrieval_residual_abs_mean': (
                self.core.retrieval.residual_weight_abs_mean
            ),
        }

    @staticmethod
    def assert_retrieval_contract(batch):
        if not bool(batch.get('retrieval_only', False)):
            raise AssertionError('missing retrieval-only dispatch marker')


class _RouteLossModel(_LossModel):
    def __init__(self):
        super().__init__()
        self.route_logits = nn.Parameter(torch.tensor([
            [0.2, 0.8], [0.8, 0.2]
        ]))
        self.route_scale = nn.Parameter(torch.zeros(2))

    def forward(self, batch):
        output = super().forward(batch)
        batch_size = batch['batch_labels'].shape[0]
        direction = torch.nn.functional.normalize(
            self.route_logits, dim=-1
        ).unsqueeze(0).expand(batch_size, -1, -1)
        output.update({
            'route_direction': direction,
            'route_raw_direction_norm': torch.linalg.vector_norm(
                self.route_logits, dim=-1
            ).unsqueeze(0).expand(batch_size, -1),
            'route_curvature_gate': route_curvature_gate(direction),
            'route_residual_scale': torch.tanh(self.route_scale),
        })
        return output


class _RouteOptimizerModel(_OptimizerModel):
    def __init__(self):
        super().__init__()
        self.core.route_sketch = ResidualRouteSketch(4, horizons=(1, 2))


class _CheckpointUpgradeRetrieval:
    @staticmethod
    def upgrade_legacy_state_dict(state_dict, prefix, copy=True):
        return dict(state_dict) if copy else state_dict


class _CheckpointUpgradeStub:
    def __init__(self, horizons=(2, 8, 24)):
        self.core = SimpleNamespace(
            retrieval=_CheckpointUpgradeRetrieval(),
            route_sketch=ResidualRouteSketch(8, horizons=horizons),
        )

    def state_dict(self):
        return {
            f'core.route_sketch.{key}': value
            for key, value in self.core.route_sketch.state_dict().items()
        }


class _AdapterGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.revisit_merge = nn.Module()
        self.core.revisit_merge.rel_adapter = nn.Linear(2, 2, bias=False)


class _TinyDataset(torch.utils.data.Dataset):
    dataset_fingerprint = 'fixed-population-v1'

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {'x': torch.ones(3)}


class MemNavCheckpointTest(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(il=SimpleNamespace(
            w_retrieval=1.0,
            w_gate=1.0,
            w_aux_direction=0.2,
            w_route_direction=0.0,
            use_route_sketch=False,
            route_horizons=(2, 8, 24),
            route_lr_multiplier=10.0,
            route_curvature_emphasis=0.0,
            w_aux_range=0.2,
            aux_range_beta=0.1,
            aux_range_grad_cap_ratio=0.0,
            w_pose_reliability=0.2,
            anchor_teacher_forcing_start=1.0,
            anchor_teacher_forcing_end=1.0,
            anchor_teacher_forcing_decay_steps=0,
            batch_size=1,
            num_workers=0,
            eval_seed=0,
            gate_lr_multiplier=10.0,
            pose_reliability_lr_multiplier=5.0,
        ))

    def _trainer(self, directory, eval_fingerprint=None):
        model = _TinyModel()
        args = TrainingArguments(
            output_dir=str(directory),
            report_to='none',
            per_device_train_batch_size=1,
        )
        eval_dataset = None
        if eval_fingerprint is not None:
            eval_dataset = _TinyDataset()
            eval_dataset.dataset_fingerprint = eval_fingerprint
        return MemNavTrainer(
            config=self._config(),
            model=model,
            args=args,
            train_dataset=_TinyDataset(),
            eval_dataset=eval_dataset,
        )

    def test_compact_checkpoint_may_only_omit_frozen_lingbot(self):
        compatible = SimpleNamespace(
            missing_keys=['core.lingbot.aggregator.weight'],
            unexpected_keys=[],
        )
        self.assertEqual(
            MemNavPolicy._validate_checkpoint_incompatibility(
                compatible, 'compact.ckpt'
            ),
            1,
        )
        missing_trainable = SimpleNamespace(
            missing_keys=['core.revisit_merge.revisit_head.weight'],
            unexpected_keys=[],
        )
        with self.assertRaisesRegex(ValueError, 'missing non-LingBot'):
            MemNavPolicy._validate_checkpoint_incompatibility(
                missing_trainable, 'broken.ckpt'
            )
        unexpected = SimpleNamespace(
            missing_keys=[], unexpected_keys=['obsolete.weight']
        )
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            MemNavPolicy._validate_checkpoint_incompatibility(
                unexpected, 'broken.ckpt'
            )

    def test_requested_initialization_checkpoint_must_exist(self):
        MemNavPolicy._validate_checkpoint_path('')
        with tempfile.NamedTemporaryFile() as checkpoint:
            MemNavPolicy._validate_checkpoint_path(checkpoint.name)
        with self.assertRaisesRegex(FileNotFoundError, 'not a file'):
            MemNavPolicy._validate_checkpoint_path('/definitely/missing/memnav.ckpt')

    def test_full_loss_logs_action_axes_and_class_balanced_gate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _LossModel()
            args = TrainingArguments(
                output_dir=tmp,
                report_to='none',
                per_device_train_batch_size=2,
            )
            trainer = MemNavTrainer(
                config=self._config(), model=model, args=args,
                train_dataset=_TinyDataset(),
            )
            self.assertEqual(trainer.label_names, ['batch_labels'])
            labels = torch.tensor([
                [[1.0, 0.0, -0.2], [2.0, 1.0, 0.2]],
                [[3.0, 2.0, -0.4], [4.0, 3.0, 0.4]],
            ])
            inputs = {
                'batch_labels': labels,
                'batch_pos_mask': torch.tensor([[True, False], [False, False]]),
                'batch_neg_mask': torch.tensor([[False, True], [True, True]]),
                'batch_cand_mask': torch.ones(2, 2, dtype=torch.bool),
                'batch_is_revisit': torch.tensor([1.0, 0.0]),
                'batch_goal_rel_pose': torch.tensor([
                    [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]
                ]),
                'batch_goal_range_code': torch.tensor([0.2, 0.5]),
                'batch_goal_range_steps': torch.tensor([6.4, 16.0]),
                'batch_gt_prefix_step_m': torch.tensor([0.25, 0.25]),
                'batch_goal_rel_rotation': torch.eye(3).repeat(2, 1, 1),
                'batch_goal_j': torch.tensor([0, -1]),
                'cur_steps': [319, 1024],
                'goal_steps': [400, 1100],
            }
            model.eval()
            eval_loss, _, _ = trainer.prediction_step(
                model, inputs, prediction_loss_only=True
            )
            self.assertTrue(torch.isfinite(eval_loss))
            trainer._metric_accumulators['eval'].clear()
            model.train()
            loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn('action_noise_mse_theta', outputs)
            self.assertIn('aux_range_loss', outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(model.scale.grad))
            trainer.log({'loss': float(loss.detach())})
            logged = trainer.state.log_history[-1]
            self.assertAlmostEqual(logged['action_noise_mse_theta'], 1.0)
            self.assertIn('diffusion_noise_mean', logged)
            self.assertIn('diffusion_noise_std', logged)
            self.assertGreater(logged['action_target_theta_std'], 0.0)
            self.assertAlmostEqual(logged['gate_revisit_recall'], 1.0)
            self.assertAlmostEqual(logged['gate_novel_recall'], 1.0)
            self.assertGreater(logged['gate_sep'], 0.0)
            self.assertAlmostEqual(logged['goal_A_fraction'], 0.5)
            self.assertAlmostEqual(logged['goal_B_fraction'], 0.5)
            self.assertIn('action_loss_goal_A', logged)
            self.assertIn('action_loss_goal_B', logged)
            self.assertIn('aux_direction_err_deg_goal_B_revisit', logged)
            self.assertIn('aux_range_code_mae_goal_B_revisit', logged)
            self.assertAlmostEqual(logged['anchor_tf_probability'], 1.0)
            self.assertIn('aux_mse_y_anchor_gap_256_511', logged)

            # A gray candidate participates in inference argmax, so the aligned
            # objective must also put it in the denominator.  The legacy hard-only
            # objective has no supported competitor on this row.
            gray_inputs = dict(inputs)
            gray_inputs['batch_neg_mask'] = torch.tensor([
                [False, False], [True, True]
            ])
            trainer.retrieval_denominator = 'all_candidates'
            _, gray_outputs = trainer.compute_loss(
                model, gray_inputs, return_outputs=True
            )
            self.assertEqual(
                float(gray_outputs['retrieval_hard_loss'].detach()), 0.0
            )
            self.assertGreater(
                float(gray_outputs['retrieval_all_candidate_loss'].detach()), 0.0
            )
            torch.testing.assert_close(
                gray_outputs['retrieval_loss'],
                gray_outputs['retrieval_all_candidate_loss'],
            )
            trainer.retrieval_denominator = 'positive_negative'
            trainer._metric_accumulators['train'].clear()

            model.zero_grad(set_to_none=True)
            novel_inputs = dict(inputs)
            novel_inputs['batch_pos_mask'] = torch.zeros(2, 2, dtype=torch.bool)
            novel_inputs['batch_neg_mask'] = torch.ones(2, 2, dtype=torch.bool)
            novel_inputs['batch_is_revisit'] = torch.zeros(2)
            novel_loss, novel_outputs = trainer.compute_loss(
                model, novel_inputs, return_outputs=True
            )
            self.assertTrue(torch.isfinite(novel_loss))
            self.assertEqual(float(novel_outputs['retrieval_loss'].detach()), 0.0)
            novel_loss.backward()
            self.assertTrue(torch.isfinite(model.scale.grad))
            trainer.log({'loss': float(novel_loss.detach())})
            novel_logged = trainer.state.log_history[-1]
            self.assertEqual(novel_logged['retrieval_loss'], 0.0)
            self.assertEqual(novel_logged['aux_direction_loss'], 0.0)
            self.assertEqual(novel_logged['aux_range_loss'], 0.0)
            self.assertTrue(all(
                not isinstance(value, float) or torch.isfinite(torch.tensor(value))
                for value in novel_logged.values()
            ))

    def test_retrieval_only_freezes_everything_except_temporal_reranker(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config()
            config.il.retrieval_only = True
            config.il.retrieval_rank_mode = 'raw_temporal'
            config.il.retrieval_denominator = 'all_candidates'
            config.il.retrieval_margin_cosine = 0.005
            config.il.retrieval_margin_weight = 1.0
            model = _RetrievalOnlyModel()
            trainer = MemNavTrainer(
                config=config,
                model=model,
                args=TrainingArguments(
                    output_dir=tmp,
                    report_to='none',
                    per_device_train_batch_size=1,
                ),
                train_dataset=_TinyDataset(),
            )
            trainable = {
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            self.assertEqual(trainable, {
                'core.retrieval.temporal_weights',
                'core.retrieval.temporal_bias',
            })
            self.assertFalse(model.core.retrieval.raw_log_temp.requires_grad)

            cosine = torch.tensor([0.94, 0.95, 0.94, 0.96])
            memory = torch.stack((
                cosine, (1.0 - cosine.square()).sqrt()
            ), -1)[None]
            inputs = {
                'retrieval_only': True,
                'batch_goal_cls': torch.tensor([[1.0, 0.0]]),
                'batch_mem_cls': memory,
                'batch_cand_mask': torch.ones(1, 4, dtype=torch.bool),
                # Supported plateau is positive; the isolated raw peak is not.
                'batch_pos_mask': torch.tensor([[True, True, True, False]]),
                'batch_neg_mask': torch.tensor([[False, False, False, True]]),
                'batch_labels': torch.ones(1),
            }
            loss, outputs = trainer.compute_loss(
                model, inputs, return_outputs=True
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(outputs['retrieval_strict_top1']), 0.0)
            self.assertGreater(
                float(outputs['retrieval_margin_loss'].detach()), 0.0
            )
            loss.backward()
            self.assertGreater(
                float(model.core.retrieval.temporal_weights.grad.abs().sum()),
                0.0,
            )
            self.assertIsNotNone(model.core.retrieval.temporal_bias.grad)
            self.assertIsNone(model.core.retrieval.raw_log_temp.grad)

    def test_anchor_teacher_forcing_schedule_and_endpoint_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config()
            config.il.anchor_teacher_forcing_start = 1.0
            config.il.anchor_teacher_forcing_end = 0.25
            config.il.anchor_teacher_forcing_decay_steps = 10
            trainer = MemNavTrainer(
                config=config,
                model=_TinyModel(),
                args=TrainingArguments(
                    output_dir=tmp,
                    report_to='none',
                    per_device_train_batch_size=1,
                ),
                train_dataset=_TinyDataset(),
            )
            self.assertAlmostEqual(trainer._anchor_teacher_forcing_probability(), 1.0)
            trainer.state.global_step = 5
            self.assertAlmostEqual(trainer._anchor_teacher_forcing_probability(), 0.625)
            trainer.state.global_step = 20
            self.assertAlmostEqual(trainer._anchor_teacher_forcing_probability(), 0.25)

            positives = torch.tensor([
                [True, False], [False, False], [False, True]
            ])
            torch.testing.assert_close(
                trainer._sample_anchor_teacher_mask(positives, 1.0),
                torch.tensor([True, False, True]),
            )
            self.assertFalse(bool(
                trainer._sample_anchor_teacher_mask(positives, 0.0).any()
            ))
            first = trainer._sample_anchor_teacher_mask(
                positives, 0.5, seed=123
            )
            second = trainer._sample_anchor_teacher_mask(
                positives, 0.5, seed=123
            )
            torch.testing.assert_close(first, second)

            torch.manual_seed(99)
            expected_global_draw = torch.rand(3)
            torch.manual_seed(99)
            trainer._sample_anchor_teacher_mask(positives, 0.5, seed=123)
            torch.testing.assert_close(torch.rand(3), expected_global_draw)

    def test_route_direction_loss_is_label_side_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config()
            config.il.use_route_sketch = True
            config.il.route_horizons = (1, 2)
            config.il.w_route_direction = 0.2
            model = _RouteLossModel()
            trainer = MemNavTrainer(
                config=config,
                model=model,
                args=TrainingArguments(
                    output_dir=tmp,
                    report_to='none',
                    per_device_train_batch_size=2,
                ),
                train_dataset=_TinyDataset(),
            )
            labels = torch.tensor([
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            ])
            inputs = {
                'batch_labels': labels,
                'batch_pos_mask': torch.tensor([
                    [True, False], [False, False]
                ]),
                'batch_neg_mask': torch.tensor([
                    [False, True], [True, True]
                ]),
                'batch_is_revisit': torch.tensor([1.0, 0.0]),
                'batch_goal_rel_pose': torch.tensor([
                    [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]
                ]),
                'batch_goal_range_code': torch.tensor([0.2, 0.5]),
                'batch_goal_range_steps': torch.tensor([6.4, 16.0]),
                'batch_gt_prefix_step_m': torch.tensor([0.25, 0.25]),
                'batch_goal_rel_rotation': torch.eye(3).repeat(2, 1, 1),
                'batch_goal_j': torch.tensor([0, -1]),
                'cur_steps': [319, 1024],
                'goal_steps': [400, 1100],
            }
            model.train()
            loss, output = trainer.compute_loss(
                model, inputs, return_outputs=True
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(
                float(output['route_direction_loss'].detach()), 0.0
            )
            loss.backward()
            self.assertTrue(torch.isfinite(model.route_logits.grad).all())
            self.assertGreater(float(model.route_logits.grad.abs().sum()), 0.0)

    def test_action_safe_range_projection_removes_conflict_and_caps_norm(self):
        action = [torch.tensor([1.0, 0.0])]
        conflicting_range = [torch.tensor([-1.0, 2.0])]
        corrected, diagnostics = project_auxiliary_gradients(
            action, conflicting_range, max_norm_ratio=0.25
        )
        torch.testing.assert_close(corrected[0], torch.tensor([0.0, 0.25]))
        self.assertLess(float(diagnostics['raw_cosine']), 0.0)
        self.assertEqual(float(diagnostics['conflict']), 1.0)
        self.assertAlmostEqual(
            float(diagnostics['corrected_norm_ratio']), 0.25
        )
        self.assertGreaterEqual(float(torch.dot(action[0], corrected[0])), 0.0)

        aligned_range = [torch.tensor([2.0, 0.0])]
        aligned_corrected, aligned_diagnostics = project_auxiliary_gradients(
            action, aligned_range, max_norm_ratio=0.25
        )
        torch.testing.assert_close(
            aligned_corrected[0], torch.tensor([0.25, 0.0])
        )
        self.assertEqual(float(aligned_diagnostics['conflict']), 0.0)

    def test_zero_value_correction_replaces_only_shared_range_gradient(self):
        model = _AdapterGradientModel()
        weight = model.core.revisit_merge.rel_adapter.weight
        action_loss = weight[0, 0]
        range_loss = -10.0 * weight[0, 0] + 10.0 * weight[0, 1]
        controller = SimpleNamespace(
            w_aux_range=0.2,
            aux_range_grad_cap_ratio=0.25,
        )
        correction, diagnostics = MemNavTrainer._action_safe_range_correction(
            controller, model, action_loss, range_loss
        )
        uncorrected_value = action_loss + controller.w_aux_range * range_loss
        corrected_value = uncorrected_value + correction
        torch.testing.assert_close(corrected_value, uncorrected_value)
        corrected_value.backward()
        expected = torch.zeros_like(weight)
        expected[0, 0] = 1.0
        expected[0, 1] = 0.25
        torch.testing.assert_close(weight.grad, expected)
        self.assertEqual(float(diagnostics['conflict']), 1.0)

    def test_compact_checkpoint_round_trip_and_dataset_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'checkpoint-5'
            trainer = self._trainer(tmp)
            expected = {
                key: value.detach().clone() for key, value in trainer.model.state_dict().items()
            }
            trainer.save_model(str(checkpoint))
            metadata = json.loads(
                (checkpoint / 'memnav_metadata.json').read_text(encoding='utf-8')
            )
            self.assertEqual(metadata['training_objective']['w_aux_range'], 0.2)
            self.assertEqual(
                metadata['training_objective']['route_lr_multiplier'], 10.0
            )
            self.assertEqual(
                metadata['training_objective']['aux_range_grad_cap_ratio'], 0.0
            )
            self.assertEqual(
                metadata['training_objective']['anchor_teacher_forcing_start'],
                1.0,
            )
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.zero_()
            trainer._load_from_checkpoint(str(checkpoint))
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, expected[key])

            trainer.w_aux_range = 0.3
            with self.assertRaisesRegex(ValueError, 'Training objective changed'):
                trainer._load_from_checkpoint(str(checkpoint))
            trainer.w_aux_range = 0.2

            trainer.train_dataset.dataset_fingerprint = 'different-population'
            with self.assertRaisesRegex(ValueError, 'Dataset fingerprint changed'):
                trainer._load_from_checkpoint(str(checkpoint))

    def test_gate_calibration_has_separate_lr_and_no_weight_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config()
            config.il.gate_lr_multiplier = 7.0
            args = TrainingArguments(
                output_dir=tmp,
                report_to='none',
                per_device_train_batch_size=1,
                learning_rate=2e-4,
                weight_decay=0.1,
            )
            model = _OptimizerModel()
            trainer = MemNavTrainer(
                config=config,
                model=model,
                args=args,
                train_dataset=_TinyDataset(),
            )
            optimizer = trainer.create_optimizer()
            gate_groups = [
                group for group in optimizer.param_groups
                if group.get('memnav_group') == 'gate_calibration'
            ]
            self.assertEqual(len(gate_groups), 1)
            gate_group = gate_groups[0]
            self.assertAlmostEqual(gate_group['lr'], 1.4e-3)
            self.assertEqual(gate_group['weight_decay'], 0.0)
            expected_ids = {
                id(model.core.retrieval.gate_log_slope),
                id(model.core.retrieval.gate_bias),
            }
            self.assertEqual(
                {id(parameter) for parameter in gate_group['params']}, expected_ids
            )
            pose_groups = [
                group for group in optimizer.param_groups
                if group.get('memnav_group') == 'pose_reliability_calibration'
            ]
            self.assertEqual(len(pose_groups), 1)
            self.assertAlmostEqual(pose_groups[0]['lr'], 1.0e-3)
            self.assertEqual(pose_groups[0]['weight_decay'], 0.0)
            self.assertEqual(
                {id(parameter) for parameter in pose_groups[0]['params']},
                {
                    id(parameter) for parameter in
                    model.core.revisit_merge.pose_encoder.reliability_head.parameters()
                },
            )
            retrieval_groups = [
                group for group in optimizer.param_groups
                if str(group.get('memnav_group', '')).startswith(
                    'retrieval_rank'
                )
            ]
            self.assertEqual(len(retrieval_groups), 2)
            self.assertTrue(all(
                abs(group['lr'] - 2e-4) < 1e-12
                for group in retrieval_groups
            ))

    def test_route_sketch_has_separate_scaled_optimizer_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config()
            config.il.use_route_sketch = True
            config.il.route_horizons = (1, 2)
            config.il.route_lr_multiplier = 6.0
            model = _RouteOptimizerModel()
            trainer = MemNavTrainer(
                config=config,
                model=model,
                args=TrainingArguments(
                    output_dir=tmp,
                    report_to='none',
                    per_device_train_batch_size=1,
                    learning_rate=2e-4,
                    weight_decay=0.1,
                ),
                train_dataset=_TinyDataset(),
            )
            optimizer = trainer.create_optimizer()
            groups = [
                group for group in optimizer.param_groups
                if str(group.get('memnav_group', '')).startswith('route_sketch')
            ]
            self.assertEqual(len(groups), 2)
            self.assertTrue(all(
                abs(group['lr'] - 1.2e-3) < 1e-12 for group in groups
            ))
            no_decay = next(
                group for group in groups
                if group['memnav_group'] == 'route_sketch_no_decay'
            )
            self.assertEqual(no_decay['weight_decay'], 0.0)
            self.assertIn(
                id(model.core.route_sketch.residual_scale),
                {id(parameter) for parameter in no_decay['params']},
            )

    def test_route_checkpoint_upgrade_only_fills_fully_legacy_namespace(self):
        stub = _CheckpointUpgradeStub()
        upgraded = MemNavPolicy.upgrade_checkpoint_state_dict(stub, {})
        expected = stub.state_dict()
        self.assertEqual(set(upgraded), set(expected))
        self.assertTrue(all(torch.equal(upgraded[key], value) for key, value in expected.items()))

        partial = {'core.route_sketch.residual_scale': torch.zeros(3)}
        upgraded_partial = MemNavPolicy.upgrade_checkpoint_state_dict(stub, partial)
        self.assertEqual(set(upgraded_partial), set(partial))

    def test_route_checkpoint_upgrade_rejects_different_horizons(self):
        source = _CheckpointUpgradeStub(horizons=(1, 8, 24)).state_dict()
        target = _CheckpointUpgradeStub(horizons=(2, 8, 24))
        with self.assertRaisesRegex(ValueError, 'horizons do not match'):
            MemNavPolicy.upgrade_checkpoint_state_dict(target, source)

    def test_compact_checkpoint_guards_fixed_eval_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'checkpoint-5'
            trainer = self._trainer(tmp, eval_fingerprint='eval-population-v1')
            trainer.save_model(str(checkpoint))
            trainer.eval_dataset.dataset_fingerprint = 'eval-population-v2'
            with self.assertRaisesRegex(ValueError, 'Evaluation fingerprint changed'):
                trainer._load_from_checkpoint(str(checkpoint))

    def test_resume_requires_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'checkpoint-5'
            trainer = self._trainer(tmp)
            trainer.save_model(str(checkpoint))
            (checkpoint / 'memnav_metadata.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing .*memnav_metadata.json'):
                trainer._load_from_checkpoint(str(checkpoint))

    def test_component_logs_are_windowed_and_not_double_prefixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer(tmp)
            trainer._accumulate('action_loss', torch.tensor(1.0), 2.0)
            trainer._accumulate('action_loss', torch.tensor(3.0), 2.0)
            trainer._accumulate('aux_pred_y_mean', torch.tensor(2.0), 1.0)
            trainer._accumulate('aux_pred_y_sq_mean', torch.tensor(5.0), 1.0)
            trainer.log({'loss': 4.0})
            logged = trainer.state.log_history[-1]
            self.assertEqual(logged['action_loss'], 2.0)
            self.assertEqual(logged['aux_pred_y_std'], 1.0)
            self.assertNotIn('train/action_loss', logged)

            trainer._accumulate('action_loss', torch.tensor(5.0), 1.0, phase='eval')
            trainer.log({'eval_loss': 6.0})
            eval_logged = trainer.state.log_history[-1]
            self.assertEqual(eval_logged['eval_action_loss'], 5.0)
            self.assertNotIn('action_loss', eval_logged)


if __name__ == '__main__':
    unittest.main()
