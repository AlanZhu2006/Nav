from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from transformers import TrainingArguments

from internnav.trainer.memnav_trainer import MemNavTrainer


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
            w_pose_reliability=0.2,
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
                'batch_is_revisit': torch.tensor([1.0, 0.0]),
                'batch_goal_rel_pose': torch.tensor([
                    [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]
                ]),
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
            loss.backward()
            self.assertTrue(torch.isfinite(model.scale.grad))
            trainer.log({'loss': float(loss.detach())})
            logged = trainer.state.log_history[-1]
            self.assertAlmostEqual(logged['action_noise_mse_theta'], 1.0)
            self.assertGreater(logged['action_target_theta_std'], 0.0)
            self.assertAlmostEqual(logged['gate_revisit_recall'], 1.0)
            self.assertAlmostEqual(logged['gate_novel_recall'], 1.0)
            self.assertGreater(logged['gate_sep'], 0.0)
            self.assertAlmostEqual(logged['goal_A_fraction'], 0.5)
            self.assertAlmostEqual(logged['goal_B_fraction'], 0.5)
            self.assertIn('action_loss_goal_A', logged)
            self.assertIn('action_loss_goal_B', logged)
            self.assertIn('aux_direction_err_deg_goal_B_revisit', logged)
            self.assertIn('aux_mse_y_anchor_gap_256_511', logged)

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
            self.assertTrue(all(
                not isinstance(value, float) or torch.isfinite(torch.tensor(value))
                for value in novel_logged.values()
            ))

    def test_compact_checkpoint_round_trip_and_dataset_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'checkpoint-5'
            trainer = self._trainer(tmp)
            expected = {
                key: value.detach().clone() for key, value in trainer.model.state_dict().items()
            }
            trainer.save_model(str(checkpoint))
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.zero_()
            trainer._load_from_checkpoint(str(checkpoint))
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, expected[key])

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
