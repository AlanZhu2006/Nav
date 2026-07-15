import pytest
import torch

from internnav.model.basemodel.memnav.metrics import (
    compute_gate_threshold_sweep,
    compute_memory_length_diagnostics,
    compute_memnav_batch_records,
    compute_memnav_batch_totals,
    finalize_memnav_metrics,
)


def _synthetic_batch():
    return {
        'batch_is_revisit': torch.tensor([1.0, 0.0]),
        'batch_null_pos': torch.tensor([False, True]),
        'batch_mem_mask': torch.tensor([[True, True], [True, True]]),
        'batch_pos_mask': torch.tensor([[True, False], [False, False]]),
        'batch_neg_mask': torch.tensor([[False, True], [True, True]]),
        'batch_goal_rel_pose': torch.zeros(2, 3),
    }


def _synthetic_outputs():
    return {
        'ret_logits': torch.tensor([[3.0, 0.0, -1.0], [0.0, 1.0, 2.0]]),
        'revisit_gate': torch.tensor([0.9, 0.2]),
        'noise': torch.zeros(2, 1, 1),
        'noise_ng': torch.ones(2, 1, 1),
        'noise_mg': torch.full((2, 1, 1), 2.0),
        'aux_pose': torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
    }


def test_memnav_metrics_respect_multi_positive_and_semantic_modes():
    totals = compute_memnav_batch_totals(_synthetic_outputs(), _synthetic_batch())
    metrics = finalize_memnav_metrics(totals)

    assert metrics['num_revisit'] == 1
    assert metrics['num_novel'] == 1
    assert metrics['action_loss'] == pytest.approx(2.5)
    assert metrics['retrieval_accuracy'] == pytest.approx(1.0)
    assert metrics['revisit_match_accuracy'] == pytest.approx(1.0)
    assert metrics['novel_null_accuracy'] == pytest.approx(1.0)
    assert metrics['revisit_pred_positive_fraction'] == pytest.approx(1.0)
    assert metrics['revisit_pred_negative_fraction'] == pytest.approx(0.0)
    assert metrics['revisit_pred_ignored_fraction'] == pytest.approx(0.0)
    assert metrics['revisit_pred_null_fraction'] == pytest.approx(0.0)
    assert metrics['novel_pred_positive_fraction'] == pytest.approx(0.0)
    assert metrics['novel_pred_negative_fraction'] == pytest.approx(0.0)
    assert metrics['novel_pred_ignored_fraction'] == pytest.approx(0.0)
    assert metrics['novel_pred_null_fraction'] == pytest.approx(1.0)
    assert metrics['revisit_top_real_match_accuracy'] == pytest.approx(1.0)
    assert metrics['revisit_top_real_negative_fraction'] == pytest.approx(0.0)
    assert metrics['revisit_top_real_ignored_fraction'] == pytest.approx(0.0)
    assert metrics['gate_accuracy_at_0_5'] == pytest.approx(1.0)
    assert metrics['gate_revisit_accuracy_at_0_5'] == pytest.approx(1.0)
    assert metrics['gate_novel_accuracy_at_0_5'] == pytest.approx(1.0)
    assert metrics['gate_revisit'] == pytest.approx(0.9)
    assert metrics['gate_novel'] == pytest.approx(0.2)
    assert metrics['gate_separation'] == pytest.approx(0.7)
    assert metrics['aux_pose_mse_revisit'] == pytest.approx(14.0 / 3.0)
    assert metrics['loss'] == pytest.approx(
        2.5 + metrics['retrieval_loss'] + 0.5 * (14.0 / 3.0)
    )


def test_memnav_metrics_reject_label_semantic_mismatch():
    batch = _synthetic_batch()
    batch['batch_null_pos'][0] = True

    with pytest.raises(ValueError, match='null targets'):
        compute_memnav_batch_totals(_synthetic_outputs(), batch)


def test_memnav_metrics_classify_retrieval_failures():
    revisit = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    batch = {
        'batch_is_revisit': revisit,
        'batch_null_pos': ~revisit.bool(),
        'batch_mem_mask': torch.ones(5, 3, dtype=torch.bool),
        'batch_pos_mask': torch.tensor([
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]),
        'batch_neg_mask': torch.tensor([
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
        ]),
        'batch_goal_rel_pose': torch.zeros(5, 3),
    }
    outputs = {
        'ret_logits': torch.tensor([
            [0.0, 3.0, 1.0, -1.0],  # revisit: explicit negative
            [0.0, 1.0, 3.0, -1.0],  # revisit: ignored gray band
            [0.0, 1.0, 2.0, 3.0],   # revisit: null
            [3.0, 0.0, 1.0, -1.0],  # novel: explicit negative
            [0.0, 1.0, 3.0, -1.0],  # novel: ignored gray band
        ]),
        'revisit_gate': torch.zeros(5),
        'noise': torch.zeros(5, 1, 1),
        'noise_ng': torch.zeros(5, 1, 1),
        'noise_mg': torch.zeros(5, 1, 1),
        'aux_pose': torch.zeros(5, 3),
    }

    metrics = finalize_memnav_metrics(compute_memnav_batch_totals(outputs, batch))

    assert metrics['revisit_pred_positive_fraction'] == pytest.approx(0.0)
    assert metrics['revisit_pred_negative_fraction'] == pytest.approx(1.0 / 3.0)
    assert metrics['revisit_pred_ignored_fraction'] == pytest.approx(1.0 / 3.0)
    assert metrics['revisit_pred_null_fraction'] == pytest.approx(1.0 / 3.0)
    assert metrics['novel_pred_positive_fraction'] == pytest.approx(0.0)
    assert metrics['novel_pred_negative_fraction'] == pytest.approx(0.5)
    assert metrics['novel_pred_ignored_fraction'] == pytest.approx(0.5)
    assert metrics['novel_pred_null_fraction'] == pytest.approx(0.0)
    assert metrics['revisit_top_real_match_accuracy'] == pytest.approx(0.0)
    assert metrics['revisit_top_real_negative_fraction'] == pytest.approx(1.0 / 3.0)
    assert metrics['revisit_top_real_ignored_fraction'] == pytest.approx(2.0 / 3.0)
    assert metrics['gate_accuracy_at_0_5'] == pytest.approx(0.4)
    assert metrics['gate_revisit_accuracy_at_0_5'] == pytest.approx(0.0)
    assert metrics['gate_novel_accuracy_at_0_5'] == pytest.approx(1.0)


def test_memnav_per_sample_records_and_memory_buckets():
    batch = _synthetic_batch()
    batch.update({
        'cur_steps': [319, 1024],
        'goal_steps': [400, 1100],
        'cache_paths': ['revisit.npz', 'novel.npz'],
    })
    outputs = _synthetic_outputs()
    outputs['noise_oracle_gate'] = torch.tensor([[[1.0]], [[3.0]]])

    records = compute_memnav_batch_records(outputs, batch)

    assert records[0]['memory_length'] == 320
    assert records[0]['joint_outcome'] == 'positive'
    assert records[0]['top_real_outcome'] == 'positive'
    assert records[0]['max_real_null_margin'] == pytest.approx(4.0)
    assert records[0]['oracle_gate_action_loss'] == pytest.approx(1.0)
    assert records[1]['memory_length'] == 1025
    assert records[1]['joint_outcome'] == 'null'
    assert records[1]['top_real_outcome'] == 'negative'
    assert records[1]['max_real_null_margin'] == pytest.approx(-1.0)
    assert records[1]['oracle_gate_action_loss'] == pytest.approx(9.0)

    buckets = compute_memory_length_diagnostics(records)
    assert buckets['up_to_320']['num_samples'] == 1
    assert buckets['up_to_320']['revisit_top_real_match_accuracy'] == pytest.approx(1.0)
    assert buckets['321_to_1024']['num_samples'] == 0
    assert buckets['1025_to_2048']['num_samples'] == 1
    assert buckets['1025_to_2048']['novel_null_accuracy'] == pytest.approx(1.0)


def test_gate_threshold_sweep_finds_balanced_separator():
    records = [
        {'is_revisit': True, 'gate': 0.9},
        {'is_revisit': True, 'gate': 0.8},
        {'is_revisit': False, 'gate': 0.7},
        {'is_revisit': False, 'gate': 0.1},
    ]

    sweep = compute_gate_threshold_sweep(records)

    assert sweep['reference']['balanced_accuracy'] == pytest.approx(0.75)
    assert 0.7 < sweep['best']['threshold'] < 0.8
    assert sweep['best']['balanced_accuracy'] == pytest.approx(1.0)
    assert sweep['best']['revisit_recall'] == pytest.approx(1.0)
    assert sweep['best']['novel_recall'] == pytest.approx(1.0)
