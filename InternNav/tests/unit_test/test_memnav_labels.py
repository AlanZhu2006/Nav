import torch

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.dataset.memnav_labels import build_retrieval_label
from internnav.model.basemodel.memnav.retrieval import RetrievalHead


def test_revisit_uses_metadata_kind_and_strong_positives():
    label, reason = build_retrieval_label(
        [0.9, 0.8, 0.05, 0.3, 0.6], 'revisit', 0.5, 0.1, anchor_margin=2)

    assert reason is None
    assert label.null_pos is False
    assert label.candidate_mask.tolist() == [False, False, True, True, True]
    assert label.pos_mask.tolist() == [False, False, False, False, True]
    assert label.neg_mask.tolist() == [False, False, True, False, False]


def test_weak_revisit_is_skipped_not_relabeled_as_novel():
    label, reason = build_retrieval_label(
        [0.9, 0.8, 0.05, 0.3, 0.49], 'revisit', 0.5, 0.1, anchor_margin=2)

    assert label is None
    assert reason == 'weak_revisit'


def test_novel_uses_null_even_with_ignored_band_frames():
    label, reason = build_retrieval_label(
        [0.8, 0.7, 0.02, 0.2, 0.1], 'novel', 0.5, 0.1, anchor_margin=2)

    assert reason is None
    assert label.null_pos is True
    assert not label.pos_mask.any()
    assert label.neg_mask.tolist() == [False, False, True, False, True]


def test_novel_with_valid_positive_is_rejected():
    label, reason = build_retrieval_label(
        [0.0, 0.0, 0.6], 'novel', 0.5, 0.1, anchor_margin=2)

    assert label is None
    assert reason == 'novel_has_positive'


def test_current_frame_can_be_a_retrieval_candidate():
    label, reason = build_retrieval_label(
        [0.0, 0.0, 0.1, 0.7], 'revisit', 0.5, 0.1, anchor_margin=2)

    assert reason is None
    assert label.candidate_mask[-1]
    assert label.pos_mask[-1]


def _collate_item(length, candidate_mask):
    return {
        'mem_cls': torch.zeros(length, 4),
        'pos_mask': torch.zeros(length, dtype=torch.bool),
        'neg_mask': torch.zeros(length, dtype=torch.bool),
        'candidate_mask': torch.tensor(candidate_mask, dtype=torch.bool),
        'null_pos': torch.tensor(True),
        'is_revisit': torch.tensor(0.0),
        'pred_actions': torch.zeros(2, 3),
        'goal_rel_pose': torch.zeros(3),
        'goal_image': torch.zeros(3, 2, 2),
        'window_images': torch.zeros(2, 3, 2, 2),
        'cache_path': 'cache.npz',
        'rgb_dir': 'rgb',
        'cur_step': length - 1,
        'goal_step': length,
    }


def test_collate_masks_early_candidates_and_padding():
    batch = memnav_collate_fn([
        _collate_item(4, [False, False, True, True]),
        _collate_item(3, [False, True, True]),
    ])

    assert batch['batch_mem_mask'].tolist() == [
        [False, False, True, True],
        [False, True, True, False],
    ]


def test_retrieval_head_never_selects_a_masked_frame():
    head = RetrievalHead(dino_dim=8, proj_dim=4)
    goal = torch.randn(2, 8)
    memory = torch.randn(2, 5, 8)
    candidate_mask = torch.tensor([
        [False, False, True, True, True],
        [False, True, True, False, False],
    ])

    match, gate, logits = head(goal, memory, candidate_mask)

    assert match[0] >= 2
    assert match[1] in (1, 2)
    assert torch.isneginf(logits[:, :-1][~candidate_mask]).all()
    assert torch.isfinite(gate).all()


def test_retrieval_head_rejects_an_empty_candidate_set():
    head = RetrievalHead(dino_dim=8, proj_dim=4)

    try:
        head(torch.randn(1, 8), torch.randn(1, 3, 8), torch.zeros(1, 3, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError('an all-masked retrieval sample must fail')
