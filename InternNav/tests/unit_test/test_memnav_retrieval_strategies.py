import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts' / 'eval' / 'diag_memnav_retrieval_strategies.py'
)
SPEC = importlib.util.spec_from_file_location('diag_memnav_retrieval_strategies', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_masked_argmax_never_selects_padding():
    scores = torch.tensor([[100.0, 1.0, 2.0]])
    candidate = torch.tensor([[False, True, True]])
    assert MODULE.masked_argmax(scores, candidate).tolist() == [2]


def test_temporal_mass_prefers_supported_band_over_isolated_peak():
    logits = torch.tensor([[0.0, 4.0, 0.0, 3.5, 3.5, 3.5, 0.0]])
    candidate = torch.ones_like(logits, dtype=torch.bool)
    assert MODULE.masked_argmax(logits, candidate).tolist() == [1]
    anchor = MODULE.temporal_mass_anchor(logits, candidate, radius=1)
    assert anchor.tolist() == [4]


def test_classification_and_nearest_positive_distance():
    anchor = torch.tensor([2, 3, 4])
    positive = torch.tensor([
        [False, False, True, False, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
    ])
    negative = torch.tensor([
        [True, False, False, False, False],
        [True, False, False, False, False],
        [False, False, False, False, True],
    ])
    assert MODULE.classify_anchor(anchor, positive, negative) == [
        'positive', 'gray', 'negative'
    ]
    assert MODULE.nearest_positive_distance(anchor, positive).tolist() == [0, 2, -1]


def test_temporal_mass_rejects_empty_candidate_row():
    logits = torch.zeros(1, 3)
    candidate = torch.zeros(1, 3, dtype=torch.bool)
    try:
        MODULE.temporal_mass_anchor(logits, candidate, radius=1)
    except ValueError as error:
        assert 'at least one candidate' in str(error)
    else:
        raise AssertionError('expected empty candidate row to fail closed')


def test_candidate_zscore_ignores_padding_and_normalizes_candidates():
    scores = torch.tensor([[1.0, 3.0, 1000.0], [4.0, 4.0, -1000.0]])
    candidate = torch.tensor([[True, True, False], [True, True, False]])
    zscore = MODULE.candidate_zscore(scores, candidate)
    torch.testing.assert_close(zscore[0, :2], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(zscore[1, :2], torch.zeros(2))
    assert zscore[0, 2] == torch.finfo(scores.dtype).min


def test_cross_rerank_uses_only_shortlisted_candidates():
    shortlist = torch.tensor([[9.0, 8.0, 7.0, 6.0]])
    rerank = torch.tensor([[0.0, 1.0, 100.0, 200.0]])
    candidate = torch.ones_like(shortlist, dtype=torch.bool)
    anchor = MODULE.topk_rerank_anchor(shortlist, rerank, candidate, 2)
    assert anchor.item() == 1
