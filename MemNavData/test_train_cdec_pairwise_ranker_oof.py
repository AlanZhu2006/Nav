import numpy as np

from MemNavData.train_cdec_pairwise_ranker_oof import (
    exact_mcnemar,
    pairwise_differences,
    parse_c_grid,
)


def test_pairwise_differences_use_only_positive_minus_known_negative():
    features = np.asarray([[[3.0], [1.0], [100.0]]])
    labels = np.asarray([[1, 0, -1]])
    result = pairwise_differences(features, labels, np.asarray([0]))
    np.testing.assert_allclose(result, [[2.0]])


def test_regularization_grid_is_sorted_unique_and_requires_search():
    assert parse_c_grid(["3", "0.1", "3"]) == [0.1, 3.0]


def test_exact_mcnemar_handles_ties_and_direction_symmetrically():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(10, 8) == exact_mcnemar(8, 10)
