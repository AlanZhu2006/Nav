import numpy as np
import pandas as pd
import pytest

from MemNavData.audit_phase_b_decision_units import (
    _cluster_bootstrap_interval,
    pairwise_auc,
    pooled_auc,
)


def test_pairwise_auc_orders_and_ties() -> None:
    assert pairwise_auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert pairwise_auc([0.1], [0.9]) == 0.0
    assert pairwise_auc([0.5], [0.5]) == 0.5


def test_pooled_auc_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pooled_auc([0.1, 0.2], [True])


def test_scene_bootstrap_resamples_whole_clusters() -> None:
    frame = pd.DataFrame({
        "scene": ["a", "a", "b", "b"],
        "delta": [1.0, 1.0, -1.0, -1.0],
        "pairs": [1, 3, 2, 2],
    })
    interval = _cluster_bootstrap_interval(
        frame,
        value_column="delta",
        weight_column="pairs",
        resamples=200,
        seed=7,
    )
    assert interval["scene_clusters"] == 2
    assert interval["resamples"] == 200
    assert np.isfinite([
        interval["lower_95"], interval["median"], interval["upper_95"]
    ]).all()
    assert interval["lower_95"] <= interval["median"] <= interval["upper_95"]
