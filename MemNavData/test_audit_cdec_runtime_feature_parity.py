import pytest

from MemNavData.audit_cdec_runtime_feature_parity import (
    deterministic_sessions,
)


def test_parity_sample_is_label_blind_stable_and_unique():
    universe = ["scene/session_2", "scene/session_1", "scene/session_3"]
    first = deterministic_sessions(universe + [universe[0]], 2)
    second = deterministic_sessions(reversed(universe), 2)
    assert first == second
    assert len(first) == len(set(first)) == 2


def test_parity_sample_count_fails_closed():
    with pytest.raises(ValueError):
        deterministic_sessions(["a"], 2)
