import numpy as np
import pytest

from MemNavData.audit_pt1_multinovel_capacity import (
    distance_bin, distances_to_recorded_path, route_on_floor, trajectory_metrics,
)
from MemNavData.probe_pt1_multinovel_support import is_unsupported, select_candidates


def test_distance_bins_cover_common_range_without_overlap():
    assert [distance_bin(v) for v in (1.99, 2, 3.99, 4, 5.99, 6, 9, 9.01)] == [
        None, "2_to_4", "2_to_4", "4_to_6", "4_to_6", "6_to_9", "6_to_9", None]


def test_transition_lengths_are_not_dropped_at_leg_boundaries():
    positions = np.column_stack((np.arange(9), np.zeros((9, 2))))
    result = trajectory_metrics(positions, [3, 6])
    assert result["leg_path_m"] == [2., 3., 3.]
    assert sum(result["leg_path_m"]) == result["recorded_path_m"] == 8.


def test_long_return_is_not_new_spatial_exploration():
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                          [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]])
    result = trajectory_metrics(positions, [3, 5])
    assert result["recorded_path_m"] == 8.
    assert result["endpoint_displacement_m"] == 0.
    assert result["relative_to_prior_path"]["C"]["path_fraction_outside_prior_1m_tube"] == 0.


def test_full_prior_history_used_in_path_distance():
    history = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    values = distances_to_recorded_path([[.1, 0, 0], [6.2, 0, 0]], history)
    np.testing.assert_allclose(values, [.1, .2])


def test_different_floors_are_not_projected_onto_each_other():
    values = distances_to_recorded_path([[0, 3, 0]], [[0, 0, 0]])
    assert values[0] == 3.
    assert not route_on_floor([[0, 0, 0], [1, 1, 0], [2, 0, 0]], 0)
    assert route_on_floor([[0, 0, 0], [1, .1, 0], [2, 0, 0]], 0)


def test_empty_history_and_no_nan_leak_into_measured_values():
    assert np.isinf(distances_to_recorded_path([[1, 0, 0]], np.empty((0, 3)))[0])
    with pytest.raises(ValueError):
        trajectory_metrics([[0, 0, 0], [np.nan, 0, 0], [2, 0, 0]], [1, 2])


def test_empty_depth_is_not_falsely_accepted_as_novel():
    assert not is_unsupported(0, [0., 0.])
    assert not is_unsupported(10, [.1, 0.])
    assert is_unsupported(10, [.099, 0.])
    assert is_unsupported(10, [])


def test_visual_candidate_selection_ignores_scores_and_follows_proposal_order():
    candidates = [dict(proposal_index=i, distance_bin="2_to_4", direction="front",
                       spurious_policy_success=i % 2) for i in (4, 1, 7, 0)]
    assert [c["proposal_index"] for c in select_candidates(candidates)] == [0, 1]
