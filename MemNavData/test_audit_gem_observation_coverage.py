"""Coverage scoring must not invent evidence across angles, floors or time."""
import numpy as np
import pytest

from MemNavData.audit_gem_observation_coverage import coverage_panel


def panel(history, query, eligible, candidates):
    return coverage_panel(history, query, eligible, candidates,
        radii=[1.0], yaws=[30.0, 60.0, 180.0])


def test_yaw_wrap_is_circular():
    value = panel([[0, 0, 0, np.deg2rad(179)]],
        [0, 0, 0, np.deg2rad(-179)], [0], [0])
    assert value['nearest']['yaw_difference_deg'] == pytest.approx(2.)
    assert value['by_radius']['1.0']['angles']['30.0']['shortlist_frames'] == 1


def test_other_floor_is_not_a_nearby_observation():
    value = panel([[0, 3, 0, 0]], [0, 0, 0, 0], [0], [0])
    assert value['nearest']['distance_3d_m'] == 3.
    assert value['by_radius']['1.0']['nearby_history_frames'] == 0


def test_nearby_observations_can_exist_but_be_absent_from_shortlist():
    value = panel([[0, 0, 0, 0], [2, 0, 0, 0]], [0, 0, 0, 0], [0, 1], [1])
    counts = value['by_radius']['1.0']['angles']['60.0']
    assert counts == dict(history_frames=1, shortlist_frames=0)


def test_newer_observation_cannot_create_old_history_coverage():
    value = panel([[4, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0], [0], [0])
    assert value['nearest']['frame'] == 0
    assert value['by_radius']['1.0']['nearby_history_frames'] == 0
    with pytest.raises(ValueError, match='eligible'):
        panel([[4, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0], [0], [1])


def test_same_place_opposite_view_is_distinct_from_no_place_observation():
    value = panel([[0, 0, 0, np.pi]], [0, 0, 0, 0], [0], [0])
    counts = value['by_radius']['1.0']
    assert counts['nearby_history_frames'] == 1
    assert counts['angles']['60.0']['history_frames'] == 0
    assert counts['angles']['180.0']['history_frames'] == 1


def test_empty_eligible_set_remains_in_the_denominator():
    value = panel([[0, 0, 0, 0]], [0, 0, 0, 0], [], [])
    assert value['eligible_frames'] == 0 and value['nearest'] is None
    assert value['by_radius']['1.0']['angles']['60.0']['history_frames'] == 0
