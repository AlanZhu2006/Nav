import numpy as np
import pytest

from MemNavData.longrange_local_tangent_attribution import (
    floor_aware_goal_distance_m,
    first_local_tangent_world,
    realized_planar_translation_m,
    signed_pointgoal_heading_deg,
    tangent_requires_alignment,
)


def test_first_local_tangent_preserves_order_and_height() -> None:
    path = [
        [0.0, 0.0, 0.0],
        [0.0, 0.2, -0.10],
        [0.35, 0.4, -0.15],
        [2.0, 1.0, 2.0],
    ]
    target, receipt = first_local_tangent_world(path, min_planar_m=0.30)
    assert np.allclose(target, [0.35, 0.4, -0.15])
    assert receipt["tangent_path_index"] == 2
    assert receipt["tangent_planar_baseline_m"] == pytest.approx(
        np.hypot(0.35, 0.15))
    assert receipt["tangent_vertical_delta_m"] == pytest.approx(0.4)


def test_first_local_tangent_rejects_vertical_only_path() -> None:
    with pytest.raises(ValueError, match="no planar tangent"):
        first_local_tangent_world([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])


@pytest.mark.parametrize(
    ("pointgoal", "heading", "requires"),
    [
        ([2.5, 0.0], 0.0, False),
        ([0.0, 2.5], 90.0, True),
        ([0.0, -2.5], -90.0, True),
    ],
)
def test_heading_contract(pointgoal, heading, requires) -> None:
    assert signed_pointgoal_heading_deg(pointgoal) == pytest.approx(heading)
    assert tangent_requires_alignment(pointgoal) is requires


def test_realized_motion_uses_post_snap_pose() -> None:
    assert realized_planar_translation_m(
        [1.0, 3.2, 2.0], [1.0, 3.2, 2.0]) == 0.0
    assert realized_planar_translation_m(
        [1.0, 3.2, 2.0], [1.3, 7.2, 2.4]) == pytest.approx(0.5)


def test_floor_aware_distance_separates_planar_and_vertical_error() -> None:
    distance, planar, vertical = floor_aware_goal_distance_m(
        [0.0, 3.2, 0.0], [0.6, 0.0, 0.8])
    assert planar == pytest.approx(1.0)
    assert vertical == pytest.approx(3.2)
    assert distance == pytest.approx(np.hypot(1.0, 3.2))
