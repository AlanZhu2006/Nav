import math

import numpy as np
import pytest

from MemNavData.episodic_route_filter import ActionCoordinateRouteCompass
from MemNavData.se2_projected_route_compass import (
    SE2_EXECUTOR_MOTION_MODEL,
    SE2_LOCAL_ODOMETRY_MODEL,
    SE2ProjectedRouteCompass,
    reconstruct_reverse_route,
    reconstruct_reverse_route_se2,
)


def test_reverse_receipts_reconstruct_a_metric_corner():
    route, headings = reconstruct_reverse_route(
        translations_m=[2.0, 2.0],
        yaw_deltas_rad=[math.pi / 2.0, 0.0],
    )

    np.testing.assert_allclose(
        route,
        [[0.0, 0.0], [-2.0, 0.0], [-2.0, 2.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        headings, [0.0, -math.pi / 2.0, -math.pi / 2.0],
        atol=1e-12,
    )


def test_projected_progress_follows_a_corner_and_outputs_body_bearing():
    compass = SE2ProjectedRouteCompass.from_reverse_executor_receipts(
        [2.0, 2.0], [math.pi / 2.0, 0.0], lookahead_m=1.0)

    initial = compass.advance(
        executed_translation_m=0.0, executed_yaw_rad=0.0)
    np.testing.assert_allclose(initial.unit_bearing, [-1.0, 0.0])

    corner = compass.advance(
        executed_translation_m=2.0, executed_yaw_rad=math.pi)
    assert corner.projected_progress_m == pytest.approx(2.0)
    assert corner.cross_track_error_m == pytest.approx(0.0, abs=1e-12)
    # At the corner the next reverse-route edge lies to the robot's right.
    np.testing.assert_allclose(corner.unit_bearing, [0.0, -1.0], atol=1e-12)

    endpoint = compass.advance(
        executed_translation_m=2.0, executed_yaw_rad=-math.pi / 2.0)
    assert endpoint.projected_progress_m == pytest.approx(4.0)
    assert endpoint.terminal_tangent_held is True
    assert endpoint.motion_model == SE2_EXECUTOR_MOTION_MODEL
    np.testing.assert_allclose(endpoint.unit_bearing, [1.0, 0.0], atol=1e-12)


def test_lateral_oscillation_does_not_masquerade_as_route_progress():
    route = np.asarray([[0.0, 0.0], [-5.0, 0.0], [-10.0, 0.0]])
    projected = SE2ProjectedRouteCompass(route, lookahead_m=2.5)
    scalar = ActionCoordinateRouteCompass(
        route,
        np.asarray([0.0, 5.0, 10.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        lookahead_m=2.5,
    )

    readout = None
    for yaw in [math.pi / 2.0] + [math.pi] * 9:
        readout = projected.advance(
            executed_translation_m=1.0, executed_yaw_rad=yaw)
        scalar.advance(executed_translation_m=1.0, executed_yaw_rad=yaw)

    assert readout is not None
    assert projected.projected_progress_m == pytest.approx(0.0, abs=1e-9)
    assert scalar.action_progress_m == pytest.approx(10.0)
    assert readout.query_path_length_m == pytest.approx(10.0)
    assert readout.route_progress_fraction == pytest.approx(0.0, abs=1e-9)


def test_off_route_pose_is_guided_from_live_position_not_route_tangent():
    route = np.asarray([[0.0, 0.0], [-5.0, 0.0], [-10.0, 0.0]])
    compass = SE2ProjectedRouteCompass(route, lookahead_m=2.5)

    readout = compass.advance(
        executed_translation_m=1.0, executed_yaw_rad=math.pi / 2.0)

    assert readout.projected_progress_m == pytest.approx(0.0)
    assert readout.cross_track_error_m == pytest.approx(1.0)
    # World delta to the lookahead is [-2.5, -1].  Expressed in a body that
    # faces +left, it is [-1, +2.5].
    expected = np.asarray([-1.0, 2.5])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(readout.unit_bearing, expected, atol=1e-12)


def test_projection_is_monotone_when_executor_moves_backwards_along_route():
    route = np.asarray([[0.0, 0.0], [-5.0, 0.0], [-10.0, 0.0]])
    compass = SE2ProjectedRouteCompass(route, lookahead_m=2.0)

    outward = compass.advance(
        executed_translation_m=4.0, executed_yaw_rad=math.pi)
    assert outward.projected_progress_m == pytest.approx(4.0)

    away = compass.advance(
        executed_translation_m=2.0, executed_yaw_rad=math.pi)
    assert away.projected_progress_m == pytest.approx(4.0)
    assert away.cross_track_error_m == pytest.approx(2.0)
    np.testing.assert_allclose(away.unit_bearing, [-1.0, 0.0], atol=1e-12)


def test_local_se2_receipt_preserves_realized_lateral_motion():
    # Original history moves one metre forward and then one metre left in its
    # previous body frame while turning 90 degrees.  The exact inverse route
    # must retain both components rather than replacing them with a chord norm.
    route, _ = reconstruct_reverse_route_se2(
        delta_forwards_m=[1.0, 1.0],
        delta_lefts_m=[1.0, 0.0],
        yaw_deltas_rad=[math.pi / 2.0, 0.0],
    )
    np.testing.assert_allclose(
        route,
        [[0.0, 0.0], [-1.0, 1.0], [-1.0, 2.0]],
        atol=1e-12,
    )

    compass = SE2ProjectedRouteCompass.from_reverse_local_se2_receipts(
        [1.0, 1.0], [1.0, 0.0], [math.pi / 2.0, 0.0], lookahead_m=1.0)
    readout = compass.advance_local_se2(
        executed_forward_m=-1.0,
        executed_left_m=1.0,
        executed_yaw_rad=math.pi,
    )
    assert readout.motion_model == SE2_LOCAL_ODOMETRY_MODEL
    assert readout.projected_progress_m == pytest.approx(math.sqrt(2.0))
    assert readout.cross_track_error_m == pytest.approx(0.0, abs=1e-12)


def test_route_and_query_must_use_the_same_motion_receipt_model():
    compass = SE2ProjectedRouteCompass.from_reverse_local_se2_receipts(
        [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="differs from frozen route"):
        compass.advance(executed_translation_m=1.0, executed_yaw_rad=0.0)


def test_projection_cannot_jump_across_a_self_intersection():
    # The origin appears again four metres later on the route.  A global
    # nearest-point projection would be ambiguous here and could silently
    # teleport progress to the later branch.  The causal projection is
    # bounded by the current local displacement instead.
    route = np.asarray([
        [0.0, 0.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])
    compass = SE2ProjectedRouteCompass.from_reverse_local_se2_receipts(
        delta_forwards_m=[1.0, 1.0, 1.0, 1.0, 1.0],
        delta_lefts_m=[0.0, 0.0, 0.0, 0.0, 0.0],
        yaw_deltas_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # Replace the straight constructor route with the self-intersecting route;
    # the invariant belongs to projection and is independent of how an edge
    # receipt was estimated.
    compass = SE2ProjectedRouteCompass(
        route, motion_model=SE2_LOCAL_ODOMETRY_MODEL)

    initial = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    assert initial.projected_progress_m == pytest.approx(0.0)

    moved = compass.advance_local_se2(
        executed_forward_m=-0.1,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    assert moved.projected_progress_m == pytest.approx(0.1)
    assert moved.progress_increment_m <= moved.progress_budget_m + 1e-12


@pytest.mark.parametrize(
    "translations,yaws",
    [([1.0], [0.0]), ([1.0, -1.0], [0.0, 0.0]), ([1.0, 1.0], [0.0])],
)
def test_reverse_route_rejects_malformed_receipts(translations, yaws):
    with pytest.raises(ValueError):
        reconstruct_reverse_route(translations, yaws)
