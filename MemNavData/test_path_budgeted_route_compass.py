import math

import numpy as np
import pytest

from MemNavData.path_budgeted_route_compass import (
    first_feasible_tangent_from_local_route,
    from_local_route,
    tangent_from_local_route,
)


def test_progress_never_exceeds_cumulative_query_path():
    compass = from_local_route(np.asarray([
        [0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0],
    ]))
    for forward, left, yaw in [
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]:
        readout = compass.advance_local_se2(
            executed_forward_m=forward,
            executed_left_m=left,
            executed_yaw_rad=yaw,
        )
        assert (readout.projected_progress_m
                <= readout.query_path_length_m + 1e-12)
        assert (readout.progress_increment_m
                <= readout.progress_budget_m + 1e-12)


def test_unused_path_budget_can_be_recovered_later():
    route = np.asarray([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    compass = from_local_route(route, lookahead_m=0.5)
    # The first lateral motion earns one metre of causal budget but projects
    # nowhere along the route.
    first = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=1.0,
        executed_yaw_rad=0.0,
    )
    assert first.projected_progress_m == pytest.approx(0.0)
    # A short diagonal correction reaches a later route point. The projection
    # may use banked path budget without exceeding total travelled distance.
    second = compass.advance_local_se2(
        executed_forward_m=-1.0,
        executed_left_m=-1.0,
        executed_yaw_rad=0.0,
    )
    assert second.projected_progress_m == pytest.approx(1.0)
    assert second.query_path_length_m == pytest.approx(1.0 + math.sqrt(2.0))


def test_exact_self_intersection_tie_prefers_earliest_reachable_arc():
    route = np.asarray([
        [0.0, 0.0], [-1.0, 0.0], [-1.0, -1.0],
        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
    ])
    compass = from_local_route(route)
    readout = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    assert readout.projected_progress_m == pytest.approx(0.0)


def test_tangent_guidance_is_invariant_to_lateral_position_drift():
    route = np.asarray([
        [0.0, 0.0], [-2.0, 0.0], [-4.0, 0.0], [-6.0, 0.0],
    ])
    chord = from_local_route(route, lookahead_m=1.0)
    tangent = tangent_from_local_route(route, lookahead_m=1.0)

    chord_readout = chord.advance_local_se2(
        executed_forward_m=-1.0,
        executed_left_m=1.0,
        executed_yaw_rad=0.0,
    )
    tangent_readout = tangent.advance_local_se2(
        executed_forward_m=-1.0,
        executed_left_m=1.0,
        executed_yaw_rad=0.0,
    )

    assert tangent_readout.projected_progress_m == pytest.approx(1.0)
    assert tangent_readout.cross_track_error_m == pytest.approx(1.0)
    np.testing.assert_allclose(tangent_readout.unit_bearing, [-1.0, 0.0])
    assert not np.allclose(
        chord_readout.unit_bearing, tangent_readout.unit_bearing)


def test_tangent_guidance_is_expressed_in_the_current_body_frame():
    compass = tangent_from_local_route(np.asarray([
        [0.0, 0.0], [-2.0, 0.0], [-4.0, 0.0],
    ]), lookahead_m=1.0)

    readout = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=math.pi / 2.0,
    )

    np.testing.assert_allclose(readout.unit_bearing, [0.0, 1.0], atol=1e-12)


def test_tangent_guidance_holds_the_terminal_route_direction():
    compass = tangent_from_local_route(np.asarray([
        [0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0],
    ]), lookahead_m=0.5)

    readout = compass.advance_local_se2(
        executed_forward_m=-2.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )

    assert readout.projected_progress_m == pytest.approx(2.0)
    assert readout.terminal_tangent_held is True
    np.testing.assert_allclose(readout.unit_bearing, [-1.0, 0.0])


def test_first_feasible_tangent_does_not_cut_the_first_corner():
    # A 2.5 m arc-ahead chord points diagonally through the corner.  The new
    # readout must preserve the first locally traversable segment.
    route = np.asarray([
        [0.0, 0.0], [0.31, 0.0], [0.31, 2.50], [0.31, 3.00],
    ])
    legacy = tangent_from_local_route(route, lookahead_m=2.5)
    local = first_feasible_tangent_from_local_route(route)

    legacy_readout = legacy.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    local_readout = local.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )

    assert legacy_readout.unit_bearing[1] > 0.9
    np.testing.assert_allclose(local_readout.unit_bearing, [1.0, 0.0])
    assert local_readout.reference_arc_m == pytest.approx(0.31)
    assert local.tangent_baseline_m == pytest.approx(0.30)
    assert np.linalg.norm(local_readout.controller_pointgoal) == pytest.approx(
        2.5)


def test_first_feasible_tangent_advances_continuously_without_a_gate():
    route = np.asarray([
        [0.0, 0.0], [0.35, 0.0], [0.70, 0.0],
        [0.70, 0.35], [0.70, 0.70],
    ])
    compass = first_feasible_tangent_from_local_route(route)

    first = compass.advance_local_se2(
        executed_forward_m=0.35,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    second = compass.advance_local_se2(
        executed_forward_m=0.35,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )

    assert first.projected_progress_m == pytest.approx(0.35)
    np.testing.assert_allclose(first.unit_bearing, [1.0, 0.0])
    assert second.projected_progress_m == pytest.approx(0.70)
    np.testing.assert_allclose(second.unit_bearing, [0.0, 1.0])
    assert second.reference_arc_m == pytest.approx(1.05)


def test_first_feasible_tangent_holds_last_local_direction_at_terminal():
    compass = first_feasible_tangent_from_local_route(np.asarray([
        [0.0, 0.0], [0.0, 0.31], [0.0, 0.62],
    ]))
    readout = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.62,
        executed_yaw_rad=0.0,
    )

    assert readout.terminal_tangent_held is True
    np.testing.assert_allclose(readout.unit_bearing, [0.0, 1.0])
    assert readout.reference_arc_m == pytest.approx(0.62)
