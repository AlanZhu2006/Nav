import numpy as np
import pytest

from MemNavData.episodic_route_filter import (
    ActionCoordinateRouteCompass,
    ActionIntegratedRouteBearing,
    MonotoneRouteFilter,
    ROUTE_FILTER_SCHEMA_VERSION,
    derive_route_transition,
)


def test_transition_is_derived_from_route_cadence():
    positions = np.stack([
        0.02 * np.arange(200, dtype=np.float64),
        np.zeros(200, dtype=np.float64),
    ], axis=1)
    receipt = derive_route_transition(
        positions, nominal_motion_m=0.30, maximum_search_frames=32)
    assert receipt.schema_version == ROUTE_FILTER_SCHEMA_VERSION
    assert receipt.mean_advance_frames == 15
    assert receipt.sigma_advance_frames == 1.0
    assert receipt.maximum_advance_frames == 19


def test_filter_uses_soft_evidence_without_similarity_gate():
    positions = np.stack([
        0.05 * np.arange(80, dtype=np.float64),
        np.zeros(80, dtype=np.float64),
    ], axis=1)
    transition = derive_route_transition(
        positions, nominal_motion_m=0.30, maximum_search_frames=20)
    tracker = MonotoneRouteFilter(
        len(positions), transition,
        source_indices=np.arange(100, 20, -1),
    )
    similarity = np.zeros(len(positions), dtype=np.float64)
    similarity[6] = 1.0
    first = tracker.update(similarity, translated=True)
    assert first.state_index == 6
    assert first.source_index == 94

    # A turn is a zero-motion observation.  It cannot move the route state
    # backward even when appearance alone prefers an earlier address.
    similarity[:] = 0.0
    similarity[2] = 1.0
    second = tracker.update(similarity, translated=False)
    assert second.state_index >= first.state_index


def test_constant_observation_still_has_a_total_readout():
    positions = np.stack([
        0.05 * np.arange(80, dtype=np.float64),
        np.zeros(80, dtype=np.float64),
    ], axis=1)
    transition = derive_route_transition(
        positions, nominal_motion_m=0.30, maximum_search_frames=20)
    tracker = MonotoneRouteFilter(len(positions), transition)
    readout = tracker.update(np.ones(len(positions)), translated=True)
    assert readout.state_index == transition.mean_advance_frames
    assert readout.observation_standard_deviation == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_transition_rejects_invalid_motion(bad):
    positions = np.stack([
        np.arange(20, dtype=np.float64),
        np.zeros(20, dtype=np.float64),
    ], axis=1)
    with pytest.raises(ValueError):
        derive_route_transition(positions, nominal_motion_m=bad)


def test_action_integrated_bearing_has_no_endpoint_fallback():
    positions = np.stack([
        np.zeros(30, dtype=np.float64),
        0.1 * np.arange(30, dtype=np.float64),
    ], axis=1)
    pose9 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    field = ActionIntegratedRouteBearing(
        positions, pose9, lookahead_m=2.5)
    first = field.guidance(0)
    assert np.allclose(first.unit_bearing, [1.0, 0.0], atol=1e-12)
    assert np.isclose(np.linalg.norm(first.controller_pointgoal), 2.5)

    field.advance_executor_yaw(np.pi / 2.0)
    turned = field.guidance(0)
    assert np.allclose(turned.unit_bearing, [0.0, -1.0], atol=1e-12)

    endpoint = field.guidance(len(positions) - 1)
    assert endpoint.terminal_tangent_held is True
    assert np.isclose(np.linalg.norm(endpoint.controller_pointgoal), 2.5)


def _identity_pose9():
    return np.asarray([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ])


def test_action_coordinate_ignores_turn_frames_and_monocular_scale():
    route = np.asarray([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [0.0, -2.0],
        [0.0, -3.0],
    ])
    action_arc = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0])
    first = ActionCoordinateRouteCompass(
        route, action_arc, _identity_pose9(), lookahead_m=1.0)
    scaled = ActionCoordinateRouteCompass(
        17.0 * route, action_arc, _identity_pose9(), lookahead_m=1.0)

    turn = first.advance(
        executed_translation_m=0.0, executed_yaw_rad=0.25)
    assert turn.action_progress_m == 0.0
    left = first.advance(
        executed_translation_m=1.0, executed_yaw_rad=-0.25)
    right = scaled.advance(
        executed_translation_m=1.0, executed_yaw_rad=0.0)

    assert left.action_progress_m == 1.0
    assert left.route_state_index == 2
    np.testing.assert_allclose(left.unit_bearing, right.unit_bearing, atol=1e-9)
    np.testing.assert_allclose(left.unit_bearing, [-1.0, 0.0], atol=1e-9)


def test_action_coordinate_terminal_readout_is_not_a_fallback():
    route = np.asarray([[0.0, 0.0], [0.0, -1.0], [0.0, -2.0]])
    compass = ActionCoordinateRouteCompass(
        route, np.asarray([0.0, 1.0, 2.0]), _identity_pose9(),
        lookahead_m=1.0, controller_radius_m=2.5)

    result = compass.advance(
        executed_translation_m=3.0, executed_yaw_rad=0.0)

    assert result.action_progress_m == 2.0
    assert result.terminal_tangent_held is True
    np.testing.assert_allclose(result.controller_pointgoal, [-2.5, 0.0])
