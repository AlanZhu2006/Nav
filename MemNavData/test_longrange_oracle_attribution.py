import math
import unittest

import numpy as np

from longrange_oracle_attribution import (
    OracleRuntimeState,
    OracleRouteProjector,
    fixed_radius_pointgoal,
    geodesic_lookahead_world,
    historical_reverse_route,
    world_delta_to_local,
)


class OracleAttributionGeometryTest(unittest.TestCase):
    def test_habitat_world_to_local_axes(self):
        np.testing.assert_allclose(
            world_delta_to_local([0.0, -2.0], 0.0), [2.0, 0.0])
        np.testing.assert_allclose(
            world_delta_to_local([-2.0, 0.0], 0.0), [0.0, 2.0])
        np.testing.assert_allclose(
            world_delta_to_local([-2.0, 0.0], math.pi / 2), [2.0, 0.0],
            atol=1e-12)

    def test_fixed_radius_discards_distance(self):
        np.testing.assert_allclose(
            fixed_radius_pointgoal([0.0, -10.0], [0.0, 0.0], 0.0),
            [2.5, 0.0])

    def test_monotone_projection_corrects_cross_track_error(self):
        route = [[0.0, 0.0], [0.0, -5.0], [5.0, -5.0]]
        projector = OracleRouteProjector(route, lookahead_m=2.5)
        first = projector.update([1.0, -2.0], 0.0)
        self.assertAlmostEqual(first.progress_m, 2.0)
        self.assertAlmostEqual(first.cross_track_m, 1.0)
        np.testing.assert_allclose(first.reference_world_xz, [0.0, -4.5])
        second = projector.update([0.1, -1.0], 0.0)
        self.assertGreaterEqual(second.progress_m, first.progress_m)

    def test_historical_route_is_reversed_to_anchor(self):
        poses = [
            {"x": 0.0, "z": float(index)} for index in range(6)
        ]
        route = historical_reverse_route(
            poses,
            query_start_xz=[0.0, 5.0],
            goal_start_frame=6,
            target_anchor=2,
        )
        np.testing.assert_allclose(route, [
            [0.0, 5.0], [0.0, 4.0], [0.0, 3.0], [0.0, 2.0],
        ])

    def test_geodesic_lookahead_interpolates(self):
        point, extent = geodesic_lookahead_world([
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [3.0, 0.0, -1.0],
        ], lookahead_m=2.5)
        np.testing.assert_allclose(point, [1.5, -1.0])
        self.assertAlmostEqual(extent, 4.0)

    def test_runtime_query_goal_survives_causal_prefix_binding(self):
        state = OracleRuntimeState()
        state.bind_query([1.0, 0.5, -2.0])
        state.bind_history({"poses": [{"x": 0.0, "z": 0.0}]})
        np.testing.assert_allclose(state.goal_floor, [1.0, 0.5, -2.0])
        self.assertIsNotNone(state.trace)

    def test_new_runtime_query_invalidates_stale_history_state(self):
        state = OracleRuntimeState()
        state.bind_query([1.0, 0.5, -2.0])
        state.bind_history({"poses": [{"x": 0.0, "z": 0.0}]})
        state.bind_pathfinder(object())
        state.route_projector = OracleRouteProjector(
            [[0.0, 0.0], [0.0, -1.0]])
        state.route_anchor = 0
        state.route_goal_start = 1
        state.bind_query([3.0, 0.5, 4.0])
        self.assertIsNone(state.trace)
        self.assertIsNone(state.pathfinder)
        self.assertIsNone(state.route_projector)
        self.assertIsNone(state.route_anchor)
        self.assertIsNone(state.route_goal_start)
        np.testing.assert_allclose(state.goal_floor, [3.0, 0.5, 4.0])


if __name__ == "__main__":
    unittest.main()
