import math

import numpy as np

from MemNavData.run_local_monocular_adjacent_motion_audit import (
    QUERY_SCHEMA_VERSIONS,
    angular_error,
    world_delta_in_body,
)


def test_world_delta_matches_habitat_forward_left_axes():
    np.testing.assert_allclose(
        world_delta_in_body(np.asarray([0.0, -1.0]), 0.0),
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        world_delta_in_body(np.asarray([-1.0, 0.0]), 0.0),
        [0.0, 1.0],
    )


def test_angular_error_wraps_at_pi():
    np.testing.assert_allclose(
        angular_error(math.pi - 0.1, -math.pi + 0.1), 0.2,
    )


def test_audit_accepts_only_frozen_reverse_query_schemas():
    assert QUERY_SCHEMA_VERSIONS == {
        "hm3d_reverse_route_odometry_queries_v1_20260902",
        "hm3d_dense_reverse_route_queries_v2_20260903",
    }
