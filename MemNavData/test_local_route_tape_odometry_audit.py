import math
import unittest

import numpy as np

from MemNavData.run_local_route_tape_odometry_audit import (
    cumulative_distance,
    model_local_translation,
    sample_polyline,
    tangent_heading,
)


def pose9(x: float, z: float, yaw: float = 0.0) -> np.ndarray:
    result = np.zeros(9, dtype=np.float64)
    result[0], result[2] = x, z
    result[4] = math.sin(yaw / 2.0)
    result[6] = math.cos(yaw / 2.0)
    return result


class RouteTapeAuditMathTest(unittest.TestCase):
    def test_forward_model_motion_uses_pose_frame(self):
        first = pose9(0.0, 0.0)
        second = pose9(0.0, 2.0)
        np.testing.assert_allclose(
            model_local_translation(first, second, 0.5), [1.0, 0.0])

    def test_polyline_lookahead_follows_corner(self):
        points = np.asarray([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        arc = cumulative_distance(points)
        np.testing.assert_allclose(sample_polyline(points, arc, 3.0), [1.0, 2.0])
        self.assertAlmostEqual(tangent_heading(points, arc, 2.0), math.pi / 2)


if __name__ == "__main__":
    unittest.main()
