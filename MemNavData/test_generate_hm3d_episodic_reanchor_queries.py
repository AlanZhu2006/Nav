import unittest

import numpy as np

from MemNavData.generate_hm3d_episodic_reanchor_queries import (
    optical_center_from_floor,
    query_yaw_from_history,
)


class ReanchorQueryConstructionTest(unittest.TestCase):
    def test_frozen_camera_height_is_added_to_floor_position(self):
        floor = np.asarray([1.2, -0.17, 4.3], dtype=np.float64)
        camera = optical_center_from_floor(floor, 0.5)
        np.testing.assert_allclose(camera, [1.2, 0.33, 4.3])
        np.testing.assert_allclose(camera[[0, 2]], floor[[0, 2]])

    def test_nonpositive_camera_height_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "camera height"):
            optical_center_from_floor(np.zeros(3), 0.0)

    def test_reverse_traversal_yaw_is_relative_to_taught_heading(self):
        yaw = query_yaw_from_history(
            0.25, base_yaw_offset_deg=180.0, perturbation_deg=-15.0)
        self.assertAlmostEqual(yaw, 0.25 + np.deg2rad(165.0))


if __name__ == "__main__":
    unittest.main()
