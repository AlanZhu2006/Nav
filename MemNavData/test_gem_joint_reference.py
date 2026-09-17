"""Small synthetic tests for the offline multi-reference experiment."""
import unittest

import cv2
import numpy as np

from MemNavData.run_gem_joint_reference import unique_query_correspondences
from MemNavData.lingbot_pnp_localization import SiftPnPConfig, solve_camera_pose_pnp


class JointReferenceTests(unittest.TestCase):
    def test_duplicate_query_is_counted_once(self):
        query = np.array([[1., 2.], [1., 2.], [4., 6.], [1., 2.]])
        indices = unique_query_correspondences(query, np.array([.3, .8, .4, .8]),
                                              np.array([2, 3, 4, 1]))
        self.assertEqual(indices.tolist(), [3, 2])

    def test_empty_correspondences(self):
        self.assertEqual(len(unique_query_correspondences(
            np.empty((0, 2)), np.empty(0), np.empty(0))), 0)

    def test_joint_adds_constraints_without_changing_frame(self):
        rng = np.random.default_rng(11)
        xyz = rng.uniform([-2., -1.5, 4.], [2., 1.5, 8.], size=(24, 3))
        k = np.array([[300., 0, 259.], [0, 300., 259.], [0, 0, 1.]])
        rvec, tvec = np.array([.05, -.15, .03]), np.array([.4, -.2, .1])
        uv = cv2.projectPoints(xyz, rvec, tvec, k, None)[0].reshape(-1, 2)
        cfg = SiftPnPConfig()
        fov = np.array([0, 0, 0, 0, 0, 0, 1, 1.4, 1.4])
        for start in range(0, 24, 6):
            result = solve_camera_pose_pnp(xyz[start:start+6], uv[start:start+6],
                                          k, config=cfg, fov_pose9=fov)
            self.assertNotIn('pose9', result)
        joint = solve_camera_pose_pnp(xyz, uv, k, config=cfg, fov_pose9=fov)
        expected = -cv2.Rodrigues(rvec)[0].T @ tvec
        self.assertEqual(joint['status'], 'ok')
        np.testing.assert_allclose(joint['pose9'][:3], expected, atol=1e-5)

    def test_shared_coordinate_similarity_changes_pose_not_projection(self):
        rng = np.random.default_rng(23)
        xyz = rng.uniform([-2., -1., 4.], [2., 1., 8.], size=(40, 3))
        k = np.array([[300., 0, 259.], [0, 300., 259.], [0, 0, 1.]])
        uv = (xyz @ k.T)
        uv = uv[:, :2] / uv[:, 2:3]
        rotation = cv2.Rodrigues(np.array([.15, -.3, .04]))[0]
        shift = np.array([1., -2., .5])
        transformed = 2.3 * (xyz @ rotation.T) + shift
        fov = np.array([0, 0, 0, 0, 0, 0, 1, 1.4, 1.4])
        result = solve_camera_pose_pnp(transformed, uv, k, config=SiftPnPConfig(),
                                       fov_pose9=fov)
        self.assertEqual(result['status'], 'ok')
        np.testing.assert_allclose(result['pose9'][:3], shift, atol=1e-5)
        self.assertLess(result['reprojection_rmse_px'], 1e-5)


if __name__ == '__main__':
    unittest.main()
