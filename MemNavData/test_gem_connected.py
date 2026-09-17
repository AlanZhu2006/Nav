"""Camera identity and monocular scale alignment of connected memory."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from NavDP.baselines.memnav.gem.connected import align_same_cameras


class ConnectedMemoryTests(unittest.TestCase):
    def test_exact_shared_cameras_under_random_sim3(self):
        rng = np.random.default_rng(83)
        for _ in range(40):
            new = np.broadcast_to(np.eye(4), (8,4,4)).copy()
            new[:, :3, :3] = Rotation.random(8, random_state=rng).as_matrix()
            new[:, :3, 3] = rng.normal(size=(8,3))
            rotation = Rotation.random(random_state=rng).as_matrix()
            scale = np.exp(rng.normal())
            translation = rng.normal(size=3)
            old = new.copy()
            old[:, :3, :3] = rotation @ new[:, :3, :3]
            old[:, :3, 3] = scale * new[:, :3, 3] @ rotation.T + translation
            depth = rng.uniform(.1,10,size=(8,96))
            actual, _ = align_same_cameras(old,new,scale*depth,depth,np.ones(96,dtype=bool))
            np.testing.assert_allclose(actual[:3,:3],scale*rotation,atol=1e-12)
            np.testing.assert_allclose(actual[:3,3],translation,atol=1e-12)

    def test_zero_camera_baseline_still_has_depth_scale(self):
        new = np.broadcast_to(np.eye(4), (8,4,4)).copy()
        old = new.copy()
        old[:,:3,:3] = Rotation.from_euler('y',.4).as_matrix()
        old[:,:3,3] = [1,2,3]
        depth = np.arange(1,33,dtype=float).reshape(8,4)
        actual,_=align_same_cameras(old,new,2*depth,depth,np.ones(4,dtype=bool))
        self.assertAlmostEqual(np.cbrt(np.linalg.det(actual[:3,:3])),2)
        np.testing.assert_allclose(actual[:3,3],[1,2,3],atol=1e-12)

    def test_invalid_shared_frame_raises_instead_of_replacing_relation(self):
        poses = np.broadcast_to(np.eye(4),(8,4,4)).copy()
        depth = np.ones((8,4))
        depth[3] = np.nan
        with self.assertRaises(ValueError):
            align_same_cameras(poses,poses,depth,np.ones((8,4)),np.ones(4,dtype=bool))


if __name__=='__main__': unittest.main()
