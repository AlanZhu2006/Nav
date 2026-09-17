"""Noisy same-camera transport must be reciprocal without tuning thresholds."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from NavDP.baselines.memnav.gem.reciprocal import reciprocal_camera_transport


class ReciprocalTests(unittest.TestCase):
    def test_swap_inverts_transform_even_with_noisy_camera_and_depth(self):
        rng=np.random.default_rng(71)
        for _ in range(80):
            a=np.broadcast_to(np.eye(4),(8,4,4)).copy()
            b=a.copy()
            a[:,:3,:3]=Rotation.random(8,random_state=rng).as_matrix()
            b[:,:3,:3]=Rotation.random(8,random_state=rng).as_matrix()
            a[:,:3,3]=rng.normal(size=(8,3))
            b[:,:3,3]=rng.normal(size=(8,3))
            x=np.exp(rng.normal(size=(8,64)))
            y=np.exp(rng.normal(size=(8,64)))
            mask=np.ones(64,dtype=bool)
            x[0,0]=np.nan
            ab,_=reciprocal_camera_transport(a,b,x,y,mask)
            ba,_=reciprocal_camera_transport(b,a,y,x,mask)
            np.testing.assert_allclose(ab@ba,np.eye(4),atol=1e-11)

    def test_exact_scale_and_camera_change(self):
        rng=np.random.default_rng(17)
        new=np.broadcast_to(np.eye(4),(8,4,4)).copy()
        new[:,:3,:3]=Rotation.random(8,random_state=rng).as_matrix()
        new[:,:3,3]=rng.normal(size=(8,3))
        old=new.copy()
        r=Rotation.from_euler('xyz',[.2,-.4,.3]).as_matrix()
        old[:,:3,:3]=r@new[:,:3,:3]
        old[:,:3,3]=3.7*new[:,:3,3]@r.T+[1,2,3]
        depth=rng.uniform(.1,8,(8,64))
        actual,_=reciprocal_camera_transport(old,new,3.7*depth,depth,np.ones(64,dtype=bool))
        np.testing.assert_allclose(actual[:3,:3],3.7*r,atol=1e-12)
        np.testing.assert_allclose(actual[:3,3],[1,2,3],atol=1e-12)


if __name__=='__main__':unittest.main()
