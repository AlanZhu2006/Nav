import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from MemNavData.gem_temporal_localization import TemporalView,balanced_samples,shared_transform


def fixture(pure_rotation=False,unit=1.):
    rng = np.random.default_rng(40)
    points = rng.uniform([-1.,-.7,3.],[1.,.7,6.],(120,3))*unit
    K = np.array([[320.,0,259.],[0,315.,259.],[0,0,1.]])
    expected = np.eye(4)
    expected[:3,:3] = 1.3*Rotation.from_euler('xyz',[2.,-8.,1.],degrees=True).as_matrix()
    expected[:3,3] = np.array([.3,-.1,.2])*unit
    views = []
    for i,c in enumerate(([-.25,.02,-.05],[-.1,-.02,.01],[0,0,0])):
        L = np.eye(4)
        L[:3,:3] = Rotation.from_euler('y',3.*(2-i),degrees=True).as_matrix()
        L[:3,3] = (np.zeros(3) if pure_rotation else np.array(c))*unit
        world_to_camera = np.linalg.inv(expected@L)
        y = points@world_to_camera[:3,:3].T+world_to_camera[:3,3]
        p = y@K.T
        views.append(TemporalView(i,points.copy(),p[:,:2]/p[:,2:3],K.copy(),L))
    initial = np.eye(4)
    initial[:3,:3] = Rotation.from_euler('xyz',[3.,-7.,0.],degrees=True).as_matrix()
    initial[:3,3] = np.array([.25,-.07,.24])*unit
    return views,initial,expected


class TemporalLocalizationTest(unittest.TestCase):
    def test_recovers_one_shared_similarity_and_gauge(self):
        outputs=[]
        for unit in (1.,11.):
            views,initial,expected=fixture(unit=unit)
            result=shared_transform(views,current=2,reference_from_current=initial,estimate_scale=True)
            self.assertTrue(result['identified'])
            self.assertTrue(result['optimizer_success'])
            np.testing.assert_allclose(result['transform'],expected,atol=1e-6)
            self.assertAlmostEqual(result['scale'],1.3,places=6)
            outputs.append(result)
        self.assertAlmostEqual(outputs[0]['scale'],outputs[1]['scale'],places=6)

    def test_pure_rotation_does_not_identify_scale(self):
        views,initial,_=fixture(pure_rotation=True)
        result=shared_transform(views,current=2,reference_from_current=initial,estimate_scale=True)
        self.assertFalse(result['identified'])
        self.assertEqual(result['numerical_rank'],6)
        self.assertLess(result['scale_information'],1e-12)

    def test_balanced_total_budget_and_causality(self):
        selections=balanced_samples([2,100,100],10)
        self.assertEqual([len(x) for x in selections],[2,4,4])
        for x in selections:
            self.assertEqual(len(x),len(np.unique(x)))
        views,initial,_=fixture()
        with self.assertRaises(ValueError):
            shared_transform(views,current=1,reference_from_current=initial,estimate_scale=False)
        with self.assertRaises(ValueError):
            shared_transform([views[0],views[0]],current=2,reference_from_current=initial,estimate_scale=False)


if __name__=='__main__':
    unittest.main()
