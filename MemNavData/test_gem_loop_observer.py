import unittest

import numpy as np

from MemNavData.gem_loop_observer import CausalKeyframePool, PairRaster, sample_current_grid


class LoopEvidenceContracts(unittest.TestCase):
    def test_causal_pool_boundary_and_episode_isolation(self):
        pool = CausalKeyframePool([8, 15, 22, 29], exclude_recent=64)
        self.assertEqual(pool.advance(71), [])
        self.assertEqual(pool.advance(72), [8])
        self.assertEqual(pool.resolve([0]), [8])
        with self.assertRaises(ValueError):
            pool.resolve([1])
        self.assertEqual(pool.advance(86), [15, 22])
        self.assertEqual(pool.resolve([2, 0]), [22, 8])
        with self.assertRaises(ValueError):
            pool.advance(86)
        with self.assertRaises(ValueError):
            pool.resolve([0, 0])
        self.assertEqual(CausalKeyframePool([8]).indexed, [])

    def test_crop_and_resize_round_trip(self):
        raster = PairRaster(288, 512, 480/512, 270/288, 0, 0, 480, 270)
        p = raster.pixels(np.array([0, 511, 512, 512*288-1]))
        np.testing.assert_array_equal(p, [[0, 0], [511, 0], [0, 1], [511, 287]])
        np.testing.assert_allclose(raster.raw(p), p*[480/512, 270/288])
        cropped = PairRaster(280, 512, 1.25, 1.2, 0, 6, 640, 350)
        np.testing.assert_allclose(cropped.raw([[20, 30]]), [[25, 43.2]])
        with self.assertRaises(ValueError):
            raster.pixels(np.array([512*288]))

    def test_uniform_budget_and_empty_evidence(self):
        self.assertEqual(len(sample_current_grid(np.zeros(10, bool))), 0)
        mask = np.arange(100) % 3 != 0
        ids = sample_current_grid(mask, limit=10)
        self.assertEqual(len(np.unique(ids)), 10)
        self.assertTrue(mask[ids].all())
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 98)


if __name__ == '__main__':
    unittest.main()
