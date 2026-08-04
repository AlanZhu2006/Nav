import unittest

import numpy as np

from MemNavData.patch_temporal_router import (
    combine_patch_temporal,
    combined_feature_names,
    patch_feature_names,
    symmetric_patch_relation_features,
    temporal_feature_names,
    temporal_score_features,
)


class PatchTemporalRouterTest(unittest.TestCase):
    def test_patch_relation_is_symmetric_and_finite(self):
        rng = np.random.default_rng(7)
        query = rng.normal(size=(16, 12))
        memory = rng.normal(size=(16, 12))
        forward = symmetric_patch_relation_features(query, memory, 0.73)
        reverse = symmetric_patch_relation_features(memory, query, 0.73)
        np.testing.assert_allclose(forward, reverse, atol=1e-12)
        self.assertEqual(forward.shape, (len(patch_feature_names()),))
        self.assertTrue(np.isfinite(forward).all())

    def test_identical_patch_grid_has_strong_mutual_support(self):
        tokens = np.eye(16, dtype=np.float64)
        features = symmetric_patch_relation_features(tokens, tokens, 1.0)
        names = patch_feature_names()
        mutual = features[names.index("mutual_match_fraction")]
        residual = features[names.index(
            "affine_residual_median_direction_mean")]
        self.assertEqual(mutual, 1.0)
        self.assertLess(residual, 1e-10)

    def test_patch_relation_rejects_non_square_or_zero_tokens(self):
        with self.assertRaises(ValueError):
            symmetric_patch_relation_features(
                np.ones((15, 4)), np.ones((15, 4)), 0.5)
        with self.assertRaises(ValueError):
            symmetric_patch_relation_features(
                np.zeros((16, 4)), np.ones((16, 4)), 0.5)

    def test_temporal_features_capture_supported_peak(self):
        frames = np.arange(80)
        scores = np.full(80, 0.20)
        scores[37:44] = np.array([0.75, 0.83, 0.91, 0.95, 0.90, 0.84, 0.76])
        features = temporal_score_features(40, frames, scores)
        names = temporal_feature_names()
        self.assertEqual(features.shape, (len(names),))
        self.assertEqual(features[names.index("temporal_candidate_cosine")], 0.95)
        self.assertEqual(features[names.index(
            "temporal_score_minus_session_max")], 0.0)
        self.assertGreater(features[names.index(
            "temporal_w4_near_peak_fraction")], 0.1)

    def test_temporal_input_validation_and_combination(self):
        with self.assertRaises(ValueError):
            temporal_score_features(1, [0, 0, 1], [0.1, 0.2, 0.3])
        patch = np.zeros((2, len(patch_feature_names())))
        temporal = np.zeros((2, len(temporal_feature_names())))
        combined = combine_patch_temporal(patch, temporal)
        self.assertEqual(combined.shape, (2, len(combined_feature_names())))


if __name__ == "__main__":
    unittest.main()
