import unittest

import numpy as np

from MemNavData.diag_patch_temporal_router import (
    select_hard_candidates,
    selection_digest,
)
from MemNavData.patch_temporal_router import (
    combine_patch_temporal,
    combined_feature_names,
    directional_combined_feature_names,
    directional_patch_feature_names,
    directional_patch_relation_features,
    patch_feature_names,
    symmetric_from_directional_patch_features,
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

    def test_directional_patch_relation_preserves_query_side(self):
        rng = np.random.default_rng(17)
        query = rng.normal(size=(16, 12))
        memory = rng.normal(size=(16, 12))
        forward = directional_patch_relation_features(query, memory, 0.73)
        reverse = directional_patch_relation_features(memory, query, 0.73)
        names = directional_patch_feature_names()
        self.assertEqual(forward.shape, (len(names),))
        np.testing.assert_allclose(
            forward[names.index("best_match_mean_query")],
            reverse[names.index("best_match_mean_memory")], atol=1e-12)
        np.testing.assert_allclose(
            forward[names.index("best_match_q90_memory")],
            reverse[names.index("best_match_q90_query")], atol=1e-12)
        self.assertEqual(
            len(directional_combined_feature_names()),
            len(combined_feature_names()))

    def test_directional_features_recover_exact_symmetric_summary(self):
        rng = np.random.default_rng(23)
        query = rng.normal(size=(64, 24))
        memory = rng.normal(size=(64, 24))
        directional = directional_patch_relation_features(
            query, memory, 0.61)
        recovered = symmetric_from_directional_patch_features(directional)
        direct = symmetric_patch_relation_features(query, memory, 0.61)
        np.testing.assert_allclose(recovered, direct, atol=1e-12)
        batch = symmetric_from_directional_patch_features(
            np.stack([directional, directional]))
        self.assertEqual(batch.shape, (2, len(patch_feature_names())))

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

    def test_feature_cache_identity_tracks_temporal_curve_not_labels(self):
        import pandas as pd

        full = pd.DataFrame([
            dict(session_id="s", query_path="q.jpg", candidate_path="a.jpg",
                 candidate_frame=1, dino_cosine=0.9, teacher_pass=1),
            dict(session_id="s", query_path="q.jpg", candidate_path="b.jpg",
                 candidate_frame=2, dino_cosine=0.8, teacher_pass=0),
        ])
        selected = full.iloc[:1].copy()
        baseline = selection_digest(
            selected, full, 1, 8, "weight", "directional")
        relabeled = full.copy()
        relabeled["teacher_pass"] = [0, 1]
        self.assertEqual(
            baseline,
            selection_digest(
                selected.assign(teacher_pass=0), relabeled,
                1, 8, "weight", "directional"))
        changed_curve = full.copy()
        changed_curve.loc[1, "dino_cosine"] = 0.7
        self.assertNotEqual(
            baseline,
            selection_digest(
                selected, changed_curve, 1, 8, "weight", "directional"))

    def test_temporal_nms_candidate_selection_removes_adjacent_duplicates(self):
        import pandas as pd

        frame = pd.DataFrame([
            dict(session_id="s", query_path="q.jpg", candidate_path="a.jpg",
                 candidate_frame=0, dino_cosine=0.99, teacher_pass=0),
            dict(session_id="s", query_path="q.jpg", candidate_path="b.jpg",
                 candidate_frame=1, dino_cosine=0.98, teacher_pass=0),
            dict(session_id="s", query_path="q.jpg", candidate_path="c.jpg",
                 candidate_frame=8, dino_cosine=0.90, teacher_pass=1),
        ])
        raw = select_hard_candidates(frame, 2)
        diverse = select_hard_candidates(frame, 2, "temporal_nms", 4)
        self.assertEqual(raw["candidate_frame"].tolist(), [0, 1])
        self.assertEqual(diverse["candidate_frame"].tolist(), [0, 8])


if __name__ == "__main__":
    unittest.main()
