import unittest

import numpy as np

from MemNavData.diag_probabilistic_memory_localizer import (
    grouped_ranking_metrics,
)
from MemNavData.probabilistic_memory_localizer import (
    evaluate_probabilistic_set,
    fit_probabilistic_set,
    scene_group_oof_probabilities,
    session_indices,
)


class ProbabilisticMemoryLocalizerTest(unittest.TestCase):
    def test_session_indices_preserve_first_seen_order(self):
        groups = np.asarray(["b", "a", "b", "c", "a"])
        result = session_indices(groups)
        self.assertEqual([row.tolist() for row in result], [[0, 2], [1, 4], [3]])

    def test_fit_learns_match_and_no_match(self):
        features = np.asarray([
            [3.0], [0.0],      # positive session
            [-2.0], [-1.5],    # no-match session
            [0.0], [4.0],      # positive session
            [-3.0], [-2.5],    # no-match session
        ])
        groups = np.asarray(["p1", "p1", "n1", "n1", "p2", "p2", "n2", "n2"])
        covis = np.asarray([0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
        model = fit_probabilistic_set(features, groups, covis, l2=0.01)
        candidate, dustbin = model.predict(features, groups)
        metrics = evaluate_probabilistic_set(
            groups, covis, candidate, dustbin)
        self.assertTrue(model.converged)
        self.assertEqual(model.positive_sessions, 2)
        self.assertEqual(metrics["joint_localization_accuracy"], 1.0)
        self.assertEqual(metrics["conditional_candidate_recall_at_1"], 1.0)

    def test_soft_multi_positive_target_is_finite(self):
        features = np.asarray([[2.0], [1.8], [-1.0], [-2.0]])
        groups = np.asarray(["p", "p", "n", "n"])
        covis = np.asarray([0.9, 0.6, 0.0, 0.0])
        model = fit_probabilistic_set(features, groups, covis)
        candidate, dustbin = model.predict(features, groups)
        self.assertTrue(np.isfinite(candidate).all())
        self.assertTrue(np.isfinite(dustbin).all())
        self.assertAlmostEqual(candidate[:2].sum() + dustbin[0], 1.0)

    def test_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            fit_probabilistic_set(
                np.ones((2, 1)), np.asarray(["a"]), np.asarray([0.0, 1.0]))

    def test_scene_oof_predictions_are_complete(self):
        features = np.asarray([
            [3.0], [0.0], [-2.0], [-1.5],
            [0.0], [4.0], [-3.0], [-2.5],
        ])
        groups = np.asarray([
            "p1", "p1", "n1", "n1", "p2", "p2", "n2", "n2"])
        scenes = np.asarray([
            "s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"])
        covis = np.asarray([0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
        candidate, dustbin = scene_group_oof_probabilities(
            features, groups, scenes, covis, l2=0.01, folds=2)
        self.assertTrue(np.isfinite(candidate).all())
        self.assertTrue(np.isfinite(dustbin).all())
        for index in session_indices(groups):
            self.assertAlmostEqual(
                float(candidate[index].sum() + dustbin[index[0]]), 1.0)

    def test_scene_oof_rejects_cross_scene_session(self):
        with self.assertRaisesRegex(ValueError, "crosses scene"):
            scene_group_oof_probabilities(
                np.asarray([[1.0], [0.0], [-1.0], [-2.0]]),
                np.asarray(["shared", "shared", "other", "other"]),
                np.asarray(["s1", "s2", "s1", "s2"]),
                np.asarray([0.9, 0.0, 0.0, 0.0]), l2=0.01, folds=2)

    def test_grouped_ranking_uses_requested_threshold(self):
        metrics = grouped_ranking_metrics(
            np.asarray(["scene", "scene"]),
            np.asarray(["session", "session"]),
            np.asarray([1, 2]), np.asarray([0.6, 0.2]),
            np.asarray([2.0, 1.0]), positive_threshold=0.7)
        self.assertEqual(metrics["scene"]["sessions_with_positive"], 0)


if __name__ == "__main__":
    unittest.main()
