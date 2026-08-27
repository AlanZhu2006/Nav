import unittest

import numpy as np

from MemNavData.listwise_router import (
    fit_listwise_linear,
    ranking_session_indices,
    scene_group_oof_scores,
)


class ListwiseRouterTest(unittest.TestCase):
    def test_fit_ranks_positive_candidate_first(self):
        features = np.asarray([
            [2.0, 0.0], [0.0, 0.0], [-1.0, 0.0],
            [3.0, 1.0], [0.0, 1.0], [-2.0, 1.0],
        ])
        groups = np.asarray(["a", "a", "a", "b", "b", "b"])
        covis = np.asarray([0.9, 0.0, 0.0, 0.8, 0.0, 0.0])
        model = fit_listwise_linear(features, groups, covis, l2=0.01)
        scores = model.score(features)
        self.assertGreater(scores[0], max(scores[1:3]))
        self.assertGreater(scores[3], max(scores[4:6]))
        self.assertEqual(model.training_sessions, 2)
        self.assertTrue(np.isfinite(model.objective))

    def test_novel_only_session_is_not_a_ranking_target(self):
        groups = np.asarray(["positive", "positive", "novel", "novel"])
        covis = np.asarray([0.8, 0.0, 0.0, 0.0])
        sessions = ranking_session_indices(groups, covis, 0.5)
        self.assertEqual(len(sessions), 1)
        np.testing.assert_array_equal(sessions[0], [0, 1])

    def test_scene_oof_is_complete_and_disjoint(self):
        features = []
        groups = []
        scenes = []
        covis = []
        for scene_index in range(3):
            for session_index in range(2):
                group = f"s{scene_index}-{session_index}"
                features.extend([[2.0, scene_index], [0.0, scene_index]])
                groups.extend([group, group])
                scenes.extend([f"scene{scene_index}"] * 2)
                covis.extend([0.9, 0.0])
        scores = scene_group_oof_scores(
            np.asarray(features), np.asarray(groups), np.asarray(scenes),
            np.asarray(covis), l2=0.01, folds=3)
        self.assertTrue(np.isfinite(scores).all())
        for start in range(0, len(scores), 2):
            self.assertGreater(scores[start], scores[start + 1])

    def test_rejects_training_without_contrastive_session(self):
        with self.assertRaises(ValueError):
            fit_listwise_linear(
                np.ones((3, 2)), ["a", "a", "b"],
                np.zeros(3), l2=0.01)


if __name__ == "__main__":
    unittest.main()
