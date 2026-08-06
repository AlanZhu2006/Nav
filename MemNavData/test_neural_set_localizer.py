import unittest

import numpy as np
import torch

from MemNavData.train_neural_set_localizer import (
    NeuralSetLocalizer,
    localization_metrics,
    pack_sessions,
    select_match_threshold,
)


class NeuralSetLocalizerTest(unittest.TestCase):
    def test_pack_adds_dustbin_target_only_for_no_match(self):
        packed = pack_sessions(
            np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["s1", "s1", "s2", "s2"]),
            np.asarray([0.8, 0.1, 0.2, 0.0]),
            positive_threshold=0.5)
        self.assertAlmostEqual(packed.target[0, :-1].sum().item(), 1.0)
        self.assertEqual(packed.target[0, -1].item(), 0.0)
        self.assertEqual(packed.target[1, -1].item(), 1.0)

    def test_model_is_permutation_equivariant_for_candidates(self):
        torch.manual_seed(0)
        model = NeuralSetLocalizer(3, 8, 0.0).eval()
        features = torch.randn(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        original = model(features, mask)
        order = torch.tensor([2, 0, 1])
        permuted = model(features[:, order], mask[:, order])
        self.assertTrue(torch.allclose(
            original[:, :3][:, order], permuted[:, :3], atol=1e-6))
        self.assertTrue(torch.allclose(original[:, -1], permuted[:, -1]))

    def test_metrics_score_correct_candidate_and_dustbin(self):
        packed = pack_sessions(
            np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["s1", "s1", "s2", "s2"]),
            np.asarray([0.8, 0.1, 0.2, 0.0]),
            positive_threshold=0.5)
        probability = np.asarray([
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ])
        report = localization_metrics(
            packed, probability, positive_threshold=0.5)
        self.assertEqual(report["joint_localization_accuracy"], 1.0)
        self.assertEqual(report["conditional_candidate_recall_at_1"], 1.0)
        threshold, calibrated = select_match_threshold(
            packed, probability, positive_threshold=0.5)
        self.assertGreater(threshold, 0.0)
        self.assertEqual(calibrated["joint_localization_accuracy"], 1.0)

    def test_match_accuracy_uses_the_reported_threshold(self):
        packed = pack_sessions(
            np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["s1", "s1", "s2", "s2"]),
            np.asarray([0.8, 0.1, 0.2, 0.0]),
            positive_threshold=0.5)
        probability = np.asarray([
            [0.35, 0.05, 0.60],
            [0.10, 0.10, 0.80],
        ])
        report = localization_metrics(
            packed, probability, positive_threshold=0.5,
            match_threshold=0.30)
        self.assertEqual(report["match_accuracy"], 1.0)
        self.assertEqual(report["match_threshold"], 0.30)


if __name__ == "__main__":
    unittest.main()
