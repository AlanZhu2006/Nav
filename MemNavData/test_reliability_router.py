import math
import tempfile
import unittest

import numpy as np

from MemNavData.reliability_router import (
    LinearReliabilityRouter,
    SelectiveThresholds,
    calibrate_zero_error_thresholds,
    selective_decisions,
    symmetric_relation_features,
)


class ReliabilityRouterTest(unittest.TestCase):
    def test_relation_features_are_symmetric_and_normalized(self):
        goal = np.array([[3.0, 4.0], [1.0, 0.0]])
        memory = np.array([[0.0, 2.0], [0.0, 2.0]])
        forward = symmetric_relation_features(goal, memory)
        reverse = symmetric_relation_features(memory, goal)
        np.testing.assert_allclose(forward, reverse)
        self.assertEqual(forward.shape, (2, 5))
        self.assertTrue(np.all(forward[:, -1] <= 1.0 + 1e-12))

    def test_zero_error_calibration_and_selective_decisions(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.05, 0.10, 0.35, 0.60, 0.80, 0.95])
        thresholds = calibrate_zero_error_thresholds(
            labels, probabilities, min_samples=2)
        decisions = selective_decisions(probabilities, thresholds)
        self.assertGreaterEqual(thresholds.accept_min, 0.35)
        self.assertLessEqual(thresholds.reject_max, 0.60)
        self.assertFalse(np.any(decisions[labels == 0] == 1))
        self.assertFalse(np.any(decisions[labels == 1] == -1))
        self.assertGreater(np.sum(decisions != 0), 0)

    def test_small_tails_defer_instead_of_guessing(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        thresholds = calibrate_zero_error_thresholds(
            labels, probabilities, min_samples=3)
        self.assertEqual(thresholds.reject_max, -math.inf)
        self.assertEqual(thresholds.accept_min, math.inf)
        np.testing.assert_array_equal(
            selective_decisions(probabilities, thresholds),
            np.zeros(4, dtype=np.int8))

    def test_portable_linear_head_round_trip(self):
        thresholds = SelectiveThresholds(
            reject_max=0.2,
            accept_min=0.8,
            reject_calibration_count=20,
            accept_calibration_count=20,
            min_samples=20,
        )
        router = LinearReliabilityRouter(
            mean=np.zeros(5),
            scale=np.ones(5),
            coefficient=np.array([0.5, -0.25, 0.1, 0.2, 2.0]),
            intercept=-0.3,
            thresholds=thresholds,
        )
        goal = np.array([[1.0, 0.0], [1.0, 1.0]])
        memory = np.array([[1.0, 0.0], [-1.0, 1.0]])
        expected = router.predict_proba(goal, memory)
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            router.save(handle.name)
            restored = LinearReliabilityRouter.load(handle.name)
        np.testing.assert_allclose(restored.predict_proba(goal, memory), expected)
        np.testing.assert_array_equal(
            restored.decisions(goal, memory),
            selective_decisions(expected, thresholds))


if __name__ == "__main__":
    unittest.main()
