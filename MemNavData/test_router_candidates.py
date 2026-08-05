import unittest

from NavDP.baselines.memnav.router_candidates import temporal_nms_candidates


class TemporalCandidateTest(unittest.TestCase):
    def test_adjacent_high_scores_do_not_fill_the_candidate_budget(self):
        scores = [0.99, 0.98, 0.97] + [0.0] * 17 + [0.90]
        selected = temporal_nms_candidates(
            scores, [True] * len(scores), top_k=2, min_frame_gap=4)
        self.assertEqual([row["anchor"] for row in selected], [0, 20])

    def test_mask_tie_break_and_nonfinite_scores(self):
        selected = temporal_nms_candidates(
            [0.7, float("nan"), 0.9, 0.9, 0.8],
            [True, True, False, True, True],
            top_k=3,
            min_frame_gap=1,
        )
        self.assertEqual([row["anchor"] for row in selected], [3, 4, 0])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            temporal_nms_candidates([0.1], [], top_k=1, min_frame_gap=1)
        with self.assertRaises(ValueError):
            temporal_nms_candidates([0.1], [True], top_k=0, min_frame_gap=1)


if __name__ == "__main__":
    unittest.main()
