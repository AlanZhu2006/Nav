import unittest

from MemNavData.test_lingbot_native_training import synthetic_rows
from MemNavData.train_lingbot_native_localizer import (
    balanced_overfit_subset,
    build_feature_matrix,
    pack_exact_sessions,
)


class PhaseBTrainingPreflightTest(unittest.TestCase):
    def test_balanced_subset_preserves_positive_and_no_match(self):
        frame = synthetic_rows()
        features, _names, predicted, target = build_feature_matrix(frame)
        packed = pack_exact_sessions(
            features,
            frame["session_id"].to_numpy(dtype=str),
            frame["scene"].to_numpy(dtype=str),
            frame["teacher_covis"].to_numpy(),
            predicted,
            target,
            frame["session_has_positive"].to_numpy(dtype=bool),
            frame["session_is_strict_no_match"].to_numpy(dtype=bool),
            positive_threshold=0.5,
            negative_threshold=0.2,
        )
        subset = balanced_overfit_subset(packed, max_sessions=2)
        self.assertEqual(len(subset.session_ids), 2)
        self.assertTrue(bool((subset.selected_match_target > 0.5).any()))
        self.assertTrue(bool((subset.no_match_target > 0.5).any()))
        self.assertEqual(subset.features.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
