import math
import unittest

import numpy as np

from MemNavData.episodic_path_field import FrozenEpisodicPath
from MemNavData.revisit_bearing_adapter import VERIFIED_BEARING_RADIUS_M


def pose9(x: float, z: float) -> np.ndarray:
    result = np.zeros(9, dtype=np.float64)
    result[0] = x
    result[2] = z
    result[6] = 1.0  # XYZW identity quaternion occupies pose9[3:7].
    return result


class FrozenEpisodicPathTest(unittest.TestCase):
    def test_long_bent_history_uses_local_path_not_endpoint_chord(self):
        # Forward history: goal -> corner -> causal tail.  The frozen Revisit
        # route reverses it, so its first 2.5 m goes east before turning north.
        translations = np.asarray([
            [4.0, 0.0, 4.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=2,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        guidance = path.guidance(pose9(0.0, 0.0), minimum_progress_m=0.0)
        # Identity camera conversion is [world z, -world x].
        self.assertAlmostEqual(guidance.unit_bearing[0], 0.0, places=7)
        self.assertAlmostEqual(guidance.unit_bearing[1], -1.0, places=7)
        self.assertAlmostEqual(
            math.hypot(*guidance.controller_pointgoal),
            VERIFIED_BEARING_RADIUS_M,
            places=7,
        )
        endpoint = np.asarray([4.0, -4.0]) / math.sqrt(32.0)
        self.assertGreater(
            np.linalg.norm(np.asarray(guidance.unit_bearing) - endpoint),
            0.5,
        )

    def test_short_path_reduces_to_terminal_bearing(self):
        translations = np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=1,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        guidance = path.guidance(pose9(0.0, 0.0), minimum_progress_m=0.0)
        self.assertEqual(guidance.unit_bearing, (1.0, 0.0))
        self.assertTrue(guidance.terminal_within_horizon)
        self.assertFalse(guidance.complete)
        self.assertAlmostEqual(guidance.reference_arc_m, 1.0)

    def test_pnp_terminal_translation_extends_anchor_route(self):
        translations = np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=1,
            anchor_index=0,
            terminal_translation=[0.0, 0.0, 2.0],
            metric_scale_m_per_raw=1.0,
        )
        self.assertAlmostEqual(path.total_length_m, 2.0)
        self.assertEqual(path.source_indices[-1], None)
        np.testing.assert_allclose(path.raw_xz[-1], [0.0, 2.0])

    def test_lingbot_gauge_change_leaves_guidance_unchanged(self):
        translations = np.asarray([
            [3.0, 0.0, 5.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        original = FrozenEpisodicPath.from_history(
            translations,
            start_index=2,
            anchor_index=0,
            metric_scale_m_per_raw=0.8,
        ).guidance(pose9(0.0, 0.0), minimum_progress_m=0.0)
        factor = 7.0
        scaled = FrozenEpisodicPath.from_history(
            translations * factor,
            start_index=2,
            anchor_index=0,
            metric_scale_m_per_raw=0.8 / factor,
        ).guidance(pose9(0.0, 0.0), minimum_progress_m=0.0)
        np.testing.assert_allclose(
            original.unit_bearing, scaled.unit_bearing, atol=1e-12)
        self.assertAlmostEqual(
            original.remaining_m, scaled.remaining_m, places=12)

    def test_projection_progress_never_moves_backward(self):
        translations = np.asarray([
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=1,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        first = path.guidance(pose9(0.0, 0.9), minimum_progress_m=0.0)
        second = path.guidance(
            pose9(0.0, 1.7), minimum_progress_m=first.progress_m)
        regressed_observation = path.guidance(
            pose9(0.0, 1.2), minimum_progress_m=second.progress_m)
        self.assertAlmostEqual(first.progress_m, 0.9)
        self.assertAlmostEqual(second.progress_m, 1.7)
        self.assertAlmostEqual(
            regressed_observation.progress_m, second.progress_m)

    def test_visual_candidate_window_is_the_same_controller_horizon(self):
        translations = np.asarray([
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=4,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        self.assertEqual(
            path.source_indices_in_control_window(minimum_progress_m=0.0),
            (4, 3, 2),
        )
        self.assertEqual(
            path.source_indices_in_control_window(minimum_progress_m=1.5),
            (2, 1, 0),
        )

    def test_visual_route_addressing_is_not_limited_by_control_horizon(self):
        translations = np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 6.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=3,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        self.assertEqual(
            path.source_indices_at_or_after_progress(
                minimum_progress_m=0.0),
            (3, 2, 1, 0),
        )
        self.assertAlmostEqual(path.progress_for_source_index(1), 4.0)
        self.assertEqual(
            path.source_indices_at_or_after_progress(
                minimum_progress_m=4.0),
            (1, 0),
        )

    def test_local_projection_cannot_jump_across_a_hairpin(self):
        # A later route segment returns spatially close to the start, but it is
        # farther than one controller horizon in temporal arc length.
        translations = np.asarray([
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=3,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        projection = path.project_monotone(
            [0.09, 0.0, 0.0], minimum_progress_m=0.0)
        self.assertLessEqual(
            projection.progress_m, VERIFIED_BEARING_RADIUS_M + 1e-12)
        self.assertEqual(projection.segment, 0)

    def test_completed_path_emits_no_spurious_direction(self):
        translations = np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        path = FrozenEpisodicPath.from_history(
            translations,
            start_index=1,
            anchor_index=0,
            metric_scale_m_per_raw=1.0,
        )
        guidance = path.guidance(
            pose9(0.0, 1.0), minimum_progress_m=path.total_length_m)
        self.assertTrue(guidance.complete)
        self.assertIsNone(guidance.unit_bearing)
        self.assertIsNone(guidance.controller_pointgoal)

    def test_invalid_or_degenerate_path_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "no non-zero spatial extent"):
            FrozenEpisodicPath.from_history(
                np.zeros((3, 3)),
                start_index=2,
                anchor_index=0,
                metric_scale_m_per_raw=1.0,
            )
        with self.assertRaisesRegex(ValueError, "metric_scale"):
            FrozenEpisodicPath.from_history(
                np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
                start_index=1,
                anchor_index=0,
                metric_scale_m_per_raw=0.0,
            )


if __name__ == "__main__":
    unittest.main()
