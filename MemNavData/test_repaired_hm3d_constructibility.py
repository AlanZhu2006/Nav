"""Run with the existing Habitat interpreter; no pytest installation needed."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import audit_repaired_hm3d_constructibility as audit


class ConstructibilityAuditTests(unittest.TestCase):
    def test_short_a_can_succeed_but_have_no_eligible_revisit_frame(self):
        poses = [{"x": i * .03, "y": 0., "z": 0., "yaw": 0.} for i in range(75)]
        history = audit.trace_history({"poses": poses, "end_position": [2.25, 0., 0.], "end_yaw": 0.})
        with patch.object(audit.builder.history_tools, "goal_distance",
                          side_effect=lambda pf, first, second: float(np.linalg.norm(first - second))):
            result = audit.revisit_frame_audit(SimpleNamespace(pathfinder=None), history)
        self.assertEqual(result["inspected_frame_indices"], [39, 47, 55])
        self.assertLess(result["max_inspected_geodesic_m"], 2.)
        self.assertEqual(result["source_candidates"], [])
        self.assertTrue(result["all_frames_in_2_to_9m_band"])

    def test_geometry_only_replay_restores_original_functions(self):
        b = audit.builder
        history = audit.trace_history({"poses": [], "end_position": [0., 0., 0.], "end_yaw": 0.})
        attempt = {"selected": {"standard": {}}, "natural_error": 'rejected: {"attempts": 5000}',
                   "scene_rank": 3, "source_episode_rank": 0}

        def sampler(*args, **kwargs):
            self.assertEqual(b.render(None, np.zeros(3), 0.), (None, None))
            self.assertIsNone(b.history_tools.goal_world_points(None, None, None))
            self.assertGreaterEqual(b.covis_curve(None, [], [])[0], .1)
            raise b.NaturalNovelConstructionError({"attempts": 5000})

        original = (b.render, b.covis_curve, b.history_tools.goal_world_points, b.pair_tools.query_geometry)
        with patch.object(audit, "restore_revisit_position", return_value=np.zeros(3)), \
             patch.object(b, "sample_natural_novel", sampler):
            result = audit.geometry_replay(SimpleNamespace(pathfinder=None), history, attempt, "scene", "ep")
        self.assertTrue(result["original_geometry_counts_reproduced"])
        self.assertEqual(result["candidates_reaching_visual_support"], 1)
        self.assertFalse(result["visual_support_recomputed"])
        self.assertEqual(original, (b.render, b.covis_curve, b.history_tools.goal_world_points,
                                    b.pair_tools.query_geometry))

    def test_geometry_replay_must_not_silently_accept_different_counts(self):
        b = audit.builder
        history = audit.trace_history({"poses": [], "end_position": [0., 0., 0.], "end_yaw": 0.})
        attempt = {"selected": {"standard": {}}, "natural_error": 'rejected: {"attempts": 5000}',
                   "scene_rank": 3, "source_episode_rank": 0}

        def sampler(*args, **kwargs):
            raise b.NaturalNovelConstructionError({"attempts": 4999})

        with patch.object(audit, "restore_revisit_position", return_value=np.zeros(3)), \
             patch.object(b, "sample_natural_novel", sampler), \
             self.assertRaisesRegex(ValueError, "Geometry replay differs"):
            audit.geometry_replay(SimpleNamespace(pathfinder=None), history, attempt, "scene", "ep")


if __name__ == "__main__":
    unittest.main()
