import unittest

import numpy as np

from NavDP.baselines.memnav.reverse_memory_graph import (
    ReverseRouteProgress,
    reverse_metric_nodes,
)


class ReverseMetricNodesTest(unittest.TestCase):
    def test_resamples_straight_path_in_reverse(self):
        pose = np.zeros((7, 3), dtype=np.float64)
        pose[:, 0] = np.arange(7)
        self.assertEqual(
            reverse_metric_nodes(
                pose, start_index=6, anchor_index=0,
                metric_scale=1.0, spacing_m=2.0),
            (4, 2, 0),
        )

    def test_metric_scale_changes_node_spacing(self):
        pose = np.zeros((7, 3), dtype=np.float64)
        pose[:, 2] = np.arange(7)
        self.assertEqual(
            reverse_metric_nodes(
                pose, start_index=6, anchor_index=0,
                metric_scale=0.5, spacing_m=2.0),
            (2, 0),
        )

    def test_anchor_is_kept_after_short_remainder(self):
        pose = np.zeros((6, 3), dtype=np.float64)
        pose[:, 0] = np.arange(6) * 0.2
        self.assertEqual(
            reverse_metric_nodes(
                pose, start_index=5, anchor_index=1,
                metric_scale=1.0, spacing_m=2.0),
            (1,),
        )

    def test_progress_is_monotone(self):
        progress = ReverseRouteProgress(0, 6, (4, 2, 0))
        self.assertEqual(progress.current_node, 4)
        self.assertFalse(progress.accept_distance(0.8, 0.6))
        self.assertTrue(progress.accept_distance(0.6, 0.6))
        self.assertEqual(progress.current_node, 2)
        self.assertTrue(progress.accept_distance(0.1, 0.6))
        self.assertTrue(progress.accept_distance(0.2, 0.6))
        self.assertTrue(progress.complete)
        self.assertIsNone(progress.current_node)

    def test_invalid_indices_fail_closed(self):
        with self.assertRaises(ValueError):
            reverse_metric_nodes(
                np.zeros((3, 3)), start_index=1, anchor_index=2,
                metric_scale=1.0, spacing_m=1.0)

    def test_anchor_at_route_start_needs_no_graph_node(self):
        self.assertEqual(
            reverse_metric_nodes(
                np.zeros((3, 3)), start_index=1, anchor_index=1,
                metric_scale=1.0, spacing_m=1.0),
            (),
        )


if __name__ == "__main__":
    unittest.main()
