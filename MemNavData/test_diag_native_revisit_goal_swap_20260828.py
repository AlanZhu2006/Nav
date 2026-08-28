import math
import unittest

import numpy as np

from MemNavData.diag_native_revisit_goal_swap_20260828 import (
    aggregate_probe,
    donor_assignment,
    endpoint_heading_deg,
    heading_delta_deg,
    probe_contrasts,
    rms_divergence,
)


def _resp(traj_scale: float, seed_offset: float = 0.0):
    base = np.linspace(0.0, 1.0, 24 * 3).reshape(24, 3)
    candidates = np.stack([base + i * 0.01 + seed_offset for i in range(4)])
    selected = candidates[0] * traj_scale
    return {
        "all_trajectory": (candidates * traj_scale).tolist(),
        "trajectory": selected.tolist(),
        "all_values": [0.1 * traj_scale, 0.2, 0.3],
    }


class DonorAssignmentTest(unittest.TestCase):
    def test_donor_always_other_scene(self):
        labels = ["000_sceneA_episode_0000", "001_sceneA_episode_0001",
                  "002_sceneB_episode_0000", "003_sceneC_episode_0000"]
        donors = donor_assignment(labels)
        self.assertEqual(set(donors), set(labels))
        for label, donor in donors.items():
            self.assertNotEqual(label.split("_", 2)[1],
                                donor.split("_", 2)[1])

    def test_single_scene_rejected(self):
        with self.assertRaises(ValueError):
            donor_assignment(["000_s_e0", "001_s_e1"])


class MetricTest(unittest.TestCase):
    def test_rms_zero_for_identical(self):
        self.assertEqual(rms_divergence([[1.0, 2.0]], [[1.0, 2.0]]), 0.0)

    def test_rms_shape_mismatch(self):
        with self.assertRaises(ValueError):
            rms_divergence([[1.0]], [[1.0, 2.0]])

    def test_endpoint_heading(self):
        traj = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        self.assertAlmostEqual(endpoint_heading_deg(traj), 45.0)

    def test_heading_delta_wraps(self):
        left = [[0.0, 0.0, 0.0], [-1.0, -0.001, 0.0]]
        right = [[0.0, 0.0, 0.0], [-1.0, 0.001, 0.0]]
        self.assertLess(heading_delta_deg(left, right), 1.0)


class ContrastTest(unittest.TestCase):
    def test_identical_swap_gives_zero_and_ratio_none_guard(self):
        responses = {
            "correct": _resp(1.0),
            "swapped": _resp(1.0),
            "seed_shift": _resp(1.0, seed_offset=0.05),
            "novel_goal": _resp(1.0),
        }
        contrasts = probe_contrasts(responses)
        self.assertEqual(contrasts["swapped"]["candidate_rms"], 0.0)
        self.assertGreater(contrasts["seed_shift"]["candidate_rms"], 0.0)
        self.assertEqual(contrasts["swap_over_seed_rms_ratio"], 0.0)

    def test_active_conditioning_yields_large_ratio(self):
        responses = {
            "correct": _resp(1.0),
            "swapped": _resp(2.0),
            "seed_shift": _resp(1.0, seed_offset=0.01),
            "novel_goal": _resp(1.0),
        }
        contrasts = probe_contrasts(responses)
        self.assertGreater(contrasts["swap_over_seed_rms_ratio"], 5.0)


class AggregateTest(unittest.TestCase):
    def test_aggregate_counts_ratios(self):
        rows = []
        for ratio, swap_h, seed_h in ((0.5, 3.0, 4.0), (2.0, 30.0, 5.0)):
            rows.append({"contrasts": {
                "swap_over_seed_rms_ratio": ratio,
                "swapped": {"selected_heading_delta_deg": swap_h},
                "seed_shift": {"selected_heading_delta_deg": seed_h},
            }})
        agg = aggregate_probe(rows)
        self.assertEqual(agg["swap_over_seed_rms_ratio"]["n"], 2)
        self.assertEqual(agg["histories_with_ratio_above_1"], 1)
        self.assertAlmostEqual(
            agg["swapped_selected_heading_delta_deg"]["median"], 16.5)


if __name__ == "__main__":
    unittest.main()
