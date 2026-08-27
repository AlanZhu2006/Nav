import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from MemNavData.audit_navdp_arrival_consensus import (
    EpisodeSource,
    GoalSpec,
    aggregate_states,
    deterministic_seed,
    roc_auc,
    select_goal_states,
    summarize_rollout,
)


class ArrivalConsensusTest(unittest.TestCase):
    def test_state_selection_is_one_per_fixed_band(self):
        source = EpisodeSource("scene", "episode_0000", Path("/tmp/episode"))
        goal = GoalSpec(
            goal_index=1,
            name="B",
            image=Path("/tmp/goal_1.jpg"),
            position_xy=np.zeros(2),
            segment_start=0,
            segment_end=7,
        )
        positions = np.asarray([
            [0.10, 0.0], [0.40, 0.0], [0.75, 0.0], [1.50, 0.0],
            [3.00, 0.0], [5.00, 0.0], [8.00, 0.0],
        ])
        states = select_goal_states(source, goal, positions)
        self.assertEqual(len(states), 6)
        self.assertEqual(
            [state.distance_band for state in states],
            [
                "arrived_025", "near_miss_050", "near_100", "mid_200",
                "far_400", "very_far",
            ],
        )
        self.assertEqual(len({state.frame_index for state in states}), 6)

    def test_rollout_summary_distinguishes_one_zero_from_consensus(self):
        candidates = np.zeros((1, 4, 3, 3), dtype=float)
        candidates[0, 1:, :, 0] = np.asarray([1.0, 2.0, 3.0])[:, None]
        payload = {
            "trajectory": candidates[:, 0],
            "all_trajectory": candidates,
            "all_values": [[0.9, 0.8, 0.7, 0.6]],
            "diffusion_seed": 17,
        }
        result = summarize_rollout(payload, stop_threshold=-0.5)
        self.assertTrue(result["selected_zero"])
        self.assertEqual(result["candidate_zero_count"], 1)
        self.assertEqual(result["candidate_zero_fraction"], 0.25)
        self.assertAlmostEqual(
            result["zero_over_nonzero_critic_margin"], 0.1)
        self.assertFalse(result["critic_fallback"])

    def test_aggregate_and_auc(self):
        rows = []
        for state, label, zeros in (
                ("positive", True, (True, True)),
                ("negative", False, (False, False))):
            for index, zero in enumerate(zeros):
                rows.append({
                    "state_id": state,
                    "scene": state,
                    "episode": "episode_0000",
                    "goal_index": 1,
                    "goal_name": "B",
                    "frame_index": index,
                    "distance_band": "arrived_025" if label else "far_400",
                    "euclidean_distance_m": 0.1 if label else 3.0,
                    "arrival_025": label,
                    "sample_index": index,
                    "selected_zero": zero,
                    "candidate_zero_fraction": 1.0 if zero else 0.0,
                    "top4_zero_fraction": 1.0 if zero else 0.0,
                    "critic_max": 0.0,
                    "critic_fallback": False,
                    "zero_over_nonzero_critic_margin": np.nan,
                    "request_latency_s": 0.1,
                })
        states = aggregate_states(pd.DataFrame(rows))
        self.assertEqual(len(states), 2)
        self.assertEqual(
            roc_auc(states["arrival_025"], states["selected_zero_rate"]),
            1.0,
        )

    def test_seed_is_stable_and_sample_specific(self):
        first = deterministic_seed(1, "state", 0)
        self.assertEqual(first, deterministic_seed(1, "state", 0))
        self.assertNotEqual(first, deterministic_seed(1, "state", 1))


if __name__ == "__main__":
    unittest.main()
