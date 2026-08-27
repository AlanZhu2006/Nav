import unittest

import pandas as pd

from MemNavData.build_router_cross_episode_pairs import (
    collect_episode_frames,
    hard_teacher_frames,
    paired_episode_names,
    query_indices,
    return_query_indices,
)


class CrossEpisodePairBuilderTest(unittest.TestCase):
    def test_collects_contiguous_frames_and_pairs_adjacent_episodes(self):
        rows = []
        for episode in ("episode_0000", "episode_0001"):
            for frame in range(5):
                rows.append({
                    "scene": "scene_a",
                    "episode": episode,
                    "candidate_frame": frame,
                    "candidate_path": f"/{episode}/{frame}.jpg",
                })
        episodes = collect_episode_frames(pd.DataFrame(rows), ["scene_a"])
        self.assertEqual(len(episodes), 2)
        self.assertEqual(paired_episode_names(episodes, "scene_a"), {
            "episode_0000": "episode_0001",
            "episode_0001": "episode_0000",
        })

    def test_rejects_incomplete_trajectory_and_odd_episode_count(self):
        incomplete = pd.DataFrame([
            {"scene": "scene_a", "episode": "episode_0000",
             "candidate_frame": 0, "candidate_path": "/0.jpg"},
            {"scene": "scene_a", "episode": "episode_0000",
             "candidate_frame": 2, "candidate_path": "/2.jpg"},
        ])
        with self.assertRaises(ValueError):
            collect_episode_frames(incomplete, ["scene_a"])
        episodes = {("scene_a", "episode_0000"): {0: "/0.jpg"}}
        with self.assertRaises(ValueError):
            paired_episode_names(episodes, "scene_a")

    def test_query_indices_are_deterministic_and_bounded(self):
        self.assertEqual(query_indices(100, stride=20, margin=10, maximum=0),
                         [10, 30, 50, 70])
        limited = query_indices(100, stride=10, margin=10, maximum=3)
        self.assertEqual(len(limited), 3)
        self.assertEqual(limited, sorted(limited))
        self.assertTrue(all(10 <= item < 90 for item in limited))

    def test_return_queries_are_post_switch_and_bounded(self):
        self.assertEqual(
            return_query_indices(
                200, switch=80, stride=30, margin=10, maximum=0),
            [90, 120, 150, 180])
        limited = return_query_indices(
            200, switch=80, stride=10, margin=10, maximum=3)
        self.assertEqual(len(limited), 3)
        self.assertTrue(all(80 < item < 190 for item in limited))
        with self.assertRaises(ValueError):
            return_query_indices(
                200, switch=200, stride=10, margin=10, maximum=3)

    def test_hard_teacher_frames_use_score_then_frame_tiebreak(self):
        selected = hard_teacher_frames(
            [30, 10, 20, 40], [0.7, 0.9, 0.9, 0.1], top_k=2)
        self.assertEqual(selected, frozenset({10, 20}))
        self.assertEqual(
            hard_teacher_frames([2, 1], [0.1, 0.2], top_k=0),
            frozenset({1, 2}))
        with self.assertRaises(ValueError):
            hard_teacher_frames([1, 1], [0.1, 0.2], top_k=1)


if __name__ == "__main__":
    unittest.main()
