import unittest
from pathlib import Path

import numpy as np

from MemNavData.diag_distill_geometry_router import (
    Episode,
    cosine_similarity_rows,
    select_episodes,
)
from MemNavData.reliability_router import symmetric_relation_features


def episode(scene: str, name: str) -> Episode:
    root = Path("/") / scene / name
    return Episode(
        scene=scene,
        name=name,
        root=root,
        rgb_dir=root / "rgb",
        switch=10,
        n_frames=20,
        intrinsic=np.eye(3),
    )


class RouterDatasetSelectionTest(unittest.TestCase):
    def test_cosine_only_path_matches_full_relation_features(self):
        goal = np.asarray([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
        memory = np.asarray([[3.0, -1.0, 2.0], [0.2, 4.0, -1.0]])
        expected = symmetric_relation_features(goal, memory)[:, -1]
        np.testing.assert_allclose(
            cosine_similarity_rows(goal, memory), expected,
            rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            cosine_similarity_rows(np.zeros((1, 3)), np.ones((1, 3)))

    def test_scene_filter_and_episode_cap_are_deterministic(self):
        episodes = [
            episode("scene_b", "episode_0002"),
            episode("scene_a", "episode_0001"),
            episode("scene_b", "episode_0000"),
            episode("scene_a", "episode_0000"),
            episode("scene_b", "episode_0001"),
        ]
        selected = select_episodes(
            episodes, ["scene_b"], maximum_per_scene=2)
        self.assertEqual(
            [(item.scene, item.name) for item in selected],
            [
                ("scene_b", "episode_0000"),
                ("scene_b", "episode_0001"),
            ],
        )

    def test_empty_scene_filter_keeps_every_scene_balanced(self):
        selected = select_episodes(
            [
                episode("scene_b", "episode_0001"),
                episode("scene_a", "episode_0001"),
                episode("scene_a", "episode_0000"),
            ],
            [],
            maximum_per_scene=1,
        )
        self.assertEqual(
            [(item.scene, item.name) for item in selected],
            [
                ("scene_a", "episode_0000"),
                ("scene_b", "episode_0001"),
            ],
        )

    def test_invalid_or_empty_selection_is_rejected(self):
        episodes = [episode("scene_a", "episode_0000")]
        with self.assertRaises(ValueError):
            select_episodes(episodes, ["missing"], maximum_per_scene=1)
        with self.assertRaises(ValueError):
            select_episodes(episodes, [], maximum_per_scene=-1)
        with self.assertRaises(ValueError):
            select_episodes([], [], maximum_per_scene=0)


if __name__ == "__main__":
    unittest.main()
