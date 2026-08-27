import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.covisibility_teacher import (
    EpisodeCovisibilityCache,
    backproject_world,
    covisibility_label,
    episode_root,
    parse_path_maps,
    projected_covisibility,
    remap_path,
)


class CovisibilityTeacherTest(unittest.TestCase):
    def test_identity_view_has_unit_overlap(self):
        depth = np.full((8, 10), 2.0, dtype=np.float64)
        intrinsic = np.array([
            [8.0, 0.0, 5.0],
            [0.0, 8.0, 4.0],
            [0.0, 0.0, 1.0],
        ])
        action = np.eye(4)
        # Match the production stride convention, which does not sample the
        # final image border excluded by the projection bounds.
        points = backproject_world(
            depth, intrinsic, action, stride=2)
        score = projected_covisibility(
            points, depth, intrinsic, action, tolerance=1e-6)
        self.assertAlmostEqual(score, 1.0)

    def test_occluding_depth_rejects_query_surface(self):
        depth = np.full((6, 8), 2.0, dtype=np.float64)
        intrinsic = np.array([
            [6.0, 0.0, 4.0],
            [0.0, 6.0, 3.0],
            [0.0, 0.0, 1.0],
        ])
        action = np.eye(4)
        points = backproject_world(depth, intrinsic, action, stride=1)
        occluder = np.full_like(depth, 1.0)
        score = projected_covisibility(
            points, occluder, intrinsic, action, tolerance=0.1)
        self.assertEqual(score, 0.0)

    def test_labels_preserve_ignore_band(self):
        self.assertEqual(covisibility_label(0.8), 1)
        self.assertEqual(covisibility_label(0.05), 0)
        self.assertEqual(covisibility_label(0.3), -1)
        with self.assertRaises(ValueError):
            covisibility_label(0.5, 0.1, 0.5)

    def test_path_helpers(self):
        raw = "/old/scene/episode/videos/chunk/rgb/1.jpg"
        mappings = parse_path_maps(["/old=/new"])
        self.assertEqual(remap_path(raw, mappings), Path(
            "/new/scene/episode/videos/chunk/rgb/1.jpg"))
        path = Path(
            "/tmp/scene/episode/videos/chunk-000/"
            "observation.images.rgb/1.jpg")
        self.assertEqual(episode_root(path), Path("/tmp/scene/episode"))
        self.assertEqual(
            episode_root(Path("/tmp/scene/episode/goal_1.jpg")),
            Path("/tmp/scene/episode"))

    def test_metadata_goal_index_and_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            (root / "meta").mkdir(parents=True)
            metadata = {
                "goals": [
                    {"covis_curve": [0.1, 0.2]},
                    {"covis_curve": [0.7, 0.8]},
                ],
            }
            with open(root / "meta" / "gen_meta.json", "w",
                      encoding="utf-8") as handle:
                json.dump(metadata, handle)
            candidate = (root / "videos" / "chunk-000"
                         / "observation.images.rgb" / "1.jpg")
            cache = EpisodeCovisibilityCache()
            self.assertEqual(
                cache.metadata_covisibility(
                    root / "goal_image.jpg", candidate, 1),
                0.2)
            self.assertEqual(
                cache.metadata_covisibility(
                    root / "goal_2.jpg", candidate, 0),
                0.7)


if __name__ == "__main__":
    unittest.main()
