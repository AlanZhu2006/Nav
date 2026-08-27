import tempfile
import unittest
from pathlib import Path

from MemNavData.collect_hm3d_fullmono_goal_a import collection_actions


class GoalAResumePlanTest(unittest.TestCase):
    def test_fresh_collection_runs_every_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            self.assertEqual(
                collection_actions(root, ["episode_0000", "episode_0001"], False),
                [("episode_0000", "run"), ("episode_0001", "run")],
            )
            self.assertTrue((root / "logs").is_dir())

    def test_resume_audits_existing_and_runs_only_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            existing = root / "episode_0000"
            existing.mkdir(parents=True)
            (existing / "receipt.json").write_text("{}\n")
            self.assertEqual(
                collection_actions(root, ["episode_0000", "episode_0001"], True),
                [("episode_0000", "audit"), ("episode_0001", "run")],
            )

    def test_empty_failed_episode_directory_is_filled_additively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            (root / "episode_0000").mkdir(parents=True)
            self.assertEqual(
                collection_actions(root, ["episode_0000"], True),
                [("episode_0000", "run")],
            )

    def test_normal_mode_never_reuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            root.mkdir()
            with self.assertRaises(RuntimeError):
                collection_actions(root, ["episode_0000"], False)

    def test_completed_scene_cannot_be_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            root.mkdir()
            (root / "completion.json").write_text("{}\n")
            with self.assertRaises(RuntimeError):
                collection_actions(root, ["episode_0000"], True)


if __name__ == "__main__":
    unittest.main()
