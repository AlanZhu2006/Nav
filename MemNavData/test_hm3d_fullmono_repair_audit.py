import tempfile
import unittest
from pathlib import Path

from MemNavData.hm3d_fullmono_repair_audit import (
    episode_tree_hashes,
    parse_indices,
)


class HM3DFullMonoRepairAuditTest(unittest.TestCase):
    def test_indices_are_sorted_and_unique(self):
        self.assertEqual(
            parse_indices("53,47,29,46,29"), [29, 46, 47, 53])

    def test_episode_inventory_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            episode = scene / "episode_0000"
            episode.mkdir()
            (episode / "receipt.json").write_text("{\"ok\": true}\n")
            result = episode_tree_hashes(scene, "episode_0000")
            self.assertEqual(set(result), {"episode_0000/receipt.json"})
            self.assertEqual(len(result["episode_0000/receipt.json"]), 64)

    def test_missing_episode_has_empty_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                episode_tree_hashes(Path(directory), "episode_0000"), {})

    def test_empty_failed_directory_has_empty_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / "episode_0000").mkdir()
            self.assertEqual(episode_tree_hashes(scene, "episode_0000"), {})


if __name__ == "__main__":
    unittest.main()
