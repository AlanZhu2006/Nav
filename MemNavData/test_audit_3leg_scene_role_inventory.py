import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_3leg_scene_role_inventory import build_receipt


class SceneRoleInventoryTest(unittest.TestCase):
    def test_only_remaining_scenes_are_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_root = root / "episodes"
            for scene, count in {
                "consumed_a": 2,
                "train_a": 1,
                "blind_a": 3,
                "blind_b": 2,
            }.items():
                for index in range(count):
                    (episode_root / scene / f"episode_{index:04d}").mkdir(
                        parents=True
                    )

            consumed = root / "consumed.json"
            consumed.write_text(json.dumps({"scenes": ["consumed_a"]}))
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "train": ["train_a"],
                        "development": ["dev_a"],
                        "final_reserved": ["final_a"],
                    }
                )
            )
            blind = root / "blind.json"
            blind.write_text(
                json.dumps(
                    {"selection": {"selected_scenes": ["blind_a", "blind_b"]}}
                )
            )

            receipt = build_receipt(
                three_leg_root=episode_root,
                consumed_manifest=consumed,
                role_split=split,
                blind_role_manifest=blind,
                expected_episode_count=8,
            )

        self.assertEqual(receipt["decision"], "stop_before_blind_confirmation")
        self.assertEqual(receipt["counts"]["three_leg_nonempty_scene_clusters"], 4)
        self.assertEqual(
            receipt["counts"]["remaining_after_consumed_train_development_final"],
            2,
        )
        self.assertEqual(receipt["counts"]["remaining_intersection_blind16"], 2)
        self.assertEqual(receipt["counts"]["remaining_outside_blind"], 0)
        self.assertNotIn("blind_a", json.dumps(receipt))

    def test_detects_nonblind_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "episodes" / "fresh_a" / "episode_0000").mkdir(parents=True)
            (root / "consumed.json").write_text(json.dumps({"scenes": []}))
            (root / "split.json").write_text(
                json.dumps({"train": [], "development": [], "final_reserved": []})
            )
            (root / "blind.json").write_text(
                json.dumps({"selection": {"selected_scenes": []}})
            )

            receipt = build_receipt(
                three_leg_root=root / "episodes",
                consumed_manifest=root / "consumed.json",
                role_split=root / "split.json",
                blind_role_manifest=root / "blind.json",
            )

        self.assertEqual(
            receipt["decision"], "nonblind_scene_disjoint_candidates_exist"
        )
        self.assertEqual(receipt["counts"]["remaining_outside_blind"], 1)


if __name__ == "__main__":
    unittest.main()
