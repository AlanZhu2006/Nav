import types
import unittest

from MemNavData.goat_contract_smoke import (
    _adapter_contract,
    _episode_scene_id,
    _parse_episode,
    _select_episodes,
)


class GoatContractSmokeTest(unittest.TestCase):
    def test_adapter_contract_separates_proposal_from_semantic_stop(self):
        contract = _adapter_contract()
        self.assertTrue(contract["no_motion_is_only_arrival_proposal"])
        self.assertTrue(contract["adapter_never_emits_subtask_stop"])
        self.assertEqual(contract["stop_action_name"], "subtask_stop")
        self.assertEqual(contract["straight_action_ids"], [1, 1])

    def test_parse_episode(self):
        self.assertEqual(_parse_episode("scene:7"), ("scene", "7"))

    def test_scene_id_strips_basis_glb(self):
        episode = types.SimpleNamespace(
            scene_id="/data/hm3d/00877-scene/scene.basis.glb"
        )
        self.assertEqual(_episode_scene_id(episode), "scene")

    def test_selection_preserves_requested_order(self):
        episodes = [
            types.SimpleNamespace(
                scene_id="/data/002-b/b.basis.glb",
                episode_id="4",
                tasks=[["chair", "image", "chair_1", 0]],
            ),
            types.SimpleNamespace(
                scene_id="/data/001-a/a.basis.glb",
                episode_id="3",
                tasks=[["table", "image", "table_1", 0]],
            ),
        ]
        selected = _select_episodes(episodes, (("a", "3"), ("b", "4")))
        self.assertEqual([ep.episode_id for ep in selected], ["3", "4"])

    def test_selection_rejects_non_image_first_subtask(self):
        episodes = [
            types.SimpleNamespace(
                scene_id="/data/001-a/a.basis.glb",
                episode_id="3",
                tasks=[["chair", "object", None]],
            ),
            types.SimpleNamespace(
                scene_id="/data/002-b/b.basis.glb",
                episode_id="4",
                tasks=[["table", "image", "table_1", 0]],
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "must begin with an ImageGoal"):
            _select_episodes(episodes, (("a", "3"), ("b", "4")))


if __name__ == "__main__":
    unittest.main()
