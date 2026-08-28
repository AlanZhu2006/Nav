import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.hm3d_fullmono_mixed_role import (
    ARMS,
    bind_parent_manifest,
    rotated_arm_order,
    selected_arm_order,
)
from MemNavData.finalize_hm3d_fullmono_mixed_role import choose_scene_prefix


class FullMonoMixedRoleContractTest(unittest.TestCase):
    def test_rotation_is_balanced_and_complete(self):
        self.assertEqual(rotated_arm_order(0), ARMS)
        self.assertEqual(rotated_arm_order(1), ARMS[1:] + ARMS[:1])
        self.assertEqual(rotated_arm_order(2), ARMS[2:] + ARMS[:2])
        for index in range(12):
            self.assertEqual(set(rotated_arm_order(index)), set(ARMS))

    def test_table1_native_cec_subset_alternates_without_changing_legacy(self):
        pair = ("mono_native", "mono_cec")
        self.assertEqual(selected_arm_order(0, ARMS), rotated_arm_order(0))
        self.assertEqual(selected_arm_order(1, pair), pair[::-1])
        self.assertEqual(selected_arm_order(2, pair), pair)
        with self.assertRaisesRegex(RuntimeError, "requires mono_native"):
            selected_arm_order(0, ("mono_cec",))
        with self.assertRaisesRegex(RuntimeError, "duplicates"):
            selected_arm_order(0, ("mono_native", "mono_native", "mono_cec"))

    def test_protocol_freezes_all_36_sources(self):
        path = Path(__file__).with_name(
            "hm3d_fullmono_mixed_role_protocol_20260820.json"
        )
        protocol = json.loads(path.read_text())
        scenes = protocol["dataset"]["scenes"]
        self.assertEqual(len(scenes), 9)
        self.assertEqual([row["rank"] for row in scenes], list(range(9)))
        self.assertEqual(len({row["scene_id"] for row in scenes}), 9)
        self.assertEqual(protocol["dataset"]["source_episode_count"], 36)
        self.assertEqual(protocol["query"]["arms"], list(ARMS))
        self.assertEqual(protocol["query"]["metric_depth_sensor_reads_allowed"], 0)

    def test_fresh_protocol_freezes_all_54_scenes(self):
        path = Path(__file__).with_name(
            "hm3d_fresh_fullmono_mixed_role_protocol_20260820.json"
        )
        protocol = json.loads(path.read_text())
        scenes = protocol["dataset"]["scenes"]
        self.assertEqual(len(scenes), 54)
        self.assertEqual([row["rank"] for row in scenes], list(range(54)))
        self.assertEqual(len({row["scene_id"] for row in scenes}), 54)
        self.assertEqual(protocol["dataset"]["target_source_episode_count"], 216)
        self.assertEqual(protocol["query"]["arms"], list(ARMS))

    def test_prefix_stops_at_first_constructibility_target(self):
        selection = {
            "initial_scene_prefix": 30, "extension_block_scenes": 6,
            "maximum_scene_prefix": 54, "target_histories": 24,
            "target_scene_clusters": 15,
        }
        fragments = [
            {"scene_index": index,
             "retained_histories": 2 if index < 15 else 0}
            for index in range(54)
        ]
        result = choose_scene_prefix(fragments, selection)
        self.assertEqual(result["selected_scene_prefix"], 30)
        self.assertTrue(result["target_met"])

        fragments = [
            {"scene_index": index,
             "retained_histories": (
                 2 if index < 10 else
                 1 if 10 <= index < 14 else
                 2 if 30 <= index < 32 else 0)}
            for index in range(54)
        ]
        result = choose_scene_prefix(fragments, selection)
        self.assertEqual(result["selected_scene_prefix"], 36)
        self.assertTrue(result["target_met"])

    def test_prefix_exhaustion_is_reported_underpowered(self):
        selection = {
            "initial_scene_prefix": 30, "extension_block_scenes": 6,
            "maximum_scene_prefix": 54, "target_histories": 24,
            "target_scene_clusters": 15,
        }
        fragments = [
            {"scene_index": index, "retained_histories": int(index < 10)}
            for index in range(54)
        ]
        result = choose_scene_prefix(fragments, selection)
        self.assertEqual(result["selected_scene_prefix"], 54)
        self.assertFalse(result["target_met"])

    def test_fresh_parent_is_bound_by_generated_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol = {
                "schema_version": (
                    "hm3d_fresh_fullmono_mixed_role_protocol_v1_20260820"),
                "dataset": {},
            }
            protocol_path.write_text(json.dumps(protocol))
            protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            parent_path = root / "parent.json"
            parent = {
                "protocol_sha256": protocol_sha,
                "query_outcomes_read": False,
                "fresh_scene_generalization": True,
            }
            parent_path.write_text(json.dumps(parent))
            parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
            (root / "parent.json.sha256").write_text(
                parent_sha + "  parent.json\n")
            observed, digest = bind_parent_manifest(
                protocol, protocol_path, parent_path)
            self.assertEqual(observed, parent)
            self.assertEqual(digest, parent_sha)
            parent_path.write_text(json.dumps({**parent, "query_outcomes_read": True}))
            with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                bind_parent_manifest(protocol, protocol_path, parent_path)

if __name__ == "__main__":
    unittest.main()
