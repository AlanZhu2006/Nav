import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_hm3d_fresh_fullmono_selection import verify


ROOT = Path(__file__).resolve().parents[1]


class FreshFullMonoSelectionTest(unittest.TestCase):
    def test_real_protocol_recomputes_all_54_fresh_scenes(self):
        result = verify(
            ROOT / "MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json",
            ROOT / ".diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt",
            ROOT / "MemNavData/hm3d_consumed_scene_audit_20260816.json",
            ROOT / "MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json",
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["prior_consumed_scene_count"], 46)
        self.assertEqual(result["fresh_scene_count"], 54)
        self.assertEqual(result["initial_scene_prefix"], 30)
        self.assertEqual(result["extension_prefixes"], [36, 42, 48, 54])
        self.assertEqual(result["selected_overlap_with_consumed"], [])
        self.assertEqual(result["fresh_scenes"][0]["scene_id"], "rJhMRvNn4DS")
        self.assertEqual(result["fresh_scenes"][-1]["scene_id"], "58NLZxWBSpk")

    def test_modified_scene_order_fails(self):
        protocol = json.loads((
            ROOT / "MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json"
        ).read_text())
        protocol["dataset"]["scenes"][0], protocol["dataset"]["scenes"][1] = (
            protocol["dataset"]["scenes"][1],
            protocol["dataset"]["scenes"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(RuntimeError, "fresh-scene order"):
                verify(
                    path,
                    ROOT / ".diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt",
                    ROOT / "MemNavData/hm3d_consumed_scene_audit_20260816.json",
                    ROOT / "MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json",
                )


if __name__ == "__main__":
    unittest.main()
