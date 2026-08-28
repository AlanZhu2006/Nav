import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_navdp_training_hm3d_disjointness_20260828 import (
    parse_listing_scene_ids,
    parse_val_member_scene_ids,
    run_audit,
)


def _listing(tmp: Path, name: str, entries: list[str]) -> Path:
    path = tmp / f"{name}.json"
    path.write_text(json.dumps(
        [{"path": f"vln_n1/traj_data/{name}/{e}"} for e in entries]))
    return path


def _members(tmp: Path, lines: list[str]) -> Path:
    path = tmp / "members.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


class AuditDisjointnessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_parse_listing_extracts_ids_and_prefixes(self):
        path = _listing(self.tmp, "hm3d_d435i",
                        ["00001-UVdNNRcVyV1.tar.gz",
                         "00798-zzzzzzzzzz9.tar.gz"])
        ids = parse_listing_scene_ids(path)
        self.assertEqual(ids, {"UVdNNRcVyV1": 1, "zzzzzzzzzz9": 798})

    def test_parse_listing_rejects_unknown_name(self):
        path = _listing(self.tmp, "hm3d_d435i", ["README.md"])
        with self.assertRaises(ValueError):
            parse_listing_scene_ids(path)

    def test_parse_members_dedupes_per_scene(self):
        path = _members(self.tmp, [
            "00800-TEEsavR23oF/",
            "00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
            "00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.navmesh",
        ])
        ids = parse_val_member_scene_ids(path)
        self.assertEqual(ids, {"TEEsavR23oF": 800, "HaxA7YrQdEC": 801})

    def test_disjoint_verdict(self):
        listing = _listing(self.tmp, "hm3d_d435i",
                           ["00001-UVdNNRcVyV1.tar.gz"])
        members = _members(self.tmp, ["00800-TEEsavR23oF/"])
        report = run_audit({"hm3d_d435i": listing}, members)
        self.assertTrue(report["disjoint"])
        self.assertEqual(report["val_intersection"], [])

    def test_overlap_flips_verdict(self):
        listing = _listing(self.tmp, "hm3d_d435i",
                           ["00800-TEEsavR23oF.tar.gz"])
        members = _members(self.tmp, ["00800-TEEsavR23oF/"])
        report = run_audit({"hm3d_d435i": listing}, members)
        self.assertFalse(report["disjoint"])
        self.assertEqual(report["val_intersection"], ["TEEsavR23oF"])

    def test_train_prefix_range_guard(self):
        # A val-range prefix with an unmatched id must still fail the
        # train-range guard even though the intersection is empty.
        listing = _listing(self.tmp, "hm3d_zed",
                           ["00850-aaaaaaaaaa1.tar.gz"])
        members = _members(self.tmp, ["00800-TEEsavR23oF/"])
        report = run_audit({"hm3d_zed": listing}, members)
        self.assertFalse(report["disjoint"])


if __name__ == "__main__":
    unittest.main()
