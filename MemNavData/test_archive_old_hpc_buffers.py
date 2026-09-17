import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import archive_old_hpc_buffers_20260909 as cleanup


class ArchiveBeforePruneTests(unittest.TestCase):
    def make(self, tmp):
        root, cold = Path(tmp)/"results", Path(tmp)/"cold"
        target = root/"experiment"/"old_run"/"buffer"
        target.mkdir(parents=True)
        (target/"rgb.jpg").write_bytes(b"unique recorded pixels")
        os.utime(target/"rgb.jpg", (time.time()-72*3600,)*2)
        plan = Path(tmp)/"plan.json"
        cleanup.write(plan, {"code_sha256": cleanup.digest(cleanup.__file__), "archive_root": str(cold),
                             "targets": [str(target)]})
        return root, cold, target, plan

    def test_verified_archive_is_recoverable_after_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cold, target, plan = self.make(tmp)
            with patch.multiple(cleanup, ROOT=root, COLD=cold, EXPERIMENTS=("experiment",)):
                cleanup.archive_one(plan, cleanup.digest(plan), 0, Path(tmp)/"receipts")
            self.assertFalse(target.exists())
            receipt = json.loads(target.with_name("buffer.ARCHIVED_20260909.json").read_text())
            self.assertEqual(receipt["deleted_files"], 1)
            with tarfile.open(receipt["archive"], "r:gz") as tar:
                self.assertEqual(tar.extractfile("buffer/rgb.jpg").read(), b"unique recorded pixels")

    def test_recent_or_symlink_data_is_not_deleted(self):
        for problem in ("recent", "symlink"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root, cold, target, plan = self.make(tmp)
                if problem == "recent":
                    os.utime(target/"rgb.jpg", None)
                else:
                    (target/"external").symlink_to(Path(tmp)/"unrelated")
                with patch.multiple(cleanup, ROOT=root, COLD=cold, EXPERIMENTS=("experiment",)):
                    with self.assertRaises(RuntimeError):
                        cleanup.archive_one(plan, cleanup.digest(plan), 0, Path(tmp)/"receipts")
                self.assertTrue((target/"rgb.jpg").is_file())

    def test_changed_source_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cold, target, plan = self.make(tmp)
            original = cleanup.inventory
            count = 0
            def inventory(path):
                nonlocal count
                count += 1
                rows, dirs = original(path)
                if count == 2:
                    rows[0]["mtime_ns"] += 1
                return rows, dirs
            with patch.multiple(cleanup, ROOT=root, COLD=cold, EXPERIMENTS=("experiment",)):
                with patch.object(cleanup, "inventory", side_effect=inventory):
                    with self.assertRaisesRegex(RuntimeError, "source changed"):
                        cleanup.archive_one(plan, cleanup.digest(plan), 0, Path(tmp)/"receipts")
            self.assertTrue((target/"rgb.jpg").is_file())

    def test_broad_or_wrong_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cold, target, plan = self.make(tmp)
            with patch.multiple(cleanup, ROOT=root, COLD=cold, EXPERIMENTS=("experiment",)):
                for path in (root, root/"experiment", target.parent):
                    with self.assertRaises(RuntimeError):
                        cleanup.validate_target(path)

    def test_cold_migration_preserves_archive_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cold, target, plan = self.make(tmp)
            final = Path(tmp)/"archive"/"cold"
            final.parent.mkdir()
            receipts = Path(tmp)/"receipts"
            with patch.multiple(cleanup, ROOT=root, COLD=cold, FINAL_COLD=final, EXPERIMENTS=("experiment",)):
                cleanup.archive_one(plan, cleanup.digest(plan), 0, receipts)
                cleanup.migrate(receipts)
            receipt = json.loads(next(receipts.glob("*.json")).read_text())
            self.assertTrue(receipt["cold_transfer_verified"])
            self.assertEqual(Path(receipt["archive"]).parent, final)
            self.assertEqual(cleanup.digest(receipt["archive"]), receipt["archive_sha256"])
            self.assertFalse(Path(receipt["previous_scratch_archive"]).exists())


if __name__ == "__main__":
    unittest.main()
