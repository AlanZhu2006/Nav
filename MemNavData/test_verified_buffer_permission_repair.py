import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch

import archive_old_hpc_buffers_20260909 as cleanup
from finish_verified_buffer_prune_20260909 import finish, write_pointer


class PermissionRepairTests(unittest.TestCase):
    def test_readonly_owned_parent_is_restored_and_archive_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cold = Path(tmp)/"results", Path(tmp)/"archives"
            target = root/"old"/"run"/"buffer"
            target.mkdir(parents=True)
            data = target/"frame.jpg"
            data.write_bytes(b"original unique data")
            os.utime(data, (time.time()-72*3600,)*2)
            target.chmod(0o555)
            target.parent.chmod(0o555)
            plan, receipts = Path(tmp)/"plan.json", Path(tmp)/"receipts"
            cleanup.write(plan, {"code_sha256": cleanup.digest(cleanup.__file__),
                "archive_root": str(cold), "targets": [str(target)]})
            try:
                with patch.multiple(cleanup, ROOT=root, COLD=cold, FINAL_COLD=Path(tmp)/"final_cold", EXPERIMENTS=("old",)):
                    with self.assertRaises(PermissionError):
                        cleanup.archive_one(plan, cleanup.digest(plan), 0, receipts)
                    receipt_path = next(receipts.glob("*.json"))
                    before = json.loads(receipt_path.read_text())
                    self.assertEqual(before["status"], "archive_verified_before_prune")
                    finish(receipt_path, cleanup)
                    after = json.loads(receipt_path.read_text())
                    original_write = cleanup.write
                    def write(path, value):
                        if Path(path).name.endswith(".ARCHIVED_20260909.json"):
                            return write_pointer(path, value, original_write, cleanup)
                        return original_write(path, value)
                    with patch.object(cleanup, "write", side_effect=write):
                        cleanup.migrate(receipts)
                    self.assertTrue(json.loads(receipt_path.read_text())["cold_transfer_verified"])
                self.assertFalse(target.exists())
                self.assertEqual(stat.S_IMODE(target.parent.stat().st_mode), 0o555)
                self.assertEqual(before["archive_sha256"], after["archive_sha256"])
                self.assertEqual(after["deleted_files"], 1)
                self.assertTrue(after["surviving_parent_mode_restored"])
            finally:
                target.parent.chmod(0o755)
                if target.exists():
                    target.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
