import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch

import archive_hpc_buffers_phase2_20260909 as phase2


class Phase2CleanupTests(unittest.TestCase):
    def test_archive_and_permission_repair_finish_in_same_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            target = root / "old" / "run" / "buffer"
            target.mkdir(parents=True)
            (target / "frame.jpg").write_bytes(b"historical observation")
            os.utime(target / "frame.jpg", (time.time() - 72 * 3600,) * 2)
            target.chmod(0o555)
            target.parent.chmod(0o555)
            population = Path(tmp) / "population.json"
            population.write_text("{}")
            plan, receipts = Path(tmp) / "plan.json", Path(tmp) / "receipts"
            cold = Path(tmp) / "archives"
            try:
                with patch.multiple(phase2.base, ROOT=root, COLD=cold, EXPERIMENTS=("old",)):
                    with patch.object(phase2, "POPULATION", population):
                        phase2.base.write(plan, {
                            "code_sha256": phase2.BASE_SHA, "archive_root": str(cold),
                            "targets": [str(target)], "wrapper_sha256": phase2.digest(phase2.HERE),
                            "protected_population_sha256": phase2.digest(population),
                        })
                        phase2.execute(plan, phase2.digest(plan), 0, receipts)
                        row = json.loads(next(receipts.glob("*.json")).read_text())
                        self.assertEqual(row["status"], "archived_and_pruned")
                        self.assertTrue(row["permission_repair"])
                        self.assertEqual(row["archive_sha256"], phase2.digest(row["archive"]))
                        self.assertFalse(target.exists())
                        self.assertTrue(target.with_name("buffer.ARCHIVED_20260909.json").is_file())
                        self.assertEqual(stat.S_IMODE(target.parent.stat().st_mode), 0o555)
            finally:
                target.parent.chmod(0o755)
                if target.exists():
                    target.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
