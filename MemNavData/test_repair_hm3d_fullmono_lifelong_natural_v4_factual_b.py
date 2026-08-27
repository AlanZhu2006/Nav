from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import repair_hm3d_fullmono_lifelong_natural_v4_factual_b as repair


def seal(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )


class MissingFactualBRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        factual = self.run / "factual_b"
        factual.mkdir(parents=True)
        self.episodes = [
            {"scene": f"scene_{i // 3}", "episode": f"episode_{i:03d}"}
            for i in range(99)
        ]
        for index, item in enumerate(self.episodes):
            if index in repair.FROZEN_MISSING_INDICES:
                continue
            output = factual / repair.label_for(index, item)
            output.mkdir()
            completion = output / "completion.json"
            completion.write_text("{}\n")
            seal(completion)
        for index in (51, 62):
            output = factual / repair.label_for(index, self.episodes[index])
            (output / "logs").mkdir(parents=True)
            (output / "logs/eval.log").write_text("opaque partial bytes\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_audit_and_archive_preserve_partial_outputs(self) -> None:
        result = repair.audit(self.run, self.episodes)
        self.assertEqual(result["missing_history_indices"], [51, 52, 62, 63])
        receipt = repair.archive_partial(
            self.run, self.episodes, 31, "missingfix1"
        )
        self.assertEqual(receipt["history_indices"], [51, 52])
        self.assertTrue(receipt["rows"][0]["partial_output_present"])
        self.assertFalse(receipt["rows"][1]["partial_output_present"])
        archive = (self.run / "failed_attempts" /
                   "factual_b_missingfix1_shard031")
        self.assertTrue((archive / "archive_receipt.json.sha256").is_file())
        self.assertFalse((self.run / "factual_b" /
                          repair.label_for(51, self.episodes[51])).exists())

    def test_completed_repair_target_is_rejected(self) -> None:
        item = self.episodes[51]
        output = self.run / "factual_b" / repair.label_for(51, item)
        completion = output / "completion.json"
        completion.write_text("{}\n")
        seal(completion)
        with self.assertRaisesRegex(RuntimeError, "overwrite completed"):
            repair.archive_partial(
                self.run, self.episodes, 31, "shouldfail"
            )


if __name__ == "__main__":
    unittest.main()
