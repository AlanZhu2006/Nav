import gzip
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_goat_bench_dataset import audit_split


class GoatDatasetAuditTest(unittest.TestCase):
    def test_distinguishes_instance_and_category_recurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "val_unseen"
            content = split / "content"
            content.mkdir(parents=True)
            payload = {
                "episodes": [
                    {
                        "tasks": [
                            ["chair", "description", "chair_1"],
                            ["chair", "image", "chair_1", 7],
                            ["chair", "image", "chair_2", 9],
                            ["chair", "image", "chair_1", 11],
                        ]
                    },
                    {"tasks": [["table", "image", "table_1", 2]]},
                ]
            }
            with gzip.open(
                content / "scene.json.gz", "wt", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle)

            result = audit_split(split)

        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["image_subtask_count"], 4)
        self.assertEqual(
            result["image_tasklist_exact_prior_instance_count"], 2
        )
        self.assertEqual(
            result["image_with_prior_description_same_instance_count"], 2
        )
        self.assertEqual(
            result["image_with_prior_image_same_instance_count"], 1
        )
        self.assertEqual(result["image_with_prior_same_category_count"], 3)
        self.assertEqual(
            result["episodes_with_exact_prior_instance_image_query"], 1
        )


if __name__ == "__main__":
    unittest.main()
