import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.finalize_shared_online_double_revisit_fresh20 import parse_statuses
from MemNavData.prepare_shared_online_double_revisit_fresh import (
    is_expected_construction_failure,
    round_robin_select,
)


class FreshPreparationTest(unittest.TestCase):
    def test_scene_round_robin_is_balanced_and_stable(self):
        rows = [
            {"scene": "a", "episode": "episode_0002", "candidate_index": 2},
            {"scene": "b", "episode": "episode_0001", "candidate_index": 4},
            {"scene": "a", "episode": "episode_0000", "candidate_index": 0},
            {"scene": "c", "episode": "episode_0003", "candidate_index": 7},
            {"scene": "b", "episode": "episode_0000", "candidate_index": 3},
        ]
        selected = round_robin_select(rows, ["b", "a", "c"], 5)
        self.assertEqual(
            [(row["scene"], row["episode"]) for row in selected],
            [
                ("b", "episode_0000"),
                ("a", "episode_0000"),
                ("c", "episode_0003"),
                ("b", "episode_0001"),
                ("a", "episode_0002"),
            ],
        )

    def test_only_declared_constructibility_failures_are_caught(self):
        self.assertTrue(
            is_expected_construction_failure(
                RuntimeError("online-A trace has too few source-anchor candidates")
            )
        )
        self.assertFalse(is_expected_construction_failure(RuntimeError("hash mismatch")))
        self.assertFalse(is_expected_construction_failure(ValueError("bad contract")))

    def test_failed_job_status_parser_requires_complete_population(self):
        rows = [
            {
                "candidate": index,
                "constructible": index == 1,
                "episode": f"episode_{index:04d}",
                "reason": None if index == 1 else "expected failure",
                "scene": f"scene{index}",
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prep.out"
            path.write_text(
                "unrelated output\n"
                + "\n".join(json.dumps(row, sort_keys=True) for row in rows)
                + "\n"
            )
            parsed = parse_statuses(path, 3)
            self.assertEqual([row["candidate"] for row in parsed], [0, 1, 2])
            with self.assertRaisesRegex(RuntimeError, "complete candidate population"):
                parse_statuses(path, 4)


if __name__ == "__main__":
    unittest.main()
