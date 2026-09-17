"""Synthetic accounting fixtures, not navigation results."""
import json
from pathlib import Path
import tempfile
import unittest

from construct_repaired_fullmono_queries import save
from probe_repaired_fullmono_construction import sha
from seal_repaired_fullmono_queries import SCHEMA, seal
from summarize_repaired_fullmono_queries import summarize
import test_repaired_fullmono_query_seal as seal_tests


class QuerySummaryTests(unittest.TestCase):
    def fixture(self, root):
        args = seal_tests.QuerySealTests().fixture(root)
        seal(args)
        path = args.out / "population.json"
        population = json.loads(path.read_text())
        evaluation = root / "evaluation"
        evaluation.mkdir()
        for task in population["tasks"]:
            folder = evaluation / f"task_{task['task_index']:03d}"
            folder.mkdir()
            role = "revisit" if task["query_id"] == "revisit" else "novel"
            successes = {"native": int(role == "novel"), "cec": int(role == "novel" or task["history_index"] == 0),
                         "raw_fixed": int(task["history_index"] == 0)}
            records = [dict(arm=a, reached=successes[a], spl=.5 * successes[a], actual_path_m=2.,
                            steps=10, total_turn_deg=20., wall_seconds=3.) for a in task["arm_order"]]
            verified = dict(schema=SCHEMA, verified=True, task=task, population_sha256=sha(path),
                records=records, cec_takeover_plans=int(role == "revisit"),
                no_takeover_exact_native=role == "novel")
            manifest = dict(schema=SCHEMA, task=task, population_sha256=sha(path), analysis_role=role,
                scene="s", runtime_source_receipt=population["runtime_source_receipt"],
                evaluation_source_receipt=population["evaluation_source_receipt"])
            raw = folder / "synthetic_archive_fixture.bin"
            raw.write_bytes(b"synthetic unit-test archive; not a rollout")
            save(folder / "manifest.json", manifest)
            save(folder / "independent_verification.json", verified)
            save(folder / "archive_receipt.json", dict(completed=True, exit_code=0,
                verification=verified, all_member_hashes_readback_verified=True,
                archive=str(raw), archive_sha256=sha(raw)))
        return path, evaluation

    def test_role_and_source_denominators_remain_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            path, evaluation = self.fixture(Path(d))
            result = summarize(path, sha(path), evaluation)
            self.assertTrue(result["complete"])
            self.assertEqual(result["source_accounting"]["completed_A"], 4)
            self.assertEqual(result["source_accounting"]["successful_A"], 3)
            self.assertEqual(result["source_accounting"]["constructible_role_pairs"], 2)
            self.assertEqual(result["roles"]["revisit"]["queries"], 2)
            self.assertEqual(result["roles"]["revisit"]["arms"]["cec"]["sr"], .5)
            self.assertEqual(result["roles"]["revisit"]["comparisons"]["native"]["gain"], 1)
            self.assertEqual(result["roles"]["novel"]["exact_native_queries"], 2)
            self.assertEqual(result["balanced_role_mixture"]["arms"]["cec"]["successes"], 3)

    def test_missing_task_is_not_zero_success(self):
        with tempfile.TemporaryDirectory() as d:
            path, evaluation = self.fixture(Path(d))
            (evaluation / "task_000/archive_receipt.json").unlink()
            result = summarize(path, sha(path), evaluation)
            self.assertFalse(result["complete"])
            self.assertEqual(result["verified_queries"], 3)
            self.assertNotIn("roles", result)
            self.assertNotIn("balanced_role_mixture", result)

    def test_incomplete_arm_receipt_is_not_final(self):
        with tempfile.TemporaryDirectory() as d:
            path, evaluation = self.fixture(Path(d))
            receipt = evaluation / "task_000/archive_receipt.json"
            data = json.loads(receipt.read_text())
            data["completed"] = False
            receipt.write_text(json.dumps(data))
            self.assertFalse(summarize(path, sha(path), evaluation)["complete"])

    def test_old_population_is_not_mixed_in(self):
        with tempfile.TemporaryDirectory() as d:
            path, evaluation = self.fixture(Path(d))
            receipt = evaluation / "task_000/manifest.json"
            data = json.loads(receipt.read_text())
            data["population_sha256"] = "0" * 64
            receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Task/population pairing changed"):
                summarize(path, sha(path), evaluation)

    def test_changed_archive_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path, evaluation = self.fixture(Path(d))
            (evaluation / "task_000/synthetic_archive_fixture.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Raw archive changed"):
                summarize(path, sha(path), evaluation)


if __name__ == "__main__":
    unittest.main()
