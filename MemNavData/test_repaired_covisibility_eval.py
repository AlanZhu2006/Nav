import copy
import itertools
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import repaired_covisibility_eval as r
from covisibility_task_archive import archive
from summarize_covisibility_repaired import exact_mcnemar, group_stats, holm


class CovisibilityEvaluationTests(unittest.TestCase):
    def population(self):
        return {"schema": r.SCHEMA, "phase": "sealed_before_query_evaluation", "arms": list(r.ARMS),
            "query_count": 6, "query_arm_count": 18, "tasks": [
                {"task_index": i, "arm_order": list(order)} for i, order in enumerate(itertools.permutations(r.ARMS))]}

    def test_frozen_six_arm_orders(self):
        p = self.population()
        for i in range(6):
            self.assertEqual(r.select_task(p, i), p["tasks"][i])
        with self.assertRaises(ValueError):
            r.select_task(p, 6)

    def test_incomplete_or_unsealed_manifest_rejected(self):
        for key, value in (("query_count", 5), ("query_arm_count", 17), ("phase", "construction")):
            p = self.population()
            p[key] = value
            with self.assertRaises(ValueError):
                r.select_task(p, 0)

    def test_shuffled_arms_rejected(self):
        p = self.population()
        p["tasks"][1]["arm_order"] = list(r.ARMS)
        with self.assertRaises(ValueError):
            r.select_task(p, 1)

    def test_command_changes_only_callback_entry(self):
        from MemNavData.run_repaired_fullmono_local import evaluator_command as original
        source = {"scene": "s", "episode": "episode_0000", "asset": "/unused.glb",
                  "source_episode": "/unused/episode_0000", "seed": 41}
        for role in ("novel", "revisit"):
            for arm in r.ARMS:
                old = original(source, Path("/unused/out"), 21750, 21751, arm=arm, role=role,
                               benchmark=Path("/unused/sealed_query"))
                new = r.evaluator_command(source, Path("/unused/out"), 21750, 21751, arm=arm, role=role)
                self.assertEqual(old[:2], new[:2])
                self.assertEqual(old[4:], new[4:])
                self.assertEqual(new[3], "eval")

    def test_evaluator_annotation_stripping(self):
        from shared_online_role_pair_contract import runtime_query
        q = dict(query_id="c10_30", analysis_role="revisit", goal_rgb="goal.jpg", goal_depth="annotation.npy",
            goal_rgb_sha256="a"*64, goal_depth_sha256="b"*64, floor_position=[0., 0., 1.], yaw_rad=.3,
            geodesic_from_a_end_m=2., initial_path_bearing_rad=.1, max_online_a_covis=.2,
            q_eligible=.2, bin="c10_30", covis_curve=[.1, .2])
        visible = runtime_query(q)
        self.assertFalse({"analysis_role", "q_eligible", "bin", "covis_curve", "max_online_a_covis"} & set(visible))

    def test_archive_preserves_absolute_receipts_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, durable = Path(tmp)/"node"/"task", Path(tmp)/"durable"
            root.mkdir(parents=True)
            durable.mkdir()
            r.dump(root/"summary.json", {"completed": True, "end_position": [1., .2, 3.], "spl": .7})
            r.dump(root/"independent_verification.json", {"verified": True})
            r.dump(root/"absolute_receipt.json", {"artifact": str(root/"depth.bin")})
            (root/"depth.bin").write_bytes(b"depth-evidence")
            before = r.sha(root/"absolute_receipt.json")
            archive(root, durable, 0)
            receipt = r.load(durable/"archive_receipt.json")
            self.assertTrue(receipt["completed"])
            self.assertEqual(receipt["archive_sha256"], r.sha(durable/"artifacts.tar.gz"))
            with tarfile.open(durable/"artifacts.tar.gz", "r:gz") as tar:
                stored = json.load(tar.extractfile("task/absolute_receipt.json"))
                self.assertEqual(stored["artifact"], str(root/"depth.bin"))
                self.assertEqual(json.load(tar.extractfile("task/summary.json"))["end_position"], [1., .2, 3.])
            self.assertEqual(before, r.sha(root/"absolute_receipt.json"))
            with self.assertRaises(ValueError):
                archive(root, durable, 0)

    def test_failed_task_never_becomes_valid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, durable = Path(tmp)/"task", Path(tmp)/"durable"
            root.mkdir()
            durable.mkdir()
            r.dump(root/"summary.json", {"completed": False, "queries": []})
            archive(root, durable, 124)
            self.assertFalse(r.load(durable/"archive_receipt.json")["completed"])
            self.assertFalse((durable/"summary.json").exists())

    def test_statistics_and_family_correction(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.)
        self.assertAlmostEqual(exact_mcnemar(12, 0), .00048828125)
        self.assertEqual(holm([.04, .001, .2]), [.08, .003, .2])
        row = dict(reached=0, spl=0., actual_path_m=2., steps=60, total_turn_deg=20., wall_seconds=5.)
        tasks = []
        for i in range(6):
            arms = {a: copy.deepcopy(row) for a in r.ARMS}
            arms["cec"].update(reached=1, spl=.8)
            tasks.append({"scene": f"scene_{i//2}", "arms": arms})
        stats = group_stats(tasks)
        self.assertEqual(stats["scenes"], 3)
        self.assertEqual(stats["comparisons"]["native"]["gain"], 6)
        self.assertEqual(stats["comparisons"]["native"]["scene_cluster_bootstrap_ci95"], [1., 1.])
        self.assertAlmostEqual(stats["arms"]["cec"]["spl"], .8)


if __name__ == "__main__":
    unittest.main()
