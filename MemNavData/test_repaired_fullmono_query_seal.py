import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from construct_repaired_fullmono_queries import SCHEMA as CONSTRUCTION_SCHEMA, save
import repaired_covisibility_eval as runtime
import repaired_fullmono_query_eval as query_adapter
from seal_repaired_fullmono_queries import ARMS, paired_tasks, read_scene, seal
from probe_repaired_fullmono_construction import sha


class QuerySealTests(unittest.TestCase):
    def fixture(self, root):
        receipt = root / "runtime.sha256"
        receipt.write_text("frozen runtime\n")
        evaluation = root / "SOURCE_BUNDLE.sha256"
        evaluation.write_text("frozen evaluation adapter\n")
        scene, collection = root / "constructed_scene", root / "collection"
        scene.mkdir()
        collection.mkdir()
        sources = [dict(source_index=i, scene="s", episode=f"ep{i}", seed=100+i, scene_rank=0)
                   for i in range(4)]
        plan = dict(sources=sources, stages=[dict(scene_prefix=30)], target_histories=1,
                    target_scene_clusters=1, base_runtime_sha256=sha(receipt))
        plan_path = root / "plan.json"
        save(plan_path, plan)
        manifest = dict(sources=sources, source_plan_sha256=sha(plan_path),
                        runtime_source_receipt=dict(path=str(receipt), sha256=sha(receipt)))
        a_rows = [dict(s, reached=i != 1) for i, s in enumerate(sources)]
        a_verifier = dict(verified=True, goal_a=[])
        reports = []
        for i, source in enumerate(sources):
            row = dict(source_index=i, scene="s", episode=f"ep{i}", construction_complete=True,
                       pair_constructible=i in (0, 3), query_rollouts=0)
            if row["pair_constructible"]:
                folder = scene / f"ep{i}"
                online = folder / "online"
                online.mkdir(parents=True)
                save(online / "receipt.json", {})
                save(online / "online_a_trace.json", dict(poses=[dict(step=0)], reached=True,
                                                           episode_seed=source["seed"]))
                queries = []
                for name, role in (("novel_front", "novel"), ("novel_rear", "novel"), ("revisit", "revisit")):
                    for suffix in ("jpg", "npy"):
                        (folder / f"{name}.{suffix}").write_bytes((name+suffix).encode())
                    queries.append(dict(query_id=name, analysis_role=role,
                        goal_rgb=name+".jpg", goal_depth=name+".npy",
                        goal_rgb_sha256=sha(folder / (name+".jpg")),
                        goal_depth_sha256=sha(folder / (name+".npy"))))
                payload = dict(schema=CONSTRUCTION_SCHEMA, completed=True, pair_constructible=True,
                    navigation_rollouts=0, query_outcomes_read=False, history_index=i, source_index=i,
                    scene="s", episode=f"ep{i}", source=source, queries=queries,
                    selected_query_ids=["novel_rear", "revisit"], online_a_episode=str(online),
                    online_a_steps=1, online_a_receipt_sha256=sha(online / "receipt.json"),
                    online_a_trace_sha256=sha(online / "online_a_trace.json"),
                    all_historical_rgb_rerender_hashes_match=True)
                path = folder / "construction.json"
                save(path, payload)
                row.update(construction=str(path), construction_sha256=sha(path))
                a_verifier["goal_a"].append(dict(source_index=i, reached=True,
                                                 trace_sha256=payload["online_a_trace_sha256"]))
            reports.append(row)
        save(collection / "manifest.json", manifest)
        save(collection / "summary.json", dict(completed=True, queries=[], goal_a=a_rows))
        save(collection / "independent_verification.json", a_verifier)
        save(scene / "a_input_receipt.json", dict(collection=str(collection), manifest=manifest,
            manifest_sha256=sha(collection / "manifest.json"),
            verification_sha256=sha(collection / "independent_verification.json")))
        summary = dict(schema=CONSTRUCTION_SCHEMA, completed=True, query_rollouts=0,
            source_plan_sha256=sha(plan_path), source_scene_rank=0, reports=reports, source_count=4)
        save(scene / "summary.json", summary)
        save(scene / "independent_verification.json", dict(schema=CONSTRUCTION_SCHEMA, verified=True,
            query_rollouts=0, source_plan_sha256=sha(plan_path), summary_sha256=sha(scene / "summary.json"),
            reports=[dict(r, verified=True) for r in reports]))
        return argparse.Namespace(plan=plan_path, plan_sha256=sha(plan_path), scenes=[scene],
            runtime_receipt=receipt, evaluation_receipt=evaluation, out=root / "sealed")

    def test_seal_keeps_all_legal_pairs_not_only_minimum_count(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            seal(args)
            population = json.loads((args.out / "population.json").read_text())
            self.assertEqual(population["histories"], 2)  # target was one, both retained
            self.assertEqual(population["query_count"], 4)
            self.assertEqual(population["query_arm_count"], 12)
            self.assertEqual(population["source_waterfall"], dict(planned_A=4, completed_A=4,
                successful_A=3, constructible_role_pairs=2, pair_scene_clusters=1))
            self.assertEqual({t["query_id"] for t in population["tasks"]}, {"novel_rear", "revisit"})

    def test_schedule_is_stable_and_each_task_has_all_three_arms(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            reports, _, binding = read_scene(args.scenes[0], args.plan_sha256)
            a = paired_tasks([0, 3], reports, [binding])
            b = paired_tasks([3, 0], list(reversed(reports)), [binding])
            self.assertEqual(a, b)
            self.assertTrue(all(set(t["arm_order"]) == set(ARMS) for t in a))
            self.assertEqual(len({tuple(t["arm_order"]) for t in a}), 4)

    def test_incomplete_prefix_never_produces_population(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            plan = json.loads(args.plan.read_text())
            plan["sources"].append(dict(source_index=4, scene="s2", episode="ep0", scene_rank=1))
            # Keep receipt hashes consistent; the missing source itself is the test.
            with patch("seal_repaired_fullmono_queries.read", side_effect=lambda p:
                       plan if Path(p) == args.plan else json.loads(Path(p).read_text())):
                result = seal(args)
            self.assertEqual(result["state"], "await_complete_prefix")
            self.assertEqual(result["missing_source_indices"], [4])
            self.assertFalse((args.out / "population.json").exists())

    def test_changed_goal_is_not_silently_sealed(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            (args.scenes[0] / "ep0" / "revisit.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Selected goal changed"):
                seal(args)

    def test_actual_a_query_loader_reuses_rollout_and_checks_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            seal(args)
            population = args.out / "population.json"
            original_run, original_verify = runtime.run, runtime.verify
            # Restore the imported covisibility module after this binding test.
            with patch.multiple(runtime, SCHEMA=runtime.SCHEMA, SCOPE=runtime.SCOPE,
                                POPULATION_SHA=runtime.POPULATION_SHA, HERE=runtime.HERE,
                                load_task=runtime.load_task), \
                 patch.object(query_adapter, "__file__", str(Path(d) / "repaired_fullmono_query_eval.py")), \
                 patch.dict("os.environ", {"REPAIRED_SOURCE_RECEIPT": str(args.runtime_receipt)}):
                query_adapter.bind_runtime(sha(population))
                for i in range(4):
                    task, payload, query, _ = runtime.load_task(population, i)
                    self.assertIn(task["query_id"], payload["selected_query_ids"])
                self.assertIs(runtime.run, original_run)
                self.assertIs(runtime.verify, original_verify)


if __name__ == "__main__":
    unittest.main()
