import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from collect_repaired_fullmono_a import SCHEMA as A_SCHEMA
from construct_repaired_fullmono_queries import check_a_sources, selected_query_ids, unpack_traces
from covisibility_task_archive import archive
from repaired_covisibility_eval import dump


class FullMonoConstructionTests(unittest.TestCase):
    def fixture(self):
        sources = [dict(scene="s", episode=f"ep{i}", seed=2026082200+i,
                        source_index=i, scene_rank=0, episode_rank=i) for i in range(4)]
        plan = dict(new_a_required=True, old_a_results_for_selection=False,
                    status="prepared_not_submitted_not_query_sealed", sources=sources)
        manifest = dict(schema=A_SCHEMA, queries_allowed=False, source_scene_rank=0, sources=copy.deepcopy(sources))
        summary = dict(completed=True, queries=[], goal_a=[dict(s, reached=int(i%2==0)) for i,s in enumerate(sources)])
        verification = dict(verified=True, goal_a=[dict(source_index=i, reached=int(i%2==0)) for i in range(4)])
        return plan, manifest, summary, verification

    def test_failed_as_are_kept_in_complete_source_validation(self):
        self.assertEqual(len(check_a_sources(*self.fixture())), 4)

    def test_missing_or_reordered_source_is_not_scientific_attrition(self):
        for partial in (False, True):
            args = list(self.fixture())
            if partial:
                args[2]["goal_a"].pop()
            else:
                args[1]["sources"].reverse()
            with self.assertRaises(ValueError):
                check_a_sources(*args)

    def test_unselected_direction_targets_are_not_extra_formal_queries(self):
        q=[dict(query_id="revisit", analysis_role="revisit"),
           dict(query_id="novel_front", analysis_role="novel"),
           dict(query_id="novel_rear", analysis_role="novel")]
        n={s: dict(constructible=True, query_id="novel_"+s) for s in ("front","rear")}
        self.assertEqual(selected_query_ids(q,n,"rear"), ["novel_rear","revisit"])
        self.assertEqual(selected_query_ids(q,n,None), [])

    def test_a_role_pair_cannot_be_formed_without_both_queries(self):
        n={"front": dict(constructible=True,query_id="novel_front")}
        for q in ([dict(query_id="novel_front",analysis_role="novel")],
                  [dict(query_id="revisit",analysis_role="revisit")]):
            with self.assertRaises(ValueError):
                selected_query_ids(q,n,"front")

    def test_readback_archive_preserves_every_a_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, durable, restored = root / "task", root / "durable", root / "restored"
            task.mkdir()
            durable.mkdir()
            restored.mkdir()
            plan, manifest, summary, verification = self.fixture()
            plan["base_runtime_sha256"] = "a"*64
            manifest["runtime_source_receipt"] = {"sha256": "a"*64}
            raw = []
            for source, v in zip(manifest["sources"], verification["goal_a"]):
                path = task / "goal_a" / source["scene"] / source["episode"] / f"{source['episode']}_leg1_trace.json"
                path.parent.mkdir(parents=True)
                data = json.dumps({"episode": source["episode"], "reached": v["reached"],
                                   "end_position": [source["source_index"], .1, 3.]}).encode()
                path.write_bytes(data)
                raw.append((path.relative_to(task), data))
                v["trace_sha256"] = hashlib.sha256(data).hexdigest()
            for name, value in (("manifest.json", manifest), ("summary.json", summary),
                                ("independent_verification.json", verification)):
                dump(task / name, value)
                dump(durable / name, value)
            archive(task, durable, 0)
            selected = unpack_traces(durable, restored, plan)
            self.assertEqual(len(selected), 4)
            for relative, data in raw:
                self.assertEqual((restored / relative).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
