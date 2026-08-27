import csv
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.summarize_graph_conditional_c_eval import (
    summarize_graph_conditional,
    validate_configuration,
)


class GraphConditionalSummaryTest(unittest.TestCase):
    @staticmethod
    def write_arm(root, arm, mode, *, spacing, gap="16", success=True):
        output = root / "scenes/00_scene" / arm
        output.mkdir(parents=True, exist_ok=True)
        record = {
            "episode": "episode_0000",
            "seed": 7,
            "mode": mode,
            "reached_C": int(success),
            "spl_C": 0.5 if success else 0.0,
            "geo_C": 3.0,
            "len_C": 4.0,
            "steps_C": 80,
            "final_dist_C": 0.5 if success else 2.0,
            "prefix_last_source_frame": 320,
            "prefix_source_frames": 321,
            "memory_prefix_frames": 321 if mode != "native" else 0,
            "navdp_prefix_decision_frames": 41,
            "c_recall_gap": 280,
            "c_gt_covis_anchor": 40,
            "router_active_episode": int(mode not in ("native", "oracle_point")),
            "retrieval_candidate_min_gap": gap if mode != "native" else "",
            "graph_subgoal_spacing_m": spacing,
            "graph_subgoal_arrival_m": 0.60 if mode != "native" else "",
            "deterministic_plan_seeds": 1,
        }
        with (output / "metric.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        (output / "episode_0000_plans.json").write_text(json.dumps({
            "protocol": "conditional_C_after_causal_source_AB_replay",
            "mode": mode,
            "legC": [{
                "requested_diffusion_seed": 11,
                "diffusion_seed": 11,
            }],
        }))

    def test_configuration_requires_frozen_graph_parameters(self):
        graph = [{"candidate_gap": 16, "graph_spacing_m": 1.25}]
        validate_configuration(
            "graph", graph, expected_gap=16, expected_spacing_m=1.25)
        with self.assertRaisesRegex(RuntimeError, "graph spacing mismatch"):
            validate_configuration(
                "graph", [{"candidate_gap": 16, "graph_spacing_m": 0.0}],
                expected_gap=16, expected_spacing_m=1.25)

    def test_end_to_end_summary_reads_direct_and_graph_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct"
            graph = root / "graph"
            for arm, mode in (
                ("navdp_native", "native"),
                ("geometry_router", "geometry_topk"),
                ("oracle_anchor", "oracle_anchor"),
                ("oracle_point", "oracle_point"),
            ):
                self.write_arm(direct, arm, mode, spacing=0.0)
            for arm, mode in (
                ("geometry_router", "geometry_topk"),
                ("oracle_anchor", "oracle_anchor"),
            ):
                self.write_arm(graph, arm, mode, spacing=1.25)
            manifest = {
                "selection": {"selected_scenes": ["scene"]},
                "episodes": {"scene": [{"episode": "episode_0000"}]},
            }
            report = summarize_graph_conditional(manifest, direct, graph)
            self.assertEqual(report["audit"]["status"], "ok")
            self.assertEqual(report["arms"]["graph_gap16"]["successes"], 1)
        with self.assertRaisesRegex(RuntimeError, "candidate gap mismatch"):
            validate_configuration(
                "graph", [{"candidate_gap": 4, "graph_spacing_m": 1.25}],
                expected_gap=16, expected_spacing_m=1.25)


if __name__ == "__main__":
    unittest.main()
