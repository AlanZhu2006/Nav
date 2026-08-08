import copy
import json
import unittest
from pathlib import Path

from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    paired_summary,
    percentile,
    truth,
)
from MemNavData.validate_expanded_navdp_router_eval import (
    resolve_dependency_paths,
    validate_selection,
)
from MemNavData.validate_frozen_router_blind import compare_value


ROOT = Path(__file__).resolve().parent


class ExpandedBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "expanded_navdp_router_eval_20260805.json").read_text()
        )

    def test_frozen_selection_is_scene_disjoint(self):
        selected = validate_selection(self.manifest)
        self.assertEqual(len(selected), 20)
        self.assertFalse(set(selected) & set(self.manifest["training_scenes"]))

    def test_training_scene_leak_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        leaked = broken["training_scenes"][0]
        broken["selection"]["selected_scenes"][0] = leaked
        with self.assertRaisesRegex(RuntimeError, "selected scene outside eligible pool"):
            validate_selection(broken)

    def test_summary_separates_novel_revisit_and_joint(self):
        base = {
            "spl_a": 0.5,
            "spl_b": 0.4,
            "final_dist_a": 0.5,
            "final_dist_b": 0.7,
            "router_active_episode_a": False,
            "router_active_episode_b": True,
            "geometry_verification_ms": [10.0],
        }
        rows = [
            dict(base, reached_a=True, reached_b=True, joint=True),
            dict(base, reached_a=True, reached_b=False, joint=False),
            dict(base, reached_a=False, reached_b=False, joint=False,
                 router_active_episode_b=False),
        ]
        summary = arm_summary(rows)
        self.assertEqual(summary["novel"]["successes"], 2)
        self.assertEqual(summary["revisit_given_novel_success"]["eligible"], 2)
        self.assertEqual(summary["revisit_given_novel_success"]["successes"], 1)
        self.assertEqual(summary["joint"]["successes"], 1)

    def test_json_and_csv_truth_values(self):
        self.assertTrue(truth(True))
        self.assertTrue(truth("True"))
        self.assertTrue(truth("1.0"))
        self.assertFalse(truth(False))
        self.assertFalse(truth("False"))
        self.assertFalse(truth("0.0"))
        self.assertFalse(truth(None))

    def test_three_arm_pairing_attributes_topk_gain(self):
        def row(joint, *, seed=7, rank=None):
            return {
                "seed": seed,
                "recall_gap": 320,
                "geo_a": 2.0,
                "geo_b": 3.0,
                "joint": joint,
                "reached_a": True,
                "reached_b": joint,
                "router_active_episode_b": joint,
                "selected_candidate_ranks": ([] if rank is None else [rank]),
            }

        key_a = ("scene", "episode_0000")
        key_b = ("scene", "episode_0001")
        top1 = {key_a: row(False), key_b: row(True, rank=1)}
        topk = {key_a: row(True, rank=3), key_b: row(True, rank=1)}
        summary = paired_summary(
            "geometry_top1", "geometry_router", top1, topk, {key_a, key_b}
        )
        self.assertEqual(summary["outcomes"]["right_only_joint_success"], 1)
        self.assertEqual(summary["outcomes"]["left_only_joint_success"], 0)
        self.assertEqual(summary["joint_sr_delta_right_minus_left"], 0.5)

    def test_percentile_uses_linear_interpolation(self):
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([4.0], 0.95), 4.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_frozen_router_numeric_comparison_is_strict(self):
        compare_value("weights", [1.0, 2.0], [1.0, 2.0 + 1e-13])
        with self.assertRaisesRegex(RuntimeError, "frozen numeric field changed"):
            compare_value("weights", [1.0, 2.0], [1.0, 2.0 + 1e-6])

    def test_local_dependency_overrides_preserve_frozen_hash_records(self):
        local_gate = Path("/scratch/local/gatecurr600.memnav.ckpt")
        resolved = resolve_dependency_paths(
            self.manifest,
            {
                "gatecurr600": local_gate,
                "navdp_checkpoint": None,
                "lingbot_map_long": None,
            },
        )
        self.assertEqual(resolved["gatecurr600"], local_gate)
        self.assertEqual(
            resolved["navdp_checkpoint"],
            Path(self.manifest["dependencies"]["navdp_checkpoint"]["path"]),
        )
        self.assertEqual(set(resolved), set(self.manifest["dependencies"]))

    def test_scene_runner_terminal_alignment_is_explicit_and_default_off(self):
        runner = (ROOT / "run_expanded_navdp_router_scene.sh").read_text()
        self.assertIn("TERMINAL_UTURN=${TERMINAL_UTURN:-off}", runner)
        self.assertIn(
            "TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE:-off}",
            runner,
        )
        self.assertIn('--terminal_uturn "${TERMINAL_UTURN}"', runner)
        self.assertIn(
            '--terminal_visual_refine "${TERMINAL_VISUAL_REFINE}"',
            runner,
        )
        self.assertNotIn("--terminal_uturn off", runner)

    def test_scene_runner_shadow_arrival_is_explicit_and_default_off(self):
        runner = (ROOT / "run_expanded_navdp_router_scene.sh").read_text()
        self.assertIn("ARRIVAL_SHADOW=${ARRIVAL_SHADOW:-off}", runner)
        self.assertIn('--arrival_shadow "${ARRIVAL_SHADOW}"', runner)
        evaluator = (ROOT / "eval_2leg_habitat.py").read_text()
        self.assertIn(
            'args.arrival_shadow == "diagnostic" and leg_index == 1',
            evaluator,
        )

    def test_phase_b_p0_arm_is_ranking_only_and_explicitly_opt_in(self):
        runner = (ROOT / "run_expanded_navdp_router_scene.sh").read_text()
        evaluator = (ROOT / "eval_2leg_habitat.py").read_text()
        server = (
            ROOT.parent / "NavDP/baselines/memnav/memnav_server.py"
        ).read_text()
        self.assertIn(
            "RUN_LEARNED_RANK_GEOMETRY=${RUN_LEARNED_RANK_GEOMETRY:-0}",
            runner,
        )
        self.assertIn("EXPECTED_PHASE_B_CKPT_SHA", runner)
        self.assertIn("--phase_b_allow_unapproved", runner)
        self.assertIn(
            "run_geometry_arm learned_rank_geometry 8 "
            "learned_rank_geometry",
            runner,
        )
        self.assertIn('f"{BASE}/phase_b_rank"', evaluator)
        self.assertIn(
            'rank_out.get("activation_uses_model_score") is not False',
            evaluator,
        )
        self.assertIn('@app.route("/phase_b_rank"', server)
        self.assertIn(
            '"activation_semantics": '
            '"diagnostic_only_geometry_gate_unchanged"',
            (ROOT / "phase_b_runtime.py").read_text(),
        )

    def test_phase_b_p0_hpc_mode_freezes_a_within_each_array_task(self):
        runner = (ROOT / "run_expanded_navdp_router_scene.sh").read_text()
        sbatch = (ROOT / "slurm_expanded_navdp_router_eval.sbatch").read_text()
        self.assertIn("P0_SHARED_PREFIX=${P0_SHARED_PREFIX:-0}", runner)
        self.assertIn(
            "P0_SHARED_PREFIX requires exactly native/geometry/learned arms",
            runner,
        )
        protocol = runner.index("Generate Goal A exactly once on this node")
        geometry = runner.index(
            "run_geometry_arm geometry_router 8 memory_geometry", protocol
        )
        learned = runner.index(
            "run_geometry_arm learned_rank_geometry 8 learned_rank_geometry",
            geometry,
        )
        native = runner.index('run_native_arm "${P0_SHARED_ARGS[@]}"', learned)
        self.assertLess(geometry, learned)
        self.assertLess(learned, native)
        self.assertIn(
            '--shared_leg1_trace_root "${SCENE_ROOT}/geometry_router"',
            runner,
        )
        self.assertIn('P0_SHARED_PREFIX="${P0_SHARED_PREFIX}"', sbatch)

    def test_global_subgoal_oracle_is_explicit_and_fail_closed(self):
        runner = (ROOT / "run_expanded_navdp_router_scene.sh").read_text()
        self.assertIn(
            "ORACLE_GLOBAL_SUBGOAL_M=${ORACLE_GLOBAL_SUBGOAL_M:-0.0}",
            runner,
        )
        self.assertIn(
            '--oracle_global_subgoal_m "${ORACLE_GLOBAL_SUBGOAL_M}"',
            runner,
        )
        self.assertIn(
            "oracle global subgoals require the native-only arm",
            runner,
        )
        evaluator = (ROOT / "eval_3leg_habitat.py").read_text()
        self.assertIn(
            '"pose_controller": "oracle_habitat_geodesic_subgoal"',
            evaluator,
        )
        self.assertIn(
            'args.trajectory_selector == "server"', evaluator)


if __name__ == "__main__":
    unittest.main()
