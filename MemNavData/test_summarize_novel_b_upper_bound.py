import ast
import copy
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from MemNavData.summarize_novel_b_upper_bound import (
    ARMS,
    EXPECTED_MANIFEST_SHA256,
    PROTOCOL,
    canonical_sha256,
    load_arm,
    load_frozen_manifest,
    summarize_rows,
)


ROOT = Path(__file__).resolve().parent


def goal_a_record(reached: bool, marker: float) -> dict:
    return {
        "reached": reached,
        "path_len": marker,
        "path_len_at_reach": marker if reached else None,
        "step_at_reach": 2 if reached else None,
        "steps": 2,
        "plans": [{
            "step": 0,
            "requested_diffusion_seed": 20260803 * 100_000,
            "diffusion_seed": 20260803 * 100_000,
            "nested": {"marker": marker},
        }],
        "memory_trace": [],
        "rollout_trace": [{"step": 0, "x": marker}],
        "end_pos": [marker, 0.0, 0.0],
        "end_psi": 0.0,
        "final_goal_dist_m": 0.5 if reached else 2.0,
    }


def paired_row(
    key: tuple[str, str],
    *,
    goal_a: dict,
    reached_b: bool,
) -> dict:
    return {
        "scene": key[0],
        "episode": key[1],
        "seed": 20260803,
        "goal_a": copy.deepcopy(goal_a),
        "goal_a_sha256": canonical_sha256(goal_a),
        "reached_a": bool(goal_a["reached"]),
        "reached_b": reached_b,
        "attempted_b": bool(goal_a["reached"]),
        "spl_a": 0.5 if goal_a["reached"] else 0.0,
        "spl_b": 0.6 if reached_b else 0.0,
        "geo_a": 2.0,
        "geo_b": 3.0,
        "path_a": float(goal_a["path_len"]),
        "path_b": 4.0,
        "final_dist_a": float(goal_a["final_goal_dist_m"]),
        "final_dist_b": 0.5 if reached_b else 2.0,
        "steps_a": int(goal_a["steps"]),
        "steps_b": 3,
    }


class NovelBUpperBoundSummaryTest(unittest.TestCase):
    def test_slurm_defaults_to_one_bounded_h100_task(self):
        slurm = (ROOT / "slurm_novel_b_upper_bound.sbatch").read_text()
        runner = (ROOT / "run_novel_b_upper_bound.sh").read_text()
        self.assertIn("#SBATCH --time=24:00:00", slurm)
        self.assertIn("#SBATCH --partition=h100_tandon", slurm)
        self.assertIn("#SBATCH --gres=gpu:h100:1", slurm)
        self.assertNotIn("#SBATCH --array", slurm)
        self.assertIn('ARMS=(native_imagegoal oracle_short_1p25m '
                      'oracle_final_point)', runner)
        self.assertIn('"server_start_count": 1', runner)
        self.assertIn("goal_a_pairing.json", runner)
        self.assertIn("HTTP_TIMEOUT_S", (
            ROOT / "eval_novel_b_habitat.py").read_text())
        self.assertIn("EXPECTED_BASE_SIF_HEAD_SHA256", slurm)
        self.assertIn(EXPECTED_MANIFEST_SHA256, runner)

    def test_evaluator_api_is_compatible_with_clean_head_base(self):
        """Do not accidentally depend on fields present only in dirty files."""
        clean_base = subprocess.run(
            ["git", "show", "HEAD:MemNavData/eval_2leg_habitat.py"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        base_tree = ast.parse(clean_base)
        base_arguments = set()
        base_names = set()
        for node in ast.walk(base_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                for argument in node.args:
                    if (
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                        and argument.value.startswith("--")
                    ):
                        base_arguments.add(
                            argument.value[2:].replace("-", "_")
                        )
        for node in base_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                base_names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                base_names.update(
                    alias.asname or alias.name.split(".")[-1]
                    for alias in node.names
                )
            elif isinstance(node, ast.Assign):
                base_names.update(
                    target.id for target in node.targets
                    if isinstance(target, ast.Name)
                )
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                base_names.add(node.target.id)

        evaluator_source = (ROOT / "eval_novel_b_habitat.py").read_text()
        evaluator_tree = ast.parse(evaluator_source)
        direct_args = {
            node.attr
            for node in ast.walk(evaluator_tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        }
        base_references = {
            node.attr
            for node in ast.walk(evaluator_tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "base"
        }
        self.assertEqual(direct_args - base_arguments, set())
        self.assertEqual(base_references - base_names, set())
        # The one clean-base optional lookup must be explicit and default-off.
        self.assertRegex(
            evaluator_source,
            re.compile(
                r'getattr\(args,\s*"oracle_observed_frontier",\s*"off"\)'
            ),
        )

    def test_manifest_identity_is_frozen(self):
        manifest, digest = load_frozen_manifest(
            ROOT / "expanded_3leg_router_eval_20260805.json"
        )
        self.assertEqual(digest, EXPECTED_MANIFEST_SHA256)
        self.assertEqual(len(manifest["selection"]["selected_scenes"]), 10)

    def test_goal_b_denominator_is_common_goal_a_success_only(self):
        eligible_key = ("scene_a", "episode_0000")
        failed_a_key = ("scene_b", "episode_0000")
        expected = {eligible_key, failed_a_key}
        success_a = goal_a_record(True, 1.5)
        failed_a = goal_a_record(False, 2.5)
        outcomes = {
            "native_imagegoal": False,
            "oracle_short_1p25m": True,
            "oracle_final_point": True,
        }
        rows = {
            arm: {
                eligible_key: paired_row(
                    eligible_key,
                    goal_a=success_a,
                    reached_b=outcomes[arm],
                ),
                failed_a_key: paired_row(
                    failed_a_key,
                    goal_a=failed_a,
                    reached_b=False,
                ),
            }
            for arm in ARMS
        }

        summary = summarize_rows(rows, expected, mode="full")

        self.assertEqual(
            summary["audit"]["common_goal_A_success_eligible"], 1
        )
        self.assertEqual(len(summary["audit"]["excluded_goal_A_failures"]), 1)
        native = summary["arms"]["native_imagegoal"]
        short = summary["arms"]["oracle_short_1p25m"]
        self.assertEqual(
            native["goal_B_given_common_goal_A_success"]["eligible"], 1
        )
        self.assertEqual(
            native["goal_B_given_common_goal_A_success"]["successes"], 0
        )
        self.assertEqual(
            short["goal_B_given_common_goal_A_success"]["successes"], 1
        )
        comparison = summary["pairwise"][
            "oracle_short_1p25m_vs_native_imagegoal"
        ]
        self.assertEqual(comparison["outcomes"]["right_only_B_success"], 1)
        self.assertEqual(comparison["B_sr_delta_right_minus_left"], 1.0)

    def test_any_nested_goal_a_field_difference_fails_closed(self):
        key = ("scene_a", "episode_0000")
        goal_a = goal_a_record(True, 1.5)
        rows = {
            arm: {key: paired_row(key, goal_a=goal_a, reached_b=False)}
            for arm in ARMS
        }
        changed = rows["oracle_final_point"][key]
        changed["goal_a"]["plans"][0]["nested"]["marker"] = 9.0
        changed["goal_a_sha256"] = canonical_sha256(changed["goal_a"])

        with self.assertRaisesRegex(
            RuntimeError, r"Goal-A field mismatch:.*nested.marker"
        ):
            summarize_rows(rows, {key}, mode="smoke")

    def test_missing_arm_episode_fails_closed(self):
        key = ("scene_a", "episode_0000")
        goal_a = goal_a_record(True, 1.5)
        rows = {
            arm: {key: paired_row(key, goal_a=goal_a, reached_b=False)}
            for arm in ARMS
        }
        rows["oracle_short_1p25m"].clear()
        with self.assertRaisesRegex(RuntimeError, "result keys differ"):
            summarize_rows(rows, {key}, mode="smoke")

    def test_loader_recomputes_goal_a_sha_and_checks_seed_echo(self):
        goal_a = goal_a_record(True, 1.5)
        goal_b = {
            "attempted": True,
            "reached": True,
            "path_len": 3.0,
            "path_len_at_reach": 3.0,
            "steps": 2,
            "plans": [{
                "step": 0,
                "requested_diffusion_seed": 20260803 * 100_000 + 10_000,
                "diffusion_seed": 20260803 * 100_000 + 10_000,
                "trajectory_selector": "server",
            }],
            "final_goal_dist_m": 0.5,
        }
        metric = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "scene": "scene_a",
            "episode": "episode_0000",
            "seed": 20260803,
            "arm": "native_imagegoal",
            "server_backend": "navdp",
            "navdp_stop_threshold": -0.5,
            "goal_A_controller": "native_imagegoal",
            "goal_B_controller": "native_imagegoal",
            "oracle_subgoal_m": None,
            "deterministic_plan_seeds": 1,
            "navdp_goal_switch_reset": "carry",
            "success_dist_m": 1.0,
            "max_steps": 1200,
            "exec_horizon": 8,
            "reached_A": 1,
            "reached_B": 1,
            "B_attempted": 1,
            "spl_A": 1.0,
            "spl_B": 1.0,
            "geo_A": 2.0,
            "geo_B": 3.0,
            "len_A": 1.5,
            "len_B": 3.0,
            "len_B_at_reach": 3.0,
            "steps_A": 2,
            "steps_B": 2,
            "final_dist_A": 0.5,
            "final_dist_B": 0.5,
            "goal_A_plan_count": 1,
            "goal_B_plan_count": 1,
            "goal_a_sha256": canonical_sha256(goal_a),
        }
        artifact = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "scene": "scene_a",
            "episode": "episode_0000",
            "seed": 20260803,
            "arm": "native_imagegoal",
            "goal_a_sha256": canonical_sha256(goal_a),
            "geodesic_m": {"A": 2.0, "B": 3.0},
            "goal_a": goal_a,
            "goal_b": goal_b,
        }
        with tempfile.TemporaryDirectory() as temporary:
            scene_root = Path(temporary)
            arm_root = scene_root / "native_imagegoal"
            arm_root.mkdir()
            with (arm_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metric))
                writer.writeheader()
                writer.writerow(metric)
            (arm_root / "summary.json").write_text(json.dumps({
                "schema_version": 1,
                "protocol": PROTOCOL,
                "arm": "native_imagegoal",
                "episodes": 1,
                "goal_A_successes": 1,
                "goal_B_eligible": 1,
                "goal_B_successes": 1,
                "goal_B_sr_given_A": 1.0,
            }) + "\n")
            audit_path = arm_root / "episode_0000_audit.json"
            audit_path.write_text(json.dumps(artifact))

            loaded = load_arm(
                scene_root,
                "native_imagegoal",
                "scene_a",
                {"episode_0000"},
                expected_seed=20260803,
            )
            self.assertTrue(loaded[("scene_a", "episode_0000")]["reached_b"])

            artifact["goal_a_sha256"] = "0" * 64
            audit_path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(RuntimeError, "artifact Goal-A SHA"):
                load_arm(
                    scene_root,
                    "native_imagegoal",
                    "scene_a",
                    {"episode_0000"},
                    expected_seed=20260803,
                )

            artifact["goal_a_sha256"] = canonical_sha256(goal_a)
            artifact["seed"] = 20260804
            audit_path.write_text(json.dumps(artifact))
            metric["seed"] = 20260804
            with (arm_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metric))
                writer.writeheader()
                writer.writerow(metric)
            with self.assertRaisesRegex(RuntimeError, "frozen manifest seed"):
                load_arm(
                    scene_root,
                    "native_imagegoal",
                    "scene_a",
                    {"episode_0000"},
                    expected_seed=20260803,
                )


if __name__ == "__main__":
    unittest.main()
