import csv
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.summarize_xnavdp_revisit_gate import (
    load_gate_arm,
    paired_revisit_summary,
    scene_cluster_interval,
)
from MemNavData.xnavdp_revisit_contract import (
    OFFICIAL_XNAVDP_COMMIT,
    OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
    XNAVDP_CHECKPOINT_TENSOR_COUNT,
    XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
    XNAVDP_MODEL_STATE_TENSOR_COUNT,
)


def _row(*, success, active=True, termination="success", blocked=0):
    return {
        "seed": 7,
        "recall_gap": 80,
        "reached_a": True,
        "reached_b": bool(success),
        "joint": bool(success),
        "spl_a": 0.8,
        "spl_b": 0.7 if success else 0.0,
        "geo_a": 3.0,
        "geo_b": 4.0,
        "path_a": 3.5,
        "path_b": 5.0,
        "final_dist_a": 0.7,
        "final_dist_b": 0.8 if success else 3.0,
        "steps_a": 100,
        "steps_b": 140,
        "termination_reason_b": termination,
        "blocked_steps_b": blocked,
        "router_active_episode_b": bool(active),
        "leg1_trace_sha256": "a" * 64,
    }


class XNavDPRevisitSummaryTest(unittest.TestCase):
    def test_loader_requires_exact_xnavdp_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene_root = Path(temporary)
            arm_root = scene_root / "memory_xnavdp_point"
            arm_root.mkdir()
            metric = {
                "episode": "episode_0000",
                "seed": 7,
                "recall_gap": 80,
                "deterministic_plan_seeds": True,
                "revisit_controller": "xnavdp_point",
                "server_backend": "hybrid_pose",
                "reached_A": 1,
                "reached_B": 1,
                "spl_A": 0.8,
                "spl_B": 0.7,
                "geo_A": 3.0,
                "geo_B": 4.0,
                "len_A": 3.5,
                "len_B": 5.0,
                "final_dist_A": 0.7,
                "terminal_final_goal_dist_m": 0.8,
                "steps_A": 100,
                "steps_B": 140,
                "termination_reason_B": "success",
                "blocked_steps_B": 0,
                "leg1_trace_sha256": "a" * 64,
                "xnavdp_history_contract_valid": True,
                "xnavdp_official_commit": OFFICIAL_XNAVDP_COMMIT,
                "xnavdp_checkpoint_sha256": (
                    OFFICIAL_XNAVDP_POSTTRAIN_SHA256),
                "xnavdp_checkpoint_load_audited": True,
                "xnavdp_model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
                "xnavdp_checkpoint_tensor_count": (
                    XNAVDP_CHECKPOINT_TENSOR_COUNT),
                "xnavdp_checkpoint_missing_count": 0,
                "xnavdp_checkpoint_unexpected_count": (
                    XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT),
                "xnavdp_checkpoint_shape_mismatch_count": 0,
            }
            metric_path = arm_root / "metric.csv"
            with metric_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metric))
                writer.writeheader()
                writer.writerow(metric)
            plans = {
                "legA": [{
                    "step": 0,
                    "requested_diffusion_seed": 10,
                    "diffusion_seed": 10,
                    "router_active": False,
                }],
                "legB": [{
                    "step": 0,
                    "requested_diffusion_seed": 11,
                    "diffusion_seed": 11,
                    "router_active": True,
                }],
            }
            (arm_root / "episode_0000_plans.json").write_text(
                json.dumps(plans))
            loaded = load_gate_arm(
                scene_root, "memory_xnavdp_point", "scene")
            self.assertTrue(
                loaded[("scene", "episode_0000")][
                    "xnavdp_history_contract_valid"])

            metric["xnavdp_official_commit"] = "wrong"
            with metric_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metric))
                writer.writeheader()
                writer.writerow(metric)
            with self.assertRaisesRegex(RuntimeError, "source receipt mismatch"):
                load_gate_arm(scene_root, "memory_xnavdp_point", "scene")

    def test_pair_reports_net_gain_common_activation_and_safety(self):
        keys = {("s0", "e0"), ("s1", "e0")}
        left = {
            ("s0", "e0"): _row(success=False),
            ("s1", "e0"): _row(success=True, active=False),
        }
        right = {
            ("s0", "e0"): _row(success=True),
            ("s1", "e0"): _row(success=True),
        }
        result = paired_revisit_summary(
            "mixed", "x", left, right, keys)
        self.assertEqual(result["outcomes"]["right_only_success"], 1)
        self.assertEqual(result["outcomes"]["left_only_success"], 0)
        self.assertEqual(result["paired_risk_difference"], 0.5)
        self.assertEqual(result["common_activation"]["eligible"], 1)
        self.assertEqual(result["activation_divergence"]["right_only"], 1)
        self.assertEqual(result["safety"]["new_stuck_failure_count"], 0)

    def test_new_paired_stuck_failure_is_explicit(self):
        key = ("s0", "e0")
        left = {key: _row(success=False, termination="max_steps")}
        right = {key: _row(success=False, termination="stuck", blocked=9)}
        result = paired_revisit_summary("mixed", "x", left, right, {key})
        self.assertEqual(result["safety"]["new_stuck_failure_count"], 1)
        self.assertEqual(result["safety"]["right_blocked_steps"], 9)

    def test_scene_cluster_interval_is_reproducible(self):
        values = {"s0": [1.0, 0.0], "s1": [-1.0, 0.0]}
        first = scene_cluster_interval(values, draws=500, seed=11)
        second = scene_cluster_interval(values, draws=500, seed=11)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.0)
        self.assertGreaterEqual(first[1], 0.0)


if __name__ == "__main__":
    unittest.main()
