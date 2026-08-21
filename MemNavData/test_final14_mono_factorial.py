from __future__ import annotations

import unittest

from MemNavData.final14_mono_factorial import (
    ARMS,
    audit_depth_plans,
    interaction_difference,
    rotated_arm_order,
)
from MemNavData.run_final14_mono_factorial_episode import (
    audit_fully_rejected_fallback,
)


def mono_receipt(frame: int = 120) -> dict:
    return {
        "depth_contract": "raw_lingbot_depth_first40_v1",
        "metric_depth_sensor_consumed": False,
        "frame_index": frame,
        "scale_active": True,
        "scale_receipt_sha256": "a" * 64,
        "scale_receipt": {
            "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
            "whole_episode_ground_cache_consumed": False,
        },
    }


class Final14MonoFactorialTest(unittest.TestCase):
    def test_rotation_covers_every_start(self) -> None:
        self.assertEqual(
            {rotated_arm_order(index)[0] for index in range(len(ARMS))},
            set(ARMS),
        )

    def test_mono_depth_contract(self) -> None:
        plans = [{
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
            "monocular_depth_receipt": mono_receipt(),
        }]
        result = audit_depth_plans("mono_cec", plans)
        self.assertEqual(result["monocular_receipt_plan_count"], 1)

    def test_metric_depth_contract(self) -> None:
        plans = [{
            "navdp_depth_source": "metric_request",
            "metric_depth_sensor_consumed": True,
            "monocular_depth_receipt": None,
        }]
        result = audit_depth_plans("metric_native", plans)
        self.assertEqual(result["metric_sensor_plan_count"], 1)

    def test_mono_rejects_bootstrap_receipt(self) -> None:
        plans = [{
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
            "monocular_depth_receipt": mono_receipt(39),
        }]
        with self.assertRaisesRegex(RuntimeError, "bootstrap"):
            audit_depth_plans("mono_native", plans)

    def test_interaction(self) -> None:
        rows = []
        values = {
            "mono_native": 0,
            "mono_cec": 1,
            "metric_native": 0,
            "metric_cec": 0,
        }
        for episode in ("e0", "e1"):
            for arm, reached in values.items():
                rows.append({
                    "scene": "s0",
                    "episode": episode,
                    "arm": arm,
                    "reached": reached,
                })
        result = interaction_difference(rows, seed=7, resamples=100)
        self.assertEqual(result["difference_in_differences"], 1.0)
        self.assertEqual(result["scene_cluster_bootstrap_95"], [1.0, 1.0])

    def test_exact_fallback_audit_executes_on_python39(self) -> None:
        plans = [{
            "requested_diffusion_seed": 7,
            "diffusion_seed": 7,
            "selected_trajectory_sha256": "b" * 64,
        }]
        payload = {
            "query_leg": plans,
            "rollout_traces": {"query": [{"step": 0}]},
        }
        self.assertTrue(audit_fully_rejected_fallback(
            arm="mono_cec",
            role="novel",
            cec_row={"certificate_accept_plans": "0"},
            cec_payload=payload,
            native_payload=payload,
        ))


if __name__ == "__main__":
    unittest.main()
