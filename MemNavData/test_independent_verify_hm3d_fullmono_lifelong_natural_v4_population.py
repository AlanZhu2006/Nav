#!/usr/bin/env python3

import hashlib
import tempfile
import unittest
from pathlib import Path

from independent_verify_hm3d_fullmono_lifelong_natural_v4_population import (
    audit_mono_plans,
    expected_attrition_reasons,
    verify_file_ledger,
)


class NaturalV4PopulationVerifierTest(unittest.TestCase):
    @staticmethod
    def mono_plan(scale_hash="scale"):
        return {
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
            "monocular_depth_receipt": {
                "depth_contract": "raw_lingbot_depth_first40_v1",
                "metric_depth_sensor_consumed": False,
                "frame_index": 40,
                "scale_active": True,
                "scale_receipt_sha256": scale_hash,
                "scale_receipt": {
                    "scale_evidence_contract":
                        "causal_first_prefix_rgb_only_v1",
                    "whole_episode_ground_cache_consumed": False,
                },
            },
        }

    def test_raw_mono_audit_rejects_scale_drift(self):
        self.assertEqual(
            audit_mono_plans([self.mono_plan(), self.mono_plan()]),
            {
                "metric_sensor_plan_count": 0,
                "monocular_receipt_plan_count": 2,
                "monocular_scale_hash_count": 1,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "multiple scale"):
            audit_mono_plans([
                self.mono_plan("first"), self.mono_plan("second")
            ])

    def test_attrition_is_recomputed_from_frozen_thresholds(self):
        protocol = {"factual_b_collection": {
            "B_goal_support_by_factual_B_minimum_inclusive": 0.2,
            "actual_B_end_to_C_geodesic_band_m": [2.0, 9.0],
        }}
        self.assertEqual(
            expected_attrition_reasons({}, {"reached_B": False}, protocol),
            {"actual_mono_B_failed"},
        )
        self.assertEqual(
            expected_attrition_reasons(
                {
                    "B_goal_max_factual_B_covis": 0.19,
                    "actual_B_end_to_C_geodesic_m": 9.1,
                },
                {"reached_B": True},
                protocol,
            ),
            {
                "B_goal_not_supported_by_factual_B",
                "actual_B_end_to_C_geodesic_outside_band",
            },
        )

    def test_file_ledger_requires_exact_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.txt"
            payload.write_text("payload\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / "FILES.sha256").write_text(
                f"{digest}  payload.txt\n"
            )
            self.assertEqual(
                verify_file_ledger(
                    root, "FILES.sha256", {"FILES.sha256"}
                ),
                1,
            )
            (root / "untracked.txt").write_text("extra\n")
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                verify_file_ledger(
                    root, "FILES.sha256", {"FILES.sha256"}
                )


if __name__ == "__main__":
    unittest.main()
