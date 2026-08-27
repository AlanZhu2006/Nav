import unittest

from MemNavData.mdtec_raw_depth_gate_d import (
    ARMS,
    audit_arm_contract,
    exact_mcnemar_two_sided,
    paired_contrast,
    rotated_arm_order,
    scene_cluster_interval,
)


def plan(source, frame=None, *, valid=True):
    row = {
        "navdp_depth_source": source,
        "metric_depth_sensor_consumed": source == "metric_request",
    }
    if frame is not None:
        active = frame >= 40
        scale = {
            "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
            "whole_episode_ground_cache_consumed": False,
            "scale_valid": valid,
        } if active else None
        row["monocular_depth_receipt"] = {
            "depth_contract": "raw_lingbot_depth_first40_v1",
            "metric_depth_sensor_consumed": False,
            "image_sha256": "a" * 64,
            "depth_png_sha256": "b" * 64,
            "frame_index": frame,
            "depth_nonzero_fraction": 1.0 if active and valid else 0.0,
            "scale_active": active,
            "scale_receipt": scale,
            "scale_receipt_sha256": "c" * 64 if active else None,
        }
    return row


class GateDContractTest(unittest.TestCase):
    def test_arm_rotation_is_balanced(self):
        self.assertEqual(rotated_arm_order(0, 0), ARMS)
        self.assertEqual(rotated_arm_order(1, 0), ARMS[1:] + ARMS[:1])
        self.assertEqual(rotated_arm_order(1, 1), ARMS[2:] + ARMS[:2])

    def test_raw_prefix_and_active_receipts(self):
        plans = [plan("monocular_sidecar", 0),
                 plan("monocular_sidecar", 32),
                 plan("monocular_sidecar", 40)]
        result = audit_arm_contract("raw_first40", {
            "plans": plans,
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
            "monocular_frame40_survived": True,
        })
        self.assertEqual(result["bootstrap_plan_count"], 2)
        self.assertEqual(result["active_plan_count"], 1)
        self.assertEqual(result["scale_receipt_unique_count"], 1)

    def test_raw_rejects_sensor_consumption(self):
        with self.assertRaisesRegex(RuntimeError, "metric depth was consumed"):
            audit_arm_contract("raw_first40", {
                "plans": [plan("monocular_sidecar", 40)],
                "navdp_depth_source": "monocular_sidecar",
                "metric_depth_sensor_consumed_any": True,
                "monocular_frame40_survived": True,
            })

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_two_sided(12, 0), 0.00048828125)

    def test_paired_and_cluster_statistics(self):
        rows = []
        values = {
            ("s0", "e0"): (1, 0),
            ("s0", "e1"): (1, 1),
            ("s1", "e0"): (0, 1),
            ("s1", "e1"): (1, 0),
        }
        for (scene, episode), (raw, metric) in values.items():
            rows.extend([
                {"scene": scene, "episode": episode,
                 "arm": "raw_first40", "reached": raw},
                {"scene": scene, "episode": episode,
                 "arm": "metric_teacher", "reached": metric},
            ])
        contrast = paired_contrast(rows, "raw_first40", "metric_teacher")
        self.assertEqual((contrast["gains"], contrast["losses"]), (2, 1))
        self.assertAlmostEqual(contrast["risk_difference"], 0.25)
        first = scene_cluster_interval(
            rows, "raw_first40", "metric_teacher", seed=7, resamples=1000)
        second = scene_cluster_interval(
            rows, "raw_first40", "metric_teacher", seed=7, resamples=1000)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
