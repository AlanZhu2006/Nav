import tempfile
import unittest
from pathlib import Path

from MemNavData.mdtec_monocular_cec_composition import (
    ARMS,
    audit_arm_leg_b,
    audit_shared_leg_a,
    exact_mcnemar_two_sided,
    paired_contrast,
    rotated_arm_order,
    scene_cluster_interval,
)
from MemNavData.independent_verify_mdtec_monocular_cec_composition import (
    load as load_verified_rows,
)


def receipt(frame, *, valid=True):
    active = frame >= 40
    return {
        "metric_depth_sensor_consumed": False,
        "frame_index": frame,
        "depth_nonzero_fraction": 1.0 if active and valid else 0.0,
        "scale_active": active,
        "scale_receipt": ({"scale_valid": valid} if active else None),
        "scale_receipt_sha256": "c" * 64 if active else None,
    }


def plan(frame, **extra):
    row = {"monocular_depth_receipt": receipt(frame)}
    row.update(extra)
    return row


class ArmRotationTest(unittest.TestCase):
    def test_two_arm_rotation_is_balanced(self):
        self.assertEqual(rotated_arm_order(0, 0), ARMS)
        self.assertEqual(rotated_arm_order(1, 0), ARMS[1:] + ARMS[:1])
        self.assertEqual(rotated_arm_order(2, 0), ARMS)


class SharedLegAAuditTest(unittest.TestCase):
    def test_valid_shared_leg_a_passes(self):
        outcome = {
            "plans": [plan(0), plan(32), plan(40)],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        audit_shared_leg_a(outcome)  # must not raise

    def test_metric_sensor_consumption_fails_closed(self):
        outcome = {
            "plans": [plan(40)],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": True,
        }
        with self.assertRaisesRegex(RuntimeError, "consumed simulator metric depth"):
            audit_shared_leg_a(outcome)

    def test_double_scale_freeze_fails_closed(self):
        plans = [plan(40), plan(41)]
        plans[1]["monocular_depth_receipt"]["scale_receipt_sha256"] = "d" * 64
        outcome = {
            "plans": plans,
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        with self.assertRaisesRegex(RuntimeError, "froze scale more than once"):
            audit_shared_leg_a(outcome)


class ArmLegBAuditTest(unittest.TestCase):
    def test_raw_native_rejects_certified_activity(self):
        outcome = {
            "plans": [{"certified_relocalization_ok": True}],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        with self.assertRaisesRegex(RuntimeError, "unexpected certified_relocalization"):
            audit_arm_leg_b("raw_native", outcome, native_outcome=None)

    def test_raw_cec_runtime_failure_fails_closed(self):
        outcome = {
            "plans": [{"certified_relocalization_ok": False}],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        with self.assertRaisesRegex(RuntimeError, "runtime/transport failure"):
            audit_arm_leg_b("raw_cec", outcome, native_outcome=None)

    def test_raw_cec_reject_must_match_native_action(self):
        native_outcome = {"plans": [
            {"requested_diffusion_seed": 1, "diffusion_seed": 1,
             "selected_trajectory_sha256": "x"},
        ]}
        matching_cec = {
            "plans": [
                {"certified_relocalization_ok": True,
                 "certified_relocalization_accepted": False,
                 "requested_diffusion_seed": 1, "diffusion_seed": 1,
                 "selected_trajectory_sha256": "x"},
            ],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        result = audit_arm_leg_b("raw_cec", matching_cec, native_outcome=native_outcome)
        self.assertEqual(result["certified_runtime_failure_count"], 0)

        diverging_cec = {
            "plans": [
                {"certified_relocalization_ok": True,
                 "certified_relocalization_accepted": False,
                 "requested_diffusion_seed": 1, "diffusion_seed": 1,
                 "selected_trajectory_sha256": "y"},
            ],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        with self.assertRaisesRegex(RuntimeError, "reject: selected trajectory diverges"):
            audit_arm_leg_b("raw_cec", diverging_cec, native_outcome=native_outcome)

    def test_raw_cec_accepted_plan_may_diverge_from_native(self):
        native_outcome = {"plans": [
            {"requested_diffusion_seed": 1, "diffusion_seed": 1,
             "selected_trajectory_sha256": "x"},
        ]}
        accepted_cec = {
            "plans": [
                {"certified_relocalization_ok": True,
                 "certified_relocalization_accepted": True,
                 "requested_diffusion_seed": 99, "diffusion_seed": 99,
                 "selected_trajectory_sha256": "z"},
            ],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": False,
        }
        result = audit_arm_leg_b("raw_cec", accepted_cec, native_outcome=native_outcome)
        self.assertEqual(result["certified_accept_count"], 1)


class ReusedStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_reused_from_gate_d(self):
        self.assertEqual(exact_mcnemar_two_sided(0, 0), 1.0)

    def test_paired_and_cluster_statistics(self):
        rows = []
        values = {
            ("s0", "e0"): (1, 0), ("s0", "e1"): (1, 1),
            ("s1", "e0"): (0, 1), ("s1", "e1"): (1, 0),
        }
        for (scene, episode), (cec, native) in values.items():
            rows.extend([
                {"scene": scene, "episode": episode, "arm": "raw_cec", "reached": cec},
                {"scene": scene, "episode": episode, "arm": "raw_native", "reached": native},
            ])
        contrast = paired_contrast(rows, "raw_cec", "raw_native")
        self.assertEqual((contrast["gains"], contrast["losses"]), (2, 1))
        first = scene_cluster_interval(rows, "raw_cec", "raw_native", seed=7, resamples=1000)
        second = scene_cluster_interval(rows, "raw_cec", "raw_native", seed=7, resamples=1000)
        self.assertEqual(first, second)


class IndependentDistanceVerificationTest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, reached: int, distance: float) -> None:
        scene = root / "scenes" / "00_scene"
        scene.mkdir(parents=True)
        (scene / "plans.json").write_text(
            '{"arm": "raw_native", "plans": []}\n')
        (scene / "depth_arms.csv").write_text(
            "arm,episode,plans_file,reached,reached_B,"
            "metric_depth_sensor_consumed_any,certified_runtime_failure_count,"
            "scene\n"
            f"raw_native,episode_0000,plans.json,{reached},{reached},False,0,scene\n"
        )
        arm = scene / "episode_0000_raw_native"
        arm.mkdir()
        (arm / "metric.csv").write_text(
            "episode,final_dist_B\n"
            f"episode_0000,{distance}\n"
        )

    def test_success_is_recomputed_from_retained_raw_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, reached=1, distance=0.75)
            rows = load_verified_rows(root, 1.0)
            self.assertEqual(rows[0]["_reached_from_distance"], 1)
            self.assertEqual(rows[0]["_raw_final_dist_B"], 0.75)

    def test_reported_success_distance_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, reached=0, distance=0.75)
            with self.assertRaisesRegex(RuntimeError, "disagrees with raw distance"):
                load_verified_rows(root, 1.0)


if __name__ == "__main__":
    unittest.main()
