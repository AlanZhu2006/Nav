import math
import unittest

import numpy as np

from MemNavData.xnavdp_revisit_contract import (
    OFFICIAL_XNAVDP_COMMIT,
    OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
    XNAVDP_ALGO,
    XNAVDP_CHECKPOINT_TENSOR_COUNT,
    XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
    XNAVDP_MODEL_STATE_TENSOR_COUNT,
    habitat_pose_to_xnavdp,
    normalize_xnavdp_response,
    pointgoal_payload,
    validate_history_receipt,
    validate_reset_receipt,
    xnavdp_state_payload,
)


def _reset_receipt():
    return {
        "algo": XNAVDP_ALGO,
        "official_commit": OFFICIAL_XNAVDP_COMMIT,
        "checkpoint_sha256": OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        "actor_mode": "posttrain",
        "embodiment": "wheeled",
        "checkpoint_load_audit": {
            "audited": True,
            "model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
            "checkpoint_tensor_count": XNAVDP_CHECKPOINT_TENSOR_COUNT,
            "missing_count": 0,
            "unexpected_count": XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
            "shape_mismatch_count": 0,
        },
        "history_frame_count": [0],
    }


def _response(seed=7):
    trajectory = np.zeros((1, 24, 3), dtype=float)
    candidates = np.zeros((1, 8, 24, 3), dtype=float)
    values = np.arange(8, dtype=float)[None]
    return {
        "algo": XNAVDP_ALGO,
        "controller": "xnavdp_point_posttrain",
        "diffusion_seed": seed,
        "trajectory": trajectory.tolist(),
        "all_trajectory": candidates.tolist(),
        "all_values": values.tolist(),
        "frames_appended": 1,
        "history_frame_count": [5],
        "official_commit": OFFICIAL_XNAVDP_COMMIT,
        "checkpoint_sha256": OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        "actor_mode": "posttrain",
        "embodiment": "wheeled",
        "checkpoint_load_audit": _reset_receipt()["checkpoint_load_audit"],
    }


class XNavDPRevisitContractTest(unittest.TestCase):
    def test_pointgoal_payload_is_forward_left_batch_one(self):
        self.assertEqual(pointgoal_payload([2.5, -0.75]), {
            "goal_x": [2.5], "goal_y": [-0.75]})
        with self.assertRaises(ValueError):
            pointgoal_payload([1.0, float("nan")])

    def test_habitat_pose_mapping_preserves_forward_left_basis(self):
        for yaw in (0.0, math.pi / 2.0, -math.pi / 3.0, math.pi):
            origin_hab = np.asarray([1.2, 0.5, -3.4])
            origin_x, quat = habitat_pose_to_xnavdp(origin_hab, yaw)
            theta = 2.0 * math.atan2(quat[2], quat[3])
            rotation = np.asarray([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ])
            local = np.asarray([1.7, -0.4])  # forward, left
            world_x = origin_x[:2] + rotation @ local

            dx = -local[0] * math.sin(yaw) - local[1] * math.cos(yaw)
            dz = -local[0] * math.cos(yaw) + local[1] * math.sin(yaw)
            expected_x = np.asarray([
                origin_hab[0] + dx,
                -(origin_hab[2] + dz),
            ])
            np.testing.assert_allclose(world_x, expected_x, atol=1e-12)

    def test_state_payload_is_batched_and_xyzw(self):
        payload = xnavdp_state_payload([2.0, 0.5, 4.0], 0.0)
        self.assertEqual(payload["robot_pos"], [[2.0, -4.0, 0.0]])
        self.assertEqual(len(payload["robot_quat"]), 1)
        np.testing.assert_allclose(
            np.linalg.norm(payload["robot_quat"][0]), 1.0, atol=1e-12)

    def test_reset_and_history_receipts_fail_closed(self):
        self.assertEqual(
            validate_reset_receipt(_reset_receipt())["actor_mode"],
            "posttrain")
        bad = _reset_receipt()
        bad["official_commit"] = "wrong"
        with self.assertRaises(ValueError):
            validate_reset_receipt(bad)
        history = _reset_receipt()
        history.update({
            "diffusion_sampled": False,
            "frames_appended": 1,
            "history_frame_count": [1],
        })
        self.assertEqual(validate_history_receipt(
            history, expected_frame_count=1)["frames_appended"], 1)
        bad_history = dict(history, frames_appended=2)
        with self.assertRaises(ValueError):
            validate_history_receipt(bad_history)
        with self.assertRaisesRegex(ValueError, "cumulative history count"):
            validate_history_receipt(history, expected_frame_count=2)

    def test_response_validation_squeezes_batch_and_checks_seed(self):
        result = normalize_xnavdp_response(
            _response(), expected_seed=7, expected_history_frame_count=5)
        self.assertEqual(np.asarray(result["trajectory"]).shape, (24, 3))
        self.assertEqual(
            np.asarray(result["all_trajectory"]).shape, (8, 24, 3))
        self.assertEqual(np.asarray(result["all_values"]).shape, (8,))
        with self.assertRaises(ValueError):
            normalize_xnavdp_response(_response(seed=8), expected_seed=7)
        bad = _response()
        bad["trajectory"][0][0][0] = float("nan")
        with self.assertRaises(ValueError):
            normalize_xnavdp_response(bad, expected_seed=7)


if __name__ == "__main__":
    unittest.main()
