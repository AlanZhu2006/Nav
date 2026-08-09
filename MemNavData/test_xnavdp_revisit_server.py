import io
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from MemNavData import xnavdp_revisit_server as server
from MemNavData.xnavdp_revisit_contract import (
    XNAVDP_CHECKPOINT_TENSOR_COUNT,
    XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
    XNAVDP_MODEL_STATE_TENSOR_COUNT,
    OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
    XNAVDP_ALGO,
    normalize_xnavdp_response,
    validate_history_receipt,
    validate_reset_receipt,
)


class _FakeNavigator:
    def __init__(self, intrinsic):
        self.image_intrinsic = np.asarray(intrinsic)
        self.memory_size = 8
        self.navi_former = type("Policy", (), {"ft_step": 6})()
        self.reset(1)

    def reset(self, batch_size):
        self.batch_size = int(batch_size)
        self.frame_count = [0]

    def reset_env(self, env_id):
        assert env_id == 0
        self.frame_count[0] = 0

    def process_image(self, images):
        return np.asarray(images, dtype=np.float32) / 255.0

    def _update_and_sample_history(self, processed, num_samples):
        assert num_samples == self.memory_size
        self.frame_count[0] += 1
        return np.zeros((1, 8, 4, 4, 3), dtype=np.float32)

    def step_pointgoal_with_guidance(
            self, goal, image, depth, robot_pos, robot_quat):
        self.frame_count[0] += 1
        candidates = np.zeros((1, 8, 24, 3), dtype=np.float32)
        candidates[0, :, :, 0] = np.arange(8)[:, None] / 10.0
        values = np.arange(8, dtype=np.float32)[None]
        return candidates[:, -1], candidates, values, None


def _rgb_bytes():
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(
        buffer, format="JPEG")
    return buffer.getvalue()


def _depth_bytes():
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 4), 10000, dtype=np.uint16)).save(
        buffer, format="PNG")
    return buffer.getvalue()


class XNavDPRevisitServerTest(unittest.TestCase):
    def setUp(self):
        server._navigator = None
        server._device = "cpu"
        server._actor_mode = "posttrain"
        server._embodiment_name = "wheeled"
        server._checkpoint_sha256 = OFFICIAL_XNAVDP_POSTTRAIN_SHA256
        server._checkpoint_load_audit = {
            "audited": True,
            "model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
            "checkpoint_tensor_count": XNAVDP_CHECKPOINT_TENSOR_COUNT,
            "missing_count": 0,
            "unexpected_count": XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
            "shape_mismatch_count": 0,
        }
        server.app.config.update(TESTING=True)
        self.client = server.app.test_client()

    def test_reset_replay_and_pointgoal_each_append_exactly_once(self):
        fake = _FakeNavigator(np.eye(3))
        with mock.patch.object(server, "_build_navigator", return_value=fake):
            reset = self.client.post("/navigator_reset", json={
                "intrinsic": np.eye(3).tolist(),
                "batch_size": 1,
                "seed": 17,
            })
        self.assertEqual(reset.status_code, 200)
        validate_reset_receipt(reset.get_json())
        self.assertEqual(fake.frame_count, [0])

        replay = self.client.post(
            "/memory_replay_step",
            data={"image": (io.BytesIO(_rgb_bytes()), "image.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(replay.status_code, 200)
        validate_history_receipt(replay.get_json())
        self.assertEqual(fake.frame_count, [1])

        point = self.client.post(
            "/pointgoal_step",
            data={
                "image": (io.BytesIO(_rgb_bytes()), "image.jpg"),
                "depth": (io.BytesIO(_depth_bytes()), "depth.png"),
                "goal_data": json.dumps({
                    "goal_x": [2.0], "goal_y": [-0.5]}),
                "state_data": json.dumps({
                    "robot_pos": [[1.0, 2.0, 0.0]],
                    "robot_quat": [[0.0, 0.0, 0.0, 1.0]],
                }),
                "diffusion_seed": "23",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(point.status_code, 200, point.get_data(as_text=True))
        payload = normalize_xnavdp_response(
            point.get_json(), expected_seed=23,
            expected_history_frame_count=2)
        self.assertEqual(payload["algo"], XNAVDP_ALGO)
        self.assertTrue(payload["rtc_robot_state_used"])
        self.assertEqual(fake.frame_count, [2])

    def test_checkpoint_coverage_audit_rejects_missing_model_tensor(self):
        model = mock.Mock()
        model.state_dict.return_value = {
            "present": np.zeros((2, 2)),
            "missing": np.zeros((1,)),
        }
        checkpoint = {"present": np.zeros((2, 2))}
        with mock.patch.object(server.torch, "load", return_value=checkpoint):
            with self.assertRaisesRegex(RuntimeError, "coverage differs"):
                server._audit_checkpoint_model_coverage(
                    model, Path("/checkpoint.ckpt"))

    def test_pointgoal_rejects_partial_robot_state_without_appending(self):
        fake = _FakeNavigator(np.eye(3))
        server._navigator = fake
        response = self.client.post(
            "/pointgoal_step",
            data={
                "image": (io.BytesIO(_rgb_bytes()), "image.jpg"),
                "depth": (io.BytesIO(_depth_bytes()), "depth.png"),
                "goal_data": json.dumps({"goal_x": [1.0], "goal_y": [0.0]}),
                "state_data": json.dumps({"robot_pos": [[0.0, 0.0, 0.0]]}),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("supplied together", response.get_json()["error"])
        self.assertEqual(fake.frame_count, [0])

    def test_tracked_official_source_changes_fail_closed(self):
        dirty = mock.Mock(stdout=" M baselines/x-navdp/eval/src/policy_agent.py\n")
        with mock.patch.object(server.subprocess, "run", return_value=dirty):
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                server._assert_clean_official_checkout(Path("/official"))


if __name__ == "__main__":
    unittest.main()
