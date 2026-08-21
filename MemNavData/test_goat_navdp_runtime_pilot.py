import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MemNavData.goat_navdp_runtime_pilot import (
    NAVDP_UPSTREAM_CRITIC_THRESHOLD,
    _camera_intrinsic,
    _depth_png_bytes,
    _load_manifest,
    _navdp_reset,
    _navdp_wire_jpeg_bytes,
    _normalize_trajectory,
    _plan_seed,
    _validated_critic_receipt,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


class GoatNavDPRuntimePilotTest(unittest.TestCase):
    def test_upstream_threshold_is_critic_score_not_metric_radius(self):
        self.assertEqual(NAVDP_UPSTREAM_CRITIC_THRESHOLD, -0.5)
        self.assertLess(NAVDP_UPSTREAM_CRITIC_THRESHOLD, 0.0)

    def test_camera_intrinsic_uses_horizontal_fov_and_square_pixels(self):
        intrinsic = _camera_intrinsic(640, 360, 42.0)
        expected = 180.0 / np.tan(np.deg2rad(21.0))
        self.assertAlmostEqual(intrinsic[0, 0], expected)
        self.assertAlmostEqual(intrinsic[1, 1], expected)
        self.assertEqual(intrinsic[0, 2], 180.0)
        self.assertEqual(intrinsic[1, 2], 320.0)

    def test_normalize_trajectory_removes_single_environment_axis(self):
        raw = np.zeros((1, 24, 3), dtype=np.float32)
        self.assertEqual(_normalize_trajectory(raw).shape, (24, 3))

    def test_normalize_trajectory_rejects_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "non-empty and finite"):
            _normalize_trajectory([[0.0, float("nan"), 0.0]])

    def test_plan_seed_is_request_stable_and_plan_specific(self):
        first = _plan_seed(7, "scene", "3", 0)
        self.assertEqual(first, _plan_seed(7, "scene", "3", 0))
        self.assertNotEqual(first, _plan_seed(7, "scene", "3", 1))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**63)

    def test_depth_wire_format_is_uint16_times_ten_thousand(self):
        encoded = _depth_png_bytes(np.asarray([[[0.5], [1.25]]], np.float32))
        decoded = np.asarray(Image.open(io.BytesIO(encoded)))
        np.testing.assert_array_equal(decoded, np.asarray([[5000, 12500]]))

    def test_navdp_wire_round_trip_preserves_rgb_at_server_model_input(self):
        import cv2

        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[:, :32] = (240, 20, 10)
        rgb[:, 32:64] = (10, 230, 20)
        rgb[:, 64:] = (20, 10, 220)

        encoded = _navdp_wire_jpeg_bytes(rgb)
        server_pil_rgb = np.asarray(
            Image.open(io.BytesIO(encoded)).convert("RGB"))
        server_model_input = cv2.cvtColor(
            server_pil_rgb, cv2.COLOR_RGB2BGR)

        error = np.abs(
            server_model_input.astype(np.int16) - rgb.astype(np.int16))
        self.assertLess(float(error.mean()), 5.0)
        np.testing.assert_array_equal(
            np.argmax(server_model_input.mean(axis=(0, 1))),
            np.argmax(rgb.mean(axis=(0, 1))),
        )

    def test_navdp_reset_requires_echoed_critic_threshold_semantics(self):
        session = _Session({
            "algo": "navdp",
            "stop_threshold": NAVDP_UPSTREAM_CRITIC_THRESHOLD,
            "threshold_semantics": "critic_score_fallback",
            "checkpoint_contract": {
                "exact_state_dict": True,
                "temporal_depth": 16,
            },
        })
        receipt = _navdp_reset(
            session,
            "http://localhost:8888",
            np.eye(3),
            7,
            NAVDP_UPSTREAM_CRITIC_THRESHOLD,
            1.0,
        )
        self.assertEqual(
            receipt["stop_threshold"], NAVDP_UPSTREAM_CRITIC_THRESHOLD)
        self.assertEqual(
            session.calls[0][1]["json"]["stop_threshold"],
            NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        )

        session = _Session({
            "algo": "navdp",
            "stop_threshold": 0.2,
            "threshold_semantics": "critic_score_fallback",
            "checkpoint_contract": {
                "exact_state_dict": True,
                "temporal_depth": 16,
            },
        })
        with self.assertRaisesRegex(RuntimeError, "changed"):
            _navdp_reset(
                session,
                "http://localhost:8888",
                np.eye(3),
                7,
                NAVDP_UPSTREAM_CRITIC_THRESHOLD,
                1.0,
            )

    def test_critic_receipt_matches_scores_and_threshold(self):
        receipt = _validated_critic_receipt({
            "all_values": [[-0.8, -0.6]],
            "critic_max": -0.6,
            "critic_min": -0.8,
            "critic_threshold": -0.5,
            "critic_fallback_applied": True,
        })
        self.assertTrue(receipt["critic_fallback_applied"])
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            _validated_critic_receipt({
                "all_values": [[-0.8, -0.6]],
                "critic_max": -0.6,
                "critic_min": -0.8,
                "critic_threshold": -0.5,
                "critic_fallback_applied": False,
            })

    def test_manifest_requires_ten_unique_scenes(self):
        payload = {
            "schema_version": "goat_navdp_runtime_pilot_manifest_v1_20260814",
            "base_seed": 1,
            "max_navigation_actions": 10,
            "episodes": [
                {"scene_id": "same", "episode_id": str(index)}
                for index in range(10)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "ten unique scenes"):
                _load_manifest(path)


if __name__ == "__main__":
    unittest.main()
