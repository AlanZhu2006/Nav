import math
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from MemNavData.habitat_xnavdp_tracking import (
    install_rgb_only_transport, world_to_local_reference,
)


class TrackingContracts(unittest.TestCase):
    def test_roundtrip_all_quadrants(self):
        points = np.array([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
        position = np.array([1.2, .3, -2.4])
        for yaw in (0., .7, -2.9, math.pi):
            forward, left = points.T
            world = np.column_stack((
                position[0]-forward*math.sin(yaw)-left*math.cos(yaw),
                position[2]-forward*math.cos(yaw)+left*math.sin(yaw)))
            result = world_to_local_reference(world, position, yaw)
            np.testing.assert_allclose(result[0], 0., atol=1e-12)
            np.testing.assert_allclose(result[1:], points, atol=1e-12)

    def test_x_wire_rgb_only_and_native_history_once(self):
        calls = []
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(ok=True, raise_for_status=lambda: None,
                                   json=lambda: {"diffusion_sampled": False})
        base = SimpleNamespace(XNAVDP_BASE="http://X", NOVEL_BASE="http://base",
                               requests=SimpleNamespace(post=post))
        install_rgb_only_transport(base)
        files = {"image": ("a.jpg", b"rgb"), "depth": ("d.png", b"sensor")}
        base.requests.post("http://X/pointgoal_step", files=files, data={"seed": "3"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(calls[0][1]["files"]), {"image"})
        self.assertEqual(calls[1][0], "http://base/memory_replay_step")
        self.assertIn("depth", files)  # Do not mutate caller/other arms.
        base.requests.post("http://base/imagegoal_step", files=files)
        self.assertEqual(len(calls), 3)
        self.assertIn("depth", calls[-1][1]["files"])

    def test_monocular_adapter_uses_bound_payload_and_rejects_sensor_wire(self):
        from flask import Flask, request, jsonify
        from MemNavData.xnavdp_mono_diagnostic_server import install
        from MemNavData.monocular_depth_runtime import (
            build_monocular_depth_payload, bind_monocular_depth_transaction,
        )
        app = Flask("x_mono_test")
        app.testing = True
        image_bytes = b"causal-rgb"
        digest = hashlib.sha256(image_bytes).hexdigest()
        payload = build_monocular_depth_payload(relative_depth=None, depth_shape=(3, 4),
                    image_sha256_value=digest, frame_index=0, scale_receipt=None)
        payload = bind_monocular_depth_transaction(payload)
        server = SimpleNamespace(app=app, _decode_rgb=lambda n: request.files["image"].read(),
                                 _navigator=SimpleNamespace(process_pointgoal=lambda p: p))
        server._decode_rgb = lambda n: np.zeros((n, 3, 4, 3), dtype=np.uint8)
        install(server)
        @app.route("/pointgoal_step", methods=["POST"])
        def query():
            server._decode_rgb(1)
            depth = server._decode_depth(1)
            self.assertEqual(depth.shape, (1, 3, 4, 1))
            self.assertEqual(float(depth.sum()), 0.)
            return jsonify({"trajectory": []})
        data = {"monocular_depth_transaction_token": payload["monocular_depth_transaction_token"],
                "monocular_depth_frame_index": "0", "goal_data": '{"goal_x":[-2.5],"goal_y":[0.0]}' }
        fake = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "XNAVDP_MONOCULAR_DEPTH_URL": "http://sidecar", "XNAVDP_DIAGNOSTIC_LOG": str(Path(tmp)/"calls.jsonl")}), \
                patch("requests.post", return_value=fake) as post:
            with app.test_client() as client:
                result = client.post("/pointgoal_step", data=dict(data, image=(io.BytesIO(image_bytes), "x.jpg")))
                self.assertEqual(result.status_code, 200)
                self.assertFalse(result.json["metric_depth_sensor_consumed"])
                self.assertEqual(result.json["xnavdp_input_diagnostic"]["goal_after_processing"][0][0], -2.5)
                self.assertEqual(post.call_args.kwargs["data"]["expected_image_sha256"], digest)
                self.assertEqual(result.json["actor_rgb_channel_order"], "RGB")
                legacy = client.post("/pointgoal_step", data=dict(data,
                    image=(io.BytesIO(image_bytes), "x.jpg"), xnavdp_rgb_contract="legacy_bgr"))
                self.assertEqual(legacy.status_code, 200)
                self.assertEqual(legacy.json["actor_rgb_channel_order"], "BGR")
                with self.assertRaises(ValueError):
                    client.post("/pointgoal_step", data=dict(data,
                        image=(io.BytesIO(image_bytes), "x.jpg"), depth=(io.BytesIO(b"gt"), "gt.png")))

    def test_certified_mixed_dispatch_reaches_x_with_same_point_and_depth_token(self):
        import requests
        calls = []
        def post(url, **kw):
            calls.append((url, kw))
            r = requests.Response()
            r.status_code = 200
            payload = ({"controller": "xnavdp_point_posttrain", "history_frame_count": [2]}
                       if url == "http://X/pointgoal_step" else {"diffusion_sampled": False})
            r._content = json.dumps(payload).encode()
            return r
        base = SimpleNamespace(args=SimpleNamespace(revisit_controller="xnavdp_point"),
            XNAVDP_BASE="http://X", NOVEL_BASE="http://base", requests=SimpleNamespace(post=post),
            XNAVDP_CLIENT_STATE={"history_frame_count": 1},
            normalize_xnavdp_response=lambda r, **kw: r)
        point = '{"goal_x":[-2.5],"goal_y":[0.0]}'
        def certified(*a, **kw):
            return base.requests.post("http://base/navdp_step_ip_mixgoal",
                files={"image": ("a.jpg", b"rgb"), "image_goal": ("g.jpg", b"goal"),
                       "depth": ("d.png", b"gt")},
                data={"goal_data": point, "diffusion_seed": "7",
                      "monocular_depth_transaction_token": "receipt"}).json()
        base.srv_plan = certified
        install_rgb_only_transport(base)
        result = base.srv_plan(robot_position=np.zeros(3), robot_yaw=0.)
        self.assertEqual(result["pose_controller"], "xnavdp_point_posttrain")
        self.assertEqual(calls[0][0], "http://X/pointgoal_step")
        self.assertEqual(set(calls[0][1]["files"]), {"image"})
        self.assertEqual(calls[0][1]["data"]["goal_data"], point)
        self.assertEqual(calls[0][1]["data"]["monocular_depth_transaction_token"], "receipt")
        self.assertIn("state_data", calls[0][1]["data"])
        self.assertEqual(base.XNAVDP_CLIENT_STATE["history_frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
