import io
import math
import unittest
from types import SimpleNamespace

from flask import Flask
import numpy as np
from scipy.spatial.transform import Rotation

from MemNavData import xnavdp_revisit_server as shared_server
from MemNavData.audit_xnavdp_observation_contract import (
    analyze, pil_wire,
)
from MemNavData.xnavdp_mono_diagnostic_server import install
from MemNavData.xnavdp_revisit_contract import habitat_pose_to_xnavdp


class RGBContractTests(unittest.TestCase):
    def test_actual_published_client_and_server_restore_rgb(self):
        raw = np.full((64, 64, 3), [220, 60, 20], dtype=np.uint8)
        result = analyze(raw, "fixture")
        self.assertLess(result["official_vs_raw_mae_255"], 2.)
        self.assertGreater(result["legacy_vs_decoded_rgb_mae_255"], 100.)
        self.assertEqual(result["corrected_vs_decoded_rgb_mae_255"], 0.)

    def test_same_decoder_contract_is_used_for_live_and_history(self):
        app = Flask("x_rgb_contract")
        server = SimpleNamespace(app=app, _decode_rgb=shared_server._decode_rgb)
        install(server)
        wire = pil_wire(np.full((32, 32, 3), [220, 60, 20], dtype=np.uint8))
        for route in ("/pointgoal_step", "/memory_replay_step"):
            for contract, expected in (("rgb_v1", [219, 60, 20]),
                                       ("legacy_bgr", [20, 60, 219])):
                with app.test_request_context(route, method="POST", data={
                        "image": (io.BytesIO(wire), "frame.jpg"),
                        "xnavdp_rgb_contract": contract}):
                    image = server._decode_rgb(1)
                    np.testing.assert_allclose(image[0, 16, 16], expected, atol=2)

    def test_rtc_coordinates_preserve_old_path_under_translation_and_turn(self):
        # Independent standard rigid transform, including backward displacement.
        old_pos, old_yaw = np.array([1.2, .4, -2.]), 2.8
        now_pos, now_yaw = np.array([1.6, .4, -1.2]), -2.9
        local = np.array([[-1., .2], [-2., -.4], [.7, -.1]])
        old_x, old_q = habitat_pose_to_xnavdp(old_pos, old_yaw)
        now_x, now_q = habitat_pose_to_xnavdp(now_pos, now_yaw)
        world = Rotation.from_quat(old_q).apply(np.column_stack((local, np.zeros(3))))+old_x
        current = Rotation.from_quat(now_q).inv().apply(world-now_x)[:, :2]
        world_hab = np.column_stack((
            old_pos[0]-local[:, 0]*math.sin(old_yaw)-local[:, 1]*math.cos(old_yaw),
            old_pos[2]-local[:, 0]*math.cos(old_yaw)+local[:, 1]*math.sin(old_yaw)))
        delta = world_hab-now_pos[[0, 2]]
        expected = np.column_stack((
            -math.sin(now_yaw)*delta[:, 0]-math.cos(now_yaw)*delta[:, 1],
            -math.cos(now_yaw)*delta[:, 0]+math.sin(now_yaw)*delta[:, 1]))
        np.testing.assert_allclose(current, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
