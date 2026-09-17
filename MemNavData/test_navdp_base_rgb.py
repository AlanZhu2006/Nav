import io
from types import SimpleNamespace

from flask import Flask, request
import numpy as np
import pytest

from MemNavData.audit_navdp_base_rgb import analyze, analyze_vint, original_preprocess, pil_wire, route_decode
from MemNavData.navdp_execution_audit_server import install


def test_actual_base_client_decoder_and_preprocessor_color_convention():
    result = analyze(np.full((64, 64, 3), [220, 60, 20], np.uint8))
    np.testing.assert_allclose(result["official_actor_channels"], [220, 60, 20], atol=2)
    np.testing.assert_allclose(result["habitat_legacy_actor_channels"], [20, 60, 220], atol=2)
    assert result["fixed_equals_pil_rgb"]
    assert result["preprocessed_official_channels"][0] > result["preprocessed_official_channels"][2]
    assert result["preprocessed_legacy_channels"][0] < result["preprocessed_legacy_channels"][2]


def test_vint_actual_current_goal_context_path_has_same_legacy_wire_issue():
    rows = analyze_vint(np.full((64, 64, 3), [220, 60, 20], np.uint8))
    assert len(rows) == 3
    for row in rows:
        np.testing.assert_allclose(row["reconstructed_actor_channels"], [20, 60, 219], atol=2)


def test_reset_bound_contract_covers_live_goal_and_replay_without_mutating_wire():
    class Agent:
        def process_image(self, images):
            self.actual = images.copy()
            return original_preprocess(images)
    app = Flask("base_rgb_audit")
    agent = Agent()
    @app.post("/navigator_reset")
    def reset():
        return {"ok": True}
    @app.post("/read")
    def read():
        wire = request.files["image"].read()
        for fn, variable in [("navdp_step_image", "image"), ("navdp_step_image", "goal"),
                             ("navdp_memory_replay_step", "image")]:
            agent.process_image(route_decode(wire, fn, variable))
        return {"ok": True}
    install(SimpleNamespace(app=app), Agent, None)
    client = app.test_client()
    wire = pil_wire(np.full((64, 64, 3), [220, 60, 20], np.uint8))
    for contract, expected in [("legacy_bgr", [20, 60, 219]), ("rgb_v1", [219, 60, 20]),
                               ("legacy_bgr", [20, 60, 219])]:
        assert client.post("/navigator_reset", json={"audit_actor_rgb_contract": contract}).status_code == 200
        response = client.post("/read", data={"image": (io.BytesIO(wire), "image.jpg")})
        assert response.status_code == 200
        receipt = response.get_json()["execution_input_audit"]
        assert receipt["contract"] == contract and len(receipt["image_calls"]) == 3
        for call in receipt["image_calls"]:
            np.testing.assert_allclose(call["input_mean_channels"], expected, atol=2)
    assert client.post("/navigator_reset", json={"audit_actor_rgb_contract": "guess"}).status_code == 400
