"""Real depth resolver + Flask boundary; sidecar HTTP only is substituted."""
import io
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, request
import numpy as np
from PIL import Image
import pytest

from MemNavData.audit_navdp_base_rgb import pil_wire, original_preprocess
from MemNavData.audit_navdp_depth_raster import depth_preprocess
from MemNavData.monocular_depth_runtime import (
    bind_monocular_depth_transaction, build_monocular_depth_payload, image_sha256,
)
from MemNavData.navdp_depth_raster_audit_server import convert_observation, install

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NavDP/baselines/navdp"))
import navdp_server


def scale_receipt():
    return {
        "schema": "mdtec_first40_scale_receipt_v1_20260819",
        "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
        "frozen_after_observation_count": 40, "active_from_frame_index": 40,
        "scale_prefix_first_frame": 0, "scale_prefix_last_frame": 39, "scale_prefix_frames": 40,
        "camera_height_m": .5, "scale_valid": True, "scale_hat": 1.,
        "ground_h_est_raw": .5, "relative_floor_iqr": .01, "valid_frame_ratio": 1.,
        "scale_clamped": False, "freeze_error": None, "whole_episode_ground_cache_consumed": False,
    }


@pytest.fixture
def route(monkeypatch, tmp_path):
    app = Flask("raster_contract_test")
    app.config["TESTING"] = True
    wire = pil_wire(np.full((270, 480, 3), [220, 60, 20], np.uint8))
    payload = bind_monocular_depth_transaction(build_monocular_depth_payload(
        relative_depth=np.full((518, 518), 2., np.float32), depth_shape=(518, 518),
        image_sha256_value=image_sha256(wire), frame_index=40, scale_receipt=scale_receipt()))
    calls = []
    def post(url, data, timeout):
        calls.append(data)
        assert data["monocular_depth_transaction_token"] == payload["monocular_depth_transaction_token"]
        return SimpleNamespace(ok=True, json=lambda:payload)
    monkeypatch.setattr(navdp_server.requests, "post", post)
    monkeypatch.setattr(navdp_server, "active_depth_source", "monocular_sidecar")
    monkeypatch.setattr(navdp_server, "monocular_depth_cache", {})
    monkeypatch.setattr(navdp_server.args, "monocular_depth_url", "http://test-sidecar.invalid")
    server = SimpleNamespace(app=app, _resolve_observation_depth=navdp_server._resolve_observation_depth)
    install(server, tmp_path / "depth_artifacts")
    @app.post("/navigator_reset")
    def reset():
        return {"ok": True}
    @app.post("/read")
    def read():
        image_bytes = request.files["image"].read()
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        depth, metadata = server._resolve_observation_depth(image_bytes, image, None, 1,
            transaction_token=payload["monocular_depth_transaction_token"], expected_frame_index=40)
        actor = depth_preprocess(depth[0, :, :, 0])
        rgb_support = original_preprocess(np.full((1,270,480,3),255,np.uint8))[0,:,:,0] > .5
        return dict(metadata, nonzero_padding=int(np.count_nonzero(actor[~rgb_support])))
    return app.test_client(), wire, payload, calls


def test_reset_bound_modes_preserve_original_cache_and_payload(route):
    client, wire, payload, calls = route
    for mode, pixels, shape in (("legacy_square",21952,[1,518,518,1]),
                                ("source_rgb",0,[1,270,480,1]),
                                ("legacy_square",21952,[1,518,518,1])):
        reset = client.post("/navigator_reset",json={"audit_depth_raster_contract":mode})
        assert reset.status_code == 200
        assert reset.get_json()["audit_depth_raster_contract"] == mode
        response = client.post("/read",data={"image":(io.BytesIO(wire),"rgb.jpg")})
        assert response.status_code == 200
        receipt = response.get_json()
        assert receipt["navdp_depth_raster"]["output_shape"] == shape
        assert receipt["nonzero_padding"] == pixels
        for key in ("depth_png_sha256", "image_sha256", "monocular_depth_transaction_token", "scale_receipt_sha256"):
            assert receipt[key] == payload[key]
    assert len(calls) == 1  # conversion does not modify or refetch the cached depth
    cached = next(iter(navdp_server.monocular_depth_cache.values()))[0]
    assert cached.shape == (518, 518)
    np.testing.assert_allclose(cached,2.)


def test_invalid_mode_rejected_without_changing_active_mode(route):
    client, wire, _, _ = route
    client.post("/navigator_reset",json={"audit_depth_raster_contract":"source_rgb"})
    assert client.post("/navigator_reset",json={"audit_depth_raster_contract":"guess"}).status_code == 400
    receipt = client.post("/read",data={"image":(io.BytesIO(wire),"rgb.jpg")}).get_json()
    assert receipt["navdp_depth_raster"]["contract"] == "source_rgb"


def test_saved_depths_identify_producer_and_actual_readout(route):
    client, wire, _, _ = route
    client.post("/navigator_reset", json={"audit_depth_raster_contract": "source_rgb"})
    receipt = client.post("/read", data={"image": (io.BytesIO(wire), "rgb.jpg")}).get_json()
    raster = receipt["navdp_depth_raster"]
    artifact = Path(raster["artifact"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == raster["artifact_sha256"]
    with np.load(artifact, allow_pickle=False) as saved:
        producer, readout = saved["producer_depth"], saved["navdp_readout_depth"]
        assert producer.shape == (1, 518, 518, 1)
        assert readout.shape == (1, 270, 480, 1)
        np.testing.assert_array_equal(saved["source_rgb_hw"], [270, 480])
        np.testing.assert_array_equal(producer, 2.)
        np.testing.assert_array_equal(readout, 2.)
        assert hashlib.sha256(producer.tobytes()).hexdigest() == raster["input_tensor_sha256"]
        assert hashlib.sha256(readout.tobytes()).hexdigest() == raster["output_tensor_sha256"]


@pytest.mark.parametrize("state", ["bootstrap_zero_depth", "frozen_scale_invalid_zero_depth"])
def test_bootstrap_zero_is_not_fabricated_geometry(state):
    depth=np.zeros((1,270,480,1),np.float32)
    result,receipt=convert_observation(depth,{"depth_source":"monocular_sidecar","scale_state":state},(270,480),"source_rgb")
    np.testing.assert_array_equal(result,depth)
    assert receipt["navdp_depth_raster"]["operation"] == "source_zero_unchanged"


def test_active_mismatched_raster_does_not_guess_or_fallback():
    with pytest.raises(ValueError):
        convert_observation(np.ones((1,270,480,1),np.float32),
            {"depth_source":"monocular_sidecar","scale_state":"raw_lingbot_metric_depth"},(270,480),"source_rgb")


def test_metric_control_not_changed_by_monocular_raster_adapter():
    depth=np.full((1,270,480,1),1.7,np.float32)
    result,receipt=convert_observation(depth,{"depth_source":"metric_request"},(270,480),"source_rgb")
    np.testing.assert_array_equal(result,depth)
    assert receipt["navdp_depth_raster"]["operation"] == "identity"
