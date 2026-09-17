#!/usr/bin/env python3
"""Private X-NavDP adapter: published actor, frame-bound LingBot depth.

No uploaded sensor depth is decoded. Official actor/RTC/PointGoal processing
is unchanged; standard RGB and frame-bound depth are supplied by the adapter.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def actor_rgb_from_decoded_rgb(decoded_rgb, contract):
    """Keep the canonical shared decoder's RGB, or replay the diagnosed bug.

    ``legacy_bgr`` is only a consumed diagnostic control. The shared X service
    and normal diagnostic path both use RGB; there is no second corrective
    swap that could undo the shared service's repair.
    """
    if contract == "rgb_v1":
        return np.ascontiguousarray(decoded_rgb)
    if contract == "legacy_bgr":
        return np.ascontiguousarray(decoded_rgb[..., ::-1])
    raise ValueError(f"Unknown diagnostic RGB contract: {contract}")


def install(server):
    from flask import g, request
    import requests
    from MemNavData.monocular_depth_runtime import (
        decode_monocular_depth_payload, validate_monocular_depth_transaction,
    )

    original_rgb = server._decode_rgb

    def rgb(batch_size):
        stream = request.files["image"].stream
        offset = stream.tell()
        g.image_sha256 = hashlib.sha256(stream.read()).hexdigest()
        stream.seek(offset)
        contract = request.form.get("xnavdp_rgb_contract", "rgb_v1")
        image = actor_rgb_from_decoded_rgb(original_rgb(batch_size), contract)
        g.rgb_receipt = {
            "contract": contract,
            "wire_image_sha256": g.image_sha256,
            "actor_channel_order": "RGB" if contract == "rgb_v1" else "BGR",
            "actor_image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "actor_channel_means": image.mean(axis=(0, 1, 2)).tolist(),
        }
        return image

    def depth(batch_size):
        if batch_size != 1 or "depth" in request.files:
            raise ValueError("X monocular diagnostic accepts RGB-only batch one")
        token = request.form["monocular_depth_transaction_token"]
        frame = int(request.form["monocular_depth_frame_index"])
        response = requests.post(os.environ["XNAVDP_MONOCULAR_DEPTH_URL"], data={
            "expected_image_sha256": g.image_sha256,
            "monocular_depth_transaction_token": token,
            "expected_frame_index": str(frame),
        }, timeout=60)
        response.raise_for_status()
        payload = response.json()
        validate_monocular_depth_transaction(payload, expected_token=token,
            expected_image_sha256=g.image_sha256, expected_frame_index=frame)
        values, metadata = decode_monocular_depth_payload(
            payload, expected_image_sha256=g.image_sha256)
        g.depth_receipt = dict(metadata, depth_source="monocular_sidecar",
                               metric_depth_sensor_consumed=False)
        return values[None, :, :, None].astype(np.float32, copy=False)

    server._decode_rgb, server._decode_depth = rgb, depth

    @server.app.after_request
    def receipt(response):
        if request.path in ("/pointgoal_step", "/memory_replay_step") and response.status_code == 200:
            result = response.get_json()
            result["xnavdp_rgb_receipt"] = g.rgb_receipt
            result["actor_rgb_channel_order"] = g.rgb_receipt["actor_channel_order"]
            rgb_log = Path(os.environ["XNAVDP_DIAGNOSTIC_LOG"]).with_name("xnavdp_rgb_observations.jsonl")
            with rgb_log.open("a") as stream:
                stream.write(json.dumps(dict(g.rgb_receipt, endpoint=request.path,
                    history_frame_count=result.get("history_frame_count"))) + "\n")
            if request.path == "/memory_replay_step":
                response.set_data(json.dumps(result, allow_nan=False))
                return response
            goal = json.loads(request.form["goal_data"])
            point = np.array([[goal["goal_x"][0], goal["goal_y"][0], 0.]])
            processed = server._navigator.process_pointgoal(point)
            result.update(depth_source="monocular_sidecar",
                          metric_depth_sensor_consumed=False,
                          monocular_depth_receipt=g.depth_receipt,
                          xnavdp_input_diagnostic={
                              "goal_before_processing": point.tolist(),
                              "goal_after_processing": processed.tolist(),
                              "image_sha256": g.image_sha256,
                              "sensor_depth_uploaded": False,
                              "rtc_pose_source": "ideal simulator odometry; no goal/path GT",
                              "rgb_contract": g.rgb_receipt["contract"],
                              "actor_rgb_sha256": g.rgb_receipt["actor_image_sha256"],
                          })
            response.set_data(json.dumps(result, allow_nan=False))
            with Path(os.environ["XNAVDP_DIAGNOSTIC_LOG"]).open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
        return response


if __name__ == "__main__":
    from MemNavData import xnavdp_revisit_server as server
    install(server)
    server.main()
