"""Private base-NavDP observation audit; no production defaults are changed.

The contract is fixed at navigator_reset, never selected from model outcomes.
legacy_bgr preserves the existing PIL-JPEG integration. rgb_v1 compensates
the server's channel swap at the actual image preprocessor, for both current
observations and goals. Original JPEGs/depth transactions stay untouched.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CONTRACTS = ("legacy_bgr", "rgb_v1")


def actor_input(server_decoded_bgr, contract):
    if contract not in CONTRACTS:
        raise ValueError(contract)
    data = np.asarray(server_decoded_bgr)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("Expected [B,H,W,3] decoded image")
    return data if contract == "legacy_bgr" else data[..., ::-1].copy()


def install(server, agent_class, output):
    from flask import g, has_request_context, request
    original = agent_class.process_image
    contract = "legacy_bgr"

    def process_image(self, images):
        actual = actor_input(images, contract)
        result = original(self, actual)
        if has_request_context():
            g.execution_input_audit["image_calls"].append({
                "input_shape": list(actual.shape),
                "input_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "input_mean_channels": actual.mean(axis=(0, 1, 2)).tolist(),
                "encoded_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            })
        return result

    agent_class.process_image = process_image

    @server.app.before_request
    def begin():
        nonlocal contract
        if request.path == "/navigator_reset":
            chosen = (request.get_json(silent=True) or {}).get(
                "audit_actor_rgb_contract", "legacy_bgr")
            if chosen not in CONTRACTS:
                return {"error": "invalid audit RGB contract"}, 400
            contract = chosen
        g.execution_input_audit = {
            "endpoint": request.path, "contract": contract,
            "actor_channel_order": "BGR" if contract == "legacy_bgr" else "RGB",
            "image_calls": [],
        }
        for name in ("image", "goal", "image_goal"):
            if name in request.files:
                stream = request.files[name].stream
                offset = stream.tell()
                g.execution_input_audit[name + "_jpeg_sha256"] = hashlib.sha256(stream.read()).hexdigest()
                stream.seek(offset)

    @server.app.after_request
    def finish(response):
        receipt = getattr(g, "execution_input_audit", None)
        if receipt is None:
            return response
        receipt["status_code"] = response.status_code
        payload = response.get_json(silent=True)
        if isinstance(payload, dict):
            for name in ("diffusion_seed", "queue_lengths", "critic_fallback_applied"):
                if name in payload:
                    receipt[name] = payload[name]
            payload["execution_input_audit"] = receipt
            response.set_data(server.app.json.dumps(payload))
        if output is not None:
            with Path(output).open("a") as handle:
                handle.write(json.dumps(receipt, allow_nan=False) + "\n")
        return response


def main():
    import navdp_server as server
    from policy_agent import NavDP_Agent
    install(server, NavDP_Agent, os.environ["NAVDP_EXECUTION_INPUT_LOG"])
    server.app.run(host="127.0.0.1", port=server.args.port)


if __name__ == "__main__":
    main()
