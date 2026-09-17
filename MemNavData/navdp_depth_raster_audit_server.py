"""Private reset-bound depth readout experiment, retaining source receipts.

The sidecar's immutable PNG/cache remains in LingBot coordinates. This wrapper
converts only the returned dense observation, after the original transaction
validator and before the unchanged NavDP encoder. It does not advance LingBot.
"""
from dataclasses import asdict
import hashlib
import itertools
import os
from pathlib import Path

import numpy as np

from MemNavData.lingbot_depth_raster import lingbot_pad_raster, to_source_rgb_raster

CONTRACTS = ("legacy_square", "source_rgb")


def convert_observation(depth, receipt, source_hw, contract):
    if contract not in CONTRACTS:
        raise ValueError("Unknown depth raster contract")
    original = np.asarray(depth)
    converted, layout = original, None
    operation = "identity"
    if contract == "source_rgb" and receipt["depth_source"] == "monocular_sidecar":
        if receipt["scale_state"] == "raw_lingbot_metric_depth":
            if original.shape != (1, 518, 518, 1):
                raise ValueError("Active sidecar depth is not on the frozen LingBot pad-518 raster")
            raster = lingbot_pad_raster(source_hw)
            converted = to_source_rgb_raster(original[0, :, :, 0], raster)[None, :, :, None]
            layout, operation = asdict(raster), "inverse_lingbot_pad_to_source_rgb"
        elif receipt["scale_state"] in ("bootstrap_zero_depth", "frozen_scale_invalid_zero_depth"):
            if original.shape != (1, *source_hw, 1) or np.count_nonzero(original):
                raise ValueError("Bootstrap/invalid-scale observation must retain source-shaped zero depth")
            operation = "source_zero_unchanged"
        else:
            raise ValueError("Unsupported sidecar scale state")
    evidence = {
        "contract": contract, "operation": operation, "layout": layout,
        "source_rgb_hw": list(source_hw),
        "input_shape": list(original.shape), "output_shape": list(converted.shape),
        "input_tensor_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
        "output_tensor_sha256": hashlib.sha256(converted.tobytes()).hexdigest(),
        "metric_scale_unchanged": True,
    }
    # depth_png_sha256/depth_shape still describe the ORIGINAL sidecar payload.
    # The new nested receipt identifies the dense readout actually given to NavDP.
    return converted, dict(receipt, navdp_depth_raster=evidence)


def install(server, output_dir=None):
    from flask import request
    original = server._resolve_observation_depth
    contract = "legacy_square"
    sequence = itertools.count()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

    @server.app.before_request
    def bind_contract():
        nonlocal contract
        if request.path == "/navigator_reset":
            chosen = (request.get_json(silent=True) or {}).get("audit_depth_raster_contract", "legacy_square")
            if chosen not in CONTRACTS:
                return {"error": "Unknown depth raster contract"}, 400
            contract = chosen

    @server.app.after_request
    def report_contract(response):
        payload = response.get_json(silent=True)
        if isinstance(payload, dict):
            payload["audit_depth_raster_contract"] = contract
            response.set_data(server.app.json.dumps(payload))
        return response

    def resolve(image_bytes, image_bgr, depth_file, batch_size, **kwargs):
        depth, receipt = original(image_bytes, image_bgr, depth_file, batch_size, **kwargs)
        if batch_size != 1 and contract == "source_rgb" and receipt["depth_source"] == "monocular_sidecar":
            raise ValueError("Monocular raster readout requires batch size one")
        converted, metadata = convert_observation(depth, receipt, tuple(image_bgr.shape[:2]), contract)
        if output_dir is not None:
            artifact = output_dir / f"{next(sequence):06d}.npz"
            with artifact.open("xb") as stream:
                np.savez_compressed(stream, producer_depth=depth, navdp_readout_depth=converted,
                                    source_rgb_hw=np.asarray(image_bgr.shape[:2]))
            metadata["navdp_depth_raster"]["artifact"] = str(artifact.resolve())
            metadata["navdp_depth_raster"]["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        return converted, metadata

    server._resolve_observation_depth = resolve


def main():
    from MemNavData.multipart_crlf_repair import install as install_transport
    install_transport()
    import navdp_server as server
    from policy_agent import NavDP_Agent
    from MemNavData.navdp_execution_audit_server import install as install_rgb
    install_rgb(server, NavDP_Agent, os.environ["NAVDP_EXECUTION_INPUT_LOG"])
    install(server, os.environ.get("NAVDP_DEPTH_RASTER_LOG_ROOT"))
    server.app.run(host="127.0.0.1", port=server.args.port)


if __name__ == "__main__":
    main()
