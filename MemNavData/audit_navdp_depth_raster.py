"""Read immutable sidecar transactions without advancing memory or inference.

This audits actual depth transport, not navigation effects. Only the existing
token lookup endpoint is used: no reset, image append, or uncached depth call.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import requests

from MemNavData.audit_navdp_base_rgb import AGENT, original_preprocess
from MemNavData.monocular_depth_runtime import (
    decode_monocular_depth_payload, validate_monocular_depth_transaction,
)


def depth_preprocess(depth):
    tree = ast.parse(AGENT.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NavDP_Agent")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_depth")
    scope = dict(np=np, cv2=cv2)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(AGENT), "exec"), scope)
    return scope["process_depth"](SimpleNamespace(image_size=224),
                                    depth[None, :, :, None].copy())[0, :, :, 0]


def analyze(payload, source_hw):
    depth, _ = decode_monocular_depth_payload(
        payload, expected_image_sha256=payload["image_sha256"])
    actor_depth = depth_preprocess(depth)
    height, width = source_hw
    rgb_support = original_preprocess(
        np.full((1, height, width, 3), 255, np.uint8))[0, :, :, 0] > .5
    padding_values = actor_depth[~rgb_support]
    return {
        "frame_index": payload["frame_index"], "depth_shape": list(depth.shape),
        "source_hw": list(source_hw), "actor_depth_shape": list(actor_depth.shape),
        "scale_hat": payload["scale_receipt"]["scale_hat"],
        "scale_clamped": payload["scale_receipt"]["scale_clamped"],
        "padding_pixels": int(padding_values.size),
        "nonzero_actor_depth_on_rgb_padding": int(np.count_nonzero(padding_values)),
        "padding_nonzero_fraction": float(np.count_nonzero(padding_values) / padding_values.size)
        if padding_values.size else 0.,
        "padding_depth_median_after_preprocess": float(np.median(padding_values))
        if padding_values.size else None,
        "nonzero_fraction_on_rgb_support": float(np.count_nonzero(actor_depth[rgb_support]) / rgb_support.sum()),
        "depth_png_sha256": payload["depth_png_sha256"],
        "transaction_token": payload["monocular_depth_transaction_token"],
        "image_sha256": payload["image_sha256"],
    }


def capture(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, seen, misses = [], set(), 0
    deadline = time.monotonic() + args.seconds
    while len(rows) < args.samples and time.monotonic() < deadline:
        progress = json.loads((args.bridge / "progress.json").read_text())
        if progress["stage"] != "evaluation":
            break
        receipt_path = args.bridge / "evaluation" / progress["scene"] / progress["arm"] / "navdp_http_receipts.jsonl"
        if not receipt_path.exists():
            time.sleep(.2)
            continue
        lines = receipt_path.read_text().splitlines()
        if not lines:
            time.sleep(.2)
            continue
        try:
            receipt = json.loads(lines[-1])
        except json.JSONDecodeError:  # writer may still be finishing the last line
            time.sleep(.2)
            continue
        evidence = receipt.get("monocular_depth_receipt") or {}
        token = evidence.get("monocular_depth_transaction_token")
        if not token or token in seen:
            time.sleep(.2)
            continue
        # This exact token is already materialized by the rollout. Server-side
        # validation either returns that immutable payload or HTTP 409.
        response = requests.post(args.endpoint.rstrip("/") + "/monocular_depth_query", data={
            "monocular_depth_transaction_token": token,
            "expected_image_sha256": evidence["image_sha256"],
            "expected_frame_index": evidence["frame_index"],
        }, timeout=3)
        if response.status_code == 409:
            misses += 1
            seen.add(token)
            continue
        response.raise_for_status()
        payload = response.json()
        validate_monocular_depth_transaction(payload, expected_token=token,
            expected_image_sha256=evidence["image_sha256"],
            expected_frame_index=evidence["frame_index"])
        assert payload["depth_png_sha256"] == evidence["depth_png_sha256"]
        assert payload["metric_depth_sensor_consumed"] is False
        assert payload["depth_prediction_cache_hit"] is True
        row = dict(analyze(payload, tuple(args.source_hw)), scene=progress["scene"], arm=progress["arm"],
                   source_receipt=str(receipt_path), immutable_transaction_read=True)
        number = len(rows)
        (out / f"payload_{number:02}.json").write_text(json.dumps(payload) + "\n")
        rows.append(row)
        seen.add(token)
        print(json.dumps(row), flush=True)
    result = dict(records=rows, samples=len(rows), expired_transaction_misses=misses,
                  scope="actual cached payloads; raster values only, NOT SR or navigation effect",
                  no_model_or_memory_advance=True,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (AGENT, Path(__file__))})
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--source-hw", type=int, nargs=2, default=[270, 480])
    capture(parser.parse_args())
