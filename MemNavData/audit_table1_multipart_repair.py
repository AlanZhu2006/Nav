"""Reproduce both failed-cell JPEG digests without models or navigation."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from flask import Flask, request
from PIL import Image
from requests import Request
from werkzeug.sansio.multipart import MultipartDecoder

from MemNavData.multipart_crlf_repair import install


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def audit(original, failed, output):
    plan = json.loads((original / "plan.json").read_text())
    app = Flask(__name__)

    @app.post("/upload")
    def upload():
        return {key: value.read().hex() for key, value in request.files.items()}

    client = app.test_client()
    unpatched = MultipartDecoder._last_partial_boundary_index
    while getattr(unpatched, "_memnav_crlf_pair_repair", False):
        unpatched = unpatched.__wrapped__
    results = []
    try:
        for index in (110, 158):
            receipt_path = failed / "evaluation" / f"task_{index:03d}" / "archive_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            archive = Path(receipt["archive"])
            assert digest(archive.read_bytes()) == receipt["archive_sha256"]
            with tarfile.open(archive, "r:gz") as stream:
                def read(name):
                    return stream.extractfile(name).read()

                if index == 110:
                    # Diagnostic reconstruction only: the archived file contains
                    # the observed extraneous CR. Production never strips bytes.
                    received = read("task/buffer/ep_0004/348.jpg")
                    assert received.endswith(b"\xff\xd9\r") and len(received) == 65191
                    image = received[:-1]
                    fields = {"graph_rescue": "0", "materialize_monocular_depth": "1"}
                    files = {"image": ("image.jpg", image)}
                    target_key, target, captured_digest = "image", image, digest(received)
                else:
                    boundary_rows = [json.loads(line) for line in read(
                        "task/evaluation/revisit/native/memory_http_boundary.jsonl").splitlines()]
                    frame = boundary_rows[-1]["frame_idx"]
                    image = read(f"task/buffer/ep_0004/{frame}.jpg")
                    cell = plan["cells"][index]
                    folder = Path(cell["benchmark"]) / cell["scene"] / cell["episode"]
                    payload = json.loads((folder / "role_pairs.json").read_text())
                    query = next(q for p in payload["pairs"] for q in p["queries"]
                                 if q["analysis_role"] == "revisit")
                    goal = (folder / query["goal_rgb"]).read_bytes()
                    assert digest(goal) == query["goal_rgb_sha256"]
                    http = [json.loads(line) for line in read("task/image_controller_http.jsonl").splitlines()][-1]
                    assert digest(image) == http["image_jpeg_sha256"]
                    fields = {"diffusion_seed": str(http["diffusion_seed"])}
                    files = {"image": ("image.jpg", image), "goal": ("goal.jpg", goal)}
                    target_key, target, captured_digest = "goal", goal, http["goal_jpeg_sha256"]
                prepared = Request("POST", "http://localhost/upload", files=files, data=fields).prepare()
                MultipartDecoder._last_partial_boundary_index = unpatched
                before = client.post("/upload", data=prepared.body,
                                     content_type=prepared.headers["Content-Type"])
                old = bytes.fromhex(before.json[target_key])
                assert digest(old) == captured_digest and old == target + b"\r"
                np.testing.assert_array_equal(decode(old), decode(target))
                install()
                after = client.post("/upload", data=prepared.body,
                                    content_type=prepared.headers["Content-Type"])
                for key, (_, value) in files.items():
                    assert bytes.fromhex(after.json[key]) == value
                results.append(dict(index=index, archive=str(archive),
                    archive_sha256=receipt["archive_sha256"], request_body_bytes=len(prepared.body),
                    target_key=target_key, expected_bytes=len(target),
                    expected_sha256=digest(target), old_received_sha256=digest(old),
                    captured_failed_sha256=captured_digest,
                    repaired_sha256=digest(bytes.fromhex(after.json[target_key])),
                    extra_byte_hex="0d", decoded_rgb_identical=True,
                    exact_bytes_after_repair=True))
    finally:
        MultipartDecoder._last_partial_boundary_index = unpatched
    result = dict(verified=True, werkzeug_version=importlib.metadata.version("werkzeug"),
                  model_inference=False, navigation_run=False, cases=results)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.original, args.failed, args.output)
