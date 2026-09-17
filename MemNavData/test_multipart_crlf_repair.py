"""CPU-only regressions for Table-I JPEG transport failures 110 and 158."""
import io

import pytest
from flask import Flask, request
from requests import Request
from werkzeug.formparser import MultiPartParser
from werkzeug.sansio import multipart

from MemNavData.multipart_crlf_repair import install


def prepared(files, data):
    return Request("POST", "http://localhost/upload", files=files, data=data).prepare()


def app():
    application = Flask(__name__)

    @application.post("/upload")
    def upload():
        return {key: value.read().hex() for key, value in request.files.items()}

    return application


@pytest.fixture
def repaired(monkeypatch):
    original = multipart.MultipartDecoder._last_partial_boundary_index
    monkeypatch.setattr(multipart.MultipartDecoder, "_last_partial_boundary_index", original)
    install()


def test_reproduce_110_before_fix(monkeypatch):
    original = multipart.MultipartDecoder._last_partial_boundary_index
    while getattr(original, "_memnav_crlf_pair_repair", False):
        original = original.__wrapped__
    monkeypatch.setattr(multipart.MultipartDecoder, "_last_partial_boundary_index", original)
    image = b"\xff\xd8" + b"x" * 65186 + b"\xff\xd9"
    req = prepared({"image": ("image.jpg", image)},
                   {"graph_rescue": "0", "materialize_monocular_depth": "1"})
    assert len(req.body) == 65539
    result = app().test_client().post("/upload", data=req.body,
                                    content_type=req.headers["Content-Type"])
    assert bytes.fromhex(result.json["image"]) == image + b"\r"


@pytest.mark.parametrize("length", range(65120, 65240))
def test_exact_bytes_around_64k_boundary(repaired, length):
    image = b"\xff\xd8" + b"x" * length + b"\xff\xd9"
    req = prepared({"image": ("image.jpg", image)},
                   {"graph_rescue": "0", "materialize_monocular_depth": "1"})
    result = app().test_client().post("/upload", data=req.body,
                                    content_type=req.headers["Content-Type"])
    assert bytes.fromhex(result.json["image"]) == image


@pytest.mark.parametrize("tail", [b"", b"\r", b"\n", b"\r\n", b"\r\n\r\n"])
def test_genuine_payload_line_endings_not_stripped(repaired, tail):
    image = b"\xff\xd8" + b"x" * 65186 + b"\xff\xd9" + tail
    req = prepared({"image": ("image.jpg", image)},
                   {"graph_rescue": "0", "materialize_monocular_depth": "1"})
    result = app().test_client().post("/upload", data=req.body,
                                    content_type=req.headers["Content-Type"])
    assert bytes.fromhex(result.json["image"]) == image


@pytest.mark.parametrize("offset", range(-5, 6))
def test_two_files_preserve_goal_at_closing_boundary(repaired, offset):
    # The real cell-158 goal is 49,039 bytes. No model is needed to reproduce
    # its transport layout; synthesize same-length bytes ending in JPEG EOI.
    goal = b"\xff\xd8" + b"g" * 49035 + b"\xff\xd9"
    image = b"\xff\xd8" + b"i" * (16000 + offset) + b"\xff\xd9"
    req = prepared({"image": ("image.jpg", image), "goal": ("goal.jpg", goal)},
                   {"diffusion_seed": "2026080610031"})
    # Move the chunk edge through the final delimiter independently of body
    # length. Both fields and every byte must survive intact.
    boundary = req.headers["Content-Type"].split("boundary=")[1].encode()
    for suffix in range(1, 40):
        parser = MultiPartParser(buffer_size=len(req.body) - suffix)
        form, files = parser.parse(io.BytesIO(req.body), boundary, len(req.body))
        assert form["diffusion_seed"] == "2026080610031"
        assert files["image"].read() == image
        assert files["goal"].read() == goal


def test_idempotent(repaired):
    before = multipart.MultipartDecoder._last_partial_boundary_index
    install()
    assert multipart.MultipartDecoder._last_partial_boundary_index is before
