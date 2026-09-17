#!/usr/bin/env python3
"""Exercise the published X client/server color chain without network or GPU.

Uses the actual client function with its HTTP call intercepted, and executes
the image-decoding assignments extracted from the published server. This
tests the full encoding convention, not just agreement between our wrappers.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_xnavdp_tracking import OFFICIAL
from MemNavData.xnavdp_mono_diagnostic_server import actor_rgb_from_decoded_rgb

EVAL = OFFICIAL / "baselines/x-navdp/eval"


def pil_wire(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def published_server_rgb(wire):
    tree = ast.parse((EVAL / "src/policy_server.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "navdp_step_xy")
    statements = [n for n in function.body if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "image"]
    assert len(statements) == 5
    namespace = dict(Image=Image, np=np, cv2=cv2, batch_size=1,
                     image_file=SimpleNamespace(stream=io.BytesIO(wire)))
    module = ast.Module(body=statements, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "published_X_RGB_decoder", "exec"), namespace)
    return namespace["image"]


def published_client_wire(rgb):
    spec = importlib.util.spec_from_file_location("x_rgb_audit_client", EVAL / "src/client_utils.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    captured = []

    def intercept(url, **kwargs):
        captured.append(kwargs["files"]["image"][1])
        return SimpleNamespace(raise_for_status=lambda: None,
                               text=json.dumps(dict(trajectory=[], all_trajectory=[], all_values=[])))

    with patch.object(client.requests, "post", side_effect=intercept):
        client.pointgoal_step(np.array([[-2.5, 0.]]), rgb[None],
                              np.ones((1, *rgb.shape[:2], 1), np.float32))
    assert len(captured) == 1
    return captured[0]


def analyze(rgb, name):
    official = published_server_rgb(published_client_wire(rgb))[0]
    wire = pil_wire(rgb)
    legacy = published_server_rgb(wire)
    decoded = np.asarray(Image.open(io.BytesIO(wire)).convert("RGB"))
    corrected = actor_rgb_from_decoded_rgb(decoded[None], "rgb_v1")[0]
    assert np.array_equal(corrected, decoded)
    assert np.array_equal(legacy[0], decoded[..., ::-1])
    mae = lambda a, b: float(np.abs(a.astype(float)-b.astype(float)).mean())
    return dict(name=name, raw_mean_rgb=rgb.mean(axis=(0, 1)).tolist(),
        official_actor_mean=official.mean(axis=(0, 1)).tolist(),
        legacy_actor_mean=legacy[0].mean(axis=(0, 1)).tolist(),
        corrected_actor_mean=corrected.mean(axis=(0, 1)).tolist(),
        official_vs_raw_mae_255=mae(official, rgb),
        legacy_vs_decoded_rgb_mae_255=mae(legacy[0], decoded),
        corrected_vs_decoded_rgb_mae_255=mae(corrected, decoded),
        corrected_vs_official_mae_255=mae(corrected, official),
        caveat="JPEG coding differs; channel semantics, not bit-identical JPEG transport")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    color = np.empty((64, 64, 3), np.uint8)
    color[:] = [220, 60, 20]
    cases = [analyze(color, "red_dominant_synthetic")]
    assert cases[0]["official_vs_raw_mae_255"] < 2
    assert cases[0]["legacy_vs_decoded_rgb_mae_255"] > 100
    old = ROOT / ".diagnostics/habitat_physics_executor_20260908/xnavdp_stack_v2"
    for history in sorted(old.glob("history_*")):
        image = history / "evaluation/cec_x__xmpc/first_query_rgb.png"
        cases.append(analyze(np.asarray(Image.open(image).convert("RGB")), history.name))
    sources = [EVAL / "src/client_utils.py", EVAL / "src/policy_server.py",
               EVAL.parent / "src/environment/tasks/observation_utils.py",
               ROOT / "MemNavData/xnavdp_revisit_server.py",
               ROOT / "MemNavData/xnavdp_mono_diagnostic_server.py"]
    result = dict(passed=True, cases=cases,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  confirmed_bug="normal PIL RGB JPEG + published server conversion delivers BGR to X actor",
                  correction="keep original causal JPEG bytes; restore actor RGB after decoding",
                  scope="read-only RGB contract probe, not an SR result or full X reproduction")
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=True, cases=len(cases), result=str(out / "summary.json"))))


if __name__ == "__main__":
    main()
