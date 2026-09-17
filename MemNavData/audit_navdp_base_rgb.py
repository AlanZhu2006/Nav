"""Exercise actual base NavDP JPEG encoder, route decoder and preprocessor.

No GPU, policy inference or network. Only the actual client's HTTP call is
intercepted; AST extraction executes original decoding/preprocessing code.
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

import cv2
import numpy as np
from PIL import Image

from MemNavData.navdp_execution_audit_server import actor_input

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "NavDP/baselines/navdp/navdp_server.py"
AGENT = ROOT / "NavDP/baselines/navdp/policy_agent.py"
CLIENT = ROOT / "NavDP/utils_tasks/client_utils.py"
VINT = ROOT / "NavDP/baselines/vint"


def pil_wire(rgb):
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=95)
    return out.getvalue()


def original_client_wire(rgb):
    spec = importlib.util.spec_from_file_location("base_rgb_client", CLIENT)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    captured = []
    def post(url, **kwargs):
        captured.append(kwargs["files"])
        return SimpleNamespace(text=json.dumps(dict(trajectory=[], all_trajectory=[], all_values=[])))
    with patch.object(client.requests, "post", side_effect=post):
        client.imagegoal_step(rgb[None], rgb[None], np.ones((1, *rgb.shape[:2], 1)))
    return captured[0]


def route_decode(wire, function_name="navdp_step_image", variable="image"):
    tree = ast.parse(SERVER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == function_name)
    body = [n for n in function.body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == variable]
    file = SimpleNamespace(stream=io.BytesIO(wire))
    namespace = dict(np=np, cv2=cv2, Image=Image, io=io, batch_size=1,
                     image_bytes=wire, image_file=file, goal_file=file)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace[variable]


def original_preprocess(images):
    tree = ast.parse(AGENT.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NavDP_Agent")
    function = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "process_image")
    namespace = dict(np=np, cv2=cv2)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(AGENT), "exec"), namespace)
    return namespace["process_image"](SimpleNamespace(image_size=224), images)


def analyze(rgb):
    files = original_client_wire(rgb)
    wire = pil_wire(rgb)
    decoded = np.asarray(Image.open(io.BytesIO(wire)).convert("RGB"))[None]
    official = route_decode(files["image"][1])
    legacy = route_decode(wire)
    fixed = actor_input(legacy, "rgb_v1")
    np.testing.assert_array_equal(legacy, decoded[..., ::-1])
    np.testing.assert_array_equal(fixed, decoded)
    goal = route_decode(wire, variable="goal")
    history = route_decode(wire, "navdp_memory_replay_step")
    np.testing.assert_array_equal(goal, legacy)
    np.testing.assert_array_equal(history, legacy)
    return {
        "raw_channels": rgb.mean(axis=(0, 1)).tolist(),
        "official_actor_channels": official.mean(axis=(0, 1, 2)).tolist(),
        "habitat_legacy_actor_channels": legacy.mean(axis=(0, 1, 2)).tolist(),
        "fixed_actor_channels": fixed.mean(axis=(0, 1, 2)).tolist(),
        "preprocessed_official_channels": original_preprocess(official).mean(axis=(0, 1, 2)).tolist(),
        "preprocessed_legacy_channels": original_preprocess(legacy).mean(axis=(0, 1, 2)).tolist(),
        "preprocessed_fixed_channels": original_preprocess(fixed).mean(axis=(0, 1, 2)).tolist(),
        "current_goal_history_contract_equal": True,
        "fixed_equals_pil_rgb": bool(np.array_equal(fixed, decoded)),
    }


def analyze_vint(rgb):
    """Exercise ViNT's actual route assignments and PIL/tensor preprocessing.

    No checkpoint or inference. This does not infer which historical bundle
    a published table used, nor the effect of channel order on its SR.
    """
    import torch
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    from typing import List
    server = ast.parse((VINT / "vint_server.py").read_text())
    agent = ast.parse((VINT / "vint_agent.py").read_text())
    base = ast.parse((VINT / "base_agent.py").read_text())
    transform = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "transform_images")
    process = next(n for n in ast.walk(base) if isinstance(n, ast.FunctionDef) and n.name == "process_image")
    env = dict(torch=torch, transforms=transforms, TF=TF, PILImage=Image, Image=Image,
               List=List, IMAGE_ASPECT_RATIO=4/3)
    exec(compile(ast.Module(body=[transform, process], type_ignores=[]), str(VINT), "exec"), env)
    wire = pil_wire(rgb)
    output = []
    for endpoint, variable in (("vint_step_imagegoal", "image"), ("vint_step_imagegoal", "goal"),
                               ("vint_observation_step", "image")):
        route = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == endpoint)
        assignments = [n for n in route.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == variable]
        file = SimpleNamespace(stream=io.BytesIO(wire))
        scope = dict(np=np, cv2=cv2, Image=Image, image_file=file, goal_file=file,
                     vint_navigator=SimpleNamespace(batch_size=1))
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(VINT), "exec"), scope)
        decoded = scope[variable][0]
        pil = env["process_image"](None, decoded)
        encoded = env["transform_images"](pil, [160, 120])
        mean = encoded.mean(dim=(0, 2, 3)).numpy()
        pixel = (mean * np.array([.229, .224, .225]) + np.array([.485, .456, .406])) * 255
        expected_bgr = np.asarray(Image.open(io.BytesIO(wire)).convert("RGB"))[..., ::-1]
        np.testing.assert_array_equal(decoded, expected_bgr)
        np.testing.assert_allclose(pixel, expected_bgr.mean(axis=(0, 1)), atol=.02)
        output.append(dict(endpoint=endpoint, variable=variable,
                           reconstructed_actor_channels=pixel.tolist(),
                           legacy_wire_becomes_bgr=True))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [analyze(np.full((64, 64, 3), value, np.uint8))
             for value in ([220, 60, 20], [20, 60, 220], [30, 210, 70])]
    sources = [SERVER, AGENT, CLIENT, Path(__file__),
               ROOT / "MemNavData/navdp_execution_audit_server.py",
               VINT / "vint_server.py", VINT / "vint_agent.py", VINT / "base_agent.py"]
    report = dict(verified=True, cases=cases,
                  vint_current_stack=analyze_vint(np.full((64, 64, 3), [220, 60, 20], np.uint8)),
                  scope="base and ViNT actor channel contracts; no SR or historical-table inference",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
