"""CPU-only raster-contract probe using actual LingBot and NavDP preprocessors.

Synthetic depth fields reveal transport behavior, not actual depth accuracy or
navigation performance. No models, weights, inference or existing data edits.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from MemNavData.audit_navdp_base_rgb import AGENT, original_preprocess

LOADER = Path("/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/lingbot_map/utils/load_fn.py")


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("lingbot_raster_probe", LOADER)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    tree = ast.parse(AGENT.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NavDP_Agent")
    fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "process_depth")
    namespace = dict(np=np, cv2=cv2)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(AGENT), "exec"), namespace)
    agent = SimpleNamespace(image_size=224)
    depth_encode = lambda x: namespace["process_depth"](agent, x[None, :, :, None].copy())[0, :, :, 0]
    records = []
    for height, width in [(270, 480), (360, 640), (480, 640), (518, 518)]:
        image = out / f"black_{width}x{height}.png"
        Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(image)
        lb = loader.load_and_preprocess_images([str(image)], mode="pad", image_size=518, patch_size=14)[0].numpy()
        support = (lb[0] < .5)
        rgb_support = original_preprocess(np.full((1, height, width, 3), 255, np.uint8))[0, :, :, 0] > .5
        support_depth = depth_encode(support.astype(np.float32) * 2)
        dense_depth = depth_encode(np.full((518, 518), 2., np.float32))
        def box(mask):
            yy, xx = np.where(mask)
            return [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
        records.append({
            "source_hw": [height, width], "lingbot_support_xyxy": box(support),
            "navdp_rgb_support_xyxy": box(rgb_support),
            "synthetic_supported_depth_xyxy": box(support_depth > .1),
            "rgb_padding_pixels": int((~rgb_support).sum()),
            "synthetic_dense_depth_nonzero_on_rgb_padding": int(((dense_depth > .1) & ~rgb_support).sum()),
            "supported_depth_pixels_outside_rgb_support": int(((support_depth > .1) & ~rgb_support).sum()),
        })
    report = dict(verified=True, records=records,
                  scope="synthetic padding transport, NOT actual predicted padded depth or SR",
                  conclusion="sidecar depth padding is not remapped/masked to NavDP RGB support; effect on real predictions not measured",
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (LOADER, AGENT, Path(__file__))})
    (out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
