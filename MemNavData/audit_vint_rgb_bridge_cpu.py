"""Actual frozen ViNT parity check: correct RGB vs repaired server BGR.

CPU-only model inference. This is not a Habitat rollout or an SR result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NavDP/baselines/vint"))
from vint_agent import ViNTAgent
from MemNavData.vint_rgb_input_bridge import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.perf_counter()
    agent = ViNTAgent(np.eye(3), str(args.checkpoint),
        str(ROOT / "NavDP/baselines/vint/configs/vint.yaml"),
        str(ROOT / "NavDP/baselines/vint/configs/robot_config.yaml"), device="cpu")
    images = [np.asarray(Image.open(args.history / f"{i:06d}.jpg").convert("RGB"))[None]
              for i in range(7)]
    goal = np.asarray(Image.open(args.goal).convert("RGB"))[None]
    agent.reset(1)
    for image in images[:-1]:
        agent.observe(image)
    expected = [x.detach().clone() for x in agent.step_imagegoal(goal, images[-1])]
    expected_context = [np.asarray(x).copy() for x in agent.memory_queue[0]]
    records = []
    install(ViNTAgent, records.append)
    agent.reset(1)
    for image in images[:-1]:
        agent.observe(image[..., ::-1].copy())
    actual = agent.step_imagegoal(goal[..., ::-1].copy(), images[-1][..., ::-1].copy())
    errors = []
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        errors.append(float((x - y).abs().max()))
    for actual_context, expected_image in zip(agent.memory_queue[0], expected_context):
        np.testing.assert_array_equal(actual_context, expected_image)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output = dict(verified=True, device="cpu", scope="actual ViNT input/model parity; not closed-loop SR",
        checkpoint_sha256=checkpoint_sha, maximum_absolute_errors=errors,
        context_size=len(expected_context), inputs=records,
        model_and_parity_wall_seconds=time.perf_counter() - started,
        weights_stop_mask_resizing_unchanged=True,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "MemNavData/vint_rgb_input_bridge.py",
                      ROOT / "NavDP/baselines/vint/vint_agent.py",
                      ROOT / "NavDP/baselines/vint/vint_server.py")})
    (args.out / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
