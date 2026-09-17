"""Private ViNT RGB repair for the corrected Table-I evaluation.

The existing server decodes RGB JPEGs and swaps them to BGR. ViNT's PIL-based
preprocessor expects RGB. Undo that swap at the agent boundary for both the
goal and every observation, including context-only replay. Weights, resizing,
normalization, distance prediction and the upstream stop mask are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def rgb_from_server_bgr(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("ViNT agent expects [batch, height, width, 3]")
    return images[..., ::-1].copy()


def install(agent_class, record=lambda _: None):
    original_goal = agent_class.step_imagegoal
    original_observe = agent_class.observe
    original_nogoal = agent_class.step_nogoal

    def convert(kind, images):
        rgb = rgb_from_server_bgr(images)
        record(dict(kind=kind, shape=list(rgb.shape),
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        return rgb

    def imagegoal(self, goal, image):
        return original_goal(self, convert("goal", goal), convert("observation", image))

    def observe(self, image):
        return original_observe(self, convert("replay_observation", image))

    def nogoal(self, image):
        return original_nogoal(self, convert("nogoal_observation", image))

    agent_class.step_imagegoal = imagegoal
    agent_class.observe = observe
    agent_class.step_nogoal = nogoal


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "NavDP/baselines/vint"))
    import vint_server as server
    from flask import g, request
    from vint_agent import ViNTAgent

    @server.app.before_request
    def begin():
        g.vint_rgb_audit = dict(contract="rgb_v1", endpoint=request.path, input_calls=[])

    install(ViNTAgent, lambda row: g.vint_rgb_audit["input_calls"].append(row))

    @server.app.after_request
    def finish(response):
        audit = dict(g.vint_rgb_audit, status_code=response.status_code)
        data = response.get_json(silent=True)
        if isinstance(data, dict):
            data["vint_rgb_input_audit"] = audit
            response.set_data(server.app.json.dumps(data))
        path = os.environ.get("VINT_RGB_INPUT_AUDIT_LOG")
        if path:
            with Path(path).open("a") as stream:
                stream.write(json.dumps(audit, allow_nan=False) + "\n")
        return response

    server.app.run(host="127.0.0.1", port=server.args.port, threaded=False)


if __name__ == "__main__":
    main()
