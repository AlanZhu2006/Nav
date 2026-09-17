"""Task-private RGB ImageGoal server for Table-I ViNT and NoMaD.

No fallback planner, no depth input, and no implicit reset/replay sampling.
Endpoint aliases preserve evaluator transport, not NavDP model identity.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.image_controller_policy import load_agent, predict


def create_app(controller, agent, transform, *, sample_num=8, audit_path=None):
    from MemNavData.multipart_crlf_repair import install
    install()
    app = Flask(__name__)
    state = dict(observations=0, resets=0, last_plan=None)

    def rgb(key):
        payload = request.files[key].read()
        value = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))[None]
        return payload, value

    def receipt(result, image=None, goal=None):
        audit = dict(contract="rgb_v1", endpoint=request.path, status_code=200,
            controller=controller, image_calls=[], image_jpeg_sha256=None,
            goal_jpeg_sha256=None, controller_depth_source="none",
            received_file_fields=sorted(request.files), received_form_fields=sorted(request.form),
            observations=state["observations"], resets=state["resets"])
        for key, data in (("image", image), ("goal", goal)):
            if data is None:
                continue
            encoded, pixels = data
            audit[f"{key}_jpeg_sha256"] = hashlib.sha256(encoded).hexdigest()
            audit["image_calls"].append(dict(kind=key, input_shape=list(pixels.shape),
                input_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), color="RGB"))
        result.update(algo=controller, controller=controller, depth_source="none",
            controller_depth_source="none", metric_depth_sensor_consumed=False,
            execution_input_audit=audit)
        if audit_path:
            with Path(audit_path).open("a") as stream:
                stream.write(json.dumps(dict(audit, diffusion_seed=result.get("diffusion_seed"),
                    controller_seed_consumed=result.get("controller_seed_consumed"),
                    goal_mask=result.get("goal_mask")), allow_nan=False) + "\n")
        return jsonify(result)

    @app.get("/healthz")
    def health():
        return jsonify(ok=True, controller=controller)

    @app.post("/navigator_reset")
    def reset():
        payload = request.get_json(force=True)
        if int(payload["batch_size"]) != 1:
            raise ValueError("Only one paired environment is supported")
        agent.reset(1)
        state.update(observations=0, resets=state["resets"] + 1, last_plan=None)
        return receipt(dict(reset=True, diffusion_sampled=False))

    @app.post("/navigator_reset_env")
    def reset_env():
        if int(request.get_json(force=True)["env_id"]) != 0:
            raise ValueError("Only environment zero exists")
        agent.reset_env(0)
        state.update(observations=0, last_plan=None)
        return receipt(dict(reset=True, diffusion_sampled=False))

    @app.post("/memory_replay_step")
    @app.post("/observation_step")
    def replay():
        if set(request.files) != {"image"}:
            raise ValueError("Replay requires RGB only")
        image = rgb("image")
        agent.observe(image[1])
        state["observations"] += 1
        return receipt(dict(observed=True, diffusion_sampled=False,
            memory_size=agent.memory_size + 1,
            queue_lengths=[len(q) for q in agent.memory_queue],
            queue_padding_strategy="repeat_first_observation"), image=image)

    @app.post("/imagegoal_step")
    def plan():
        if set(request.files) != {"image", "goal"} or set(request.form) != {"diffusion_seed"}:
            raise ValueError("Controller accepts only current RGB, goal RGB and paired seed")
        image, goal = rgb("image"), rgb("goal")
        result = predict(agent, transform, controller, image[1], goal[1],
                         seed=int(request.form["diffusion_seed"]), sample_num=sample_num)
        state["observations"] += 1
        return receipt(result, image=image, goal=goal)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("vint", "nomad"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    agent, transform = load_agent(args.controller, ROOT, args.checkpoint, args.device)
    create_app(args.controller, agent, transform,
        audit_path=os.environ.get("IMAGE_CONTROLLER_AUDIT")).run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
