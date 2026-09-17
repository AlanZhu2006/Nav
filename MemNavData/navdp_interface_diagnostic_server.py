#!/usr/bin/env python3
"""Private NavDP server with read-only pointgoal / trajectory-mask receipts.

No source-network edits, extra inference, weight changes, or RNG draws. The
original predict function is compiled with two observation calls surrounding
its existing short-trajectory mask. Returned trajectories stay untouched.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap


def instrument_mask(function, capture):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))

    class Insert(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            target = node.targets[0]
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "all_trajectory"
                    and "trajectory_length" in ast.unparse(target)):
                self.count += 1
                before = ast.parse('_interface_capture("pre_mask", all_trajectory, critic_values)').body[0]
                after = ast.parse('_interface_capture("post_mask", all_trajectory, critic_values)').body[0]
                return [before, node, after]
            return node

    insert = Insert()
    insert.visit(tree)
    if insert.count != 1:
        raise RuntimeError(f"Expected one frozen mask in {function.__name__}, found {insert.count}")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _interface_capture=capture)
    exec(compile(tree, function.__code__.co_filename + ":observed", "exec"), namespace)
    return namespace[function.__name__]


def main():
    from flask import g, has_request_context, request
    from policy_agent import NavDP_Agent
    from policy_network import NavDP_Policy

    output = Path(os.environ["NAVDP_INTERFACE_LOG"])
    output.parent.mkdir(parents=True, exist_ok=True)

    def capture(stage, trajectories, values):
        if has_request_context():
            g.interface[stage] = trajectories.detach().cpu().numpy().tolist()
            if stage == "pre_mask":
                g.interface["critic_values"] = values.detach().cpu().numpy().tolist()

    original_point = NavDP_Agent.process_pointgoal

    def point(self, goals):
        processed = original_point(self, goals)
        if has_request_context():
            g.interface["pointgoal_before_clip"] = goals.tolist()
            g.interface["pointgoal_after_clip"] = processed.tolist()
        return processed

    NavDP_Agent.process_pointgoal = point
    for name in ("predict_ip_action", "predict_imagegoal_action"):
        setattr(NavDP_Policy, name, instrument_mask(getattr(NavDP_Policy, name), capture))

    import navdp_server as server

    @server.app.before_request
    def begin():
        g.interface = {"endpoint": request.path,
                       "diffusion_seed": request.form.get("diffusion_seed")}
        for field in ("image", "image_goal", "goal"):
            if field in request.files:
                stream = request.files[field].stream
                offset = stream.tell()
                g.interface[field + "_sha256"] = hashlib.sha256(stream.read()).hexdigest()
                stream.seek(offset)

    @server.app.after_request
    def finish(response):
        payload = response.get_json(silent=True)
        g.interface["status_code"] = response.status_code
        if isinstance(payload, dict):
            for key in ("trajectory", "all_trajectory", "all_values", "navdp_stop_evidence",
                        "navdp_critic_max", "queue_lengths", "diffusion_sampled"):
                if key in payload:
                    g.interface[key] = payload[key]
            if "pre_mask" in g.interface:
                payload["navdp_interface_diagnostic"] = g.interface
                response.set_data(server.app.json.dumps(payload))
        with output.open("a") as stream:
            stream.write(json.dumps(g.interface, allow_nan=False) + "\n")
        return response

    server.app.run(host="127.0.0.1", port=server.args.port)


if __name__ == "__main__":
    main()
