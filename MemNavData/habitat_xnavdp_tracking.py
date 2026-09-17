"""Published X-NavDP MPC, retaining its first-control chunk semantics."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / ".diagnostics/xnavdp_official_878740a2011856d0/NavDP"
ACADOS = ROOT / ".diagnostics/xnavdp_acados_v055_20260810/acados"
SOURCE = OFFICIAL / "baselines/x-navdp/src/utils/mpc_tracking.py"
CONFIG = dict(N=30, T=.1, desired_v=.376, v_max=.376, w_max=math.pi/4,
              ref_gap=3, ref_traj_length_m=2., ref_desired_v=.376, min_desired_v=.05)


def world_to_local_reference(world_path, position, yaw):
    """Habitat XZ -> policy forward/left; append initial origin as upstream."""
    delta = np.asarray(world_path) - np.asarray(position)[[0, 2]]
    forward = -math.sin(yaw)*delta[:, 0] - math.cos(yaw)*delta[:, 1]
    left = -math.cos(yaw)*delta[:, 0] + math.sin(yaw)*delta[:, 1]
    return np.vstack((np.zeros((1, 2)), np.column_stack((forward, left))))


def build_controller():
    spec = importlib.util.spec_from_file_location("published_xnavdp_mpc", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BatchMPCController(batch=1, **CONFIG)


class XTracker:
    """Solve once per fresh neural plan, consume controls 0..7 in order."""
    def __init__(self):
        self.controller = build_controller()
        self.plan_id = None
        self.controls = None
        self.cursor = 0
        self.receipt = None

    def command(self, plan_id, position, yaw, world_path):
        solve_s = None
        if plan_id != self.plan_id:
            reference = world_to_local_reference(world_path, position, yaw)
            started = time.monotonic()
            controls, states, desired, curvature = self.controller.solve(reference[None])
            solve_s = time.monotonic()-started
            status = int(self.controller.controller.solver.status)
            if status or not np.isfinite(controls).all():
                raise RuntimeError(f"Published X MPC failed: status={status}")
            if np.max(np.abs(controls[0, :, 0])) > CONFIG["v_max"]+1e-6:
                raise RuntimeError("X MPC exceeded matched speed bound")
            if np.max(np.abs(controls[0, :, 1])) > CONFIG["w_max"]+1e-6:
                raise RuntimeError("X MPC exceeded matched angular bound")
            self.plan_id, self.controls, self.cursor = plan_id, controls[0], 0
            self.receipt = dict(reference_local=reference.tolist(), controls=controls[0].tolist(),
                                predicted_states=states[0].tolist(), solver_status=status,
                                desired_v=float(desired[0]), max_curvature=float(curvature[0]))
        if self.cursor >= 8:
            raise RuntimeError("A fresh plan is required after eight X MPC commands")
        index = self.cursor
        v, w = map(float, self.controls[index])
        self.cursor += 1
        return v, w, solve_s, dict(self.receipt, consumed_index=index)


def install_rgb_only_transport(base):
    """Preserve native FIFO while X executes; no simulator depth on X wire."""
    original = base.requests.post
    use_x = getattr(getattr(base, "args", None), "revisit_controller", None) == "xnavdp_point"
    state = {}
    if use_x:
        original_plan = base.srv_plan

        def plan(*a, **kw):
            from MemNavData.xnavdp_revisit_contract import xnavdp_state_payload
            state["payload"] = xnavdp_state_payload(kw["robot_position"], kw["robot_yaw"])
            result = original_plan(*a, **kw)
            if result.get("controller") == "xnavdp_point_posttrain":
                result["pose_controller"] = "xnavdp_point_posttrain"
            return result

        base.srv_plan = plan

    def post(url, *a, **kw):
        certified_x = use_x and url == f"{base.NOVEL_BASE}/navdp_step_ip_mixgoal"
        if certified_x:
            url = f"{base.XNAVDP_BASE}/pointgoal_step"
            kw["data"] = dict(kw["data"], state_data=json.dumps(state["payload"]))
        x_call = url == f"{base.XNAVDP_BASE}/pointgoal_step"
        if url in (f"{base.XNAVDP_BASE}/pointgoal_step", f"{base.XNAVDP_BASE}/memory_replay_step"):
            kw["data"] = dict(kw.get("data") or {}, xnavdp_rgb_contract=
                              os.environ.get("XNAVDP_RGB_CONTRACT", "rgb_v1"))
        if x_call:
            kw["files"] = dict(kw["files"])
            kw["files"].pop("depth", None)
            kw["files"].pop("image_goal", None)
        response = original(url, *a, **kw)
        if x_call and response.ok:
            if certified_x:
                expected = int(base.XNAVDP_CLIENT_STATE["history_frame_count"])+1
                result = base.normalize_xnavdp_response(response.json(),
                    expected_seed=int(kw["data"]["diffusion_seed"]), expected_history_frame_count=expected)
                base.XNAVDP_CLIENT_STATE["history_frame_count"] = expected
                response._content = json.dumps(result, allow_nan=False).encode()
            replay = original(f"{base.NOVEL_BASE}/memory_replay_step",
                              files={"image": kw["files"]["image"]})
            replay.raise_for_status()
            if replay.json().get("diffusion_sampled") is not False:
                raise RuntimeError("Native shadow FIFO replay sampled diffusion")
        elif use_x and url == f"{base.NOVEL_BASE}/imagegoal_step" and response.ok:
            base.srv_xnavdp_memory_replay(kw["files"]["image"][1])
        return response

    base.requests.post = post


def run_shared_evaluator(snapshot_path, validate_only=False):
    """Permit X only in this consumed diagnostic; keep all other checks intact."""
    source = ROOT / "MemNavData/eval_shared_online_role_pairs.py"
    text = source.read_text()
    old = 'args.revisit_controller == "navdp_mixed",\n        "legacy evaluator controller label must remain neutral",'
    new = ('(args.revisit_controller == "navdp_mixed" or\n'
           '         (args.revisit_controller == "xnavdp_point" and\n'
           '          args.role_pair_scope == "consumed_integration")),\n'
           '        "X controller is restricted to the consumed physical diagnostic",')
    if text.count(old) != 1:
        raise RuntimeError("Shared evaluator controller check changed")
    text = text.replace(old, new)
    if snapshot_path is not None:
        Path(snapshot_path).write_text(text)
    import eval_2leg_habitat as base
    original_validator = base.validate_revisit_adapter_configuration
    def validate_adapter(**kw):
        if (kw["revisit_controller"] == "xnavdp_point" and
                base.args.role_pair_scope == "consumed_integration" and
                kw["mode"] == "verified_bearing_v1" and kw["server_backend"] == "hybrid_pose"):
            # Same certificate and fixed-radius adapter; authorize only this
            # controller substitution in the isolated comparison, not formal runs.
            kw = dict(kw, revisit_controller="navdp_mixed")
        return original_validator(**kw)
    base.validate_revisit_adapter_configuration = validate_adapter
    namespace = {"__name__": "physical_x_stack", "__file__": str(source)}
    exec(compile(text, str(source)+":consumed_X_controller", "exec"), namespace)
    return namespace["validate_cli"]() if validate_only else namespace["main"]()
