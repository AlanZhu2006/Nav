"""Explicit heading actions for a forward-only PointGoal input domain.

The upstream memory adapter has already decided whether a point request exists.
This component neither authorizes memory nor chooses a direction: it makes the
issued local target representable before the next frozen-policy plan. Habitat
executes bounded yaw changes with fresh RGB, not a virtual rotation of old RGB
or trajectories. It is kinematic, not a claim of full-body physical safety.
"""
import inspect
import math

import numpy as np

from MemNavData.bounded_pursuit import PursuitCommand


EXECUTOR_ODOMETRY_FIELDS = frozenset({
    "executed_translation_m", "executed_yaw_rad", "executed_forward_m",
    "executed_left_m", "executor_local_se2_source", "executor_local_se2_contract",
})


def rgb_only_memory_form(data):
    """The endpoint-bearing study does not need simulator odometry receipts.

    Keep ideal state in the tracker/evaluator, outside the geometry model's
    request boundary. These old diagnostic fields are not used by the current
    endpoint estimator; removing them is not a new odometry estimator.
    """
    return {key: value for key, value in data.items() if key not in EXECUTOR_ODOMETRY_FIELDS}


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FrontGoalAdapter:
    def __init__(self, *, enabled, max_turn_rad):
        if not math.isfinite(max_turn_rad) or not 0 < max_turn_rad <= math.pi:
            raise ValueError("A finite positive angular action bound is required")
        self.enabled = bool(enabled)
        self.max_turn_rad = float(max_turn_rad)
        self.active = False
        self.target_yaw = None
        self.actions = 0
        self.events = []

    def consider(self, response, yaw, action_index):
        if self.active:
            raise RuntimeError("A pending heading action cannot be resampled")
        if response.get("revisit_adapter_takeover") is not True:
            return False
        point = np.asarray(response.get("memory_controller_pointgoal"), dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(yaw):
            raise ValueError("An issued point request must contain a finite 2-D target")
        rearward = bool(point[0] < 0)
        if not rearward:
            return False
        turn = math.atan2(point[1], point[0])
        self.active = self.enabled
        self.target_yaw = float(yaw + turn) if self.active else None
        self.events.append({
            "event": "rearward_point_request", "action_index": int(action_index),
            "pointgoal": point.tolist(), "yaw_before_rad": float(yaw),
            "requested_turn_rad": turn, "target_yaw_rad": self.target_yaw,
            "activated": self.active, "anchor": response.get("anchor"),
            "memory_frame_idx": response.get("memory_frame_idx"),
        })
        return self.active

    def command(self, position, yaw):
        if not self.active:
            raise RuntimeError("No heading alignment is active")
        pos = np.asarray(position, dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all() or not math.isfinite(yaw):
            raise ValueError("Expected a finite low-level pose")
        delta = max(-self.max_turn_rad, min(self.max_turn_rad, wrap(self.target_yaw-yaw)))
        return PursuitCommand(pos.copy(), float(yaw+delta), 0., 0., 0., -1,
                              "pointgoal_heading_alignment")

    def observe(self, yaw, action_index):
        if not self.active or not math.isfinite(yaw):
            raise RuntimeError("Unexpected heading observation")
        self.actions += 1
        residual = wrap(self.target_yaw-yaw)
        if abs(residual) <= 1e-8:  # arithmetic completion, not confidence/arrival
            self.active = False
            self.events.append({"event": "heading_complete", "action_index": int(action_index),
                                "yaw_after_rad": float(yaw), "residual_rad": residual,
                                "fresh_replan_required": True})

    def receipt(self):
        return {"enabled": self.enabled, "active": self.active, "actions": self.actions,
                "max_turn_rad": self.max_turn_rad, "events": self.events}


def install_loop_hook(base, original, adapter, output):
    """Process-local integration: do not edit the production evaluator.

    The first request has already appended its decision RGB and sampled NavDP.
    If its issued goal is rearward, its trajectory is recorded but NOT executed.
    Later turn frames append once; only nominal decision frames enter the NavDP
    FIFO. No new diffusion sample is drawn until a fresh post-turn observation.
    The initial discarded sample is retained/counted, not hidden as zero cost.
    """
    source = inspect.getsource(original)
    marker = "        if cec_bounded_alignment_remaining_rad is not None:\n"
    addition = '''        if _front_goal_adapter.active:
            memory_response = srv_memory(frame)
            memory_frame_idx = memory_response.get("frame_idx")
            if memory_frame_idx is not None:
                memory_trace.append(dict(frame_idx=int(memory_frame_idx), step=int(step),
                                         x=float(pos[0]), z=float(pos[2]), yaw=float(psi)))
            if step % args.exec_horizon == 0:
                srv_navdp_memory_replay(frame)
            pos, psi, dl = pursuit_step(pos, psi, None, pf)
            bind_next_executor_receipt(frame_pose_position, frame_pose_yaw, dl)
            path_len += dl
            history.append(np.array([pos[0], pos[2]]))
            if (len(history) > args.stuck_window and
                    np.linalg.norm(history[-1] - history[-args.stuck_window]) < args.stuck_dist):
                return result(step + 1, termination_reason="stuck")
            way_world = None
            cec_bounded_alignment_force_replan = True
            continue

'''
    if source.count(marker) != 1:
        raise RuntimeError("Evaluator heading insertion point changed")
    changed = source.replace(marker, addition + marker)
    output.write_text(changed)
    base.__dict__["_front_goal_adapter"] = adapter
    namespace = {}
    exec(compile(changed, str(output), "exec"), base.__dict__, namespace)
    return namespace[original.__name__]
