"""Consumed-history physical heading alignment; not a production CEC change.

Only the first accepted rearward goal is eligible. The target heading is held
through one velocity-driven turn; new RGB, not rotated old waypoints, feeds the
next plan. Habitat pose is low-level feedback, never a goal-direction oracle.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class PhysicalAlignment:
    max_yaw_rate = math.pi / 4
    dt = .1
    tolerance = math.radians(1)
    max_actions = 80

    def __init__(self, enabled):
        self.enabled = enabled
        self.considered = False
        self.active = False
        self.target_yaw = None
        self.action_count = 0
        self.events = []

    def consider(self, response, yaw, action_index):
        if self.considered or response.get("certified_relocalization_accepted") is not True:
            return
        if response.get("router_active") is not True:
            return
        point = response.get("memory_controller_pointgoal")
        if not isinstance(point, list) or len(point) != 2:
            raise RuntimeError("Accepted diagnostic goal has no controller pointgoal")
        forward, left = map(float, point)
        if (not all(map(math.isfinite, (forward, left)))
                or abs(math.hypot(forward, left) - 2.5) > 1e-5
                or response.get("certified_relocalization_guidance_mode") != "endpoint_bearing"):
            raise RuntimeError("Diagnostic alignment must retain certified endpoint bearing / 2.5 m")
        self.considered = True
        turn = math.atan2(left, forward)
        self.active = self.enabled and forward < 0
        self.target_yaw = wrap(yaw + turn) if self.active else None
        receipt = {
            "event": "first_certified_goal", "action": action_index,
            "enabled": self.enabled, "rearward": forward < 0,
            "activated": self.active, "pointgoal": point,
            "requested_turn_rad": turn, "yaw_before_rad": yaw,
            "target_yaw_rad": self.target_yaw,
            "anchor": response.get("anchor"),
            "memory_frame_idx": response.get("memory_frame_idx"),
            "authority_response_sha256": hashlib.sha256(json.dumps(
                response, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        }
        self.events.append(receipt)

    def command(self, yaw):
        if not self.active:
            raise RuntimeError("No physical alignment is pending")
        if self.action_count >= self.max_actions:
            raise RuntimeError("Physical alignment exhausted its fixed 80-command bound")
        remaining = wrap(self.target_yaw - yaw)
        return 0.0, max(-self.max_yaw_rate, min(self.max_yaw_rate, remaining / self.dt))

    def observe_after_action(self, yaw, action_index):
        self.action_count += 1
        residual = wrap(self.target_yaw - yaw)
        if abs(residual) <= self.tolerance:
            self.active = False
            self.events.append({
                "event": "turn_completed", "action": action_index,
                "actions": self.action_count, "yaw_after_rad": yaw,
                "residual_rad": residual, "fresh_replan_required": True,
            })

    def receipt(self):
        return {"enabled": self.enabled, "active": self.active,
                "actions": self.action_count, "events": self.events}


def install_loop_hook(base, original, alignment, output):
    """Add one pending-turn branch to a process-local copy of the evaluator.

    The production evaluator, its old idealized yaw switch, and its validators
    remain untouched. The normal branch is byte-identical source. Both arms use
    this copy; disabled / forward-first cases do not enter the new branch.
    """
    source = inspect.getsource(original)
    marker = "        if cec_bounded_alignment_remaining_rad is not None:\n"
    addition = '''        if _bullet_alignment.active:
            memory_response = srv_memory(frame)
            memory_frame_idx = memory_response.get("frame_idx")
            if memory_frame_idx is not None:
                memory_trace.append(dict(frame_idx=int(memory_frame_idx),
                                         step=int(step), x=float(pos[0]),
                                         z=float(pos[2]), yaw=float(psi)))
            # Preserve the nominal eight-command decision-image stride while
            # suppressing diffusion during the atomic turn. LingBot receives
            # every fresh frame once. No simulator motion is supplied here.
            if step % args.exec_horizon == 0:
                srv_navdp_memory_replay(frame)
            pos, psi, dl = pursuit_step(pos, psi, None, pf)
            bind_next_executor_receipt(frame_pose_position, frame_pose_yaw, dl)
            path_len += dl
            history.append(np.array([pos[0], pos[2]]))
            way_world = None
            cec_bounded_alignment_force_replan = True
            continue

'''
    if source.count(marker) != 1:
        raise RuntimeError("Physical alignment insertion point changed")
    modified = source.replace(marker, addition + marker)
    output.write_text(modified)
    base.__dict__["_bullet_alignment"] = alignment
    namespace = {}
    exec(compile(modified, str(output), "exec"), base.__dict__, namespace)
    return namespace[original.__name__]
