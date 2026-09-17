"""Reuse GEM's existing decision; adapt only the downstream goal interface.

Native/reject: original ImageGoal. Accept: certified historical anchor JPEG.
The certified bearing remains available to the common physical turn adapter,
but neither an image controller nor this transport receives a metric target.
"""
import hashlib
import inspect
import json
from pathlib import Path


def reset_source(source, controller):
    old = 'if novel_info.get("depth_source") != args.navdp_depth_source:'
    if controller not in ("vint", "nomad") or source.count(old) != 1:
        raise ValueError("Unexpected controller or reset integration point")
    new = (f'if novel_info.get("controller") != {controller!r} '
           'or novel_info.get("controller_depth_source") != "none":')
    return source.replace(old, new).replace(
        'raise RuntimeError("NavDP reset did not honor the frozen depth arm")',
        'raise RuntimeError("RGB controller reset violated its input contract")')


def replay_source(source):
    old = "expected_queue = min(len(plan_steps), final_memory_size)"
    if source.count(old) != 1:
        raise ValueError("Unexpected replay integration point")
    # Released RGB controllers repeat their first observation to fill context.
    return source.replace(old, "expected_queue = final_memory_size  # repeated-first-frame context")


class ImageControllerGoalAdapter:
    def __init__(self, base, controller, out):
        self.base, self.controller, self.out = base, controller, Path(out)
        self.certificate, self.certificate_goal, self.anchor_cache = None, None, {}
        self.plan_receipt = None

    def append(self, name, row):
        with (self.out / name).open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def install(self):
        base = self.base
        post, plan = base.requests.post, base.srv_plan
        source = reset_source(inspect.getsource(base.srv_reset), self.controller)
        destination = self.out / "rgb_controller_reset.py"
        destination.write_text(source)
        namespace = {}
        exec(compile(source, str(destination), "exec"), base.__dict__, namespace)
        base.srv_reset = namespace["srv_reset"]
        import eval_shared_online_role_pairs as shared
        source = replay_source(inspect.getsource(shared.replay_online_a))
        destination = self.out / "rgb_controller_replay.py"
        destination.write_text(source)
        namespace = {}
        exec(compile(source, str(destination), "exec"), shared.replay_online_a.__globals__, namespace)
        shared.replay_online_a = namespace["replay_online_a"]
        # Binary anchor reads do not pass through the JSON append auditor.
        anchor_session = base.requests.Session()

        def forward(url, *a, **kw):
            if url == f"{base.BASE}/certified_relocalize":
                response = post(url, *a, **kw)
                response.raise_for_status()
                self.certificate = response.json()
                self.certificate_goal = kw["files"]["goal"][1]
                return response
            if url not in (f"{base.NOVEL_BASE}/imagegoal_step", f"{base.NOVEL_BASE}/navdp_step_ip_mixgoal"):
                return post(url, *a, **kw)
            files, data = kw["files"], kw.get("data", {})
            mixed = url.endswith("/navdp_step_ip_mixgoal")
            original_goal = files["image_goal" if mixed else "goal"][1]
            goal, anchor, anchor_sha = original_goal, None, None
            if mixed:
                c = self.certificate
                if not c or c.get("accepted") is not True or original_goal != self.certificate_goal:
                    raise RuntimeError("ImageGoal substitution lacks a current accepted GEM decision")
                anchor, anchor_sha = int(c["selected_anchor"]), c["selected_anchor_image_sha256"]
                key = (anchor, anchor_sha)
                if key not in self.anchor_cache:
                    r = anchor_session.post(f"{base.BASE}/certified_anchor_image",
                        files={"goal": ("goal.jpg", original_goal)},
                        data={"selected_anchor": str(anchor), "expected_anchor_sha256": anchor_sha}, timeout=180)
                    r.raise_for_status()
                    if (hashlib.sha256(r.content).hexdigest() != anchor_sha
                            or r.headers.get("X-CEC-Anchor-Index") != str(anchor)):
                        raise RuntimeError("Certified historical image identity changed")
                    self.anchor_cache[key] = r.content
                goal = self.anchor_cache[key]
            # Explicit allowlist: neither serialized depth nor PointGoal crosses
            # into the RGB-only controller. The original goal stays with GEM/scorer.
            response = post(f"{base.NOVEL_BASE}/imagegoal_step",
                files={"image": files["image"], "goal": ("goal.jpg", goal)},
                data={"diffusion_seed": str(int(data["diffusion_seed"]))})
            response.raise_for_status()
            result = response.json()
            if result.get("controller") != self.controller or result.get("controller_depth_source") != "none":
                raise RuntimeError("Unexpected downstream controller")
            audit = result["execution_input_audit"]
            if audit["goal_jpeg_sha256"] != hashlib.sha256(goal).hexdigest():
                raise RuntimeError("Controller did not consume the issued image")
            if self.controller == "nomad" and (result.get("goal_mask") != [0]
                    or result.get("controller_seed_consumed") is not True
                    or result.get("diffusion_seed") != int(data["diffusion_seed"])):
                raise RuntimeError("NoMaD did not consume goal and paired diffusion seed")
            self.plan_receipt = dict(controller=self.controller, accepted=mixed, anchor=anchor,
                original_goal_sha256=hashlib.sha256(original_goal).hexdigest(),
                controller_goal_sha256=hashlib.sha256(goal).hexdigest(), anchor_sha256=anchor_sha,
                diffusion_seed=int(data["diffusion_seed"]), policy_pointgoal_consumed=False,
                controller_seed_consumed=result["controller_seed_consumed"], goal_mask=result["goal_mask"],
                predicted_temporal_distance=result["predicted_temporal_distance"],
                distance_used_to_suppress_trajectory=result["distance_used_to_suppress_trajectory"],
                controller_depth_source="none", execution_input_audit=audit)
            self.append("image_controller_requests.jsonl", self.plan_receipt)
            return response

        def srv_plan(*a, **kw):
            self.plan_receipt = None
            result = plan(*a, **kw)
            if self.plan_receipt is None:
                if result.get("geometry_stream_stop") is True:
                    return result
                raise RuntimeError("Planning did not reach the configured image controller")
            result["pose_controller"] = self.controller + ("_certified_imagegoal" if self.plan_receipt["accepted"] else "_native_imagegoal")
            # This field is an alignment target in the legacy evaluator, not a
            # PointGoal policy input. Actual HTTP receipt above proves that scope.
            result["controller_depth_source"] = "none"
            return result

        base.requests.post, base.srv_plan = forward, srv_plan
