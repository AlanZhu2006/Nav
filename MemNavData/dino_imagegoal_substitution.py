"""Diagnostic ImageGoal replacement using the existing causal DINO probe.

Only the supplied experiment process installs this hook. The normal native
append is replaced by one retrieval append, once at the first query plan.
All later appends, depth transactions and NavDP requests remain native.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select_history_image(probe, images, expected_hashes):
    """Use only raw visual candidates, never the probe's learned gate/pose."""
    if probe["frame_idx"] != len(images):
        raise ValueError("first query frame does not follow the frozen history")
    if probe["candidate_ceiling"] != len(images) - 1:
        raise ValueError("retrieval boundary differs from the causal A prefix")
    candidates = probe["certified_visual_candidates"]
    if not candidates:
        raise ValueError("this nonempty-history pilot requires a DINO candidate")
    ranked = sorted(candidates, key=lambda c: (-float(c["score"]), int(c["anchor"])))
    chosen = ranked[0]
    anchor = int(chosen["anchor"])
    if not 8 <= anchor < len(images):
        raise ValueError("retrieved image is not in the CEC-eligible causal history")
    image = Path(images[anchor]).read_bytes()
    if digest(image) != expected_hashes[anchor]:
        raise ValueError("retrieved image differs from the actually replayed RGB")
    return image, {
        "anchor": anchor, "dino_cosine": float(chosen["score"]),
        "anchor_rgb_sha256": digest(image),
        "candidate_ceiling": int(probe["candidate_ceiling"]),
        "candidate_minimum": 8, "shortlist": candidates,
        "retrieval_rule": "first_raw_dino_candidate_before_geometry",
        "candidate_lifecycle": "frozen_at_first_query",
        "learned_gate_consumed": False,
        "geometric_pose_consumed": False, "role_consumed": False,
    }


class ImageGoalSubstitution:
    def __init__(self, base, images, expected_hashes, out):
        self.base, self.images, self.hashes = base, images, expected_hashes
        self.out = Path(out)
        self.transport, self.native_plan = base.requests.post, base.srv_plan
        self.in_plan = False
        self.goal = None
        self.replacement = None
        self.selection = None
        self.rows = []

    def install(self):
        self.base.requests.post, self.base.srv_plan = self.post, self.plan

    def post(self, url, *args, **kwargs):
        if not self.in_plan:
            return self.transport(url, *args, **kwargs)
        if url == self.base.BASE + "/memory_step" and self.replacement is not None:
            return self.transport(url, *args, **kwargs)
        if url == self.base.BASE + "/memory_step" and self.replacement is None:
            # One append, not probe + append: retain the original image and
            # materialize_monocular_depth fields/transaction.
            files = dict(kwargs["files"], goal=("goal.jpg", self.goal))
            started = time.perf_counter()
            response = self.transport(self.base.BASE + "/retrieval_probe_step",
                                      *args, **dict(kwargs, files=files))
            response.raise_for_status()
            probe = response.json()
            self.replacement, self.selection = select_history_image(
                probe, self.images, self.hashes)
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "retrieved_imagegoal.jpg").write_bytes(self.replacement)
            self.selection.update(original_goal_sha256=digest(self.goal),
                probe_wall_ms=1000 * (time.perf_counter() - started),
                probe_timing=probe.get("retrieval_probe_timing"),
                one_query_append=True)
            (self.out / "retrieval_selection.json").write_text(
                json.dumps(self.selection, indent=2, allow_nan=False) + "\n")
            return response
        if url == self.base.NOVEL_BASE + "/imagegoal_step":
            if self.replacement is None:
                raise RuntimeError("retrieval must precede the first ImageGoal plan")
            if "goal_data" in (kwargs.get("data") or {}):
                raise ValueError("image-only baseline must not carry a PointGoal")
            files = dict(kwargs["files"], goal=("retrieved.jpg", self.replacement))
            response = self.transport(url, *args, **dict(kwargs, files=files))
            response.raise_for_status()
            row = dict(plan_index=len(self.rows),
                original_goal_sha256=digest(self.goal),
                controller_goal_sha256=digest(self.replacement),
                current_rgb_sha256=digest(files["image"][1]),
                anchor=self.selection["anchor"],
                diffusion_seed=(kwargs.get("data") or {}).get("diffusion_seed"),
                pointgoal_supplied=False,
                actual_input_audit=response.json().get("execution_input_audit"))
            self.rows.append(row)
            with (self.out / "imagegoal_substitution.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return response
        raise ValueError("unexpected endpoint during image-only planning: " + url)

    def plan(self, image, goal, *args, **kwargs):
        if self.goal is not None and self.goal != goal:
            raise ValueError("one frozen query per evaluator process")
        self.goal, self.in_plan = goal, True
        try:
            result = self.native_plan(image, goal, *args, **kwargs)
            result.update(pose_controller="navdp_retrieved_imagegoal",
                          anchor=self.selection["anchor"],
                          retrieved_imagegoal_used=True)
            return result
        finally:
            self.in_plan = False
