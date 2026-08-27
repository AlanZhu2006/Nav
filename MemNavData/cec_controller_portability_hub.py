#!/usr/bin/env python3
"""One CEC proof layer with controller-native accepted-branch adapters.

Every arm observes the same causal RGB stream and runs the same CEC proof.
The frozen reject policy is explicit: either shared monocular NavDP receives
the original ImageGoal, or an ImageGoal-capable selected controller receives
that unchanged goal itself.  CEC acceptance authorizes one configured
controller adapter:

* NavDP receives its frozen ImageGoal + 2.5 m bearing mixed goal;
* ViNT, GNM and NoMaD receive the hash-bound certified history anchor as
  ImageGoal;
* iPlanner and ViPlanner receive the normalized 2.5 m PointGoal and the same
  LingBot monocular-depth sidecar.

This service never reads a Novel/Revisit role label.  CEC authorizes each
decision independently.  The inactive temporal controller context is
shadow-maintained so later action-level switches remain causal.  The service
owns no actuator interface.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import io
import json
import math
import threading
import time
from typing import Any, Mapping, MutableMapping

from flask import Flask, jsonify, request
import numpy as np
import requests

from MemNavData.controller_portability_contract import (
    CEC_PROOF_HYBRID,
    CecProjection,
    ComparisonPlan,
    controller_spec,
    project_cec_proof,
    validate_comparison_plan,
)
from MemNavData.cec_handoff_contract import (
    build_handoff_packet,
    project_handoff_packet,
)
from MemNavData.monocular_depth_runtime import (
    decode_monocular_depth_payload,
)


PORTABILITY_HUB_SCHEMA = "cec_controller_portability_hub_v2"
PROPOSAL_ORDER = "geometry_first"
SUPPORTED_HTTP_CONTROLLERS = frozenset({
    "navdp", "vint", "gnm", "nomad", "iplanner", "viplanner",
})
# Controllers whose native server keeps a short causal RGB context (a
# memory_queue of the last few frames) instead of reading a metric-depth
# sidecar.  These need context shadowing via /observation_step, not a
# monocular depth sidecar.
SHORT_CONTEXT_CONTROLLERS = frozenset({"vint", "gnm", "nomad"})
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "role", "goal_role", "query_role", "is_revisit", "is_novel",
    "oracle_pose", "gt_pose", "ground_truth_pose", "habitat_pose",
})


class PortabilityHubError(RuntimeError):
    """A stateful upstream is uncertain and the episode must be reset."""


@dataclass(frozen=True)
class PortabilityHubConfig:
    controller: str
    memnav_url: str
    controller_url: str
    fallback_navdp_url: str
    camera_height_m: float
    reject_policy: str = "shared_native_exact"
    connect_timeout_s: float = 3.0
    request_timeout_s: float = 180.0
    # Native-control counterfactual: run the identical probe/certificate
    # pipeline and receipts, but never grant CEC control authority.  The
    # configured reject policy determines which native controller acts.
    force_reject_native: bool = False
    emit_handoff_packets: bool = False

    @property
    def timeout(self) -> tuple[float, float]:
        return (float(self.connect_timeout_s), float(self.request_timeout_s))


def _file(name: str, payload: bytes, media_type: str):
    return (name, io.BytesIO(payload), media_type)


def _json_object(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, MutableMapping):
        raise PortabilityHubError(f"{label} returned non-object JSON")
    return dict(payload)


def _finite_intrinsic(value: Any) -> list[list[float]]:
    try:
        matrix = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intrinsic must be a finite 3x3 matrix") from exc
    if (len(matrix) != 3 or any(len(row) != 3 for row in matrix)
            or not np.isfinite(np.asarray(matrix)).all()):
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    return matrix


def _reject_privileged_fields(payload: Mapping[str, Any]) -> None:
    leaked = sorted(set(payload) & FORBIDDEN_RUNTIME_FIELDS)
    if leaked:
        raise ValueError("privileged runtime fields are forbidden: " + ", ".join(leaked))


def _finite_trajectory(payload: Mapping[str, Any]) -> None:
    if "trajectory" not in payload:
        raise PortabilityHubError("controller response lacks trajectory")
    try:
        trajectory = np.asarray(payload["trajectory"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityHubError("controller trajectory is not numeric") from exc
    if (trajectory.size == 0 or trajectory.ndim < 2
            or trajectory.shape[-1] != 3
            or not np.isfinite(trajectory).all()):
        raise PortabilityHubError("controller trajectory must be finite xyz")


class CecControllerPortabilityRouter:
    """Stateful role-free CEC router for one frozen downstream controller."""

    def __init__(
        self,
        config: PortabilityHubConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        if config.controller not in SUPPORTED_HTTP_CONTROLLERS:
            raise ValueError(
                f"controller {config.controller!r} has no audited HTTP adapter")
        depth_source = (
            "none" if config.controller in SHORT_CONTEXT_CONTROLLERS
            else "monocular_sidecar")
        self.plan = ComparisonPlan(
            controller=config.controller,
            protocol=CEC_PROOF_HYBRID,
            depth_source=depth_source,
            query_population="mixed_role",
            reject_policy=config.reject_policy,
            fallback_controller=(
                config.controller
                if config.reject_policy == "controller_native_exact"
                else "navdp"),
        )
        self.spec = validate_comparison_plan(self.plan)
        if (not math.isfinite(config.camera_height_m)
                or not 0.1 <= config.camera_height_m <= 2.0):
            raise ValueError("camera_height_m must be finite and in [0.1, 2.0]")
        self.config = config
        self.session = session or requests.Session()
        self.initialized = False
        self.reset_required = False
        self.step_index = 0
        self.query_index = 0
        self.last_action_state = "unresolved"
        self._anchor_jpeg: bytes | None = None
        self._anchor_index: int | None = None
        self._anchor_sha256: str | None = None
        self._goal_sha256: str | None = None
        self._causal_history_sha256: str | None = None

    def _post_json(self, url: str, label: str, **kwargs) -> dict[str, Any]:
        return _json_object(
            self.session.post(url, timeout=self.config.timeout, **kwargs), label)

    def _advance_causal_history(self, image: bytes, goal: bytes) -> str:
        """Extend the role-free history chain after one observed decision.

        The reset digest binds the complete frozen online-A trace.  Each later
        decision extends it with the actual RGB and active goal, so packets
        emitted after controller trajectories diverge cannot accidentally
        claim the same causal history.
        """
        if self._causal_history_sha256 is None:
            raise PortabilityHubError("causal-history chain is unavailable")
        payload = {
            "prior_sha256": self._causal_history_sha256,
            "current_rgb_sha256": hashlib.sha256(image).hexdigest(),
            "goal_rgb_sha256": hashlib.sha256(goal).hexdigest(),
            "decision_index": int(self.step_index),
        }
        self._causal_history_sha256 = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        return self._causal_history_sha256

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _reject_privileged_fields(payload)
        intrinsic = _finite_intrinsic(payload.get("intrinsic"))
        common = dict(payload)
        causal_history_sha256 = common.pop("causal_history_sha256", None)
        if self.config.emit_handoff_packets:
            if (not isinstance(causal_history_sha256, str)
                    or len(causal_history_sha256) != 64
                    or any(character not in "0123456789abcdef"
                           for character in causal_history_sha256)):
                raise ValueError(
                    "handoff packet mode requires causal_history_sha256")
        common["intrinsic"] = intrinsic
        fallback_payload = dict(common)
        fallback_payload["depth_source"] = "monocular_sidecar"
        memory_payload = {
            "camera_height": float(self.config.camera_height_m),
            "camera_intrinsic": intrinsic,
            "seed": payload.get("seed"),
            "episode_len": payload.get("episode_len"),
        }
        self.initialized = False
        self.reset_required = False
        self.step_index = 0
        self.query_index = 0
        self.last_action_state = "unresolved"
        self._anchor_jpeg = None
        self._anchor_index = None
        self._anchor_sha256 = None
        self._goal_sha256 = None
        self._causal_history_sha256 = causal_history_sha256
        try:
            memory = self._post_json(
                f"{self.config.memnav_url.rstrip('/')}/navigator_reset",
                "MemNav reset", json=memory_payload)
            fallback = self._post_json(
                f"{self.config.fallback_navdp_url.rstrip('/')}/navigator_reset",
                "fallback NavDP reset", json=fallback_payload)
            if (self.spec.key == "navdp"
                    and self.config.controller_url.rstrip("/")
                    == self.config.fallback_navdp_url.rstrip("/")):
                controller = fallback
            else:
                controller = self._post_json(
                    f"{self.config.controller_url.rstrip('/')}/navigator_reset",
                    f"{self.spec.display_name} reset", json=common)
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                f"atomic reset failed: {type(exc).__name__}: {exc}") from exc

        certificate = memory.get("certified_relocalization")
        if not isinstance(certificate, Mapping) or certificate.get("enabled") is not True:
            self.reset_required = True
            raise PortabilityHubError("MemNav did not enable CEC")
        if (fallback.get("depth_source") != "monocular_sidecar"
                or fallback.get("metric_depth_sensor_consumed_by_config") is not False
                or fallback.get("monocular_depth_url_configured") is not True):
            self.reset_required = True
            raise PortabilityHubError(
                "shared fallback did not establish the full-mono contract")
        if self.spec.key != "navdp":
            receipt = controller.get("portability_receipt")
            if (not isinstance(receipt, Mapping)
                    or receipt.get("controller") != self.spec.key
                    or receipt.get("comparison_protocol") != CEC_PROOF_HYBRID
                    or receipt.get("cec_accept_adapter")
                    != self.spec.cec_accept_adapter):
                self.reset_required = True
                raise PortabilityHubError(
                    "alternate controller did not return the CEC proxy receipt")
        self.initialized = True
        return {
            "ok": True,
            "algo": "cec_controller_portability",
            "schema": PORTABILITY_HUB_SCHEMA,
            "controller": self.spec.key,
            "cec_accept_adapter": self.spec.cec_accept_adapter,
            "reject_controller": self.plan.fallback_controller,
            "reject_policy": self.plan.reject_policy,
            "force_reject_native": bool(self.config.force_reject_native),
            "handoff_packets_enabled": bool(
                self.config.emit_handoff_packets),
            "controller_depth_source": self.plan.depth_source,
            "depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_by_config": False,
            "monocular_depth_url_configured": True,
            "role_label_visible": False,
            "metric_depth_sensor_consumed_by_policy": False,
        }

    def _probe(
        self, image: bytes, goal: bytes, form: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = {}
        if form.get("candidate_ceiling_override") not in (None, ""):
            data["candidate_ceiling_override"] = str(
                int(form["candidate_ceiling_override"]))
        try:
            return self._post_json(
                f"{self.config.memnav_url.rstrip('/')}/retrieval_probe_step",
                "CEC retrieval probe",
                files={
                    "image": _file("image.jpg", image, "image/jpeg"),
                    "goal": _file("goal.jpg", goal, "image/jpeg"),
                },
                data=data,
            )
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                f"CEC causal stream failed: {type(exc).__name__}: {exc}") from exc

    def _certificate(
        self, goal: bytes, candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return self._post_json(
                f"{self.config.memnav_url.rstrip('/')}/certified_relocalize",
                "CEC certificate",
                files={"goal": _file("goal.jpg", goal, "image/jpeg")},
                data={
                    "candidates": json.dumps(candidates),
                    "proposal_order": PROPOSAL_ORDER,
                    "graph_rescue": "0",
                    "learned_rescue": "0",
                },
            )
        except Exception as exc:
            return {
                "ok": False,
                "accepted": False,
                "reason": "certificate_endpoint_failure_action_fallback",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _fetch_certified_anchor(
        self, goal: bytes, certificate: Mapping[str, Any],
    ) -> bytes:
        anchor = certificate.get("selected_anchor")
        expected = certificate.get("selected_anchor_image_sha256")
        response = self.session.post(
            f"{self.config.memnav_url.rstrip('/')}/certified_anchor_image",
            files={"goal": _file("goal.jpg", goal, "image/jpeg")},
            data={
                "selected_anchor": str(anchor),
                "expected_anchor_sha256": str(expected),
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        image = bytes(response.content)
        actual = hashlib.sha256(image).hexdigest()
        if (not image or actual != expected
                or response.headers.get("X-CEC-Anchor-SHA256") != expected
                or response.headers.get("X-CEC-Anchor-Index") != str(anchor)):
            raise PortabilityHubError("certified anchor response failed identity checks")
        return image

    def _mono_depth_png(self, image: bytes) -> tuple[bytes, dict[str, Any]]:
        image_sha = hashlib.sha256(image).hexdigest()
        payload = self._post_json(
            f"{self.config.memnav_url.rstrip('/')}/monocular_depth_query",
            "monocular depth sidecar",
            data={"expected_image_sha256": image_sha},
        )
        _depth, metadata = decode_monocular_depth_payload(
            payload, expected_image_sha256=image_sha)
        png = base64.b64decode(payload["depth_png_base64"], validate=True)
        return png, metadata

    def _fallback(
        self, image: bytes, original_goal: bytes,
        controller_form: Mapping[str, str],
    ) -> dict[str, Any]:
        controller_started = time.perf_counter()
        controller_native = (
            self.config.reject_policy == "controller_native_exact")
        fallback_url = (
            self.config.controller_url
            if controller_native else self.config.fallback_navdp_url)
        result = self._post_json(
            f"{fallback_url.rstrip('/')}/imagegoal_step",
            (f"{self.spec.display_name} native fallback"
             if controller_native else "shared native fallback"),
            files={
                "image": _file("image.jpg", image, "image/jpeg"),
                "goal": _file("goal.jpg", original_goal, "image/jpeg"),
            },
            data=dict(controller_form),
        )
        if not controller_native and (
                result.get("depth_source") != "monocular_sidecar"
                or result.get("metric_depth_sensor_consumed") is not False):
            raise PortabilityHubError(
                "fallback response did not prove monocular depth consumption")
        requested_seed = controller_form.get("diffusion_seed")
        if controller_native:
            receipt = result.get("portability_receipt")
            if (not isinstance(receipt, Mapping)
                    or receipt.get("controller") != self.spec.key
                    or receipt.get("endpoint") != "imagegoal_step"
                    or receipt.get("reject_policy")
                    != "controller_native_exact"):
                raise PortabilityHubError(
                    "controller-native fallback lost its portability receipt")
            if requested_seed is not None:
                result["diffusion_seed"] = int(requested_seed)
            result["cec_seed_semantics"] = (
                "paired_request_id_not_consumed_by_deterministic_controller")
            result["cec_controller_seed_consumed"] = False
        else:
            if (requested_seed is not None
                    and int(result.get("diffusion_seed", -1))
                    != int(requested_seed)):
                raise PortabilityHubError(
                    "fallback NavDP did not consume the paired diffusion seed")
            result["cec_seed_semantics"] = "navdp_diffusion_rng_consumed"
            result["cec_controller_seed_consumed"] = True
        _finite_trajectory(result)
        result["cec_controller_ms"] = (
            (time.perf_counter() - controller_started) * 1000.0)
        result["cec_depth_sidecar_ms"] = None
        return result

    def _shadow_vint_context(self, image: bytes) -> None:
        """Keep a short-context controller's (ViNT/GNM/NoMaD) causal RGB
        history advancing while NavDP owns a query."""
        if self.spec.key not in SHORT_CONTEXT_CONTROLLERS:
            return
        result = self._post_json(
            f"{self.config.controller_url.rstrip('/')}/observation_step",
            f"{self.spec.display_name} shadow observation",
            files={"image": _file("image.jpg", image, "image/jpeg")},
        )
        receipt = result.get("portability_receipt")
        if (result.get("observed") is not True
                or not isinstance(receipt, Mapping)
                or receipt.get("controller") != self.spec.key
                or receipt.get("endpoint") != "observation_step"):
            raise PortabilityHubError(
                f"{self.spec.display_name} shadow observation lost its "
                "portability receipt")

    def _shadow_fallback_context(self, image: bytes) -> bool:
        """Keep mono NavDP causal while another controller owns this action."""
        if (self.spec.key == "navdp"
                and self.config.controller_url.rstrip("/")
                == self.config.fallback_navdp_url.rstrip("/")):
            return False
        result = self._post_json(
            f"{self.config.fallback_navdp_url.rstrip('/')}/memory_replay_step",
            "fallback NavDP shadow observation",
            files={"image": _file("image.jpg", image, "image/jpeg")},
        )
        if result.get("diffusion_sampled") is not False:
            raise PortabilityHubError(
                "fallback shadow observation unexpectedly sampled an action")
        return True

    @staticmethod
    def _cec_form(projection: CecProjection) -> dict[str, str]:
        payload = projection.payload
        form = {
            "cec_proof_sha256": projection.proof_sha256,
            "cec_action_authorized": "1",
            "cec_selected_anchor": str(payload["cec_selected_anchor"]),
        }
        if projection.adapter == "verified_anchor_imagegoal":
            form.update({
                "goal_source": "certified_history_anchor",
                "cec_anchor_sha256": str(payload["cec_anchor_sha256"]),
            })
        return form

    def _accepted_controller(
        self,
        image: bytes,
        original_goal: bytes,
        projection: CecProjection,
        controller_form: Mapping[str, str],
    ) -> dict[str, Any]:
        data = {**dict(controller_form), **self._cec_form(projection)}
        payload = projection.payload
        base = self.config.controller_url.rstrip("/")
        depth_sidecar_ms = None
        if projection.adapter == "bearing_mixedgoal":
            data["goal_data"] = json.dumps({
                "goal_x": payload["goal_x"],
                "goal_y": payload["goal_y"],
            })
            controller_started = time.perf_counter()
            result = self._post_json(
                f"{base}/navdp_step_ip_mixgoal", "CEC NavDP mixed step",
                files={
                    "image": _file("image.jpg", image, "image/jpeg"),
                    "image_goal": _file(
                        "goal.jpg", original_goal, "image/jpeg"),
                }, data=data)
        elif projection.adapter == "bearing_pointgoal":
            depth_started = time.perf_counter()
            depth_png, depth_receipt = self._mono_depth_png(image)
            depth_sidecar_ms = (
                (time.perf_counter() - depth_started) * 1000.0)
            data["goal_data"] = json.dumps({
                "goal_x": payload["goal_x"],
                "goal_y": payload["goal_y"],
            })
            controller_started = time.perf_counter()
            result = self._post_json(
                f"{base}/pointgoal_step", f"CEC {self.spec.display_name} step",
                files={
                    "image": _file("image.jpg", image, "image/jpeg"),
                    "depth": _file("depth.png", depth_png, "image/png"),
                }, data=data)
            result["monocular_depth_receipt"] = depth_receipt
        elif projection.adapter == "verified_anchor_imagegoal":
            if self._anchor_jpeg is None:
                raise PortabilityHubError(
                    f"{self.spec.display_name} takeover lacks a certified "
                    "anchor")
            controller_started = time.perf_counter()
            result = self._post_json(
                f"{base}/imagegoal_step",
                f"CEC {self.spec.display_name} step",
                files={
                    "image": _file("image.jpg", image, "image/jpeg"),
                    "goal": _file(
                        "certified_anchor.jpg", self._anchor_jpeg, "image/jpeg"),
                }, data=data)
        else:
            raise PortabilityHubError(
                f"unsupported HTTP CEC adapter {projection.adapter!r}")
        result["cec_controller_ms"] = (
            (time.perf_counter() - controller_started) * 1000.0)
        result["cec_depth_sidecar_ms"] = depth_sidecar_ms
        _finite_trajectory(result)
        if self.spec.key != "navdp" and (
                result.get("cec_proof_sha256") != projection.proof_sha256):
            raise PortabilityHubError(
                "alternate controller response lost the CEC proof receipt")
        requested_seed = controller_form.get("diffusion_seed")
        if self.spec.key == "navdp":
            if (requested_seed is not None
                    and result.get("diffusion_seed") is None):
                raise PortabilityHubError(
                    "NavDP accepted action did not echo its consumed seed")
            result["cec_seed_semantics"] = "navdp_diffusion_rng_consumed"
            result["cec_controller_seed_consumed"] = True
        else:
            # ViNT/GNM/NoMaD/iPlanner/ViPlanner are deterministic inference
            # wrappers and have no NavDP diffusion RNG.  Preserve the paired
            # request ID for
            # evaluator alignment without claiming that their model consumed
            # it as a random seed.
            if requested_seed is not None:
                result["diffusion_seed"] = int(requested_seed)
            result["cec_seed_semantics"] = (
                "paired_request_id_not_consumed_by_deterministic_controller")
            result["cec_controller_seed_consumed"] = False
        return result

    def plan_imagegoal(
        self,
        *,
        image: bytes,
        goal: bytes,
        form: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision_started = time.perf_counter()
        if not self.initialized or self.reset_required:
            raise PortabilityHubError("router requires a successful reset")
        if not image or not goal:
            raise ValueError("image and goal are required")
        _reject_privileged_fields(form or {})
        form = dict(form or {})
        controller_form = {}
        if form.get("diffusion_seed") not in (None, ""):
            controller_form["diffusion_seed"] = str(int(form["diffusion_seed"]))
        self.step_index += 1
        goal_sha256 = hashlib.sha256(goal).hexdigest()
        goal_session_expected_start = goal_sha256 != self._goal_sha256
        if goal_session_expected_start:
            self.query_index += 1
            self._goal_sha256 = goal_sha256
            self._anchor_jpeg = None
            self._anchor_index = None
            self._anchor_sha256 = None
        probe_started = time.perf_counter()
        probe = self._probe(image, goal, form)
        probe_ms = (time.perf_counter() - probe_started) * 1000.0
        candidates = probe.get("certified_visual_candidates")
        if not isinstance(candidates, list):
            candidates = []

        certificate = None
        projection = None
        handoff_packet = None
        packet_history_sha256 = self._causal_history_sha256
        fallback_context_shadowed = False
        alternate_context_shadowed = False
        certificate_ms = 0.0
        projection_ms = 0.0
        shadow_ms = 0.0
        try:
            certificate_started = time.perf_counter()
            certificate = self._certificate(goal, candidates)
            certificate_ms = (
                (time.perf_counter() - certificate_started) * 1000.0)
            projection_started = time.perf_counter()
            if (certificate.get("accepted") is True
                    and (self.config.emit_handoff_packets
                         or (not self.config.force_reject_native
                             and self.spec.cec_accept_adapter
                             == "verified_anchor_imagegoal"))):
                selected_anchor = certificate.get("selected_anchor")
                selected_sha = certificate.get(
                    "selected_anchor_image_sha256")
                if (self._anchor_jpeg is None
                        or self._anchor_index != selected_anchor
                        or self._anchor_sha256 != selected_sha):
                    self._anchor_jpeg = self._fetch_certified_anchor(
                        goal, certificate)
                    self._anchor_index = int(selected_anchor)
                    self._anchor_sha256 = str(selected_sha)
            if (self.config.emit_handoff_packets
                    and certificate.get("accepted") is True):
                if self._anchor_jpeg is None:
                    raise PortabilityHubError(
                        "accepted handoff lacks certified anchor bytes")
                if self._causal_history_sha256 is None:
                    raise PortabilityHubError(
                        "accepted handoff lacks causal-history identity")
                handoff_packet = build_handoff_packet(
                    certificate,
                    current_rgb=image,
                    goal_rgb=goal,
                    anchor_jpeg=self._anchor_jpeg,
                    causal_history_sha256=packet_history_sha256,
                )
                projection = project_handoff_packet(
                    self.spec.key,
                    handoff_packet,
                    current_rgb=image,
                    goal_rgb=goal,
                    anchor_jpeg=self._anchor_jpeg,
                    causal_history_sha256=packet_history_sha256,
                )
            else:
                projection = project_cec_proof(
                    self.spec.key, certificate, anchor_jpeg=self._anchor_jpeg,
                    shadow_only=self.config.force_reject_native,
                    reject_policy=self.config.reject_policy)
            projection_ms = (
                (time.perf_counter() - projection_started) * 1000.0)
            shadow_takeover = bool(projection.takeover)
            takeover_authorized = (
                shadow_takeover and not self.config.force_reject_native)
            if takeover_authorized:
                result = self._accepted_controller(
                    image, goal, projection, controller_form)
                shadow_started = time.perf_counter()
                fallback_context_shadowed = self._shadow_fallback_context(
                    image)
                shadow_ms = (
                    (time.perf_counter() - shadow_started) * 1000.0)
                self.last_action_state = "takeover"
            else:
                result = self._fallback(image, goal, controller_form)
                shadow_started = time.perf_counter()
                if self.config.reject_policy == "controller_native_exact":
                    fallback_context_shadowed = self._shadow_fallback_context(
                        image)
                else:
                    self._shadow_vint_context(image)
                shadow_ms = (
                    (time.perf_counter() - shadow_started) * 1000.0)
                alternate_context_shadowed = bool(
                    self.config.reject_policy == "shared_native_exact"
                    and self.spec.key in SHORT_CONTEXT_CONTROLLERS)
                self.last_action_state = (
                    "forced_reject"
                    if shadow_takeover and self.config.force_reject_native
                    else "fallback")
        except PortabilityHubError:
            self.reset_required = True
            raise
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                "CEC/controller contract failed; reset is required: "
                f"{type(exc).__name__}: {exc}") from exc

        result.update({
            "cec_portability_schema": PORTABILITY_HUB_SCHEMA,
            "cec_decision_scope": "per_action",
            "cec_action_state": self.last_action_state,
            "cec_takeover": takeover_authorized,
            "cec_shadow_takeover": shadow_takeover,
            "cec_forced_reject_native": bool(self.config.force_reject_native),
            "cec_accept_controller": self.spec.key,
            "cec_accept_adapter": self.spec.cec_accept_adapter,
            "cec_reject_controller": self.plan.fallback_controller,
            "cec_reject_policy": self.plan.reject_policy,
            "cec_step_index": self.step_index,
            "cec_query_index": self.query_index,
            "cec_goal_sha256": goal_sha256,
            "cec_goal_session_expected_start": goal_session_expected_start,
            "cec_goal_session_index": probe.get("goal_session_index"),
            "cec_goal_session_started": probe.get("goal_session_started"),
            "cec_long_term_memory_preserved": probe.get(
                "long_term_memory_preserved"),
            "cec_goal_start_frame": probe.get("goal_start_frame"),
            "cec_candidate_ceiling": probe.get("candidate_ceiling"),
            "cec_frame_idx": probe.get("frame_idx"),
            # eval_2leg_habitat records every causal write through this common
            # compatibility field.  The CEC probe already performed the sole
            # append for this decision frame; exposing the same index prevents
            # plan frames from appearing as holes in the long-term trace.
            "memory_frame_idx": probe.get("frame_idx"),
            "cec_selected_anchor": projection.payload.get(
                "cec_selected_anchor"),
            "cec_proof_sha256": projection.proof_sha256,
            "cec_handoff_packet": handoff_packet,
            "cec_handoff_packet_sha256": (
                None if handoff_packet is None
                else handoff_packet["packet_sha256"]),
            "cec_handoff_single_use": (
                None if handoff_packet is None
                else handoff_packet["single_use"]),
            "cec_causal_history_before_decision_sha256": (
                packet_history_sha256
                if self.config.emit_handoff_packets else None),
            "cec_projected_goal": dict(projection.payload),
            "cec_controller_portability_receipt": result.get(
                "portability_receipt"),
            "cec_fallback_context_shadowed": fallback_context_shadowed,
            "cec_alternate_context_shadowed": alternate_context_shadowed,
            "cec_probe_ms": probe_ms,
            "cec_certificate_ms": certificate_ms,
            "cec_projection_ms": projection_ms,
            "cec_controller_ms": result.get("cec_controller_ms"),
            "cec_depth_sidecar_ms": result.get("cec_depth_sidecar_ms"),
            "cec_context_shadow_ms": shadow_ms,
            "cec_total_decision_ms": (
                (time.perf_counter() - decision_started) * 1000.0),
            "role_label_visible": False,
            "metric_depth_sensor_consumed": False,
            "metric_depth_sensor_consumed_by_policy": False,
        })
        if certificate is not None:
            result["cec_reason"] = certificate.get("reason")
            result["cec_certificate"] = certificate.get("certificate")
        if self.config.emit_handoff_packets:
            result["cec_causal_history_after_decision_sha256"] = (
                self._advance_causal_history(image, goal))
        return result

    def memory_step(self, image: bytes) -> dict[str, Any]:
        """Replay one causal RGB frame into CEC without sampling an action."""
        if not self.initialized or self.reset_required:
            raise PortabilityHubError("router requires a successful reset")
        if not image:
            raise ValueError("image is required")
        try:
            return self._post_json(
                f"{self.config.memnav_url.rstrip('/')}/memory_step",
                "CEC memory replay",
                files={"image": _file("image.jpg", image, "image/jpeg")},
            )
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                f"CEC memory replay failed: {type(exc).__name__}: {exc}") from exc

    def replay_goal_session(
        self, goal: bytes, expected_start_frame: int,
    ) -> dict[str, Any]:
        """Restore one frozen query lifecycle boundary without inference."""
        if not self.initialized or self.reset_required:
            raise PortabilityHubError("router requires a successful reset")
        if not goal:
            raise ValueError("goal is required")
        expected_start_frame = int(expected_start_frame)
        if expected_start_frame < 0:
            raise ValueError("expected_start_frame must be non-negative")
        goal_sha256 = hashlib.sha256(goal).hexdigest()
        if goal_sha256 == self._goal_sha256:
            raise ValueError("replayed goal session did not switch goals")
        try:
            receipt = self._post_json(
                f"{self.config.memnav_url.rstrip('/')}/goal_session_replay",
                "CEC goal-session replay",
                files={"goal": _file("goal.jpg", goal, "image/jpeg")},
                data={"expected_start_frame": str(expected_start_frame)},
            )
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                "CEC goal-session replay failed: "
                f"{type(exc).__name__}: {exc}") from exc
        if (receipt.get("diffusion_sampled") is not False
                or receipt.get("memory_appended") is not False
                or int(receipt.get("goal_start_frame", -1))
                != expected_start_frame
                or receipt.get("goal_session_started") is not True):
            self.reset_required = True
            raise PortabilityHubError(
                "CEC goal-session replay returned an invalid receipt")
        self.query_index += 1
        self._goal_sha256 = goal_sha256
        self._anchor_jpeg = None
        self._anchor_index = None
        self._anchor_sha256 = None
        return {
            **receipt,
            "cec_portability_schema": PORTABILITY_HUB_SCHEMA,
            "cec_goal_sha256": goal_sha256,
            "cec_query_index": int(self.query_index),
            "cec_goal_session_expected_start": True,
            "cec_goal_session_replayed": True,
            "role_label_visible": False,
        }

    def controller_memory_replay(self, image: bytes) -> dict[str, Any]:
        """Replay one decision RGB into bounded controller context only."""
        if not self.initialized or self.reset_required:
            raise PortabilityHubError("router requires a successful reset")
        if not image:
            raise ValueError("image is required")
        try:
            fallback = self._post_json(
                f"{self.config.fallback_navdp_url.rstrip('/')}/memory_replay_step",
                "fallback NavDP memory replay",
                files={"image": _file("image.jpg", image, "image/jpeg")},
            )
            if fallback.get("diffusion_sampled") is not False:
                raise PortabilityHubError(
                    "fallback replay unexpectedly sampled an action")
            self._shadow_vint_context(image)
            return {
                **fallback,
                "cec_portability_schema": PORTABILITY_HUB_SCHEMA,
                "alternate_context_shadowed": self.spec.key in SHORT_CONTEXT_CONTROLLERS,
            }
        except Exception as exc:
            self.reset_required = True
            if isinstance(exc, PortabilityHubError):
                raise
            raise PortabilityHubError(
                "controller memory replay failed: "
                f"{type(exc).__name__}: {exc}") from exc

    def reset_short_context(self, env_id: int) -> dict[str, Any]:
        """Clear bounded controller state without reopening the active goal.

        This endpoint is also used by collision recovery *within* a query.
        Clearing ``_goal_sha256`` here made the next action look like a new
        semantic goal to the hub even though MemNav correctly kept the same
        long-term goal session.  A full episode reset still clears the goal
        identity in :meth:`reset`; a short FIFO reset must not.
        """
        if not self.initialized or self.reset_required:
            raise PortabilityHubError("router requires a successful reset")
        payload = {"env_id": int(env_id)}
        try:
            fallback = self._post_json(
                f"{self.config.fallback_navdp_url.rstrip('/')}/navigator_reset_env",
                "fallback short-context reset", json=payload)
            alternate = None
            if (self.spec.key != "navdp"
                    or self.config.controller_url.rstrip("/")
                    != self.config.fallback_navdp_url.rstrip("/")):
                alternate = self._post_json(
                    f"{self.config.controller_url.rstrip('/')}/navigator_reset_env",
                    "alternate short-context reset", json=payload)
        except Exception as exc:
            self.reset_required = True
            raise PortabilityHubError(
                "short-context reset failed: "
                f"{type(exc).__name__}: {exc}") from exc
        self.last_action_state = "unresolved"
        self._anchor_jpeg = None
        self._anchor_index = None
        self._anchor_sha256 = None
        return {
            "ok": True,
            "algo": "cec_controller_portability",
            "fallback": fallback,
            "alternate": alternate,
            "long_term_cec_history_preserved": True,
            "active_goal_session_preserved": self._goal_sha256 is not None,
        }


def create_app(router: CecControllerPortabilityRouter) -> Flask:
    app = Flask(__name__)
    call_lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "ok": True,
            "schema": PORTABILITY_HUB_SCHEMA,
            "controller": router.spec.key,
            "cec_accept_adapter": router.spec.cec_accept_adapter,
            "initialized": router.initialized,
            "reset_required": router.reset_required,
            "cec_decision_scope": "per_action",
            "cec_last_action_state": router.last_action_state,
            "force_reject_native": bool(router.config.force_reject_native),
            "reject_policy": router.plan.reject_policy,
            "reject_controller": router.plan.fallback_controller,
            "handoff_packets_enabled": bool(
                router.config.emit_handoff_packets),
        })

    @app.post("/navigator_reset")
    def navigator_reset():
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.reset(request.get_json(silent=True) or {}))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/imagegoal_step")
    def imagegoal_step():
        missing = [key for key in ("image", "goal") if key not in request.files]
        if missing:
            return jsonify({"error": "missing files: " + ", ".join(missing)}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.plan_imagegoal(
                    image=request.files["image"].read(),
                    goal=request.files["goal"].read(),
                    form=request.form.to_dict(flat=True),
                ))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                router.reset_required = True
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/memory_step")
    def memory_step():
        if "image" not in request.files:
            return jsonify({"error": "missing file: image"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.memory_step(request.files["image"].read()))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/memory_replay_step")
    def memory_replay_step():
        if "image" not in request.files:
            return jsonify({"error": "missing file: image"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.controller_memory_replay(
                    request.files["image"].read()))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/goal_session_replay")
    def goal_session_replay():
        if "goal" not in request.files:
            return jsonify({"error": "missing file: goal"}), 400
        raw_start = request.form.get("expected_start_frame")
        if raw_start is None:
            return jsonify({"error": "expected_start_frame is required"}), 400
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.replay_goal_session(
                    request.files["goal"].read(), int(raw_start)))
            except (TypeError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    @app.post("/navigator_reset_env")
    def navigator_reset_env():
        payload = request.get_json(silent=True) or {}
        if not call_lock.acquire(blocking=False):
            return jsonify({"error": "hub_busy"}), 409
        try:
            try:
                return jsonify(router.reset_short_context(
                    int(payload.get("env_id", 0))))
            except (TypeError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400
            except PortabilityHubError as exc:
                return jsonify({"error": str(exc), "reset_required": True}), 503
        finally:
            call_lock.release()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18890)
    parser.add_argument("--controller", choices=sorted(SUPPORTED_HTTP_CONTROLLERS),
                        required=True)
    parser.add_argument("--memnav-url", default="http://127.0.0.1:18888")
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--fallback-navdp-url", default="http://127.0.0.1:8888")
    parser.add_argument("--camera-height-m", type=float, required=True)
    parser.add_argument(
        "--reject-policy",
        choices=["shared_native_exact", "controller_native_exact"],
        default="shared_native_exact",
        help=("shared_native_exact rejects to mono NavDP; "
              "controller_native_exact rejects to the selected controller "
              "with the unchanged original ImageGoal"),
    )
    parser.add_argument("--connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--force-reject-native", action="store_true",
        help=("native-control counterfactual: run the identical CEC "
              "probe/certificate pipeline and receipts, but never grant "
              "takeover authority; every action follows --reject-policy"))
    parser.add_argument(
        "--emit-handoff-packets", action="store_true",
        help=("seal every accepted live CEC decision as a single-use, "
              "input-bound handoff packet before controller projection"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("portability hub must bind to loopback; use an SSH tunnel")
    router = CecControllerPortabilityRouter(PortabilityHubConfig(
        controller=args.controller,
        memnav_url=args.memnav_url.rstrip("/"),
        controller_url=args.controller_url.rstrip("/"),
        fallback_navdp_url=args.fallback_navdp_url.rstrip("/"),
        camera_height_m=args.camera_height_m,
        reject_policy=args.reject_policy,
        connect_timeout_s=args.connect_timeout_s,
        request_timeout_s=args.request_timeout_s,
        force_reject_native=bool(args.force_reject_native),
        emit_handoff_packets=bool(args.emit_handoff_packets),
    ))
    create_app(router).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
