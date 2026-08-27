"""Pure contracts for controller-portability comparisons of CEC.

The purpose of this module is to prevent unlike systems from being collapsed
into one misleading success-rate table.  CEC's deployed output is a certified
scale-free bearing.  A controller can therefore participate in one of three
different experiments:

* native ImageGoal evaluation;
* full role-free CEC, which requires both ImageGoal fallback and PointGoal
  execution in the same controller;
* accepted-bearing execution, which requires PointGoal support but scores a
  certificate rejection as uncovered rather than silently substituting a
  different controller.

Map/odometry planners form a fourth, explicitly non-headline diagnostic tier.
This file intentionally has no Torch, Habitat, ROS, or HTTP dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CONTROLLER_PORTABILITY_SCHEMA_VERSION = 2
CEC_FIXED_RADIUS_M = 2.5
CEC_POINTGOAL_UNITS = "lingbot_raw_direction_only"

NATIVE_IMAGEGOAL = "native_imagegoal"
ROLE_FREE_CEC_FULL = "role_free_cec_full"
CEC_BEARING_EXECUTOR = "cec_bearing_executor"
CEC_PROOF_HYBRID = "cec_proof_hybrid"
MAP_BACKEND_DIAGNOSTIC = "map_backend_diagnostic"

PROTOCOLS = frozenset({
    NATIVE_IMAGEGOAL,
    ROLE_FREE_CEC_FULL,
    CEC_BEARING_EXECUTOR,
    CEC_PROOF_HYBRID,
    MAP_BACKEND_DIAGNOSTIC,
})
DEPTH_SOURCES = frozenset({
    "none",
    "metric_sensor",
    "monocular_sidecar",
    "metric_map",
})
QUERY_POPULATIONS = frozenset({"any", "mixed_role", "revisit_only"})
REJECT_POLICIES = frozenset({
    "not_applicable",
    "native_exact",
    "controller_native_exact",
    "score_uncovered",
    "shared_native_exact",
})
CEC_ACCEPT_ADAPTERS = frozenset({
    "bearing_mixedgoal",
    "bearing_pointgoal",
    "verified_anchor_imagegoal",
    "bearing_metric_map_goal",
})


@dataclass(frozen=True)
class ControllerSpec:
    key: str
    display_name: str
    task_interfaces: frozenset[str]
    required_observations: frozenset[str]
    local_path: str | None
    official_repository: str
    official_commit: str
    license_note: str
    controller_family: str
    exact_imagegoal_fallback: bool
    cec_accept_adapter: str


@dataclass(frozen=True)
class ComparisonPlan:
    controller: str
    protocol: str
    depth_source: str
    query_population: str
    reject_policy: str
    fallback_controller: str | None = None
    role_label_visible: bool = False
    uses_oracle_pose: bool = False


@dataclass(frozen=True)
class CecProjection:
    """Auditable projection of one CEC proof into a controller-native goal."""

    proof_sha256: str
    takeover: bool
    controller: str
    adapter: str
    endpoint: str
    payload: Mapping[str, Any]


CONTROLLERS: Mapping[str, ControllerSpec] = {
    "navdp": ControllerSpec(
        key="navdp",
        display_name="NavDP",
        task_interfaces=frozenset({"imagegoal", "pointgoal", "mixed_goal"}),
        required_observations=frozenset({"rgb", "depth"}),
        local_path="NavDP/baselines/navdp",
        official_repository="https://github.com/InternRobotics/NavDP",
        official_commit="3c53e437be03899859f16cfbdbd0951612b8dcad",
        license_note="repository license; frozen project reference controller",
        controller_family="visual_diffusion_policy",
        exact_imagegoal_fallback=True,
        cec_accept_adapter="bearing_mixedgoal",
    ),
    "vint": ControllerSpec(
        key="vint",
        display_name="ViNT",
        task_interfaces=frozenset({"imagegoal", "nogoal"}),
        required_observations=frozenset({"rgb"}),
        local_path="NavDP/baselines/vint",
        official_repository=(
            "https://github.com/robodhruv/visualnav-transformer"),
        official_commit="dca79815b704e5aa9c6bdc3082351f9e3b2848c2",
        license_note="MIT",
        controller_family="visual_goal_conditioned_policy",
        exact_imagegoal_fallback=True,
        cec_accept_adapter="verified_anchor_imagegoal",
    ),
    "gnm": ControllerSpec(
        key="gnm",
        display_name="GNM",
        task_interfaces=frozenset({"imagegoal", "nogoal"}),
        required_observations=frozenset({"rgb"}),
        local_path="NavDP/baselines/gnm",
        official_repository=(
            "https://github.com/robodhruv/visualnav-transformer"),
        official_commit="dca79815b704e5aa9c6bdc3082351f9e3b2848c2",
        license_note="MIT",
        controller_family="visual_goal_conditioned_policy",
        exact_imagegoal_fallback=True,
        cec_accept_adapter="verified_anchor_imagegoal",
    ),
    "nomad": ControllerSpec(
        key="nomad",
        display_name="NoMaD",
        task_interfaces=frozenset({"imagegoal", "nogoal"}),
        required_observations=frozenset({"rgb"}),
        local_path="NavDP/baselines/nomad",
        official_repository=(
            "https://github.com/robodhruv/visualnav-transformer"),
        official_commit="dca79815b704e5aa9c6bdc3082351f9e3b2848c2",
        license_note="MIT",
        controller_family="visual_goal_conditioned_diffusion_policy",
        exact_imagegoal_fallback=True,
        cec_accept_adapter="verified_anchor_imagegoal",
    ),
    "iplanner": ControllerSpec(
        key="iplanner",
        display_name="iPlanner",
        task_interfaces=frozenset({"pointgoal"}),
        required_observations=frozenset({"depth"}),
        local_path="NavDP/baselines/iplanner",
        official_repository="https://github.com/leggedrobotics/iPlanner",
        official_commit="4a8d823ff9d09c3f626b727e7e00484b38f80d49",
        license_note="MIT",
        controller_family="learned_local_path_planner",
        exact_imagegoal_fallback=False,
        cec_accept_adapter="bearing_pointgoal",
    ),
    "viplanner": ControllerSpec(
        key="viplanner",
        display_name="ViPlanner",
        task_interfaces=frozenset({"pointgoal"}),
        required_observations=frozenset({"rgb", "depth", "semantics"}),
        local_path="NavDP/baselines/viplanner",
        official_repository="https://github.com/leggedrobotics/viplanner",
        official_commit="6fcf3c60f6fa3b28b3a11af054d6033825923789",
        license_note="BSD-3-Clause",
        controller_family="semantic_learned_local_path_planner",
        exact_imagegoal_fallback=False,
        cec_accept_adapter="bearing_pointgoal",
    ),
    "ego_planner": ControllerSpec(
        key="ego_planner",
        display_name="EGO-Planner",
        task_interfaces=frozenset({"metric_map_goal"}),
        required_observations=frozenset({"odometry", "occupancy_3d"}),
        local_path=None,
        official_repository="https://github.com/ZJU-FAST-Lab/ego-planner",
        official_commit="bfda51284c8c1b476043255a8145ef925a3778a5",
        license_note="GPL-3.0; keep isolated from the main codebase",
        controller_family="map_based_quadrotor_trajectory_optimizer",
        exact_imagegoal_fallback=False,
        cec_accept_adapter="bearing_metric_map_goal",
    ),
}


def controller_spec(key: str) -> ControllerSpec:
    try:
        return CONTROLLERS[key]
    except KeyError as exc:
        raise ValueError(f"unknown controller {key!r}") from exc


def validate_comparison_plan(plan: ComparisonPlan) -> ControllerSpec:
    """Validate that a proposed result can support its intended claim."""

    if plan.protocol not in PROTOCOLS:
        raise ValueError(f"unknown comparison protocol {plan.protocol!r}")
    if plan.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"unknown depth source {plan.depth_source!r}")
    if plan.query_population not in QUERY_POPULATIONS:
        raise ValueError(
            f"unknown query population {plan.query_population!r}")
    if plan.reject_policy not in REJECT_POLICIES:
        raise ValueError(f"unknown reject policy {plan.reject_policy!r}")
    if plan.role_label_visible:
        raise ValueError("runtime role labels are forbidden in formal comparisons")
    if plan.uses_oracle_pose:
        raise ValueError("oracle pose is permitted only in a separately named upper bound")

    spec = controller_spec(plan.controller)
    needs_depth = "depth" in spec.required_observations
    if needs_depth and plan.depth_source not in {
            "metric_sensor", "monocular_sidecar"}:
        raise ValueError(f"{spec.display_name} requires an explicit depth source")
    if not needs_depth and plan.depth_source not in {"none", "metric_map"}:
        raise ValueError(
            f"{spec.display_name} does not consume controller depth")

    if plan.protocol == NATIVE_IMAGEGOAL:
        if "imagegoal" not in spec.task_interfaces:
            raise ValueError(
                f"{spec.display_name} is not a native ImageGoal controller")
        if plan.reject_policy != "not_applicable":
            raise ValueError("native ImageGoal evaluation has no CEC rejection")
        if plan.query_population not in {"any", "mixed_role"}:
            raise ValueError("native ImageGoal evaluation must not select Revisits")

    elif plan.protocol == ROLE_FREE_CEC_FULL:
        required = {"imagegoal", "pointgoal"}
        if not required.issubset(spec.task_interfaces):
            raise ValueError(
                f"{spec.display_name} cannot execute both CEC branches")
        if not spec.exact_imagegoal_fallback:
            raise ValueError(
                f"{spec.display_name} cannot provide exact ImageGoal fallback")
        if plan.query_population != "mixed_role":
            raise ValueError("full role-free CEC requires a mixed-role population")
        if plan.reject_policy != "native_exact":
            raise ValueError("full role-free CEC requires exact native fallback")

    elif plan.protocol == CEC_BEARING_EXECUTOR:
        if "pointgoal" not in spec.task_interfaces:
            raise ValueError(
                f"{spec.display_name} cannot consume the CEC bearing token")
        if plan.query_population != "revisit_only":
            raise ValueError(
                "bearing-executor portability is a Revisit-only experiment")
        if plan.reject_policy != "score_uncovered":
            raise ValueError(
                "PointGoal-only controllers must score CEC reject as uncovered")

    elif plan.protocol == CEC_PROOF_HYBRID:
        if plan.query_population != "mixed_role":
            raise ValueError("CEC proof portability requires a mixed-role population")
        if plan.reject_policy == "shared_native_exact":
            if plan.fallback_controller != "navdp":
                raise ValueError(
                    "shared CEC proof portability freezes mono NavDP as the "
                    "fallback")
        elif plan.reject_policy == "controller_native_exact":
            if ("imagegoal" not in spec.task_interfaces
                    or not spec.exact_imagegoal_fallback):
                raise ValueError(
                    f"{spec.display_name} cannot provide controller-native "
                    "exact ImageGoal fallback")
            if plan.fallback_controller != spec.key:
                raise ValueError(
                    "controller-native CEC proof portability must fall back "
                    "to the same controller")
        else:
            raise ValueError(
                "CEC proof portability requires either the shared exact "
                "native fallback or controller-native exact fallback")
        if spec.cec_accept_adapter not in CEC_ACCEPT_ADAPTERS:
            raise ValueError(
                f"{spec.display_name} has no audited CEC proof adapter")

    elif plan.protocol == MAP_BACKEND_DIAGNOSTIC:
        if "metric_map_goal" not in spec.task_interfaces:
            raise ValueError(
                f"{spec.display_name} is not a metric-map planner")
        if plan.depth_source != "metric_map":
            raise ValueError("map backend diagnostics require a metric map")
        if plan.reject_policy != "not_applicable":
            raise ValueError("map backend diagnostics do not test CEC rejection")

    return spec


def _finite_direction(value: Any) -> tuple[float, float]:
    try:
        if len(value) != 2 or any(isinstance(item, bool) for item in value):
            raise ValueError
        forward, left = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("CEC direction must contain two finite values") from exc
    norm = math.hypot(forward, left)
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("CEC direction must be finite and non-zero")
    return forward, left


def _normalized_pointgoal(direction: Any) -> tuple[float, float]:
    forward, left = _finite_direction(direction)
    scale = CEC_FIXED_RADIUS_M / math.hypot(forward, left)
    return forward * scale, left * scale


def _proof_identity(proof: Mapping[str, Any]) -> str:
    public = {
        "schema": proof.get("certified_relocalization_schema_version"),
        "frame_idx": proof.get("frame_idx"),
        "accepted": proof.get("accepted"),
        "reason": proof.get("reason"),
        "selected_anchor": proof.get("selected_anchor"),
        "selected_anchor_image_sha256": proof.get(
            "selected_anchor_image_sha256"),
        "direction_vector": proof.get("direction_vector", proof.get("aux_pose")),
        "pointgoal_units": proof.get("pointgoal_units"),
        "certificate": proof.get("certificate"),
    }
    encoded = json.dumps(
        public, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cec_proof_sha256(proof: Mapping[str, Any]) -> str:
    """Return the canonical public identity of one CEC decision.

    The helper is deliberately public so a controller-independent handoff
    artifact can bind exactly the same proof that :func:`project_cec_proof`
    consumes.  Keeping one implementation avoids a subtle but serious audit
    failure where the packet and the live router hash different field sets.
    """

    if not isinstance(proof, Mapping):
        raise ValueError("CEC proof must be a mapping")
    return _proof_identity(proof)


def project_cec_proof(
    controller: str,
    proof: Mapping[str, Any],
    *,
    anchor_jpeg: bytes | None = None,
    shadow_only: bool = False,
    reject_policy: str = "shared_native_exact",
) -> CecProjection:
    """Project the same certified proof into one controller-native interface.

    ``shadow_only`` audits what an accepted certificate would authorize
    without requiring the (expensive) certified anchor JPEG to have been
    fetched -- used by the forced-reject-native baseline, which intentionally
    skips the anchor fetch for an anchor-based controller because the
    takeover is never granted.  The returned projection still reports
    ``takeover=True`` for logging, but its payload cannot carry a real
    hash-bound anchor.

    A valid rejection selects the explicitly frozen ImageGoal fallback for the
    current action.  CEC is evaluated again at the next decision, matching the
    canonical per-action exact-fallback contract.
    """

    if not isinstance(proof, Mapping):
        raise ValueError("CEC proof must be a mapping")
    leaked = sorted(set(proof) & {
        "role", "goal_role", "query_role", "is_revisit", "is_novel",
        "oracle_pose", "gt_pose", "habitat_pose",
    })
    if leaked:
        raise ValueError("CEC proof contains privileged fields: " + ", ".join(leaked))
    if proof.get("accepted") is not True and proof.get("accepted") is not False:
        raise ValueError("CEC proof must contain a boolean accepted decision")
    proof_sha256 = _proof_identity(proof)
    spec = controller_spec(controller)
    if proof.get("accepted") is False:
        if reject_policy == "shared_native_exact":
            reject_controller = "navdp"
            reject_adapter = "shared_native_exact"
        elif reject_policy == "controller_native_exact":
            if ("imagegoal" not in spec.task_interfaces
                    or not spec.exact_imagegoal_fallback):
                raise ValueError(
                    f"{spec.display_name} cannot provide controller-native "
                    "exact ImageGoal fallback")
            reject_controller = spec.key
            reject_adapter = "controller_native_exact"
        else:
            raise ValueError(f"unsupported CEC reject policy {reject_policy!r}")
        return CecProjection(
            proof_sha256=proof_sha256,
            takeover=False,
            controller=reject_controller,
            adapter=reject_adapter,
            endpoint="imagegoal_step",
            payload={"fallback_this_action": True},
        )

    if proof.get("ok") is not True:
        raise ValueError("an accepted CEC proof must have ok=true")
    certificate = proof.get("certificate")
    if not isinstance(certificate, Mapping) or certificate.get("accepted") is not True:
        raise ValueError("an accepted CEC proof requires an accepted certificate")
    if proof.get("pointgoal_units") != CEC_POINTGOAL_UNITS:
        raise ValueError("accepted CEC proof has the wrong direction units")
    raw_selected_anchor = proof.get("selected_anchor")
    if isinstance(raw_selected_anchor, bool) or not isinstance(
            raw_selected_anchor, int):
        raise ValueError("selected anchor must be a non-negative integer")
    try:
        selected_anchor = int(raw_selected_anchor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("accepted CEC proof requires a selected anchor") from exc
    if selected_anchor < 0:
        raise ValueError("selected anchor must be a non-negative integer")
    direction = proof.get("direction_vector", proof.get("aux_pose"))
    fixed = _normalized_pointgoal(direction)
    if spec.cec_accept_adapter in {"bearing_mixedgoal", "bearing_pointgoal"}:
        payload = {
            **fixed_bearing_payload(fixed),
            "cec_selected_anchor": selected_anchor,
        }
        if spec.cec_accept_adapter == "bearing_mixedgoal":
            payload["preserve_original_imagegoal"] = True
            endpoint = "navdp_step_ip_mixgoal"
        else:
            endpoint = "pointgoal_step"
    elif spec.cec_accept_adapter == "verified_anchor_imagegoal":
        if not isinstance(anchor_jpeg, bytes) or not anchor_jpeg:
            if not shadow_only:
                raise ValueError(
                    "ViNT CEC takeover requires the certified anchor JPEG")
            payload = {
                "cec_selected_anchor": selected_anchor,
                "shadow_anchor_unresolved": True,
            }
        else:
            anchor_sha256 = hashlib.sha256(anchor_jpeg).hexdigest()
            advertised = proof.get("selected_anchor_image_sha256")
            if advertised is not None and advertised != anchor_sha256:
                raise ValueError("certified anchor JPEG does not match the proof")
            payload = {
                "cec_selected_anchor": selected_anchor,
                "cec_anchor_sha256": anchor_sha256,
                "goal_source": "certified_history_anchor",
            }
        endpoint = "imagegoal_step"
    elif spec.cec_accept_adapter == "bearing_metric_map_goal":
        payload = {
            "local_metric_goal": [fixed[0], fixed[1], 0.0],
            "cec_selected_anchor": selected_anchor,
            "occupancy_required": True,
        }
        endpoint = "metric_map_goal"
    else:  # guarded by the frozen controller registry
        raise ValueError(f"unsupported CEC adapter {spec.cec_accept_adapter!r}")

    return CecProjection(
        proof_sha256=proof_sha256,
        takeover=True,
        controller=controller,
        adapter=spec.cec_accept_adapter,
        endpoint=endpoint,
        payload=payload,
    )


def fixed_bearing_payload(
    pointgoal: Sequence[float], *, atol: float = 1e-6,
) -> dict[str, list[float]]:
    """Validate a frozen ``[forward, left]`` CEC token for PointGoal APIs."""

    try:
        if len(pointgoal) != 2:
            raise ValueError
        if isinstance(pointgoal[0], bool) or isinstance(pointgoal[1], bool):
            raise ValueError
        forward = float(pointgoal[0])
        left = float(pointgoal[1])
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PointGoal must be two finite numeric values") from exc
    if not math.isfinite(forward) or not math.isfinite(left):
        raise ValueError("PointGoal must be finite")
    norm = math.hypot(forward, left)
    if not math.isclose(norm, CEC_FIXED_RADIUS_M, abs_tol=atol, rel_tol=0.0):
        raise ValueError(
            f"CEC PointGoal norm {norm:.9g} is not frozen radius "
            f"{CEC_FIXED_RADIUS_M:.9g} m")
    return {"goal_x": [forward], "goal_y": [left]}


def is_headline_eligible(plan: ComparisonPlan) -> bool:
    """Return true only after full validation and for sensor-matched tiers."""

    validate_comparison_plan(plan)
    return (
        plan.protocol != MAP_BACKEND_DIAGNOSTIC
        and plan.controller != "ego_planner"
    )


__all__ = [
    "CEC_BEARING_EXECUTOR",
    "CEC_FIXED_RADIUS_M",
    "CEC_POINTGOAL_UNITS",
    "CEC_PROOF_HYBRID",
    "CONTROLLERS",
    "CONTROLLER_PORTABILITY_SCHEMA_VERSION",
    "ComparisonPlan",
    "CecProjection",
    "ControllerSpec",
    "MAP_BACKEND_DIAGNOSTIC",
    "NATIVE_IMAGEGOAL",
    "ROLE_FREE_CEC_FULL",
    "cec_proof_sha256",
    "controller_spec",
    "fixed_bearing_payload",
    "is_headline_eligible",
    "project_cec_proof",
    "validate_comparison_plan",
]
