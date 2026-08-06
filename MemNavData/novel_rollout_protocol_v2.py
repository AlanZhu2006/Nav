"""Pure protocol boundary for paired frozen-NavDP H8/H24 rollouts.

The module deliberately contains no Habitat, ROS, HTTP, Torch, or image
decoding code.  A real adapter must reset the simulator and NavDP server,
replay the exact manifest FIFO, expose queue hashes, plan once per eight-step
commitment, and return diagnostic pursuit observations.  This layer verifies
that contract and derives rollout utility labels; it cannot make an
un-audited adapter causal merely by accepting its receipts.

Each arm starts independently from one frozen decision state.  Native and
residual arms use an identical three-seed schedule.  A residual world subgoal
is immutable across H24, while the adapter must reproject it into the arm's
current local frame at t0, t8, and t16.  FIFO hashes may therefore diverge
after the first commitment, but the complete prepared state and the t0 append
must be byte-identical across arms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from numbers import Integral, Real
from typing import Callable, Mapping, Protocol, Sequence


PROTOCOL_VERSION = "novel_paired_h24_v3"
COMMITMENT_STEPS = 8
COMMITMENT_COUNT = 3
REGRESSION_MARGIN_M = 0.25
USEFUL_MARGIN_M = 0.25


class RolloutProtocolError(RuntimeError):
    """Raised when a rollout receipt cannot prove the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RolloutProtocolError(message)


def _is_finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the strict canonical encoding used by audit sidecars."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RolloutProtocolError(
            f"artifact is not canonical-JSON-compatible: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_runtime_geometry_signature(
    environment_sha256: str,
    navmesh_sha256: str,
    runtime_geometry: RuntimeGeometrySpec,
) -> str:
    """Bind environment, mesh bytes, simulator build, agent, and mesh config."""
    _require(_valid_sha256(environment_sha256), "environment hash is invalid")
    _require(_valid_sha256(navmesh_sha256), "navmesh hash is invalid")
    _validate_runtime_geometry(runtime_geometry)
    return canonical_sha256({
        "environment_sha256": environment_sha256,
        "navmesh_sha256": navmesh_sha256,
        "runtime_geometry": asdict(runtime_geometry),
    })


def canonical_pose_sha256(pose_xz_yaw: Sequence[float]) -> str:
    """Hash the exact finite Habitat x-z-yaw tuple used by this protocol."""
    _require(
        isinstance(pose_xz_yaw, Sequence)
        and not isinstance(pose_xz_yaw, (str, bytes, bytearray))
        and len(pose_xz_yaw) == 3
        and all(_is_finite_real(value) for value in pose_xz_yaw),
        "world pose must be finite [x, z, yaw]",
    )
    x, z, yaw = map(float, pose_xz_yaw)
    _require(-math.pi <= yaw < math.pi, "world yaw must be wrapped to [-pi, pi)")
    return canonical_sha256({"x": x, "z": z, "yaw": yaw})


def world_goal_to_local(
    goal_xz_m: Sequence[float],
    pose_xz_yaw: Sequence[float],
) -> tuple[float, float]:
    """Convert Habitat world x-z to NavDP ``[forward, left]`` exactly."""
    _require(
        isinstance(goal_xz_m, Sequence)
        and not isinstance(goal_xz_m, (str, bytes, bytearray))
        and len(goal_xz_m) == 2
        and all(_is_finite_real(value) for value in goal_xz_m),
        "world subgoal must be finite [x, z]",
    )
    canonical_pose_sha256(pose_xz_yaw)
    goal_x, goal_z = map(float, goal_xz_m)
    current_x, current_z, yaw = map(float, pose_xz_yaw)
    dx, dz = goal_x - current_x, goal_z - current_z
    sine, cosine = math.sin(yaw), math.cos(yaw)
    return (
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    )


@dataclass(frozen=True)
class RuntimeGeometrySpec:
    """Canonical inputs that can change Habitat collision/geodesic geometry.

    ``navmesh_settings_sha256`` must hash the adapter's complete canonical
    NavMeshSettings mapping, including fields not named here.  The explicit
    agent dimensions make the most safety-critical settings human-readable.
    """

    habitat_sim_version: str
    agent_radius_m: float
    agent_height_m: float
    agent_max_climb_m: float
    agent_max_slope_deg: float
    navmesh_source: str
    navmesh_settings_sha256: str


def _validate_runtime_geometry(spec: RuntimeGeometrySpec) -> None:
    _require(isinstance(spec, RuntimeGeometrySpec), "bad runtime geometry type")
    _require(
        isinstance(spec.habitat_sim_version, str)
        and spec.habitat_sim_version
        and spec.habitat_sim_version == spec.habitat_sim_version.strip(),
        "Habitat-Sim version must be a trimmed non-empty string",
    )
    for name in (
        "agent_radius_m",
        "agent_height_m",
        "agent_max_climb_m",
        "agent_max_slope_deg",
    ):
        value = getattr(spec, name)
        _require(
            _is_finite_real(value) and float(value) >= 0.0,
            f"runtime geometry {name} must be finite and non-negative",
        )
    _require(
        float(spec.agent_radius_m) > 0.0
        and float(spec.agent_height_m) > 0.0,
        "agent radius and height must be positive",
    )
    _require(
        spec.navmesh_source in ("loaded_frozen", "recomputed"),
        "navmesh source must say loaded_frozen or recomputed",
    )
    _require(
        _valid_sha256(spec.navmesh_settings_sha256),
        "navmesh settings hash is invalid",
    )


@dataclass(frozen=True)
class FrozenDecisionState:
    """Deployment-visible identity plus hashes for one causal decision."""

    state_id: str
    session_id: str
    goal_epoch: str
    goal_sha256: str
    manifest_fifo_sha256: str
    current_rgb_sha256: str
    current_depth_sha256: str
    start_pose_sha256: str
    environment_id: str
    environment_sha256: str
    navmesh_sha256: str
    runtime_geometry: RuntimeGeometrySpec


@dataclass(frozen=True)
class CandidateArm:
    """One native or metric residual proposal evaluated from the state."""

    candidate_id: str
    candidate_type: str
    world_subgoal_xz_m: tuple[float, float] | None = None

    @property
    def is_native(self) -> bool:
        return self.candidate_type == "native"


@dataclass(frozen=True)
class PreparationReceipt:
    """Evidence returned after reset, FIFO replay, and start-state render."""

    state_id: str
    manifest_fifo_sha256: str
    processed_fifo_sha256: str
    processed_fifo_item_sha256: tuple[str, ...]
    queue_length: int
    current_rgb_sha256: str
    current_depth_sha256: str
    start_pose_sha256: str
    environment_sha256: str
    navmesh_sha256: str
    runtime_geometry_signature: str
    world_pose_xz_yaw: tuple[float, float, float]
    initial_goal_distance_m: float | None
    goal_reachable: bool
    diffusion_calls: int


@dataclass(frozen=True)
class PlanRequest:
    state_id: str
    candidate_id: str
    candidate_type: str
    goal_sha256: str
    commitment_index: int
    diffusion_seed: int
    current_rgb_sha256: str
    current_depth_sha256: str
    current_pose_sha256: str
    current_world_pose_xz_yaw: tuple[float, float, float]
    fixed_world_subgoal_xz_m: tuple[float, float] | None


@dataclass(frozen=True)
class PlanReceipt:
    """Auditable response to one native or mixed point-goal plan call."""

    state_id: str
    candidate_id: str
    candidate_type: str
    goal_sha256: str
    commitment_index: int
    diffusion_seed: int
    current_rgb_sha256: str
    current_depth_sha256: str
    current_pose_sha256: str
    current_world_pose_xz_yaw: tuple[float, float, float]
    fixed_world_subgoal_xz_m: tuple[float, float] | None
    local_subgoal_forward_left_m: tuple[float, float] | None
    plan_sha256: str
    fifo_sha256_before: str
    fifo_sha256_after: str
    queue_length_before: int
    queue_length_after: int
    diffusion_calls_delta: int


@dataclass(frozen=True)
class StepReceipt:
    """One diagnostic pursuit step after a frozen plan response."""

    global_step_index: int
    pose_sha256: str
    world_pose_xz_yaw: tuple[float, float, float]
    rgb_sha256: str
    depth_sha256: str
    goal_distance_m: float | None
    goal_reachable: bool
    moved_m: float
    collision_detected: bool
    full_step_rejected: bool = False
    creep_used: bool = False
    zero_motion: bool = False


@dataclass(frozen=True)
class CommitmentReceipt:
    state_id: str
    candidate_id: str
    commitment_index: int
    plan_sha256: str
    fifo_mutations: int
    steps: tuple[StepReceipt, ...]


@dataclass(frozen=True)
class ArmOutcome:
    """Verified trace and scalar utility for one independently reset arm."""

    candidate_id: str
    candidate_type: str
    preparation: PreparationReceipt
    plans: tuple[PlanReceipt, ...]
    commitments: tuple[CommitmentReceipt, ...]
    rollout_label_valid: bool
    invalid_reason: str | None
    reachable: bool
    initial_goal_distance_m: float | None
    goal_distance_h8_m: float | None
    goal_distance_h24_m: float | None
    geodesic_progress_h8_m: float
    geodesic_progress_h24_m: float
    collision_h8: bool
    trace_sha256: str


@dataclass(frozen=True)
class PairedRolloutArtifact:
    protocol_version: str
    run_signature_sha256: str
    state: FrozenDecisionState
    diffusion_seeds: tuple[int, int, int]
    outcomes: tuple[ArmOutcome, ...]
    labels_by_candidate: Mapping[str, Mapping[str, object]]
    artifact_sha256: str


class RolloutBackend(Protocol):
    """Minimal real/fake adapter consumed by the pure verifier."""

    def prepare_arm(self, state: FrozenDecisionState) -> PreparationReceipt:
        """Reset simulator/server, replay FIFO without diffusion, and render."""

    def plan(self, request: PlanRequest) -> PlanReceipt:
        """Append the current observation exactly once and sample one plan."""

    def pursue(
        self,
        plan: PlanReceipt,
        steps: int,
    ) -> CommitmentReceipt:
        """Execute exactly ``steps`` diagnostic local-controller steps."""


BackendFactory = Callable[[str], RolloutBackend]


def _validate_state(state: FrozenDecisionState) -> None:
    _require(isinstance(state, FrozenDecisionState), "state has wrong type")
    for name in ("state_id", "session_id", "goal_epoch", "environment_id"):
        value = getattr(state, name)
        _require(
            isinstance(value, str) and value and value == value.strip(),
            f"state.{name} must be a trimmed non-empty string",
        )
    for name in (
        "goal_sha256",
        "manifest_fifo_sha256",
        "current_rgb_sha256",
        "current_depth_sha256",
        "start_pose_sha256",
        "environment_sha256",
        "navmesh_sha256",
    ):
        _require(_valid_sha256(getattr(state, name)), f"state.{name} invalid")
    _validate_runtime_geometry(state.runtime_geometry)


def _validate_arm(arm: CandidateArm) -> None:
    _require(isinstance(arm, CandidateArm), "candidate arm has wrong type")
    _require(
        isinstance(arm.candidate_id, str)
        and arm.candidate_id
        and arm.candidate_id == arm.candidate_id.strip(),
        "candidate_id must be a trimmed non-empty string",
    )
    _require(
        arm.candidate_type in ("native", "memory_graph", "frontier"),
        f"unsupported candidate_type {arm.candidate_type!r}",
    )
    if arm.is_native:
        _require(arm.candidate_id == "native", "native id must equal 'native'")
        _require(
            arm.world_subgoal_xz_m is None,
            "native arm cannot carry a metric subgoal",
        )
    else:
        _require(arm.candidate_id != "native", "residual id cannot be native")
        _require(
            isinstance(arm.world_subgoal_xz_m, tuple)
            and len(arm.world_subgoal_xz_m) == 2
            and all(_is_finite_real(value) for value in arm.world_subgoal_xz_m),
            "residual arm requires a finite world x-z subgoal",
        )


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, int, int]:
    _require(
        isinstance(seeds, Sequence)
        and not isinstance(seeds, (str, bytes, bytearray))
        and len(seeds) == COMMITMENT_COUNT,
        f"exactly {COMMITMENT_COUNT} diffusion seeds are required",
    )
    frozen: list[int] = []
    for index, seed in enumerate(seeds):
        _require(
            isinstance(seed, Integral) and not isinstance(seed, bool)
            and 0 <= int(seed) < 2**63,
            f"diffusion seed {index} is invalid",
        )
        frozen.append(int(seed))
    _require(len(set(frozen)) == len(frozen), "diffusion seeds must be unique")
    return tuple(frozen)  # type: ignore[return-value]


def _validate_preparation(
    state: FrozenDecisionState,
    receipt: PreparationReceipt,
) -> None:
    _require(isinstance(receipt, PreparationReceipt), "bad preparation type")
    _require(receipt.state_id == state.state_id, "prepared wrong state")
    _require(
        receipt.manifest_fifo_sha256 == state.manifest_fifo_sha256,
        "adapter did not echo the manifest FIFO hash",
    )
    _require(
        _valid_sha256(receipt.processed_fifo_sha256),
        "processed FIFO hash is invalid",
    )
    _require(
        isinstance(receipt.processed_fifo_item_sha256, tuple)
        and all(_valid_sha256(value)
                for value in receipt.processed_fifo_item_sha256),
        "processed FIFO item hashes are invalid",
    )
    _require(
        isinstance(receipt.queue_length, Integral)
        and not isinstance(receipt.queue_length, bool)
        and int(receipt.queue_length) == len(receipt.processed_fifo_item_sha256),
        "prepared queue length disagrees with item hashes",
    )
    _require(
        receipt.current_rgb_sha256 == state.current_rgb_sha256,
        "start RGB differs from frozen state",
    )
    _require(
        receipt.current_depth_sha256 == state.current_depth_sha256,
        "start depth differs from frozen state",
    )
    _require(
        receipt.start_pose_sha256 == state.start_pose_sha256,
        "start pose differs from frozen state",
    )
    _require(
        receipt.environment_sha256 == state.environment_sha256,
        "prepared environment differs from frozen state",
    )
    _require(
        receipt.navmesh_sha256 == state.navmesh_sha256,
        "prepared navmesh differs from frozen state",
    )
    _require(
        receipt.runtime_geometry_signature
        == canonical_runtime_geometry_signature(
            state.environment_sha256,
            state.navmesh_sha256,
            state.runtime_geometry,
        ),
        "runtime geometry signature disagrees with environment/navmesh",
    )
    _require(
        canonical_pose_sha256(receipt.world_pose_xz_yaw)
        == receipt.start_pose_sha256,
        "numeric start pose disagrees with its canonical hash",
    )
    _require(
        isinstance(receipt.diffusion_calls, Integral)
        and not isinstance(receipt.diffusion_calls, bool)
        and int(receipt.diffusion_calls) == 0,
        "FIFO preparation must not call diffusion",
    )
    _require(
        type(receipt.goal_reachable) is bool,
        "preparation reachability must be boolean",
    )
    if receipt.goal_reachable:
        _require(
            _is_finite_real(receipt.initial_goal_distance_m)
            and float(receipt.initial_goal_distance_m) >= 0.0,
            "reachable start requires a finite non-negative goal distance",
        )
    else:
        _require(
            receipt.initial_goal_distance_m is None,
            "unreachable start must not report a goal distance",
        )


def _validate_plan(
    request: PlanRequest,
    receipt: PlanReceipt,
    expected_fifo_before: str,
    expected_queue_length: int,
) -> None:
    _require(isinstance(receipt, PlanReceipt), "bad plan receipt type")
    for name in (
        "state_id",
        "candidate_id",
        "candidate_type",
        "goal_sha256",
        "commitment_index",
        "diffusion_seed",
        "current_rgb_sha256",
        "current_depth_sha256",
        "current_pose_sha256",
        "current_world_pose_xz_yaw",
        "fixed_world_subgoal_xz_m",
    ):
        _require(
            getattr(receipt, name) == getattr(request, name),
            f"plan receipt changed {name}",
        )
    _require(_valid_sha256(receipt.plan_sha256), "plan hash is invalid")
    _require(
        canonical_pose_sha256(receipt.current_world_pose_xz_yaw)
        == receipt.current_pose_sha256,
        "plan numeric pose disagrees with its canonical hash",
    )
    _require(
        receipt.fifo_sha256_before == expected_fifo_before,
        "plan started from an unexpected FIFO",
    )
    _require(
        _valid_sha256(receipt.fifo_sha256_after),
        "post-plan FIFO hash is invalid",
    )
    _require(
        int(receipt.queue_length_before) == int(expected_queue_length),
        "plan started from an unexpected FIFO length",
    )
    _require(
        isinstance(receipt.queue_length_after, Integral)
        and not isinstance(receipt.queue_length_after, bool)
        and int(receipt.queue_length_after)
        == min(8, int(expected_queue_length) + 1),
        "plan must append exactly one item to the eight-frame FIFO",
    )
    _require(
        isinstance(receipt.diffusion_calls_delta, Integral)
        and not isinstance(receipt.diffusion_calls_delta, bool)
        and int(receipt.diffusion_calls_delta) == 1,
        "each commitment must call diffusion exactly once",
    )
    if request.candidate_type == "native":
        _require(
            receipt.local_subgoal_forward_left_m is None,
            "native plan cannot report a point-goal projection",
        )
    else:
        local = receipt.local_subgoal_forward_left_m
        _require(
            isinstance(local, tuple)
            and len(local) == 2
            and all(_is_finite_real(value) for value in local),
            "residual plan requires a finite current-frame projection",
        )
        expected_local = world_goal_to_local(
            request.fixed_world_subgoal_xz_m,
            request.current_world_pose_xz_yaw,
        )
        _require(
            all(math.isclose(
                float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7)
                for actual, expected in zip(local, expected_local)),
            "residual local projection disagrees with fixed world subgoal",
        )


def _validate_commitment(
    state_id: str,
    candidate_id: str,
    commitment_index: int,
    plan: PlanReceipt,
    receipt: CommitmentReceipt,
) -> None:
    _require(isinstance(receipt, CommitmentReceipt), "bad commitment type")
    _require(receipt.state_id == state_id, "commitment changed state")
    _require(receipt.candidate_id == candidate_id, "commitment changed arm")
    _require(
        receipt.commitment_index == commitment_index,
        "commitment index mismatch",
    )
    _require(receipt.plan_sha256 == plan.plan_sha256, "executed wrong plan")
    _require(receipt.fifo_mutations == 0, "pursuit mutated NavDP FIFO")
    _require(
        isinstance(receipt.steps, tuple)
        and len(receipt.steps) == COMMITMENT_STEPS,
        f"commitment must contain exactly {COMMITMENT_STEPS} steps",
    )
    base_step = commitment_index * COMMITMENT_STEPS
    for offset, step in enumerate(receipt.steps):
        _require(isinstance(step, StepReceipt), "bad pursuit-step type")
        _require(
            step.global_step_index == base_step + offset,
            "pursuit step index is not contiguous",
        )
        for name in ("pose_sha256", "rgb_sha256", "depth_sha256"):
            _require(_valid_sha256(getattr(step, name)), f"step {name} invalid")
        _require(
            canonical_pose_sha256(step.world_pose_xz_yaw) == step.pose_sha256,
            "step numeric pose disagrees with its canonical hash",
        )
        _require(
            _is_finite_real(step.moved_m) and float(step.moved_m) >= 0.0,
            "step movement must be finite and non-negative",
        )
        for name in (
            "goal_reachable",
            "collision_detected",
            "full_step_rejected",
            "creep_used",
            "zero_motion",
        ):
            _require(type(getattr(step, name)) is bool, f"step {name} invalid")
        if step.goal_reachable:
            _require(
                _is_finite_real(step.goal_distance_m)
                and float(step.goal_distance_m) >= 0.0,
                "reachable step requires a finite non-negative distance",
            )
        else:
            _require(
                step.goal_distance_m is None,
                "unreachable step must not report a goal distance",
            )
        if step.full_step_rejected or step.creep_used:
            _require(
                step.collision_detected,
                "rejection/creep diagnostics require collision_detected",
            )


def run_candidate_arm(
    backend: RolloutBackend,
    state: FrozenDecisionState,
    arm: CandidateArm,
    diffusion_seeds: Sequence[int],
) -> ArmOutcome:
    """Run and verify one independently prepared candidate arm."""
    _validate_state(state)
    _validate_arm(arm)
    seeds = _validate_seeds(diffusion_seeds)
    preparation = backend.prepare_arm(state)
    _validate_preparation(state, preparation)

    plans: list[PlanReceipt] = []
    commitments: list[CommitmentReceipt] = []
    fifo_sha = preparation.processed_fifo_sha256
    queue_length = int(preparation.queue_length)
    current_rgb_sha = preparation.current_rgb_sha256
    current_depth_sha = preparation.current_depth_sha256
    current_pose_sha = preparation.start_pose_sha256
    current_world_pose = preparation.world_pose_xz_yaw
    for commitment_index, diffusion_seed in enumerate(seeds):
        request = PlanRequest(
            state_id=state.state_id,
            candidate_id=arm.candidate_id,
            candidate_type=arm.candidate_type,
            goal_sha256=state.goal_sha256,
            commitment_index=commitment_index,
            diffusion_seed=diffusion_seed,
            current_rgb_sha256=current_rgb_sha,
            current_depth_sha256=current_depth_sha,
            current_pose_sha256=current_pose_sha,
            current_world_pose_xz_yaw=current_world_pose,
            fixed_world_subgoal_xz_m=arm.world_subgoal_xz_m,
        )
        plan = backend.plan(request)
        _validate_plan(request, plan, fifo_sha, queue_length)
        commitment = backend.pursue(plan, COMMITMENT_STEPS)
        _validate_commitment(
            state.state_id,
            arm.candidate_id,
            commitment_index,
            plan,
            commitment,
        )
        plans.append(plan)
        commitments.append(commitment)
        fifo_sha = plan.fifo_sha256_after
        queue_length = int(plan.queue_length_after)
        endpoint = commitment.steps[-1]
        current_rgb_sha = endpoint.rgb_sha256
        current_depth_sha = endpoint.depth_sha256
        current_pose_sha = endpoint.pose_sha256
        current_world_pose = endpoint.world_pose_xz_yaw

    all_steps = tuple(
        step for commitment in commitments for step in commitment.steps)
    reachable = bool(
        preparation.goal_reachable
        and all(step.goal_reachable for step in all_steps)
    )
    invalid_reason = None if reachable else "unreachable_goal_distance"
    d0 = preparation.initial_goal_distance_m
    d8 = all_steps[COMMITMENT_STEPS - 1].goal_distance_m
    d24 = all_steps[-1].goal_distance_m
    if reachable:
        _require(d0 is not None and d8 is not None and d24 is not None,
                 "reachable trace lost a distance")
        progress_h8 = float(d0) - float(d8)
        progress_h24 = float(d0) - float(d24)
    else:
        d0 = d0 if _is_finite_real(d0) else None
        d8 = d8 if _is_finite_real(d8) else None
        d24 = d24 if _is_finite_real(d24) else None
        progress_h8 = 0.0
        progress_h24 = 0.0
    collision_h8 = any(
        step.collision_detected for step in commitments[0].steps)
    trace_payload = {
        "candidate": asdict(arm),
        "preparation": asdict(preparation),
        "plans": [asdict(plan) for plan in plans],
        "commitments": [asdict(commitment) for commitment in commitments],
    }
    return ArmOutcome(
        candidate_id=arm.candidate_id,
        candidate_type=arm.candidate_type,
        preparation=preparation,
        plans=tuple(plans),
        commitments=tuple(commitments),
        rollout_label_valid=reachable,
        invalid_reason=invalid_reason,
        reachable=reachable,
        initial_goal_distance_m=(float(d0) if d0 is not None else None),
        goal_distance_h8_m=(float(d8) if d8 is not None else None),
        goal_distance_h24_m=(float(d24) if d24 is not None else None),
        geodesic_progress_h8_m=progress_h8,
        geodesic_progress_h24_m=progress_h24,
        collision_h8=collision_h8 if reachable else False,
        trace_sha256=canonical_sha256(trace_payload),
    )


def _preparation_signature(receipt: PreparationReceipt) -> tuple[object, ...]:
    return (
        receipt.state_id,
        receipt.manifest_fifo_sha256,
        receipt.processed_fifo_sha256,
        receipt.processed_fifo_item_sha256,
        receipt.queue_length,
        receipt.current_rgb_sha256,
        receipt.current_depth_sha256,
        receipt.start_pose_sha256,
        receipt.environment_sha256,
        receipt.navmesh_sha256,
        receipt.runtime_geometry_signature,
        receipt.world_pose_xz_yaw,
        receipt.initial_goal_distance_m,
        receipt.goal_reachable,
        receipt.diffusion_calls,
    )


def _neutral_invalid_labels() -> dict[str, object]:
    return {
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": 0.0,
        "harm": False,
        "useful": False,
        "reachable": False,
        "collision_h8": False,
        "regression_h24": False,
        "rollout_label_valid": False,
    }


def _derive_labels(
    outcomes: Sequence[ArmOutcome],
) -> dict[str, dict[str, object]]:
    native = next(outcome for outcome in outcomes if outcome.candidate_id == "native")
    labels: dict[str, dict[str, object]] = {}
    if not native.rollout_label_valid:
        return {outcome.candidate_id: _neutral_invalid_labels()
                for outcome in outcomes}
    for outcome in outcomes:
        if not outcome.rollout_label_valid:
            labels[outcome.candidate_id] = _neutral_invalid_labels()
            continue
        advantage = (
            outcome.geodesic_progress_h24_m
            - native.geodesic_progress_h24_m
        )
        if outcome.candidate_id == "native":
            advantage = 0.0
            regression = False
            useful = False
        else:
            regression = advantage <= -REGRESSION_MARGIN_M
            useful = bool(
                advantage >= USEFUL_MARGIN_M
                and not outcome.collision_h8
                and not regression
            )
        harm = bool(outcome.collision_h8 or regression)
        labels[outcome.candidate_id] = {
            "geodesic_progress_h8_m": outcome.geodesic_progress_h8_m,
            "geodesic_progress_h24_m": outcome.geodesic_progress_h24_m,
            "advantage_h24_m": advantage,
            "harm": harm,
            "useful": useful,
            "reachable": True,
            "collision_h8": outcome.collision_h8,
            "regression_h24": regression,
            "rollout_label_valid": True,
        }
    return labels


def _validate_outcome_trace(
    state: FrozenDecisionState,
    diffusion_seeds: tuple[int, int, int],
    outcome: ArmOutcome,
) -> None:
    """Re-verify a deserialized outcome without trusting stored summaries."""
    _require(isinstance(outcome, ArmOutcome), "bad arm outcome type")
    _require(
        isinstance(outcome.plans, tuple)
        and len(outcome.plans) == COMMITMENT_COUNT,
        f"outcome must contain exactly {COMMITMENT_COUNT} plans",
    )
    _require(
        isinstance(outcome.commitments, tuple)
        and len(outcome.commitments) == COMMITMENT_COUNT,
        f"outcome must contain exactly {COMMITMENT_COUNT} commitments",
    )
    first_plan = outcome.plans[0]
    arm = CandidateArm(
        candidate_id=outcome.candidate_id,
        candidate_type=outcome.candidate_type,
        world_subgoal_xz_m=first_plan.fixed_world_subgoal_xz_m,
    )
    _validate_arm(arm)
    _validate_preparation(state, outcome.preparation)

    fifo_sha = outcome.preparation.processed_fifo_sha256
    queue_length = int(outcome.preparation.queue_length)
    current_rgb_sha = outcome.preparation.current_rgb_sha256
    current_depth_sha = outcome.preparation.current_depth_sha256
    current_pose_sha = outcome.preparation.start_pose_sha256
    current_world_pose = outcome.preparation.world_pose_xz_yaw
    for commitment_index, diffusion_seed in enumerate(diffusion_seeds):
        request = PlanRequest(
            state_id=state.state_id,
            candidate_id=arm.candidate_id,
            candidate_type=arm.candidate_type,
            goal_sha256=state.goal_sha256,
            commitment_index=commitment_index,
            diffusion_seed=diffusion_seed,
            current_rgb_sha256=current_rgb_sha,
            current_depth_sha256=current_depth_sha,
            current_pose_sha256=current_pose_sha,
            current_world_pose_xz_yaw=current_world_pose,
            fixed_world_subgoal_xz_m=arm.world_subgoal_xz_m,
        )
        plan = outcome.plans[commitment_index]
        _validate_plan(request, plan, fifo_sha, queue_length)
        commitment = outcome.commitments[commitment_index]
        _validate_commitment(
            state.state_id,
            arm.candidate_id,
            commitment_index,
            plan,
            commitment,
        )
        fifo_sha = plan.fifo_sha256_after
        queue_length = int(plan.queue_length_after)
        endpoint = commitment.steps[-1]
        current_rgb_sha = endpoint.rgb_sha256
        current_depth_sha = endpoint.depth_sha256
        current_pose_sha = endpoint.pose_sha256
        current_world_pose = endpoint.world_pose_xz_yaw

    steps = tuple(
        step
        for commitment in outcome.commitments
        for step in commitment.steps
    )
    reachable = bool(
        outcome.preparation.goal_reachable
        and all(step.goal_reachable for step in steps)
    )
    expected_reason = None if reachable else "unreachable_goal_distance"
    d0 = outcome.preparation.initial_goal_distance_m
    d8 = steps[COMMITMENT_STEPS - 1].goal_distance_m
    d24 = steps[-1].goal_distance_m
    if reachable:
        _require(d0 is not None and d8 is not None and d24 is not None,
                 "reachable trace lost a distance")
        progress_h8 = float(d0) - float(d8)
        progress_h24 = float(d0) - float(d24)
    else:
        d0 = d0 if _is_finite_real(d0) else None
        d8 = d8 if _is_finite_real(d8) else None
        d24 = d24 if _is_finite_real(d24) else None
        progress_h8 = 0.0
        progress_h24 = 0.0
    expected_collision = (
        any(step.collision_detected for step in outcome.commitments[0].steps)
        if reachable else False
    )
    expected_values = {
        "rollout_label_valid": reachable,
        "invalid_reason": expected_reason,
        "reachable": reachable,
        "initial_goal_distance_m": float(d0) if d0 is not None else None,
        "goal_distance_h8_m": float(d8) if d8 is not None else None,
        "goal_distance_h24_m": float(d24) if d24 is not None else None,
        "geodesic_progress_h8_m": progress_h8,
        "geodesic_progress_h24_m": progress_h24,
        "collision_h8": expected_collision,
    }
    for name, expected in expected_values.items():
        _require(
            getattr(outcome, name) == expected,
            f"outcome summary {name} disagrees with verified trace",
        )
    trace_payload = {
        "candidate": asdict(arm),
        "preparation": asdict(outcome.preparation),
        "plans": [asdict(plan) for plan in outcome.plans],
        "commitments": [
            asdict(commitment) for commitment in outcome.commitments
        ],
    }
    _require(
        canonical_sha256(trace_payload) == outcome.trace_sha256,
        "outcome trace hash mismatch",
    )


def collect_paired_rollouts(
    backend_factory: BackendFactory,
    state: FrozenDecisionState,
    arms: Sequence[CandidateArm],
    diffusion_seeds: Sequence[int],
    *,
    run_signature_sha256: str,
) -> PairedRolloutArtifact:
    """Collect same-state arms and reject cross-arm contamination.

    ``arms`` may be supplied in any order; the returned artifact is canonical:
    native first, followed by residual candidate id.  This supports an explicit
    arm-order reversal smoke without changing the artifact hash.
    """
    _validate_state(state)
    seeds = _validate_seeds(diffusion_seeds)
    _require(_valid_sha256(run_signature_sha256), "run signature is invalid")
    _require(
        isinstance(arms, Sequence)
        and not isinstance(arms, (str, bytes, bytearray))
        and len(arms) >= 2,
        "paired rollout requires native and at least one residual",
    )
    for arm in arms:
        _validate_arm(arm)
    ids = [arm.candidate_id for arm in arms]
    _require(len(ids) == len(set(ids)), "candidate ids must be unique")
    _require(ids.count("native") == 1, "exactly one native arm is required")

    outcomes = [
        run_candidate_arm(backend_factory(arm.candidate_id), state, arm, seeds)
        for arm in arms
    ]
    preparation_signatures = {
        _preparation_signature(outcome.preparation) for outcome in outcomes
    }
    _require(
        len(preparation_signatures) == 1,
        "arms did not start from byte-identical state/FIFO/distance",
    )
    t0_appends = {
        (
            outcome.plans[0].fifo_sha256_after,
            outcome.plans[0].queue_length_after,
        )
        for outcome in outcomes
    }
    _require(
        len(t0_appends) == 1,
        "native/residual t0 observation append is not byte-identical",
    )
    canonical_outcomes = tuple(sorted(
        outcomes,
        key=lambda outcome: (
            outcome.candidate_id != "native",
            outcome.candidate_id,
        ),
    ))
    labels = _derive_labels(canonical_outcomes)
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "run_signature_sha256": run_signature_sha256,
        "state": asdict(state),
        "diffusion_seeds": list(seeds),
        "outcomes": [asdict(outcome) for outcome in canonical_outcomes],
        "labels_by_candidate": labels,
    }
    artifact_sha = canonical_sha256(body)
    return PairedRolloutArtifact(
        protocol_version=PROTOCOL_VERSION,
        run_signature_sha256=run_signature_sha256,
        state=state,
        diffusion_seeds=seeds,
        outcomes=canonical_outcomes,
        labels_by_candidate=labels,
        artifact_sha256=artifact_sha,
    )


def collect_native_rollout(
    backend_factory: BackendFactory,
    state: FrozenDecisionState,
    diffusion_seeds: Sequence[int],
    *,
    run_signature_sha256: str,
) -> PairedRolloutArtifact:
    """Collect an explicitly native-only audit row.

    Native-only rows are useful for independent controller health checks.  The
    paired collector intentionally continues to require at least one residual
    arm, so a native-only row cannot masquerade as a training comparison.
    """
    _validate_state(state)
    seeds = _validate_seeds(diffusion_seeds)
    _require(_valid_sha256(run_signature_sha256), "run signature is invalid")
    native = CandidateArm("native", "native")
    outcome = run_candidate_arm(
        backend_factory(native.candidate_id), state, native, seeds)
    labels = _derive_labels((outcome,))
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "run_signature_sha256": run_signature_sha256,
        "state": asdict(state),
        "diffusion_seeds": list(seeds),
        "outcomes": [asdict(outcome)],
        "labels_by_candidate": labels,
    }
    artifact_sha = canonical_sha256(body)
    return PairedRolloutArtifact(
        protocol_version=PROTOCOL_VERSION,
        run_signature_sha256=run_signature_sha256,
        state=state,
        diffusion_seeds=seeds,
        outcomes=(outcome,),
        labels_by_candidate=labels,
        artifact_sha256=artifact_sha,
    )


def _strict_mapping(
    value: object,
    expected_keys: Sequence[str],
    context: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    keys = set(value.keys())
    _require(
        all(isinstance(key, str) for key in keys),
        f"{context} contains a non-string key",
    )
    expected = set(expected_keys)
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    _require(not unknown, f"{context} contains unknown keys: {unknown}")
    _require(not missing, f"{context} is missing keys: {missing}")
    return value


def _strict_string(value: object, context: str) -> str:
    _require(isinstance(value, str), f"{context} must be a string")
    return value


def _strict_int(value: object, context: str) -> int:
    _require(
        isinstance(value, Integral) and not isinstance(value, bool),
        f"{context} must be an integer",
    )
    return int(value)


def _strict_bool(value: object, context: str) -> bool:
    _require(type(value) is bool, f"{context} must be boolean")
    return value


def _strict_float(value: object, context: str) -> float:
    _require(_is_finite_real(value), f"{context} must be finite numeric")
    return float(value)


def _strict_optional_float(value: object, context: str) -> float | None:
    return None if value is None else _strict_float(value, context)


def _strict_sequence(value: object, context: str) -> Sequence[object]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{context} must be an array",
    )
    return value


def _strict_float_tuple(
    value: object,
    length: int,
    context: str,
) -> tuple[float, ...]:
    sequence = _strict_sequence(value, context)
    _require(len(sequence) == length, f"{context} must have length {length}")
    return tuple(
        _strict_float(item, f"{context}[{index}]")
        for index, item in enumerate(sequence)
    )


def _parse_runtime_geometry(value: object) -> RuntimeGeometrySpec:
    keys = (
        "habitat_sim_version", "agent_radius_m", "agent_height_m",
        "agent_max_climb_m", "agent_max_slope_deg", "navmesh_source",
        "navmesh_settings_sha256",
    )
    row = _strict_mapping(value, keys, "state.runtime_geometry")
    spec = RuntimeGeometrySpec(
        habitat_sim_version=_strict_string(
            row["habitat_sim_version"],
            "state.runtime_geometry.habitat_sim_version",
        ),
        agent_radius_m=_strict_float(
            row["agent_radius_m"], "state.runtime_geometry.agent_radius_m"),
        agent_height_m=_strict_float(
            row["agent_height_m"], "state.runtime_geometry.agent_height_m"),
        agent_max_climb_m=_strict_float(
            row["agent_max_climb_m"],
            "state.runtime_geometry.agent_max_climb_m",
        ),
        agent_max_slope_deg=_strict_float(
            row["agent_max_slope_deg"],
            "state.runtime_geometry.agent_max_slope_deg",
        ),
        navmesh_source=_strict_string(
            row["navmesh_source"], "state.runtime_geometry.navmesh_source"),
        navmesh_settings_sha256=_strict_string(
            row["navmesh_settings_sha256"],
            "state.runtime_geometry.navmesh_settings_sha256",
        ),
    )
    _validate_runtime_geometry(spec)
    return spec


def _parse_state(value: object) -> FrozenDecisionState:
    keys = (
        "state_id", "session_id", "goal_epoch", "goal_sha256",
        "manifest_fifo_sha256", "current_rgb_sha256",
        "current_depth_sha256", "start_pose_sha256", "environment_id",
        "environment_sha256", "navmesh_sha256", "runtime_geometry",
    )
    row = _strict_mapping(value, keys, "state")
    string_keys = keys[:-1]
    result = FrozenDecisionState(
        **{
            key: _strict_string(row[key], f"state.{key}")
            for key in string_keys
        },
        runtime_geometry=_parse_runtime_geometry(row["runtime_geometry"]),
    )
    _validate_state(result)
    return result


def _parse_preparation(value: object) -> PreparationReceipt:
    keys = (
        "state_id", "manifest_fifo_sha256", "processed_fifo_sha256",
        "processed_fifo_item_sha256", "queue_length", "current_rgb_sha256",
        "current_depth_sha256", "start_pose_sha256", "environment_sha256",
        "navmesh_sha256", "runtime_geometry_signature",
        "world_pose_xz_yaw", "initial_goal_distance_m", "goal_reachable",
        "diffusion_calls",
    )
    row = _strict_mapping(value, keys, "preparation")
    item_hashes = _strict_sequence(
        row["processed_fifo_item_sha256"],
        "preparation.processed_fifo_item_sha256",
    )
    return PreparationReceipt(
        state_id=_strict_string(row["state_id"], "preparation.state_id"),
        manifest_fifo_sha256=_strict_string(
            row["manifest_fifo_sha256"], "preparation.manifest_fifo_sha256"),
        processed_fifo_sha256=_strict_string(
            row["processed_fifo_sha256"], "preparation.processed_fifo_sha256"),
        processed_fifo_item_sha256=tuple(
            _strict_string(item, "preparation FIFO item hash")
            for item in item_hashes
        ),
        queue_length=_strict_int(row["queue_length"], "preparation.queue_length"),
        current_rgb_sha256=_strict_string(
            row["current_rgb_sha256"], "preparation.current_rgb_sha256"),
        current_depth_sha256=_strict_string(
            row["current_depth_sha256"], "preparation.current_depth_sha256"),
        start_pose_sha256=_strict_string(
            row["start_pose_sha256"], "preparation.start_pose_sha256"),
        environment_sha256=_strict_string(
            row["environment_sha256"], "preparation.environment_sha256"),
        navmesh_sha256=_strict_string(
            row["navmesh_sha256"], "preparation.navmesh_sha256"),
        runtime_geometry_signature=_strict_string(
            row["runtime_geometry_signature"],
            "preparation.runtime_geometry_signature",
        ),
        world_pose_xz_yaw=_strict_float_tuple(
            row["world_pose_xz_yaw"], 3, "preparation.world_pose_xz_yaw"
        ),  # type: ignore[arg-type]
        initial_goal_distance_m=_strict_optional_float(
            row["initial_goal_distance_m"],
            "preparation.initial_goal_distance_m",
        ),
        goal_reachable=_strict_bool(
            row["goal_reachable"], "preparation.goal_reachable"),
        diffusion_calls=_strict_int(
            row["diffusion_calls"], "preparation.diffusion_calls"),
    )


def _parse_plan(value: object) -> PlanReceipt:
    keys = (
        "state_id", "candidate_id", "candidate_type", "goal_sha256",
        "commitment_index", "diffusion_seed", "current_rgb_sha256",
        "current_depth_sha256", "current_pose_sha256",
        "current_world_pose_xz_yaw", "fixed_world_subgoal_xz_m",
        "local_subgoal_forward_left_m", "plan_sha256", "fifo_sha256_before",
        "fifo_sha256_after", "queue_length_before", "queue_length_after",
        "diffusion_calls_delta",
    )
    row = _strict_mapping(value, keys, "plan")
    fixed = row["fixed_world_subgoal_xz_m"]
    local = row["local_subgoal_forward_left_m"]
    return PlanReceipt(
        state_id=_strict_string(row["state_id"], "plan.state_id"),
        candidate_id=_strict_string(row["candidate_id"], "plan.candidate_id"),
        candidate_type=_strict_string(
            row["candidate_type"], "plan.candidate_type"),
        goal_sha256=_strict_string(row["goal_sha256"], "plan.goal_sha256"),
        commitment_index=_strict_int(
            row["commitment_index"], "plan.commitment_index"),
        diffusion_seed=_strict_int(row["diffusion_seed"], "plan.diffusion_seed"),
        current_rgb_sha256=_strict_string(
            row["current_rgb_sha256"], "plan.current_rgb_sha256"),
        current_depth_sha256=_strict_string(
            row["current_depth_sha256"], "plan.current_depth_sha256"),
        current_pose_sha256=_strict_string(
            row["current_pose_sha256"], "plan.current_pose_sha256"),
        current_world_pose_xz_yaw=_strict_float_tuple(
            row["current_world_pose_xz_yaw"], 3,
            "plan.current_world_pose_xz_yaw",
        ),  # type: ignore[arg-type]
        fixed_world_subgoal_xz_m=(
            None if fixed is None else _strict_float_tuple(
                fixed, 2, "plan.fixed_world_subgoal_xz_m")
        ),  # type: ignore[arg-type]
        local_subgoal_forward_left_m=(
            None if local is None else _strict_float_tuple(
                local, 2, "plan.local_subgoal_forward_left_m")
        ),  # type: ignore[arg-type]
        plan_sha256=_strict_string(row["plan_sha256"], "plan.plan_sha256"),
        fifo_sha256_before=_strict_string(
            row["fifo_sha256_before"], "plan.fifo_sha256_before"),
        fifo_sha256_after=_strict_string(
            row["fifo_sha256_after"], "plan.fifo_sha256_after"),
        queue_length_before=_strict_int(
            row["queue_length_before"], "plan.queue_length_before"),
        queue_length_after=_strict_int(
            row["queue_length_after"], "plan.queue_length_after"),
        diffusion_calls_delta=_strict_int(
            row["diffusion_calls_delta"], "plan.diffusion_calls_delta"),
    )


def _parse_step(value: object) -> StepReceipt:
    keys = (
        "global_step_index", "pose_sha256", "world_pose_xz_yaw",
        "rgb_sha256", "depth_sha256", "goal_distance_m", "goal_reachable",
        "moved_m", "collision_detected", "full_step_rejected", "creep_used",
        "zero_motion",
    )
    row = _strict_mapping(value, keys, "step")
    return StepReceipt(
        global_step_index=_strict_int(
            row["global_step_index"], "step.global_step_index"),
        pose_sha256=_strict_string(row["pose_sha256"], "step.pose_sha256"),
        world_pose_xz_yaw=_strict_float_tuple(
            row["world_pose_xz_yaw"], 3, "step.world_pose_xz_yaw"
        ),  # type: ignore[arg-type]
        rgb_sha256=_strict_string(row["rgb_sha256"], "step.rgb_sha256"),
        depth_sha256=_strict_string(row["depth_sha256"], "step.depth_sha256"),
        goal_distance_m=_strict_optional_float(
            row["goal_distance_m"], "step.goal_distance_m"),
        goal_reachable=_strict_bool(
            row["goal_reachable"], "step.goal_reachable"),
        moved_m=_strict_float(row["moved_m"], "step.moved_m"),
        collision_detected=_strict_bool(
            row["collision_detected"], "step.collision_detected"),
        full_step_rejected=_strict_bool(
            row["full_step_rejected"], "step.full_step_rejected"),
        creep_used=_strict_bool(row["creep_used"], "step.creep_used"),
        zero_motion=_strict_bool(row["zero_motion"], "step.zero_motion"),
    )


def _parse_commitment(value: object) -> CommitmentReceipt:
    keys = (
        "state_id", "candidate_id", "commitment_index", "plan_sha256",
        "fifo_mutations", "steps",
    )
    row = _strict_mapping(value, keys, "commitment")
    steps = _strict_sequence(row["steps"], "commitment.steps")
    return CommitmentReceipt(
        state_id=_strict_string(row["state_id"], "commitment.state_id"),
        candidate_id=_strict_string(
            row["candidate_id"], "commitment.candidate_id"),
        commitment_index=_strict_int(
            row["commitment_index"], "commitment.commitment_index"),
        plan_sha256=_strict_string(
            row["plan_sha256"], "commitment.plan_sha256"),
        fifo_mutations=_strict_int(
            row["fifo_mutations"], "commitment.fifo_mutations"),
        steps=tuple(_parse_step(step) for step in steps),
    )


def _parse_outcome(value: object) -> ArmOutcome:
    keys = (
        "candidate_id", "candidate_type", "preparation", "plans",
        "commitments", "rollout_label_valid", "invalid_reason", "reachable",
        "initial_goal_distance_m", "goal_distance_h8_m", "goal_distance_h24_m",
        "geodesic_progress_h8_m", "geodesic_progress_h24_m", "collision_h8",
        "trace_sha256",
    )
    row = _strict_mapping(value, keys, "outcome")
    plans = _strict_sequence(row["plans"], "outcome.plans")
    commitments = _strict_sequence(row["commitments"], "outcome.commitments")
    invalid_reason = row["invalid_reason"]
    _require(
        invalid_reason is None or isinstance(invalid_reason, str),
        "outcome.invalid_reason must be null or string",
    )
    return ArmOutcome(
        candidate_id=_strict_string(row["candidate_id"], "outcome.candidate_id"),
        candidate_type=_strict_string(
            row["candidate_type"], "outcome.candidate_type"),
        preparation=_parse_preparation(row["preparation"]),
        plans=tuple(_parse_plan(plan) for plan in plans),
        commitments=tuple(
            _parse_commitment(commitment) for commitment in commitments),
        rollout_label_valid=_strict_bool(
            row["rollout_label_valid"], "outcome.rollout_label_valid"),
        invalid_reason=invalid_reason,
        reachable=_strict_bool(row["reachable"], "outcome.reachable"),
        initial_goal_distance_m=_strict_optional_float(
            row["initial_goal_distance_m"], "outcome.initial_goal_distance_m"),
        goal_distance_h8_m=_strict_optional_float(
            row["goal_distance_h8_m"], "outcome.goal_distance_h8_m"),
        goal_distance_h24_m=_strict_optional_float(
            row["goal_distance_h24_m"], "outcome.goal_distance_h24_m"),
        geodesic_progress_h8_m=_strict_float(
            row["geodesic_progress_h8_m"], "outcome.geodesic_progress_h8_m"),
        geodesic_progress_h24_m=_strict_float(
            row["geodesic_progress_h24_m"], "outcome.geodesic_progress_h24_m"),
        collision_h8=_strict_bool(row["collision_h8"], "outcome.collision_h8"),
        trace_sha256=_strict_string(row["trace_sha256"], "outcome.trace_sha256"),
    )


def _parse_labels(value: object) -> dict[str, dict[str, object]]:
    _require(isinstance(value, Mapping), "labels_by_candidate must be an object")
    expected = (
        "geodesic_progress_h8_m", "geodesic_progress_h24_m",
        "advantage_h24_m", "harm", "useful", "reachable", "collision_h8",
        "regression_h24", "rollout_label_valid",
    )
    result: dict[str, dict[str, object]] = {}
    for candidate_id, raw_labels in value.items():
        candidate_id = _strict_string(candidate_id, "label candidate id")
        labels = _strict_mapping(
            raw_labels, expected, f"labels_by_candidate[{candidate_id!r}]")
        result[candidate_id] = {
            "geodesic_progress_h8_m": _strict_float(
                labels["geodesic_progress_h8_m"], "label progress h8"),
            "geodesic_progress_h24_m": _strict_float(
                labels["geodesic_progress_h24_m"], "label progress h24"),
            "advantage_h24_m": _strict_float(
                labels["advantage_h24_m"], "label advantage h24"),
            **{
                key: _strict_bool(labels[key], f"label {key}")
                for key in expected[3:]
            },
        }
    return result


def artifact_from_dict(value: object) -> PairedRolloutArtifact:
    """Strictly reconstruct and semantically verify one JSON-shaped artifact."""
    keys = (
        "protocol_version", "run_signature_sha256", "state",
        "diffusion_seeds", "outcomes", "labels_by_candidate",
        "artifact_sha256",
    )
    row = _strict_mapping(value, keys, "artifact")
    seeds_raw = _strict_sequence(row["diffusion_seeds"], "diffusion_seeds")
    seeds = _validate_seeds(tuple(
        _strict_int(seed, f"diffusion_seeds[{index}]")
        for index, seed in enumerate(seeds_raw)
    ))
    outcomes_raw = _strict_sequence(row["outcomes"], "outcomes")
    artifact = PairedRolloutArtifact(
        protocol_version=_strict_string(
            row["protocol_version"], "protocol_version"),
        run_signature_sha256=_strict_string(
            row["run_signature_sha256"], "run_signature_sha256"),
        state=_parse_state(row["state"]),
        diffusion_seeds=seeds,
        outcomes=tuple(_parse_outcome(item) for item in outcomes_raw),
        labels_by_candidate=_parse_labels(row["labels_by_candidate"]),
        artifact_sha256=_strict_string(
            row["artifact_sha256"], "artifact_sha256"),
    )
    artifact_to_dict(artifact)
    return artifact


def artifact_to_dict(artifact: PairedRolloutArtifact) -> dict[str, object]:
    _require(
        isinstance(artifact, PairedRolloutArtifact),
        "artifact has wrong type",
    )
    _require(
        artifact.protocol_version == PROTOCOL_VERSION,
        "artifact protocol version is not current",
    )
    _require(
        _valid_sha256(artifact.run_signature_sha256),
        "artifact run signature is invalid",
    )
    _validate_state(artifact.state)
    seeds = _validate_seeds(artifact.diffusion_seeds)
    _require(
        isinstance(artifact.outcomes, tuple) and len(artifact.outcomes) >= 1,
        "artifact requires at least one outcome",
    )
    candidate_ids = [outcome.candidate_id for outcome in artifact.outcomes]
    _require(
        len(candidate_ids) == len(set(candidate_ids)),
        "artifact candidate ids must be unique",
    )
    _require(
        candidate_ids.count("native") == 1,
        "artifact requires exactly one native outcome",
    )
    expected_order = sorted(
        artifact.outcomes,
        key=lambda outcome: (
            outcome.candidate_id != "native",
            outcome.candidate_id,
        ),
    )
    _require(
        list(artifact.outcomes) == expected_order,
        "artifact outcomes are not in canonical order",
    )
    for outcome in artifact.outcomes:
        _validate_outcome_trace(artifact.state, seeds, outcome)
    expected_labels = _derive_labels(artifact.outcomes)
    _require(
        canonical_json_bytes(dict(artifact.labels_by_candidate))
        == canonical_json_bytes(expected_labels),
        "artifact labels disagree with verified traces",
    )
    body = {
        "protocol_version": artifact.protocol_version,
        "run_signature_sha256": artifact.run_signature_sha256,
        "state": asdict(artifact.state),
        "diffusion_seeds": list(artifact.diffusion_seeds),
        "outcomes": [asdict(outcome) for outcome in artifact.outcomes],
        "labels_by_candidate": dict(artifact.labels_by_candidate),
    }
    _require(
        canonical_sha256(body) == artifact.artifact_sha256,
        "artifact content no longer matches its hash",
    )
    return {**body, "artifact_sha256": artifact.artifact_sha256}


def _artifact_disk_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_artifact(path: str | Path) -> PairedRolloutArtifact:
    """Validate disk bytes, JSON schema, semantic trace, hash, and sidecar."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _require(path.is_file(), f"artifact file is missing: {path}")
    _require(sidecar.is_file(), f"artifact sidecar is missing: {sidecar}")
    encoded = path.read_bytes()
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RolloutProtocolError(
                    f"JSON contains non-finite constant {value!r}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RolloutProtocolError(f"artifact JSON is invalid: {error}") from error
    artifact = artifact_from_dict(payload)
    canonical_encoded = _artifact_disk_bytes(artifact_to_dict(artifact))
    _require(
        encoded == canonical_encoded,
        "artifact JSON does not use the canonical disk encoding",
    )
    file_sha = hashlib.sha256(encoded).hexdigest()
    expected_sidecar = f"{file_sha}  {path.name}\n".encode("utf-8")
    _require(
        sidecar.read_bytes() == expected_sidecar,
        "artifact sidecar is noncanonical or disagrees with JSON bytes",
    )
    return artifact


def atomic_write_artifact(
    path: str | Path,
    artifact: PairedRolloutArtifact,
    *,
    resume: bool = False,
) -> str:
    """Atomically persist an artifact or verify an identical resume target."""
    path = Path(path)
    payload = artifact_to_dict(artifact)
    encoded = _artifact_disk_bytes(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    lock = path.with_name(path.name + ".lock")
    file_sha = hashlib.sha256(encoded).hexdigest()
    if path.exists() or sidecar.exists():
        _require(resume, f"output already exists: {path}")
        _require(path.is_file() and sidecar.is_file(), "partial resume artifact")
        existing_artifact = load_artifact(path)
        _require(
            artifact_to_dict(existing_artifact) == payload,
            "resume artifact content differs",
        )
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RolloutProtocolError(
            f"another writer owns the output lock: {lock}") from error
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary_sidecar = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        _require(
            not path.exists() and not sidecar.exists(),
            "output appeared after lock acquisition",
        )
        _require(
            not temporary.exists() and not temporary_sidecar.exists(),
            "temporary output already exists",
        )
        temporary.write_bytes(encoded)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_sidecar.write_text(
            f"{file_sha}  {path.name}\n", encoding="utf-8")
        with temporary_sidecar.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        os.close(lock_fd)
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    _require(
        artifact_to_dict(load_artifact(path)) == payload,
        "written artifact failed strict round-trip validation",
    )
    return "written"
