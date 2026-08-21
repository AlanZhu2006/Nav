"""Pure helpers for deterministic, shared-prefix navigation evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


LEG1_TRACE_SCHEMA_VERSION = 1
EPISODE_SEED_STRIDE = 100_000
LEG_SEED_STRIDE = 10_000
MAX_PLANS_PER_LEG = LEG_SEED_STRIDE
MAX_RESAMPLES_PER_PLAN = 99
APPROVED_SHARED_TRACE_ROUTES = frozenset({
    "phase",
    "memory_advantage",
    "memory_geometry",
    "learned_rank_geometry",
    "native_sidecar",
    "certified_relocalization",
})
NATIVE_OBSERVATION_REPLAY_CONTRACT = (
    "frozen_native_navdp_observation_only_memory_replay_v1"
)


def diffusion_plan_seed(
        episode_seed: int, leg_index: int, plan_index: int) -> int:
    """Map an episode, leg, and replan index to a collision-free seed."""
    values = (episode_seed, leg_index, plan_index)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError("seed coordinates must be integers")
    if episode_seed < 0:
        raise ValueError("episode_seed must be non-negative")
    if not 0 <= leg_index < 10:
        raise ValueError("leg_index must be in [0, 10)")
    if not 0 <= plan_index < MAX_PLANS_PER_LEG:
        raise ValueError(
            f"plan_index must be in [0, {MAX_PLANS_PER_LEG})")
    return (
        episode_seed * EPISODE_SEED_STRIDE
        + leg_index * LEG_SEED_STRIDE
        + plan_index
    )


def diffusion_resample_seed(primary_seed: int, resample_index: int) -> int:
    """Derive a deterministic seed outside the primary plan-seed namespace."""
    values = (primary_seed, resample_index)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError("resample seed coordinates must be integers")
    if primary_seed < 0:
        raise ValueError("primary_seed must be non-negative")
    if not 1 <= resample_index <= MAX_RESAMPLES_PER_PLAN:
        raise ValueError(
            f"resample_index must be in [1, {MAX_RESAMPLES_PER_PLAN}]")
    return primary_seed * 100 + resample_index


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_shared_trace_source(payload: dict) -> None:
    """Accept native-phase or automatic-router traces from the hybrid server."""
    backend = payload.get("source_backend")
    route = payload.get("source_hybrid_route")
    if backend == "hybrid_pose":
        if route not in APPROVED_SHARED_TRACE_ROUTES:
            raise ValueError("shared trace source route is not approved")
        return
    if backend == "navdp":
        if route != "phase":
            raise ValueError("native shared trace must use the phase label")
        if payload.get("source_control_contract") != (
                NATIVE_OBSERVATION_REPLAY_CONTRACT):
            raise ValueError(
                "native shared trace lacks the observation-only replay seal"
            )
        checkpoint_sha = payload.get("source_navdp_checkpoint_sha256")
        if (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
                or any(char not in "0123456789abcdef"
                       for char in checkpoint_sha)):
            raise ValueError("native shared trace checkpoint hash is invalid")
        if payload.get("source_memory_observer_present") is not False:
            raise ValueError(
                "native trace must disclose that memory was absent during control"
            )
        return
    raise ValueError("shared trace source backend is unsupported")


def write_leg1_trace(path: Path, payload: dict) -> str:
    """Validate and atomically materialize a canonical trace JSON."""
    validate_leg1_trace(payload)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def validate_leg1_trace(
        payload: dict, *, expected_episode: str | None = None,
        expected_seed: int | None = None,
        expected_goal_sha256: str | None = None,
        expected_source_scene: str | None = None) -> dict:
    """Fail closed on a shared Novel-prefix trace before replay."""
    if payload.get("schema_version") != LEG1_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported leg-1 trace schema")
    episode = payload.get("episode")
    if not isinstance(episode, str) or not episode.startswith("episode_"):
        raise ValueError("invalid trace episode")
    if expected_episode is not None and episode != expected_episode:
        raise ValueError("shared trace episode mismatch")
    seed = payload.get("episode_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid trace episode_seed")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("shared trace seed mismatch")
    goal_sha = payload.get("goal_sha256")
    if (not isinstance(goal_sha, str) or len(goal_sha) != 64
            or any(char not in "0123456789abcdef" for char in goal_sha)):
        raise ValueError("invalid trace goal_sha256")
    if expected_goal_sha256 is not None and goal_sha != expected_goal_sha256:
        raise ValueError("shared trace Goal-A image mismatch")
    source_scene = payload.get("source_scene")
    if not isinstance(source_scene, str) or not source_scene:
        raise ValueError("invalid trace source_scene")
    if expected_source_scene is not None and source_scene != expected_source_scene:
        raise ValueError("shared trace source scene mismatch")
    source_backend = payload.get("source_backend")
    if not isinstance(source_backend, str) or not source_backend:
        raise ValueError("invalid trace source_backend")
    source_route = payload.get("source_hybrid_route")
    if not isinstance(source_route, str) or not source_route:
        raise ValueError("invalid trace source_hybrid_route")
    candidate_gap = payload.get("source_retrieval_candidate_min_gap")
    if (isinstance(candidate_gap, bool) or not isinstance(candidate_gap, int)
            or candidate_gap <= 0):
        raise ValueError("invalid trace source candidate gap")
    for field, lower_bound in (
            ("source_graph_subgoal_spacing_m", 0.0),
            ("source_graph_subgoal_arrival_m", 1e-12)):
        value = _finite_number(payload.get(field), field)
        if value < lower_bound:
            raise ValueError(f"invalid trace {field}")
    goal_source_episode = payload.get("goal_source_episode")
    if (not isinstance(goal_source_episode, str)
            or not goal_source_episode.startswith("episode_")):
        raise ValueError("invalid trace goal_source_episode")

    reached = payload.get("reached")
    if not isinstance(reached, bool):
        raise ValueError("trace reached must be bool")
    steps = payload.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("invalid trace steps")
    poses = payload.get("poses")
    if not isinstance(poses, list) or len(poses) != steps:
        raise ValueError("trace pose count must equal steps")
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict) or pose.get("step") != index:
            raise ValueError("trace steps must be dense and ordered")
        for field in ("x", "y", "z", "yaw"):
            _finite_number(pose.get(field), f"poses[{index}].{field}")
        image_sha = pose.get("jpg_sha256")
        if (not isinstance(image_sha, str) or len(image_sha) != 64
                or any(char not in "0123456789abcdef" for char in image_sha)):
            raise ValueError(f"poses[{index}].jpg_sha256 is invalid")

    for field in ("path_len", "final_goal_dist_m", "end_yaw"):
        _finite_number(payload.get(field), field)
    end_position = payload.get("end_position")
    if not isinstance(end_position, list) or len(end_position) != 3:
        raise ValueError("end_position must contain three coordinates")
    for index, value in enumerate(end_position):
        _finite_number(value, f"end_position[{index}]")
    plans = payload.get("plans")
    if not isinstance(plans, list):
        raise ValueError("trace plans must be a list")
    previous_plan_step = -1
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            raise ValueError("trace plans must contain objects")
        plan_step = plan.get("step")
        if (isinstance(plan_step, bool) or not isinstance(plan_step, int)
                or not 0 <= plan_step < steps):
            raise ValueError(f"plans[{index}].step is outside the trace")
        if plan_step <= previous_plan_step:
            raise ValueError("trace plan steps must be strictly increasing")
        previous_plan_step = plan_step
    return payload


def load_leg1_trace(
        path: Path, *, expected_episode: str, expected_seed: int,
        expected_goal_sha256: str,
        expected_source_scene: str | None = None) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    validate_leg1_trace(
        payload,
        expected_episode=expected_episode,
        expected_seed=expected_seed,
        expected_goal_sha256=expected_goal_sha256,
        expected_source_scene=expected_source_scene,
    )
    return payload, file_sha256(path)
