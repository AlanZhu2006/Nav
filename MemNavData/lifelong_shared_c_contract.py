"""Pure contracts for a shared-C, B2-only controller comparison.

The old portability study independently executed C in every arm, so its B2
comparison did not share a causal start state.  This contract freezes one
controller-specific factual C rollout before any B2 outcome is generated,
then replays that exact RGB/pose/action-decision trace in every B2 arm.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "lifelong_shared_c_trace_v1_20260825"
POPULATION_SCHEMA = "lifelong_shared_c_population_v1_20260825"
RESULT_SCHEMA = "lifelong_shared_c_b2_eval_v1_20260825"
ARMS = ("all_prior", "initial_leg_only", "forced_reject_native")
PRIMARY_ARMS = ARMS[:2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def validate_trace(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema_version") == TRACE_SCHEMA,
            "shared-C trace schema changed")
    for key in (
        "scene", "episode", "controller", "benchmark_sha256",
        "online_A_trace_sha256", "online_B_trace_sha256", "goal_C_sha256",
    ):
        require(isinstance(payload.get(key), str) and payload[key],
                f"shared-C trace missing {key}")
    require(str(payload["episode"]).startswith("episode_"),
            "shared-C episode identity changed")
    for key in ("benchmark_sha256", "online_A_trace_sha256",
                "online_B_trace_sha256", "goal_C_sha256"):
        value = payload[key]
        require(len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                f"shared-C {key} is not SHA-256")
    require(isinstance(payload.get("episode_seed"), int)
            and payload["episode_seed"] >= 0,
            "shared-C episode seed changed")
    a_ceiling = int(payload["online_A_candidate_ceiling"])
    b_ceiling = int(payload["online_B_candidate_ceiling"])
    start_frame = int(payload["C_goal_start_frame"])
    require(a_ceiling >= 0 and b_ceiling > a_ceiling,
            "shared-C factual ceilings are invalid")
    require(start_frame == b_ceiling + 1,
            "shared-C goal boundary is not immediately after B")
    require(int(payload["C_candidate_ceiling"]) == a_ceiling,
            "shared-C did not use the initial-leg ceiling")
    require(payload.get("runtime_role_visible") is False,
            "shared-C runtime consumed a role label")
    poses = payload.get("poses")
    plans = payload.get("plans")
    memory = payload.get("memory_trace")
    require(isinstance(poses, list) and poses,
            "shared-C trace has no rendered poses")
    require(isinstance(plans, list) and plans,
            "shared-C trace has no controller decisions")
    require(isinstance(memory, list) and len(memory) == len(poses),
            "shared-C memory/pose lengths differ")
    expected_frames = list(range(start_frame, start_frame + len(memory)))
    require([int(row["frame_idx"]) for row in memory] == expected_frames,
            "shared-C memory indices are not contiguous")
    steps = [int(row["step"]) for row in poses]
    require(steps == list(range(len(poses))),
            "shared-C pose steps are not contiguous")
    plan_steps = [int(row["step"]) for row in plans]
    require(plan_steps == sorted(set(plan_steps)),
            "shared-C plan steps are duplicated or unordered")
    require(set(plan_steps).issubset(set(steps)),
            "shared-C plan step escaped the pose trace")
    for pose in poses:
        for key in ("x", "y", "z", "yaw"):
            _finite(pose[key], f"pose {key}")
        digest = pose.get("jpg_sha256")
        require(isinstance(digest, str) and len(digest) == 64,
                "shared-C pose JPEG hash is missing")
    for key in ("start_position", "end_position"):
        value = payload.get(key)
        require(isinstance(value, list) and len(value) == 3,
                f"shared-C {key} changed")
        [_finite(item, key) for item in value]
    _finite(payload["start_yaw"], "start_yaw")
    _finite(payload["end_yaw"], "end_yaw")
    require(isinstance(payload.get("reached_C"), bool),
            "shared-C reached flag is not boolean")
    require(payload.get("termination_reason") is None
            or isinstance(payload["termination_reason"], str),
            "shared-C termination reason changed type")
    return payload


def write_trace(path: Path, payload: dict[str, Any]) -> str:
    validate_trace(payload)
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"shared-C trace exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), f"stale shared-C temporary: {temporary}")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def load_trace(path: Path, *, expected_sha256: str | None = None) -> dict:
    require(path.is_file(), f"shared-C trace is missing: {path}")
    if expected_sha256 is not None:
        require(sha256_file(path) == expected_sha256,
                "shared-C trace hash changed")
    return validate_trace(json.loads(path.read_text()))


__all__ = [
    "ARMS", "POPULATION_SCHEMA", "PRIMARY_ARMS", "RESULT_SCHEMA",
    "TRACE_SCHEMA", "load_trace", "require", "sha256_file", "validate_trace",
    "write_trace",
]
