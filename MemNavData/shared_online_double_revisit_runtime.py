"""Runtime contracts for replaying an audited online-A navigation prefix."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any


ONLINE_SCHEMA = "shared_online_a_materialized_v1"
BENCHMARK_SCHEMA = "shared_online_double_revisit_v1_20260812"
BENCHMARK_SCHEMAS = (
    BENCHMARK_SCHEMA,
    "shared_online_double_revisit_v2_route_negative_20260812",
)
VARIANTS = (
    "v0_exact_online_frame",
    "v1_controlled_pose_perturbation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def should_activate_certified_stagnation_intervention(
    *,
    mode: str,
    server_backend: str,
    hybrid_route: str,
    policy_backend: str | None,
    attempted: bool,
    plans: list[dict[str, Any]],
) -> bool:
    """Authorize the one-shot residual only at an audited stuck event.

    This helper deliberately does *not* detect stagnation.  The evaluator's
    frozen odometric detector owns that event; this predicate only checks that
    the requested intervention is the certified, automatic Revisit path and
    that an accepted certificate already exists for the current goal.
    """
    if mode not in ("off", "budget_control", "rescue"):
        raise ValueError(f"unknown certified stagnation mode {mode!r}")
    return bool(
        mode in ("budget_control", "rescue")
        and server_backend == "hybrid_pose"
        and hybrid_route == "certified_relocalization"
        and policy_backend == "navdp_auto"
        and not attempted
        and any(
            isinstance(plan, dict)
            and plan.get("certified_relocalization_accepted") is True
            for plan in plans
        )
    )


def should_activate_certified_graph_rescue(**kwargs: Any) -> bool:
    """Backward-compatible alias for callers that predate the control arm."""
    return should_activate_certified_stagnation_intervention(**kwargs)


def load_frozen_episode(
    benchmark_path: Path,
    *,
    variant: str,
    expected_scene: str,
) -> dict[str, Any]:
    """Load one per-episode benchmark and verify all source identities."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown shared-online variant {variant!r}")
    payload = json.loads(benchmark_path.read_text())
    require(
        payload.get("schema_version") in BENCHMARK_SCHEMAS,
        "shared-online benchmark schema changed",
    )
    require(
        payload.get("scene") == expected_scene,
        "benchmark scene differs from evaluator asset",
    )
    source = Path(payload["source_online_episode"])
    require(source.is_dir(), "source online-A episode is missing")
    require(
        sha256_file(source / "receipt.json")
        == payload["source_online_receipt_sha256"],
        "source online-A receipt hash changed",
    )
    require(
        sha256_file(source / "online_a_trace.json")
        == payload["source_online_trace_sha256"],
        "source online-A trace hash changed",
    )
    receipt = json.loads((source / "receipt.json").read_text())
    require(receipt.get("schema_version") == ONLINE_SCHEMA, "online schema changed")
    trace = json.loads((source / "online_a_trace.json").read_text())
    require(trace.get("reached") is True, "frozen online-A did not reach Goal A")
    require(
        len(trace.get("poses") or []) == int(payload["online_a_steps"]),
        "online-A trace length changed",
    )
    require(
        sha256_file(source / "goal_a.jpg") == payload["goal_a"]["sha256"],
        "Goal-A image hash changed",
    )
    selected = payload["variants"][variant]
    for role in ("B", "C"):
        asset = selected["assets"][role]
        goal_dir = benchmark_path.parent / variant
        rgb = goal_dir / asset["rgb"]
        depth = goal_dir / asset["depth"]
        require(
            sha256_file(rgb) == asset["rgb_sha256"],
            f"Goal-{role} RGB hash changed",
        )
        require(
            sha256_file(depth) == asset["depth_sha256"],
            f"Goal-{role} depth hash changed",
        )
    return {
        "benchmark": payload,
        "variant": selected,
        "source": source,
        "receipt": receipt,
        "trace": trace,
    }


def replay_online_a(
    frozen: dict[str, Any],
    *,
    memory_step: Callable[[bytes], dict[str, Any]],
    navdp_replay_step: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    """Restore one online-A prefix without executing or sampling a policy.

    Long-term MemNav receives every physical frame.  NavDP receives only the
    original decision frames, matching ``run_policy_leg`` and its eight-frame
    local FIFO.  Both callbacks must already target freshly reset servers.
    """
    source = Path(frozen["source"])
    trace = frozen["trace"]
    poses = trace["poses"]
    steps = [int(pose["step"]) for pose in poses]
    require(steps == list(range(len(poses))), "online-A poses are not contiguous")

    plan_steps = [int(plan["step"]) for plan in trace["plans"]]
    require(
        len(plan_steps) == len(set(plan_steps)),
        "online-A plan steps contain duplicates",
    )
    require(
        set(plan_steps).issubset(set(steps)),
        "online-A plan step lies outside the pose trace",
    )
    plan_step_set = set(plan_steps)
    memory_trace = []
    replayed_steps = []
    final_queue_lengths = None
    final_memory_size = None
    for pose in poses:
        step = int(pose["step"])
        image = source / "rgb" / f"{step:06d}.jpg"
        require(
            sha256_file(image) == pose["jpg_sha256"],
            f"online-A RGB hash changed at step {step}",
        )
        encoded = image.read_bytes()
        memory_receipt = memory_step(encoded)
        frame_idx = memory_receipt.get("frame_idx")
        if frame_idx is not None:
            require(
                int(frame_idx) == step,
                "MemNav replay index differs from frozen online frame",
            )
            memory_trace.append(
                {
                    "frame_idx": int(frame_idx),
                    "step": step,
                    "x": float(pose["x"]),
                    "z": float(pose["z"]),
                    "yaw": float(pose["yaw"]),
                }
            )
        if step in plan_step_set:
            navdp_receipt = navdp_replay_step(encoded)
            require(
                navdp_receipt.get("diffusion_sampled") is False,
                "NavDP replay unexpectedly sampled diffusion",
            )
            final_queue_lengths = navdp_receipt.get("queue_lengths")
            final_memory_size = int(navdp_receipt.get("memory_size", -1))
            replayed_steps.append(step)

    require(replayed_steps == plan_steps, "NavDP decision replay order changed")
    require(bool(plan_steps), "online-A trace contains no decision frames")
    require(final_memory_size is not None and final_memory_size > 0,
            "NavDP replay omitted memory size")
    expected_queue = min(len(plan_steps), final_memory_size)
    require(
        final_queue_lengths == [expected_queue],
        "NavDP queue length differs from frozen decision count",
    )
    require(
        not memory_trace or len(memory_trace) == len(poses),
        "MemNav replay did not index every online-A frame",
    )
    return {
        "online_frames": len(poses),
        "decision_frames": len(plan_steps),
        "decision_steps": plan_steps,
        "navdp_memory_size": final_memory_size,
        "navdp_queue_lengths": final_queue_lengths,
        "memory_trace": memory_trace,
        "all_rgb_hashes_verified": True,
        "diffusion_samples_during_replay": 0,
    }


def summarize_c_tail(
    curve: list[float],
    *,
    maximum_allowed: float,
) -> dict[str, Any]:
    """Summarize whether online B creates a recent-memory shortcut for C."""
    if not math.isfinite(float(maximum_allowed)) or not 0.0 <= maximum_allowed <= 1.0:
        raise ValueError("maximum_allowed must be finite and in [0,1]")
    values = [float(value) for value in curve]
    require(bool(values), "online-B co-visibility curve is empty")
    require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
        "online-B co-visibility curve is invalid",
    )
    argmax = max(range(len(values)), key=values.__getitem__)
    maximum = values[argmax]
    return {
        "ok": maximum <= float(maximum_allowed),
        "maximum_covisibility": maximum,
        "argmax_b_frame": int(argmax),
        "endpoint_covisibility": values[-1],
        "maximum_allowed": float(maximum_allowed),
        "frames": len(values),
        "curve": values,
    }


__all__ = [
    "BENCHMARK_SCHEMA",
    "BENCHMARK_SCHEMAS",
    "ONLINE_SCHEMA",
    "VARIANTS",
    "load_frozen_episode",
    "replay_online_a",
    "sha256_file",
    "summarize_c_tail",
]
