#!/usr/bin/env python3
"""Add fail-closed native-first audit endpoints to an unmodified NavDP server.

The wrapper refuses to import NavDP unless ``navdp_server.py`` and
``policy_agent.py`` match the pinned clean-source hashes below.  It then adds:

* ``/memory_audit``: read-only hashes of the real processed FIFO and padded
  model tensor;
* ``/navdp_plan_atomic``: one transactional FIFO append followed by exactly
  one seeded, read-only ``native`` or ``image_point`` diffusion call;
* ``/navdp_step_ip_mixgoal_resample``: image+point resampling from the FIFO
  already advanced by ``/imagegoal_step`` for this decision frame.

The second endpoint verifies that the supplied current image is byte-identical
to the last processed FIFO item and that no FIFO byte changes during mixed
inference.  Thus an abstention executes the already-sampled native response,
while an activation pays one additional read-only diffusion call without
duplicating the current observation.  This is an orchestration boundary, not
a trained model and not a claim of closed-loop non-inferiority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np


EXPECTED_NAVDP_SERVER_SHA256 = (
    "d27a282b08577555f45ae260f87b251eb5d597d0a087ac23dce79e4eb246028e"
)
EXPECTED_POLICY_AGENT_SHA256 = (
    "1f9dda348e03591a721606411333232e9b7053d9c68c864bc0f4ed698aa089a9"
)
EXPECTED_DETERMINISTIC_SEED_SHA256 = (
    "1679bd8eb6ce70fde02fcc15abcd4fbff8ae1a0b8a0fa350a357c574b98b4463"
)
AUDIT_PROTOCOL = "navdp_native_first_fifo_v1"
ATOMIC_PLAN_PROTOCOL = "navdp_native_first_atomic_plan_v2"
ATOMIC_PLAN_MODES = frozenset(("native", "image_point"))
MAX_DIFFUSION_SEED = 2**63 - 1


class NativeFirstAuditError(RuntimeError):
    """Raised when exact native/FIFO equivalence cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeFirstAuditError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_file(
    path: str | Path,
    expected_sha256: str,
    description: str,
) -> str:
    path = Path(path)
    _require(path.is_file(), f"missing {description}: {path}")
    _require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef"
                for character in expected_sha256),
        f"invalid expected SHA for {description}",
    )
    actual = sha256_file(path)
    _require(
        actual == expected_sha256,
        f"{description} source mismatch: {actual} != {expected_sha256}",
    )
    return actual


def canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeFirstAuditError(
            f"audit value is not canonical JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode one provenance object in the collector's canonical format."""
    try:
        return (json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeFirstAuditError(
            f"provenance value is not canonical JSON: {error}") from error


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_server_provenance(
    path: str | Path,
    provenance: Mapping[str, str],
) -> str:
    """Persist exact live-server identity without parsing stdout logs."""
    output = Path(path).resolve()
    payload = canonical_json_bytes(dict(provenance))
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    if output.exists() or sidecar.exists():
        _require(
            output.is_file() and sidecar.is_file(),
            "server provenance output pair is incomplete",
        )
        _require(
            output.read_bytes() == payload
            and sidecar.read_bytes() == sidecar_payload,
            "server provenance output pair differs from the live server",
        )
        return digest
    _atomic_write_bytes(output, payload)
    _atomic_write_bytes(sidecar, sidecar_payload)
    return digest


def ndarray_sha256(value: object) -> str:
    """Hash ndarray dtype/shape and exact C-order bytes."""
    array = np.asarray(value)
    _require(
        np.issubdtype(array.dtype, np.number),
        "only numeric arrays can enter FIFO audit",
    )
    _require(
        bool(np.isfinite(array).all()),
        "non-finite numeric array cannot enter FIFO audit",
    )
    contiguous = np.ascontiguousarray(array)
    header = json.dumps({
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def padded_fifo_tensor(
    queues: Sequence[Sequence[np.ndarray]],
    memory_size: int,
) -> np.ndarray | None:
    """Reproduce NavDP's left-zero-padding without mutating its queues."""
    _require(
        isinstance(memory_size, int)
        and not isinstance(memory_size, bool)
        and memory_size >= 1,
        "memory_size must be a positive integer",
    )
    if not queues:
        return None
    rows = []
    for env_index, queue in enumerate(queues):
        _require(
            1 <= len(queue) <= memory_size,
            f"environment {env_index} FIFO length is outside [1, memory_size]",
        )
        arrays = [np.asarray(item) for item in queue]
        reference = arrays[0]
        _require(
            all(item.shape == reference.shape and item.dtype == reference.dtype
                for item in arrays),
            f"environment {env_index} FIFO is ragged",
        )
        stacked = np.stack(arrays, axis=0)
        if stacked.shape[0] < memory_size:
            stacked = np.pad(
                stacked,
                ((memory_size - stacked.shape[0], 0),
                 *[(0, 0) for _ in reference.shape]),
                mode="constant",
                constant_values=0,
            )
        rows.append(stacked)
    reference = rows[0]
    _require(
        all(row.shape == reference.shape and row.dtype == reference.dtype
            for row in rows),
        "batch FIFO tensor is ragged",
    )
    return np.stack(rows, axis=0)


def snapshot_memory(agent: object) -> dict[str, object]:
    """Return content hashes for the exact queue consumed by NavDP."""
    queues = getattr(agent, "memory_queue", None)
    memory_size = getattr(agent, "memory_size", None)
    raw_batch_size = getattr(agent, "batch_size", None)
    _require(isinstance(queues, list), "agent memory_queue is unavailable")
    _require(
        isinstance(memory_size, int) and not isinstance(memory_size, bool),
        "agent memory_size is unavailable",
    )
    # The unmodified NavDP server stores ``batch_size`` as the zero-dimensional
    # NumPy integer produced by ``np.array(payload["batch_size"])``.  Treat it
    # as the same scalar contract as a Python integer, while rejecting floats,
    # booleans, and non-scalar arrays.
    batch_array = np.asarray(raw_batch_size)
    _require(
        batch_array.shape == ()
        and np.issubdtype(batch_array.dtype, np.integer)
        and not np.issubdtype(batch_array.dtype, np.bool_),
        "agent batch_size is not an integer scalar",
    )
    batch_size = int(batch_array.item())
    _require(
        batch_size >= 1 and len(queues) == batch_size,
        "agent FIFO count disagrees with batch_size",
    )
    queue_item_sha256 = [
        [ndarray_sha256(item) for item in queue]
        for queue in queues
    ]
    queue_lengths = [len(queue) for queue in queues]
    if all(length == 0 for length in queue_lengths):
        padded = None
        padded_sha = None
    else:
        _require(
            all(length > 0 for length in queue_lengths),
            "mixed empty/non-empty batch FIFOs are not auditable",
        )
        padded = padded_fifo_tensor(queues, memory_size)
        _require(padded is not None, "padded FIFO unexpectedly absent")
        padded_sha = ndarray_sha256(padded)
    identity = {
        "memory_size": memory_size,
        "queue_lengths": queue_lengths,
        "queue_item_sha256": queue_item_sha256,
        "padded_model_tensor_sha256": padded_sha,
    }
    return {
        **identity,
        "fifo_sha256": canonical_json_sha256(identity),
    }


def _copy_memory_queue(agent: object) -> list[list[np.ndarray]]:
    """Copy the transaction state so a failed plan can roll back exactly."""
    queues = getattr(agent, "memory_queue", None)
    _require(isinstance(queues, list), "agent memory_queue is unavailable")
    return [
        [np.asarray(item).copy() for item in queue]
        for queue in queues
    ]


def _restore_memory_queue(
    agent: object,
    queues: Sequence[Sequence[np.ndarray]],
) -> None:
    """Restore the FIFO after any rejected atomic-plan transaction."""
    agent.memory_queue = [
        [np.asarray(item).copy() for item in queue]
        for queue in queues
    ]


def _normalize_required_seed(seed: object) -> int:
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        "diffusion_seed must be an integer",
    )
    _require(
        0 <= seed <= MAX_DIFFUSION_SEED,
        f"diffusion_seed must be in [0, {MAX_DIFFUSION_SEED}]",
    )
    return seed


def _parse_required_seed(value: object) -> int:
    _require(value is not None and value != "", "diffusion_seed is required")
    _require(not isinstance(value, bool), "diffusion_seed must be an integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as error:
        raise NativeFirstAuditError(
            "diffusion_seed must be an integer") from error
    _require(
        str(value).strip() == str(seed),
        "diffusion_seed must use canonical integer syntax",
    )
    return _normalize_required_seed(seed)


def _append_processed_current_once(
    agent: object,
    processed_current: np.ndarray,
    before: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Advance every environment FIFO exactly once and prove the transition."""
    queues = getattr(agent, "memory_queue", None)
    memory_size = getattr(agent, "memory_size", None)
    _require(isinstance(queues, list), "agent memory_queue is unavailable")
    _require(
        isinstance(memory_size, int)
        and not isinstance(memory_size, bool)
        and memory_size >= 1,
        "agent memory_size is unavailable",
    )
    _require(
        len(processed_current) == len(queues),
        "current-image batch differs from NavDP FIFO batch",
    )
    current_hashes: list[str] = []
    for env_index, current_value in enumerate(processed_current):
        current = np.asarray(current_value)
        current_hash = ndarray_sha256(current)
        queue = queues[env_index]
        _require(
            isinstance(queue, list) and len(queue) <= memory_size,
            f"environment {env_index} FIFO is malformed",
        )
        if queue:
            reference = np.asarray(queue[-1])
            _require(
                current.shape == reference.shape
                and current.dtype == reference.dtype,
                f"environment {env_index} current image disagrees with FIFO",
            )
        if len(queue) == memory_size:
            del queue[0]
        queue.append(current.copy())
        current_hashes.append(current_hash)

    after = snapshot_memory(agent)
    before_items = before["queue_item_sha256"]
    _require(isinstance(before_items, list), "invalid FIFO before snapshot")
    expected_items = []
    for env_index, (items, current_hash) in enumerate(
            zip(before_items, current_hashes)):
        _require(isinstance(items, list), "invalid FIFO item hash list")
        retained = items[-(memory_size - 1):] if memory_size > 1 else []
        expected_items.append([*retained, current_hash])
        _require(
            after["queue_item_sha256"][env_index][-1] == current_hash,
            f"environment {env_index} current image is not the FIFO tail",
        )
    _require(
        after["queue_item_sha256"] == expected_items,
        "atomic plan did not append current exactly once",
    )
    return after, current_hashes


def _require_current_tail_hashes(
    agent: object,
    current_hashes: Sequence[str],
    phase: str,
) -> None:
    queues = getattr(agent, "memory_queue", None)
    _require(
        isinstance(queues, list) and len(queues) == len(current_hashes),
        f"FIFO batch changed {phase}",
    )
    for env_index, (queue, expected_hash) in enumerate(
            zip(queues, current_hashes)):
        _require(
            isinstance(queue, list) and bool(queue),
            f"environment {env_index} FIFO became empty {phase}",
        )
        _require(
            ndarray_sha256(queue[-1]) == expected_hash,
            f"environment {env_index} current image is not the FIFO tail {phase}",
        )


def _validate_prediction_outputs(
    outputs: object,
    stop_threshold: object,
    mode: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    bool,
]:
    _require(
        isinstance(outputs, (tuple, list)) and len(outputs) == 4,
        f"NavDP {mode} predictor returned an unexpected structure",
    )
    all_trajectory, all_values, good_trajectory, _bad_trajectory = outputs
    all_trajectory = np.asarray(all_trajectory)
    all_values = np.asarray(all_values)
    good_trajectory = np.asarray(good_trajectory).copy()
    _require(
        all(
            np.issubdtype(value.dtype, np.number)
            and bool(np.isfinite(value).all())
            for value in (all_trajectory, all_values, good_trajectory)
        ),
        f"NavDP {mode} predictor returned non-finite outputs",
    )
    _require(all_values.size > 0, f"NavDP {mode} predictor returned no values")
    threshold_array = np.asarray(stop_threshold)
    _require(
        threshold_array.shape == ()
        and np.issubdtype(threshold_array.dtype, np.number)
        and not np.issubdtype(threshold_array.dtype, np.bool_)
        and bool(np.isfinite(threshold_array)),
        "agent stop_threshold is unavailable",
    )
    threshold = float(threshold_array.item())
    _require(
        good_trajectory.ndim == 4
        and good_trajectory.shape[1] >= 1
        and good_trajectory.shape[-2:] == (24, 3),
        f"NavDP {mode} good trajectory has no executable candidate",
    )
    _require(
        all_trajectory.ndim == 4
        and all_trajectory.shape[0] == good_trajectory.shape[0]
        and all_trajectory.shape[-2:] == good_trajectory.shape[-2:],
        f"NavDP {mode} candidate trajectories changed shape",
    )
    _require(
        all_values.ndim == 2
        and all_values.shape[:1] == all_trajectory.shape[:1]
        and all_values.shape[1] == all_trajectory.shape[1],
        f"NavDP {mode} critic values disagree with candidate trajectories",
    )
    raw_execute = good_trajectory[:, 0].copy()
    critic_max = float(all_values.max())
    fallback_applied = critic_max < threshold
    if fallback_applied:
        good_trajectory[:, :, :, 0] = 0.0
        good_trajectory[:, :, :, 1] = np.sign(
            good_trajectory[:, :, :, 1].mean())
    execute = good_trajectory[:, 0]
    _require(
        bool(np.array_equal(execute[:, :, 2], raw_execute[:, :, 2])),
        f"NavDP {mode} low-critic fallback changed trajectory theta",
    )
    return (
        execute,
        raw_execute,
        all_trajectory,
        all_values,
        critic_max,
        threshold,
        fallback_applied,
    )


def atomic_native_first_plan(
    agent: object,
    *,
    mode: str,
    image_goal: np.ndarray,
    current_images: np.ndarray,
    current_depths: np.ndarray,
    diffusion_seed: int,
    apply_seed: Callable[[int], object],
    point_goal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Commit one observation and sample exactly one read-only NavDP plan.

    Both modes share this transaction, so neither mode can receive a fresher
    FIFO than the other by construction.  Failed preprocessing, seeding,
    prediction, or audit restores the pre-request FIFO.
    """
    _require(mode in ATOMIC_PLAN_MODES, f"unsupported atomic plan mode: {mode}")
    seed = _normalize_required_seed(diffusion_seed)
    _require(callable(apply_seed), "apply_seed must be callable")

    before_queue = _copy_memory_queue(agent)
    before = snapshot_memory(agent)
    try:
        # A malformed goal/current request must never consume an observation.
        # Preprocessing is inside the transaction too: the pinned NavDP
        # implementation is pure, and the after-append proof would reject any
        # hidden FIFO mutation here.
        current = np.asarray(
            agent.process_image(np.asarray(current_images).copy()))
        goal = np.asarray(agent.process_image(np.asarray(image_goal).copy()))
        _require(
            len(current) == int(agent.batch_size)
            and len(goal) == int(agent.batch_size),
            "current/goal batch differs from NavDP batch_size",
        )
        current_batch_hash = ndarray_sha256(current)
        goal_batch_hash = ndarray_sha256(goal)
        goal_item_hashes = [ndarray_sha256(item) for item in goal]

        processed_point_goal = None
        point_goal_hash = None
        if mode == "image_point":
            _require(
                point_goal is not None,
                "image_point mode requires point_goal",
            )
            processed_point_goal = np.asarray(
                agent.process_pointgoal(np.asarray(point_goal).copy()))
            _require(
                len(processed_point_goal) == int(agent.batch_size),
                "point-goal batch differs from NavDP batch_size",
            )
            point_goal_hash = ndarray_sha256(processed_point_goal)
        else:
            _require(point_goal is None, "native mode forbids point_goal")

        depth = np.asarray(
            agent.process_depth(np.asarray(current_depths).copy()))
        _require(
            len(depth) == int(agent.batch_size),
            "depth batch differs from NavDP batch_size",
        )

        after_append, current_item_hashes = _append_processed_current_once(
            agent, current, before)
        input_images = padded_fifo_tensor(agent.memory_queue, agent.memory_size)
        _require(input_images is not None, "atomic plan FIFO is empty")

        seeded = apply_seed(seed)
        _require(
            seeded == seed,
            f"apply_seed returned {seeded!r}, expected {seed}",
        )
        _require(
            snapshot_memory(agent) == after_append,
            "seed application mutated NavDP FIFO",
        )

        if mode == "native":
            outputs = agent.navi_former.predict_imagegoal_action(
                goal, input_images, depth)
        else:
            _require(
                processed_point_goal is not None,
                "processed point goal unexpectedly absent",
            )
            outputs = agent.navi_former.predict_ip_action(
                processed_point_goal, goal, input_images, depth)
        (
            execute,
            raw_execute,
            all_trajectory,
            all_values,
            critic_max,
            stop_threshold,
            fallback_applied,
        ) = _validate_prediction_outputs(outputs, agent.stop_threshold, mode)

        _require_current_tail_hashes(
            agent, current_item_hashes, "after diffusion inference")
        after_inference = snapshot_memory(agent)
        _require(
            after_inference == after_append,
            "diffusion inference mutated NavDP FIFO",
        )
    except Exception:
        _restore_memory_queue(agent, before_queue)
        raise

    receipt_core: dict[str, object] = {
        "protocol": ATOMIC_PLAN_PROTOCOL,
        "mode": mode,
        "diffusion_seed": seed,
        "diffusion_call_count": 1,
        "goal_sha256": goal_batch_hash,
        "goal_item_sha256": goal_item_hashes,
        "current_sha256": current_batch_hash,
        "current_item_sha256": current_item_hashes,
        "fifo_before_sha256": before["fifo_sha256"],
        "fifo_after_append_sha256": after_append["fifo_sha256"],
        "fifo_item_sha256_before": before["queue_item_sha256"],
        "fifo_item_sha256": after_append["queue_item_sha256"],
        "fifo_lengths_before": before["queue_lengths"],
        "fifo_lengths_after": after_append["queue_lengths"],
        "point_goal_sha256": point_goal_hash,
        "critic_max": critic_max,
        "stop_threshold": stop_threshold,
        "low_critic_fallback_applied": fallback_applied,
        "raw_selected_trajectory": raw_execute.tolist(),
        "executable_trajectory": execute.tolist(),
        "inference_fifo_unchanged": True,
        "append_count_per_environment": 1,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_json_sha256(receipt_core),
    }
    return execute, all_trajectory, all_values, receipt


def verify_same_prefix_plan_receipts(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> str:
    """Fail closed unless native and residual plans used one exact prefix."""
    _require(
        {first.get("mode"), second.get("mode")} == ATOMIC_PLAN_MODES,
        "paired receipts must contain native and image_point modes",
    )
    for name, receipt in (("first", first), ("second", second)):
        _require(
            receipt.get("protocol") == ATOMIC_PLAN_PROTOCOL,
            f"{name} receipt protocol mismatch",
        )
        _require(
            receipt.get("diffusion_call_count") == 1,
            f"{name} receipt diffusion call count is not one",
        )
        _require(
            receipt.get("append_count_per_environment") == 1,
            f"{name} receipt append count is not one",
        )
        _require(
            receipt.get("inference_fifo_unchanged") is True,
            f"{name} receipt did not prove read-only inference",
        )
        try:
            critic_max = float(receipt.get("critic_max"))
            stop_threshold = float(receipt.get("stop_threshold"))
            raw_selected = np.asarray(
                receipt.get("raw_selected_trajectory"), dtype=np.float64)
            executable = np.asarray(
                receipt.get("executable_trajectory"), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise NativeFirstAuditError(
                f"{name} receipt trajectory diagnostics are invalid") from error
        fallback = receipt.get("low_critic_fallback_applied")
        _require(
            not isinstance(receipt.get("critic_max"), (bool, np.bool_))
            and not isinstance(receipt.get("stop_threshold"), (bool, np.bool_))
            and np.isfinite(critic_max)
            and np.isfinite(stop_threshold)
            and isinstance(fallback, bool)
            and fallback == (critic_max < stop_threshold),
            f"{name} receipt low-critic decision is invalid",
        )
        _require(
            raw_selected.ndim == 3
            and raw_selected.shape[-2:] == (24, 3)
            and executable.shape == raw_selected.shape
            and bool(np.isfinite(raw_selected).all())
            and bool(np.isfinite(executable).all()),
            f"{name} receipt trajectories are malformed",
        )
        if fallback:
            _require(
                bool(np.all(executable[:, :, 0] == 0.0))
                and bool(np.all(
                    executable[:, :, 1] == executable[0, 0, 1]))
                and float(executable[0, 0, 1]) in (-1.0, 0.0, 1.0)
                and bool(np.array_equal(
                    executable[:, :, 2], raw_selected[:, :, 2])),
                f"{name} receipt fallback trajectory is invalid",
            )
        else:
            _require(
                bool(np.array_equal(executable, raw_selected)),
                f"{name} receipt changed a non-fallback trajectory",
            )
        claimed_sha = receipt.get("receipt_sha256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        _require(
            claimed_sha == canonical_json_sha256(unsigned),
            f"{name} receipt hash mismatch",
        )
    compared = (
        "diffusion_seed",
        "goal_sha256",
        "goal_item_sha256",
        "current_sha256",
        "current_item_sha256",
        "fifo_before_sha256",
        "fifo_after_append_sha256",
        "fifo_item_sha256_before",
        "fifo_item_sha256",
        "fifo_lengths_before",
        "fifo_lengths_after",
        "stop_threshold",
    )
    for field in compared:
        _require(
            first.get(field) == second.get(field),
            f"paired receipt {field} mismatch",
        )
    identity = {field: first.get(field) for field in compared}
    return canonical_json_sha256(identity)


def _processed_current_matches_fifo(
    agent: object,
    images: np.ndarray,
) -> list[str]:
    processed = np.asarray(agent.process_image(images))
    queues = agent.memory_queue
    _require(
        len(processed) == len(queues),
        "current-image batch differs from NavDP FIFO batch",
    )
    hashes = []
    for env_index, current in enumerate(processed):
        _require(queues[env_index], "mixed resample requires a prior native step")
        current_hash = ndarray_sha256(current)
        fifo_hash = ndarray_sha256(queues[env_index][-1])
        _require(
            current_hash == fifo_hash,
            f"environment {env_index} current image is not the FIFO tail",
        )
        hashes.append(current_hash)
    return hashes


def resample_point_image_goal_read_only(
    agent: object,
    point_goal: np.ndarray,
    image_goal: np.ndarray,
    current_images: np.ndarray,
    current_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Run mixed conditioning from an already-advanced FIFO, without append."""
    before = snapshot_memory(agent)
    current_hashes = _processed_current_matches_fifo(agent, current_images)
    input_images = padded_fifo_tensor(agent.memory_queue, agent.memory_size)
    _require(input_images is not None, "mixed resample FIFO is empty")
    input_depth = agent.process_depth(np.asarray(current_depths).copy())
    input_point_goal = agent.process_pointgoal(np.asarray(point_goal).copy())
    input_image_goal = agent.process_image(np.asarray(image_goal).copy())
    outputs = agent.navi_former.predict_ip_action(
        input_point_goal,
        input_image_goal,
        input_images,
        input_depth,
    )
    _require(
        isinstance(outputs, (tuple, list)) and len(outputs) == 4,
        "NavDP mixed predictor returned an unexpected structure",
    )
    all_trajectory, all_values, good_trajectory, _bad_trajectory = outputs
    all_trajectory = np.asarray(all_trajectory)
    all_values = np.asarray(all_values)
    good_trajectory = np.asarray(good_trajectory).copy()
    _require(
        all(np.issubdtype(value.dtype, np.number)
            and bool(np.isfinite(value).all())
            for value in (all_trajectory, all_values, good_trajectory)),
        "NavDP mixed predictor returned non-finite outputs",
    )
    if float(all_values.max()) < float(agent.stop_threshold):
        good_trajectory[:, :, :, 0] = 0.0
        good_trajectory[:, :, :, 1] = np.sign(
            good_trajectory[:, :, :, 1].mean())
    after = snapshot_memory(agent)
    _require(before == after, "mixed resample mutated NavDP FIFO bytes")
    _require(
        good_trajectory.ndim >= 2 and good_trajectory.shape[1] >= 1,
        "NavDP good trajectory has no executable candidate",
    )
    audit = {
        "protocol": AUDIT_PROTOCOL,
        "fifo_before": before,
        "fifo_after": after,
        "current_processed_sha256": current_hashes,
        "memory_mutated": False,
    }
    return good_trajectory[:, 0], all_trajectory, all_values, audit


def _decode_rgb(base: object, file_storage: object, batch_size: int) -> np.ndarray:
    image = base.Image.open(file_storage.stream).convert("RGB")
    image = base.cv2.cvtColor(np.asarray(image), base.cv2.COLOR_RGB2BGR)
    return image.reshape((batch_size, -1, image.shape[1], 3))


def _decode_depth(base: object, file_storage: object, batch_size: int) -> np.ndarray:
    depth = base.Image.open(file_storage.stream).convert("I")
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    return depth.reshape((batch_size, -1, depth.shape[1], 1))


def register_audit_routes(base: object, provenance: Mapping[str, str]) -> None:
    """Register routes on the already-imported, source-verified Flask app."""
    _require(hasattr(base, "app"), "NavDP server app is unavailable")

    @base.app.route("/native_first_provenance", methods=["GET"])
    def native_first_provenance():
        return base.jsonify({
            "status": "ready",
            "protocol": ATOMIC_PLAN_PROTOCOL,
            "provenance": dict(provenance),
        })

    @base.app.route("/memory_audit", methods=["GET", "POST"])
    def memory_audit():
        agent = base.navdp_navigator
        if agent is None:
            return base.jsonify({"error": "navigator is not initialized"}), 409
        try:
            snapshot = snapshot_memory(agent)
        except NativeFirstAuditError as error:
            return base.jsonify({"error": str(error)}), 409
        return base.jsonify({
            "algo": "navdp",
            "protocol": AUDIT_PROTOCOL,
            "provenance": dict(provenance),
            **snapshot,
        })

    @base.app.route("/navdp_plan_atomic", methods=["POST"])
    def atomic_plan():
        """Advance the FIFO once and run one selected diffusion branch."""
        agent = base.navdp_navigator
        if agent is None:
            return base.jsonify({"error": "navigator is not initialized"}), 409
        try:
            batch_size = int(agent.batch_size)
            mode = str(base.request.form.get("mode", ""))
            current = _decode_rgb(
                base, base.request.files["image"], batch_size)
            image_goal = _decode_rgb(
                base, base.request.files["image_goal"], batch_size)
            depth = _decode_depth(
                base, base.request.files["depth"], batch_size)
            seed = _parse_required_seed(
                base.request.form.get("diffusion_seed"))
            point_goal = None
            if mode == "image_point":
                point_data = json.loads(base.request.form.get("goal_data"))
                point_x = np.asarray(point_data["goal_x"])
                point_y = np.asarray(point_data["goal_y"])
                _require(
                    point_x.shape == (batch_size,)
                    and point_y.shape == (batch_size,)
                    and bool(np.isfinite(point_x).all())
                    and bool(np.isfinite(point_y).all()),
                    "point goal must contain one finite x/y pair per environment",
                )
                point_goal = np.stack(
                    (point_x, point_y, np.zeros_like(point_x)), axis=1)
            execute, all_trajectory, all_values, receipt = (
                atomic_native_first_plan(
                    agent,
                    mode=mode,
                    image_goal=image_goal,
                    current_images=current,
                    current_depths=depth,
                    diffusion_seed=seed,
                    apply_seed=base.apply_seed,
                    point_goal=point_goal,
                )
            )
        except (
            NativeFirstAuditError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return base.jsonify({"error": str(error)}), 409
        return base.jsonify({
            "trajectory": execute.tolist(),
            "raw_selected_trajectory": receipt["raw_selected_trajectory"],
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
            "critic_max": receipt["critic_max"],
            "stop_threshold": receipt["stop_threshold"],
            "low_critic_fallback_applied": (
                receipt["low_critic_fallback_applied"]),
            "receipt": receipt,
            "provenance": dict(provenance),
        })

    @base.app.route(
        "/navdp_step_ip_mixgoal_resample",
        methods=["POST"],
    )
    def mixed_resample():
        agent = base.navdp_navigator
        if agent is None:
            return base.jsonify({"error": "navigator is not initialized"}), 409
        try:
            batch_size = int(agent.batch_size)
            point_data = json.loads(base.request.form.get("goal_data"))
            point_x = np.asarray(point_data["goal_x"])
            point_y = np.asarray(point_data["goal_y"])
            _require(
                point_x.shape == (batch_size,)
                and point_y.shape == (batch_size,)
                and bool(np.isfinite(point_x).all())
                and bool(np.isfinite(point_y).all()),
                "point goal must contain one finite x/y pair per environment",
            )
            point_goal = np.stack(
                (point_x, point_y, np.zeros_like(point_x)), axis=1)
            current = _decode_rgb(
                base, base.request.files["image"], batch_size)
            image_goal = _decode_rgb(
                base, base.request.files["image_goal"], batch_size)
            depth = _decode_depth(
                base, base.request.files["depth"], batch_size)
            diffusion_seed = base.apply_seed(
                base.request.form.get("diffusion_seed"))
            execute, all_trajectory, all_values, audit = (
                resample_point_image_goal_read_only(
                    agent,
                    point_goal,
                    image_goal,
                    current,
                    depth,
                )
            )
        except (
            NativeFirstAuditError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return base.jsonify({"error": str(error)}), 409
        return base.jsonify({
            "trajectory": execute.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
            "diffusion_seed": diffusion_seed,
            "memory_mutated": False,
            "queue_lengths": audit["fifo_after"]["queue_lengths"],
            "fifo_sha256": audit["fifo_after"]["fifo_sha256"],
            "current_processed_sha256": audit["current_processed_sha256"],
            "protocol": AUDIT_PROTOCOL,
            "provenance": dict(provenance),
        })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--navdp-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--provenance-output", required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    navdp_dir = Path(args.navdp_dir).resolve()
    server_path = navdp_dir / "navdp_server.py"
    policy_path = navdp_dir / "policy_agent.py"
    seed_path = navdp_dir / "deterministic_seed.py"
    server_sha = verify_source_file(
        server_path, EXPECTED_NAVDP_SERVER_SHA256, "navdp_server.py")
    policy_sha = verify_source_file(
        policy_path, EXPECTED_POLICY_AGENT_SHA256, "policy_agent.py")
    seed_sha = verify_source_file(
        seed_path,
        EXPECTED_DETERMINISTIC_SEED_SHA256,
        "deterministic_seed.py",
    )
    checkpoint_sha = verify_source_file(
        args.checkpoint,
        args.expected_checkpoint_sha256,
        "NavDP checkpoint",
    )
    wrapper_sha = verify_source_file(
        __file__, args.expected_wrapper_sha256, "native-first wrapper")
    sys.path.insert(0, str(navdp_dir))
    old_argv = sys.argv
    sys.argv = [str(server_path), "--port", str(args.port),
                "--checkpoint", str(Path(args.checkpoint).resolve())]
    try:
        base = importlib.import_module("navdp_server")
    finally:
        sys.argv = old_argv
    _require(
        Path(base.__file__).resolve() == server_path,
        "imported NavDP server from an unexpected path",
    )
    provenance = {
        "navdp_server_sha256": server_sha,
        "policy_agent_sha256": policy_sha,
        "deterministic_seed_sha256": seed_sha,
        "checkpoint_sha256": checkpoint_sha,
        "wrapper_sha256": wrapper_sha,
    }
    register_audit_routes(base, provenance)
    provenance_sha = write_server_provenance(
        args.provenance_output, provenance)
    print(json.dumps({
        "status": "native_first_audit_server_ready",
        "protocol": AUDIT_PROTOCOL,
        "host": args.host,
        "port": args.port,
        "provenance": provenance,
        "provenance_output": str(Path(args.provenance_output).resolve()),
        "provenance_sha256": provenance_sha,
    }, sort_keys=True), flush=True)
    base.app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
