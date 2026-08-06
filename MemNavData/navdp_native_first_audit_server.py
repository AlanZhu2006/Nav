#!/usr/bin/env python3
"""Add fail-closed native-first audit endpoints to an unmodified NavDP server.

The wrapper refuses to import NavDP unless ``navdp_server.py`` and
``policy_agent.py`` match the pinned clean-source hashes below.  It then adds:

* ``/memory_audit``: read-only hashes of the real processed FIFO and padded
  model tensor;
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
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


EXPECTED_NAVDP_SERVER_SHA256 = (
    "d27a282b08577555f45ae260f87b251eb5d597d0a087ac23dce79e4eb246028e"
)
EXPECTED_POLICY_AGENT_SHA256 = (
    "1f9dda348e03591a721606411333232e9b7053d9c68c864bc0f4ed698aa089a9"
)
AUDIT_PROTOCOL = "navdp_native_first_fifo_v1"


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
    batch_size = getattr(agent, "batch_size", None)
    _require(isinstance(queues, list), "agent memory_queue is unavailable")
    _require(
        isinstance(memory_size, int) and not isinstance(memory_size, bool),
        "agent memory_size is unavailable",
    )
    _require(
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size >= 1
        and len(queues) == batch_size,
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
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    navdp_dir = Path(args.navdp_dir).resolve()
    server_path = navdp_dir / "navdp_server.py"
    policy_path = navdp_dir / "policy_agent.py"
    server_sha = verify_source_file(
        server_path, EXPECTED_NAVDP_SERVER_SHA256, "navdp_server.py")
    policy_sha = verify_source_file(
        policy_path, EXPECTED_POLICY_AGENT_SHA256, "policy_agent.py")
    checkpoint_sha = verify_source_file(
        args.checkpoint,
        args.expected_checkpoint_sha256,
        "NavDP checkpoint",
    )
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
        "checkpoint_sha256": checkpoint_sha,
        "wrapper_sha256": sha256_file(__file__),
    }
    register_audit_routes(base, provenance)
    print(json.dumps({
        "status": "native_first_audit_server_ready",
        "protocol": AUDIT_PROTOCOL,
        "host": args.host,
        "port": args.port,
        "provenance": provenance,
    }, sort_keys=True), flush=True)
    base.app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
