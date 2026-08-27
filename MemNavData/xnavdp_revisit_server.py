#!/usr/bin/env python3
"""Deterministic, provenance-checked X-NavDP PointGoal HTTP adapter.

This server imports the *unmodified* official X-NavDP evaluation agent from a
commit-pinned checkout.  It adds only the contracts required by the Habitat
revisit benchmark: source/checkpoint receipts, deterministic request seeds,
an observation-only history replay endpoint, and fail-closed JSON validation.

It intentionally exposes no ImageGoal endpoint.  Goal images remain the
memory router's responsibility; this process is only a PointGoal controller.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

import cv2
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import torch

try:
    from .xnavdp_revisit_contract import (
        OFFICIAL_XNAVDP_COMMIT,
        OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        XNAVDP_ALGO,
        XNAVDP_CHECKPOINT_TENSOR_COUNT,
        XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        XNAVDP_MODEL_STATE_TENSOR_COUNT,
    )
except ImportError:  # Script execution from MemNavData/.
    from xnavdp_revisit_contract import (
        OFFICIAL_XNAVDP_COMMIT,
        OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        XNAVDP_ALGO,
        XNAVDP_CHECKPOINT_TENSOR_COUNT,
        XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        XNAVDP_MODEL_STATE_TENSOR_COUNT,
    )


app = Flask(__name__)
_lock = threading.RLock()
_navigator = None
_agent_class = None
_official_root: Path | None = None
_checkpoint: Path | None = None
_device = "cuda:0"
_embodiment_name = "wheeled"
_embodiment_index = 0
_actor_mode = "posttrain"
_checkpoint_sha256: str | None = None
_checkpoint_load_audit: dict[str, Any] = {"audited": False}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _official_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_clean_official_checkout(root: Path) -> None:
    """Reject tracked source edits hidden behind an otherwise valid HEAD."""

    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "official X-NavDP checkout has tracked modifications: "
            f"{status.splitlines()[0]}")


def configure(
    *,
    official_root: Path,
    checkpoint: Path,
    device: str,
    embodiment: str,
    actor_mode: str,
) -> None:
    """Validate immutable assets and make the official agent importable."""

    global _agent_class, _official_root, _checkpoint, _device
    global _embodiment_name, _embodiment_index, _actor_mode
    global _checkpoint_sha256

    official_root = Path(official_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    commit = _official_commit(official_root)
    if commit != OFFICIAL_XNAVDP_COMMIT:
        raise RuntimeError(
            f"official X-NavDP commit {commit} differs from frozen pin "
            f"{OFFICIAL_XNAVDP_COMMIT}")
    _assert_clean_official_checkout(official_root)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != OFFICIAL_XNAVDP_POSTTRAIN_SHA256:
        raise RuntimeError(
            "X-NavDP checkpoint SHA256 differs from the frozen release")
    embodiment_map = {"wheeled": 0, "humanoid": 1, "quadruped": 2}
    if embodiment not in embodiment_map:
        raise ValueError(f"unsupported embodiment: {embodiment}")
    if actor_mode not in {"posttrain", "base"}:
        raise ValueError(f"unsupported actor mode: {actor_mode}")

    eval_root = official_root / "baselines/x-navdp/eval"
    if not (eval_root / "src/policy_agent.py").is_file():
        raise RuntimeError("official X-NavDP evaluation source is incomplete")
    eval_root_text = str(eval_root)
    if eval_root_text not in sys.path:
        sys.path.insert(0, eval_root_text)
    from src.policy_agent import NavDP_Agent

    _agent_class = NavDP_Agent
    _official_root = official_root
    _checkpoint = checkpoint
    _device = str(device)
    _embodiment_name = embodiment
    _embodiment_index = embodiment_map[embodiment]
    _actor_mode = actor_mode
    _checkpoint_sha256 = checkpoint_sha


def _receipt() -> dict[str, Any]:
    return {
        "algo": XNAVDP_ALGO,
        "official_commit": OFFICIAL_XNAVDP_COMMIT,
        "checkpoint_sha256": _checkpoint_sha256,
        "actor_mode": _actor_mode,
        "embodiment": _embodiment_name,
        "checkpoint_load_audit": dict(_checkpoint_load_audit),
    }


def _require_navigator():
    if _navigator is None:
        raise RuntimeError("navigator is not initialized; call /navigator_reset")
    return _navigator


def _build_navigator(intrinsic: np.ndarray):
    global _checkpoint_load_audit
    if _agent_class is None or _checkpoint is None:
        raise RuntimeError("server was not configured")
    _checkpoint_load_audit = {"audited": False}
    navigator = _agent_class(
        intrinsic,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        navi_model=str(_checkpoint),
        device=_device,
        embodiment=_embodiment_index,
        is_real=False,
    )
    if _actor_mode == "base":
        # Exact shared base actor, while keeping the same preprocessing and Q
        # readout.  This mode is engineering-only; the formal X arm is posttrain.
        navigator.navi_former.ft_step = 0
    _checkpoint_load_audit = _audit_checkpoint_model_coverage(
        navigator.navi_former, _checkpoint)
    return navigator


def _audit_checkpoint_model_coverage(
    model: torch.nn.Module, checkpoint_path: Path,
) -> dict[str, Any]:
    """Prove that ``strict=False`` left no eval-model tensor uninitialized.

    The released checkpoint intentionally contains ImageGoal/pixel modules that
    the official PointGoal evaluation class does not construct.  Extra tensors
    are therefore expected, while a missing or shape-mismatched model tensor is
    always fatal.
    """

    load_args = {
        "map_location": "cpu",
        "weights_only": True,
        "mmap": True,
    }
    try:
        checkpoint_state = torch.load(str(checkpoint_path), **load_args)
    except TypeError:  # Compatibility with older Torch releases.
        load_args.pop("weights_only", None)
        load_args.pop("mmap", None)
        checkpoint_state = torch.load(str(checkpoint_path), **load_args)
    if (isinstance(checkpoint_state, dict)
            and isinstance(checkpoint_state.get("state_dict"), dict)):
        checkpoint_state = checkpoint_state["state_dict"]
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError("X-NavDP checkpoint is not a state-dict mapping")

    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = sorted(
        key for key in model_keys & checkpoint_keys
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape))
    audit = {
        "audited": True,
        "model_tensor_count": len(model_state),
        "checkpoint_tensor_count": len(checkpoint_state),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
    }
    expected = {
        "audited": True,
        "model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
        "checkpoint_tensor_count": XNAVDP_CHECKPOINT_TENSOR_COUNT,
        "missing_count": 0,
        "unexpected_count": XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        "shape_mismatch_count": 0,
    }
    if audit != expected:
        raise RuntimeError(
            "X-NavDP checkpoint/model coverage differs from the frozen audit: "
            f"observed={audit}, expected={expected}, "
            f"first_missing={missing[:3]}, "
            f"first_shape_mismatch={shape_mismatch[:3]}")
    return audit


@contextmanager
def deterministic_rng(seed: int | None) -> Iterator[int | None]:
    """Isolate Torch and NumPy global RNG mutation for one HTTP request."""

    if seed is None:
        yield None
        return
    seed = int(seed)
    numpy_state = np.random.get_state()
    devices: list[int] = []
    if str(_device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(_device).index
        devices = [torch.cuda.current_device() if index is None else index]
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2 ** 32))
            yield seed
    finally:
        np.random.set_state(numpy_state)


def _decode_rgb(batch_size: int) -> np.ndarray:
    image = Image.open(request.files["image"].stream).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if bgr.shape[0] % batch_size:
        raise ValueError("RGB wire height is not divisible by batch size")
    return bgr.reshape((batch_size, -1, bgr.shape[1], 3))


def _decode_depth(batch_size: int) -> np.ndarray:
    depth = Image.open(request.files["depth"].stream).convert("I")
    depth = np.asarray(depth, dtype=np.float32)[..., None] / 10000.0
    if depth.shape[0] % batch_size:
        raise ValueError("depth wire height is not divisible by batch size")
    return depth.reshape((batch_size, -1, depth.shape[1], 1))


def _json_form(name: str, default: Any = None) -> Any:
    raw = request.form.get(name)
    if raw is None:
        return default
    return json.loads(raw)


def _request_seed() -> int | None:
    raw = request.form.get("diffusion_seed")
    return None if raw is None else int(raw)


@app.errorhandler(Exception)
def _handle_error(error):
    status = 400 if isinstance(error, (KeyError, TypeError, ValueError)) else 500
    return jsonify({
        **_receipt(),
        "error": f"{type(error).__name__}: {error}",
    }), status


@app.route("/health", methods=["GET"])
def health():
    return jsonify({**_receipt(), "initialized": _navigator is not None})


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    global _navigator
    with _lock:
        body = request.get_json(force=True)
        intrinsic = np.asarray(body["intrinsic"], dtype=np.float64)
        batch_size = int(body["batch_size"])
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic must be a finite 3x3 matrix")
        if batch_size != 1:
            raise ValueError("formal revisit controller is frozen to batch size 1")
        if _navigator is None:
            _navigator = _build_navigator(intrinsic)
        else:
            existing = np.asarray(_navigator.image_intrinsic, dtype=np.float64)
            if existing.shape != intrinsic.shape or not np.allclose(
                    existing, intrinsic, rtol=0.0, atol=1e-12):
                raise ValueError("camera intrinsic changed without restarting server")
        _navigator.reset(batch_size)
        return jsonify({
            **_receipt(),
            "batch_size": batch_size,
            "history_frame_count": list(_navigator.frame_count),
        })


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    with _lock:
        navigator = _require_navigator()
        env_id = int(request.get_json(force=True)["env_id"])
        if env_id != 0:
            raise ValueError("batch-one server only accepts env_id=0")
        navigator.reset_env(env_id)
        return jsonify({
            **_receipt(),
            "env_id": env_id,
            "history_frame_count": list(navigator.frame_count),
        })


@app.route("/memory_replay_step", methods=["POST"])
def memory_replay_step():
    """Append one decision RGB without sampling diffusion or changing RTC."""

    with _lock:
        navigator = _require_navigator()
        image = _decode_rgb(navigator.batch_size)
        processed = navigator.process_image(image)
        before = list(navigator.frame_count)
        navigator._update_and_sample_history(
            processed, num_samples=navigator.memory_size)
        after = list(navigator.frame_count)
        deltas = [int(a) - int(b) for a, b in zip(after, before)]
        if deltas != [1]:
            raise RuntimeError(
                f"history replay frame-count delta is {deltas}, expected [1]")
        return jsonify({
            **_receipt(),
            "diffusion_sampled": False,
            "frames_appended": 1,
            "history_frame_count": after,
        })


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    with _lock:
        navigator = _require_navigator()
        image = _decode_rgb(navigator.batch_size)
        depth = _decode_depth(navigator.batch_size)
        goal_data = _json_form("goal_data")
        goal_x = np.asarray(goal_data["goal_x"], dtype=np.float32)
        goal_y = np.asarray(goal_data["goal_y"], dtype=np.float32)
        if goal_x.shape != (1,) or goal_y.shape != (1,):
            raise ValueError("PointGoal payload must have batch-one x/y vectors")
        goal = np.stack(
            (goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
        if not np.isfinite(goal).all():
            raise ValueError("PointGoal contains NaN or infinity")

        state_data = _json_form("state_data", {})
        robot_pos = state_data.get("robot_pos")
        robot_quat = state_data.get("robot_quat")
        if (robot_pos is None) != (robot_quat is None):
            raise ValueError("robot_pos and robot_quat must be supplied together")
        if robot_pos is not None:
            robot_pos = np.asarray(robot_pos, dtype=np.float64)
            robot_quat = np.asarray(robot_quat, dtype=np.float64)
            if (robot_pos.shape != (1, 3) or robot_quat.shape != (1, 4)
                    or not np.isfinite(robot_pos).all()
                    or not np.isfinite(robot_quat).all()):
                raise ValueError("robot state must be finite [1,3]/[1,4]")

        seed = _request_seed()
        before = list(navigator.frame_count)
        with deterministic_rng(seed):
            trajectory, candidates, values, _ = (
                navigator.step_pointgoal_with_guidance(
                    goal, image, depth, robot_pos, robot_quat))
        after = list(navigator.frame_count)
        deltas = [int(a) - int(b) for a, b in zip(after, before)]
        if deltas != [1]:
            raise RuntimeError(
                f"PointGoal request frame-count delta is {deltas}, expected [1]")

        trajectory = np.asarray(trajectory, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if (trajectory.shape != (1, 24, 3)
                or candidates.ndim != 4
                or candidates.shape[0] != 1
                or candidates.shape[2:] != (24, 3)
                or values.shape != candidates.shape[:2]
                or not np.isfinite(trajectory).all()
                or not np.isfinite(candidates).all()
                or not np.isfinite(values).all()):
            raise RuntimeError(
                "official X-NavDP returned an invalid trajectory/Q shape")
        return jsonify({
            **_receipt(),
            "controller": (
                "xnavdp_point_posttrain" if _actor_mode == "posttrain"
                else "xnavdp_point_base"),
            "trajectory": trajectory.tolist(),
            "all_trajectory": candidates.tolist(),
            "all_values": values.tolist(),
            "diffusion_seed": seed,
            "frames_appended": 1,
            "history_frame_count": after,
            "rtc_robot_state_used": robot_pos is not None,
        })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    response = jsonify({**_receipt(), "status": "shutting_down"})

    def exit_later():
        time.sleep(0.15)
        os._exit(0)

    threading.Thread(target=exit_later, daemon=True).start()
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root", type=Path,
        default=Path(".diagnostics/xnavdp_official_878740a2011856d0/NavDP"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            ".diagnostics/xnavdp_official_878740a2011856d0/"
            "x-navdp_posttrain.ckpt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--embodiment", choices=("wheeled", "humanoid", "quadruped"),
        default="wheeled")
    parser.add_argument(
        "--actor-mode", choices=("posttrain", "base"),
        default="posttrain")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18889)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure(
        official_root=args.official_root,
        checkpoint=args.checkpoint,
        device=args.device,
        embodiment=args.embodiment,
        actor_mode=args.actor_mode,
    )
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
