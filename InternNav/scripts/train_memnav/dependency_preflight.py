#!/usr/bin/env python
"""Fail-fast MemNav training dependency and configuration preflight.

This intentionally avoids loading multi-GB KV payloads or constructing LingBot.  It
checks the exact Python environment, configured paths, one representative versioned
cache pair (headers only), warm-start checkpoint schema, and output writability before
the long training process begins. GPU context ownership is checked separately by
``gpu_preflight.py`` so only that transient failure is requeued.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile

# Executing ``python scripts/train_memnav/this_file.py`` puts only this script's
# directory on sys.path. Resolve the InternNav root from the file itself so the
# preflight is independent of cwd and caller-provided PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def require_path(name: str, value: str | None, *, file: bool = False) -> Path:
    if not value:
        raise RuntimeError(f"{name} is empty")
    path = Path(value)
    ok = path.is_file() if file else path.is_dir()
    if not ok:
        kind = "file" if file else "directory"
        raise RuntimeError(f"{name} {kind} missing: {path}")
    if not os.access(path, os.R_OK):
        raise RuntimeError(f"{name} is not readable: {path}")
    return path


def first_file(root: Path, filename: str) -> Path:
    try:
        return next(root.rglob(filename))
    except StopIteration as exc:
        raise RuntimeError(f"no {filename} found below {root}") from exc


def check_imports() -> None:
    required = [
        "torch", "transformers", "diffusers", "numpy", "pandas", "pyarrow",
        "PIL", "cv2",
    ]
    if os.environ.get("MEMNAV_REPORT_TO", "wandb") == "wandb":
        required.append("wandb")
    versions = []
    for name in required:
        module = importlib.import_module(name)
        versions.append(f"{name}={getattr(module, '__version__', '?')}")
    # This import used to fail in clean worktrees because encoder/__init__.py eagerly
    # imported the unrelated, ignored Long-CLIP checkout. Keep it as a regression check.
    from internnav.model.basemodel.memnav.memnav_policy import MemNavNet  # noqa: F401
    from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset  # noqa: F401
    # Exercise the exact model-selection path used by the production train.py entrypoint.
    # The full-batch preflight used to import MemNav classes directly and therefore
    # missed train.py's eager import of the unrelated, gitignored Long-CLIP checkout.
    from scripts.train.train import _load_runtime
    runtime = _load_runtime("memnav")
    if runtime.dataset_class is not MemNav_Dataset:
        raise RuntimeError("train.py selected an unexpected MemNav dataset class")
    expected_init = os.environ.get("MEMNAV_INIT_CKPT", "")
    configured_init = str(runtime.exp_cfg.il.ckpt_to_load or "")
    if configured_init != expected_init:
        raise RuntimeError(
            "MEMNAV_INIT_CKPT is not wired to train config il.ckpt_to_load: "
            f"env={expected_init!r} config={configured_init!r}"
        )
    expected_fusion = os.environ.get("MEMNAV_GATE_FUSION", "complementary")
    if runtime.exp_cfg.il.gate_fusion != expected_fusion:
        raise RuntimeError(
            "MEMNAV_GATE_FUSION is not wired to train config: "
            f"env={expected_fusion!r} config={runtime.exp_cfg.il.gate_fusion!r}"
        )
    expected_top1_weight = float(os.environ.get("MEMNAV_RETRIEVAL_TOP1_WEIGHT", "0.0"))
    expected_top1_margin = float(os.environ.get("MEMNAV_RETRIEVAL_TOP1_MARGIN", "0.2"))
    if (runtime.exp_cfg.il.w_retrieval_top1 != expected_top1_weight
            or runtime.exp_cfg.il.retrieval_top1_margin != expected_top1_margin):
        raise RuntimeError("retrieval top-1 settings are not wired to train config")
    expected_goal_a = os.environ.get("MEMNAV_GOAL_A_MIN_K", "").strip()
    expected_goal_a = int(expected_goal_a) if expected_goal_a else None
    if runtime.exp_cfg.il.goal_a_min_k != expected_goal_a:
        raise RuntimeError("MEMNAV_GOAL_A_MIN_K is not wired to train config")
    expected_swap_weight = float(os.environ.get("MEMNAV_GOAL_SWAP_WEIGHT", "0.0"))
    expected_swap_margin = float(os.environ.get("MEMNAV_GOAL_SWAP_MARGIN", "0.05"))
    expected_swap_angle = float(os.environ.get("MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG", "30.0"))
    if (runtime.exp_cfg.il.w_goal_swap != expected_swap_weight
            or runtime.exp_cfg.il.goal_swap_margin != expected_swap_margin
            or runtime.exp_cfg.il.goal_swap_min_angle_deg != expected_swap_angle
            or runtime.exp_cfg.il.goal_swap_negatives != (expected_swap_weight > 0)):
        raise RuntimeError("goal-swap settings are not wired to train config")
    print("[dependency-preflight] imports OK: " + " ".join(versions), flush=True)


def check_cache(feature_root: Path) -> None:
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_files

    agg = first_file(feature_root, "lingbot_cache.npz")
    cam = agg.with_name("lingbot_cam_cache.npz")
    if not cam.is_file():
        raise RuntimeError(f"representative cache lacks camera pair: {cam}")
    layout = validate_cache_files(
        agg,
        cam,
        expected_num_scale_frames=int(os.environ.get("MEMNAV_NUM_SCALE", "8")),
        expected_sliding_window=int(os.environ.get("MEMNAV_WINDOW", "32")),
        require_versioned=os.environ.get("MEMNAV_REQUIRE_VERSIONED_CACHE", "").lower()
        in ("1", "true", "yes"),
    )
    print(
        f"[dependency-preflight] cache OK: {agg} frames={layout.num_frames} "
        f"interval={layout.keyframe_interval} "
        f"window={int(os.environ.get('MEMNAV_WINDOW', '32'))}",
        flush=True,
    )


def check_checkpoint() -> None:
    value = os.environ.get("MEMNAV_INIT_CKPT", "")
    if not value:
        print("[dependency-preflight] warm start: fresh initialization", flush=True)
        return
    path = require_path("MEMNAV_INIT_CKPT", value, file=True)
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a state dict: {path}")
    required = {"core.retrieval.gate_a", "core.revisit_merge.revisit_head.weight",
                "core.decoder.layers.0.self_attn.in_proj_weight"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"warm-start checkpoint lacks MemNav tensors: {missing}")
    print(f"[dependency-preflight] warm start OK: {path} ({len(state)} tensors)", flush=True)


def check_output() -> None:
    name = os.environ.get("NAME", "memnav_mp3d")
    output = Path("checkpoints") / name / "ckpts"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".preflight-", dir=output, delete=True):
        pass
    print(f"[dependency-preflight] output writable: {output.resolve()}", flush=True)


def main() -> int:
    try:
        root = require_path("MEMNAV_ROOT_DIR", os.environ.get("MEMNAV_ROOT_DIR"))
        feature = require_path("MEMNAV_FEATURE_ROOT", os.environ.get("MEMNAV_FEATURE_ROOT"))
        lingbot = require_path("LINGBOT_REPO", os.environ.get("LINGBOT_REPO"))
        require_path("LINGBOT_WEIGHTS", os.environ.get("LINGBOT_WEIGHTS"), file=True)
        if not (lingbot / "lingbot_map" / "models" / "gct_stream.py").is_file():
            raise RuntimeError(f"LINGBOT_REPO is incomplete: {lingbot}")
        print(f"[dependency-preflight] data root readable: {root}", flush=True)
        check_imports()
        check_cache(feature)
        check_checkpoint()
        check_output()
    except Exception as exc:
        print(f"[dependency-preflight] FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("[dependency-preflight] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
