#!/usr/bin/env python3
"""Audit what X-NavDP actually changes relative to the frozen NavDP model.

This is a structural/checkpoint audit, not a navigation evaluation.  It loads
both checkpoints on CPU with ``weights_only=True`` and reports exact tensor
equality by module.  Optional pinned source inspection records whether the
released evaluation path exposes ImageGoal inference or only PointGoal.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
OFFICIAL_XNAVDP_COMMIT = "878740a2011856d0e3782dd6ccd880fd2eccd70f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor_mapping(path: Path) -> Mapping[str, torch.Tensor]:
    """Safely load a plain state dict and reject non-tensor payloads."""

    try:
        state = torch.load(
            str(path), map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"checkpoint is not a non-empty mapping: {path}")
    invalid = [key for key, value in state.items()
               if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise RuntimeError(
            f"checkpoint contains non-tensor state entries: {invalid[:5]}")
    return state


def _prefix(key: str) -> str:
    return key.split(".", 1)[0]


def compare_state_dicts(
    base: Mapping[str, torch.Tensor],
    post: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return exact, shape-aware per-module checkpoint differences."""

    base_keys = set(base)
    post_keys = set(post)
    shared = sorted(base_keys & post_keys)
    shape_mismatch = [
        key for key in shared if tuple(base[key].shape) != tuple(post[key].shape)
    ]
    comparable = [key for key in shared if key not in set(shape_mismatch)]
    exact = [key for key in comparable if torch.equal(base[key], post[key])]
    exact_set = set(exact)
    changed = [key for key in comparable if key not in exact_set]
    base_only = sorted(base_keys - post_keys)
    post_only = sorted(post_keys - base_keys)

    module_rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "base_tensors": 0,
            "post_tensors": 0,
            "shared_same_shape": 0,
            "exact_equal": 0,
            "changed": 0,
            "shape_mismatch": 0,
            "base_only": 0,
            "post_only": 0,
        })
    for key in base:
        module_rows[_prefix(key)]["base_tensors"] += 1
    for key in post:
        module_rows[_prefix(key)]["post_tensors"] += 1
    for key in comparable:
        module_rows[_prefix(key)]["shared_same_shape"] += 1
    for key in exact:
        module_rows[_prefix(key)]["exact_equal"] += 1
    for key in changed:
        module_rows[_prefix(key)]["changed"] += 1
    for key in shape_mismatch:
        module_rows[_prefix(key)]["shape_mismatch"] += 1
    for key in base_only:
        module_rows[_prefix(key)]["base_only"] += 1
    for key in post_only:
        module_rows[_prefix(key)]["post_only"] += 1

    def fully_equal(prefix: str) -> bool:
        row = module_rows.get(prefix)
        return bool(
            row
            and row["base_tensors"] > 0
            and row["base_tensors"] == row["post_tensors"]
            and row["exact_equal"] == row["base_tensors"]
            and row["changed"] == 0
            and row["shape_mismatch"] == 0
            and row["base_only"] == 0
            and row["post_only"] == 0
        )

    return {
        "base_tensor_count": len(base),
        "post_tensor_count": len(post),
        "base_parameter_count": int(sum(value.numel() for value in base.values())),
        "post_parameter_count": int(sum(value.numel() for value in post.values())),
        "shared_key_count": len(shared),
        "shared_same_shape_count": len(comparable),
        "exact_equal_count": len(exact),
        "changed_count": len(changed),
        "shape_mismatch_count": len(shape_mismatch),
        "base_only_count": len(base_only),
        "post_only_count": len(post_only),
        "shape_mismatch_keys": shape_mismatch,
        "base_only_prefixes": sorted({_prefix(key) for key in base_only}),
        "post_only_prefixes": sorted({_prefix(key) for key in post_only}),
        "modules": dict(sorted(module_rows.items())),
        "findings": {
            "image_encoder_present_in_post": any(
                key.startswith("image_encoder.") for key in post),
            "image_encoder_exactly_equal_to_base": fully_equal("image_encoder"),
            "point_encoder_exactly_equal_to_base": fully_equal("point_encoder"),
            "base_decoder_exactly_equal_to_base": fully_equal("decoder"),
            "rgbd_encoder_exactly_equal_to_base": fully_equal("rgbd_encoder"),
            "fine_tuned_decoder_present": any(
                key.startswith("decoder_ft.") for key in post),
            "twin_q_heads_present": (
                any(key.startswith("q1_heads.") for key in post)
                and any(key.startswith("q2_heads.") for key in post)),
        },
    }


def inspect_official_source(root: Path) -> dict[str, Any]:
    """Record task-interface facts from a pinned official sparse checkout."""

    policy = root / "baselines/x-navdp/src/x_navdp/models/x_navdp_policy.py"
    eval_policy = (
        root / "baselines/x-navdp/eval/src/policy_network_embodiment.py")
    observation = root / "baselines/x-navdp/src/training/observation.py"
    for path in (policy, eval_policy, observation):
        if not path.is_file():
            raise RuntimeError(f"official source file is missing: {path}")
    commit = None
    git_head = root / ".git" / "HEAD"
    if git_head.is_file():
        import subprocess
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
    train_text = policy.read_text(encoding="utf-8")
    eval_text = eval_policy.read_text(encoding="utf-8")
    observation_text = observation.read_text(encoding="utf-8")
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "commit_matches_pin": commit == OFFICIAL_XNAVDP_COMMIT,
        "files": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (policy, eval_policy, observation)
        },
        "training_model_declares_image_encoder": (
            "self.image_encoder = ImageGoalBackbone" in train_text),
        "training_action_path_reads_pointgoal": (
            "if 'pointgoal' in observations.keys()" in train_text),
        "eval_model_declares_image_encoder": (
            "image_encoder" in eval_text or "ImageGoal" in eval_text),
        "observation_adapter_exports_pointgoal": (
            '"pointgoal": observation["goal_pose"]' in observation_text),
        "observation_adapter_exports_imagegoal": (
            '"imagegoal"' in observation_text),
    }


def audit(base_path: Path, post_path: Path,
          official_root: Path | None = None) -> dict[str, Any]:
    base = load_tensor_mapping(base_path)
    post = load_tensor_mapping(post_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "CPU-only exact checkpoint/source compatibility audit; no policy "
            "quality or deployment claim"),
        "provenance": {
            "base_checkpoint": str(base_path.resolve()),
            "base_checkpoint_bytes": base_path.stat().st_size,
            "base_checkpoint_sha256": sha256_file(base_path),
            "post_checkpoint": str(post_path.resolve()),
            "post_checkpoint_bytes": post_path.stat().st_size,
            "post_checkpoint_sha256": sha256_file(post_path),
        },
        "comparison": compare_state_dicts(base, post),
        "official_source": (
            inspect_official_source(official_root)
            if official_root is not None else None),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--post-checkpoint", type=Path, required=True)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        args.base_checkpoint, args.post_checkpoint, args.official_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
