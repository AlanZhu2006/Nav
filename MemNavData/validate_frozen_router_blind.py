#!/usr/bin/env python3
"""Fail closed if a blind router run refits or retunes the frozen model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_KEYS = (
    "candidate_min_frame_gap",
    "candidate_selection",
    "coefficient",
    "feature_names",
    "feature_version",
    "grid_size",
    "intercept",
    "lingbot_commit",
    "lingbot_weight_sha256",
    "mean",
    "patch_relation",
    "scale",
    "thresholds",
    "top_k",
    "train_scenes",
)
LISTWISE_KEYS = (
    "coefficient",
    "l2",
    "mean",
    "positive_threshold",
    "scale",
    "training_sessions",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compare_value(name: str, reference: Any, candidate: Any) -> None:
    if isinstance(reference, list) and reference and all(
            isinstance(value, (int, float)) for value in reference):
        require(
            np.allclose(reference, candidate, rtol=0.0, atol=1e-12),
            f"frozen numeric field changed: {name}",
        )
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        require(
            bool(np.isclose(reference, candidate, rtol=0.0, atol=1e-12)),
            f"frozen numeric field changed: {name}",
        )
    else:
        require(reference == candidate, f"frozen field changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-reference-sha", required=True)
    parser.add_argument("--expected-heldout-scene", action="append", required=True)
    args = parser.parse_args()

    actual_reference_sha = sha256(args.reference)
    require(
        actual_reference_sha == args.expected_reference_sha,
        "reference router SHA256 mismatch",
    )
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    report = json.loads(args.report.read_text())

    expected_heldout = sorted(args.expected_heldout_scene)
    require(
        sorted(candidate["heldout_scenes"]) == expected_heldout,
        "candidate heldout scenes do not match the blind manifest",
    )
    require(
        sorted(report["heldout_scenes"]) == expected_heldout,
        "report heldout scenes do not match the blind manifest",
    )
    require(
        not candidate.get("deployment_approved", True),
        "blind diagnostic must not self-approve deployment",
    )

    for key in MODEL_KEYS:
        compare_value(key, reference[key], candidate[key])
    for key in LISTWISE_KEYS:
        compare_value(
            f"listwise_ranker.{key}",
            reference["listwise_ranker"][key],
            candidate["listwise_ranker"][key],
        )

    print(json.dumps({
        "status": "ok",
        "reference_sha256": actual_reference_sha,
        "reference_heldout_scenes": sorted(reference["heldout_scenes"]),
        "blind_heldout_scenes": expected_heldout,
        "frozen_fields_compared": list(MODEL_KEYS),
        "frozen_listwise_fields_compared": list(LISTWISE_KEYS),
        "deployment_approved": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
