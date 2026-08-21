#!/usr/bin/env python3
"""Export the all-train CDEC pairwise fit as a fail-closed runtime artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

try:
    from MemNavData.cdec_pairwise_runtime import (
        CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
        CDEC_DINO_INFERENCE_BATCH_SIZE,
        CDEC_PATCH_GRID_SIZE,
        sha256,
    )
    from MemNavData.patch_temporal_router import (
        directional_patch_feature_names,
    )
    from MemNavData.train_cdec_pairwise_ranker_oof import SCHEMA_VERSION
except ModuleNotFoundError:  # direct script invocation
    from cdec_pairwise_runtime import (  # type: ignore
        CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
        CDEC_DINO_INFERENCE_BATCH_SIZE,
        CDEC_PATCH_GRID_SIZE,
        sha256,
    )
    from patch_temporal_router import (  # type: ignore
        directional_patch_feature_names,
    )
    from train_cdec_pairwise_ranker_oof import SCHEMA_VERSION  # type: ignore


def build_artifact(report: dict, *, report_path: Path,
                   report_sha256: str) -> dict:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pairwise training report schema")
    protocol = report.get("protocol")
    if (not isinstance(protocol, dict)
            or protocol.get("development_or_blind_read") is not False
            or protocol.get("activation_or_NULL_learned") is not False
            or protocol.get("groups") != "scene"):
        raise ValueError("training report violates the frozen split/authority contract")
    fit = report.get("deployment_fit_on_all_train_scenes")
    if not isinstance(fit, dict):
        raise ValueError("all-train deployment fit is missing")
    feature_names = tuple(map(str, fit.get("feature_names", ())))
    if feature_names != directional_patch_feature_names():
        raise ValueError("training feature schema changed")
    coverage = report.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("scenes") != 40:
        raise ValueError("all-train scene scope changed")
    inputs = report.get("inputs")
    selection = report.get("selection_artifact")
    if not isinstance(inputs, dict) or not isinstance(selection, dict):
        raise ValueError("training input receipts are missing")
    return {
        "schema_version": CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
        # This exporter intentionally cannot approve its own model.  Promotion
        # requires a separate, immutable same-process certificate gate report.
        "deployment_approved": False,
        "approval_state": "pending_same_process_dual_proposal_certificate_gate",
        "training_scope": {
            "scenes": int(coverage["scenes"]),
            "sessions": int(coverage["sessions"]),
            "grouping": "scene",
            "development_or_blind_read": False,
            "labels_available_at_runtime": False,
        },
        "runtime_semantics": {
            "authority": "rank_frozen_causal_shortlist_only",
            "activation_authority": "independent_atomic_pnp_certificate",
            "fallback": "native_imagegoal",
            "score_calibration": "uncalibrated_pairwise_utility",
            "cascade": "geometry_proposal_then_learned_on_certificate_reject",
            "accepted_geometry_can_be_overridden": False,
        },
        "model": {
            "family": "zero_intercept_logistic_pairwise_ranker",
            "selected_C": float(fit["selected_C"]),
            "intercept": 0.0,
            "patch_grid_size": CDEC_PATCH_GRID_SIZE,
            "dino_inference_batch_size": CDEC_DINO_INFERENCE_BATCH_SIZE,
            "relation_storage_dtype": "float32",
            "feature_names": list(feature_names),
            "coefficient": list(map(float, fit["coefficient"])),
            "mean": list(map(float, fit["mean"])),
            "scale": list(map(float, fit["scale"])),
        },
        "receipts": {
            "training_report": str(report_path.resolve()),
            "training_report_sha256": report_sha256,
            "rows_csv_sha256": str(inputs["rows_csv_sha256"]),
            "patch_cache_sha256": str(inputs["patch_cache_sha256"]),
            "oof_selection_sha256": str(selection["sha256"]),
        },
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    observed = sha256(args.report)
    if observed != args.expected_report_sha256:
        raise RuntimeError("pairwise training report SHA256 changed")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    artifact = build_artifact(
        report, report_path=args.report, report_sha256=observed)
    atomic_json(args.output, artifact)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "deployment_approved": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
