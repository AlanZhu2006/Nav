#!/usr/bin/env python3
"""Independently recount the frozen train40 certificate challenge.

This verifier intentionally does not import the production summarizer.  It
reconstructs the center-view certificate and GT-only actionability label from
the raw collector CSV, checks the exact manifest cover, and compares all
confusion counts (overall and stratified) with the published audit JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def center_pnp(payload: str) -> dict:
    hypotheses = json.loads(payload)
    centers = [item for item in hypotheses if int(item["offset"]) == 0]
    if len(centers) != 1 or not isinstance(
            centers[0].get("pnp_lightglue"), dict):
        raise RuntimeError("row lacks one LightGlue-PnP center hypothesis")
    return centers[0]["pnp_lightglue"]


def certificate(pnp: dict) -> bool:
    fields = (
        "inliers", "query_inlier_coverage", "reference_inlier_coverage",
        "reprojection_rmse_px",
    )
    return bool(
        pnp.get("status") == "ok"
        and all(finite(pnp.get(name)) for name in fields)
        and int(pnp["inliers"]) >= 16
        and float(pnp["query_inlier_coverage"]) >= 0.05
        and float(pnp["reference_inlier_coverage"]) >= 0.05
        and float(pnp["reprojection_rmse_px"]) <= 2.0
    )


def actionable(pnp: dict) -> bool:
    return bool(
        pnp.get("status") == "ok"
        and finite(pnp.get("relative_position_error_m"))
        and float(pnp["relative_position_error_m"]) <= 0.75
    )


def support_band(value: object) -> str:
    if not finite(value):
        raise RuntimeError("formal challenge support metadata is non-finite")
    if float(value) <= 0.10:
        return "strict_or_low_le_0p10"
    if float(value) <= 0.50:
        return "boundary_0p10_to_0p50"
    return "strong_gt_0p50"


def history_band(value: object) -> str:
    if not finite(value) or float(value) <= 0:
        raise RuntimeError("formal challenge history gap is invalid")
    if float(value) <= 32:
        return "within_lingbot_window_le_32"
    if float(value) <= 96:
        return "delayed_33_to_96"
    return "long_delay_gt_96"


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "sessions": int(len(frame)),
        "scenes": int(frame["scene"].nunique()),
        "accepted": int(frame["certificate"].sum()),
        "actionable": int(frame["actionable"].sum()),
        "true_positive": int((
            frame["certificate"] & frame["actionable"]).sum()),
        "false_positive": int((
            frame["certificate"] & ~frame["actionable"]).sum()),
        "false_negative": int((
            ~frame["certificate"] & frame["actionable"]).sum()),
        "true_negative": int((
            ~frame["certificate"] & ~frame["actionable"]).sum()),
    }


def grouped_counts(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    return {
        str(name): counts(group)
        for name, group in frame.groupby(column, sort=True)
    }


def compare_counts(observed: dict, reported: dict, context: str) -> None:
    for name, value in observed.items():
        if int(reported.get(name, -1)) != value:
            raise RuntimeError(
                f"{context}.{name} mismatch: {value} != {reported.get(name)}")


def verify(rows: pd.DataFrame, audit: dict, manifest: dict) -> dict:
    expected_sessions = [str(item) for item in manifest["sessions"]]
    if manifest.get("schema_version") != (
            "train40_certificate_challenge_manifest_v1"):
        raise RuntimeError("challenge manifest schema changed")
    if (len(expected_sessions) != 480
            or len(set(expected_sessions)) != 480):
        raise RuntimeError("challenge manifest session cover is invalid")
    if len(rows) != rows["session_id"].nunique() or len(rows) != 480:
        raise RuntimeError("raw collector is not one row per train40 session")
    if set(rows["session_id"].astype(str)) != set(expected_sessions):
        raise RuntimeError("raw collector differs from the frozen session universe")
    records = []
    for row in rows.itertuples(index=False):
        pnp = center_pnp(str(row.hypotheses_json))
        gap = int(row.causal_decision_frame) - int(row.candidate_frame)
        records.append({
            "session_id": str(row.session_id),
            "scene": str(row.scene),
            "certificate": certificate(pnp),
            "actionable": actionable(pnp),
            "selected_anchor_support_band": support_band(row.teacher_covis),
            "session_support_band": support_band(row.session_max_covis),
            "history_gap_band": history_band(gap),
            "causal_state_name": str(row.causal_state_name),
        })
    frame = pd.DataFrame(records)
    overall = counts(frame)
    compare_counts(overall, {
        "sessions": audit["rows"],
        "scenes": audit["scenes"],
        "accepted": audit["certificate"]["accepted"],
        "actionable": audit["ground_truth_actionable"],
        "true_positive": audit["certificate"]["true_positive"],
        "false_positive": audit["certificate"]["false_positive"],
        "false_negative": audit["certificate"]["false_negative"],
        "true_negative": audit["certificate"]["true_negative"],
    }, "overall")
    strata = {
        "selected_anchor_support": grouped_counts(
            frame, "selected_anchor_support_band"),
        "session_max_support": grouped_counts(frame, "session_support_band"),
        "history_gap": grouped_counts(frame, "history_gap_band"),
        "causal_state": grouped_counts(frame, "causal_state_name"),
    }
    reported_strata = audit["stratified_actionability"]
    for axis, groups in strata.items():
        if set(groups) != set(reported_strata[axis]):
            raise RuntimeError(f"{axis} group cover changed")
        for group, values in groups.items():
            compare_counts(values, reported_strata[axis][group], f"{axis}.{group}")
    return {
        "schema_version": (
            "independent_train40_certificate_challenge_verification_v1"),
        "status": "independent_raw_recount_passed_not_closed_loop",
        "overall": overall,
        "stratified_counts": strata,
        "manifest_session_universe_sha256": manifest[
            "session_universe_sha256"],
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = verify(pd.read_csv(args.rows), audit, manifest)
    result["inputs"] = {
        "rows_sha256": sha256_file(args.rows),
        "audit_sha256": sha256_file(args.audit),
        "manifest_sha256": sha256_file(args.manifest),
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
