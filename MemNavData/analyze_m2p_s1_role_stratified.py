#!/usr/bin/env python3
"""Role-stratified analysis for the M2P S-1 dual-context collection.

The primary unit is factual ``goal_c_t0`` (true Revisit).  Novel-B start,
Novel-B midpoint, and counterfactual goals are retained as controls but are
never pooled into the primary Revisit rate.  The script also joins the pinned
production ``goal_append_warm`` rows and a train-only causal teacher to audit
runtime parity and support recency without reading development/blind data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_TEACHER_SHA = (
    "dfd56abff5e17f1d687eab511bdf545e3e1bcff5d74509a6b128263cd7c85027")
EXPECTED_PRODUCTION_SHA = (
    "193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_state(session_id: str) -> str:
    if "/goal_c_t0/" in session_id:
        return "true_revisit_c_t0"
    if "/goal_b_midpoint_t1/" in session_id:
        return "novel_b_midpoint_t1"
    if "/goal_b_t0/" in session_id:
        return "novel_b_t0"
    raise ValueError(f"unknown causal state in {session_id}")


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _cdf(values: Iterable[float], threshold: float) -> dict[str, Any]:
    finite = [float(value) for value in values
              if math.isfinite(float(value))]
    hits = sum(value <= threshold for value in finite)
    return {"hits": hits, "total": len(finite),
            "rate": hits / len(finite) if finite else None}


def error_summary(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values
              if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "median_deg": float(np.median(finite)) if finite else None,
        "cdf_le_15": _cdf(finite, 15.0),
        "cdf_le_30": _cdf(finite, 30.0),
        "cdf_le_45": _cdf(finite, 45.0),
    }


def quadrant_summary(frame: pd.DataFrame, *,
                     anchored_column: str) -> dict[str, Any]:
    anchored = frame[anchored_column].le(30.0)
    full = frame["full_prefix_error_deg"].le(30.0)
    both = int((anchored & full).sum())
    anchored_only = int((anchored & ~full).sum())
    full_only = int((~anchored & full).sum())
    neither = int((~anchored & ~full).sum())
    total = len(frame)
    return {
        "n": total,
        "both_good": both,
        "anchored_only_good": anchored_only,
        "full_prefix_only_good": full_only,
        "neither_good": neither,
        "anchored_cdf30": (both + anchored_only) / total if total else None,
        "full_prefix_cdf30": (both + full_only) / total if total else None,
        "oracle_union_cdf30": (
            (both + anchored_only + full_only) / total if total else None),
        "full_minus_anchored_paired": {
            "gains": full_only,
            "losses": anchored_only,
            "risk_difference": (
                (full_only - anchored_only) / total if total else None),
            "exact_mcnemar_p": exact_mcnemar_p(full_only, anchored_only),
        },
        "oracle_union_minus_anchored": (
            full_only / total if total else None),
    }


def cluster_bootstrap(frame: pd.DataFrame, *, resamples: int,
                      seed: int) -> dict[str, Any]:
    scenes = sorted(frame["scene"].astype(str).unique())
    if len(scenes) < 2:
        return {"available": False, "reason": "fewer_than_two_scenes"}
    by_scene = {scene: frame.loc[frame["scene"].eq(scene)]
                for scene in scenes}
    rng = np.random.default_rng(seed)
    full_minus_anchored = []
    union_minus_anchored = []
    for _ in range(resamples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        sample = pd.concat([by_scene[str(scene)] for scene in selected],
                           ignore_index=True)
        anchored = sample["anchored_error_deg"].le(30.0)
        full = sample["full_prefix_error_deg"].le(30.0)
        full_minus_anchored.append(float(full.mean() - anchored.mean()))
        union_minus_anchored.append(float((anchored | full).mean()
                                          - anchored.mean()))

    def interval(values: list[float]) -> dict[str, Any]:
        return {
            "median": float(np.median(values)),
            "ci95": [float(np.percentile(values, 2.5)),
                     float(np.percentile(values, 97.5))],
        }
    return {
        "available": True,
        "scenes": len(scenes),
        "resamples": resamples,
        "full_minus_anchored": interval(full_minus_anchored),
        "oracle_union_minus_anchored": interval(union_minus_anchored),
    }


def support_table(teacher: pd.DataFrame) -> pd.DataFrame:
    records = []
    for session_id, group in teacher.groupby("session_id", sort=True):
        decision_values = group["decision_frame"].drop_duplicates().tolist()
        require(len(decision_values) == 1,
                f"teacher decision changes in {session_id}")
        decision = int(decision_values[0])
        positives = group.loc[group["teacher_covis"].ge(0.5)]
        if positives.empty:
            latest = best = None
            latest_gap = best_gap = None
        else:
            latest = int(positives["candidate_frame"].max())
            best_row = positives.sort_values(
                ["teacher_covis", "candidate_frame"],
                ascending=[False, False]).iloc[0]
            best = int(best_row["candidate_frame"])
            latest_gap = decision - 1 - latest
            best_gap = decision - 1 - best
        records.append({
            "session_id": str(session_id),
            "teacher_positive_count": int(len(positives)),
            "latest_positive_frame": latest,
            "latest_positive_gap": latest_gap,
            "best_positive_frame": best,
            "best_positive_gap": best_gap,
            "teacher_session_max_covis": float(group["teacher_covis"].max()),
        })
    return pd.DataFrame(records)


def recall_gap_band(value: Any) -> str:
    if pd.isna(value):
        return "no_teacher_support"
    gap = int(value)
    if gap <= 32:
        return "within_gct_window_le32"
    if gap <= 128:
        return "mid_gap_33_128"
    return "long_gap_gt128"


def partial_futility_decision(*, scenes: int,
                              primary_quadrants: dict[str, Any],
                              identity_ok: bool,
                              production_parity_ok: bool) -> dict[str, Any]:
    """Apply the frozen partial-run rule; it can never declare success."""
    if not 10 <= scenes < 40:
        return {
            "scope": "not_evaluated",
            "continue_full40": None,
            "reason": None,
        }
    catastrophic = (
        primary_quadrants["full_prefix_only_good"] == 0
        and primary_quadrants["anchored_only_good"] >= 5
        and primary_quadrants["full_prefix_cdf30"]
        <= primary_quadrants["anchored_cdf30"])
    stop_reasons = []
    if not identity_ok:
        stop_reasons.append("query_state_identity_failure")
    if not production_parity_ok:
        stop_reasons.append("production_cdf30_parity_failure")
    if catastrophic:
        stop_reasons.append("catastrophic_full_prefix_dominance_failure")
    reason = ("not_futile_requires_full40" if not stop_reasons
              else stop_reasons[0])
    return {
        "scope": "futility_only_never_success_claim",
        "continue_full40": bool(
            identity_ok and production_parity_ok and not catastrophic),
        "reason": reason,
        "stop_reasons": stop_reasons,
    }


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    require(sha256_file(args.teacher_train) == args.expected_teacher_sha,
            "train-only teacher SHA mismatch")
    require(sha256_file(args.production_rows) == args.expected_production_sha,
            "production rows SHA mismatch")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    require(report.get("configuration", {}).get(
        "development_or_blind_read") in (False, 0),
        "collector reports development/blind access")
    collected = pd.DataFrame(report["rows"])
    require(not collected.empty, "collector report has no rows")
    require(collected["session_id"].is_unique,
            "collector repeats a session")
    collected = collected.rename(columns={
        "local_direction_error_deg": "anchored_error_deg",
        "global_direction_error_deg": "full_prefix_error_deg",
        "local_raw_norm": "anchored_raw_norm",
        "global_raw_norm": "full_prefix_raw_norm",
    })
    collected["causal_state"] = collected["session_id"].map(causal_state)
    collected["full_to_anchored_norm_ratio"] = (
        collected["full_prefix_raw_norm"]
        / collected["anchored_raw_norm"].clip(lower=1e-9))

    teacher = pd.read_csv(args.teacher_train)
    require(set(teacher["split_role"].astype(str)) == {"train"},
            "teacher artifact is not train-only")
    require(teacher["scene"].nunique() == 40
            and teacher["session_id"].nunique() == 480
            and len(teacher) == 14172,
            "teacher train universe changed")
    collected = collected.merge(
        support_table(teacher), on="session_id", how="left",
        validate="one_to_one")
    require(collected["teacher_session_max_covis"].notna().all(),
            "collector session absent from teacher")
    collected["recall_gap_band"] = collected[
        "latest_positive_gap"].map(recall_gap_band)

    production = pd.read_csv(args.production_rows)
    production = production[[
        "session_id", "candidate_frame",
        "relative_position_direction_error_deg_center",
    ]].rename(columns={
        "candidate_frame": "production_anchor",
        "relative_position_direction_error_deg_center": (
            "production_anchored_error_deg"),
    })
    collected = collected.merge(
        production,
        left_on=["session_id", "dino_anchor"],
        right_on=["session_id", "production_anchor"],
        how="left", validate="one_to_one")
    require(collected["production_anchored_error_deg"].notna().all(),
            "production DINO anchor is absent")
    collected["anchored_good"] = collected["anchored_error_deg"].le(30.0)
    collected["production_anchored_good"] = collected[
        "production_anchored_error_deg"].le(30.0)
    collected["full_prefix_good"] = collected[
        "full_prefix_error_deg"].le(30.0)

    strata: dict[str, Any] = {}
    for (state, variant), frame in collected.groupby(
            ["causal_state", "goal_variant"], sort=True):
        key = f"{state}/{variant}"
        strata[key] = {
            "same_process_quadrants": quadrant_summary(
                frame, anchored_column="anchored_error_deg"),
            "production_reference_quadrants": quadrant_summary(
                frame, anchored_column="production_anchored_error_deg"),
            "anchored_error": error_summary(frame["anchored_error_deg"]),
            "full_prefix_error": error_summary(
                frame["full_prefix_error_deg"]),
            "session_labels": {
                str(label): int(count) for label, count in
                frame["session_label"].value_counts().sort_index().items()
            },
        }

    primary = collected.loc[
        collected["causal_state"].eq("true_revisit_c_t0")
        & collected["goal_variant"].eq("factual")].copy()
    require(not primary.empty, "no factual Revisit-C rows")
    primary_quadrants = quadrant_summary(
        primary, anchored_column="anchored_error_deg")
    by_gap = {}
    for band, frame in primary.groupby("recall_gap_band", sort=True):
        by_gap[str(band)] = {
            "same_process_quadrants": quadrant_summary(
                frame, anchored_column="anchored_error_deg"),
            "latest_positive_gap": {
                "min": (int(frame["latest_positive_gap"].min())
                        if frame["latest_positive_gap"].notna().any()
                        else None),
                "median": (float(frame["latest_positive_gap"].median())
                           if frame["latest_positive_gap"].notna().any()
                           else None),
                "max": (int(frame["latest_positive_gap"].max())
                        if frame["latest_positive_gap"].notna().any()
                        else None),
            },
        }

    parity = collected["anchored_good"].eq(
        collected["production_anchored_good"])
    scenes = collected["scene"].nunique()
    identity_ok = bool(
        collected["all_query_state_identity"].astype(bool).all())
    production_parity_ok = bool(parity.all())
    futility = partial_futility_decision(
        scenes=scenes,
        primary_quadrants=primary_quadrants,
        identity_ok=identity_ok,
        production_parity_ok=production_parity_ok)

    summary = {
        "scope": "train_only_role_stratified_s1_not_closed_loop_or_sr",
        "sessions": len(collected),
        "scenes": scenes,
        "episodes": collected[["scene", "episode"]].drop_duplicates().shape[0],
        "development_or_blind_read": False,
        "primary_unit": "true_revisit_c_t0/factual",
        "primary_true_revisit": {
            "same_process_quadrants": primary_quadrants,
            "production_reference_quadrants": quadrant_summary(
                primary,
                anchored_column="production_anchored_error_deg"),
            "scene_cluster_bootstrap": cluster_bootstrap(
                primary, resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed),
            "by_latest_positive_recall_gap": by_gap,
            "production_actionability_parity": {
                "agreement": int(primary["anchored_good"].eq(
                    primary["production_anchored_good"]).sum()),
                "sessions": len(primary),
                "agreement_rate": float(primary["anchored_good"].eq(
                    primary["production_anchored_good"]).mean()),
                "same_process_hits": int(primary["anchored_good"].sum()),
                "production_hits": int(
                    primary["production_anchored_good"].sum()),
            },
        },
        "causal_role_controls": strata,
        "production_parity": {
            "same_anchor": int(collected["production_anchor"].eq(
                collected["dino_anchor"]).sum()),
            "sessions": len(collected),
            "cdf30_agreement": int(parity.sum()),
            "cdf30_agreement_rate": float(parity.mean()),
            "continuous_absolute_difference_deg": {
                "median": float(np.median(np.abs(
                    collected["anchored_error_deg"]
                    - collected["production_anchored_error_deg"]))),
                "max": float(np.max(np.abs(
                    collected["anchored_error_deg"]
                    - collected["production_anchored_error_deg"]))),
            },
        },
        "all_query_state_identity": bool(
            collected["all_query_state_identity"].astype(bool).all()),
        "futility_decision": futility,
        "authorization": {
            "effectiveness_claim": False,
            "selective_m2p_training": False,
            "closed_loop": False,
        },
        "input_sha256": {
            "collector_report": sha256_file(args.report),
            "teacher_train": sha256_file(args.teacher_train),
            "production_rows": sha256_file(args.production_rows),
        },
    }
    return summary, collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--production-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha", default=EXPECTED_TEACHER_SHA)
    parser.add_argument("--expected-production-sha",
                        default=EXPECTED_PRODUCTION_SHA)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    for path in (args.report, args.teacher_train, args.production_rows):
        require(path.is_file(), f"missing input: {path}")
    require(args.bootstrap_resamples >= 1, "bootstrap count must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary, rows = analyze(args)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    rows.to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
