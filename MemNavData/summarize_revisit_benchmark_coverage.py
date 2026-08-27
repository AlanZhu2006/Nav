#!/usr/bin/env python3
"""Consolidate existing Revisit evidence into a benchmark-coverage audit.

This script is deliberately analysis-only.  It reads train/consumed artifacts,
does not inspect development or blind data, and does not fit or tune a method.
Its purpose is to distinguish evidence volume from evidence coverage before a
new expensive closed-loop run is approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-evidence", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--closed-loop-report", type=Path, required=True)
    parser.add_argument("--observability-audit", type=Path, required=True)
    parser.add_argument("--mixed-safety-audit", type=Path, required=True)
    parser.add_argument("--nnr-report", type=Path, required=True)
    parser.add_argument("--sweep-results", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def zero_event_upper(total: int, alpha: float = 0.05) -> float | None:
    """Exact one-sided binomial upper confidence bound for zero events."""
    if total <= 0:
        return None
    return 1.0 - alpha ** (1.0 / total)


def quantiles(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return {}
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def geometry_inventory(evidence_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    columns = [
        "session_id", "scene", "goal_role", "goal_variant", "teacher_covis",
        "geometry_hard_pass",
    ]
    evidence = pd.read_csv(evidence_path, usecols=columns)
    sessions = evidence.groupby(
        ["session_id", "scene", "goal_role", "goal_variant"], as_index=False
    ).agg(
        max_covis=("teacher_covis", "max"),
        any_geometry_pass=("geometry_hard_pass", "max"),
        candidates=("teacher_covis", "size"),
    )
    sessions["support_band"] = pd.cut(
        sessions["max_covis"],
        [-1e-9, 0.1, 0.5, 1.000001],
        labels=["strict_or_low_le_0p10", "boundary_0p10_to_0p50", "strong_gt_0p50"],
        include_lowest=True,
    )
    bands: dict[str, Any] = {}
    for band, group in sessions.groupby("support_band", observed=False):
        bands[str(band)] = {
            "sessions": int(len(group)),
            "scenes": int(group["scene"].nunique()),
            "geometry_any_pass_rate": float(group["any_geometry_pass"].mean()),
        }
    expected = report.get("input", {})
    if int(expected.get("sessions", -1)) != len(sessions):
        raise RuntimeError("geometry report/session inventory mismatch")
    return {
        "sessions": int(len(sessions)),
        "scenes": int(sessions["scene"].nunique()),
        "candidates": int(len(evidence)),
        "support_bands": bands,
        "stable_support_precision": float(report["stable_support_precision"]),
        "positive_candidate_recall": float(report["hard_positive_candidate_recall"]),
    }


def closed_loop_summary(report: dict[str, Any]) -> dict[str, Any]:
    arms = report["arms"]
    output: dict[str, Any] = {}
    for name in (
        "native", "geometry_router", "known_revisit_direct",
        "certified_relocalization",
    ):
        revisit = arms[name]["revisit_given_novel_success"]
        output[name] = {
            "eligible": int(revisit["eligible"]),
            "successes": int(revisit["successes"]),
            "success_rate": float(revisit["sr"]),
            "mean_spl": float(revisit["mean_spl"]),
        }
    contrast = report["contrasts"]["certified_minus_known_revisit_direct"][
        "conditional_b"
    ]
    output["certified_vs_direct"] = {
        "gains": int(len(contrast["gains"])),
        "losses": int(len(contrast["losses"])),
        "mcnemar_exact_two_sided_p": float(contrast["mcnemar_exact_two_sided_p"]),
        "risk_difference": float(contrast["risk_difference_right_minus_left"]),
        "scene_cluster_bootstrap_95": [
            float(value) for value in contrast["scene_cluster_bootstrap_risk_difference_95"]
        ],
    }
    output["runtime"] = report["certified_runtime"]
    return output


def observability_summary(audit: dict[str, Any]) -> dict[str, Any]:
    rows = pd.DataFrame(audit["rows"])
    reached = rows.loc[rows["trace_reached_a"].astype(bool)].copy()
    if len(reached) != int(audit["summary"]["shared_a_successes"]):
        raise RuntimeError("observability eligible denominator mismatch")
    return {
        "eligible_after_a": int(len(reached)),
        "supported_ge_0p20": int(reached["online_max_covis"].ge(0.2).sum()),
        "strong_ge_0p50": int(reached["online_max_covis"].ge(0.5).sum()),
        "online_max_covis": quantiles(reached["online_max_covis"]),
        "online_path_nearest_distance_m": quantiles(
            reached["online_path_nearest_distance_m"]
        ),
        "online_argmax_goal_yaw_error_deg": quantiles(
            reached["online_argmax_goal_yaw_error_deg"]
        ),
        "online_recall_gap_frames": quantiles(reached["online_recall_gap"]),
    }


def nnr_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != (
            "shared_online_novel_revisit_paired_report_v1_20260814"):
        raise RuntimeError("unexpected actual-online NNR report schema")
    if not report.get("all_shared_prefixes_equal"):
        raise RuntimeError("actual-online NNR A/B prefixes are not paired")
    if not report.get("all_treatment_prefixes_equal"):
        raise RuntimeError("actual-online NNR treatment prefixes differ")
    episodes = int(report["constructible_population_size"])
    if len(report["records"]) != episodes:
        raise RuntimeError("actual-online NNR record cover changed")
    arms = {
        name: {
            "successes": int(report["arms"][name]["successes"]),
            "episodes": int(report["arms"][name]["episodes"]),
            "success_rate": float(
                report["arms"][name]["SR_C_given_frozen_online_AB"]),
        }
        for name in (
            "native", "known_direct", "certified", "certified_budget",
            "certified_graph")
    }
    contrasts = {}
    for name in (
            "certified_minus_native", "known_direct_minus_native",
            "certified_graph_minus_certified_budget"):
        value = report["contrasts"][name]
        contrasts[name] = {
            "gains": int(value["gains"]),
            "losses": int(value["losses"]),
            "mcnemar_exact_two_sided_p": float(
                value["exact_mcnemar_two_sided_p"]),
            "risk_difference_pp": float(value["risk_difference_pp"]),
            "scene_cluster_bootstrap_95ci_pp": [
                float(item) for item in value[
                    "scene_cluster_bootstrap_95ci_pp"]],
        }
    return {
        "scope": str(report["scope"]),
        "source_population": int(report["source_population_size"]),
        "constructible_population": episodes,
        "construction_rejections": int(len(report["construction_rejections"])),
        "scene_clusters": int(report["scene_clusters"]),
        "arms": arms,
        "contrasts": contrasts,
        "graph_intervention_episodes": int(report["intervention_episodes"]),
        "actual_graph_plan_count": int(report["actual_graph_plan_count"]),
        "all_shared_prefixes_equal": True,
        "all_treatment_prefixes_equal": True,
    }


def sweep_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    kept = frame.loc[frame["local_rmse"].le(0.3)].copy()
    kept["success"] = kept["yaw_err_deg"].lt(30.0) & np.where(
        kept["t_dir_err_deg"].notna(),
        kept["t_dir_err_deg"].lt(30.0),
        kept["rel_pos_err_m"].lt(0.5),
    )
    pool = kept.loc[kept["kind"].isin(["grid", "rand"])].copy()
    pool["abs_yaw"] = pool["dyaw_deg"].abs()
    pool["covis_band"] = pd.cut(
        pool["covis_goal_in_anchor"],
        [-1e-9, 0.1, 0.2, 0.5, 1.000001],
        labels=["le_0p10", "0p10_to_0p20", "0p20_to_0p50", "gt_0p50"],
        include_lowest=True,
    )
    pool["yaw_band"] = pd.cut(
        pool["abs_yaw"],
        [-1e-9, 45.0, 95.0, 181.0],
        labels=["lt_45", "45_to_95", "gt_95"],
        include_lowest=True,
    )
    cells: dict[str, Any] = {}
    for (covis, yaw), group in pool.groupby(
        ["covis_band", "yaw_band"], observed=False
    ):
        if not len(group):
            continue
        cells[f"{covis}|{yaw}"] = {
            "n": int(len(group)),
            "success_rate": float(group["success"].mean()),
            "trajectories": int(group["traj"].nunique()),
        }
    return {
        "rows_raw": int(len(frame)),
        "rows_kept": int(len(kept)),
        "dropped_bad_local_fit": int(len(frame) - len(kept)),
        "scenes": int(kept["scene"].nunique()),
        "trajectories": int(kept["traj"].nunique()),
        "grid_rows": int(kept["kind"].eq("grid").sum()),
        "negative_rows": int(kept["kind"].eq("neg").sum()),
        "self_insertion_control_success": float(
            kept.loc[kept["kind"].eq("control"), "success"].mean()
        ),
        "cells": cells,
    }


def markdown(payload: dict[str, Any]) -> str:
    closed = payload["closed_loop_supported_revisit"]
    obs = payload["closed_loop_observability"]
    geo = payload["train40_inventory"]
    sweep = payload["controlled_viewpoint_sweep"]
    safety = payload["mixed_role_safety_smoke"]
    nnr = payload["actual_online_nnr"]
    direct = closed["certified_vs_direct"]
    lines = [
        "# Revisit benchmark coverage audit",
        "",
        "Date: 2026-08-14",
        "",
        "Scope: train/consumed artifacts only; no development or blind read; no tuning.",
        "",
        "## Headline",
        "",
        "The existing evidence is large for supported Revisit utility but narrow in "
        "open-set safety and history age. Re-running the same known-Revisit population "
        "is not the next statistical bottleneck.",
        "",
        "## Evidence already covered",
        "",
        "| axis | evidence | status |",
        "|---|---:|---|",
        f"| Supported Revisit closed loop | certified "
        f"{closed['certified_relocalization']['successes']}/"
        f"{closed['certified_relocalization']['eligible']} vs native "
        f"{closed['native']['successes']}/{closed['native']['eligible']} | strong |",
        f"| Certificate vs raw direct | +{direct['gains']}/-{direct['losses']}, "
        f"p={direct['mcnemar_exact_two_sided_p']:.3g} | directional, not significant |",
        f"| Actual-online support in closed-loop population | >=0.20: "
        f"{obs['supported_ge_0p20']}/{obs['eligible_after_a']}; >=0.50: "
        f"{obs['strong_ge_0p50']}/{obs['eligible_after_a']} | population is high-support |",
        f"| Controlled viewpoint mechanism | {sweep['rows_kept']} rows, "
        f"{sweep['trajectories']} trajectories, {sweep['scenes']} scenes | large N, weak scene breadth |",
        f"| Train-only candidate/existence inventory | {geo['sessions']} sessions, "
        f"{geo['candidates']} candidates, {geo['scenes']} scenes | broad offline support |",
        f"| Role-free Novel safety | {safety['accepts']}/{safety['novel_legs']} accepts; "
        f"one-sided 95% FPR upper {100*safety['zero_event_upper_95']:.1f}% | smoke only |",
        f"| Delayed actual-online Revisit | certified "
        f"{nnr['arms']['certified']['successes']}/"
        f"{nnr['arms']['certified']['episodes']} vs native "
        f"{nnr['arms']['native']['successes']}/"
        f"{nnr['arms']['native']['episodes']}; +"
        f"{nnr['contrasts']['certified_minus_native']['gains']}/-"
        f"{nnr['contrasts']['certified_minus_native']['losses']}, p="
        f"{nnr['contrasts']['certified_minus_native']['mcnemar_exact_two_sided_p']:.3g} "
        "| strong internal paired result |",
        "| Public/cross-dataset benchmark | none | missing |",
        "",
        "## Why fresh160 is not a hardness benchmark",
        "",
        f"Among the {obs['eligible_after_a']} episodes reaching A, all had actual-online "
        f"max co-visibility >=0.20 and {obs['strong_ge_0p50']} had >=0.50. Median max "
        f"co-visibility was {obs['online_max_covis']['p50']:.3f}. It is therefore a valid "
        "test of supported Revisit execution, not a test of the low-overlap boundary.",
        "",
        "## Existing train40 challenge inventory",
        "",
        "| support band (descriptive only) | sessions | scenes | any old-geometry pass |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "strict_or_low_le_0p10", "boundary_0p10_to_0p50", "strong_gt_0p50"
    ):
        item = geo["support_bands"][name]
        lines.append(
            f"| {name} | {item['sessions']} | {item['scenes']} | "
            f"{100*item['geometry_any_pass_rate']:.1f}% |"
        )
    lines += [
        "",
        "These 480 sessions are sufficient to build the next certificate stress test "
        "without generating new trajectories or reading blind data.",
        "",
        "## Architecture decision from actual-online NNR",
        "",
        f"The minimal certified arm reached {nnr['arms']['certified']['successes']}/"
        f"{nnr['arms']['certified']['episodes']}; certified+graph reached "
        f"{nnr['arms']['certified_graph']['successes']}/"
        f"{nnr['arms']['certified_graph']['episodes']}. The graph causal contrast was "
        f"+{nnr['contrasts']['certified_graph_minus_certified_budget']['gains']}/-"
        f"{nnr['contrasts']['certified_graph_minus_certified_budget']['losses']} despite "
        f"{nnr['actual_graph_plan_count']} emitted graph plans in "
        f"{nnr['graph_intervention_episodes']} episodes. Keep the minimal certificate "
        "adapter; graph rescue adds complexity without measured utility.",
        "",
        "## Frozen next experiment",
        "",
        "1. Run the frozen certificate on all 480 train40 sessions. Report actionability "
        "precision/coverage by selected support, session support, history age, and causal "
        "state; do not fit new thresholds.",
        "2. If the exhaustive challenge passes, freeze the minimal controller and run one "
        "scene-disjoint mixed-role confirmation. The policy never sees the role label.",
        "3. Only then use a public secondary benchmark (MemoNav-derived MP3D first).",
        "",
        "## Explicit non-actions",
        "",
        "- Do not rerun another same-scene known-Revisit fresh160.",
        "- Do not open blind16 for architecture or threshold selection.",
        "- Do not interpret co-visibility as the deployment classifier; it is only a "
        "difficulty coordinate.",
        "- Do not claim viewpoint generalization from the two-scene controlled sweep.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    inputs = {
        "geometry_evidence": args.geometry_evidence,
        "geometry_report": args.geometry_report,
        "closed_loop_report": args.closed_loop_report,
        "observability_audit": args.observability_audit,
        "mixed_safety_audit": args.mixed_safety_audit,
        "nnr_report": args.nnr_report,
        "sweep_results": args.sweep_results,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    geometry_report = load_json(args.geometry_report)
    closed_loop_report = load_json(args.closed_loop_report)
    observability_audit = load_json(args.observability_audit)
    mixed_safety = load_json(args.mixed_safety_audit)
    nnr_report = load_json(args.nnr_report)
    novel_legs = int(mixed_safety["novel_legs_audited"])
    accepts = int(mixed_safety["novel_certificate_accepts"])
    if accepts != 0:
        raise RuntimeError("zero-event safety bound is invalid after a Novel accept")
    payload = {
        "schema_version": "revisit_benchmark_coverage_audit_v2",
        "scope": "train_and_consumed_only_no_development_no_blind_no_tuning",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "train40_inventory": geometry_inventory(
            args.geometry_evidence, geometry_report
        ),
        "closed_loop_supported_revisit": closed_loop_summary(closed_loop_report),
        "closed_loop_observability": observability_summary(observability_audit),
        "actual_online_nnr": nnr_summary(nnr_report),
        "controlled_viewpoint_sweep": sweep_summary(args.sweep_results),
        "mixed_role_safety_smoke": {
            "novel_legs": novel_legs,
            "accepts": accepts,
            "takeovers": int(mixed_safety["novel_adapter_takeovers"]),
            "zero_event_upper_95": zero_event_upper(novel_legs),
        },
        "decision": {
            "rerun_same_known_revisit": False,
            "open_blind_now": False,
            "next_offline_test": "train40_exhaustive_480_certificate_challenge",
            "next_closed_loop_test": (
                "frozen_scene_disjoint_mixed_role_after_train40_challenge"),
            "selected_architecture": "minimal_role_free_certified_residual",
            "graph_rescue": "drop_no_measured_gain",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
