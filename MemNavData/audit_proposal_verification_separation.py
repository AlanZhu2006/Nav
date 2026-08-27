#!/usr/bin/env python3
"""Audit semantic proposal versus geometric selection in paper role-pair runs.

This is a read-only, post-hoc diagnostic.  It does not define a new method and
must not be used to tune the frozen certificate on a consumed evaluation set.
It reconstructs paired closed-loop outcomes and asks whether the current
geometry argmax changes the DINO top-1 anchor, how each choice relates to the
benchmark's audit-only maximum-covisibility frame, and where those changes
coincide with outcome discordances.

The plan logs contain PnP for the selected certificate proposal only.  A
candidate passing the Fundamental-MAGSAC precheck is therefore reported as
``precheck_eligible`` and never mislabeled as certificate accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = ("native", "raw_fixed_bearing", "geometry_fixed", "certified")
ROLES = ("novel", "revisit")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_metric(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_role = {row["analysis_role"]: row for row in rows}
    if set(by_role) != set(ROLES) or len(rows) != len(ROLES):
        raise ValueError(f"expected one metric row per role: {path}")
    return by_role


def _first_query_plan(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    plans = payload.get("query_leg")
    if not isinstance(plans, list) or not plans:
        raise ValueError(f"missing query_leg plans: {path}")
    if not isinstance(plans[0], dict):
        raise ValueError(f"first query plan is not an object: {path}")
    return plans[0]


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _success(row: Mapping[str, str]) -> bool:
    return math.isclose(float(row["reached"]), 1.0, abs_tol=1e-12)


def _benchmark_queries(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _read_json(path)
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise ValueError(f"expected exactly one role pair: {path}")
    queries = pairs[0].get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"missing benchmark queries: {path}")
    result = {query["analysis_role"]: query for query in queries}
    if set(result) != set(ROLES):
        raise ValueError(f"benchmark roles changed: {path}")
    return result


def _candidate_by_anchor(
    plan: Mapping[str, Any], anchor: int | None
) -> Mapping[str, Any] | None:
    if anchor is None:
        return None
    trials = plan.get("router_candidate_trials")
    if not isinstance(trials, list):
        return None
    found = [trial for trial in trials
             if isinstance(trial, dict)
             and _as_int(trial.get("anchor")) == anchor]
    if len(found) > 1:
        raise ValueError(f"duplicate candidate trial for anchor {anchor}")
    return found[0] if found else None


def _precheck_eligible(candidate: Mapping[str, Any] | None) -> bool | None:
    if candidate is None:
        return None
    inliers = _as_int(candidate.get("fundamental_inliers"))
    query = _as_float(candidate.get("fundamental_query_hull_coverage"))
    reference = _as_float(
        candidate.get("fundamental_reference_hull_coverage"))
    if inliers is None or query is None or reference is None:
        return None
    return inliers >= 16 and query >= 0.05 and reference >= 0.05


def _candidate_fields(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    keys = (
        "dino_rank",
        "dino_cosine",
        "fundamental_inliers",
        "fundamental_inlier_ratio",
        "fundamental_query_hull_coverage",
        "fundamental_reference_hull_coverage",
        "lightglue_score_median",
    )
    return {key: candidate.get(key) for key in keys}


def _distance(anchor: int | None, reference: int | None) -> int | None:
    if anchor is None or reference is None:
        return None
    return abs(anchor - reference)


def _first_list_anchor(value: Any) -> int | None:
    if not isinstance(value, list) or not value:
        return None
    return _as_int(value[0])


def _identity_from_directory(path: Path) -> tuple[str, str]:
    name = path.name
    marker = "_episode_"
    if marker not in name:
        raise ValueError(f"cannot parse evaluation directory: {name}")
    prefix, suffix = name.split(marker, 1)
    scene = prefix.split("_", 1)[1]
    return scene, f"episode_{suffix}"


def collect(run_root: Path, protocol: str) -> list[dict[str, Any]]:
    evaluation_root = run_root / "evaluation" / protocol
    benchmark_root = run_root / "benchmarks" / protocol
    directories = sorted(
        path for path in evaluation_root.iterdir()
        if path.is_dir() and "_episode_" in path.name)
    if not directories:
        raise ValueError(f"no evaluation directories: {evaluation_root}")

    records: list[dict[str, Any]] = []
    for directory in directories:
        scene, episode = _identity_from_directory(directory)
        benchmark_path = benchmark_root / scene / episode / "role_pairs.json"
        queries = _benchmark_queries(benchmark_path)
        metrics = {
            arm: _read_metric(directory / arm / "metric.csv")
            for arm in ARMS
        }
        for role in ROLES:
            query = queries[role]
            plan_paths = {
                arm: directory / arm / f"{episode}_pair_00_{role}_plans.json"
                for arm in ARMS if arm != "native"
            }
            plans = {arm: _first_query_plan(path)
                     for arm, path in plan_paths.items()}
            raw_plan = plans["raw_fixed_bearing"]
            geometry_plan = plans["geometry_fixed"]
            certified_plan = plans["certified"]

            raw_anchor = _as_int(raw_plan.get("retrieved_anchor"))
            dino_anchor = _first_list_anchor(
                certified_plan.get("router_candidate_order_dino"))
            if dino_anchor is None:
                trials = certified_plan.get("router_candidate_trials")
                if isinstance(trials, list):
                    rank_one = [trial for trial in trials
                                if isinstance(trial, dict)
                                and _as_int(trial.get("dino_rank")) == 1]
                    if len(rank_one) == 1:
                        dino_anchor = _as_int(rank_one[0].get("anchor"))
            geometry_anchor = _as_int(
                geometry_plan.get("router_selected_anchor"))
            certified_anchor = _as_int(
                certified_plan.get("router_selected_anchor"))
            max_covis_frame = _as_int(query.get("max_online_a_covis_frame"))
            dino_trial = _candidate_by_anchor(certified_plan, dino_anchor)
            certified_trial = _candidate_by_anchor(
                certified_plan, certified_anchor)
            pnp = certified_plan.get("certified_relocalization_pnp")
            if not isinstance(pnp, dict):
                pnp = {}

            outcomes = {arm: _success(metrics[arm][role]) for arm in ARMS}
            records.append({
                "scene": scene,
                "episode": episode,
                "role": role,
                "max_online_a_covis": _as_float(
                    query.get("max_online_a_covis")),
                "max_online_a_covis_frame": max_covis_frame,
                "geodesic_from_a_end_m": _as_float(
                    query.get("geodesic_from_a_end_m")),
                "outcomes": outcomes,
                "termination": {
                    arm: metrics[arm][role].get("termination_reason")
                    for arm in ARMS
                },
                "final_goal_dist_m": {
                    arm: _as_float(metrics[arm][role].get("final_goal_dist_m"))
                    for arm in ARMS
                },
                "dino_anchor": dino_anchor,
                "raw_arm_anchor": raw_anchor,
                "raw_arm_matches_shortlist_dino_top1": (
                    raw_anchor is not None and dino_anchor is not None
                    and raw_anchor == dino_anchor),
                "geometry_anchor": geometry_anchor,
                "certified_anchor": certified_anchor,
                "certified_selected_dino_rank": _as_int(
                    certified_plan.get(
                        "router_selected_candidate_dino_rank")),
                "dino_to_max_covis_frame_gap": _distance(
                    dino_anchor, max_covis_frame),
                "certified_to_max_covis_frame_gap": _distance(
                    certified_anchor, max_covis_frame),
                "geometry_changed_dino_top1": (
                    certified_anchor is not None
                    and dino_anchor is not None
                    and certified_anchor != dino_anchor),
                "dino_top1_precheck_eligible": _precheck_eligible(dino_trial),
                "dino_top1_evidence": _candidate_fields(dino_trial),
                "certified_selected_evidence": _candidate_fields(
                    certified_trial),
                "selected_certificate_accepted": (
                    certified_plan.get(
                        "certified_relocalization_accepted") is True),
                "selected_pnp": {
                    key: pnp.get(key) for key in (
                        "status", "inliers", "inlier_ratio",
                        "query_inlier_coverage",
                        "reference_inlier_coverage",
                        "reprojection_rmse_px",
                    )
                },
            })
    return records


def _paired(
    records: Iterable[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    records = list(records)
    gains = []
    losses = []
    left_successes = 0
    right_successes = 0
    for record in records:
        outcomes = record["outcomes"]
        left_value = bool(outcomes[left])
        right_value = bool(outcomes[right])
        left_successes += int(left_value)
        right_successes += int(right_value)
        identity = [record["scene"], record["episode"], record["role"]]
        if right_value and not left_value:
            gains.append(identity)
        elif left_value and not right_value:
            losses.append(identity)
    return {
        "left": left,
        "right": right,
        "n": len(records),
        "left_successes": left_successes,
        "right_successes": right_successes,
        "right_gains": gains,
        "right_losses": losses,
        "paired_gain_loss": [len(gains), len(losses)],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {role: [record for record in records
                      if record["role"] == role]
               for role in ROLES}
    revisit = by_role["revisit"]
    novel = by_role["novel"]
    changed = [record for record in revisit
               if record["geometry_changed_dino_top1"]]
    closer = Counter()
    for record in changed:
        dino_gap = record["dino_to_max_covis_frame_gap"]
        certified_gap = record["certified_to_max_covis_frame_gap"]
        if dino_gap is None or certified_gap is None:
            closer["unknown"] += 1
        elif dino_gap < certified_gap:
            closer["dino_closer"] += 1
        elif certified_gap < dino_gap:
            closer["geometry_closer"] += 1
        else:
            closer["tie"] += 1

    cert_raw_discordant = [record for record in records
                           if record["outcomes"]["certified"]
                           != record["outcomes"]["raw_fixed_bearing"]]
    novel_covis = [record["max_online_a_covis"] for record in novel
                   if record["max_online_a_covis"] is not None]
    revisit_covis = [record["max_online_a_covis"] for record in revisit
                     if record["max_online_a_covis"] is not None]
    return {
        "schema_version": "proposal_verification_separation_audit_v1",
        "scope": "consumed_posthoc_diagnostic_not_method_selection",
        "records": len(records),
        "scenes": len({record["scene"] for record in records}),
        "roles": {role: len(rows) for role, rows in by_role.items()},
        "paired": {
            role: {
                "certified_minus_raw_fixed": _paired(
                    rows, "raw_fixed_bearing", "certified"),
                "raw_fixed_minus_native": _paired(
                    rows, "native", "raw_fixed_bearing"),
            }
            for role, rows in {"all": records, **by_role}.items()
        },
        "candidate_selection": {
            "revisit_geometry_changed_dino_top1": len(changed),
            "revisit_geometry_changed_fraction": (
                len(changed) / len(revisit) if revisit else None),
            "changed_anchor_distance_to_max_covis": dict(closer),
            "revisit_dino_top1_precheck_eligible": sum(
                record["dino_top1_precheck_eligible"] is True
                for record in revisit),
            "revisit_dino_top1_precheck_unknown": sum(
                record["dino_top1_precheck_eligible"] is None
                for record in revisit),
        },
        "support": {
            "novel_max_covis_min_max": (
                [min(novel_covis), max(novel_covis)]
                if novel_covis else None),
            "revisit_max_covis_min_max": (
                [min(revisit_covis), max(revisit_covis)]
                if revisit_covis else None),
        },
        "certified_raw_discordant_records": cert_raw_discordant,
        "limitations": [
            "unselected candidates have Fundamental precheck evidence but no PnP result",
            "precheck_eligible does not mean certificate accepted",
            "audit-only co-visibility and role labels are not deployment inputs",
            "consumed outcomes cannot authorize threshold or method selection",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", default="natural_direction")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact", action="store_true",
        help="replace full discordant records with their identities")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(collect(args.run_root.resolve(), args.protocol))
    if args.compact:
        report["certified_raw_discordant_records"] = [
            {
                key: record[key] for key in (
                    "scene", "episode", "role", "raw_arm_anchor",
                    "dino_anchor", "certified_anchor",
                    "dino_top1_precheck_eligible", "outcomes")
            }
            for record in report["certified_raw_discordant_records"]
        ]
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
