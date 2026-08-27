#!/usr/bin/env python3
"""Paired audit of geometry and scene-OOF learned certificate proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

try:
    from MemNavData.summarize_lingbot_lightglue_localization import (
        center_hypothesis,
        has_certificate,
        is_actionable,
        summarize,
    )
except ModuleNotFoundError:  # direct invocation
    from summarize_lingbot_lightglue_localization import (  # type: ignore
        center_hypothesis,
        has_certificate,
        is_actionable,
        summarize,
    )


SCHEMA_VERSION = "cdec_dual_proposal_certificate_audit_v2_20260813"
GEOMETRY_ORIGIN = "lightglue_fundamental_rank_v1"
CDEC_ORIGIN = "cdec_scene_oof_pairwise_rank_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def policy_records(rows: pd.DataFrame) -> dict[str, dict]:
    if len(rows) != 480 or rows["session_id"].duplicated().any():
        raise RuntimeError("each policy must contain 480 unique sessions")
    result = {}
    for row in rows.itertuples(index=False):
        center = center_hypothesis(str(row.hypotheses_json))
        pnp = center.get("pnp_lightglue")
        if not isinstance(pnp, Mapping):
            raise RuntimeError(f"missing PnP payload: {row.session_id}")
        certificate = has_certificate(pnp)
        actionable = is_actionable(pnp)
        result[str(row.session_id)] = {
            "session_id": str(row.session_id),
            "scene": str(row.scene),
            "candidate_frame": int(row.candidate_frame),
            "candidate_path": str(row.candidate_path),
            "teacher_candidate_label": int(row.label),
            "session_has_positive": bool(row.session_has_positive),
            "session_is_strict_no_match": bool(row.session_is_strict_no_match),
            "certificate": certificate,
            "actionable": actionable,
            "certified_actionable": certificate and actionable,
            "certificate_false_positive": certificate and not actionable,
            "pnp_status": str(pnp.get("status")),
            "pnp_inliers": int(pnp.get("inliers", 0)),
        }
    return result


def summarize_policy(records: dict[str, dict]) -> dict:
    rows = list(records.values())
    return {
        "sessions": len(rows),
        "teacher_positive_top1": sum(
            row["teacher_candidate_label"] == 1 for row in rows),
        "certificate_accepted": sum(row["certificate"] for row in rows),
        "ground_truth_actionable": sum(row["actionable"] for row in rows),
        "certified_actionable": sum(row["certified_actionable"] for row in rows),
        "certificate_false_positive": sum(
            row["certificate_false_positive"] for row in rows),
        "accepted_scenes": len({
            row["scene"] for row in rows if row["certificate"]}),
    }


def cascade(
    primary: dict[str, dict], fallback: dict[str, dict]
) -> tuple[dict, dict[str, dict]]:
    if set(primary) != set(fallback):
        raise RuntimeError("proposal session universes differ")
    selected = {}
    for session in sorted(primary):
        first = primary[session]
        second = fallback[session]
        chosen = first if first["certificate"] else second
        selected[session] = {
            **chosen,
            "selected_source": "primary" if first["certificate"] else "fallback",
            "second_certificate_invoked": not first["certificate"],
        }
    summary = summarize_policy(selected)
    summary.update({
        "second_certificate_invocations": sum(
            row["second_certificate_invoked"] for row in selected.values()),
        "primary_accepts": sum(row["certificate"] for row in primary.values()),
        "fallback_rescues": sum(
            (not primary[key]["certificate"]) and fallback[key]["certificate"]
            for key in primary),
    })
    return summary, selected


def paired(first: dict[str, dict], second: dict[str, dict], field: str) -> dict:
    if set(first) != set(second):
        raise RuntimeError("paired session universes differ")
    gains = sum(first[key][field] and not second[key][field] for key in first)
    losses = sum(second[key][field] and not first[key][field] for key in first)
    return {
        "gains": int(gains),
        "losses": int(losses),
        "exact_mcnemar_p": exact_mcnemar(int(gains), int(losses)),
    }


def same_anchor_repeatability(
    first: dict[str, dict], second: dict[str, dict]
) -> dict:
    """Audit the natural within-process repeats where both proposals agree."""
    if set(first) != set(second):
        raise RuntimeError("proposal session universes differ")
    decision_fields = (
        "certificate", "actionable", "certified_actionable",
        "certificate_false_positive", "pnp_status", "pnp_inliers",
    )
    same_anchor = [
        key for key in first
        if first[key]["candidate_frame"] == second[key]["candidate_frame"]
    ]
    mismatches = [
        key for key in same_anchor
        if first[key]["candidate_path"] != second[key]["candidate_path"]
        or any(first[key][field] != second[key][field]
               for field in decision_fields)
    ]
    return {
        "same_anchor_sessions": len(same_anchor),
        "decision_equal": len(same_anchor) - len(mismatches),
        "decision_mismatches": len(mismatches),
        "mismatch_session_ids": sorted(mismatches),
    }


def audit(args: argparse.Namespace) -> dict:
    dual_mode = args.dual_rows is not None or args.dual_report is not None
    separate_mode = any(value is not None for value in (
        args.geometry_rows, args.geometry_report,
        args.cdec_rows, args.cdec_report))
    if dual_mode == separate_mode:
        raise ValueError(
            "provide exactly one dual collection or two separate collections")
    if dual_mode:
        if args.dual_rows is None or args.dual_report is None:
            raise ValueError("dual rows and report must be provided together")
        paths = (args.dual_rows, args.dual_report)
    else:
        paths = (
            args.geometry_rows, args.geometry_report,
            args.cdec_rows, args.cdec_report,
        )
    for path in paths:
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
    if dual_mode:
        assert args.dual_rows is not None and args.dual_report is not None
        dual_rows = pd.read_csv(args.dual_rows)
        dual_report = json.loads(args.dual_report.read_text())
        if dual_report.get("config", {}).get("selection_mode") != (
                "cdec_geometry_dual_ranked"):
            raise RuntimeError("dual collector selection mode changed")
        if "candidate_selection_origin" not in dual_rows:
            raise RuntimeError("dual rows lack proposal origin")
        geometry_rows = dual_rows.loc[
            dual_rows["candidate_selection_origin"].eq(GEOMETRY_ORIGIN)
        ].copy()
        cdec_rows = dual_rows.loc[
            dual_rows["candidate_selection_origin"].eq(CDEC_ORIGIN)
        ].copy()
        if (len(geometry_rows) != 480 or len(cdec_rows) != 480
                or len(dual_rows) != 960):
            raise RuntimeError("dual collector must contain 480 rows per proposal")
        geometry_report = dual_report
        cdec_report = dual_report
        input_payload = {
            "dual_rows": str(args.dual_rows.resolve()),
            "dual_rows_sha256": sha256(args.dual_rows),
            "dual_report": str(args.dual_report.resolve()),
            "dual_report_sha256": sha256(args.dual_report),
        }
    else:
        assert all(value is not None for value in (
            args.geometry_rows, args.geometry_report,
            args.cdec_rows, args.cdec_report))
        geometry_rows = pd.read_csv(args.geometry_rows)
        cdec_rows = pd.read_csv(args.cdec_rows)
        geometry_report = json.loads(args.geometry_report.read_text())
        cdec_report = json.loads(args.cdec_report.read_text())
        if geometry_report.get("config", {}).get("selection_mode") != (
                "lightglue_ranked"):
            raise RuntimeError("geometry collector selection mode changed")
        if cdec_report.get("config", {}).get("selection_mode") != (
                "cdec_oof_ranked"):
            raise RuntimeError("CDEC collector selection mode changed")
        input_payload = {
            "geometry_rows": str(args.geometry_rows.resolve()),
            "geometry_rows_sha256": sha256(args.geometry_rows),
            "geometry_report": str(args.geometry_report.resolve()),
            "geometry_report_sha256": sha256(args.geometry_report),
            "cdec_rows": str(args.cdec_rows.resolve()),
            "cdec_rows_sha256": sha256(args.cdec_rows),
            "cdec_report": str(args.cdec_report.resolve()),
            "cdec_report_sha256": sha256(args.cdec_report),
        }
    if (geometry_report.get("config", {}).get("neighbor_offsets") != [0]
            or cdec_report.get("config", {}).get("neighbor_offsets") != [0]
            or geometry_report.get("config", {}).get("full_replay") is not True
            or cdec_report.get("config", {}).get("full_replay") is not True):
        raise RuntimeError("collector differs from one-view full-replay contract")

    expected = set(map(str, geometry_rows["session_id"]))
    if expected != set(map(str, cdec_rows["session_id"])) or len(expected) != 480:
        raise RuntimeError("paired 480-session universe changed")
    geometry = policy_records(geometry_rows)
    cdec = policy_records(cdec_rows)
    for session in geometry:
        for field in (
            "scene", "session_has_positive", "session_is_strict_no_match",
        ):
            if geometry[session][field] != cdec[session][field]:
                raise RuntimeError(
                    f"paired session metadata differs ({field}): {session}")
    repeatability = same_anchor_repeatability(geometry, cdec)
    cdec_first_summary, cdec_first = cascade(cdec, geometry)
    geometry_first_summary, geometry_first = cascade(geometry, cdec)
    geometry_audit = summarize(geometry_rows, geometry_report, expected)
    cdec_audit = summarize(cdec_rows, cdec_report, expected)

    cdec_vs_geometry = paired(cdec, geometry, "certified_actionable")
    cdec_first_vs_geometry = paired(
        cdec_first, geometry, "certified_actionable")
    geometry_first_vs_geometry = paired(
        geometry_first, geometry, "certified_actionable")
    cdec_first_no_extra_false_positive = (
        cdec_first_summary["certificate_false_positive"]
        <= summarize_policy(geometry)["certificate_false_positive"]
    )
    geometry_first_no_extra_false_positive = (
        geometry_first_summary["certificate_false_positive"]
        <= summarize_policy(geometry)["certificate_false_positive"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "train_only_oof_proposal_certificate_audit_not_closed_loop",
        "question": (
            "Can a task-trained scene-OOF proposal expose complementary "
            "anchors that the unchanged atomic LingBot-depth PnP certificate "
            "can safely authorize?"
        ),
        "scope": {
            "train_scenes_only": True,
            "row_scene_held_out_for_cdec": True,
            "development_or_blind_read": False,
            "one_view_full_replay": True,
            "activation_learned": False,
            "same_gpu_same_lingbot_process": dual_mode,
        },
        "inputs": input_payload,
        "policies": {
            "geometry": summarize_policy(geometry),
            "cdec_oof": summarize_policy(cdec),
            "cdec_first_then_geometry_on_reject": cdec_first_summary,
            "geometry_first_then_cdec_on_reject": geometry_first_summary,
        },
        "paired": {
            "cdec_oof_minus_geometry_certified_actionable": cdec_vs_geometry,
            "cdec_first_cascade_minus_geometry_certified_actionable": (
                cdec_first_vs_geometry),
            "geometry_first_cascade_minus_geometry_certified_actionable": (
                geometry_first_vs_geometry),
        },
        "proposal_identity": {
            "same_selected_anchor": sum(
                geometry[key]["candidate_frame"] == cdec[key]["candidate_frame"]
                for key in geometry),
            "both_certificates_accept": sum(
                geometry[key]["certificate"] and cdec[key]["certificate"]
                for key in geometry),
            "geometry_only_accepts": sum(
                geometry[key]["certificate"] and not cdec[key]["certificate"]
                for key in geometry),
            "cdec_only_accepts": sum(
                cdec[key]["certificate"] and not geometry[key]["certificate"]
                for key in geometry),
            "same_anchor_repeatability": repeatability,
        },
        "method_gate": {
            "pass": bool(
                geometry_first_vs_geometry["gains"] > 0
                and geometry_first_vs_geometry["losses"] == 0
                and geometry_first_no_extra_false_positive
                and repeatability["decision_mismatches"] == 0),
            "deployment_order": "geometry_first_then_cdec_on_reject",
            "reason": (
                "The learned top-1 did not significantly replace geometry; "
                "therefore it may only supply a reject-only fallback and may "
                "not override a geometry proposal whose PnP certificate passed."
            ),
            "requirements": {
                "at_least_one_certified_actionable_rescue": (
                    geometry_first_vs_geometry["gains"] > 0),
                "cannot_lose_geometry_certified_actionable": (
                    geometry_first_vs_geometry["losses"] == 0),
                "no_extra_certificate_false_positive": (
                    geometry_first_no_extra_false_positive),
                "same_anchor_certificate_decisions_repeatable": (
                    repeatability["decision_mismatches"] == 0),
            },
            "authority_if_passed": (
                "authorizes consumed-pool closed-loop comparison only; not "
                "held-out or paper-final confirmation"),
        },
        "diagnostic_cdec_first_gate_not_authoritative": {
            "paired": cdec_first_vs_geometry,
            "no_extra_certificate_false_positive": (
                cdec_first_no_extra_false_positive),
        },
        "component_audits": {
            "geometry": geometry_audit,
            "cdec_oof": cdec_audit,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-rows", type=Path)
    parser.add_argument("--geometry-report", type=Path)
    parser.add_argument("--cdec-rows", type=Path)
    parser.add_argument("--cdec-report", type=Path)
    parser.add_argument("--dual-rows", type=Path)
    parser.add_argument("--dual-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args)
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
