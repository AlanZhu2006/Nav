#!/usr/bin/env python3
"""Compare learned Pi3X proof and the incumbent certificate on one endpoint.

The historical certificate report defines actionability by metric position
error.  The deployed compass, however, consumes only a scale-free bearing.
This reporting-only verifier therefore evaluates both frozen methods with the
same navigation-direction target: an accepted positive session is correct when
its emitted bearing is within 30 degrees.  Strict no-match sessions remain
reject targets.  No threshold or model is selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(value: object) -> bool:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _center_pnp(payload: str) -> dict[str, Any]:
    hypotheses = json.loads(payload)
    centers = [item for item in hypotheses if int(item["offset"]) == 0]
    if len(centers) != 1 or not isinstance(
        centers[0].get("pnp_lightglue"), dict
    ):
        raise ValueError("certificate row lacks one center LightGlue-PnP result")
    return centers[0]["pnp_lightglue"]


def _certificate_accepts(pnp: dict[str, Any]) -> bool:
    fields = (
        "inliers",
        "query_inlier_coverage",
        "reference_inlier_coverage",
        "reprojection_rmse_px",
    )
    return bool(
        pnp.get("status") == "ok"
        and all(_finite(pnp.get(name)) for name in fields)
        and int(pnp["inliers"]) >= 16
        and float(pnp["query_inlier_coverage"]) >= 0.05
        and float(pnp["reference_inlier_coverage"]) >= 0.05
        and float(pnp["reprojection_rmse_px"]) <= 2.0
    )


def _mcnemar_exact(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(gains, losses) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _paired(rows: Iterable[dict[str, Any]], learned_key: str,
            certificate_key: str) -> dict[str, Any]:
    materialized = list(rows)
    gains = sum(
        bool(row[learned_key]) and not bool(row[certificate_key])
        for row in materialized
    )
    losses = sum(
        bool(row[certificate_key]) and not bool(row[learned_key])
        for row in materialized
    )
    return {
        "learned_gain_certificate_loss": gains,
        "learned_loss_certificate_gain": losses,
        "exact_mcnemar_p": _mcnemar_exact(gains, losses),
    }


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if row["strict_negative"]]
    known = positives + negatives
    accepted = [row for row in known if row[f"{prefix}_accepted"]]
    correct = [row for row in positives if row[f"{prefix}_correct_accept"]]
    wrong_positive = [
        row for row in positives
        if row[f"{prefix}_accepted"] and not row[f"{prefix}_direction_correct"]
    ]
    false_accepts = [row for row in negatives if row[f"{prefix}_accepted"]]
    catastrophes = [
        row for row in accepted
        if _finite(row[f"{prefix}_bearing_error_deg"])
        and float(row[f"{prefix}_bearing_error_deg"]) > 90.0
    ]
    return {
        "positive_sessions": len(positives),
        "strict_negative_sessions": len(negatives),
        "accepted_known_sessions": len(accepted),
        "correct_positive_accepts": len(correct),
        "wrong_positive_accepts": len(wrong_positive),
        "strict_negative_false_accepts": len(false_accepts),
        "precision": len(correct) / len(accepted) if accepted else 1.0,
        "positive_recall": len(correct) / len(positives),
        "strict_negative_fpr": len(false_accepts) / len(negatives),
        "accepted_bearing_catastrophic_gt90deg": len(catastrophes),
    }


def compare(certificate_csv: Path, learned_predictions_csv: Path) -> dict[str, Any]:
    with certificate_csv.open(newline="") as handle:
        certificate_rows = list(csv.DictReader(handle))
    with learned_predictions_csv.open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    selected = [row for row in prediction_rows if _boolean(row["selected"])]
    selected_by_session = {row["session_id"]: row for row in selected}
    certificate_by_session = {row["session_id"]: row for row in certificate_rows}
    if len(selected_by_session) != len(selected):
        raise ValueError("learned predictions contain duplicate selected sessions")
    if len(certificate_by_session) != len(certificate_rows):
        raise ValueError("certificate CSV contains duplicate sessions")
    if set(selected_by_session) != set(certificate_by_session):
        raise ValueError("learned and certificate session universes differ")

    records = []
    for session_id in sorted(certificate_by_session):
        certificate = certificate_by_session[session_id]
        learned = selected_by_session[session_id]
        if certificate["scene"] != learned["scene"]:
            raise ValueError(f"scene differs for {session_id}")
        positive = _boolean(certificate["session_has_positive"])
        strict_negative = _boolean(certificate["session_is_strict_no_match"])
        expected_label = 1 if positive else 0 if strict_negative else -1
        if int(learned["session_label_reporting_only"]) != expected_label:
            raise ValueError(f"reporting label differs for {session_id}")
        pnp = _center_pnp(certificate["hypotheses_json"])
        certificate_error = pnp.get("relative_position_direction_error_deg")
        certificate_direction_correct = bool(
            _finite(certificate_error) and float(certificate_error) <= 30.0
        )
        certificate_accepted = _certificate_accepts(pnp)
        learned_error = learned["bearing_error_deg_reporting_only"]
        learned_direction_correct = bool(
            _finite(learned_error) and float(learned_error) <= 30.0
        )
        learned_navigation_label = int(
            learned["navigation_action_label_reporting_only"]
        )
        if positive and learned_navigation_label != int(learned_direction_correct):
            raise ValueError(f"learned bearing label differs for {session_id}")
        if strict_negative and learned_navigation_label != 0:
            raise ValueError(f"strict negative became a positive target: {session_id}")
        learned_accepted = _boolean(learned["accepted"])
        records.append({
            "session_id": session_id,
            "scene": certificate["scene"],
            "positive": positive,
            "strict_negative": strict_negative,
            "certificate_accepted": certificate_accepted,
            "certificate_direction_correct": certificate_direction_correct,
            "certificate_correct_accept": bool(
                positive and certificate_accepted and certificate_direction_correct
            ),
            "certificate_bearing_error_deg": certificate_error,
            "learned_accepted": learned_accepted,
            "learned_direction_correct": learned_direction_correct,
            "learned_correct_accept": bool(
                positive and learned_accepted and learned_direction_correct
            ),
            "learned_bearing_error_deg": learned_error,
        })

    positives = [row for row in records if row["positive"]]
    negatives = [row for row in records if row["strict_negative"]]
    return {
        "schema_version": 1,
        "status": "reporting_only_endpoint_aligned_train40_not_closed_loop",
        "endpoint": {
            "positive_correct": "accepted bearing error <= 30 degrees",
            "strict_negative_correct": "reject",
            "note": (
                "This differs from the historical 0.75 m metric-actionability "
                "label and does not authorize deployment."
            ),
        },
        "inputs": {
            "certificate_csv_sha256": _sha256(certificate_csv),
            "learned_predictions_csv_sha256": _sha256(learned_predictions_csv),
        },
        "population": {
            "sessions": len(records),
            "scenes": len({row["scene"] for row in records}),
            "positive_sessions": len(positives),
            "strict_negative_sessions": len(negatives),
            "ambiguous_sessions_excluded": len(records) - len(positives) - len(negatives),
        },
        "certificate": _metrics(records, "certificate"),
        "learned_spatial_proof": _metrics(records, "learned"),
        "paired_positive_correct_coverage": _paired(
            positives, "learned_correct_accept", "certificate_correct_accept"
        ),
        "paired_strict_negative_false_accept": _paired(
            negatives, "learned_accepted", "certificate_accepted"
        ),
        "changed_positive_scene_clusters": {
            "learned_gains": len({
                row["scene"] for row in positives
                if row["learned_correct_accept"]
                and not row["certificate_correct_accept"]
            }),
            "learned_losses": len({
                row["scene"] for row in positives
                if row["certificate_correct_accept"]
                and not row["learned_correct_accept"]
            }),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate-csv", type=Path, required=True)
    parser.add_argument("--learned-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.certificate_csv, args.learned_predictions_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
