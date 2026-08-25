#!/usr/bin/env python3
"""Independently derive the frozen Final14 support-spectrum result.

This script performs no policy inference and does not select an operating
point.  It binds the already independently verified Final14 summary to the two
sealed benchmark manifests and checks that unsupported, hard-support, and
standard-support queries share the same causal online-A history population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "final14_support_spectrum_reanalysis_v1_20260823"
ARMS = ("native", "raw_fixed_bearing", "certified")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def manifest_rows(manifest: dict[str, Any], role: str) -> tuple[list[tuple[str, str]], list[float]]:
    identities: list[tuple[str, str]] = []
    covisibility: list[float] = []
    for episode in manifest.get("episodes", []):
        identity = (str(episode["scene"]), str(episode["episode"]))
        identities.append(identity)
        selected = []
        for pair in episode.get("pairs", []):
            selected.extend(
                query for query in pair.get("queries", [])
                if query.get("analysis_role") == role
            )
        require(len(selected) == 1, f"{identity}: expected one {role} query")
        value = float(selected[0]["max_online_a_covis"])
        require(math.isfinite(value), f"{identity}: non-finite covisibility")
        covisibility.append(value)
    require(len(identities) == len(set(identities)), "duplicate history identity")
    return identities, covisibility


def distribution(values: list[float]) -> dict[str, float | int]:
    require(bool(values), "empty support distribution")
    return {
        "n": len(values),
        "minimum": min(values),
        "median": median(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def metric_row(protocol: dict[str, Any], role: str) -> dict[str, Any]:
    metrics = protocol["metrics"]
    for arm in ARMS:
        require(int(metrics[arm][role]["n"]) == 21,
                f"{arm}/{role}: unexpected denominator")
    cec_native = protocol["contrasts"]["certified_minus_native"][role]
    cec_raw = protocol["contrasts"][
        "certified_minus_raw_fixed_bearing"
    ][role]
    return {
        "metrics": {
            arm: {
                "successes": int(metrics[arm][role]["successes"]),
                "n": int(metrics[arm][role]["n"]),
                "sr": float(metrics[arm][role]["SR"]),
                "spl": float(metrics[arm][role]["SPL"]),
            }
            for arm in ARMS
        },
        "cec_minus_native": {
            "gains": int(cec_native["gains"]),
            "losses": int(cec_native["losses"]),
            "exact_mcnemar_two_sided_p": float(
                cec_native["exact_mcnemar_two_sided_p"]
            ),
        },
        "cec_minus_raw_fixed": {
            "gains": int(cec_raw["gains"]),
            "losses": int(cec_raw["losses"]),
            "exact_mcnemar_two_sided_p": float(
                cec_raw["exact_mcnemar_two_sided_p"]
            ),
        },
    }


def derive(root: Path) -> dict[str, Any]:
    summary_path = root / "paper_role_pair_summary.json"
    verifier_path = root / "paper_role_pair_independent_verification.json"
    natural_path = root / "benchmarks/natural_direction/manifest.json"
    hard_path = root / "benchmarks/hard_support/manifest.json"
    summary = read_json(summary_path)
    verifier = read_json(verifier_path)
    natural = read_json(natural_path)
    hard = read_json(hard_path)

    require(verifier.get("verified") is True, "Final14 verifier did not pass")
    require(verifier.get("summary_sha256") == sha256(summary_path),
            "verifier is not bound to the supplied summary")
    require(summary.get("all_required_outputs_complete") is True,
            "Final14 outputs are incomplete")

    unsupported_ids, unsupported_covis = manifest_rows(natural, "novel")
    strong_ids, strong_covis = manifest_rows(natural, "revisit")
    weak_ids, weak_covis = manifest_rows(hard, "revisit")
    require(unsupported_ids == weak_ids == strong_ids,
            "support bands do not share an identical ordered population")
    require(len(unsupported_ids) == 21, "support-spectrum population changed")
    require(max(unsupported_covis) < 0.10, "unsupported band contract changed")
    require(min(weak_covis) >= 0.25 and max(weak_covis) < 0.55,
            "hard-support band contract changed")
    require(min(strong_covis) >= 0.55 and max(strong_covis) <= 0.90,
            "standard-support band contract changed")

    protocols = summary["protocols"]
    natural_result = protocols["natural_direction"]
    hard_result = protocols["hard_support"]
    require(int(natural_result["certificate_safety"]["runtime_failure_plans"]) == 0,
            "natural protocol contains runtime failures")
    require(int(hard_result["certificate_safety"]["runtime_failure_plans"]) == 0,
            "hard protocol contains runtime failures")

    bands = [
        {
            "name": "unsupported",
            "construction_protocol": "natural_direction",
            "analysis_role": "novel",
            "support_contract": "max_online_a_covis < 0.10",
            "covisibility": distribution(unsupported_covis),
            "cec_authorized_queries": int(
                natural_result["certificate_safety"]["novel_takeover_episodes"]
            ),
            **metric_row(natural_result, "novel"),
        },
        {
            "name": "weak",
            "construction_protocol": "hard_support",
            "analysis_role": "revisit",
            "support_contract": "0.25 <= max_online_a_covis < 0.55",
            "covisibility": distribution(weak_covis),
            "cec_authorized_queries": int(
                hard_result["certificate_safety"]["revisit_activation_queries"]
            ),
            **metric_row(hard_result, "revisit"),
        },
        {
            "name": "strong",
            "construction_protocol": "natural_direction",
            "analysis_role": "revisit",
            "support_contract": "0.55 <= max_online_a_covis <= 0.90",
            "covisibility": distribution(strong_covis),
            "cec_authorized_queries": int(
                natural_result["certificate_safety"]["revisit_activation_queries"]
            ),
            **metric_row(natural_result, "revisit"),
        },
    ]
    require([row["cec_authorized_queries"] for row in bands] == [2, 19, 21],
            "CEC authorization transfer changed")
    require(all(row["cec_minus_native"]["losses"] == 0 for row in bands),
            "CEC has an unaccounted native loss in the spectrum")

    return {
        "schema_version": SCHEMA,
        "claim_scope": (
            "post-hoc derived analysis of a prospectively frozen, independently "
            "verified Final14 controlled metric-depth population"
        ),
        "new_navigation_rollouts": 0,
        "method_or_threshold_changed": False,
        "same_ordered_history_population": True,
        "histories": len(unsupported_ids),
        "scene_clusters": len({scene for scene, _episode in unsupported_ids}),
        "history_identities": [list(value) for value in unsupported_ids],
        "support_bands": bands,
        "source_receipts": {
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            "independent_verifier": {
                "path": str(verifier_path), "sha256": sha256(verifier_path)
            },
            "natural_manifest": {
                "path": str(natural_path), "sha256": sha256(natural_path)
            },
            "hard_manifest": {"path": str(hard_path), "sha256": sha256(hard_path)},
        },
        "interpretation_guard": (
            "This controlled MP3D result demonstrates a support-conditioned "
            "authorization transfer; it is not fresh-scene Full-Mono evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
