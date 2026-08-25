"""Pure contracts for the frozen HM3D full-mono lifelong experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_SCHEMA = "hm3d_fullmono_lifelong_accumulation_protocol_v1_20260824"
EXPANSION_PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_result_blind_power_expansion_v2_20260825"
)
PREFIX_SCHEMA = "hm3d_fullmono_lifelong_factual_prefix_v1_20260824"
RESULT_SCHEMA = "hm3d_fullmono_lifelong_eval_v1_20260824"
ARMS = ("all_prior", "initial_leg_only", "forced_reject_native")
QUERY_NAMES = ("C", "B2", "C2")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") in {
        PROTOCOL_SCHEMA, EXPANSION_PROTOCOL_SCHEMA,
    },
            "lifelong protocol schema changed")
    require(payload.get("post_prefix_query_outcomes_read_before_freeze") is False,
            "protocol was not frozen before query outcomes")
    require(payload["guards"]["no_post_prefix_outcome_filtering"] is True,
            "post-prefix outcome filtering is not forbidden")
    require(tuple(row["name"] for row in payload["query_runtime"]["arms"])
            == ARMS, "lifelong arm order changed")
    if payload["schema_version"] == EXPANSION_PROTOCOL_SCHEMA:
        construction = payload["novel_b_construction"]
        require(int(construction["temporal_samples_per_donor"]) >= 2,
                "expansion did not sample donor time")
        require(int(construction["maximum_candidates_per_recipient"]) >= 2,
                "expansion did not increase result-blind proposals")
        require(construction["candidate_outcomes_read"] is False,
                "expansion candidate generation read B outcomes")
        runtime = payload["query_runtime"]
        factual_c = runtime["factual_C_prefix"]
        require(factual_c["executed_once_before_any_B2_treatment"] is True
                and factual_c["selection_reads_B2_navigation_outcomes"] is False
                and factual_c[
                    "replayed_by_exact_RGB_pose_and_goal_session_receipts"
                ] is True,
                "expansion did not freeze one factual C before B2")
        require(runtime["primary_endpoint"]
                == "B2_success_after_the_same_sealed_factual_C_prefix"
                and runtime["C2_is_not_part_of_the_power_expansion"] is True,
                "expansion endpoint is not shared-C B2")
    return payload


def bind_parent(protocol: dict[str, Any], parent_root: Path) -> dict[str, Path]:
    parent = protocol["parent"]
    expected_root = Path(parent["run_root"])
    require(parent_root.resolve() == expected_root.resolve(),
            "full-mono parent run root changed")
    paths = {
        "manifest": parent_root / parent["parent_manifest"],
        "population": parent_root / parent["fullmono_population_receipt"],
    }
    require(sha256_file(paths["manifest"]) == parent["parent_manifest_sha256"],
            "parent manifest changed")
    require(sha256_file(paths["population"])
            == parent["fullmono_population_receipt_sha256"],
            "parent population receipt changed")
    return paths


def rotated_arm_order(index: int) -> tuple[str, ...]:
    offset = int(index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def donor_rank(row: dict[str, Any]) -> tuple[float, float, int, str, int]:
    """Frozen result-blind ranking for a cross-history Novel-B donor."""

    return (
        abs(float(row["a_to_b_geodesic_m"]) - 4.0),
        float(row["max_recipient_a_covis"]),
        int(row["donor_episode_rank"]),
        str(row["donor_episode"]),
        int(row.get("donor_frame_index", -1)),
    )


def select_donor(
    rows: Iterable[dict[str, Any]],
    *,
    recipient_episode: str,
    maximum_a_covis: float = 0.10,
    minimum_geodesic_m: float = 2.0,
    maximum_geodesic_m: float = 9.0,
    minimum_b_to_c_m: float = 2.0,
    maximum_b_to_c_m: float = 9.0,
) -> dict[str, Any] | None:
    """Select without reading any B/C/B2/C2 navigation outcome."""

    accepted = []
    for row in rows:
        if str(row["donor_episode"]) == str(recipient_episode):
            continue
        a_to_b = float(row["a_to_b_geodesic_m"])
        b_to_c = float(row["b_to_c_geodesic_m"])
        support = float(row["max_recipient_a_covis"])
        if not (minimum_geodesic_m <= a_to_b <= maximum_geodesic_m):
            continue
        if not (minimum_b_to_c_m <= b_to_c <= maximum_b_to_c_m):
            continue
        if not support < maximum_a_covis:
            continue
        accepted.append(row)
    return min(accepted, key=donor_rank) if accepted else None


def select_donors(
    rows: Iterable[dict[str, Any]],
    *,
    recipient_episode: str,
    maximum_candidates: int,
    maximum_per_donor: int = 1,
    prefer_distinct_direction_strata: bool = True,
    maximum_a_covis: float = 0.10,
    minimum_geodesic_m: float = 2.0,
    maximum_geodesic_m: float = 9.0,
    minimum_b_to_c_m: float = 2.0,
    maximum_b_to_c_m: float = 9.0,
) -> list[dict[str, Any]]:
    """Freeze several temporal donor hypotheses without reading outcomes."""
    require(maximum_candidates > 0, "maximum_candidates must be positive")
    require(maximum_per_donor > 0, "maximum_per_donor must be positive")
    eligible = []
    for row in rows:
        if str(row["donor_episode"]) == str(recipient_episode):
            continue
        a_to_b = float(row["a_to_b_geodesic_m"])
        b_to_c = float(row["b_to_c_geodesic_m"])
        support = float(row["max_recipient_a_covis"])
        if not (minimum_geodesic_m <= a_to_b <= maximum_geodesic_m):
            continue
        if not (minimum_b_to_c_m <= b_to_c <= maximum_b_to_c_m):
            continue
        if not support < maximum_a_covis:
            continue
        eligible.append(row)
    ordered = sorted(eligible, key=donor_rank)
    selected: list[dict[str, Any]] = []
    donor_counts: dict[str, int] = {}
    used_strata: set[str] = set()
    while len(selected) < maximum_candidates:
        available = [
            row for row in ordered if row not in selected
            and donor_counts.get(str(row["donor_episode"]), 0)
            < maximum_per_donor
        ]
        if not available:
            break
        distinct = [
            row for row in available
            if str(row.get("assigned_direction_stratum")) not in used_strata
        ]
        choice = (
            distinct[0]
            if prefer_distinct_direction_strata and distinct
            else available[0]
        )
        selected.append(choice)
        donor = str(choice["donor_episode"])
        donor_counts[donor] = donor_counts.get(donor, 0) + 1
        used_strata.add(str(choice.get("assigned_direction_stratum")))
    return selected


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    require(gains >= 0 and losses >= 0, "discordant counts must be non-negative")
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(int(gains), int(losses)) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def common_c_success(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """B2 is paired only after a shared successful C action prefix."""

    return bool(int(first["reached_C"]) and int(second["reached_C"]))


__all__ = [
    "ARMS",
    "PREFIX_SCHEMA",
    "PROTOCOL_SCHEMA",
    "EXPANSION_PROTOCOL_SCHEMA",
    "QUERY_NAMES",
    "RESULT_SCHEMA",
    "bind_parent",
    "common_c_success",
    "donor_rank",
    "exact_mcnemar",
    "load_protocol",
    "require",
    "rotated_arm_order",
    "select_donor",
    "select_donors",
    "sha256_file",
]
