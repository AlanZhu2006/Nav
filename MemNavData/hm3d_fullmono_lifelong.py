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
POWERED_EXPANSION_PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_result_blind_power_expansion_v3_20260826"
)
DIRECT_NATURAL_PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_direct_natural_power_v4_20260827"
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
        PROTOCOL_SCHEMA,
        EXPANSION_PROTOCOL_SCHEMA,
        POWERED_EXPANSION_PROTOCOL_SCHEMA,
        DIRECT_NATURAL_PROTOCOL_SCHEMA,
    },
            "lifelong protocol schema changed")
    require(payload.get("post_prefix_query_outcomes_read_before_freeze") is False,
            "protocol was not frozen before query outcomes")
    require(payload["guards"]["no_post_prefix_outcome_filtering"] is True,
            "post-prefix outcome filtering is not forbidden")
    require(tuple(row["name"] for row in payload["query_runtime"]["arms"])
            == ARMS, "lifelong arm order changed")
    if payload["schema_version"] in {
        EXPANSION_PROTOCOL_SCHEMA,
        POWERED_EXPANSION_PROTOCOL_SCHEMA,
    }:
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
    if payload["schema_version"] == POWERED_EXPANSION_PROTOCOL_SCHEMA:
        construction = payload["novel_b_construction"]
        require(int(construction["temporal_samples_per_donor"]) >= 8,
                "powered expansion donor sampling is too sparse")
        require(int(construction["maximum_candidates_per_recipient"]) >= 4,
                "powered expansion did not increase candidate density")
        require(int(construction["maximum_candidates_per_donor_history"]) >= 2,
                "powered expansion still permits only one donor frame")
        require(float(construction[
            "minimum_candidate_planar_separation_m"
        ]) >= 2.0,
                "powered expansion candidates can share a success region")
        gate = payload["construction_power_gate"]
        require(int(gate["minimum_candidate_histories"]) > 0,
                "powered expansion has no candidate-count gate")
        require(int(gate["minimum_scene_clusters"]) > 0,
                "powered expansion has no scene-count gate")
        require(gate["halt_before_factual_B_if_not_met"] is True,
                "powered expansion can run B while underpowered")
    if payload["schema_version"] == DIRECT_NATURAL_PROTOCOL_SCHEMA:
        construction = payload["novel_b_construction"]
        require(construction["source"]
                == "sealed_direct_natural_B_audit_all_candidates",
                "v4 does not consume the sealed Natural-B ledger")
        require(int(construction["maximum_candidates_per_recipient"]) == 4,
                "v4 Natural-B candidate ceiling changed")
        require(float(construction[
            "minimum_candidate_planar_separation_m"
        ]) == 2.0, "v4 Natural-B separation changed")
        require(construction["candidate_outcomes_read"] is False,
                "v4 candidate materialization read navigation outcomes")
        audit = payload["sealed_natural_b_audit"]
        require(int(audit["expected_scene_fragments"]) == 54
                and int(audit["expected_candidate_histories"]) == 99
                and int(audit["expected_recipient_histories"]) == 61
                and int(audit["expected_scene_clusters"]) == 35,
                "v4 Natural-B audit ledger changed")
        require(audit["independent_verification_required"] is True,
                "v4 omitted independent construction verification")
        runtime = payload["query_runtime"]
        require(runtime["sequence"] == ["A", "B", "C", "B2", "C2"],
                "v4 is not the frozen five-leg sequence")
        require(runtime["primary_endpoint"]
                == "B2_success_after_the_same_sealed_factual_C_prefix",
                "v4 primary accumulation endpoint changed")
        require(runtime["C2_role"]
                == "secondary_full_sequence_survival_and_anchor_provenance",
                "v4 C2 role changed")
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
    minimum_planar_separation_m: float = 0.0,
    maximum_a_covis: float = 0.10,
    minimum_geodesic_m: float = 2.0,
    maximum_geodesic_m: float = 9.0,
    minimum_b_to_c_m: float = 2.0,
    maximum_b_to_c_m: float = 9.0,
) -> list[dict[str, Any]]:
    """Freeze several temporal donor hypotheses without reading outcomes."""
    require(maximum_candidates > 0, "maximum_candidates must be positive")
    require(maximum_per_donor > 0, "maximum_per_donor must be positive")
    require(minimum_planar_separation_m >= 0.0,
            "minimum planar separation must be non-negative")

    def floor_position(row: dict[str, Any]) -> tuple[float, float, float]:
        value = row.get("_position", row.get("goal_floor_position"))
        require(value is not None and len(value) == 3,
                "candidate floor position missing")
        return tuple(float(component) for component in value)

    def separated(row: dict[str, Any]) -> bool:
        if minimum_planar_separation_m <= 0.0:
            return True
        x, _y, z = floor_position(row)
        for prior in selected:
            px, _py, pz = floor_position(prior)
            if math.hypot(x - px, z - pz) < minimum_planar_separation_m:
                return False
        return True

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
            and separated(row)
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
    "POWERED_EXPANSION_PROTOCOL_SCHEMA",
    "DIRECT_NATURAL_PROTOCOL_SCHEMA",
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
