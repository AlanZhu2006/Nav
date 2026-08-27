#!/usr/bin/env python3
"""Independent raw-fragment verifier for the Natural-B construction audit.

This verifier deliberately does not import the construction or aggregation
implementation.  It recounts the frozen scene fragments, validates the
serialized Novel-B contract, and compares the recount with the sealed summary.
It verifies constructibility only; it never authorizes or reads navigation
evaluation outcomes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "hm3d_fullmono_lifelong_natural_b_audit_v1_20260827"
VERIFY_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_independent_verification_v1_20260827"
)
STRATA = ("front", "side", "rear")
MAX_CANDIDATES_PER_RECIPIENT = 4
REFERENCE_MINIMUM_CANDIDATES = 96
REFERENCE_MINIMUM_SCENES = 15


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text().split()
    require(
        len(fields) == 2 and fields[1] == path.name,
        f"invalid SHA-256 sidecar: {sidecar}",
    )
    actual = sha256(path)
    require(fields[0] == actual, f"SHA-256 mismatch: {path}")
    return actual


def direction_in_stratum(relative_degrees: float, stratum: str) -> bool:
    magnitude = abs(float(relative_degrees))
    if stratum == "front":
        return magnitude <= 60.0 + 1e-9
    if stratum == "side":
        return 60.0 < magnitude <= 120.0 + 1e-9
    if stratum == "rear":
        return 120.0 < magnitude <= 180.0 + 1e-9
    return False


def recount_fragments(paths: list[Path], expected_scenes: int) -> dict[str, Any]:
    require(len(paths) == expected_scenes, "raw fragment count changed")
    scene_indices: set[int] = set()
    scene_names: set[str] = set()
    identities: set[tuple[str, str]] = set()
    source_histories = 0
    controlled_histories = 0
    constructible_recipients = 0
    candidate_count = 0
    status_counts: collections.Counter[str] = collections.Counter()
    direction_strata: collections.Counter[str] = collections.Counter()
    candidate_covis: list[float] = []
    constructible_scenes: set[str] = set()
    protocol_hashes: set[str] = set()

    for path in paths:
        verify_sidecar(path)
        payload = json.loads(path.read_text())
        require(payload.get("schema_version") == AUDIT_SCHEMA,
                f"fragment schema changed: {path}")
        require(payload.get("query_policy_outcomes_read") is False,
                f"query outcomes were read: {path}")
        require(payload.get("navigation_outcomes_read") is False,
                f"navigation outcomes were read: {path}")
        require(payload.get("evaluation_authorized") is False,
                f"fragment improperly authorizes evaluation: {path}")

        scene = str(payload["scene"])
        scene_index = int(payload["scene_index"])
        require(scene_index not in scene_indices, "duplicate scene index")
        require(scene not in scene_names, "duplicate scene name")
        require(path.parent.name == f"{scene_index:02d}_{scene}",
                f"fragment path/identity mismatch: {path}")
        scene_indices.add(scene_index)
        scene_names.add(scene)
        protocol_hashes.add(str(payload["protocol_sha256"]))
        source_histories += int(payload["source_materialized_A_histories"])

        recipients = payload["recipients"]
        require(
            int(payload["controlled_revisit_constructible_histories"])
            == len(recipients),
            f"controlled-Revisit count mismatch: {path}",
        )
        controlled_histories += len(recipients)
        local_constructible = 0
        local_candidates = 0
        for row in recipients:
            require(str(row["scene"]) == scene,
                    "recipient scene differs from fragment")
            episode = str(row["episode"])
            identity = (scene, episode)
            require(identity not in identities, "duplicate recipient identity")
            identities.add(identity)
            status = str(row["status"])
            require(status in {"constructible", "no_natural_B_candidate"},
                    f"unknown recipient status: {status}")
            status_counts[status] += 1
            candidates = row["candidates"]
            require(int(row["candidate_count"]) == len(candidates),
                    "recipient candidate count mismatch")
            require(len(candidates) <= MAX_CANDIDATES_PER_RECIPIENT,
                    "recipient exceeds candidate cap")
            require((status == "constructible") == bool(candidates),
                    "recipient status/candidates disagree")
            slots: set[int] = set()
            for candidate in candidates:
                slot = int(candidate["candidate_slot"])
                require(0 <= slot < MAX_CANDIDATES_PER_RECIPIENT,
                        "candidate slot outside frozen range")
                require(slot not in slots, "duplicate candidate slot")
                slots.add(slot)
                require(
                    str(candidate["candidate_identity"])
                    == f"{episode}__natural_b_{slot:02d}",
                    "candidate identity changed",
                )
                require(candidate.get("support_band") == "unsupported_novel",
                        "candidate support band changed")
                query_distance = float(candidate["query_geodesic_m"])
                paired_distance = float(
                    candidate["paired_revisit_separation_m"]
                )
                covis = float(candidate["max_online_a_covis"])
                require(2.0 <= query_distance <= 9.0,
                        "candidate A-to-B distance outside contract")
                require(2.0 <= paired_distance <= 9.0,
                        "candidate B-to-C distance outside contract")
                require(0.0 <= covis < 0.10,
                        "candidate online-A support violates Novel contract")
                stratum = str(candidate["assigned_direction_stratum"])
                require(stratum in STRATA, "candidate direction stratum changed")
                require(direction_in_stratum(
                    float(candidate[
                        "initial_path_direction_relative_to_a_end_deg"
                    ]), stratum
                ), "candidate direction falls outside assigned stratum")
                require(
                    candidate.get("goal_yaw_contract")
                    == "identity_hash_eight_world_yaw_bins",
                    "candidate yaw contract changed",
                )
                direction_strata[stratum] += 1
                candidate_covis.append(covis)
            if candidates:
                local_constructible += 1
                constructible_recipients += 1
                constructible_scenes.add(scene)
            local_candidates += len(candidates)
            candidate_count += len(candidates)
        require(
            int(payload["natural_B_constructible_recipients"])
            == local_constructible,
            f"fragment constructible-recipient count mismatch: {path}",
        )
        require(
            int(payload["natural_B_candidate_histories"]) == local_candidates,
            f"fragment candidate count mismatch: {path}",
        )

    require(scene_indices == set(range(expected_scenes)),
            "scene indices are not the complete frozen range")
    require(len(protocol_hashes) == 1, "fragments used multiple protocols")
    return {
        "scene_fragments": len(paths),
        "source_materialized_A_histories": source_histories,
        "controlled_revisit_constructible_histories": controlled_histories,
        "natural_B_constructible_recipients": constructible_recipients,
        "natural_B_candidate_histories": candidate_count,
        "natural_B_constructible_scene_clusters": len(constructible_scenes),
        "status_counts": dict(sorted(status_counts.items())),
        "direction_strata": dict(sorted(direction_strata.items())),
        "candidate_max_online_A_covis": {
            "minimum": min(candidate_covis) if candidate_covis else None,
            "median": statistics.median(candidate_covis)
            if candidate_covis else None,
            "maximum": max(candidate_covis) if candidate_covis else None,
        },
        "protocol_sha256": next(iter(protocol_hashes)),
    }


def compare_summary(recount: dict[str, Any], summary: dict[str, Any]) -> None:
    for key in (
        "scene_fragments",
        "source_materialized_A_histories",
        "controlled_revisit_constructible_histories",
        "natural_B_constructible_recipients",
        "natural_B_candidate_histories",
        "natural_B_constructible_scene_clusters",
        "status_counts",
        "direction_strata",
    ):
        require(summary.get(key) == recount[key],
                f"summary/recount mismatch: {key}")
    for key in ("minimum", "median", "maximum"):
        reported = summary["candidate_max_online_A_covis"][key]
        counted = recount["candidate_max_online_A_covis"][key]
        if reported is None or counted is None:
            require(reported is counted,
                    f"summary/recount covis mismatch: {key}")
        else:
            require(math.isclose(float(reported), float(counted),
                                 rel_tol=0.0, abs_tol=1e-12),
                    f"summary/recount covis mismatch: {key}")

    contract = summary["construction_contract"]
    require(
        contract == {
            "maximum_candidates_per_controlled_revisit_history": 4,
            "minimum_candidate_planar_separation_m": 2.0,
            "A_to_B_geodesic_m": [2.0, 9.0],
            "B_to_C_geodesic_m": [2.0, 9.0],
            "B_max_online_A_covis_exclusive": 0.10,
            "same_scene_navmesh": True,
            "goal_rendered_at_frozen_camera_height": True,
            "cross_online_history_donor_required": False,
        },
        "summary construction contract changed",
    )
    gate = summary["v3_source_gate_reference"]
    independently_met = (
        recount["natural_B_candidate_histories"]
        >= REFERENCE_MINIMUM_CANDIDATES
        and recount["natural_B_constructible_scene_clusters"]
        >= REFERENCE_MINIMUM_SCENES
    )
    require(int(gate["minimum_candidate_histories"])
            == REFERENCE_MINIMUM_CANDIDATES,
            "candidate reference gate changed")
    require(int(gate["minimum_scene_clusters"])
            == REFERENCE_MINIMUM_SCENES,
            "scene reference gate changed")
    require(gate["met"] is independently_met, "gate decision mismatch")
    require(gate["evaluation_authority_conferred"] is False,
            "gate improperly conferred evaluation authority")
    require(summary.get("query_policy_outcomes_read") is False,
            "summary reports query-outcome access")
    require(summary.get("navigation_outcomes_read") is False,
            "summary reports navigation-outcome access")
    require(summary.get("evaluation_authorized") is False,
            "summary improperly authorizes evaluation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.audit_root.glob("*/natural_b_audit.json"))
    recount = recount_fragments(paths, args.expected_scenes)
    summary_sha256 = verify_sidecar(args.summary)
    summary = json.loads(args.summary.read_text())
    require(summary.get("schema_version") == AUDIT_SCHEMA,
            "summary schema changed")
    compare_summary(recount, summary)

    report = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "scope": "independent constructibility recount; no evaluation authority",
        "scene_fragment_hashes_verified": len(paths),
        "summary_sha256": summary_sha256,
        "recount": recount,
        "reference_gate_met": bool(
            summary["v3_source_gate_reference"]["met"]
        ),
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
        "limitation": (
            "pairwise 2 m candidate separation is enforced by the sealed "
            "constructor but candidate positions are not serialized, so this "
            "verifier cannot recompute that one geometric predicate"
        ),
    }
    require(not args.out.exists(), f"verification output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
