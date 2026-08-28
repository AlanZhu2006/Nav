#!/usr/bin/env python3
"""Independent raw-fragment verifier for the Natural-B expansion audit."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "hm3d_fullmono_lifelong_natural_b_expansion_audit_v1_20260828"
VERIFY_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_independent_"
    "verification_v1_20260828"
)
SLOT_START = 4
SLOT_STOP = 16
MAX_NEW = 2
MINIMUM_SEPARATION_M = 2.0
STRATA = ("front", "side", "rear")


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
    require(len(fields) == 2 and fields[1] == path.name,
            f"invalid SHA-256 sidecar: {sidecar}")
    actual = sha256(path)
    require(fields[0] == actual, f"SHA-256 mismatch: {path}")
    return actual


def planar_distance(first: list[float], second: list[float]) -> float:
    return math.hypot(first[0] - second[0], first[2] - second[2])


def direction_in_stratum(relative_degrees: float, stratum: str) -> bool:
    magnitude = abs(float(relative_degrees))
    if stratum == "front":
        return magnitude <= 60.0 + 1e-9
    if stratum == "side":
        return 60.0 < magnitude <= 120.0 + 1e-9
    if stratum == "rear":
        return 120.0 < magnitude <= 180.0 + 1e-9
    return False


def original_candidates(manifest: dict[str, Any]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    identities = set()
    for item in manifest["episodes"]:
        scene = str(item["scene"])
        construction = item["lifelong_construction"]
        recipient = str(construction["recipient_episode"])
        slot = int(construction["candidate_slot"])
        identity = str(construction["candidate_identity"])
        require(0 <= slot < SLOT_START, "original slot outside 0--3")
        require(identity == f"{recipient}__natural_b_{slot:02d}",
                "original candidate identity changed")
        require((scene, identity) not in identities,
                "duplicate original candidate identity")
        identities.add((scene, identity))
        position = [float(value) for value in construction["goal_floor_position"]]
        require(len(position) == 3 and all(math.isfinite(v) for v in position),
                "invalid original candidate position")
        result[(scene, recipient)].append({
            "candidate_identity": identity,
            "candidate_slot": slot,
            "goal_floor_position": position,
        })
    for rows in result.values():
        rows.sort(key=lambda row: int(row["candidate_slot"]))
    return dict(result)


def recount(
    paths: list[Path],
    *,
    expected_scenes: int,
    expected_protocol_sha256: str,
    original_manifest_sha256: str,
    original: dict[tuple[str, str], list[dict]],
) -> dict[str, Any]:
    require(len(paths) == expected_scenes, "raw fragment count changed")
    scene_indices = set()
    scene_names = set()
    recipient_identities = set()
    source_histories = 0
    controlled = 0
    original_count = 0
    expansion_recipients = 0
    expansion_candidates = 0
    expansion_scenes = set()
    status_counts: collections.Counter[str] = collections.Counter()
    strata: collections.Counter[str] = collections.Counter()
    covis_values = []

    for path in paths:
        verify_sidecar(path)
        payload = json.loads(path.read_text())
        require(payload.get("schema_version") == AUDIT_SCHEMA,
                f"fragment schema changed: {path}")
        require(payload.get("expansion_protocol_sha256")
                == expected_protocol_sha256,
                "expansion protocol hash changed")
        require(payload.get("original_v4_manifest_sha256")
                == original_manifest_sha256,
                "original v4 manifest hash changed")
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "fragment read navigation outcomes")
        require(payload.get("evaluation_authorized") is False,
                "fragment authorized navigation")
        scene = str(payload["scene"])
        scene_index = int(payload["scene_index"])
        require(scene_index not in scene_indices and scene not in scene_names,
                "duplicate scene fragment")
        require(path.parent.name == f"{scene_index:02d}_{scene}",
                "fragment path/identity mismatch")
        scene_indices.add(scene_index)
        scene_names.add(scene)
        source_histories += int(payload["source_materialized_A_histories"])
        recipients = payload["recipients"]
        require(int(payload["controlled_revisit_constructible_histories"])
                == len(recipients), "controlled-Revisit count mismatch")
        controlled += len(recipients)
        local_original = 0
        local_recipients = 0
        local_candidates = 0
        for row in recipients:
            require(str(row["scene"]) == scene, "recipient scene changed")
            episode = str(row["episode"])
            key = (scene, episode)
            require(key not in recipient_identities, "duplicate recipient")
            recipient_identities.add(key)
            expected_original = original.get(key, [])
            require(row["original_candidates"] == expected_original,
                    "serialized original candidate references changed")
            require(int(row["original_candidate_count"])
                    == len(expected_original),
                    "original candidate count mismatch")
            local_original += len(expected_original)
            original_count += len(expected_original)

            attempts = row["candidate_slot_attempts"]
            require([int(item["slot"]) for item in attempts]
                    == list(range(SLOT_START, SLOT_STOP)),
                    "expansion slot ledger changed")
            candidates = row["candidates"]
            require(int(row["candidate_count"]) == len(candidates),
                    "expansion candidate count mismatch")
            require(len(candidates) <= MAX_NEW,
                    "recipient exceeds expansion candidate cap")
            candidate_by_slot = {
                int(candidate["candidate_slot"]): candidate
                for candidate in candidates
            }
            require(len(candidate_by_slot) == len(candidates),
                    "duplicate expansion candidate slot")
            accepted_so_far = 0
            for item in attempts:
                slot = int(item["slot"])
                identity = f"{episode}__natural_b_{slot:02d}"
                require(str(item["identity"]) == identity,
                        "slot-attempt identity changed")
                status = str(item["status"])
                if accepted_so_far >= MAX_NEW:
                    require(status == "not_attempted_after_recipient_cap",
                            "slot was attempted after candidate cap")
                    require(item["sampling_diagnostics"] is None,
                            "unattempted slot has diagnostics")
                    require(slot not in candidate_by_slot,
                            "unattempted slot has candidate")
                    continue
                require(status in {
                    "constructible", "no_natural_B_candidate"
                }, "invalid attempted-slot status")
                if status == "constructible":
                    require(slot in candidate_by_slot,
                            "constructible slot lacks candidate")
                    accepted_so_far += 1
                else:
                    require(slot not in candidate_by_slot,
                            "rejected slot has candidate")
            require(accepted_so_far == len(candidates),
                    "slot ledger/candidate list mismatch")

            prior_positions = [
                list(item["goal_floor_position"])
                for item in expected_original
            ]
            for slot in sorted(candidate_by_slot):
                candidate = candidate_by_slot[slot]
                require(SLOT_START <= slot < SLOT_STOP,
                        "candidate slot outside expansion range")
                require(str(candidate["candidate_identity"])
                        == f"{episode}__natural_b_{slot:02d}",
                        "candidate identity changed")
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
                        "candidate violates Novel support contract")
                stratum = str(candidate["assigned_direction_stratum"])
                require(stratum in STRATA, "candidate stratum changed")
                require(direction_in_stratum(float(candidate[
                    "initial_path_direction_relative_to_a_end_deg"
                ]), stratum), "candidate outside assigned direction stratum")
                position = [
                    float(value) for value in candidate["goal_floor_position"]
                ]
                require(len(position) == 3
                        and all(math.isfinite(v) for v in position),
                        "invalid expansion candidate position")
                require(all(planar_distance(position, prior)
                            >= MINIMUM_SEPARATION_M - 1e-6
                            for prior in prior_positions),
                        "candidate violates pairwise separation")
                prior_positions.append(position)
                strata[stratum] += 1
                covis_values.append(covis)
            status = str(row["status"])
            require(status in {
                "constructible", "no_additional_natural_B_candidate"
            }, "recipient status changed")
            require((status == "constructible") == bool(candidates),
                    "recipient status/candidates disagree")
            status_counts[status] += 1
            if candidates:
                expansion_recipients += 1
                local_recipients += 1
                expansion_scenes.add(scene)
            expansion_candidates += len(candidates)
            local_candidates += len(candidates)
        require(int(payload["original_candidate_histories_referenced"])
                == local_original, "fragment original count mismatch")
        require(int(payload["expansion_constructible_recipients"])
                == local_recipients, "fragment recipient count mismatch")
        require(int(payload["expansion_candidate_histories"])
                == local_candidates, "fragment candidate count mismatch")

    require(scene_indices == set(range(expected_scenes)),
            "scene indices are not the complete frozen range")
    return {
        "scene_fragments": len(paths),
        "source_materialized_A_histories": source_histories,
        "controlled_revisit_constructible_histories": controlled,
        "original_candidate_histories_referenced": original_count,
        "expansion_constructible_recipients": expansion_recipients,
        "expansion_candidate_histories": expansion_candidates,
        "expansion_scene_clusters": len(expansion_scenes),
        "status_counts": dict(sorted(status_counts.items())),
        "direction_strata": dict(sorted(strata.items())),
        "candidate_max_online_A_covis": {
            "minimum": min(covis_values) if covis_values else None,
            "median": statistics.median(covis_values)
            if covis_values else None,
            "maximum": max(covis_values) if covis_values else None,
        },
    }


def compare_summary(recounted: dict[str, Any], summary: dict[str, Any]) -> None:
    for key in (
        "scene_fragments",
        "source_materialized_A_histories",
        "controlled_revisit_constructible_histories",
        "original_candidate_histories_referenced",
        "expansion_constructible_recipients",
        "expansion_candidate_histories",
        "expansion_scene_clusters",
        "status_counts",
        "direction_strata",
    ):
        require(summary.get(key) == recounted[key],
                f"summary/recount mismatch: {key}")
    for key in ("minimum", "median", "maximum"):
        first = summary["candidate_max_online_A_covis"][key]
        second = recounted["candidate_max_online_A_covis"][key]
        if first is None or second is None:
            require(first is second, f"summary/recount mismatch: covis {key}")
        else:
            require(math.isclose(float(first), float(second),
                                 rel_tol=0.0, abs_tol=1e-12),
                    f"summary/recount mismatch: covis {key}")
    require(summary.get("query_policy_outcomes_read") is False
            and summary.get("navigation_outcomes_read") is False
            and summary.get("evaluation_authorized") is False,
            "summary violates result-blind scope")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expansion-protocol", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expansion_protocol_sha = sha256(args.expansion_protocol)
    protocol = json.loads(args.expansion_protocol.read_text())
    require(protocol.get("schema_version") == AUDIT_SCHEMA,
            "expansion protocol schema changed")
    original_manifest_sha = sha256(args.original_manifest)
    require(original_manifest_sha
            == protocol["frozen_sources"]["original_v4_manifest_sha256"],
            "original manifest differs from expansion protocol")
    manifest = json.loads(args.original_manifest.read_text())
    original = original_candidates(manifest)
    paths = sorted(args.audit_root.glob("*/natural_b_expansion_audit.json"))
    recounted = recount(
        paths,
        expected_scenes=args.expected_scenes,
        expected_protocol_sha256=expansion_protocol_sha,
        original_manifest_sha256=original_manifest_sha,
        original=original,
    )
    require(recounted["source_materialized_A_histories"] == 130,
            "source A history count changed")
    require(recounted["controlled_revisit_constructible_histories"] == 80,
            "controlled-Revisit population changed")
    require(recounted["original_candidate_histories_referenced"] == 99,
            "original candidate ledger changed")
    summary_sha = verify_sidecar(args.summary)
    summary = json.loads(args.summary.read_text())
    require(summary.get("schema_version") == AUDIT_SCHEMA,
            "summary schema changed")
    compare_summary(recounted, summary)
    report = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "scope": "independent expansion constructibility recount only",
        "expansion_protocol_sha256": expansion_protocol_sha,
        "original_v4_manifest_sha256": original_manifest_sha,
        "summary_sha256": summary_sha,
        "recount": recounted,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
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
