#!/usr/bin/env python3
"""Result-blind additional Natural-B audit for lifelong power expansion.

The original v4 audit deterministically attempted slots 0--3.  This audit
keeps those candidates immutable and attempts slots 4--15 for every frozen
controlled-Revisit recipient, accepting at most two new candidates per
recipient.  Candidate generation reads no factual-B result and no C/B2/C2
navigation outcome.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from audit_hm3d_fullmono_lifelong_constructibility import (
    reconstruct_revisit_position,
)
from build_final14_role_pair_scene import (
    NaturalNovelConstructionError,
    _candidate_json,
    sample_natural_novel,
)
from construct_hm3d_fullmono_lifelong_ab import load_histories
from generate_twoleg import make_sim
from hm3d_fullmono_lifelong import bind_parent, load_protocol, require, sha256_file


SCHEMA = "hm3d_fullmono_lifelong_natural_b_expansion_audit_v1_20260828"
PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_audit_v1_20260828"
)
SLOT_START = 4
SLOT_STOP = 16
MAXIMUM_NEW_CANDIDATES_PER_RECIPIENT = 2
LEGACY_EPISODE_RANK_MULTIPLIER = 4
MINIMUM_CANDIDATE_SEPARATION_M = 2.0


def load_expansion_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") == PROTOCOL_SCHEMA,
            "expansion audit protocol schema changed")
    sources = payload["frozen_sources"]
    expansion = payload["candidate_expansion"]
    guards = payload["guards"]
    require(int(sources["expected_scene_fragments"]) == 54,
            "expansion scene count changed")
    require(int(sources["expected_controlled_revisit_histories"]) == 80,
            "controlled-Revisit count changed")
    require(int(expansion["slot_start_inclusive"]) == SLOT_START
            and int(expansion["slot_stop_exclusive"]) == SLOT_STOP,
            "expansion slot range changed")
    require(int(expansion["maximum_new_candidates_per_recipient"])
            == MAXIMUM_NEW_CANDIDATES_PER_RECIPIENT,
            "expansion per-recipient cap changed")
    require(int(expansion["legacy_episode_rank_multiplier"])
            == LEGACY_EPISODE_RANK_MULTIPLIER,
            "expansion seed contract changed")
    require(float(expansion["minimum_candidate_planar_separation_m"])
            == MINIMUM_CANDIDATE_SEPARATION_M,
            "candidate separation changed")
    require(guards["no_C_B2_C2_outcome_access"] is True
            and guards["no_candidate_selection_by_factual_B_outcome"] is True
            and guards["audit_does_not_authorize_navigation"] is True,
            "expansion result-blind guards changed")
    return payload


def _original_positions(
    manifest: dict[str, Any], scene: str
) -> dict[str, list[dict[str, Any]]]:
    by_recipient: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    identities: set[str] = set()
    for item in manifest["episodes"]:
        if str(item["scene"]) != scene:
            continue
        construction = item["lifelong_construction"]
        recipient = str(construction["recipient_episode"])
        slot = int(construction["candidate_slot"])
        identity = str(construction["candidate_identity"])
        require(0 <= slot < SLOT_START, "original candidate slot changed")
        require(identity == f"{recipient}__natural_b_{slot:02d}",
                "original candidate identity changed")
        require(identity not in identities, "duplicate original candidate")
        identities.add(identity)
        position = [float(value) for value in construction["goal_floor_position"]]
        require(len(position) == 3 and all(np.isfinite(position)),
                "invalid original candidate position")
        by_recipient[recipient].append({
            "candidate_identity": identity,
            "candidate_slot": slot,
            "goal_floor_position": position,
        })
    for rows in by_recipient.values():
        rows.sort(key=lambda row: int(row["candidate_slot"]))
    return dict(by_recipient)


def aggregate_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    require(payloads, "expansion audit has no scene payloads")
    rows = []
    for payload in payloads:
        require(payload.get("schema_version") == SCHEMA,
                "expansion audit schema changed")
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "expansion audit read navigation outcomes")
        require(payload.get("evaluation_authorized") is False,
                "expansion audit authorized navigation")
        rows.extend(payload["recipients"])
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    scenes = {str(row["scene"]) for row in rows if row["candidates"]}
    statuses = collections.Counter(str(row["status"]) for row in rows)
    strata = collections.Counter(
        str(candidate["assigned_direction_stratum"])
        for candidate in candidates
    )
    covis = [float(candidate["max_online_a_covis"]) for candidate in candidates]
    return {
        "schema_version": SCHEMA,
        "scope": "result-blind additional Natural-B constructibility audit",
        "scene_fragments": len(payloads),
        "source_materialized_A_histories": sum(
            int(payload["source_materialized_A_histories"])
            for payload in payloads
        ),
        "controlled_revisit_constructible_histories": len(rows),
        "original_candidate_histories_referenced": sum(
            int(row["original_candidate_count"]) for row in rows
        ),
        "expansion_constructible_recipients": sum(
            bool(row["candidates"]) for row in rows
        ),
        "expansion_candidate_histories": len(candidates),
        "expansion_scene_clusters": len(scenes),
        "status_counts": dict(sorted(statuses.items())),
        "direction_strata": dict(sorted(strata.items())),
        "candidate_max_online_A_covis": {
            "minimum": min(covis) if covis else None,
            "median": statistics.median(covis) if covis else None,
            "maximum": max(covis) if covis else None,
        },
        "construction_contract": {
            "slot_start_inclusive": SLOT_START,
            "slot_stop_exclusive": SLOT_STOP,
            "maximum_new_candidates_per_recipient": (
                MAXIMUM_NEW_CANDIDATES_PER_RECIPIENT
            ),
            "legacy_episode_rank_multiplier": LEGACY_EPISODE_RANK_MULTIPLIER,
            "minimum_candidate_planar_separation_m": (
                MINIMUM_CANDIDATE_SEPARATION_M
            ),
            "A_to_B_geodesic_m": [2.0, 9.0],
            "B_to_C_geodesic_m": [2.0, 9.0],
            "B_max_online_A_covis_exclusive": 0.10,
        },
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
    }


def audit_scene(
    *,
    parent_root: Path,
    source_protocol_path: Path,
    expansion_protocol_path: Path,
    construction_root: Path,
    original_manifest_path: Path,
    scene_index: int,
    out: Path,
) -> dict[str, Any]:
    source_protocol = load_protocol(source_protocol_path)
    expansion_protocol = load_expansion_protocol(expansion_protocol_path)
    sources = expansion_protocol["frozen_sources"]
    require(sha256_file(source_protocol_path)
            == sources["source_construction_protocol_sha256"],
            "source construction protocol changed")
    require(parent_root.resolve() == Path(sources["parent_root"]).resolve(),
            "parent root changed")
    require(construction_root.resolve()
            == Path(sources["source_construction_root"]).resolve(),
            "source construction root changed")
    require(original_manifest_path.resolve()
            == Path(sources["original_v4_manifest"]).resolve(),
            "original v4 manifest path changed")
    require(sha256_file(original_manifest_path)
            == sources["original_v4_manifest_sha256"],
            "original v4 manifest changed")

    parent_paths = bind_parent(source_protocol, parent_root)
    parent = json.loads(parent_paths["manifest"].read_text())
    require(0 <= scene_index < len(parent["scenes"]),
            "scene index out of range")
    scene = str(parent["scenes"][scene_index])
    original_manifest = json.loads(original_manifest_path.read_text())
    original_by_recipient = _original_positions(original_manifest, scene)

    fragment = construction_root / f"{scene_index:02d}_{scene}"
    completion_path = fragment / "completion.json"
    completion = json.loads(completion_path.read_text())
    require(completion.get("query_policy_outcomes_read") is False,
            "sealed construction read query outcomes")
    require(completion.get("protocol_sha256") == sha256_file(source_protocol_path),
            "sealed construction protocol changed")
    online_root = (
        parent_root / "construction" / "scenes"
        / f"{scene_index:02d}_{scene}" / "online_a"
    )
    recipients: list[dict[str, Any]] = []
    if int(completion["materialized_A_histories"]) > 0:
        _manifest, histories = load_histories(online_root, scene)
        attempts = {str(row["episode"]): row for row in completion["attempts"]}
        require(len(attempts) == len(histories), "sealed attempt count changed")
        asset = Path(parent["assets"][scene]["glb_path"])
        require(sha256_file(asset) == parent["assets"][scene]["glb_sha256"],
                "scene asset changed")
        simulator = make_sim(str(asset), "", agent_radius=0.30)
        try:
            for history in histories:
                episode = str(history["receipt"]["episode"])
                attempt = attempts[episode]
                if not bool(attempt["revisit_A_constructible"]):
                    continue
                revisit_position = reconstruct_revisit_position(
                    simulator,
                    history,
                    scene=scene,
                    episode=episode,
                    selected=attempt["selected_revisit_A"],
                )
                original = list(original_by_recipient.get(episode, []))
                positions = [
                    np.asarray(row["goal_floor_position"], dtype=np.float64)
                    for row in original
                ]
                candidates = []
                slot_attempts = []
                for slot in range(SLOT_START, SLOT_STOP):
                    identity = f"{episode}__natural_b_{slot:02d}"
                    if len(candidates) >= MAXIMUM_NEW_CANDIDATES_PER_RECIPIENT:
                        slot_attempts.append({
                            "slot": slot,
                            "identity": identity,
                            "status": "not_attempted_after_recipient_cap",
                            "sampling_diagnostics": None,
                            "error": None,
                        })
                        continue
                    try:
                        candidate, diagnostics = sample_natural_novel(
                            simulator,
                            history,
                            scene=scene,
                            episode=identity,
                            scene_rank=scene_index,
                            episode_rank=(
                                int(history["episode_rank"])
                                * LEGACY_EPISODE_RANK_MULTIPLIER + slot
                            ),
                            paired_revisit_position=revisit_position,
                            camera_height=float(
                                history["receipt"]["camera_height_m"]
                            ),
                            minimum_paired_distance_m=2.0,
                            maximum_paired_distance_m=9.0,
                            separated_from_positions=positions,
                            minimum_candidate_separation_m=(
                                MINIMUM_CANDIDATE_SEPARATION_M
                            ),
                        )
                        positions.append(np.asarray(
                            candidate["_position"], dtype=np.float64
                        ))
                        serialized = _candidate_json(candidate)
                        serialized["candidate_slot"] = slot
                        serialized["candidate_identity"] = identity
                        serialized["goal_floor_position"] = [
                            float(value) for value in candidate["_position"]
                        ]
                        candidates.append(serialized)
                        slot_attempts.append({
                            "slot": slot,
                            "identity": identity,
                            "status": "constructible",
                            "sampling_diagnostics": diagnostics,
                            "error": None,
                        })
                    except NaturalNovelConstructionError as exc:
                        slot_attempts.append({
                            "slot": slot,
                            "identity": identity,
                            "status": "no_natural_B_candidate",
                            "sampling_diagnostics": exc.diagnostics,
                            "error": str(exc),
                        })
                recipients.append({
                    "scene": scene,
                    "episode": episode,
                    "recipient_episode_rank": int(history["episode_rank"]),
                    "status": (
                        "constructible" if candidates
                        else "no_additional_natural_B_candidate"
                    ),
                    "original_candidate_count": len(original),
                    "original_candidates": original,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "candidate_slot_attempts": slot_attempts,
                })
        finally:
            simulator.close()

    payload = {
        "schema_version": SCHEMA,
        "scope": "result-blind additional Natural-B audit",
        "scene": scene,
        "scene_index": int(scene_index),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "expansion_protocol_sha256": sha256_file(expansion_protocol_path),
        "original_v4_manifest_sha256": sha256_file(original_manifest_path),
        "construction_completion_sha256": sha256_file(completion_path),
        "source_materialized_A_histories": int(
            completion["materialized_A_histories"]
        ),
        "controlled_revisit_constructible_histories": len(recipients),
        "original_candidate_histories_referenced": sum(
            int(row["original_candidate_count"]) for row in recipients
        ),
        "expansion_constructible_recipients": sum(
            bool(row["candidates"]) for row in recipients
        ),
        "expansion_candidate_histories": sum(
            int(row["candidate_count"]) for row in recipients
        ),
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
        "recipients": recipients,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), f"expansion audit output exists: {out}")
    out.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    out.with_name(out.name + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    scene_parser = modes.add_parser("scene")
    scene_parser.add_argument("--parent-root", type=Path, required=True)
    scene_parser.add_argument("--source-protocol", type=Path, required=True)
    scene_parser.add_argument("--expansion-protocol", type=Path, required=True)
    scene_parser.add_argument("--construction-root", type=Path, required=True)
    scene_parser.add_argument("--original-manifest", type=Path, required=True)
    scene_parser.add_argument("--scene-index", type=int, required=True)
    scene_parser.add_argument("--out", type=Path, required=True)
    aggregate_parser = modes.add_parser("aggregate")
    aggregate_parser.add_argument("--audit-root", type=Path, required=True)
    aggregate_parser.add_argument("--expected-scenes", type=int, required=True)
    aggregate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "scene":
        result = audit_scene(
            parent_root=args.parent_root,
            source_protocol_path=args.source_protocol,
            expansion_protocol_path=args.expansion_protocol,
            construction_root=args.construction_root,
            original_manifest_path=args.original_manifest,
            scene_index=args.scene_index,
            out=args.out,
        )
        print(json.dumps({
            "scene": result["scene"],
            "controlled_revisit": result[
                "controlled_revisit_constructible_histories"
            ],
            "original_candidates": result[
                "original_candidate_histories_referenced"
            ],
            "expansion_recipients": result[
                "expansion_constructible_recipients"
            ],
            "expansion_candidates": result["expansion_candidate_histories"],
        }, sort_keys=True))
        return

    paths = sorted(args.audit_root.glob("*/natural_b_expansion_audit.json"))
    require(len(paths) == args.expected_scenes,
            "expansion audit fragment count changed")
    result = aggregate_payloads([json.loads(path.read_text()) for path in paths])
    require(not args.out.exists(), f"aggregate output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
