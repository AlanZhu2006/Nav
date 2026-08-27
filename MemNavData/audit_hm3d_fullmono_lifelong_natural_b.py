#!/usr/bin/env python3
"""Result-blind feasibility audit for independently rendered Novel-B goals.

This audit asks whether the lifelong benchmark's Novel-B target needs to be
borrowed from another successful online-A trace.  It reuses each sealed v3
actual-online A history and its already frozen controlled Revisit-C pose, then
constructs one deterministic natural Novel goal directly on the same HM3D
navmesh.  It never executes or reads B, C, or B2 navigation outcomes.

The output is an audit only.  It does not authorize a new evaluation or mutate
the sealed v3 population.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

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
from hm3d_fullmono_lifelong import (
    bind_parent,
    load_protocol,
    require,
    sha256_file,
)


SCHEMA = "hm3d_fullmono_lifelong_natural_b_audit_v1_20260827"
MAXIMUM_CANDIDATES_PER_RECIPIENT = 4
MINIMUM_CANDIDATE_SEPARATION_M = 2.0
REFERENCE_MINIMUM_CANDIDATES = 96
REFERENCE_MINIMUM_SCENES = 15


def aggregate_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    require(payloads, "natural-B audit has no scene payloads")
    rows = []
    for payload in payloads:
        require(payload.get("schema_version") == SCHEMA,
                "natural-B audit schema changed")
        require(payload.get("query_policy_outcomes_read") is False,
                "natural-B audit read query policy outcomes")
        require(payload.get("navigation_outcomes_read") is False,
                "natural-B audit read navigation outcomes")
        rows.extend(payload["recipients"])
    reasons = collections.Counter(str(row["status"]) for row in rows)
    constructed = [row for row in rows if row["status"] == "constructible"]
    candidates = [
        candidate
        for row in constructed
        for candidate in row["candidates"]
    ]
    scenes = {str(row["scene"]) for row in constructed}
    strata = collections.Counter(
        str(candidate["assigned_direction_stratum"])
        for candidate in candidates
    )
    return {
        "schema_version": SCHEMA,
        "scope": (
            "result-blind direct natural Novel-B constructibility audit; "
            "not an evaluation authorization"
        ),
        "scene_fragments": len(payloads),
        "source_materialized_A_histories": sum(
            int(payload["source_materialized_A_histories"])
            for payload in payloads
        ),
        "controlled_revisit_constructible_histories": len(rows),
        "natural_B_constructible_recipients": len(constructed),
        "natural_B_candidate_histories": len(candidates),
        "natural_B_constructible_scene_clusters": len(scenes),
        "status_counts": dict(sorted(reasons.items())),
        "direction_strata": dict(sorted(strata.items())),
        "candidate_max_online_A_covis": {
            "minimum": min(
                float(candidate["max_online_a_covis"])
                for candidate in candidates
            ) if candidates else None,
            "median": statistics.median(
                float(candidate["max_online_a_covis"])
                for candidate in candidates
            ) if candidates else None,
            "maximum": max(
                float(candidate["max_online_a_covis"])
                for candidate in candidates
            ) if candidates else None,
        },
        "construction_contract": {
            "maximum_candidates_per_controlled_revisit_history": (
                MAXIMUM_CANDIDATES_PER_RECIPIENT
            ),
            "minimum_candidate_planar_separation_m": (
                MINIMUM_CANDIDATE_SEPARATION_M
            ),
            "A_to_B_geodesic_m": [2.0, 9.0],
            "B_to_C_geodesic_m": [2.0, 9.0],
            "B_max_online_A_covis_exclusive": 0.10,
            "same_scene_navmesh": True,
            "goal_rendered_at_frozen_camera_height": True,
            "cross_online_history_donor_required": False,
        },
        "v3_source_gate_reference": {
            "minimum_candidate_histories": REFERENCE_MINIMUM_CANDIDATES,
            "minimum_scene_clusters": REFERENCE_MINIMUM_SCENES,
            "met": (
                len(candidates) >= REFERENCE_MINIMUM_CANDIDATES
                and len(scenes) >= REFERENCE_MINIMUM_SCENES
            ),
            "evaluation_authority_conferred": False,
        },
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
    }


def audit_scene(
    *,
    parent_root: Path,
    protocol_path: Path,
    construction_root: Path,
    scene_index: int,
    out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    parent_paths = bind_parent(protocol, parent_root)
    parent = json.loads(parent_paths["manifest"].read_text())
    require(0 <= scene_index < len(parent["scenes"]),
            "scene index out of range")
    scene = str(parent["scenes"][scene_index])
    fragment = construction_root / f"{scene_index:02d}_{scene}"
    completion_path = fragment / "completion.json"
    completion = json.loads(completion_path.read_text())
    require(completion.get("query_policy_outcomes_read") is False,
            "sealed construction read query outcomes")
    require(completion.get("protocol_sha256") == sha256_file(protocol_path),
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
                candidates = []
                candidate_positions = []
                slot_attempts = []
                for slot in range(MAXIMUM_CANDIDATES_PER_RECIPIENT):
                    identity = f"{episode}__natural_b_{slot:02d}"
                    try:
                        candidate, diagnostics = sample_natural_novel(
                            simulator,
                            history,
                            scene=scene,
                            episode=identity,
                            scene_rank=scene_index,
                            episode_rank=(
                                int(history["episode_rank"])
                                * MAXIMUM_CANDIDATES_PER_RECIPIENT + slot
                            ),
                            paired_revisit_position=revisit_position,
                            camera_height=float(
                                history["receipt"]["camera_height_m"]),
                            minimum_paired_distance_m=2.0,
                            maximum_paired_distance_m=9.0,
                            separated_from_positions=candidate_positions,
                            minimum_candidate_separation_m=(
                                MINIMUM_CANDIDATE_SEPARATION_M
                            ),
                        )
                        candidate_positions.append(candidate["_position"])
                        candidate_json = _candidate_json(candidate)
                        candidate_json["candidate_slot"] = slot
                        candidate_json["candidate_identity"] = identity
                        candidates.append(candidate_json)
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
                status = (
                    "constructible" if candidates
                    else "no_natural_B_candidate"
                )
                recipients.append({
                    "scene": scene,
                    "episode": episode,
                    "recipient_episode_rank": int(history["episode_rank"]),
                    "status": status,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "candidate_slot_attempts": slot_attempts,
                })
        finally:
            simulator.close()

    payload = {
        "schema_version": SCHEMA,
        "scope": "result-blind independently rendered Novel-B audit",
        "scene": scene,
        "scene_index": int(scene_index),
        "protocol_sha256": sha256_file(protocol_path),
        "construction_completion_sha256": sha256_file(completion_path),
        "source_materialized_A_histories": int(
            completion["materialized_A_histories"]),
        "controlled_revisit_constructible_histories": len(recipients),
        "natural_B_constructible_recipients": sum(
            row["status"] == "constructible" for row in recipients
        ),
        "natural_B_candidate_histories": sum(
            int(row["candidate_count"]) for row in recipients
        ),
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "evaluation_authorized": False,
        "recipients": recipients,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), f"audit output exists: {out}")
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
    scene_parser.add_argument("--protocol", type=Path, required=True)
    scene_parser.add_argument("--construction-root", type=Path, required=True)
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
            protocol_path=args.protocol,
            construction_root=args.construction_root,
            scene_index=args.scene_index,
            out=args.out,
        )
        print(json.dumps({
            "scene": result["scene"],
            "controlled_revisit": result[
                "controlled_revisit_constructible_histories"],
            "natural_B_recipients": result[
                "natural_B_constructible_recipients"],
            "natural_B_candidates": result["natural_B_candidate_histories"],
        }, sort_keys=True))
        return

    paths = sorted(args.audit_root.glob("*/natural_b_audit.json"))
    require(len(paths) == args.expected_scenes,
            "natural-B audit fragment count changed")
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
