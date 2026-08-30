#!/usr/bin/env python3
"""Materialize the sealed result-blind Natural-B expansion ledger.

The expansion audit fixed slots 4--15 while retaining at most two candidates
per actual-online A history.  This program deterministically reconstructs each
accepted candidate and the already sealed controlled-Revisit target.  It does
not run or inspect factual-B, Leg-3, B2, or C2 policy outcomes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from build_final14_role_pair_scene import (
    NaturalNovelConstructionError,
    _candidate_json,
    role_contract,
    sample_natural_novel,
    write_manifest,
    write_protocol_episode,
)
from construct_hm3d_fullmono_lifelong_ab import load_histories
from generate_twoleg import make_sim
from hm3d_fullmono_lifelong import (
    NATURAL_B_EXPANSION_EXECUTION_PROTOCOL_SCHEMA,
    bind_parent,
    load_protocol,
    require,
    sha256_file,
)
from materialize_hm3d_fullmono_lifelong_natural_ab import (
    equivalent,
    reconstruct_revisit_candidate,
    verify_sha_sidecar,
)


SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_scene_v1_20260830"
)
AUDIT_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_audit_v1_20260828"
)


def require_expansion_separation(
    original_positions: list[list[float]],
    materialized: list[dict[str, Any]],
    minimum_m: float,
) -> None:
    """Recompute separation against original and earlier expansion goals."""

    prior = [np.asarray(row, dtype=np.float64) for row in original_positions]
    require(all(row.shape == (3,) and np.all(np.isfinite(row)) for row in prior),
            "original candidate position is invalid")
    for candidate in materialized:
        current = np.asarray(candidate["_position"], dtype=np.float64)
        require(current.shape == (3,) and np.all(np.isfinite(current)),
                "expansion candidate position is invalid")
        for previous in prior:
            distance = float(np.linalg.norm(
                current[[0, 2]] - previous[[0, 2]]
            ))
            require(distance >= minimum_m - 1e-9,
                    "expansion candidate violates frozen planar separation")
        prior.append(current)


def _audit_candidate_without_receipt_fields(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for key in ("candidate_slot", "candidate_identity", "goal_floor_position"):
        payload.pop(key)
    return payload


def materialize_scene(
    *,
    parent_root: Path,
    protocol_path: Path,
    audit_run_root: Path,
    source_construction_root: Path,
    source_protocol_path: Path,
    scene_index: int,
    out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    require(protocol["schema_version"]
            == NATURAL_B_EXPANSION_EXECUTION_PROTOCOL_SCHEMA,
            "materializer requires the frozen v5 expansion protocol")
    audit_contract = protocol["sealed_natural_b_expansion_audit"]
    source_contract = protocol["source_controlled_revisit"]
    construction = protocol["novel_b_construction"]
    require(audit_run_root.resolve()
            == Path(audit_contract["run_root"]).resolve(),
            "expansion audit root changed")
    require(source_construction_root.resolve()
            == Path(source_contract["construction_root"]).resolve(),
            "controlled-Revisit construction root changed")
    require(source_protocol_path.resolve()
            == Path(source_contract["protocol"]).resolve(),
            "controlled-Revisit protocol path changed")
    require(sha256_file(source_protocol_path)
            == source_contract["protocol_sha256"],
            "controlled-Revisit protocol changed")

    summary_path = audit_run_root / audit_contract["summary"]
    verifier_path = audit_run_root / audit_contract["independent_verification"]
    require(sha256_file(summary_path) == audit_contract["summary_sha256"],
            "sealed expansion summary changed")
    require(sha256_file(verifier_path)
            == audit_contract["independent_verification_sha256"],
            "sealed expansion verifier changed")
    verifier = json.loads(verifier_path.read_text())
    require(verifier.get("verified") is True
            and verifier.get("navigation_outcomes_read") is False
            and verifier.get("query_policy_outcomes_read") is False,
            "expansion audit was not independently verified")

    original_manifest_path = Path(audit_contract["original_v4_manifest"])
    require(sha256_file(original_manifest_path)
            == audit_contract["original_v4_manifest_sha256"],
            "original v4 candidate manifest changed")
    parent_paths = bind_parent(protocol, parent_root)
    parent = json.loads(parent_paths["manifest"].read_text())
    require(0 <= scene_index < len(parent["scenes"]),
            "scene index outside parent manifest")
    scene = str(parent["scenes"][scene_index])

    fragment_path = (
        audit_run_root / audit_contract["scene_fragments"]
        / f"{scene_index:02d}_{scene}" / "natural_b_expansion_audit.json"
    )
    verify_sha_sidecar(fragment_path)
    audit = json.loads(fragment_path.read_text())
    require(audit.get("schema_version") == AUDIT_SCHEMA,
            "expansion audit fragment schema changed")
    require(audit["scene"] == scene
            and int(audit["scene_index"]) == scene_index,
            "expansion audit fragment identity changed")
    require(audit.get("query_policy_outcomes_read") is False
            and audit.get("navigation_outcomes_read") is False
            and audit.get("evaluation_authorized") is False,
            "expansion audit fragment crossed the outcome boundary")
    require(audit["expansion_protocol_sha256"]
            == protocol["amendment_of"]["protocol_sha256"],
            "expansion audit protocol binding changed")
    require(audit["original_v4_manifest_sha256"]
            == audit_contract["original_v4_manifest_sha256"],
            "expansion fragment references another original ledger")

    source_fragment = source_construction_root / f"{scene_index:02d}_{scene}"
    source_completion_path = source_fragment / "completion.json"
    source_completion = json.loads(source_completion_path.read_text())
    require(sha256_file(source_completion_path)
            == audit["construction_completion_sha256"],
            "controlled-Revisit construction receipt changed")
    require(source_completion.get("protocol_sha256")
            == source_contract["protocol_sha256"],
            "controlled-Revisit completion protocol changed")
    attempts = {
        str(row["episode"]): row for row in source_completion["attempts"]
    }
    online_root = (
        parent_root / "construction" / "scenes"
        / f"{scene_index:02d}_{scene}" / "online_a"
    )
    histories: list[dict[str, Any]] = []
    if online_root.is_dir():
        _manifest, histories = load_histories(online_root, scene)
    by_episode = {
        str(history["receipt"]["episode"]): history for history in histories
    }
    require(int(audit["source_materialized_A_histories"]) == len(histories),
            "source actual-online A history count changed")
    require(not out.exists(), f"expansion materialization exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))

    accepted: list[dict[str, Any]] = []
    materialized_recipients = 0
    simulator = None
    slot_start = int(construction["slot_start_inclusive"])
    slot_stop = int(construction["slot_stop_exclusive"])
    maximum_new = int(construction["maximum_new_candidates_per_recipient"])
    episode_rank_multiplier = int(construction["legacy_episode_rank_multiplier"])
    minimum_separation = float(
        construction["minimum_candidate_planar_separation_m"]
    )
    try:
        if audit["recipients"]:
            asset = Path(parent["assets"][scene]["glb_path"])
            require(sha256_file(asset) == parent["assets"][scene]["glb_sha256"],
                    "scene asset changed")
            simulator = make_sim(str(asset), "", agent_radius=0.30)
        for recipient in audit["recipients"]:
            episode = str(recipient["episode"])
            require(episode in by_episode and episode in attempts,
                    "expansion recipient disappeared from source history")
            history = by_episode[episode]
            source_attempt = attempts[episode]
            require(source_attempt["revisit_A_constructible"] is True,
                    "expansion recipient has no controlled Revisit")
            revisit = reconstruct_revisit_candidate(
                simulator,
                history,
                scene=scene,
                episode=episode,
                selected=source_attempt["selected_revisit_A"],
                camera_height=float(history["receipt"]["camera_height_m"]),
            )
            original_positions = [
                [float(value) for value in row["goal_floor_position"]]
                for row in recipient["original_candidates"]
            ]
            positions = [np.asarray(row, dtype=np.float64)
                         for row in original_positions]
            audit_by_slot = {
                int(row["candidate_slot"]): row
                for row in recipient["candidates"]
            }
            attempt_by_slot = {
                int(row["slot"]): row
                for row in recipient["candidate_slot_attempts"]
            }
            require(set(attempt_by_slot) == set(range(slot_start, slot_stop)),
                    "expansion slot ledger changed")
            materialized: list[dict[str, Any]] = []
            for slot in range(slot_start, slot_stop):
                identity = f"{episode}__natural_b_{slot:02d}"
                observed = attempt_by_slot[slot]
                require(observed["identity"] == identity,
                        "expansion candidate identity changed")
                if len(materialized) >= maximum_new:
                    require(observed["status"]
                            == "not_attempted_after_recipient_cap"
                            and observed["sampling_diagnostics"] is None,
                            "post-cap expansion attempt changed")
                    continue
                try:
                    natural, diagnostics = sample_natural_novel(
                        simulator,
                        history,
                        scene=scene,
                        episode=identity,
                        scene_rank=scene_index,
                        episode_rank=(
                            int(history["episode_rank"])
                            * episode_rank_multiplier + slot
                        ),
                        paired_revisit_position=revisit["_position"],
                        camera_height=float(
                            history["receipt"]["camera_height_m"]
                        ),
                        minimum_paired_distance_m=2.0,
                        maximum_paired_distance_m=9.0,
                        separated_from_positions=positions,
                        minimum_candidate_separation_m=minimum_separation,
                    )
                    require(observed["status"] == "constructible",
                            "expansion audit/materialization status changed")
                    require(slot in audit_by_slot,
                            "materialized expansion candidate absent from audit")
                    equivalent(
                        _candidate_json(natural),
                        _audit_candidate_without_receipt_fields(
                            audit_by_slot[slot]
                        ),
                        path=f"{scene}/{identity}",
                    )
                    equivalent(
                        diagnostics,
                        observed["sampling_diagnostics"],
                        path=f"{scene}/{identity}.diagnostics",
                    )
                    positions.append(np.asarray(
                        natural["_position"], dtype=np.float64
                    ))
                    natural["_materialization_slot"] = slot
                    materialized.append(natural)
                except NaturalNovelConstructionError as error:
                    require(observed["status"] == "no_natural_B_candidate",
                            "expansion rejection status changed")
                    require(slot not in audit_by_slot,
                            "expansion audit contains rejected candidate")
                    equivalent(
                        error.diagnostics,
                        observed["sampling_diagnostics"],
                        path=f"{scene}/{identity}.diagnostics",
                    )

            require(len(materialized) == int(recipient["candidate_count"]),
                    "expansion recipient candidate count changed")
            require_expansion_separation(
                original_positions, materialized, minimum_separation
            )
            if materialized:
                materialized_recipients += 1
            for natural in materialized:
                slot = int(natural["_materialization_slot"])
                synthetic_episode = f"{episode}__natural_b_{slot:02d}"
                destination = (
                    temporary / "role_pairs" / scene / synthetic_episode
                )
                payload = write_protocol_episode(
                    destination=destination,
                    online_episode=history["root"],
                    receipt=history["receipt"],
                    history=history,
                    natural=natural,
                    revisit=revisit,
                    protocol="hm3d_fullmono_lifelong_natural_b_expansion_v5",
                    scene_rank=scene_index,
                    episode_rank=int(history["episode_rank"]),
                )
                payload["episode"] = synthetic_episode
                payload["lifelong_construction"] = {
                    "recipient_episode": episode,
                    "synthetic_candidate_episode": synthetic_episode,
                    "candidate_slot": slot,
                    "candidate_identity": synthetic_episode,
                    "candidate_source": "sealed_natural_B_expansion_audit",
                    "goal_floor_position": [
                        float(value) for value in natural["_position"]
                    ],
                    "B_max_recipient_A_covis": float(
                        natural["max_online_a_covis"]
                    ),
                    "B_to_C_geodesic_m": float(
                        natural["paired_revisit_separation_m"]
                    ),
                    "assigned_direction_stratum": natural[
                        "assigned_direction_stratum"
                    ],
                    "audit_fragment_sha256": sha256_file(fragment_path),
                    "query_policy_outcomes_read": False,
                    "navigation_outcomes_read": False,
                }
                sidecar = destination / "role_pairs.json"
                sidecar_payload = dict(payload)
                sidecar_payload.pop("role_pairs_sha256", None)
                sidecar.write_text(json.dumps(
                    sidecar_payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n")
                payload = sidecar_payload
                payload["role_pairs_sha256"] = sha256_file(sidecar)
                accepted.append(payload)

        require(len(accepted) == int(audit["expansion_candidate_histories"]),
                "scene expansion total differs from sealed audit")
        require(materialized_recipients
                == int(audit["expansion_constructible_recipients"]),
                "scene expansion recipient total differs from sealed audit")
        role_root = temporary / "role_pairs"
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "sealed Natural-B expansion plus controlled Revisit-A assets "
                "for the conference continual-memory population"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": (
                sha256_file(online_root / "manifest.json")
                if online_root.is_dir() else None
            ),
            "construction_seed": 20260830,
            "contract": {
                **role_contract(support="standard"),
                "lifelong_direct_natural_B_to_C_geodesic_m": [2.0, 9.0],
                "lifelong_candidate_planar_separation_m": 2.0,
                "lifelong_expansion_slot_range": [slot_start, slot_stop],
            },
            "episodes": accepted,
        }
        if accepted:
            write_manifest(role_root, manifest)
        else:
            role_root.mkdir(parents=True, exist_ok=True)
            manifest_path = role_root / "manifest.json"
            manifest_path.write_text(json.dumps(
                manifest, indent=2, sort_keys=True, allow_nan=False
            ) + "\n")
            (role_root / "manifest.json.sha256").write_text(
                sha256_file(manifest_path) + "  manifest.json\n"
            )
        completion = {
            "schema_version": SCHEMA,
            "status": "complete",
            "scene": scene,
            "scene_index": scene_index,
            "protocol_sha256": sha256_file(protocol_path),
            "audit_fragment_sha256": sha256_file(fragment_path),
            "source_construction_completion_sha256": sha256_file(
                source_completion_path
            ),
            "materialized_A_histories": len(histories),
            "attempted_histories": len(audit["recipients"]),
            "materialized_recipient_histories": materialized_recipients,
            "constructible_AB_C_histories": len(accepted),
            "audit_candidate_count_reproduced": True,
            "candidate_positions_serialized": True,
            "separation_against_original_and_expansion_recomputed": True,
            "query_policy_outcomes_read": False,
            "navigation_outcomes_read": False,
            "role_pair_manifest_sha256": sha256_file(
                role_root / "manifest.json"
            ),
        }
        completion_path = temporary / "completion.json"
        completion_path.write_text(json.dumps(
            completion, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "completion.json.sha256").write_text(
            sha256_file(completion_path) + "  completion.json\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if simulator is not None:
            simulator.close()
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-run-root", type=Path, required=True)
    parser.add_argument("--source-construction-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_scene(
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        audit_run_root=args.audit_run_root,
        source_construction_root=args.source_construction_root,
        source_protocol_path=args.source_protocol,
        scene_index=args.scene_index,
        out=args.out,
    )
    print(json.dumps({
        "scene": result["scene"],
        "materialized_recipients": result[
            "materialized_recipient_histories"
        ],
        "materialized_candidates": result["constructible_AB_C_histories"],
        "navigation_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
