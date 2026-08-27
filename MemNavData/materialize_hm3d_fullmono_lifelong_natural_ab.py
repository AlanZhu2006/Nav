#!/usr/bin/env python3
"""Materialize the sealed direct-Natural-B audit ledger as A/B/C assets.

The script reads every candidate identity from the completed result-blind audit,
deterministically reconstructs its rendered goal, and pairs it with the already
sealed controlled Revisit-A target.  It never runs or reads a B/C/B2/C2 policy.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import build_shared_online_double_revisit as history_tools
import build_shared_online_role_pairs as pair_tools
from build_final14_role_pair_scene import (
    DEPTH_TOLERANCE_M,
    ELIGIBLE_FRAME_FLOOR,
    NaturalNovelConstructionError,
    _candidate_json,
    deterministic_pose_grid,
    role_contract,
    sample_natural_novel,
    write_manifest,
    write_protocol_episode,
)
from construct_hm3d_fullmono_lifelong_ab import load_histories
from generate_twoleg import covis_curve, covis_frac, make_sim, render
from hm3d_fullmono_lifelong import (
    DIRECT_NATURAL_PROTOCOL_SCHEMA,
    bind_parent,
    load_protocol,
    require,
    sha256_file,
)


SCHEMA = "hm3d_fullmono_lifelong_natural_ab_scene_v1_20260827"
AUDIT_SCHEMA = "hm3d_fullmono_lifelong_natural_b_audit_v1_20260827"
MAXIMUM_CANDIDATES_PER_RECIPIENT = 4
MINIMUM_CANDIDATE_SEPARATION_M = 2.0


def verify_sha_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing sidecar: {sidecar}")
    fields = sidecar.read_text().split()
    require(len(fields) == 2 and fields[0] == sha256_file(path)
            and fields[1] == path.name, f"bad sidecar: {sidecar}")


def equivalent(first: Any, second: Any, *, path: str = "candidate") -> None:
    """Require recursively equivalent serialized metadata with tight float tolerance."""

    if isinstance(first, bool) or isinstance(second, bool):
        require(first is second, f"{path}: boolean changed")
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        require(math.isclose(float(first), float(second), rel_tol=0.0,
                             abs_tol=1e-9), f"{path}: numeric value changed")
        return
    if isinstance(first, dict) and isinstance(second, dict):
        require(set(first) == set(second), f"{path}: keys changed")
        for key in first:
            equivalent(first[key], second[key], path=f"{path}.{key}")
        return
    if isinstance(first, list) and isinstance(second, list):
        require(len(first) == len(second), f"{path}: list length changed")
        for index, (left, right) in enumerate(zip(first, second)):
            equivalent(left, right, path=f"{path}[{index}]")
        return
    require(first == second, f"{path}: value changed")


def require_planar_separation(candidates: list[dict[str, Any]]) -> None:
    for index, first in enumerate(candidates):
        a = np.asarray(first["_position"], dtype=np.float64)
        for second in candidates[:index]:
            b = np.asarray(second["_position"], dtype=np.float64)
            require(
                float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))
                >= MINIMUM_CANDIDATE_SEPARATION_M - 1e-9,
                "materialized candidates violate 2 m planar separation",
            )


def reconstruct_revisit_candidate(
    simulator,
    history: dict[str, Any],
    *,
    scene: str,
    episode: str,
    selected: dict[str, Any],
    camera_height: float,
) -> dict[str, Any]:
    """Render the already selected Revisit pose without rerunning selection."""

    frame = int(selected["source_frame"])
    attempt = int(selected["render_attempt"])
    grid = deterministic_pose_grid(f"{scene}/{episode}/{frame}")
    require(1 <= attempt <= len(grid), "sealed revisit attempt escaped grid")
    radius, direction, yaw_offset = grid[attempt - 1]
    source_floor = np.asarray(history["floor_positions"][frame], dtype=np.float64)
    source_yaw = float(history["poses"][frame]["yaw"])
    raw = source_floor + np.asarray([
        radius * math.cos(direction), 0.0, radius * math.sin(direction)
    ])
    snapped = np.asarray(simulator.pathfinder.snap_point(raw), dtype=np.float64)
    yaw = history_tools.wrap_radians(
        source_yaw + math.radians(yaw_offset)
    )
    camera = snapped + np.asarray([0.0, camera_height, 0.0])
    rgb, depth = render(simulator, camera, yaw)
    translation = float(
        np.linalg.norm(snapped[[0, 2]] - source_floor[[0, 2]])
    )
    pixel_mae = history_tools.pixel_mae(rgb, history["rgbs"][frame])
    query_distance = history_tools.goal_distance(
        simulator.pathfinder,
        np.asarray(history["trace"]["end_position"], dtype=np.float64),
        snapped,
    )
    geometry = pair_tools.query_geometry(
        simulator.pathfinder,
        np.asarray(history["trace"]["end_position"], dtype=np.float64),
        snapped,
    )
    require(geometry is not None, "sealed revisit became unreachable")
    measured_distance, initial_bearing = geometry
    require(abs(measured_distance - query_distance) <= 1e-5,
            "sealed revisit distance changed")
    points = history_tools.goal_world_points(depth, camera, yaw)
    anchor_covis = float(covis_frac(
        points, history["transforms"][frame], history["depths"][frame]
    ))
    curve = covis_curve(
        points, history["transforms"], history["depths"],
        tol=DEPTH_TOLERANCE_M,
    )
    eligible = curve[ELIGIBLE_FRAME_FLOOR:]
    require(len(eligible) > 0, "online history has no eligible revisit frames")
    best_frame = ELIGIBLE_FRAME_FLOOR + int(np.argmax(eligible))
    best_covis = float(curve[best_frame])
    gap = abs(best_frame - frame)
    yaw_delta = history_tools.angle_delta_degrees(yaw, source_yaw)
    target = 0.72
    ranking = (
        abs(best_covis - target),
        abs(float(query_distance) - 3.0),
        -(len(history["poses"]) - 1 - frame),
        frame,
        attempt,
    )
    candidate = {
        "support_band": "standard",
        "source_frame": frame,
        "render_attempt": attempt,
        "translation_m": translation,
        "yaw_delta_deg": yaw_delta,
        "source_anchor_covis": anchor_covis,
        "max_online_a_covis": best_covis,
        "max_online_a_covis_frame": best_frame,
        "argmax_gap_frames": gap,
        "eligible_online_a_frame_floor": ELIGIBLE_FRAME_FLOOR,
        "pixel_mae": pixel_mae,
        "query_geodesic_m": float(query_distance),
        "initial_path_bearing_rad": float(initial_bearing),
        "ranking": list(ranking),
        "_position": snapped,
        "_yaw": float(yaw),
        "_rgb": rgb,
        "_depth": depth,
        "_covis_curve": [float(value) for value in curve],
    }
    equivalent(_candidate_json(candidate), selected, path="sealed_revisit")
    return candidate


def materialize_scene(
    *,
    parent_root: Path,
    protocol_path: Path,
    audit_run_root: Path,
    source_construction_root: Path,
    scene_index: int,
    out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    require(protocol["schema_version"] == DIRECT_NATURAL_PROTOCOL_SCHEMA,
            "materializer requires the frozen direct-Natural v4 protocol")
    audit_contract = protocol["sealed_natural_b_audit"]
    require(audit_run_root.resolve()
            == Path(audit_contract["run_root"]).resolve(),
            "Natural-B audit root changed")
    require(source_construction_root.resolve()
            == Path(audit_contract["source_v3_construction_root"]).resolve(),
            "controlled-Revisit source construction changed")
    summary_path = audit_run_root / audit_contract["summary"]
    verifier_path = audit_run_root / audit_contract["independent_verification"]
    require(sha256_file(summary_path) == audit_contract["summary_sha256"],
            "sealed Natural-B summary changed")
    require(sha256_file(verifier_path)
            == audit_contract["independent_verification_sha256"],
            "sealed Natural-B independent verification changed")
    verifier = json.loads(verifier_path.read_text())
    require(verifier.get("verified") is True
            and verifier.get("reference_gate_met") is True,
            "Natural-B audit was not independently verified")

    parent_paths = bind_parent(protocol, parent_root)
    parent = json.loads(parent_paths["manifest"].read_text())
    require(0 <= scene_index < len(parent["scenes"]),
            "scene index outside parent manifest")
    scene = str(parent["scenes"][scene_index])
    fragment_path = (
        audit_run_root / audit_contract["scene_fragments"]
        / f"{scene_index:02d}_{scene}" / "natural_b_audit.json"
    )
    verify_sha_sidecar(fragment_path)
    audit = json.loads(fragment_path.read_text())
    require(audit.get("schema_version") == AUDIT_SCHEMA,
            "Natural-B audit fragment schema changed")
    require(audit["scene"] == scene
            and int(audit["scene_index"]) == scene_index,
            "Natural-B audit fragment identity changed")
    require(audit.get("query_policy_outcomes_read") is False
            and audit.get("navigation_outcomes_read") is False,
            "Natural-B audit fragment read navigation outcomes")

    source_fragment = source_construction_root / f"{scene_index:02d}_{scene}"
    source_completion_path = source_fragment / "completion.json"
    source_completion = json.loads(source_completion_path.read_text())
    require(sha256_file(source_completion_path)
            == audit["construction_completion_sha256"],
            "controlled-Revisit construction receipt changed")
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
    require(not out.exists(), f"materialization output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    accepted: list[dict[str, Any]] = []
    materialized_recipients = 0
    simulator = None
    try:
        if audit["recipients"]:
            asset = Path(parent["assets"][scene]["glb_path"])
            require(sha256_file(asset) == parent["assets"][scene]["glb_sha256"],
                    "scene asset changed")
            simulator = make_sim(str(asset), "", agent_radius=0.30)
        for recipient in audit["recipients"]:
            episode = str(recipient["episode"])
            require(episode in by_episode and episode in attempts,
                    "audit recipient disappeared from source history")
            history = by_episode[episode]
            source_attempt = attempts[episode]
            require(source_attempt["revisit_A_constructible"] is True,
                    "audit recipient has no controlled Revisit")
            revisit = reconstruct_revisit_candidate(
                simulator, history, scene=scene, episode=episode,
                selected=source_attempt["selected_revisit_A"],
                camera_height=float(history["receipt"]["camera_height_m"]),
            )
            audit_by_slot = {
                int(row["candidate_slot"]): row
                for row in recipient["candidates"]
            }
            attempt_by_slot = {
                int(row["slot"]): row
                for row in recipient["candidate_slot_attempts"]
            }
            require(set(attempt_by_slot)
                    == set(range(MAXIMUM_CANDIDATES_PER_RECIPIENT)),
                    "audit candidate slot ledger changed")
            materialized: list[dict[str, Any]] = []
            for slot in range(MAXIMUM_CANDIDATES_PER_RECIPIENT):
                identity = f"{episode}__natural_b_{slot:02d}"
                observed = attempt_by_slot[slot]
                require(observed["identity"] == identity,
                        "audit candidate identity changed")
                try:
                    natural, diagnostics = sample_natural_novel(
                        simulator,
                        history,
                        scene=scene,
                        episode=identity,
                        scene_rank=scene_index,
                        episode_rank=(
                            int(history["episode_rank"])
                            * MAXIMUM_CANDIDATES_PER_RECIPIENT + slot
                        ),
                        paired_revisit_position=revisit["_position"],
                        camera_height=float(
                            history["receipt"]["camera_height_m"]
                        ),
                        minimum_paired_distance_m=2.0,
                        maximum_paired_distance_m=9.0,
                        separated_from_positions=[
                            row["_position"] for row in materialized
                        ],
                        minimum_candidate_separation_m=(
                            MINIMUM_CANDIDATE_SEPARATION_M
                        ),
                    )
                    require(observed["status"] == "constructible",
                            "audit/materialization candidate status changed")
                    require(slot in audit_by_slot,
                            "materialized candidate absent from audit ledger")
                    sealed = dict(audit_by_slot[slot])
                    sealed.pop("candidate_slot")
                    sealed.pop("candidate_identity")
                    equivalent(_candidate_json(natural), sealed,
                               path=f"{scene}/{identity}")
                    equivalent(diagnostics, observed["sampling_diagnostics"],
                               path=f"{scene}/{identity}.diagnostics")
                    natural["_materialization_slot"] = slot
                    materialized.append(natural)
                except NaturalNovelConstructionError as error:
                    require(observed["status"] == "no_natural_B_candidate",
                            "audit/materialization rejection changed")
                    require(slot not in audit_by_slot,
                            "audit ledger contains rejected candidate")
                    equivalent(error.diagnostics,
                               observed["sampling_diagnostics"],
                               path=f"{scene}/{identity}.diagnostics")
            require(len(materialized) == int(recipient["candidate_count"]),
                    "recipient materialized candidate count changed")
            require_planar_separation(materialized)
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
                    protocol="hm3d_fullmono_lifelong_direct_natural_v4",
                    scene_rank=scene_index,
                    episode_rank=int(history["episode_rank"]),
                )
                payload["episode"] = synthetic_episode
                payload["lifelong_construction"] = {
                    "recipient_episode": episode,
                    "synthetic_candidate_episode": synthetic_episode,
                    "candidate_slot": slot,
                    "candidate_identity": synthetic_episode,
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
                    sidecar_payload, indent=2, sort_keys=True, allow_nan=False
                ) + "\n")
                payload = sidecar_payload
                payload["role_pairs_sha256"] = sha256_file(sidecar)
                accepted.append(payload)

        require(len(accepted) == int(audit["natural_B_candidate_histories"]),
                "scene candidate total differs from sealed audit")
        require(materialized_recipients
                == int(audit["natural_B_constructible_recipients"]),
                "scene recipient total differs from sealed audit")
        role_root = temporary / "role_pairs"
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "sealed direct Natural-B plus controlled Revisit-A assets "
                "for full-monocular five-leg accumulation"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": (
                sha256_file(online_root / "manifest.json")
                if online_root.is_dir() else None
            ),
            "construction_seed": 20260827,
            "contract": {
                **role_contract(support="standard"),
                "lifelong_direct_natural_B_to_C_geodesic_m": [2.0, 9.0],
                "lifelong_candidate_planar_separation_m": 2.0,
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
            "pairwise_separation_recomputed": True,
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
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_scene(
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        audit_run_root=args.audit_run_root,
        source_construction_root=args.source_construction_root,
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
