#!/usr/bin/env python3
"""Build result-blind A->Novel-B plus Revisit-A pairs for one HM3D scene."""

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
    _candidate_json,
    role_contract,
    search_revisit_candidates,
    write_manifest,
    write_protocol_episode,
)
from final14_role_pair_contract import direction_in_stratum, relative_direction_degrees
from generate_twoleg import covis_curve, make_sim
from hm3d_fullmono_lifelong import (
    EXPANSION_PROTOCOL_SCHEMA,
    bind_parent,
    load_protocol,
    require,
    select_donor,
    select_donors,
    sha256_file,
)


SCHEMA = "hm3d_fullmono_lifelong_ab_scene_v1_20260824"
PROTOCOL_NAME = "hm3d_fullmono_lifelong_ab"
DEPTH_TOLERANCE_M = 0.30


def direction_stratum(relative_degrees: float) -> str:
    for name in ("front", "side", "rear"):
        if direction_in_stratum(relative_degrees, name):
            return name
    raise RuntimeError("initial bearing escaped direction strata")


def donor_candidate(
    simulator,
    recipient: dict[str, Any],
    donor: dict[str, Any],
    revisit: dict[str, Any],
    *,
    donor_episode_rank: int,
    donor_frame_index: int | None = None,
    donor_frame_temporal_rank: int = 0,
) -> dict[str, Any] | None:
    """Measure one donor without consulting any navigation result."""

    recipient_endpoint = np.asarray(
        recipient["trace"]["end_position"], dtype=np.float64
    )
    if donor_frame_index is None:
        donor_frame_index = len(donor["poses"]) - 1
    require(0 <= int(donor_frame_index) < len(donor["poses"]),
            "donor frame index escaped the factual trace")
    donor_frame_index = int(donor_frame_index)
    donor_pose = donor["poses"][donor_frame_index]
    donor_floor = np.asarray(
        [donor_pose[axis] for axis in ("x", "y", "z")], dtype=np.float64
    )
    donor_yaw = float(donor_pose["yaw"])
    if abs(float(donor_floor[1] - recipient_endpoint[1])) > 0.20:
        return None
    a_to_b = pair_tools.query_geometry(
        simulator.pathfinder, recipient_endpoint, donor_floor
    )
    if a_to_b is None:
        return None
    a_to_b_distance, initial_bearing = a_to_b
    b_to_c = pair_tools.query_geometry(
        simulator.pathfinder, donor_floor, revisit["_position"]
    )
    if b_to_c is None:
        return None
    b_to_c_distance, _ = b_to_c

    goal_points = history_tools.goal_world_points(
        donor["depths"][donor_frame_index],
        donor["camera_positions"][donor_frame_index], donor_yaw
    )
    curve = covis_curve(
        goal_points,
        recipient["transforms"],
        recipient["depths"],
        tol=DEPTH_TOLERANCE_M,
    )
    maximum = float(curve.max()) if len(curve) else 0.0
    best_frame = int(np.argmax(curve)) if len(curve) else None
    recipient_yaw = float(recipient["trace"]["end_yaw"])
    relative = relative_direction_degrees(initial_bearing, recipient_yaw)
    return {
        "donor_episode": str(donor["receipt"]["episode"]),
        "donor_episode_rank": int(donor_episode_rank),
        "donor_frame_index": donor_frame_index,
        "donor_frame_step": int(donor_pose["step"]),
        "donor_frame_temporal_rank": int(donor_frame_temporal_rank),
        "support_band": "unsupported_novel",
        "query_geodesic_m": float(a_to_b_distance),
        "a_to_b_geodesic_m": float(a_to_b_distance),
        "b_to_c_geodesic_m": float(b_to_c_distance),
        "initial_path_bearing_rad": float(initial_bearing),
        "initial_path_direction_relative_to_a_end_deg": float(relative),
        "assigned_direction_stratum": direction_stratum(relative),
        "max_online_a_covis": maximum,
        "max_recipient_a_covis": maximum,
        "max_online_a_covis_frame": best_frame,
        "eligible_online_a_frame_floor": 0,
        "paired_revisit_separation_m": float(b_to_c_distance),
        "goal_world_yaw_bin": int(
            round((donor_yaw % (2.0 * math.pi)) / (math.pi / 4.0)) % 8
        ),
        "goal_yaw_contract": "donor_factual_temporal_observation_yaw",
        "sampling_seed": 0,
        "accepted_proposal_mode": "same_scene_cross_history_factual_temporal_frame",
        "sampling_diagnostics": {
            "attempts": 1,
            "donor_episode": str(donor["receipt"]["episode"]),
            "donor_frame_index": donor_frame_index,
            "query_policy_outcomes_read": False,
        },
        "_position": donor_floor,
        "_yaw": donor_yaw,
        "_rgb": donor["rgbs"][donor_frame_index],
        "_depth": donor["depths"][donor_frame_index],
        "_covis_curve": [float(value) for value in curve],
    }


def temporal_frame_indices(length: int, samples: int) -> list[int]:
    require(length > 0 and samples > 0, "temporal sampling inputs are invalid")
    return sorted({
        int(round(float(value)))
        for value in np.linspace(0, length - 1, min(length, samples))
    })


def load_histories(online_root: Path, scene: str) -> tuple[dict, list[dict]]:
    manifest_path = online_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == "shared_online_a_materialized_v1",
            "online-A manifest schema changed")
    rows = [row for row in manifest["episodes"] if str(row["scene"]) == scene]
    histories = []
    for rank, row in enumerate(rows):
        episode = str(row["episode"])
        root = online_root / scene / episode
        receipt = json.loads((root / "receipt.json").read_text())
        history = history_tools.load_online_history(root, receipt)
        history.update({"root": root, "receipt": receipt, "episode_rank": rank})
        histories.append(history)
    return manifest, histories


def write_upstream_empty_scene(
    *,
    parent_root: Path,
    protocol_path: Path,
    parent_manifest_path: Path,
    parent_population_path: Path,
    population_fragment: dict[str, Any],
    scene: str,
    scene_index: int,
    online_root: Path,
    out: Path,
) -> dict[str, Any]:
    """Seal a parent-certified zero-history scene as explicit attrition.

    The frozen parent population includes every source scene, including scenes
    whose fixed-attempt Goal-A generation produced no online history.  Those
    scenes legitimately have no ``online_a`` directory.  They contribute zero
    rows; they must not abort the result-blind population scan or be silently
    dropped without a hash-bound receipt.
    """

    require(not online_root.exists(),
            f"{scene}: non-directory online-A path exists")
    require(int(population_fragment["scene_index"]) == int(scene_index),
            f"{scene}: parent population scene index changed")
    require(str(population_fragment["scene"]) == scene,
            f"{scene}: parent population scene identity changed")
    require(int(population_fragment["materialized_histories"]) == 0,
            f"{scene}: missing online-A root despite materialized histories")
    require(int(population_fragment["goal_a_successes"]) == 0,
            f"{scene}: missing online-A root despite Goal-A success")
    require(int(population_fragment["retained_histories"]) == 0,
            f"{scene}: missing online-A root despite retained histories")

    parent_scene_root = (
        parent_root / "construction" / "scenes"
        / f"{scene_index:02d}_{scene}"
    )
    parent_completion_path = parent_scene_root / "completion.json"
    parent_completion_sidecar = parent_scene_root / "completion.json.sha256"
    require(parent_completion_path.is_file()
            and parent_completion_sidecar.is_file(),
            f"{scene}: upstream completion receipt missing")
    expected_parent_hash = str(
        population_fragment["construction_completion_sha256"]
    )
    require(sha256_file(parent_completion_path) == expected_parent_hash,
            f"{scene}: upstream completion changed")
    sidecar_tokens = parent_completion_sidecar.read_text().strip().split()
    require(len(sidecar_tokens) == 2
            and sidecar_tokens[0] == expected_parent_hash
            and sidecar_tokens[1] == "completion.json",
            f"{scene}: upstream completion sidecar changed")
    upstream = json.loads(parent_completion_path.read_text())
    require(upstream.get("status") == "complete",
            f"{scene}: upstream scene is not complete")
    require(upstream.get("query_policy_outcomes_read") is False,
            f"{scene}: upstream construction read policy outcomes")
    require(str(upstream.get("scene")) == scene
            and int(upstream.get("scene_index", -1)) == int(scene_index),
            f"{scene}: upstream completion identity changed")
    materialization = upstream.get("materialization", {})
    require(int(materialization.get("materialized", -1)) == 0,
            f"{scene}: upstream completion is not zero-history")
    require(materialization.get("manifest_sha256") is None,
            f"{scene}: upstream zero-history scene has a manifest")
    upstream_attrition = upstream.get("construction_attrition", [])
    require(isinstance(upstream_attrition, list) and upstream_attrition,
            f"{scene}: upstream zero-history attrition missing")

    require(not out.exists(), f"construction output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    try:
        role_root = temporary / "role_pairs"
        role_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "result-blind cross-history Novel-B collection paired with "
                "a controlled Revisit-A goal for full-mono lifelong evaluation"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": None,
            "construction_seed": 20260824,
            "contract": role_contract(support="standard"),
            "episodes": [],
        }
        manifest_path = role_root / "manifest.json"
        manifest_path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (role_root / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        attrition = [{
            "scene": scene,
            "episode": None,
            "stage": "upstream_actual_A_materialization",
            "reason": "upstream_parent_certified_zero_histories",
            "upstream_construction_attrition": upstream_attrition,
        }]
        completion = {
            "schema_version": SCHEMA,
            "status": "complete",
            "scene": scene,
            "scene_index": int(scene_index),
            "protocol_sha256": sha256_file(protocol_path),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_population_receipt_sha256": sha256_file(
                parent_population_path
            ),
            "online_A_manifest_sha256": None,
            "upstream_parent_completion_sha256": expected_parent_hash,
            "materialized_A_histories": 0,
            "attempted_histories": 0,
            "constructible_AB_C_histories": 0,
            "query_policy_outcomes_read": False,
            "attempts": [],
            "attrition": attrition,
            "role_pair_manifest_sha256": sha256_file(manifest_path),
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
    return completion


def build_scene(
    *,
    parent_root: Path,
    protocol_path: Path,
    scene_index: int,
    out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    expansion = protocol["schema_version"] == EXPANSION_PROTOCOL_SCHEMA
    construction = protocol["novel_b_construction"]
    parent_paths = bind_parent(protocol, parent_root)
    parent_manifest_path = parent_paths["manifest"]
    parent_population_path = parent_paths["population"]
    parent = json.loads(parent_manifest_path.read_text())
    parent_population = json.loads(parent_population_path.read_text())
    require(0 <= scene_index < len(parent["scenes"]), "scene index out of range")
    require(len(parent_population["fragments"]) == len(parent["scenes"]),
            "parent population fragment count changed")
    scene = str(parent["scenes"][scene_index])
    population_fragment = parent_population["fragments"][scene_index]
    require(parent_population.get("policy_outcomes_read") is False,
            "parent population read policy outcomes")
    asset_row = parent["assets"][scene]
    asset = Path(asset_row["glb_path"])
    require(asset.is_file() and sha256_file(asset) == asset_row["glb_sha256"],
            f"{scene}: scene asset changed")
    online_root = (
        parent_root / "construction" / "scenes"
        / f"{scene_index:02d}_{scene}" / "online_a"
    )
    if not online_root.is_dir():
        return write_upstream_empty_scene(
            parent_root=parent_root,
            protocol_path=protocol_path,
            parent_manifest_path=parent_manifest_path,
            parent_population_path=parent_population_path,
            population_fragment=population_fragment,
            scene=scene,
            scene_index=scene_index,
            online_root=online_root,
            out=out,
        )
    online_manifest, histories = load_histories(online_root, scene)
    require(int(population_fragment["scene_index"]) == int(scene_index)
            and str(population_fragment["scene"]) == scene,
            f"{scene}: parent population identity changed")
    require(int(population_fragment["materialized_histories"]) == len(histories),
            f"{scene}: online-A history count changed")
    require(len(histories) != 1,
            f"{scene}: cross-history construction requires zero or >=2 histories")
    require(not out.exists(), f"construction output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    accepted = []
    attempts = []
    attrition = []
    simulator = make_sim(str(asset), "", agent_radius=0.30)
    try:
        for recipient in histories:
            episode = str(recipient["receipt"]["episode"])
            revisit_candidates, revisit_diagnostics = search_revisit_candidates(
                simulator,
                recipient,
                scene=scene,
                episode=episode,
                camera_height=float(recipient["receipt"]["camera_height_m"]),
            )
            revisit = revisit_candidates["standard"]
            measured = []
            if revisit is not None:
                for donor in histories:
                    if donor is recipient:
                        continue
                    indices = (
                        temporal_frame_indices(
                            len(donor["poses"]),
                            int(construction["temporal_samples_per_donor"]),
                        )
                        if expansion else [len(donor["poses"]) - 1]
                    )
                    for temporal_rank, frame_index in enumerate(indices):
                        row = donor_candidate(
                            simulator,
                            recipient,
                            donor,
                            revisit,
                            donor_episode_rank=int(donor["episode_rank"]),
                            donor_frame_index=frame_index,
                            donor_frame_temporal_rank=temporal_rank,
                        )
                        if row is not None:
                            measured.append(row)
            donors = (
                select_donors(
                    measured,
                    recipient_episode=episode,
                    maximum_candidates=int(
                        construction["maximum_candidates_per_recipient"]),
                    maximum_per_donor=int(
                        construction["maximum_candidates_per_donor_history"]),
                    prefer_distinct_direction_strata=bool(construction[
                        "prefer_distinct_initial_direction_strata"]),
                )
                if expansion else [
                    row for row in [select_donor(
                        measured, recipient_episode=episode)] if row is not None
                ]
            )
            attempt = {
                "scene": scene,
                "episode": episode,
                "recipient_episode_rank": int(recipient["episode_rank"]),
                "revisit_A_constructible": revisit is not None,
                "donors_measured": len(measured),
                "donor_selected": bool(donors),
                "selected_donor": (
                    _candidate_json(donors[0]) if donors else None),
                "selected_donors": [
                    _candidate_json(donor) for donor in donors
                ],
                "frozen_candidate_count": len(donors),
                "selected_revisit_A": _candidate_json(revisit),
                "revisit_diagnostics": revisit_diagnostics,
                "query_policy_outcomes_read": False,
            }
            attempts.append(attempt)
            if revisit is None or not donors:
                missing = []
                if revisit is None:
                    missing.append("controlled_revisit_A")
                if not donors:
                    missing.append("unsupported_cross_history_B")
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "AB_C_constructibility",
                    "reason": "missing:" + ",".join(missing),
                })
                continue
            for candidate_index, donor in enumerate(donors):
                synthetic_episode = (
                    f"{episode}__b{candidate_index:02d}" if expansion else episode
                )
                destination = temporary / "role_pairs" / scene / synthetic_episode
                payload = write_protocol_episode(
                    destination=destination,
                    online_episode=recipient["root"],
                    receipt=recipient["receipt"],
                    history=recipient,
                    natural=donor,
                    revisit=revisit,
                    protocol=(
                        "hm3d_fullmono_lifelong_result_blind_power_expansion"
                        if expansion else PROTOCOL_NAME
                    ),
                    scene_rank=scene_index,
                    episode_rank=int(recipient["episode_rank"]),
                )
                payload["episode"] = synthetic_episode
                payload["lifelong_construction"] = {
                    "recipient_episode": episode,
                    "synthetic_candidate_episode": synthetic_episode,
                    "candidate_index": candidate_index,
                    "donor_episode": donor["donor_episode"],
                    "donor_factual_frame_index": int(
                        donor["donor_frame_index"]),
                    "donor_factual_frame_step": int(donor["donor_frame_step"]),
                    "B_max_recipient_A_covis": donor["max_recipient_a_covis"],
                    "B_to_C_geodesic_m": donor["b_to_c_geodesic_m"],
                    "query_policy_outcomes_read": False,
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

        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "result-blind cross-history Novel-B collection paired with "
                "a controlled Revisit-A goal for full-mono lifelong evaluation"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": sha256_file(
                online_root / "manifest.json"
            ),
            "construction_seed": 20260824,
            "contract": role_contract(support="standard"),
            "episodes": accepted,
        }
        # Empty source scenes are retained as explicit attrition.  The shared
        # role-pair validator requires a non-empty manifest, so only write its
        # standard receipt when there are accepted histories.
        role_root = temporary / "role_pairs"
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
            "scene_index": int(scene_index),
            "protocol_sha256": sha256_file(protocol_path),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "online_A_manifest_sha256": sha256_file(
                online_root / "manifest.json"
            ),
            "materialized_A_histories": len(histories),
            "attempted_histories": len(attempts),
            "constructible_AB_C_histories": len(accepted),
            "query_policy_outcomes_read": False,
            "attempts": attempts,
            "attrition": attrition,
            "role_pair_manifest_sha256": sha256_file(
                role_root / "manifest.json"
            ),
        }
        (temporary / "completion.json").write_text(json.dumps(
            completion, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "completion.json.sha256").write_text(
            sha256_file(temporary / "completion.json") + "  completion.json\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        simulator.close()
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_scene(
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        scene_index=args.scene_index,
        out=args.out,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "scene", "materialized_A_histories", "attempted_histories",
            "constructible_AB_C_histories", "query_policy_outcomes_read",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
