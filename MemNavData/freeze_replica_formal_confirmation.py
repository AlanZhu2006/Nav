#!/usr/bin/env python3
"""Freeze the outcome-blind ten-scene Replica confirmation population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_COMPATIBILITY_SHA256 = (
    "f6c14266b5aa5462d8a0d21c1219212e482f95b18fe73808eae204f3fe5752cf"
)
EXPECTED_CONSTRUCTIBILITY_SHA256 = (
    "d4a74d41a3d4a874323b3efb28da445165233ef6338822fd1be0b0be5bd6782c"
)
EXPECTED_ELIGIBLE_SCENES = (
    "apartment_1",
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1",
    "room_2",
)
PILOT_SCENE = "room_0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def freeze(
    compatibility_path: Path,
    constructibility_path: Path,
    out: Path,
) -> dict:
    require(not out.exists(), f"output already exists: {out}")
    require(
        sha256_file(compatibility_path) == EXPECTED_COMPATIBILITY_SHA256,
        "Replica compatibility receipt changed",
    )
    compatibility = json.loads(compatibility_path.read_text())
    require(compatibility["passed"] is True, "compatibility gate did not pass")
    require(
        compatibility["paper_navigation_evaluation_authorized"] is True,
        "Replica navigation was not authorized",
    )
    require(
        tuple(compatibility["eligible_scenes"]) == EXPECTED_ELIGIBLE_SCENES,
        "eligible Replica scene population changed",
    )
    require(
        sha256_file(constructibility_path)
        == EXPECTED_CONSTRUCTIBILITY_SHA256,
        "Replica Goal-A constructibility receipt changed",
    )
    constructibility = json.loads(constructibility_path.read_text())
    require(
        constructibility["query_outcomes_read"] is False
        and constructibility["navigation_outcomes_generated"] is False,
        "constructibility diagnostic is not outcome blind",
    )
    require(
        constructibility["fresh_histories_excluding_room_0"] == 20
        and constructibility[
            "fresh_scene_count_with_history_excluding_room_0"] == 5
        and constructibility["target_met"] is False,
        "constructibility result changed",
    )
    rows = {str(row["scene"]): row for row in compatibility["scenes"]}
    replica_root = Path(compatibility["replica_root"]).resolve()
    scenes = []
    for index, scene in enumerate(EXPECTED_ELIGIBLE_SCENES):
        row = rows[scene]
        require(row["eligible"] is True, f"scene lost eligibility: {scene}")
        diameter = float(row["geodesic_probe"]["maximum_sampled_geodesic_m"])
        long_history = diameter >= 4.5
        stage = replica_root / scene / "habitat/replica_stage.stage_config.json"
        navmesh = replica_root / scene / "habitat/mesh_semantic.navmesh"
        files = {}
        for relative, record in sorted(row["files"].items()):
            path = replica_root / scene / relative
            require(path.is_file(), f"missing compatible asset: {path}")
            files[relative] = {
                "path": str(path),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
        scenes.append({
            "index": index,
            "scene": scene,
            "analysis_status": (
                "locked_consumed_pilot" if scene == PILOT_SCENE
                else "fresh_cross_dataset_stress"
            ),
            "compatibility_probe_max_geodesic_m": diameter,
            "source_stratum": "long_history" if long_history else "diagnostic",
            "source_attempts": 4,
            "generator_seed": 2026081500 + index,
            "online_a_seed": 2026081600 + index,
            "generator_contract": {
                "n_legs": 2,
                "goal_a_source_only": True,
                "requested_episodes": 4,
                "dA_min_m": 4.5 if long_history else 2.0,
                "dA_max_m": 6.5 if long_history else 5.0,
                "b_min_m": 2.0,
                "episode_attempt_multiplier": 30,
                "max_attempts": 100,
                "allow_incomplete": False,
            },
            "stage": str(stage),
            "navmesh": str(navmesh),
            "files": files,
        })

    payload = {
        "schema_version": "replica_cross_dataset_stress_freeze_v2_20260814",
        "frozen_at_date_asia_shanghai": "2026-08-14",
        "scope": (
            "ten compatible Replica scenes attempted; the five fresh scenes "
            "constructible under the unchanged Goal-A contract form an "
            "explicitly underpowered cross-dataset stress population"
        ),
        "compatibility_receipt": str(compatibility_path.resolve()),
        "compatibility_receipt_sha256": EXPECTED_COMPATIBILITY_SHA256,
        "goal_a_constructibility_receipt": str(
            constructibility_path.resolve()),
        "goal_a_constructibility_receipt_sha256": (
            EXPECTED_CONSTRUCTIBILITY_SHA256),
        "replica_root": str(replica_root),
        "scene_admission_uses_navigation_outcome": False,
        "query_outcomes_read": False,
        "formal_confirmation_authorized": False,
        "cross_dataset_stress_evaluation_authorized": True,
        "confirmation_veto_reason": (
            "unchanged Goal-A source contract yields 20 histories in only "
            "5 fresh scene clusters, below the frozen 8-cluster target"),
        "scene_count": len(scenes),
        "pilot_scene_excluded_from_primary": PILOT_SCENE,
        "primary_scenes": [
            scene for scene in EXPECTED_ELIGIBLE_SCENES if scene != PILOT_SCENE
        ],
        "primary_target": {"histories": 20, "scene_clusters": 8},
        "pre_navigation_constructibility": {
            "fresh_histories": 20,
            "fresh_scene_clusters": 5,
            "target_met": False,
        },
        "protocols": ["support_controlled", "natural_direction"],
        "arms": [
            "native", "raw_direct", "raw_fixed_bearing",
            "geometry_fixed", "certified",
        ],
        "controller_contract": {
            "navdp_frozen": True,
            "max_steps": 600,
            "success_radius_m": 1.0,
            "exec_horizon": 8,
            "deterministic_plan_seeds": True,
            "residual_distance_m": 2.5,
            "terminal_uturn": False,
            "terminal_visual_refine": False,
            "graph_rescue": False,
            "cdec_rescue": False,
        },
        "certificate_contract": {
            "minimum_pnp_inliers": 16,
            "minimum_query_hull_coverage": 0.05,
            "minimum_reference_hull_coverage": 0.05,
            "maximum_reprojection_rmse_px": 2.0,
            "replica_adaptation": False,
        },
        "reporting": {
            "primary": (
                "fresh cross-dataset stress only; no formal confirmation claim"),
            "ten_scene_aggregate": "descriptive only",
            "statistical_unit": "paired query",
            "uncertainty_cluster": "scene",
            "bootstrap_resamples": 100000,
            "no_intermediate_sr_read_before_completion": True,
        },
        "scenes": scenes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(
        f"{digest}  {out.name}\n"
    )
    return {"manifest": str(out), "sha256": digest, "scenes": len(scenes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--constructibility", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        args.compatibility, args.constructibility, args.out
    ), sort_keys=True))


if __name__ == "__main__":
    main()
