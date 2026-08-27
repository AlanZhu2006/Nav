#!/usr/bin/env python3
"""Build both frozen role-pair protocols for every materialized scene history.

Construction is policy-outcome blind.  Every materialized online-A history is
attempted once.  A history enters the evaluation population only when the
existing controlled-pose V1 Revisit builder and both pre-registered Novel
samplers are constructible; every exclusion is retained verbatim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import build_single_revisit_source as revisit_builder
import build_shared_online_role_pairs as pair_builder
from shared_online_role_pair_contract import validate_manifest


def sha256_file(path: Path) -> str:
    return pair_builder.sha256_file(path)


def revisit_contract() -> dict:
    return {
        "minimum_eligible_online_frame": 39,
        "source_anchor_end_margin_frames": 16,
        "source_anchor_stride_frames": 8,
        "minimum_query_geodesic_m": 2.0,
        "maximum_query_geodesic_m": 9.0,
        "target_query_geodesic_m": 3.0,
        "maximum_revisit_candidates": 4,
        "v1_min_translation_m": 0.20,
        "v1_max_translation_m": 0.50,
        "v1_min_yaw_delta_deg": 10.0,
        "v1_max_yaw_delta_deg": 25.0,
        "v1_min_source_frame_covis": 0.45,
        "v1_min_max_online_a_covis": 0.50,
        "v1_max_max_online_a_covis": 0.98,
        "v1_max_argmax_gap_frames": 20,
        "v1_min_pixel_mae": 5.0,
    }


def role_contract(initial_bearing_tolerance_deg: float) -> dict:
    return {
        "online_history": "frozen_native_navdp_goal_a_rgb_depth_pose_trace",
        "query_execution": "independent_reset_and_exact_online_a_replay",
        "runtime_role_visibility": "none",
        "analysis_role_location": "sidecar_only_never_forwarded_to_policy",
        "pairs_per_online_history": 1,
        "minimum_query_geodesic_m": 2.0,
        "maximum_query_geodesic_m": 9.0,
        "maximum_role_distance_error_m": 0.50,
        "maximum_role_initial_path_bearing_error_deg": float(
            initial_bearing_tolerance_deg
        ),
        "novel_max_online_a_covis_exclusive": 0.10,
        "revisit_min_online_a_covis_inclusive": 0.50,
        "revisit_max_online_a_covis_inclusive": 0.98,
        "minimum_clearance_m": 0.30,
        "same_floor_tolerance_m": 0.20,
        "minimum_novel_pair_separation_m": 1.0,
        "novel_candidate_attempts": 5000,
        "novel_covis_stride": 4,
        "covis_depth_tolerance_m": 0.30,
    }


def write_manifest(root: Path, payload: dict) -> str:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    digest = sha256_file(path)
    (root / "manifest.json.sha256").write_text(digest + "  manifest.json\n")
    return digest


def build(online_root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    online_manifest_path = online_root / "manifest.json"
    online_manifest = json.loads(online_manifest_path.read_text())
    if online_manifest.get("schema_version") != "shared_online_a_materialized_v1":
        raise RuntimeError("unexpected online-A materialization schema")

    r_contract = revisit_contract()
    protocol_specs = {
        "support_controlled": (20260814, role_contract(30.0)),
        "natural_direction": (20260815, role_contract(180.0)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    revisit_rows = []
    protocol_rows = {name: [] for name in protocol_specs}
    attrition = []
    try:
        for source in online_manifest["episodes"]:
            scene = str(source["scene"])
            episode = str(source["episode"])
            online_episode = online_root / scene / episode
            revisit_destination = temporary / "revisit_source" / scene / episode
            protocol_destinations = {
                name: temporary / name / scene / episode
                for name in protocol_specs
            }
            try:
                revisit_destination.mkdir(parents=True)
                revisit_row = revisit_builder.build_episode(
                    online_episode, revisit_destination, r_contract
                )
                built_rows = {}
                for name, (seed, contract) in protocol_specs.items():
                    protocol_destinations[name].mkdir(parents=True)
                    built_rows[name] = pair_builder.build_episode(
                        online_episode,
                        temporary / "revisit_source",
                        revisit_row,
                        protocol_destinations[name],
                        contract=contract,
                        global_seed=seed,
                    )
            except Exception as error:
                shutil.rmtree(revisit_destination, ignore_errors=True)
                for destination in protocol_destinations.values():
                    shutil.rmtree(destination, ignore_errors=True)
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "role_pair_constructibility",
                    "reason": f"{type(error).__name__}: {error}",
                })
            else:
                revisit_rows.append(revisit_row)
                for name, row in built_rows.items():
                    protocol_rows[name].append(row)

        revisit_root = temporary / "revisit_source"
        revisit_manifest = {
            "schema_version": revisit_builder.SCHEMA_VERSION,
            "purpose": (
                "single-query controlled-pose V1 Revisit candidates for the "
                "frozen paper role-pair benchmark"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": sha256_file(online_manifest_path),
            "contract": r_contract,
            "episodes": revisit_rows,
        }
        revisit_sha = write_manifest(revisit_root, revisit_manifest)

        for name, (seed, contract) in protocol_specs.items():
            protocol_root = temporary / name
            purpose = (
                "support-controlled role-free Novel/Revisit queries"
                if name == "support_controlled"
                else "natural-direction role-free Novel/Revisit queries"
            )
            manifest = {
                "schema_version": pair_builder.SCHEMA_VERSION,
                "purpose": purpose,
                "source_online_root": str(online_root.resolve()),
                "source_online_manifest_sha256": sha256_file(online_manifest_path),
                "source_revisit_root": str((out / "revisit_source").resolve()),
                "source_revisit_manifest_sha256": revisit_sha,
                "construction_seed": seed,
                "contract": contract,
                "episodes": protocol_rows[name],
            }
            if manifest["episodes"]:
                validate_manifest(manifest)
            write_manifest(protocol_root, manifest)

        construction = {
            "schema_version": "paper_role_pair_scene_build_v2_20260814",
            "query_contract": "one_independent_revisit_query_after_online_a",
            "source_materialized_histories": len(online_manifest["episodes"]),
            "constructible_histories": len(revisit_rows),
            "attrition_count": len(attrition),
            "attrition": attrition,
            "protocols": {
                name: {
                    "episodes": len(rows),
                    "pairs": sum(len(row["pairs"]) for row in rows),
                }
                for name, rows in protocol_rows.items()
            },
            "policy_outcomes_read": False,
        }
        (temporary / "construction_receipt.json").write_text(
            json.dumps(construction, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return construction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.online_root, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
