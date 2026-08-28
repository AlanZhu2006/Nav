#!/usr/bin/env python3
"""Construct one outcome-blind HM3D Table-1 reserve fragment.

The source online-A histories were materialized before query evaluation.  This
builder excludes every identity that entered the earlier 28-history formal
population, then asks only whether the remaining histories support one
standard Revisit query and one unsupported Novel query.  A Novel query may use
the first constructible front/side/rear stratum in a deterministic, identity-
bound order; navigation outcomes are never opened.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import build_final14_role_pair_scene as role_builder
    import build_shared_online_double_revisit as history_tools
    from final14_role_pair_contract import (
        assigned_direction_stratum,
    )
    from generate_twoleg import make_sim
    from shared_online_role_pair_contract import SCHEMA_VERSION
except ImportError:
    from MemNavData import build_final14_role_pair_scene as role_builder
    from MemNavData import build_shared_online_double_revisit as history_tools
    from MemNavData.final14_role_pair_contract import (
        assigned_direction_stratum,
    )
    from MemNavData.generate_twoleg import make_sim
    from MemNavData.shared_online_role_pair_contract import SCHEMA_VERSION
try:
    from hm3d_table1_fresh_query_contract import (
        CONSTRUCTION_SEED,
        SCENE_SCHEMA,
        SEED_NAMESPACE,
        identity_set,
        require,
        stratum_order,
    )
except ImportError:
    from MemNavData.hm3d_table1_fresh_query_contract import (
        CONSTRUCTION_SEED,
        SCENE_SCHEMA,
        SEED_NAMESPACE,
        identity_set,
        require,
        stratum_order,
    )


def construct_one(online_episode: Path, *, scene_rank: int,
                  episode_rank: int) -> dict[str, Any]:
    receipt = json.loads((online_episode / "receipt.json").read_text())
    history = history_tools.load_online_history(online_episode, receipt)
    scene = str(receipt["scene"])
    episode = str(receipt["episode"])
    simulator = make_sim(
        str(receipt["source_asset"]), "", agent_radius=0.30,
    )
    errors: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    natural = None
    selected_stratum = None
    try:
        revisit, revisit_diagnostics = role_builder.search_revisit_candidates(
            simulator,
            history,
            scene=scene,
            episode=episode,
            camera_height=float(receipt["camera_height_m"]),
        )
        if revisit["standard"] is not None:
            for stratum in stratum_order(
                scene_rank, episode_rank, scene, episode,
            ):
                try:
                    natural, one_diagnostic = role_builder.sample_natural_novel(
                        simulator,
                        history,
                        scene=scene,
                        episode=episode,
                        scene_rank=scene_rank,
                        episode_rank=episode_rank,
                        paired_revisit_position=np.asarray(
                            revisit["standard"]["_position"], dtype=np.float64,
                        ),
                        camera_height=float(receipt["camera_height_m"]),
                        direction_stratum=stratum,
                        sampling_seed_namespace=SEED_NAMESPACE,
                    )
                    diagnostics[stratum] = one_diagnostic
                    selected_stratum = stratum
                    break
                except role_builder.NaturalNovelConstructionError as error:
                    errors[stratum] = str(error)
    finally:
        simulator.close()
    return {
        "receipt": receipt,
        "history": history,
        "standard": revisit["standard"],
        "natural": natural,
        "selected_stratum": selected_stratum,
        "stratum_order": stratum_order(
            scene_rank, episode_rank, scene, episode,
        ),
        "revisit_diagnostics": revisit_diagnostics,
        "natural_diagnostics": diagnostics,
        "natural_errors": errors,
    }


def build(*, source_run_root: Path, original_manifest_path: Path,
          scene_index: int, out: Path, maximum_histories: int = 3) -> dict:
    require(scene_index >= 0, "scene index must be non-negative")
    require(maximum_histories > 0, "history cap must be positive")
    require(not out.exists(), f"output already exists: {out}")
    parent_path = source_run_root / "sealed_inputs/parent_manifest.json"
    parent = json.loads(parent_path.read_text())
    scenes = list(parent["scenes"])
    require(scene_index < len(scenes), "scene index outside parent manifest")
    scene = str(scenes[scene_index])
    source_order = [
        str(row["episode"]) for row in parent["episodes"].get(scene, [])
    ]
    label = f"{scene_index:02d}_{scene}"
    online_root = (
        source_run_root / "construction/scenes" / label / "online_a"
    )
    consumed_manifest = json.loads(original_manifest_path.read_text())
    consumed = identity_set(consumed_manifest)
    require(all(identity[0] in scenes for identity in consumed),
            "consumed identity is outside the source population")

    online_manifest_path = online_root / "manifest.json"
    materialized: dict[str, dict] = {}
    online_manifest_sha = None
    if online_manifest_path.is_file():
        online_manifest = json.loads(online_manifest_path.read_text())
        require(
            online_manifest.get("schema_version")
            == "shared_online_a_materialized_v1",
            f"{scene}: online-A schema changed",
        )
        materialized = {
            str(row["episode"]): row for row in online_manifest["episodes"]
        }
        require(set(materialized).issubset(set(source_order)),
                f"{scene}: materialized history is outside source order")
        online_manifest_sha = history_tools.sha256_file(online_manifest_path)
    else:
        require(not source_order, f"{scene}: missing online-A manifest")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    rows = []
    attempts = []
    attrition = []
    retained = 0
    try:
        for episode_rank, episode in enumerate(source_order):
            if episode not in materialized:
                continue
            identity = (scene, episode)
            if identity in consumed:
                attempts.append({
                    "scene": scene,
                    "episode": episode,
                    "source_episode_rank": episode_rank,
                    "excluded_consumed_identity": True,
                    "attempted_query_construction": False,
                    "retained": False,
                })
                continue
            result = construct_one(
                online_root / scene / episode,
                scene_rank=scene_index,
                episode_rank=episode_rank,
            )
            constructible = (
                result["standard"] is not None
                and result["natural"] is not None
            )
            attempt = {
                "scene": scene,
                "episode": episode,
                "source_episode_rank": episode_rank,
                "excluded_consumed_identity": False,
                "attempted_query_construction": True,
                "standard_revisit_constructible": result["standard"] is not None,
                "natural_novel_constructible": result["natural"] is not None,
                "preferred_direction_stratum": assigned_direction_stratum(
                    scene_index, episode_rank,
                ),
                "stratum_attempt_order": result["stratum_order"],
                "selected_direction_stratum": result["selected_stratum"],
                "retained": False,
                "cap_excluded": False,
                "revisit_diagnostics": result["revisit_diagnostics"],
                "natural_diagnostics": result["natural_diagnostics"],
                "natural_errors": result["natural_errors"],
            }
            if not constructible:
                missing = []
                if result["standard"] is None:
                    missing.append("standard_revisit")
                if result["natural"] is None:
                    missing.append("unsupported_novel_all_strata")
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "fresh_query_constructibility",
                    "reason": "missing:" + ",".join(missing),
                })
                attempts.append(attempt)
                continue
            if retained >= maximum_histories:
                attempt["cap_excluded"] = True
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "per_scene_history_cap",
                    "reason": f"retained_first_{maximum_histories}",
                })
                attempts.append(attempt)
                continue
            attempt["retained"] = True
            retained += 1
            destination = temporary / "natural_direction" / scene / episode
            rows.append(role_builder.write_protocol_episode(
                destination=destination,
                online_episode=online_root / scene / episode,
                receipt=result["receipt"],
                history=result["history"],
                natural=result["natural"],
                revisit=result["standard"],
                protocol="table1_fresh_query",
                scene_rank=scene_index,
                episode_rank=episode_rank,
            ))
            attempts.append(attempt)

        contract = role_builder.role_contract(support="standard")
        contract.update({
            "query_population": "fresh_query_reserve_excluding_consumed_28",
            "novel_direction_rule": (
                "preferred_final14_stratum_then_identity_bound_fallback_"
                "across_remaining_strata"
            ),
            "navigation_outcomes_used_for_construction": False,
        })
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "HM3D Table-1 fresh-query controller-portability reserve"
            ),
            "source_online_root": str(online_root.resolve()),
            "source_online_manifest_sha256": online_manifest_sha,
            "construction_seed": CONSTRUCTION_SEED,
            "contract": contract,
            "episodes": rows,
        }
        role_builder.write_manifest(temporary / "natural_direction", manifest)
        receipt = {
            "schema_version": SCENE_SCHEMA,
            "scene": scene,
            "scene_index": scene_index,
            "construction_seed": CONSTRUCTION_SEED,
            "sampling_seed_namespace": SEED_NAMESPACE,
            "source_parent_manifest_sha256": history_tools.sha256_file(parent_path),
            "source_online_manifest_sha256": online_manifest_sha,
            "consumed_manifest_sha256": history_tools.sha256_file(
                original_manifest_path,
            ),
            "source_materialized_histories": len(materialized),
            "consumed_identities_excluded": sum(
                (scene, episode) in consumed for episode in materialized
            ),
            "reserve_histories_attempted": sum(
                row.get("attempted_query_construction") is True for row in attempts
            ),
            "retained_histories": len(rows),
            "maximum_histories_per_scene": maximum_histories,
            "attrition": attrition,
            "attempts": attempts,
            "query_policy_outcomes_read": False,
        }
        (temporary / "construction_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-histories", type=int, default=3)
    args = parser.parse_args()
    result = build(
        source_run_root=args.source_run_root.resolve(),
        original_manifest_path=args.original_manifest.resolve(),
        scene_index=args.scene_index,
        out=args.out.resolve(),
        maximum_histories=args.maximum_histories,
    )
    print(json.dumps({
        "scene": result["scene"],
        "reserve_histories_attempted": result["reserve_histories_attempted"],
        "retained_histories": result["retained_histories"],
        "query_policy_outcomes_read": result["query_policy_outcomes_read"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
