#!/usr/bin/env python3
"""Materialize and construct one outcome-blind MP3D Table-1 scene."""

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
    from final14_role_pair_contract import assigned_direction_stratum
    from generate_twoleg import make_sim
    from materialize_hm3d_fullmono_online_a import materialize_scene
    from mp3d_table1_new_query_contract import (
        CONSTRUCTION_SEED,
        SCENE_SCHEMA,
        SEED_NAMESPACE,
        SOURCE_LEDGER_SCHEMAS,
        assert_new_query_identity,
        require,
        stratum_order,
    )
    from shared_online_role_pair_contract import SCHEMA_VERSION
except ImportError:
    from MemNavData import build_final14_role_pair_scene as role_builder
    from MemNavData import build_shared_online_double_revisit as history_tools
    from MemNavData.final14_role_pair_contract import assigned_direction_stratum
    from MemNavData.generate_twoleg import make_sim
    from MemNavData.materialize_hm3d_fullmono_online_a import materialize_scene
    from MemNavData.mp3d_table1_new_query_contract import (
        CONSTRUCTION_SEED,
        SCENE_SCHEMA,
        SEED_NAMESPACE,
        SOURCE_LEDGER_SCHEMAS,
        assert_new_query_identity,
        require,
        stratum_order,
    )
    from MemNavData.shared_online_role_pair_contract import SCHEMA_VERSION


def construct_one(online_episode: Path, *, scene_rank: int,
                  episode_rank: int) -> dict[str, Any]:
    receipt = json.loads((online_episode / "receipt.json").read_text())
    history = history_tools.load_online_history(online_episode, receipt)
    scene = str(receipt["scene"])
    episode = str(receipt["episode"])
    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
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


def build(*, source_ledger_path: Path, scene_index: int,
          out: Path) -> dict[str, Any]:
    require(scene_index >= 0, "scene index must be non-negative")
    require(not out.exists(), f"output already exists: {out}")
    ledger = json.loads(source_ledger_path.read_text())
    require(ledger.get("schema_version") in SOURCE_LEDGER_SCHEMAS,
            "source ledger schema changed")
    require(ledger.get("previous_goal_b_policy_outcomes_read") is False,
            "source ledger read previous Goal-B outcomes")
    scenes = list(ledger["scenes"])
    require(scene_index < len(scenes), "scene index outside source ledger")
    source = scenes[scene_index]
    require(int(source["scene_index"]) == scene_index,
            "source scene index changed")
    scene = str(source["scene"])
    source_episodes = list(source["episodes"])
    source_order = [str(row["episode"]) for row in source_episodes]
    require(len(source_order) == len(set(source_order)),
            f"{scene}: duplicate source episodes")
    by_episode = {str(row["episode"]): row for row in source_episodes}
    for row in source_episodes:
        trace = Path(row["trace_path"])
        require(trace.is_file()
                and history_tools.sha256_file(trace) == row["trace_sha256"],
                f"{scene}/{row['episode']}: source trace changed")
        for path_field, hash_field in (
            ("source_metadata_path", "source_metadata_sha256"),
            ("source_parquet_path", "source_parquet_sha256"),
        ):
            path = Path(row[path_field])
            require(path.is_file()
                    and history_tools.sha256_file(path) == row[hash_field],
                    f"{scene}/{row['episode']}: {path_field} changed")
        old_goal = Path(row["consumed_goal_b"]["goal_rgb_path"])
        require(old_goal.is_file()
                and history_tools.sha256_file(old_goal)
                == row["consumed_goal_b"]["goal_rgb_sha256"],
                f"{scene}/{row['episode']}: consumed Goal-B changed")
    trace_root = Path(source_episodes[0]["trace_path"]).parents[1]
    asset = Path(source["asset_path"])
    require(asset.is_file()
            and history_tools.sha256_file(asset) == source["asset_sha256"],
            f"{scene}: source asset changed")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    rows = []
    attempts = []
    attrition = []
    try:
        online_root = temporary / "online_a"
        materialization = materialize_scene(
            trace_root=trace_root,
            scene=scene,
            asset=asset,
            episode_root=Path(source["episode_root"]),
            source_episode_order=source_order,
            out=online_root,
            purpose=(
                "actual-online monocular MP3D Goal-A histories for a "
                "role-hidden new-query Table-1 replication"
            ),
        )
        online_manifest = json.loads((online_root / "manifest.json").read_text())
        materialized = {
            str(row["episode"]): row for row in online_manifest["episodes"]
        }
        for episode_rank, episode in enumerate(source_order):
            if episode not in materialized:
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "actual_mono_goal_a_eligibility",
                    "reason": "not_materialized_by_frozen_goal_a_contract",
                })
                attempts.append({
                    "scene": scene,
                    "episode": episode,
                    "source_episode_rank": episode_rank,
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
                "attempted_query_construction": True,
                "standard_revisit_constructible": result["standard"] is not None,
                "natural_novel_constructible": result["natural"] is not None,
                "preferred_direction_stratum": assigned_direction_stratum(
                    scene_index, episode_rank,
                ),
                "stratum_attempt_order": result["stratum_order"],
                "selected_direction_stratum": result["selected_stratum"],
                "retained": False,
                "rejected_consumed_goal_identity": False,
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
                    "stage": "new_query_constructibility",
                    "reason": "missing:" + ",".join(missing),
                })
                attempts.append(attempt)
                continue
            destination = temporary / "natural_direction" / scene / episode
            row = role_builder.write_protocol_episode(
                destination=destination,
                online_episode=online_root / scene / episode,
                receipt=result["receipt"],
                history=result["history"],
                natural=result["natural"],
                revisit=result["standard"],
                protocol="mp3d_table1_new_query",
                scene_rank=scene_index,
                episode_rank=episode_rank,
            )
            # The scene is assembled under an atomic temporary directory.
            # Bind the persisted query record to the post-rename online-A
            # location, not to the transient ``*.tmp.*`` path.
            stable_online_episode = out / "online_a" / scene / episode
            row["online_a_episode"] = str(stable_online_episode.resolve())
            metadata_path = destination / "role_pairs.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["online_a_episode"] = row["online_a_episode"]
            metadata_path.write_text(json.dumps(
                metadata, indent=2, sort_keys=True, allow_nan=False,
            ) + "\n")
            row["role_pairs_sha256"] = history_tools.sha256_file(metadata_path)
            try:
                source_record = by_episode[episode]
                forbidden = source_record.get(
                    "consumed_queries", [source_record["consumed_goal_b"]],
                )
                assert_new_query_identity(row, forbidden)
            except RuntimeError:
                shutil.rmtree(destination)
                attempt["rejected_consumed_goal_identity"] = True
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "consumed_goal_b_exclusion",
                    "reason": "new_query_matches_consumed_goal_b_hash_or_pose",
                })
                attempts.append(attempt)
                continue
            attempt["retained"] = True
            rows.append(row)
            attempts.append(attempt)

        contract = role_builder.role_contract(support="standard")
        contract.update({
            "query_population": "reused_scene_history_new_outcome_blind_query",
            "novel_direction_rule": (
                "preferred_stratum_then_identity_bound_fallback"
            ),
            "consumed_source_goal_b_excluded_by_hash_and_pose": True,
            "previous_goal_b_policy_outcomes_read": False,
            "query_navigation_outcomes_used_for_construction": False,
        })
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "MP3D Table-1 role-hidden new-query cross-controller replication"
            ),
            "source_online_root": str((out / "online_a").resolve()),
            "source_online_manifest_sha256": history_tools.sha256_file(
                online_root / "manifest.json"),
            "construction_seed": CONSTRUCTION_SEED,
            "contract": contract,
            "episodes": rows,
        }
        role_builder.write_manifest(temporary / "natural_direction", manifest)
        receipt = {
            "schema_version": SCENE_SCHEMA,
            "scene": scene,
            "scene_index": scene_index,
            "source_ledger_sha256": history_tools.sha256_file(source_ledger_path),
            "construction_seed": CONSTRUCTION_SEED,
            "sampling_seed_namespace": SEED_NAMESPACE,
            "source_history_count": len(source_order),
            "goal_a_successes": int(materialization["goal_a_successes"]),
            "materialized_histories": int(materialization["materialized"]),
            "query_construction_attempts": sum(
                row["attempted_query_construction"] for row in attempts
            ),
            "retained_histories": len(rows),
            "consumed_goal_b_rejections": sum(
                row.get("rejected_consumed_goal_identity") is True
                for row in attempts
            ),
            "attrition": attrition,
            "attempts": attempts,
            "previous_goal_b_policy_outcomes_read": False,
            "query_policy_outcomes_read": False,
        }
        (temporary / "construction_receipt.json").write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        source_ledger_path=args.source_ledger.resolve(),
        scene_index=args.scene_index,
        out=args.out.resolve(),
    )
    print(json.dumps({
        "scene": result["scene"],
        "goal_a_successes": result["goal_a_successes"],
        "retained_histories": result["retained_histories"],
        "query_policy_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
