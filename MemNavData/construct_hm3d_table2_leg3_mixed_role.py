#!/usr/bin/env python3
"""Construct one new mixed-role Leg-3 pair after a factual mono A/B prefix.

This program is construction-only.  It renders and verifies the already
sealed A/B observations, creates two previously unused query identities and
never starts or reads a navigation policy outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import build_final14_role_pair_scene as role_builder
import build_shared_online_double_revisit as history_tools
import build_shared_online_role_pairs as pair_tools
from deterministic_eval_protocol import validate_leg1_trace
from generate_twoleg import cam_to_world_hab, make_sim, render
from materialize_online_a_traces import native_control_audit
from hm3d_table2_leg3_mixed_role import (
    FRAGMENT_SCHEMA,
    PREFIX_RECEIPT_SCHEMA,
    compose_actual_ab_trace,
    load_protocol,
    require,
    sha256_file,
    stratum_order,
)


def wrap_radians(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def validate_dependency_contract() -> None:
    """Fail before Habitat load if an older role-builder shadows the bundle."""

    require(hasattr(role_builder, "NaturalNovelConstructionError"),
            "role builder lacks structured Natural-Novel rejection")
    parameters = inspect.signature(
        role_builder.sample_natural_novel
    ).parameters
    for name in (
        "maximum_paired_distance_m", "separated_from_positions",
        "minimum_candidate_separation_m", "direction_stratum",
        "sampling_seed_namespace",
    ):
        require(name in parameters,
                f"role builder lacks required argument {name}")


def pose_identity(position: np.ndarray, yaw: float) -> tuple[float, ...]:
    values = np.asarray(position, dtype=np.float64)
    return tuple(round(float(value), 4) for value in values) + (
        round(wrap_radians(yaw), 4),
    )


def find_old_query(benchmark: dict[str, Any], role: str) -> dict[str, Any]:
    rows = [
        query for pair in benchmark["pairs"] for query in pair["queries"]
        if query["analysis_role"] == role
    ]
    require(len(rows) == 1, f"source benchmark has no unique {role} query")
    return rows[0]


def copy_a_history(source: Path, destination: Path, steps: int) -> None:
    for kind, suffix in (("rgb", ".jpg"), ("depth", ".png")):
        target = destination / kind
        target.mkdir(parents=True, exist_ok=True)
        for step in range(steps):
            src = source / kind / f"{step:06d}{suffix}"
            require(src.is_file(), f"missing source A {kind} frame {step}")
            shutil.copy2(src, target / f"{step:06d}{suffix}")


def render_b_history(simulator, trace_b: dict[str, Any], *, camera_height: float,
                     destination: Path, offset: int) -> dict[str, list[Any]]:
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    floor_positions: list[np.ndarray] = []
    camera_positions: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    for pose in trace_b["poses"]:
        floor = np.asarray(
            [pose[axis] for axis in ("x", "y", "z")], dtype=np.float64,
        )
        camera = floor + np.asarray([0.0, camera_height, 0.0])
        yaw = float(pose["yaw"])
        rgb, depth = render(simulator, camera, yaw)
        encoded = history_tools.jpeg_bytes(rgb)
        require(hashlib.sha256(encoded).hexdigest() == pose["jpg_sha256"],
                f"factual B RGB mismatch at step {pose['step']}")
        step = offset + int(pose["step"])
        (destination / "rgb" / f"{step:06d}.jpg").write_bytes(encoded)
        history_tools.write_depth_png(
            destination / "depth" / f"{step:06d}.png", depth,
        )
        rgbs.append(rgb)
        depths.append(depth)
        floor_positions.append(floor)
        camera_positions.append(camera)
        transforms.append(cam_to_world_hab(camera, yaw))
    return {
        "rgbs": rgbs,
        "depths": depths,
        "floor_positions": floor_positions,
        "camera_positions": camera_positions,
        "transforms": transforms,
    }


def combined_history(history_a: dict[str, Any], trace_ab: dict[str, Any],
                     b: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "trace": trace_ab,
        "poses": trace_ab["poses"],
        "rgbs": list(history_a["rgbs"]) + list(b["rgbs"]),
        "depths": list(history_a["depths"]) + list(b["depths"]),
        "floor_positions": (
            list(history_a["floor_positions"]) + list(b["floor_positions"])
        ),
        "camera_positions": (
            list(history_a["camera_positions"]) + list(b["camera_positions"])
        ),
        "transforms": (
            list(history_a["transforms"]) + list(b["transforms"])
        ),
    }


def write_prefix(*, root: Path, source_a: Path, receipt_a: dict[str, Any],
                 trace_a: dict[str, Any], trace_b: dict[str, Any],
                 goal_b_rgb: Path, simulator) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_ab = compose_actual_ab_trace(
        trace_a, trace_b, episode=str(trace_b["episode"]),
    )
    validate_leg1_trace(trace_ab)
    steps_a = len(trace_a["poses"])
    copy_a_history(source_a, root, steps_a)
    b = render_b_history(
        simulator, trace_b, camera_height=float(receipt_a["camera_height_m"]),
        destination=root, offset=steps_a,
    )
    trace_path = root / "online_a_trace.json"
    trace_path.write_text(json.dumps(
        trace_ab, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    shutil.copy2(goal_b_rgb, root / "goal_a.jpg")
    audit = native_control_audit(trace_ab)
    require(audit["ok"] is True, "composed A/B prefix contains memory control")
    receipt = dict(receipt_a)
    receipt.update({
        "schema_version": "shared_online_a_materialized_v1",
        "episode": str(trace_ab["episode"]),
        "online_a_reached": True,
        "online_a_steps": len(trace_ab["poses"]),
        "online_a_trace_sha256": sha256_file(trace_path),
        "online_a_control_audit": audit,
        "goal_a_sha256": sha256_file(root / "goal_a.jpg"),
        "prefix_receipt_schema": PREFIX_RECEIPT_SCHEMA,
        "prefix_semantics": "actual_mono_Novel_A_then_Novel_B",
        "prefix_A_steps": steps_a,
        "prefix_B_steps": len(trace_b["poses"]),
        "source_online_A_receipt_sha256": sha256_file(source_a / "receipt.json"),
        "source_online_A_trace_sha256": sha256_file(
            source_a / "online_a_trace.json"
        ),
        "source_online_B_trace_sha256": hashlib.sha256(
            (json.dumps(trace_b, indent=2, sort_keys=True, allow_nan=False)
             + "\n").encode()
        ).hexdigest(),
        "rgb_frame_hashes": [
            str(pose["jpg_sha256"]) for pose in trace_ab["poses"]
        ],
    })
    receipt_path = root / "receipt.json"
    receipt_path.write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    history_a = history_tools.load_online_history(source_a, receipt_a)
    return receipt, combined_history(history_a, trace_ab, b)


def is_forbidden_candidate(candidate: dict[str, Any], *,
                           forbidden_poses: set[tuple[float, ...]],
                           forbidden_hashes: set[str]) -> bool:
    encoded = history_tools.jpeg_bytes(candidate["_rgb"])
    return (
        hashlib.sha256(encoded).hexdigest() in forbidden_hashes
        or pose_identity(candidate["_position"], candidate["_yaw"])
        in forbidden_poses
    )


def construct(*, protocol_path: Path, source_root: Path,
              population_index: int, out: Path) -> dict[str, Any]:
    validate_dependency_contract()
    protocol = load_protocol(protocol_path)
    source_contract = protocol["source_population"]
    require(source_root.resolve() == Path(source_contract["run_root"]).resolve(),
            "Table-2 source root changed")
    population_path = source_root / source_contract["population"]
    seal_path = source_root / source_contract["seal"]
    require(sha256_file(population_path) == source_contract["population_sha256"],
            "Table-2 source population changed")
    require(sha256_file(seal_path) == source_contract["seal_sha256"],
            "Table-2 source seal changed")
    population = json.loads(population_path.read_text())
    require(population.get("selection_reads_C_B2_C2_navigation_outcomes") is False,
            "source population read post-prefix outcomes")
    rows = population["accepted"]
    require(0 <= population_index < len(rows),
            "population index outside sealed A/B prefixes")
    source_row = rows[population_index]
    benchmark_path = population_path.parent / source_row["benchmark"]
    require(sha256_file(benchmark_path) == source_row["benchmark_sha256"],
            "source A/B benchmark changed")
    benchmark = json.loads(benchmark_path.read_text())
    scene, episode = str(benchmark["scene"]), str(benchmark["episode"])
    scene_ledger = protocol["scene_ledger"]
    parent_manifest_path = Path(scene_ledger["parent_manifest"])
    require(sha256_file(parent_manifest_path)
            == scene_ledger["parent_manifest_sha256"],
            "Table-2 parent scene ledger changed")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    require(scene in parent_manifest["scenes"],
            "Table-2 source scene is outside the parent ledger")
    scene_rank = list(parent_manifest["scenes"]).index(scene)
    source_a = Path(benchmark["source_online_A_episode"])
    require(sha256_file(source_a / "receipt.json")
            == benchmark["source_online_A_receipt_sha256"],
            "source online-A receipt changed")
    require(sha256_file(source_a / "online_a_trace.json")
            == benchmark["source_online_A_trace_sha256"],
            "source online-A trace changed")
    receipt_a = json.loads((source_a / "receipt.json").read_text())
    trace_a = json.loads((source_a / "online_a_trace.json").read_text())
    trace_b_path = benchmark_path.parent / benchmark["online_B_trace"]
    require(sha256_file(trace_b_path) == benchmark["online_B_trace_sha256"],
            "source online-B trace changed")
    trace_b = json.loads(trace_b_path.read_text())
    scene_file = Path(benchmark["source_scene_asset"])
    require(sha256_file(scene_file) == benchmark["source_scene_asset_sha256"],
            "HM3D scene asset changed")
    old_novel, old_revisit = (
        find_old_query(json.loads((
            source_root / "ab_population/role_pairs" / scene / episode
            / "role_pairs.json"
        ).read_text()), role)
        for role in ("novel", "revisit")
    )
    forbidden_queries = (old_novel, old_revisit)
    forbidden_hashes = {str(row["goal_rgb_sha256"]) for row in forbidden_queries}
    forbidden_poses = {
        pose_identity(np.asarray(row["floor_position"]), float(row["yaw_rad"]))
        for row in forbidden_queries
    }

    require(not out.exists(), f"Table-2 fragment exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    simulator = make_sim(str(scene_file), "", agent_radius=0.30)
    try:
        prefix = temporary / "causal_prefix" / scene / episode
        prefix.mkdir(parents=True)
        goal_b_rgb = benchmark_path.parent / benchmark["goals"]["B"]["rgb"]
        require(sha256_file(goal_b_rgb)
                == benchmark["goals"]["B"]["rgb_sha256"],
                "source Goal-B RGB changed")
        receipt_ab, history_ab = write_prefix(
            root=prefix, source_a=source_a, receipt_a=receipt_a,
            trace_a=trace_a, trace_b=trace_b, goal_b_rgb=goal_b_rgb,
            simulator=simulator,
        )
        revisit, revisit_diagnostics = role_builder.search_revisit_candidates(
            simulator, history_ab, scene=scene,
            episode=f"{episode}__table2_leg3",
            camera_height=float(receipt_ab["camera_height_m"]),
        )
        selected_revisit = revisit["standard"]
        attrition_reason = None
        if selected_revisit is None:
            attrition_reason = "no_new_standard_revisit_after_combined_AB"
        elif is_forbidden_candidate(
            selected_revisit, forbidden_poses=forbidden_poses,
            forbidden_hashes=forbidden_hashes,
        ):
            attrition_reason = "revisit_collides_with_consumed_B_or_C_identity"

        natural = None
        selected_stratum = None
        natural_diagnostics: dict[str, Any] = {}
        natural_errors: dict[str, str] = {}
        order = stratum_order(population_index, scene, episode)
        if attrition_reason is None:
            for stratum in order:
                try:
                    candidate, diagnostics = role_builder.sample_natural_novel(
                        simulator, history_ab, scene=scene,
                        episode=f"{episode}__table2_leg3_novel",
                        scene_rank=population_index,
                        episode_rank=population_index,
                        paired_revisit_position=np.asarray(
                            selected_revisit["_position"], dtype=np.float64,
                        ),
                        camera_height=float(receipt_ab["camera_height_m"]),
                        maximum_paired_distance_m=9.0,
                        separated_from_positions=[
                            np.asarray(row["floor_position"], dtype=np.float64)
                            for row in forbidden_queries
                        ],
                        minimum_candidate_separation_m=0.50,
                        direction_stratum=stratum,
                        sampling_seed_namespace="hm3d_table2_leg3_novel_20260829",
                    )
                    natural_diagnostics[stratum] = diagnostics
                    if is_forbidden_candidate(
                        candidate, forbidden_poses=forbidden_poses,
                        forbidden_hashes=forbidden_hashes,
                    ):
                        natural_errors[stratum] = (
                            "candidate_collides_with_consumed_B_or_C_identity"
                        )
                        continue
                    natural, selected_stratum = candidate, stratum
                    break
                except role_builder.NaturalNovelConstructionError as error:
                    natural_errors[stratum] = str(error)
                    natural_diagnostics[stratum] = error.diagnostics
            if natural is None:
                attrition_reason = "no_new_unsupported_novel_after_combined_AB"

        role_payload = None
        if attrition_reason is None:
            role_root = temporary / "role_pair" / scene / episode
            role_payload = role_builder.write_protocol_episode(
                destination=role_root,
                online_episode=prefix,
                receipt=receipt_ab,
                history=history_ab,
                natural=natural,
                revisit=selected_revisit,
                protocol="hm3d_table2_leg3_mixed_role_20260829",
                scene_rank=scene_rank,
                episode_rank=population_index,
            )
            role_payload["table2_source_population_index"] = population_index
            role_payload["table2_prefix_A_steps"] = len(trace_a["poses"])
            role_payload["table2_prefix_B_steps"] = len(trace_b["poses"])
            role_payload["table2_selected_revisit_segment"] = (
                "A" if int(selected_revisit["source_frame"])
                < len(trace_a["poses"]) else "B"
            )
            sidecar = role_root / "role_pairs.json"
            stored = dict(role_payload)
            stored.pop("role_pairs_sha256", None)
            sidecar.write_text(json.dumps(
                stored, indent=2, sort_keys=True, allow_nan=False,
            ) + "\n")
            role_payload = stored
            role_payload["role_pairs_sha256"] = sha256_file(sidecar)

        completion = {
            "schema_version": FRAGMENT_SCHEMA,
            "status": "complete",
            "population_index": population_index,
            "parent_scene_rank": scene_rank,
            "scene": scene,
            "episode": episode,
            "protocol_sha256": sha256_file(protocol_path),
            "source_population_sha256": sha256_file(population_path),
            "source_benchmark_sha256": sha256_file(benchmark_path),
            "source_online_A_trace_sha256": sha256_file(
                source_a / "online_a_trace.json"
            ),
            "source_online_B_trace_sha256": sha256_file(trace_b_path),
            "leg3_query_policy_outcomes_read": False,
            "old_goal_C_navigation_outcomes_read": False,
            "eligible": attrition_reason is None,
            "attrition_reason": attrition_reason,
            "preferred_stratum": order[0],
            "stratum_attempt_order": list(order),
            "selected_stratum": selected_stratum,
            "revisit_diagnostics": revisit_diagnostics,
            "natural_diagnostics": natural_diagnostics,
            "natural_errors": natural_errors,
            "combined_prefix_steps": len(history_ab["poses"]),
            "combined_prefix_receipt_sha256": sha256_file(
                prefix / "receipt.json"
            ),
            "combined_prefix_trace_sha256": sha256_file(
                prefix / "online_a_trace.json"
            ),
            "role_pair_sha256": (
                None if role_payload is None
                else role_payload["role_pairs_sha256"]
            ),
            "selected_revisit_segment": (
                None if selected_revisit is None else (
                    "A" if int(selected_revisit["source_frame"])
                    < len(trace_a["poses"]) else "B"
                )
            ),
            "forbidden_goal_hashes": sorted(forbidden_hashes),
        }
        completion_path = temporary / "completion.json"
        completion_path.write_text(json.dumps(
            completion, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        (temporary / "completion.json.sha256").write_text(
            sha256_file(completion_path) + "  completion.json\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        simulator.close()
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--population-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = construct(
        protocol_path=args.protocol.resolve(),
        source_root=args.source_root.resolve(),
        population_index=args.population_index,
        out=args.out.resolve(),
    )
    print(json.dumps({
        "population_index": result["population_index"],
        "scene": result["scene"],
        "episode": result["episode"],
        "eligible": result["eligible"],
        "attrition_reason": result["attrition_reason"],
        "selected_stratum": result["selected_stratum"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
