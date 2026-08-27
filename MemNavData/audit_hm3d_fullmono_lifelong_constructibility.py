#!/usr/bin/env python3
"""Result-blind constructibility waterfall for the HM3D lifelong benchmark.

The audit deliberately reads no B/C/B2 policy result.  It reconstructs the
already-sealed geometric proposal process from the factual online-A traces and
reports where candidate histories are lost.  In scene mode the script also
requires its reconstructed v3 selection to match the immutable construction
receipt exactly.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import build_shared_online_double_revisit as history_tools
import build_shared_online_role_pairs as pair_tools
from build_final14_role_pair_scene import (
    ELIGIBLE_FRAME_FLOOR,
    END_MARGIN_FRAMES,
    SOURCE_FRAME_STRIDE,
    deterministic_pose_grid,
    source_frame_candidates,
)
from construct_hm3d_fullmono_lifelong_ab import (
    DEPTH_TOLERANCE_M,
    direction_stratum,
    load_histories,
    temporal_frame_indices,
)
from final14_role_pair_contract import relative_direction_degrees
from generate_twoleg import covis_curve
from hm3d_fullmono_lifelong import (
    bind_parent,
    load_protocol,
    require,
    select_donors,
    sha256_file,
)


SCHEMA = "hm3d_fullmono_lifelong_constructibility_audit_v1_20260827"
STAGES = (
    "temporal_proposal",
    "same_floor",
    "a_to_b_reachable",
    "b_to_c_reachable",
    "a_to_b_in_band",
    "b_to_c_in_band",
    "novel_support",
)


def make_pathfinder_sim(
    glb: str,
    *,
    agent_radius: float = 0.30,
    agent_height: float = 1.5,
):
    """Build the production-equivalent navmesh without creating a renderer."""

    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.enable_physics = False
    backend.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [agent])
    )
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = float(agent_radius)
    settings.agent_height = float(agent_height)
    require(simulator.recompute_navmesh(simulator.pathfinder, settings),
            "renderer-free navmesh construction failed")
    return simulator


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(values).items()))


def _selected_identity(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["donor_episode"]), int(row["donor_frame_index"])


def classify_measurement(
    *,
    floor_delta_m: float,
    a_to_b_reachable: bool,
    b_to_c_reachable: bool,
    a_to_b_geodesic_m: float | None,
    b_to_c_geodesic_m: float | None,
    max_recipient_a_covis: float | None,
    same_floor_tolerance_m: float,
    a_to_b_band_m: tuple[float, float],
    b_to_c_band_m: tuple[float, float],
    maximum_a_covis: float,
) -> tuple[list[str], str]:
    """Return cumulative passed stages and the first failed contract."""

    passed = ["temporal_proposal"]
    if floor_delta_m > same_floor_tolerance_m:
        return passed, "floor_mismatch"
    passed.append("same_floor")
    if not a_to_b_reachable:
        return passed, "a_to_b_unreachable"
    passed.append("a_to_b_reachable")
    if not b_to_c_reachable:
        return passed, "b_to_c_unreachable"
    passed.append("b_to_c_reachable")
    require(a_to_b_geodesic_m is not None, "reachable A-to-B has no distance")
    require(b_to_c_geodesic_m is not None, "reachable B-to-C has no distance")
    if not a_to_b_band_m[0] <= a_to_b_geodesic_m <= a_to_b_band_m[1]:
        return passed, "a_to_b_outside_band"
    passed.append("a_to_b_in_band")
    if not b_to_c_band_m[0] <= b_to_c_geodesic_m <= b_to_c_band_m[1]:
        return passed, "b_to_c_outside_band"
    passed.append("b_to_c_in_band")
    require(max_recipient_a_covis is not None, "measured candidate has no covis")
    if not max_recipient_a_covis < maximum_a_covis:
        return passed, "recipient_history_support_not_novel"
    passed.append("novel_support")
    return passed, "eligible"


def summarize_recipient_measurements(
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one recipient without retaining heavy per-candidate arrays."""

    stage_counts = {
        stage: sum(stage in row["passed_stages"] for row in measurements)
        for stage in STAGES
    }
    return {
        "candidate_stage_counts": stage_counts,
        "candidate_first_rejection": _counter(
            str(row["first_rejection"]) for row in measurements
        ),
        "recipient_reaches_stage": {
            stage: bool(stage_counts[stage]) for stage in STAGES
        },
    }


def aggregate_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    require(payloads, "constructibility audit has no scene payloads")
    for payload in payloads:
        require(payload.get("schema_version") == SCHEMA,
                "constructibility audit schema changed")
        require(payload.get("query_policy_outcomes_read") is False,
                "constructibility audit read query policy outcomes")
        require(payload.get("navigation_outcomes_read") is False,
                "constructibility audit read navigation outcomes")
        require(payload.get("sealed_selection_reproduced") is True,
                "constructibility audit did not reproduce sealed selection")

    recipients = [
        row for payload in payloads for row in payload["recipients"]
    ]
    candidate_stage_counts = {
        stage: sum(
            int(row["candidate_stage_counts"][stage]) for row in recipients
        )
        for stage in STAGES
    }
    recipient_stage_counts = {
        stage: sum(
            bool(row["recipient_reaches_stage"][stage]) for row in recipients
        )
        for stage in STAGES
    }
    first_rejection = collections.Counter()
    for row in recipients:
        first_rejection.update(row["candidate_first_rejection"])
    source_reasons = _counter(
        str(row["controlled_revisit_source_status"]) for row in recipients
    )
    scenes_with_selected = {
        str(payload["scene"])
        for payload in payloads
        if int(payload["selected_candidate_count"]) > 0
    }
    return {
        "schema_version": SCHEMA,
        "scope": "result-blind HM3D lifelong construction waterfall",
        "scene_fragments": len(payloads),
        "source_materialized_A_histories": len(recipients),
        "controlled_revisit_source_status": source_reasons,
        "candidate_stage_counts": candidate_stage_counts,
        "recipient_stage_counts": recipient_stage_counts,
        "candidate_first_rejection": dict(sorted(first_rejection.items())),
        "sealed_selected_candidates": sum(
            int(payload["selected_candidate_count"]) for payload in payloads
        ),
        "sealed_selected_recipients": sum(
            int(payload["selected_recipient_count"]) for payload in payloads
        ),
        "sealed_selected_scene_clusters": len(scenes_with_selected),
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "all_sealed_selections_reproduced": True,
    }


def reconstruct_revisit_position(
    simulator,
    history: dict[str, Any],
    *,
    scene: str,
    episode: str,
    selected: dict[str, Any],
) -> np.ndarray:
    """Reconstruct a sealed controlled-Revisit pose without rendering it."""

    frame = int(selected["source_frame"])
    attempt = int(selected["render_attempt"])
    grid = deterministic_pose_grid(f"{scene}/{episode}/{frame}")
    require(1 <= attempt <= len(grid), "sealed revisit attempt escaped grid")
    radius, direction, _yaw_offset = grid[attempt - 1]
    source_floor = np.asarray(history["floor_positions"][frame], dtype=np.float64)
    raw = source_floor + np.asarray([
        radius * math.cos(direction), 0.0, radius * math.sin(direction)
    ])
    snapped = np.asarray(simulator.pathfinder.snap_point(raw), dtype=np.float64)
    translation = float(np.linalg.norm(snapped[[0, 2]] - source_floor[[0, 2]]))
    require(abs(translation - float(selected["translation_m"])) <= 1e-4,
            "sealed revisit translation did not reconstruct")
    endpoint = np.asarray(history["trace"]["end_position"], dtype=np.float64)
    geometry = pair_tools.query_geometry(simulator.pathfinder, endpoint, snapped)
    require(geometry is not None, "sealed revisit geometry disappeared")
    require(abs(float(geometry[0]) - float(selected["query_geodesic_m"])) <= 1e-4,
            "sealed revisit distance did not reconstruct")
    return snapped


def measure_temporal_candidate(
    simulator,
    recipient: dict[str, Any],
    donor: dict[str, Any],
    revisit_position: np.ndarray,
    *,
    donor_frame_index: int,
    donor_temporal_rank: int,
    construction: dict[str, Any],
) -> dict[str, Any]:
    """Measure a frozen donor frame and expose each result-blind gate."""

    endpoint = np.asarray(recipient["trace"]["end_position"], dtype=np.float64)
    pose = donor["poses"][int(donor_frame_index)]
    donor_floor = np.asarray([pose[x] for x in ("x", "y", "z")], dtype=np.float64)
    floor_delta = abs(float(donor_floor[1] - endpoint[1]))
    same_floor_tolerance = float(construction.get("same_floor_tolerance_m", 0.20))
    a_to_b = None
    b_to_c = None
    support = None
    if floor_delta <= same_floor_tolerance:
        a_to_b = pair_tools.query_geometry(
            simulator.pathfinder, endpoint, donor_floor
        )
        if a_to_b is not None:
            b_to_c = pair_tools.query_geometry(
                simulator.pathfinder, donor_floor, revisit_position
            )
        if b_to_c is not None:
            goal_points = history_tools.goal_world_points(
                donor["depths"][donor_frame_index],
                donor["camera_positions"][donor_frame_index],
                float(pose["yaw"]),
            )
            curve = covis_curve(
                goal_points,
                recipient["transforms"],
                recipient["depths"],
                tol=DEPTH_TOLERANCE_M,
            )
            support = float(curve.max()) if len(curve) else 0.0

    a_band = tuple(float(x) for x in construction["recipient_A_to_B_geodesic_band_m"])
    b_band = tuple(float(x) for x in construction["B_to_C_geodesic_band_m"])
    maximum_support = float(construction["B_max_recipient_A_covis_exclusive"])
    passed, rejection = classify_measurement(
        floor_delta_m=floor_delta,
        a_to_b_reachable=a_to_b is not None,
        b_to_c_reachable=b_to_c is not None,
        a_to_b_geodesic_m=None if a_to_b is None else float(a_to_b[0]),
        b_to_c_geodesic_m=None if b_to_c is None else float(b_to_c[0]),
        max_recipient_a_covis=support,
        same_floor_tolerance_m=same_floor_tolerance,
        a_to_b_band_m=(a_band[0], a_band[1]),
        b_to_c_band_m=(b_band[0], b_band[1]),
        maximum_a_covis=maximum_support,
    )
    row = {
        "donor_episode": str(donor["receipt"]["episode"]),
        "donor_episode_rank": int(donor["episode_rank"]),
        "donor_frame_index": int(donor_frame_index),
        "donor_frame_temporal_rank": int(donor_temporal_rank),
        "goal_floor_position": [float(x) for x in donor_floor],
        "floor_delta_m": floor_delta,
        "a_to_b_geodesic_m": None if a_to_b is None else float(a_to_b[0]),
        "b_to_c_geodesic_m": None if b_to_c is None else float(b_to_c[0]),
        "max_recipient_a_covis": support,
        "passed_stages": passed,
        "first_rejection": rejection,
    }
    if rejection == "eligible":
        initial_bearing = float(a_to_b[1])
        recipient_yaw = float(recipient["trace"]["end_yaw"])
        row.update({
            "assigned_direction_stratum": direction_stratum(
                relative_direction_degrees(initial_bearing, recipient_yaw)
            ),
            "_position": donor_floor,
        })
    return row


def controlled_revisit_source_diagnostic(
    simulator,
    history: dict[str, Any],
) -> dict[str, Any]:
    sources = source_frame_candidates(simulator, history)
    endpoint = np.asarray(history["trace"]["end_position"], dtype=np.float64)
    sampled = list(range(
        ELIGIBLE_FRAME_FLOOR,
        len(history["poses"]) - END_MARGIN_FRAMES,
        SOURCE_FRAME_STRIDE,
    ))
    distances = []
    for frame in sampled:
        try:
            value = history_tools.goal_distance(
                simulator.pathfinder, endpoint, history["floor_positions"][frame]
            )
        except RuntimeError:
            continue
        distances.append(float(value))
    if sources:
        status = "constructible"
    elif not sampled:
        status = "no_runtime_eligible_sample"
    elif not distances:
        status = "runtime_samples_unreachable"
    elif max(distances) < 2.0:
        status = "runtime_history_extent_below_2m"
    elif min(distances) > 9.0:
        status = "runtime_history_extent_above_9m"
    else:
        status = "stride_or_geometry_gap"
    return {
        "status": status,
        "online_A_frames": len(history["poses"]),
        "runtime_window_samples": len(sampled),
        "reachable_runtime_window_samples": len(distances),
        "minimum_runtime_window_distance_m": min(distances) if distances else None,
        "maximum_runtime_window_distance_m": max(distances) if distances else None,
        "source_candidates": len(sources),
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
    require(0 <= scene_index < len(parent["scenes"]), "scene index out of range")
    scene = str(parent["scenes"][scene_index])
    fragment = construction_root / f"{scene_index:02d}_{scene}"
    completion_path = fragment / "completion.json"
    completion = json.loads(completion_path.read_text())
    require(completion.get("query_policy_outcomes_read") is False,
            "sealed construction read query outcomes")
    require(completion.get("protocol_sha256") == sha256_file(protocol_path),
            "sealed construction protocol changed")

    online_root = (
        parent_root / "construction" / "scenes" / f"{scene_index:02d}_{scene}"
        / "online_a"
    )
    if int(completion["materialized_A_histories"]) == 0:
        recipients: list[dict[str, Any]] = []
        reconstructed = int(completion["constructible_AB_C_histories"]) == 0
    else:
        _manifest, histories = load_histories(online_root, scene)
        attempts = {str(row["episode"]): row for row in completion["attempts"]}
        require(len(attempts) == len(histories), "sealed attempt count changed")
        asset = Path(parent["assets"][scene]["glb_path"])
        require(sha256_file(asset) == parent["assets"][scene]["glb_sha256"],
                "scene asset changed")
        simulator = make_pathfinder_sim(str(asset), agent_radius=0.30)
        recipients = []
        try:
            for recipient in histories:
                episode = str(recipient["receipt"]["episode"])
                attempt = attempts[episode]
                source = controlled_revisit_source_diagnostic(simulator, recipient)
                require((source["status"] == "constructible")
                        == bool(attempt["revisit_A_constructible"]),
                        "controlled-Revisit source diagnosis changed")
                measurements: list[dict[str, Any]] = []
                selected_rows: list[dict[str, Any]] = []
                if attempt["revisit_A_constructible"]:
                    revisit_position = reconstruct_revisit_position(
                        simulator, recipient, scene=scene, episode=episode,
                        selected=attempt["selected_revisit_A"],
                    )
                    for donor in histories:
                        if donor is recipient:
                            continue
                        indices = temporal_frame_indices(
                            len(donor["poses"]),
                            int(protocol["novel_b_construction"][
                                "temporal_samples_per_donor"]),
                        )
                        for temporal_rank, frame_index in enumerate(indices):
                            measurements.append(measure_temporal_candidate(
                                simulator,
                                recipient,
                                donor,
                                revisit_position,
                                donor_frame_index=frame_index,
                                donor_temporal_rank=temporal_rank,
                                construction=protocol["novel_b_construction"],
                            ))
                    eligible = [
                        row for row in measurements
                        if row["first_rejection"] == "eligible"
                    ]
                    construction = protocol["novel_b_construction"]
                    selected_rows = select_donors(
                        eligible,
                        recipient_episode=episode,
                        maximum_candidates=int(
                            construction["maximum_candidates_per_recipient"]),
                        maximum_per_donor=int(
                            construction["maximum_candidates_per_donor_history"]),
                        prefer_distinct_direction_strata=bool(
                            construction["prefer_distinct_initial_direction_strata"]),
                        minimum_planar_separation_m=float(
                            construction["minimum_candidate_planar_separation_m"]),
                        maximum_a_covis=float(
                            construction["B_max_recipient_A_covis_exclusive"]),
                        minimum_geodesic_m=float(
                            construction["recipient_A_to_B_geodesic_band_m"][0]),
                        maximum_geodesic_m=float(
                            construction["recipient_A_to_B_geodesic_band_m"][1]),
                        minimum_b_to_c_m=float(
                            construction["B_to_C_geodesic_band_m"][0]),
                        maximum_b_to_c_m=float(
                            construction["B_to_C_geodesic_band_m"][1]),
                    )
                sealed = [
                    _selected_identity(row) for row in attempt["selected_donors"]
                ]
                rebuilt = [_selected_identity(row) for row in selected_rows]
                require(rebuilt == sealed,
                        f"{scene}/{episode}: sealed donor selection changed")
                summary = summarize_recipient_measurements(measurements)
                recipients.append({
                    "scene": scene,
                    "episode": episode,
                    "controlled_revisit_source_status": source["status"],
                    "controlled_revisit_source_diagnostic": source,
                    "candidate_stage_counts": summary["candidate_stage_counts"],
                    "candidate_first_rejection": summary[
                        "candidate_first_rejection"],
                    "recipient_reaches_stage": summary["recipient_reaches_stage"],
                    "sealed_selected_candidates": len(sealed),
                    "sealed_selection_reproduced": rebuilt == sealed,
                })
        finally:
            simulator.close()
        reconstructed = all(
            row["sealed_selection_reproduced"] for row in recipients
        ) and sum(row["sealed_selected_candidates"] for row in recipients) == int(
            completion["constructible_AB_C_histories"]
        )

    payload = {
        "schema_version": SCHEMA,
        "scene": scene,
        "scene_index": int(scene_index),
        "protocol_sha256": sha256_file(protocol_path),
        "construction_completion_sha256": sha256_file(completion_path),
        "source_materialized_A_histories": int(
            completion["materialized_A_histories"]),
        "selected_candidate_count": int(
            completion["constructible_AB_C_histories"]),
        "selected_recipient_count": sum(
            bool(row["sealed_selected_candidates"]) for row in recipients
        ),
        "sealed_selection_reproduced": bool(reconstructed),
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "recipients": recipients,
    }
    require(payload["sealed_selection_reproduced"],
            f"{scene}: sealed selection was not reproduced")
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), f"audit output exists: {out}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out.with_name(out.name + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    scene_parser = subparsers.add_parser("scene")
    scene_parser.add_argument("--parent-root", type=Path, required=True)
    scene_parser.add_argument("--protocol", type=Path, required=True)
    scene_parser.add_argument("--construction-root", type=Path, required=True)
    scene_parser.add_argument("--scene-index", type=int, required=True)
    scene_parser.add_argument("--out", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
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
            "histories": result["source_materialized_A_histories"],
            "selected": result["selected_candidate_count"],
            "reproduced": result["sealed_selection_reproduced"],
        }, sort_keys=True))
        return

    paths = sorted(args.audit_root.glob("*/constructibility_audit.json"))
    require(len(paths) == args.expected_scenes,
            "constructibility audit fragment count changed")
    result = aggregate_payloads([json.loads(path.read_text()) for path in paths])
    require(not args.out.exists(), f"aggregate audit output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
