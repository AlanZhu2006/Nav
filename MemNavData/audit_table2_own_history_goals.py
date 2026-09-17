"""Remeasure fixed C goals against BOTH already-recorded B histories.

This is an offline, consumed-case construction audit, not a navigation rerun.
GT geometry is used only for benchmark diagnostics, never sent to a policy.
No targets, endpoints, headings, images, or model parameters are changed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import ROOT, dump, load, sha, history_at


CASE = ROOT / ".diagnostics/table2_mixed_actual_local_20260911_v3/pLe4wQe7qrG"
MENUS = ROOT / ".diagnostics/table2_forward_only_audit_20260911_v1/render/C_after_revisit"


def audit(out, *, search=False):
    import io
    from PIL import Image
    from MemNavData.generate_twoleg import make_sim, render, cam_to_world_hab, covis_curve, covis_frac
    from MemNavData.build_shared_online_double_revisit import goal_world_points, jpeg_bytes, pixel_mae
    from MemNavData.build_final14_role_pair_scene import deterministic_pose_grid
    from MemNavData.audit_pt1_multinovel_capacity import distance_bin, shortest
    from MemNavData.generate_twoleg import first_path_yaw
    from MemNavData.final14_role_pair_contract import DEPTH_TOLERANCE_M, relative_direction_degrees
    from MemNavData.table2_balanced_sampling import novel_supported_as_task, REVISIT_VISUAL_BUDGET
    from MemNavData.table2_mixed_local import revisit_source_frames

    out.mkdir(parents=True, exist_ok=False)
    source = load(CASE / "source.json")
    bquery = load(CASE / "B_goals/revisit/query.json")
    a = history_at(bquery["prefix_root"])
    queries = [load(path) for path in sorted(MENUS.glob("*/*/query.json"))]
    if not queries:
        raise ValueError("No frozen C candidate menu")
    sim = make_sim(source["asset"], "", agent_radius=.30)
    height = source["camera_height_m"]
    inputs = {str(CASE / "source.json"): sha(CASE / "source.json"),
              str(CASE / "B_goals/revisit/query.json"): sha(CASE / "B_goals/revisit/query.json")}
    histories, rows = {}, []
    try:
        for arm in ("native", "cec"):
            trace_path = CASE / f"rollouts/B/revisit/{arm}/actual_trace.json"
            trace = load(trace_path)
            if not trace["reached"]:
                raise ValueError("This declared consumed fixture must have completed B")
            first = trace["poses"][0]
            np.testing.assert_allclose([first[k] for k in "xyz"], a["trace"]["end_position"], atol=1e-8, rtol=0)
            delta = relative_direction_degrees(first["yaw"], a["trace"]["end_yaw"])
            if abs(delta) > 1e-7:
                raise ValueError("The B trace does not join its original actual A")
            inputs[str(trace_path)] = sha(trace_path)
            transforms, depths = list(a["transforms"]), list(a["depths"])
            rgbs, poses = list(a["rgbs"]), list(a["poses"])
            for pose in trace["poses"]:
                camera = np.asarray([pose[k] for k in "xyz"]) + [0, height, 0]
                rgb, depth = render(sim, camera, pose["yaw"])
                import hashlib
                encoded = jpeg_bytes(rgb)
                if hashlib.sha256(encoded).hexdigest() != pose["jpg_sha256"]:
                    raise ValueError("Re-rendered executed RGB differs from the original trace")
                transforms.append(cam_to_world_hab(camera, pose["yaw"]))
                # Match the existing offline history materializer's uint16 PNG.
                depths.append(np.clip(np.asarray(depth, dtype=np.float64) * 10000., 0, 65535).astype(np.uint16).astype(np.float32) / 10000.)
                # The original materializer stores JPEG, and pixel_mae uses
                # its decoded pixels rather than the uncompressed render.
                rgbs.append(np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB")))
                poses.append(pose)
            current_camera = np.asarray(trace["end_position"]) + [0, height, 0]
            _, current_depth = render(sim, current_camera, trace["end_yaw"])
            histories[arm] = dict(trace=trace, transforms=transforms, depths=depths, rgbs=rgbs, poses=poses,
                                  current_t=cam_to_world_hab(current_camera, trace["end_yaw"]),
                                  current_depth=current_depth)
            print(f"HISTORY {arm}: A={len(a['poses'])}, B={len(trace['poses'])}", flush=True)
        if search:
            # Reuse the already frozen native spatial pool, including views
            # not chosen as its first valid per-cell target. This is a local
            # capacity probe, not a new formal or arm-independent task set.
            menus = load(MENUS / "construction.json")
            key = f"{source['scene']}/{source['episode']}/C/{Path(queries[0]['prefix_root']).name}"
            queries = [dict(c, analysis_role="novel", candidate_source="frozen_spatial_pool")
                       for c in menus["diagnostics"]["novel"]["spatial"]["candidates"]]
            h = histories["native"]
            frames = revisit_source_frames(len(h["poses"]))
            grids = [list(deterministic_pose_grid(key + f"/{frame}")) for frame in frames]
            revisit_candidates = []
            start = np.asarray(h["trace"]["end_position"])
            for attempt in range(max(map(len, grids), default=0)):
                for frame, grid in zip(frames, grids):
                    if attempt >= len(grid):
                        continue
                    radius, angle, dyaw = grid[attempt]
                    if radius > .8 or not 12 <= abs(dyaw) <= 45:
                        continue
                    pose = h["poses"][frame]
                    origin = np.asarray([pose[k] for k in "xyz"])
                    raw = origin + radius * np.asarray([math.cos(angle), 0, math.sin(angle)])
                    pos = np.asarray(sim.pathfinder.snap_point(raw))
                    if (not np.isfinite(pos).all() or not sim.pathfinder.is_navigable(pos)
                            or abs(pos[1]-start[1]) > .20 or np.linalg.norm((pos-raw)[[0, 2]]) > .20
                            or not .20 <= np.linalg.norm((pos-origin)[[0, 2]]) <= .80):
                        continue
                    path = shortest(sim.pathfinder, start, pos)
                    if path is None or distance_bin(path[0]) is None or np.max(np.abs(path[1][:, 1]-start[1])) > .20:
                        continue
                    yaw = (pose["yaw"] + math.radians(dyaw) + math.pi) % (2*math.pi) - math.pi
                    revisit_candidates.append(dict(floor_position=pos.tolist(), yaw_rad=yaw,
                        source_online_frame=frame, source_pose_attempt=attempt, analysis_role="revisit",
                        cell=distance_bin(path[0]) + "/unrestricted", candidate_source="original_pose_grid"))
                    if len(revisit_candidates) >= REVISIT_VISUAL_BUDGET:
                        break
                if len(revisit_candidates) >= REVISIT_VISUAL_BUDGET:
                    break
            queries += revisit_candidates
            print(f"POOL novel={len(queries)-len(revisit_candidates)} revisit={len(revisit_candidates)}", flush=True)
        for query in queries:
            position = np.asarray(query["floor_position"])
            rgb, depth = render(sim, position + [0, height, 0], query["yaw_rad"])
            import hashlib
            digest = hashlib.sha256(jpeg_bytes(rgb)).hexdigest()
            if not search and digest != query["goal_rgb_sha256"]:
                raise ValueError("Frozen C image changed")
            points = goal_world_points(depth, position + [0, height, 0], query["yaw_rad"])
            row = dict(goal_rgb=query.get("goal_rgb"), goal_sha256=digest,
                       intended_role=query["analysis_role"], original_cell=query["cell"], arms={},
                       floor_position=query["floor_position"], yaw_rad=query["yaw_rad"])
            source_mae_ok = True
            if search and query["analysis_role"] == "revisit":
                mae = pixel_mae(rgb, histories["native"]["rgbs"][query["source_online_frame"]])
                source_mae_ok = mae >= 5.
                row.update(source_online_frame=query["source_online_frame"],
                           source_pose_attempt=query["source_pose_attempt"], source_pixel_mae=mae)
            if not search:
                inputs[query["goal_rgb"]] = sha(query["goal_rgb"])
            for arm, h in histories.items():
                trace = h["trace"]
                path = shortest(sim.pathfinder, np.asarray(trace["end_position"]), position)
                if path is None:
                    raise ValueError("A fixed C goal became unreachable in the loaded asset")
                distance, route = path
                angle = relative_direction_degrees(first_path_yaw(route, np.asarray(trace["end_position"])), trace["end_yaw"])
                curve = covis_curve(points, h["transforms"], h["depths"], tol=DEPTH_TOLERANCE_M)
                visible = float(covis_frac(points, h["current_t"], h["current_depth"], tol=DEPTH_TOLERANCE_M))
                maximum, eligible = float(curve.max()), float(curve[8:].max())
                novel = novel_supported_as_task(len(points), curve, visible)
                revisit = source_mae_ok and len(points) > 0 and visible < .10 and .55 <= maximum <= .90 and eligible >= .55
                same_floor = bool(np.max(np.abs(route[:, 1] - trace["end_position"][1])) <= .20)
                geometry_ok = distance_bin(distance) is not None and same_floor
                role_ok = (novel and abs(angle) <= 60.) if query["analysis_role"] == "novel" else revisit
                row["arms"][arm] = dict(geodesic_m=float(distance), distance_bin=distance_bin(distance),
                    initial_route_angle_deg=angle, front=abs(angle) <= 60., history_frames=len(curve),
                    max_history_covis=maximum, max_runtime_eligible_covis=eligible,
                    max_A_covis=float(curve[:len(a["poses"])].max()), max_B_covis=float(curve[len(a["poses"]):].max()),
                    current_view_covis=visible, novel_support_compliant=bool(novel),
                    revisit_support_window_compliant=bool(revisit),
                    supported_by_own_history=eligible >= .55,
                    same_floor=same_floor, task_geometry_support_compliant=bool(geometry_ok and role_ok))
            row["common_geometry_support_compliant"] = all(v["task_geometry_support_compliant"] for v in row["arms"].values())
            rows.append(row)
            if not search or len(rows) % 40 == 0 or row["common_geometry_support_compliant"]:
                print(f"GOAL {len(rows)} {row['intended_role']} {row['original_cell']}: common={row['common_geometry_support_compliant']}", flush=True)
    finally:
        sim.close()
    n, g = (histories[arm]["trace"] for arm in ("native", "cec"))
    result = dict(scope="consumed actual B trajectories; C construction diagnostics only",
        diagnostic_candidate_search=search,
        navigation_rerun=False, formal_population=False, own_B_memory_used=True,
        shared_A_fixture=True, original_goals_modified=False,
        evaluated_frozen_goals_only=not search, inputs=inputs,
        endpoint_separation_m=float(np.linalg.norm(np.asarray(n["end_position"]) - g["end_position"])),
        endpoint_heading_separation_deg=abs(math.degrees(math.atan2(math.sin(n["end_yaw"] - g["end_yaw"]), math.cos(n["end_yaw"] - g["end_yaw"])))),
        queries=rows, query_count=len(rows),
        common_geometry_support_counts={role: sum(r["intended_role"] == role and r["common_geometry_support_compliant"] for r in rows) for role in ("novel", "revisit")},
        common_unique_goal_image_counts={role: len({r["goal_sha256"] for r in rows
            if r["intended_role"] == role and r["common_geometry_support_compliant"]}) for role in ("novel", "revisit")},
        limitation="One consumed successful B pair, not population evidence or an exhaustive common-goal search; no new navigation SR. Source-view perturbations are native-history based, not newly certified relative to GEM frames.")
    dump(out / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    audit(args.out.resolve(), search=args.search)
