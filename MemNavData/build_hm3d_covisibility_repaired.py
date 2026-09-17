"""Construct fixed-position, varying-yaw support queries without model inference.

GT geometry is confined to offline annotation. Runtime history remains the
original causal RGB stream. Importing this module needs only NumPy, not Habitat.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "hm3d_covisibility_repaired_protocol_20260909.json"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def dump_new(path, payload):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def support_bin(value, protocol):
    require(math.isfinite(value) and 0 <= value <= 1, "invalid support")
    edges = protocol["bin_edges"]
    for i, label in enumerate(protocol["bin_ids"]):
        if edges[i] <= value < edges[i+1] or (i == len(edges)-2 and value == edges[-1]):
            return label
    return None


def select_candidates(rows, history_id, protocol):
    selected = {}
    for label in protocol["bin_ids"]:
        candidates = [row for row in rows if row["bin"] == label and not row["exact_history_jpeg"]]
        def key(row):
            text = f"{protocol['selection_seed']}/{history_id}/{label}/{row['yaw_index']}"
            return hashlib.sha256(text.encode()).hexdigest()
        selected[label] = min(candidates, key=key) if candidates else None
    return selected


def projection_intrinsics(matrix, width, height):
    p = np.asarray(matrix, dtype=np.float64)
    require(p.shape == (4, 4) and p[0, 0] > 0 and p[1, 1] > 0, "invalid projection matrix")
    require(abs(p[0, 2]) < 1e-7 and abs(p[1, 2]) < 1e-7, "expected centered sensor")
    return np.array([[p[0, 0] * width / 2, 0, width / 2],
                     [0, p[1, 1] * height / 2, height / 2], [0, 0, 1.]])


def surface_points(depth, camera_to_world, intrinsic, protocol):
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(0, h, protocol["pixel_stride"]),
                       np.arange(0, w, protocol["pixel_stride"]), indexing="ij")
    d = depth[v, u].astype(np.float64)
    valid = np.isfinite(d) & (d > protocol["depth_min_m_exclusive"]) & (d < protocol["depth_max_m_exclusive"])
    x = (u[valid] - intrinsic[0, 2]) * d[valid] / intrinsic[0, 0]
    y = -(v[valid] - intrinsic[1, 2]) * d[valid] / intrinsic[1, 1]
    local = np.stack([x, y, -d[valid]], axis=1)
    return local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def visibility(points, transform, depth, intrinsic, tolerance):
    if not len(points):
        return 0.0
    pc = (points - transform[:3, 3]) @ transform[:3, :3]
    z = -pc[:, 2]
    denom = np.maximum(z, 1e-6)
    u = intrinsic[0, 0] * pc[:, 0] / denom + intrinsic[0, 2]
    v = -intrinsic[1, 1] * pc[:, 1] / denom + intrinsic[1, 2]
    h, w = depth.shape
    inside = (z > .05) & (u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)
    ui = np.clip(u.astype(int), 0, w-1)
    vi = np.clip(v.astype(int), 0, h-1)
    observed = depth[vi, ui]
    visible = inside & np.isfinite(observed) & (observed > 0) & (np.abs(z-observed) <= tolerance)
    return float(np.count_nonzero(visible) / len(points))


def annotate(depth, camera_to_world, transforms, depths, intrinsic, protocol):
    points = surface_points(depth, camera_to_world, intrinsic, protocol)
    require(len(points) > 0, "goal has no valid surface points")
    curve = np.array([visibility(points, t, d, intrinsic, protocol["occlusion_depth_tolerance_m"])
                      for t, d in zip(transforms, depths)], dtype=float)
    floor = protocol["eligible_frame_floor"]
    require(len(curve) > floor, "history has no eligible frames")
    best = floor + int(np.argmax(curve[floor:]))
    return {"q_eligible": float(curve[best]), "argmax_eligible": best,
            "q_all": float(curve.max()), "argmax_all": int(curve.argmax()),
            "valid_goal_points": len(points), "covis_curve": curve.tolist()}


def source_manifest(protocol):
    path = Path(protocol["source_manifest"])
    require(sha(path) == protocol["source_manifest_sha256"], "frozen source manifest changed")
    manifest = load(path)
    episodes = manifest["episodes"]
    require(len(episodes) == protocol["histories"], "wrong history count")
    require(len({h["scene"] for h in episodes}) == protocol["scenes"], "wrong scene count")
    require(len({(h["scene"], h["episode"]) for h in episodes}) == len(episodes), "duplicate history")
    return path, episodes


def build(index, root, protocol_path=PROTOCOL):
    import generate_twoleg as geo
    import build_shared_online_double_revisit as history_tools
    import build_shared_online_role_pairs as pairs

    started = time.monotonic()
    protocol = load(protocol_path)
    manifest_path, episodes = source_manifest(protocol)
    require(0 <= index < len(episodes), "invalid fixed history index")
    h = episodes[index]
    source = Path(h["online_a_episode"])
    require(sha(source/"receipt.json") == h["online_a_receipt_sha256"], "history receipt changed")
    require(sha(source/"online_a_trace.json") == h["online_a_trace_sha256"], "A trajectory changed")
    receipt = load(source/"receipt.json")
    require(sha(receipt["source_asset"]) == receipt["source_asset_sha256"], "scene asset changed")
    history = history_tools.load_online_history(source, receipt)
    require(history["trace"]["reached"] is True, "source A did not reach its goal")
    query_root = manifest_path.parent/h["scene"]/h["episode"]
    require(sha(query_root/"role_pairs.json") == h["role_pairs_sha256"], "original query identity changed")
    queries = [q for pair in h["pairs"] for q in pair["queries"]]
    originals = {role: [q for q in queries if q["analysis_role"] == role] for role in ("novel", "revisit")}
    require(all(len(q) == 1 for q in originals.values()), "expected one original role pair")
    revisit, novel = originals["revisit"][0], originals["novel"][0]
    out = root/f"history_{index:02d}"
    out.mkdir(parents=True, exist_ok=False)
    sim = geo.make_sim(receipt["source_asset"], "", agent_radius=.30)
    try:
        sensors = sim._sensors
        projection = np.array(sensors["depth"]._sensor_object.render_camera.projection_matrix)
        color_projection = np.array(sensors["color"]._sensor_object.render_camera.projection_matrix)
        np.testing.assert_allclose(projection, color_projection, atol=1e-7, rtol=0)
        intrinsic = projection_intrinsics(projection, geo.W, geo.H)
        # Re-render annotation depth, not the policy's RGB history. Avoid PNG saturation.
        rendered_depths = []
        for frame, camera in enumerate(history["camera_positions"]):
            rgb, depth = geo.render(sim, camera, history["poses"][frame]["yaw"])
            require(hashlib.sha256(history_tools.jpeg_bytes(rgb)).hexdigest() == history["poses"][frame]["jpg_sha256"],
                    f"history re-render no longer matches original RGB: {frame}")
            rendered_depths.append(depth)
        position = np.asarray(revisit["floor_position"], dtype=float)
        camera = position + [0., receipt["camera_height_m"], 0.]
        start = np.asarray(h["online_a_endpoint"]["floor_position"], dtype=float)
        distance, bearing = pairs.query_geometry(sim.pathfinder, start, position)
        require(abs(distance-revisit["geodesic_from_a_end_m"]) < .05, "goal geodesic changed")
        historical_sha = {pose["jpg_sha256"] for pose in history["poses"]}
        candidates = []
        for yaw_index in range(protocol["candidate_count"]):
            yaw = history_tools.wrap_radians(revisit["yaw_rad"] + math.radians(yaw_index*protocol["yaw_grid_step_deg"]))
            rgb, depth = geo.render(sim, camera, yaw)
            jpg_sha = hashlib.sha256(history_tools.jpeg_bytes(rgb)).hexdigest()
            info = annotate(depth, geo.cam_to_world_hab(camera, yaw), history["transforms"],
                            rendered_depths, intrinsic, protocol)
            row = dict(info, yaw_index=yaw_index, yaw_rad=yaw, goal_rgb_sha256=jpg_sha,
                       exact_history_jpeg=jpg_sha in historical_sha, bin=support_bin(info["q_eligible"], protocol))
            candidates.append(row)
        selected = select_candidates(candidates, f"{h['scene']}/{h['episode']}", protocol)
        outputs = []
        for label, row in selected.items():
            if row is None:
                continue
            folder = out/label
            folder.mkdir()
            rgb, depth = geo.render(sim, camera, row["yaw_rad"])
            jpg = history_tools.jpeg_bytes(rgb)
            require(hashlib.sha256(jpg).hexdigest() == row["goal_rgb_sha256"], "selected render changed")
            (folder/"goal.jpg").write_bytes(jpg)
            np.save(folder/"annotation_depth.npy", depth, allow_pickle=False)
            query = dict(row, query_id=label, analysis_role="revisit", floor_position=position.tolist(),
                         geodesic_from_a_end_m=float(distance), initial_path_bearing_rad=float(bearing),
                         goal_rgb=f"{label}/goal.jpg", goal_depth=f"{label}/annotation_depth.npy",
                         goal_depth_sha256=sha(folder/"annotation_depth.npy"), max_online_a_covis=row["q_eligible"],
                         eligible_online_a_frame_floor=protocol["eligible_frame_floor"])
            outputs.append(query)
        # Original Natural Novel is an extra, separately reported control, not a support-bin substitute.
        folder = out/"natural_novel"
        folder.mkdir()
        for field in ("goal_rgb", "goal_depth"):
            require(sha(query_root/novel[field]) == novel[field+"_sha256"], "original Novel asset changed")
        novel_camera = np.asarray(novel["floor_position"]) + [0., receipt["camera_height_m"], 0.]
        rgb, depth = geo.render(sim, novel_camera, novel["yaw_rad"])
        require(hashlib.sha256(history_tools.jpeg_bytes(rgb)).hexdigest() == novel["goal_rgb_sha256"], "Novel re-render differs")
        info = annotate(depth, geo.cam_to_world_hab(novel_camera, novel["yaw_rad"]), history["transforms"],
                        rendered_depths, intrinsic, protocol)
        (folder/"goal.jpg").write_bytes((query_root/novel["goal_rgb"]).read_bytes())
        np.save(folder/"annotation_depth.npy", depth, allow_pickle=False)
        control = dict(novel, **info)
        control.update(query_id="natural_novel", bin=None, goal_rgb="natural_novel/goal.jpg",
                       goal_depth="natural_novel/annotation_depth.npy", goal_depth_sha256=sha(folder/"annotation_depth.npy"),
                       max_online_a_covis=info["q_eligible"], original_annotation_max=novel["max_online_a_covis"])
        outputs.append(control)
        dump_new(out/"candidate_scan.json", candidates)
        payload = {"schema": protocol["schema"], "completed": True, "phase": "construction_only_no_SR",
                   "history_index": index, "scene": h["scene"], "episode": h["episode"],
                   "scope": protocol["scope"], "protocol_sha256": sha(protocol_path),
                   "source_manifest_sha256": protocol["source_manifest_sha256"],
                   "online_a_episode": str(source), "online_a_steps": len(history["poses"]),
                   "online_a_receipt_sha256": h["online_a_receipt_sha256"],
                   "online_a_trace_sha256": h["online_a_trace_sha256"],
                   "online_a_endpoint": h["online_a_endpoint"], "source": {
                       "scene": h["scene"], "episode": h["episode"], "asset": receipt["source_asset"],
                       "source_episode": receipt["source_episode"], "seed": int(history["trace"]["episode_seed"])},
                   "all_historical_rgb_rerender_hashes_match": True, "projection_matrix": projection.tolist(),
                   "annotation_intrinsic": intrinsic.tolist(), "annotation_depth_source": "rerendered_float32",
                   "fixed_revisit_goal_position": position.tolist(), "fixed_revisit_geodesic_m": float(distance),
                   "missing_bins": {label: "no nonidentical goal in fixed 72-yaw grid"
                                    for label, row in selected.items() if row is None},
                   "queries": outputs, "candidate_scan_sha256": sha(out/"candidate_scan.json"),
                   "navigation_rollouts": 0, "wall_seconds": time.monotonic()-started}
        dump_new(out/"construction.json", payload)
        print(json.dumps({"history": index, "scene": h["scene"], "selected": {
            q["query_id"]: round(q["q_eligible"], 5) for q in outputs}, "seconds": payload["wall_seconds"]}), flush=True)
    finally:
        sim.close()


def seal(root, protocol_path=PROTOCOL):
    protocol = load(protocol_path)
    _, sources = source_manifest(protocol)
    records, tasks = [], []
    bins = {name: [] for name in protocol["bin_ids"]}
    for index, h in enumerate(sources):
        path = root/f"history_{index:02d}"/"construction.json"
        item = load(path)
        require(item["completed"] and item["protocol_sha256"] == sha(protocol_path), "incomplete or changed construction")
        require((item["scene"], item["episode"]) == (h["scene"], h["episode"]), "wrong source order")
        scan_path = path.parent/"candidate_scan.json"
        require(sha(scan_path) == item["candidate_scan_sha256"], "candidate scan changed")
        selection = select_candidates(load(scan_path), f"{h['scene']}/{h['episode']}", protocol)
        for query in item["queries"]:
            for field in ("goal_rgb", "goal_depth"):
                require(sha(path.parent/query[field]) == query[field+"_sha256"], "selected goal changed")
            label = query["bin"]
            if label is not None:
                require(support_bin(query["q_eligible"], protocol) == label, "wrong bin")
                require(query["yaw_index"] == selection[label]["yaw_index"], "selection was changed")
                require(query["floor_position"] == item["fixed_revisit_goal_position"], "physical goal differs across bins")
                bins[label].append({"history_index": index, "scene": h["scene"], "q": query["q_eligible"]})
            tasks.append({"history_index": index, "query_id": query["query_id"],
                          "construction": str(path), "construction_sha256": sha(path)})
        expected_ids = {label for label, selected in selection.items() if selected is not None} | {"natural_novel"}
        require({q["query_id"] for q in item["queries"]} == expected_ids, "missing/extra selected query")
        records.append({"path": str(path), "sha256": sha(path), "missing_bins": item["missing_bins"]})
    # Interleave bins/history on the cluster; assignment never depends on SR.
    tasks.sort(key=lambda t: hashlib.sha256(f"{protocol['selection_seed']}/{t['history_index']}/{t['query_id']}/schedule".encode()).hexdigest())
    permutations = list(itertools.permutations(protocol["arms"]))
    for index, task in enumerate(tasks):
        task.update(task_index=index, arm_order=permutations[index % len(permutations)])
    common = set(range(len(sources)))
    statistics = {}
    for label, values in bins.items():
        common &= {v["history_index"] for v in values}
        q = [v["q"] for v in values]
        statistics[label] = {"histories": len(q), "scenes": len({v["scene"] for v in values}),
                             "min": min(q) if q else None, "median": float(np.median(q)) if q else None,
                             "max": max(q) if q else None, "values": values}
    result = {"schema": protocol["schema"], "scope": protocol["scope"], "phase": "sealed_before_query_evaluation",
              "protocol_sha256": sha(protocol_path), "source_manifest_sha256": protocol["source_manifest_sha256"],
              "histories": records, "bins": statistics, "complete_five_bin_history_indices": sorted(common),
              "tasks": tasks, "query_count": len(tasks), "query_arm_count": len(tasks)*len(protocol["arms"]),
              "arms": protocol["arms"], "navigation_rollouts_at_seal": 0}
    dump_new(root/"population.json", result)
    print(json.dumps({"query_count": len(tasks), "query_arm_count": result["query_arm_count"],
                      "bins": {b: v["histories"] for b, v in statistics.items()}, "common": len(common)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "seal"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.stage == "build":
        require(args.index is not None, "--index required for construction")
        build(args.index, args.root.resolve(), args.protocol)
    else:
        seal(args.root.resolve(), args.protocol)
