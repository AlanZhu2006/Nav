"""Render a separately declared, available-direction construction diagnostic.

No policy imports or rollouts; old frozen constructors and histories remain
unchanged. Geometry uses the actual rendering camera and float depth.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCHEMA = "repaired_fullmono_available_direction_constructibility_20260909_v1"
NAMESPACE = "repaired_fullmono_available_direction_20260909"
STRATA = ("front", "side", "rear")
SOURCE_ROOT = Path("/scratch/yz11502/Research/Nav-axis-uturn-results")
SOURCES = (
    ("rJhMRvNn4DS", SOURCE_ROOT / "repaired_hm3d_gate_20260908/gate_ee865620b4540675/integration",
     "3a8eb572734e0144242b3eac3c5c30f4dddd08acbbe8f8826bb3a8dc0aab5bfc",
     "5d0d104de37440cc2bb86789be2ef742495367d0948a35eac7c92bec0395f348"),
    ("jgPBycuV1Jq", SOURCE_ROOT / "repaired_hm3d_extension_20260908/extension_c8cf8c60e7efd55f/shard_0/integration",
     "2362a5f3d485910d9ccdd29c50a3f0450df3f92306a040b1895f02f530c456a4",
     "0265f87e7281a78a416a49e820a26cf855417b0f7bff64295ad560c211e142f8"),
)
ANNOTATION = {"pixel_stride": 4, "depth_min_m_exclusive": .05,
              "depth_max_m_exclusive": 100., "occlusion_depth_tolerance_m": .30,
              "eligible_frame_floor": 39}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_new(path, payload):
    with Path(path).open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def choose_direction(scene, episode, available):
    if len(available) != len(set(available)) or any(s not in STRATA for s in available):
        raise ValueError("Invalid direction candidates")
    def key(s):
        return hashlib.sha256(f"repaired_fullmono_choose_20260909/{scene}/{episode}/{s}".encode()).hexdigest()
    return min(available, key=key) if available else None


@contextmanager
def measured_projection(builder, intrinsic):
    from build_hm3d_covisibility_repaired import surface_points, visibility
    from generate_twoleg import cam_to_world_hab

    def points(depth, camera, yaw):
        result = surface_points(depth, cam_to_world_hab(camera, yaw), intrinsic, ANNOTATION)
        if not len(result):
            raise ValueError("Target has no valid surface points")
        return result

    def fraction(points, transform, depth, tol=.30):
        return visibility(points, transform, depth, intrinsic, tol)

    def curve(points, transforms, depths, tol=.30):
        if len(transforms) != len(depths):
            raise ValueError("History geometry/depth cardinality mismatch")
        return np.asarray([fraction(points, t, d, tol) for t, d in zip(transforms, depths)])

    with patch.object(builder.history_tools, "goal_world_points", points), \
         patch.object(builder, "covis_frac", fraction), patch.object(builder, "covis_curve", curve):
        yield


def save_candidate(builder, folder, role, candidate):
    folder.mkdir(parents=True, exist_ok=False)
    rgb, depth = folder / "goal.jpg", folder / "annotation_depth.npy"
    rgb.write_bytes(builder.history_tools.jpeg_bytes(candidate["_rgb"]))
    np.save(depth, candidate["_depth"], allow_pickle=False)
    return builder._query_record(query_id=folder.name, role=role, candidate=candidate,
        episode_root=folder.parent, rgb_path=rgb, depth_path=depth,
        rgb_sha=sha(rgb), depth_sha=sha(depth))


def build(root):
    import build_final14_role_pair_scene as b
    from build_hm3d_covisibility_repaired import projection_intrinsics
    from shared_online_role_pair_contract import runtime_query, validate_query
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for scene, original, construction_sha, verification_sha in SOURCES:
        if sha(original / "construction_summary.json") != construction_sha \
                or sha(original / "independent_verification.json") != verification_sha:
            raise ValueError("Historical construction or verification changed")
        if not load(original / "independent_verification.json")["verified"]:
            raise ValueError("Original A was not verified")
        source = next(s for s in load(original / "manifest.json")["sources"] if s["scene"] == scene)
        episode = source["episode"]
        online = original / "construction" / scene / "online_a" / scene / episode
        receipt = load(online / "receipt.json")
        if sha(source["asset"]) != receipt["source_asset_sha256"]:
            raise ValueError("Scene asset changed")
        factual = original / "goal_a" / scene / episode / f"{episode}_leg1_trace.json"
        if sha(factual) != sha(online / "online_a_trace.json"):
            raise ValueError("Not the actual repaired mono-A trace")
        history = b.history_tools.load_online_history(online, receipt)
        if not history["trace"]["reached"]:
            raise ValueError("A failed")
        report = next(r for r in load(original / "construction_summary.json")["reports"] if r["scene"] == scene)
        prior = next(a for a in report["construction"]["attempts"] if a["episode"] == episode)
        folder = root / scene / episode
        folder.mkdir(parents=True)
        sim = b.make_sim(source["asset"], "", agent_radius=.30)
        try:
            projections = {s: np.asarray(sim._sensors[s]._sensor_object.render_camera.projection_matrix)
                           for s in ("color", "depth")}
            np.testing.assert_allclose(projections["color"], projections["depth"], atol=1e-7, rtol=0)
            intrinsic = projection_intrinsics(projections["depth"], 480, 270)
            depths = []
            for i, camera in enumerate(history["camera_positions"]):
                rgb, depth = b.render(sim, camera, history["poses"][i]["yaw"])
                if hashlib.sha256(b.history_tools.jpeg_bytes(rgb)).hexdigest() != history["poses"][i]["jpg_sha256"]:
                    raise ValueError(f"Factual RGB differs at {scene} frame {i}")
                depths.append(depth)
            history["depths"] = depths
            with measured_projection(b, intrinsic):
                selected, revisit_diag = b.search_revisit_candidates(sim, history, scene=scene,
                    episode=episode, camera_height=receipt["camera_height_m"])
                revisit = selected["standard"]
                novelty, queries = {}, []
                if revisit is not None:
                    q = save_candidate(b, folder / "revisit", "revisit", revisit)
                    validate_query(q)
                    queries.append(q)
                    for stratum in STRATA:
                        try:
                            candidate, diagnostics = b.sample_natural_novel(sim, history,
                                scene=scene, episode=episode, scene_rank=source["source_scene_rank"],
                                episode_rank=prior["source_episode_rank"],
                                paired_revisit_position=revisit["_position"], camera_height=receipt["camera_height_m"],
                                direction_stratum=stratum, sampling_seed_namespace=NAMESPACE)
                        except b.NaturalNovelConstructionError as error:
                            novelty[stratum] = {"constructible": False, "counts": error.diagnostics}
                        else:
                            q = save_candidate(b, folder / f"novel_{stratum}", "novel", candidate)
                            validate_query(q)
                            if q["max_online_a_covis"] >= .10:
                                raise ValueError("Novel support bound changed")
                            queries.append(q)
                            novelty[stratum] = {"constructible": True, "counts": diagnostics,
                                                "query_id": q["query_id"]}
                direction = choose_direction(scene, episode, [s for s, r in novelty.items() if r["constructible"]])
                for q in queries:
                    public = runtime_query(q)
                    if any(k in public for k in ("analysis_role", "max_online_a_covis", "assigned_direction_stratum")):
                        raise ValueError("Role/support leaked through runtime_query")
            payload = {"schema": SCHEMA, "scene": scene, "episode": episode, "source": source,
                "scope": "consumed two-history construction diagnostic; not confirmation",
                "source_original": str(original), "online_a_episode": str(online),
                "online_a_trace_sha256": sha(factual), "online_a_receipt_sha256": sha(online / "receipt.json"),
                "source_verification_sha256": verification_sha,
                "annotation_intrinsic": intrinsic.tolist(), "annotation": ANNOTATION,
                "all_historical_rgb_rerender_hashes_match": True, "annotation_depth": "rerendered_float32",
                "prior_assigned_direction": b.assigned_direction_stratum(source["source_scene_rank"], prior["source_episode_rank"]),
                "revisit_diagnostics": revisit_diag, "novel_by_direction": novelty,
                "selected_novel_direction": direction, "pair_constructible": revisit is not None and direction is not None,
                "queries": queries, "navigation_rollouts": 0, "query_outcomes_read": False}
            dump_new(folder / "construction.json", payload)
            reports.append({"scene": scene, "path": str(folder / "construction.json"),
                            "sha256": sha(folder / "construction.json"), "pair_constructible": payload["pair_constructible"],
                            "selected_novel_direction": direction})
            print(json.dumps(reports[-1]), flush=True)
        finally:
            sim.close()
    summary = {"schema": SCHEMA, "completed": True, "navigation_rollouts": 0,
               "query_outcomes_read": False, "scene_count": len(reports),
               "pair_count": sum(r["pair_constructible"] for r in reports), "reports": reports,
               "protocol_sha256": sha(Path(__file__).with_name("REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md"))}
    dump_new(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.out), indent=2))
