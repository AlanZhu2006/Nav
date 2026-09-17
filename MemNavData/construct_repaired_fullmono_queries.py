"""Construct all role pairs from one complete, newly executed mono-A scene.

Input A rollouts stay archived.  Only trace carriers, reproducible RGB history,
offline annotation depth and query assets are materialized here. No policy is
loaded and no query navigation outcome is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from collect_repaired_fullmono_a import SCHEMA as A_SCHEMA, scene_sources
from probe_repaired_fullmono_construction import (
    ANNOTATION, NAMESPACE, STRATA, choose_direction, measured_projection, save_candidate, sha,
)

SCHEMA = "repaired_fullmono_role_pair_construction_20260909_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def check_a_sources(plan, manifest, summary, verification):
    require(manifest["schema"] == A_SCHEMA and not manifest["queries_allowed"], "Wrong A collection")
    require(summary["completed"] and not summary["queries"] and verification["verified"], "Incomplete A collection")
    planned = scene_sources(plan, manifest["source_scene_rank"])
    actual = manifest["sources"]
    require(len(actual) == len(planned) == len(summary["goal_a"]) == len(verification["goal_a"]),
            "A/source count changed")
    for p, source, row, v in zip(planned, actual, summary["goal_a"], verification["goal_a"]):
        require(all(source[k] == p[k] for k in ("scene", "episode", "seed", "source_index")),
                "A task identity changed")
        require(all(row[k] == source[k] for k in ("scene", "episode", "seed", "source_index")),
                "A summary identity changed")
        require(v["source_index"] == source["source_index"] and v["reached"] == row["reached"],
                "A verification identity changed")
    return actual


def unpack_traces(collection, out, plan):
    manifest, summary = read(collection / "manifest.json"), read(collection / "summary.json")
    verification, archive = read(collection / "independent_verification.json"), read(collection / "archive_receipt.json")
    require(archive["completed"] and archive["exit_code"] == 0 and archive["all_member_hashes_readback_verified"],
            "A archive is not complete")
    require(sha(archive["archive"]) == archive["archive_sha256"], "A archive changed")
    require(manifest["runtime_source_receipt"]["sha256"] == plan["base_runtime_sha256"], "A used another runtime")
    sources = check_a_sources(plan, manifest, summary, verification)
    with tarfile.open(archive["archive"], "r:gz") as tar:
        for filename, expected in (("manifest.json", manifest), ("summary.json", summary),
                                   ("independent_verification.json", verification)):
            require(json.load(tar.extractfile("task/" + filename)) == expected, "A archive/receipt mismatch")
        for source, verified in zip(sources, verification["goal_a"]):
            name = f"goal_a/{source['scene']}/{source['episode']}/{source['episode']}_leg1_trace.json"
            raw = tar.extractfile("task/" + name).read()
            require(hashlib.sha256(raw).hexdigest() == verified["trace_sha256"], "A trace changed")
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(raw)
    save(out / "a_input_receipt.json", {"collection": str(collection), "manifest": manifest,
        "manifest_sha256": sha(collection / "manifest.json"), "archive": archive,
        "verification_sha256": sha(collection / "independent_verification.json")})
    return sources


def selected_query_ids(queries, novelty, direction):
    if direction is None:
        return []
    require(direction in novelty and novelty[direction]["constructible"], "Selected Novel is unavailable")
    revisit = [q["query_id"] for q in queries if q["analysis_role"] == "revisit"]
    require(len(revisit) == 1, "Expected one standard Revisit")
    selected = novelty[direction]["query_id"]
    require(sum(q["query_id"] == selected and q["analysis_role"] == "novel" for q in queries) == 1,
            "Selected Novel query is missing")
    return [selected, revisit[0]]


def construct_one(source, online, out):
    import build_final14_role_pair_scene as b
    from build_hm3d_covisibility_repaired import projection_intrinsics
    from shared_online_role_pair_contract import runtime_query, validate_query

    receipt = read(online / "receipt.json")
    history = b.history_tools.load_online_history(online, receipt)
    require(history["trace"]["reached"], "Cannot form a conditional history from failed A")
    out.mkdir(parents=True, exist_ok=False)
    sim = b.make_sim(source["asset"], "", agent_radius=.30)
    try:
        projections = {s: np.asarray(sim._sensors[s]._sensor_object.render_camera.projection_matrix)
                       for s in ("color", "depth")}
        np.testing.assert_allclose(projections["color"], projections["depth"], atol=1e-7, rtol=0)
        intrinsic = projection_intrinsics(projections["depth"], 480, 270)
        depths = []
        for i, camera in enumerate(history["camera_positions"]):
            rgb, depth = b.render(sim, camera, history["poses"][i]["yaw"])
            require(hashlib.sha256(b.history_tools.jpeg_bytes(rgb)).hexdigest() == history["poses"][i]["jpg_sha256"],
                    "Re-render is not the factual A RGB")
            depths.append(depth)
        history["depths"] = depths
        with measured_projection(b, intrinsic):
            selected, revisit_diagnostics = b.search_revisit_candidates(sim, history, scene=source["scene"],
                episode=source["episode"], camera_height=receipt["camera_height_m"])
            revisit, novelty, queries = selected["standard"], {}, []
            if revisit is not None:
                queries.append(save_candidate(b, out / "revisit", "revisit", revisit))
                for stratum in STRATA:
                    try:
                        candidate, diagnostic = b.sample_natural_novel(sim, history, scene=source["scene"],
                            episode=source["episode"], scene_rank=source["source_scene_rank"],
                            episode_rank=source["source_episode_rank"], paired_revisit_position=revisit["_position"],
                            camera_height=receipt["camera_height_m"], direction_stratum=stratum,
                            sampling_seed_namespace=NAMESPACE)
                    except b.NaturalNovelConstructionError as error:
                        novelty[stratum] = {"constructible": False, "counts": error.diagnostics}
                    else:
                        q = save_candidate(b, out / f"novel_{stratum}", "novel", candidate)
                        queries.append(q)
                        novelty[stratum] = {"constructible": True, "counts": diagnostic, "query_id": q["query_id"]}
            direction = choose_direction(source["scene"], source["episode"],
                                         [s for s, row in novelty.items() if row["constructible"]])
            for q in queries:
                validate_query(q)
                q["q_eligible"] = max(q["covis_curve"][ANNOTATION["eligible_frame_floor"]:])
                q["bin"] = "unsupported" if q["analysis_role"] == "novel" else "standard_revisit"
                require(not {"analysis_role", "q_eligible", "bin", "max_online_a_covis"}.intersection(runtime_query(q)),
                        "Analysis labels leak into runtime query")
                if q["analysis_role"] == "novel":
                    require(q["max_online_a_covis"] < .10, "Novel support changed")
        selected_ids = selected_query_ids(queries, novelty, direction)
        result = {"schema": SCHEMA, "completed": True, "navigation_rollouts": 0, "query_outcomes_read": False,
            "source": source, "source_index": source["source_index"], "history_index": source["source_index"],
            "scene": source["scene"], "episode": source["episode"], "online_a_episode": str(online),
            "online_a_steps": len(history["poses"]), "online_a_receipt_sha256": sha(online / "receipt.json"),
            "online_a_trace_sha256": sha(online / "online_a_trace.json"),
            "all_historical_rgb_rerender_hashes_match": True, "annotation_intrinsic": intrinsic.tolist(),
            "annotation_depth": "rerendered_float32", "annotation": ANNOTATION,
            "revisit_diagnostics": revisit_diagnostics, "novel_by_direction": novelty,
            "selected_novel_direction": direction, "selected_query_ids": selected_ids,
            "pair_constructible": len(selected_ids) == 2, "queries": queries}
        save(out / "construction.json", result)
        return result
    finally:
        sim.close()


def build(args):
    from materialize_hm3d_fullmono_online_a import materialize_scene
    require(sha(args.plan) == args.plan_sha256, "Source plan changed")
    plan = read(args.plan)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = unpack_traces(args.collection, args.out, plan)
    original_manifest = read(args.collection / "manifest.json")
    require(original_manifest["source_plan_sha256"] == args.plan_sha256, "A used another source plan")
    scene = sources[0]["scene"]
    materialized = materialize_scene(trace_root=args.out / "goal_a" / scene, scene=scene,
        asset=Path(sources[0]["asset"]), episode_root=Path(sources[0]["source_episode_root"]),
        source_episode_order=[s["episode"] for s in sources], out=args.out / "online_a",
        purpose="new actual repaired mono-A; role queries not evaluated")
    failures = {r["episode"]: r for r in materialized["attrition"]}
    reports = []
    for source in sources:
        if source["episode"] in failures:
            row = {"source_index": source["source_index"], "scene": scene, "episode": source["episode"],
                   "construction_complete": True, "pair_constructible": False,
                   "attrition": failures[source["episode"]], "query_rollouts": 0}
        else:
            online = args.out / "online_a" / scene / source["episode"]
            path = args.out / "queries" / scene / source["episode"]
            payload = construct_one(source, online, path)
            row = {"source_index": source["source_index"], "scene": scene, "episode": source["episode"],
                   "construction_complete": True, "pair_constructible": payload["pair_constructible"],
                   "construction": str(path / "construction.json"), "construction_sha256": sha(path / "construction.json"),
                   "selected_novel_direction": payload["selected_novel_direction"], "query_rollouts": 0}
        reports.append(row)
        print(json.dumps(row), flush=True)
    summary = {"schema": SCHEMA, "completed": True, "source_plan_sha256": args.plan_sha256,
        "source_scene_rank": original_manifest["source_scene_rank"], "source_count": len(sources),
        "materialization": materialized, "reports": reports, "query_rollouts": 0,
        "pair_count": sum(r["pair_constructible"] for r in reports)}
    save(args.out / "summary.json", summary)
    return summary


def verify(args):
    """Check saved identities, completeness, assets and the declared role bounds.

    RGB is checked against actual A hashes. This reader does not independently
    rerender the full geometric support curves; that remains a builder result.
    """
    from shared_online_role_pair_contract import validate_query
    plan = read(args.plan)
    require(sha(args.plan) == args.plan_sha256, "Source plan changed")
    summary, inputs = read(args.out / "summary.json"), read(args.out / "a_input_receipt.json")
    require(summary["schema"] == SCHEMA and summary["completed"] and summary["query_rollouts"] == 0,
            "Incomplete construction scene")
    require(summary["source_plan_sha256"] == args.plan_sha256, "Construction used another source plan")
    planned = scene_sources(plan, summary["source_scene_rank"])
    require(len(summary["reports"]) == len(planned) == summary["source_count"], "Source coverage changed")
    require(sha(Path(inputs["collection"]) / "manifest.json") == inputs["manifest_sha256"], "A manifest changed")
    checked = []
    for source, row in zip(planned, summary["reports"]):
        require(all(row[k] == source[k] for k in ("source_index", "scene", "episode")), "Constructed task identity changed")
        trace_path = args.out / "goal_a" / source["scene"] / source["episode"] / f"{source['episode']}_leg1_trace.json"
        trace = read(trace_path)
        if "attrition" in row:
            require(not row["pair_constructible"] and sha(trace_path) == row["attrition"]["trace_sha256"],
                    "Wrong A attrition")
            reason = row["attrition"]["reason"]
            if reason == "mono_a_failed":
                require(not trace["reached"], "Successful A mislabeled as failed")
            else:
                require(reason == "insufficient_history_for_runtime_anchor_contract"
                        and trace["reached"] and len(trace["poses"]) <= 55, "Unknown scientific attrition")
        else:
            path = Path(row["construction"])
            require(sha(path) == row["construction_sha256"], "Construction receipt changed")
            payload = read(path)
            require(payload["schema"] == SCHEMA and payload["completed"] and payload["navigation_rollouts"] == 0,
                    "Not a completed query-free construction")
            require(sha(trace_path) == payload["online_a_trace_sha256"], "History differs from actual A")
            online = Path(payload["online_a_episode"])
            require(sha(online / "online_a_trace.json") == sha(trace_path)
                    and sha(online / "receipt.json") == payload["online_a_receipt_sha256"], "Published history changed")
            for pose in trace["poses"]:
                require(sha(online / "rgb" / f"{pose['step']:06d}.jpg") == pose["jpg_sha256"], "Causal RGB changed")
            for q in payload["queries"]:
                validate_query(q)
                require(len(q["covis_curve"]) == len(trace["poses"]), "Support curve does not cover actual A")
                require(q["q_eligible"] == max(q["covis_curve"][39:]), "Eligible support differs from curve")
                if q["analysis_role"] == "novel":
                    require(max(q["covis_curve"]) == q["max_online_a_covis"] < .10, "Novel bound changed")
                else:
                    require(.55 <= q["q_eligible"] == q["max_online_a_covis"] <= .90, "Revisit bound changed")
                for key in ("goal_rgb", "goal_depth"):
                    require(sha(path.parent / q[key]) == q[key + "_sha256"], "Goal asset changed")
            available = [s for s, item in payload["novel_by_direction"].items() if item["constructible"]]
            direction = choose_direction(source["scene"], source["episode"], available)
            require(direction == payload["selected_novel_direction"], "Direction choice changed")
            selected_ids = selected_query_ids(payload["queries"], payload["novel_by_direction"], direction)
            require(selected_ids == payload["selected_query_ids"] and bool(selected_ids) == row["pair_constructible"],
                    "Formal pair selection changed")
        checked.append(dict(row, verified=True))
    require(summary["pair_count"] == sum(r["pair_constructible"] for r in checked), "Pair count differs")
    result = {"schema": SCHEMA, "verified": True, "source_count": len(checked),
        "source_scene_rank": summary["source_scene_rank"], "source_plan_sha256": args.plan_sha256,
        "summary_sha256": sha(args.out / "summary.json"), "query_rollouts": 0,
        "verification_scope": "complete source identities, actual A/RGB hashes, query assets, saved support bounds and fixed selection",
        "independent_geometric_rerender": False, "reports": checked, "pair_count": summary["pair_count"]}
    save(args.out / "independent_verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if not args.verify_only:
        build(args)
    print(json.dumps(verify(args), indent=2))
