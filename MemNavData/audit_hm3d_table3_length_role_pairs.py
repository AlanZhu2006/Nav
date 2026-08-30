#!/usr/bin/env python3
"""Independent renderer-free audit of the sealed HM3D length population."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from hm3d_table3_length_contract import runtime_query, validate_manifest


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_curve(query: dict, steps: int, label: str) -> None:
    curve = query.get("covis_curve")
    require(isinstance(curve, list) and len(curve) == steps,
            f"{label}: co-visibility curve length changed")
    values = [float(value) for value in curve]
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in values),
            f"{label}: invalid co-visibility curve")
    floor = int(query["eligible_online_a_frame_floor"])
    end_margin = int(query["eligible_online_a_end_margin_frames"])
    require(end_margin >= 0 and floor < steps - end_margin,
            f"{label}: eligible support interval is empty")
    eligible = values[floor:steps - end_margin]
    maximum = max(eligible)
    require(abs(maximum - float(query["max_online_a_covis"])) <= 1e-12,
            f"{label}: eligible support maximum changed")
    require(floor + eligible.index(maximum)
            == int(query["max_online_a_covis_frame"]),
            f"{label}: eligible support argmax changed")


def audit(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    digest = sha256(manifest_path)
    require((root / "manifest.json.sha256").read_text().split()
            == [digest, "manifest.json"], "manifest receipt changed")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    contract = manifest["contract"]
    counts = Counter()
    scenes: dict[str, set[str]] = {}
    for episode in manifest["episodes"]:
        scene, name = str(episode["scene"]), str(episode["episode"])
        counts[episode["bin_name"]] += 1
        scenes.setdefault(episode["bin_name"], set()).add(scene)
        episode_root = root / scene / name
        sidecar = episode_root / "role_pairs.json"
        copy = dict(episode)
        sidecar_sha = copy.pop("role_pairs_sha256")
        require(sha256(sidecar) == sidecar_sha,
                f"{scene}/{name}: sidecar hash changed")
        require(json.loads(sidecar.read_text()) == copy,
                f"{scene}/{name}: sidecar/manifest mismatch")
        navmesh = Path(episode["runtime_navmesh"])
        require(navmesh.is_file()
                and sha256(navmesh) == episode["runtime_navmesh_sha256"],
                f"{scene}/{name}: pinned runtime navmesh changed")
        online = Path(episode["online_a_episode"])
        require(sha256(online / "receipt.json")
                == episode["online_a_receipt_sha256"],
                f"{scene}/{name}: online receipt changed")
        require(sha256(online / "online_a_trace.json")
                == episode["online_a_trace_sha256"],
                f"{scene}/{name}: online trace changed")
        trace = json.loads((online / "online_a_trace.json").read_text())
        require(trace["reached"] is True
                and len(trace["poses"]) == int(episode["online_a_steps"]),
                f"{scene}/{name}: online history changed")
        if contract["online_history"] == "controlled_causal_rgb_geodesic_survey":
            receipt = json.loads((online / "receipt.json").read_text())
            require(
                receipt.get("schema_version")
                == "hm3d_table3_causal_survey_materialized_v1_20260830"
                and receipt.get("history_source")
                == "controlled_causal_rgb_geodesic_survey"
                and trace.get("schema_version")
                == "hm3d_table3_causal_survey_trace_v1_20260830"
                and trace.get("source_hybrid_route") == "causal_survey",
                f"{scene}/{name}: causal-survey provenance changed",
            )
            survey = receipt.get("survey_contract")
            intrinsic = receipt.get("camera_intrinsic")
            require(
                isinstance(survey, dict)
                and survey.get("runtime_memory_input") == "RGB only"
                and survey.get("construction_only_simulator_depth") is True
                and survey.get("metric_depth_for_query_control_or_CEC") is False
                and int(trace.get("metric_depth_sensor_reads", -1)) == 0,
                f"{scene}/{name}: survey runtime modality changed",
            )
            require(
                isinstance(intrinsic, list) and len(intrinsic) == 3
                and all(isinstance(row, list) and len(row) == 3
                        for row in intrinsic)
                and all(math.isfinite(float(value))
                        for row in intrinsic for value in row)
                and int(receipt.get("episode_seed", -1))
                == int(trace.get("episode_seed", -2)) >= 0,
                f"{scene}/{name}: survey camera/seed receipt changed",
            )
            hashes = receipt.get("rgb_frame_hashes")
            require(isinstance(hashes, list) and len(hashes) == len(trace["poses"]),
                    f"{scene}/{name}: survey RGB receipt changed")
            for pose, expected in zip(trace["poses"], hashes):
                step = int(pose["step"])
                rgb = online / "rgb" / f"{step:06d}.jpg"
                require(sha256(rgb) == expected == pose["jpg_sha256"],
                        f"{scene}/{name}: survey RGB changed at {step}")
        for query in episode["pairs"][0]["queries"]:
            label = f"{scene}/{name}/{query['analysis_role']}"
            rgb = episode_root / query["goal_rgb"]
            depth = episode_root / query["goal_depth"]
            require(sha256(rgb) == query["goal_rgb_sha256"]
                    and sha256(depth) == query["goal_depth_sha256"],
                    f"{label}: goal asset changed")
            finite_curve(query, int(episode["online_a_steps"]), label)
            projected = runtime_query(query)
            require("analysis_role" not in projected
                    and "covis_curve" not in projected,
                    f"{label}: runtime role/support leak")
    minimum_histories = int(contract["minimum_histories_per_bin"])
    minimum_scenes = int(contract["minimum_scene_clusters_per_bin"])
    for spec in contract["bins_m"]:
        name = spec["name"]
        require(counts[name] >= minimum_histories,
                f"{name}: history power gate failed")
        require(len(scenes.get(name, set())) >= minimum_scenes,
                f"{name}: scene-cluster power gate failed")
    return {
        "schema_version": "hm3d_table3_length_role_pair_audit_v1_20260830",
        "ok": True, "manifest_sha256": digest,
        "histories_by_bin": dict(counts),
        "scene_clusters_by_bin": {key: len(value) for key, value in scenes.items()},
        "query_policy_outcomes_read": False,
        "runtime_role_visibility": "none",
        "online_history": contract["online_history"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
