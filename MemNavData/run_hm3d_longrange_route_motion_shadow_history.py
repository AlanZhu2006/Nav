#!/usr/bin/env python3
"""Replay one consumed geometry-stop history with a no-authority PnP shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from MemNavData.hm3d_longrange_route_tangent_experiment import (
    audit_mono_depth,
    audit_route_tangent,
    require,
)
from MemNavData.run_final14_mono_factorial_episode import (
    load_payloads,
    load_rows,
    run_command,
)


SCHEMA = "hm3d_longrange_route_motion_shadow_history_v1_20260903"
PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_motion_shadow_protocol_v1_20260903"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def local_se2_delta(reference: dict, query: dict) -> dict[str, float]:
    dx = float(query["x"]) - float(reference["x"])
    dz = float(query["z"]) - float(reference["z"])
    yaw = float(reference["yaw"])
    return {
        "forward_m": float(-dx * math.sin(yaw) - dz * math.cos(yaw)),
        "left_m": float(-dx * math.cos(yaw) + dz * math.sin(yaw)),
        "yaw_rad": wrap_angle(float(query["yaw"]) - yaw),
        "translation_m": float(math.hypot(dx, dz)),
    }


def pose_error(shadow_motion: dict | None, truth: dict | None) -> dict | None:
    if not isinstance(shadow_motion, dict) or not isinstance(truth, dict):
        return None
    forward_error = float(shadow_motion["forward_m"]) - truth["forward_m"]
    left_error = float(shadow_motion["left_m"]) - truth["left_m"]
    yaw_error = wrap_angle(float(shadow_motion["yaw_rad"]) - truth["yaw_rad"])
    return {
        "translation_vector_error_m": float(math.hypot(
            forward_error, left_error)),
        "translation_magnitude_error_m": float(abs(
            float(shadow_motion["translation_m"]) - truth["translation_m"])),
        "yaw_error_deg": float(abs(math.degrees(yaw_error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluator-source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--formal-run-root", type=Path,
        default=(Path(os.environ["FORMAL_RUN_ROOT"])
                 if os.environ.get("FORMAL_RUN_ROOT") else None),
    )
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=(Path(os.environ["PROTOCOL"])
                 if os.environ.get("PROTOCOL") else None),
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        default=os.environ.get("EXPECTED_PROTOCOL_SHA256"),
    )
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--arms", default="mono_cec_route_tangent")
    parser.add_argument("--history-contract", default="causal_survey")
    parser.add_argument(
        "--role-pair-scope", default="table3_longrange_route_tangent")
    args = parser.parse_args()

    require(args.formal_run_root is not None,
            "formal parent run root was not supplied")
    require(args.protocol is not None,
            "diagnostic protocol was not supplied")
    require(args.expected_protocol_sha256 is not None,
            "expected diagnostic protocol hash was not supplied")
    require(args.arms == "mono_cec_route_tangent",
            "diagnostic must execute only the unchanged primary arm")
    require(args.history_contract == "causal_survey",
            "diagnostic requires the frozen causal survey")
    require(args.role_pair_scope == "table3_longrange_route_tangent",
            "diagnostic role-pair scope changed")
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "consumed diagnostic protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "consumed diagnostic protocol schema changed")
    require(args.history_index in protocol["selected_history_indices"],
            "history is outside the frozen geometry-stop diagnostic")
    formal_parent = protocol["formal_parent"]
    require(str(args.formal_run_root) == formal_parent["run_root"],
            "formal parent run root changed")
    parent_summary = args.formal_run_root / "result" / "summary.json"
    parent_verification = (
        args.formal_run_root / "result" / "independent_verification.json")
    require(parent_summary.is_file()
            and sha256_file(parent_summary)
            == formal_parent["summary_sha256"],
            "formal parent summary changed")
    require(parent_verification.is_file()
            and sha256_file(parent_verification)
            == formal_parent["independent_verification_sha256"],
            "formal parent independent verification changed")
    verification = json.loads(parent_verification.read_text())
    require(verification.get("verified") is True,
            "formal parent result is not independently verified")
    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256,
            "parent benchmark manifest changed")
    require(protocol["formal_parent"]["benchmark_manifest_sha256"]
            == args.expected_manifest_sha256,
            "protocol points to another benchmark")
    manifest = json.loads(manifest_path.read_text())
    require(0 <= args.history_index < len(manifest["episodes"]),
            "history index is outside the benchmark")
    item = manifest["episodes"][args.history_index]
    scene = str(item["scene"])
    episode = str(item["episode"])

    formal_matches = list((args.formal_run_root / "evaluation").glob(
        f"{args.history_index:03d}_*/completion.json"))
    require(len(formal_matches) == 1,
            "formal parent completion is missing or ambiguous")
    formal_path = formal_matches[0]
    formal = json.loads(formal_path.read_text())
    require(formal["scene"] == scene and formal["episode"] == episode,
            "formal completion identity changed")
    require(formal["outcomes"]["mono_cec_route_tangent"] == 0
            and formal["termination_reason"]["mono_cec_route_tangent"]
            == "geometry_stream_failure",
            "selected history is not a formal route geometry stop")

    source_episode = Path(item["online_a_episode"])
    receipt = json.loads((source_episode / "receipt.json").read_text())
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file()
            and sha256_file(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    navmesh = Path(item["runtime_navmesh"])
    require(navmesh.is_file()
            and sha256_file(navmesh) == item["runtime_navmesh_sha256"],
            "pinned runtime navmesh changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / label
    require(not output.exists(), f"diagnostic output exists: {output}")
    (output / "logs").mkdir(parents=True)
    arm_root = output / "mono_cec_route_tangent_shadow"
    arm_root.mkdir()

    command = [
        args.hab_python, "-u",
        str(args.evaluator_source_root
            / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--pinned_navmesh", str(navmesh),
        "--expected_pinned_navmesh_sha256", item[
            "runtime_navmesh_sha256"],
        "--host", "127.0.0.1",
        "--port", str(args.memnav_port),
        "--novel_port", str(args.navdp_port),
        "--server_backend", "hybrid_pose",
        "--success_dist", "1.0",
        "--max_steps", str(args.max_steps),
        "--exec_horizon", "8",
        "--trajectory_selector", "server",
        "--trajectory_selector_scope", "all",
        "--leg1_mode", "shared_trace",
        "--leg1_goal_source", "own",
        "--seed", "0",
        "--terminal_uturn", "off",
        "--terminal_visual_refine", "off",
        "--deterministic_plan_seeds",
        "--retrieval_override", "off",
        "--certified_cdec_rescue", "off",
        "--certified_stagnation_graph", "off",
        "--revisit_controller", "navdp_mixed",
        "--role_pair_scope", "table3_longrange_route_tangent",
        "--role_pair_query_role", "revisit",
        "--navdp_depth_source", "monocular_sidecar",
        "--hybrid_route", "certified_relocalization",
        "--revisit_adapter", "verified_bearing_v1",
        "--certified_guidance_mode", "monocular_route_tangent",
        "--out", str(arm_root),
    ]
    elapsed = run_command(command, output / "logs" / "eval.log")
    rows = load_rows(arm_root)
    require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
            "diagnostic did not run one hidden-role Revisit")
    payload = load_payloads(arm_root, episode, rows)["revisit"]
    require(payload.get("analysis_role_not_forwarded") is True,
            "analysis role reached the runtime")
    plans = payload["query_leg"]
    depth_audit = audit_mono_depth(
        plans, arm="mono_cec_route_tangent")
    route_audit = audit_route_tangent(plans)
    stop_plans = [
        plan for plan in plans if plan.get("geometry_stream_stop") is True
    ]
    require(len(stop_plans) == 1,
            "selected formal geometry stop was not reproduced exactly once")

    stop = stop_plans[0]
    edge = stop.get("local_tangent_failure_edge_diagnostic")
    require(isinstance(edge, dict),
            "geometry stop omitted the same-edge estimator diagnostic")
    require(edge.get("primary_motion_model")
            == "fundamental_magsac_then_depth_pnp_ransac",
            "diagnostic changed the primary route estimator")
    primary_validity = edge.get("primary_local_motion_validity")
    shadow = edge.get("unfiltered_pnp_shadow")
    require(isinstance(primary_validity, dict)
            and primary_validity.get("accepted") is False,
            "typed stop lacks a rejected primary estimate")
    require(isinstance(shadow, dict)
            and shadow.get("authority")
            == "diagnostic_only_no_control_effect",
            "unfiltered PnP shadow gained authority")

    pose_by_frame = {}
    for split in ("legA", "query"):
        for pose in payload["memory_traces"][split]:
            pose_by_frame[int(pose["frame_idx"])] = pose
    reference_frame = int(edge["match_reference_frame"])
    query_frame = int(edge["match_query_frame"])
    require(reference_frame in pose_by_frame and query_frame in pose_by_frame,
            "same-edge evaluator pose receipt is missing")
    truth = local_se2_delta(
        pose_by_frame[reference_frame], pose_by_frame[query_frame])
    failure = {
        "primary_stop_reproduced": True,
        "edge_kind": edge["edge_kind"],
        "reference_frame": reference_frame,
        "query_frame": query_frame,
        "reference_rgb_sha256": edge["match_reference_sha256"],
        "query_rgb_sha256": edge["match_query_sha256"],
        "primary_pnp": edge["primary_pnp"],
        "primary_local_motion_validity": primary_validity,
        "unfiltered_pnp_shadow": shadow,
        "evaluator_pose_interval": truth,
        "unfiltered_shadow_pose_error": pose_error(
            shadow.get("motion"), truth),
    }

    completion = {
        "schema_version": SCHEMA,
        "claim_scope": "consumed same-edge mechanism diagnostic only",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "protocol_sha256": args.expected_protocol_sha256,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "formal_parent_completion_sha256": sha256_file(formal_path),
        "formal_parent_geometry_stop": True,
        "primary_estimator": "fundamental_magsac_then_depth_pnp_ransac",
        "shadow_estimator": "direct_depth_pnp_ransac",
        "shadow_control_authority": False,
        "wall_time_seconds": elapsed,
        "outcome": int(rows[0]["reached"]),
        "termination_reason": rows[0]["termination_reason"],
        "final_goal_3d_dist_m": float(rows[0]["final_goal_3d_dist_m"]),
        "depth_audit": depth_audit,
        "route_audit": route_audit,
        "failure_edge": failure,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()
    path = output / "completion.json"
    path.write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  completion.json\n"
    )
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "primary_stop_reproduced": failure is not None,
        "unfiltered_shadow_valid": failure["unfiltered_pnp_shadow"]
        ["local_motion_validity"]["accepted"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
