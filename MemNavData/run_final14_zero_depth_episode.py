#!/usr/bin/env python3
"""Run the missing zero-depth native arm on one frozen Final14 history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from MemNavData.final14_mono_factorial import require
from MemNavData.final14_zero_depth import (
    ARM,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    audit_zero_depth_plans,
)
from MemNavData.run_final14_mono_factorial_episode import (
    compare_shared_replay,
    load_payloads,
    load_rows,
    run_command,
    sha256,
)


SCHEMA = "final14_zero_depth_episode_v1_20260828"


def verify_receipt(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"receipt missing for {path}")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(fields == [digest, path.name], f"receipt invalid for {path}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    manifest_path = args.bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 natural manifest changed")
    manifest = json.loads(manifest_path.read_text())
    histories = manifest["episodes"]
    require(len(histories) == 21, "Final14 natural population changed")
    require(0 <= args.history_index < len(histories),
            "history index outside frozen population")
    item = histories[args.history_index]
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    source_receipt = json.loads((source_episode / "receipt.json").read_text())
    scene_file = Path(source_receipt["source_asset"])
    require(scene_file.is_file()
            and sha256(scene_file) == source_receipt["source_asset_sha256"],
            "source scene asset changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output_root = args.run_root / "evaluation" / "natural_direction" / label
    require(not output_root.exists(), f"output already exists: {output_root}")
    (output_root / "logs").mkdir(parents=True)
    arm_root = output_root / ARM
    arm_root.mkdir()
    reference = (args.reference_root / "evaluation" / "natural_direction" /
                 label)
    reference_completion_sha = verify_receipt(reference / "completion.json")

    command = [
        args.hab_python, "-u",
        str(args.source_root / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", args.host,
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
        "--role_pair_scope", "consumed_integration",
        "--hybrid_route", HYBRID_ROUTE,
        "--revisit_adapter", REVISIT_ADAPTER,
        "--navdp_depth_source", DEPTH_SOURCE,
        "--out", str(arm_root),
    ]
    elapsed = run_command(command, output_root / "logs/eval_zero_native.log")
    summary = json.loads((arm_root / "summary.json").read_text())
    require(summary.get("queries") == 2
            and summary.get("arm") == EVALUATOR_ARM
            and summary.get("navdp_depth_source") == DEPTH_SOURCE
            and summary.get("runtime_role_visibility") == "none",
            "zero-depth evaluator summary contract changed")
    rows = load_rows(arm_root)
    require(len(rows) == 2
            and {row["analysis_role"] for row in rows} == {"novel", "revisit"},
            "zero-depth paired role population changed")
    by_role = {row["analysis_role"]: row for row in rows}
    payloads = load_payloads(arm_root, episode, rows)
    audits = {}
    for role in ("novel", "revisit"):
        require(payloads[role].get("analysis_role_not_forwarded") is True,
                f"zero-depth/{role}: analysis role leaked")
        audits[role] = audit_zero_depth_plans(payloads[role]["query_leg"])

    reference_rows = load_rows(reference / "mono_native")
    reference_by_role = {
        row["analysis_role"]: row for row in reference_rows
    }
    reference_payloads = load_payloads(
        reference / "mono_native", episode, reference_rows
    )
    compare_shared_replay(
        reference_by_role, reference_payloads,
        by_role, payloads, ARM,
    )

    completion = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "consumed_final14_zero_depth_native_attribution",
        "fresh_confirmation": False,
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "reference_factorial_completion_sha256": reference_completion_sha,
        "arm": ARM,
        "runtime_role_visibility": "none",
        "shared_history_policy": "original_metric_navdp_goal_a_rgb_replay",
        "prefix_equality_to_verified_factorial": True,
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "smoke": bool(args.smoke),
        "wall_time_seconds": elapsed,
        "outcomes": {
            role: int(by_role[role]["reached"])
            for role in ("novel", "revisit")
        },
        "final_distance_m": {
            role: float(by_role[role]["final_goal_dist_m"])
            for role in ("novel", "revisit")
        },
        "depth_audits": audits,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode()
    completion_path = output_root / "completion.json"
    completion_path.write_bytes(encoded)
    completion_path.with_name(completion_path.name + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "outcomes": completion["outcomes"],
        "output": str(output_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
