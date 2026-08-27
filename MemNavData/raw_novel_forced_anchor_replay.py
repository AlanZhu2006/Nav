#!/usr/bin/env python3
"""Consumed-data counterfactual for raw-DINO Novel direction proposals.

This module deliberately stops before Habitat control.  It replays the exact
causal online-A JPEG history into the production MemNav/LingBot server, appends
the first Novel query observation once, and then asks the unchanged
``/posegoal_query`` endpoint for bearings under different forced historical
anchors.  The factual anchor is the one selected by raw DINO in the completed
formal run; counterfactual anchors are frozen before any model forward pass.

The experiment answers a narrow attribution question:

    Does the DINO-selected visual context let LingBot recover a more
    route-aligned Novel bearing than a uniformly sampled eligible context?

It uses only already-consumed development scenes and never launches Habitat.
Ground-truth route angles are used only by ``summarize`` after proposals have
been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_SCHEMA = "raw_novel_forced_anchor_manifest_v1_20260816"
UNIT_SCHEMA = "raw_novel_forced_anchor_unit_v1_20260816"
REPORT_SCHEMA = "raw_novel_forced_anchor_report_v1_20260816"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angular_error_deg(left: float, right: float) -> float:
    return abs(wrap_degrees(float(left) - float(right)))


def bearing_from_aux_pose(aux_pose: list[float]) -> float:
    require(len(aux_pose) == 2, "aux_pose must have two coordinates")
    forward, left = map(float, aux_pose)
    require(
        math.isfinite(forward) and math.isfinite(left),
        "aux_pose must be finite",
    )
    require(math.hypot(forward, left) > 1e-9, "aux_pose has zero direction")
    return float(math.degrees(math.atan2(left, forward)))


def first_takeover(plans: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in plans if row.get("revisit_adapter_takeover") is True]
    require(bool(rows), "raw-fixed Novel query has no takeover")
    return min(rows, key=lambda row: int(row["step"]))


def eligible_anchor_indices(plan: dict[str, Any]) -> list[int]:
    frame_index = int(plan["frame_idx"])
    ceiling = int(plan["candidate_ceiling"])
    count = int(plan["candidate_count"])
    require(count > 0, "proposal has no eligible anchors")
    upper = min(frame_index - 32, ceiling)
    lower = upper - count + 1
    require(0 <= lower <= upper, "invalid eligible interval")
    values = list(range(lower, upper + 1))
    require(len(values) == count, "eligible interval size changed")
    return values


def deterministic_counterfactual_anchors(
    eligible: list[int],
    factual_anchor: int,
    *,
    count: int,
    seed: int,
    identity: str,
) -> list[int]:
    """Sample without replacement using an identity-bound frozen seed."""

    pool = sorted(set(map(int, eligible)) - {int(factual_anchor)})
    require(bool(pool), "no non-factual eligible anchor")
    take = min(int(count), len(pool))
    require(take > 0, "counterfactual count must be positive")
    identity_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(identity_seed)
    selected = sorted(map(int, rng.choice(pool, size=take, replace=False)))
    require(factual_anchor not in selected, "factual anchor leaked into controls")
    return selected


def raw_buffer_episode(contract: dict[str, Any]) -> int:
    """Recover the MemNav reset slot for the raw-fixed Novel query.

    A freshly launched server owns ``ep_0000``.  Each non-native arm then
    performs a Novel reset and a Revisit reset.  Native never resets MemNav.
    """

    order = list(contract["arm_order"])
    position = order.index("raw_fixed_bearing")
    non_native_before = sum(arm != "native" for arm in order[:position])
    return 1 + 2 * non_native_before


def load_direction_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text())
    require(
        payload.get("schema_version")
        == "raw_novel_cohort_shift_audit_v1_20260816",
        "unexpected direction audit schema",
    )
    records = {
        (str(row["cohort"]), str(row["unit"])): row
        for row in payload["records"]
    }
    require(len(records) == len(payload["records"]), "duplicate direction rows")
    return records


def prepare_manifest(args: argparse.Namespace) -> None:
    cohort_root = args.cohort_root.resolve()
    references = load_direction_index(args.direction_report.resolve())
    rows = []
    for unit in sorted(path for path in cohort_root.iterdir() if path.is_dir()):
        contract_path = unit / "episode_contract.json"
        if not contract_path.is_file():
            continue
        contract = json.loads(contract_path.read_text())
        plan_paths = sorted(
            (unit / "raw_fixed_bearing").glob("*novel_plans.json")
        )
        require(len(plan_paths) == 1, f"{unit}: expected one Novel plan ledger")
        plan_payload = json.loads(plan_paths[0].read_text())
        plan = first_takeover(plan_payload["query_leg"])
        key = (args.cohort, unit.name)
        require(key in references, f"{unit}: missing direction audit row")
        reference = references[key]
        factual_anchor = int(plan["anchor"])
        eligible = eligible_anchor_indices(plan)
        require(factual_anchor in eligible, f"{unit}: factual anchor not eligible")
        identity = f"{args.cohort}/{unit.name}"
        sampled = deterministic_counterfactual_anchors(
            eligible,
            factual_anchor,
            count=args.anchor_samples,
            seed=args.seed,
            identity=identity,
        )
        selection_index = int(contract["selection_index"])
        buffer_slot = int(args.buffer_slot_offset) + selection_index
        buffer_episode = raw_buffer_episode(contract)
        with (unit / "raw_fixed_bearing" / "metric.csv").open(newline="") as h:
            metrics = list(csv.DictReader(h))
        novel_metrics = [row for row in metrics if row["analysis_role"] == "novel"]
        require(len(novel_metrics) == 1, f"{unit}: expected one Novel metric")
        route = reference.get("initial_geodesic_reconstruction")
        require(isinstance(route, dict), f"{unit}: missing route reconstruction")
        rows.append(
            {
                "identity": identity,
                "unit": unit.name,
                "selection_index": selection_index,
                "scene": str(contract["scene"]),
                "episode": str(contract["episode"]),
                "seed": int(novel_metrics[0]["seed"]),
                "frame_idx": int(plan["frame_idx"]),
                "goal_start_frame": int(plan["goal_start_frame"]),
                "candidate_ceiling": int(plan["candidate_ceiling"]),
                "eligible_anchor_lower": eligible[0],
                "eligible_anchor_upper": eligible[-1],
                "eligible_anchor_count": len(eligible),
                "factual_anchor": factual_anchor,
                "counterfactual_anchors": sampled,
                "logged_raw_score": float(plan["raw_score"]),
                "logged_aux_pose": list(map(float, plan["aux_pose"])),
                "logged_bearing_deg": float(reference["first_bearing_angle_deg"]),
                "shortest_path_target_deg": float(route["first_segment_angle_deg"]),
                "direct_goal_target_deg": float(
                    reference["first_direct_goal_angle_deg"]
                ),
                "remote_buffer_slot": buffer_slot,
                "remote_buffer_episode": buffer_episode,
                "remote_episode_dir": str(
                    args.remote_buffer_root
                    / f"{args.protocol}_{buffer_slot}"
                    / f"ep_{buffer_episode:04d}"
                ),
                "local_episode_dir": unit.name,
            }
        )
    require(bool(rows), f"{cohort_root}: no units found")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "scope": (
            "consumed Phase-2 first Novel proposal; factual raw-DINO anchor "
            "versus identity-seeded uniform eligible anchors"
        ),
        "cohort": args.cohort,
        "cohort_root": str(cohort_root),
        "direction_report": str(args.direction_report.resolve()),
        "protocol": args.protocol,
        "remote_buffer_root": str(args.remote_buffer_root),
        "buffer_slot_offset": int(args.buffer_slot_offset),
        "anchor_sampling": {
            "scheme": "uniform_without_replacement_excluding_factual",
            "samples_per_query": int(args.anchor_samples),
            "seed": int(args.seed),
            "frozen_before_model_forward": True,
        },
        "final14_accessed": False,
        "habitat_rollout": False,
        "records": rows,
    }
    encoded = canonical_json_bytes(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_bytes(encoded),
                "records": len(rows),
                "scene_clusters": len({row["scene"] for row in rows}),
                "forced_queries": sum(
                    1 + len(row["counterfactual_anchors"]) for row in rows
                ),
            },
            indent=2,
        )
    )


def http_json_post(
    url: str,
    *,
    json_payload: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes]] | None = None,
    data: dict[str, str] | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    import requests

    response = requests.post(
        url, json=json_payload, files=files, data=data, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    require(isinstance(payload, dict), f"{url}: non-object response")
    return payload


def validate_proposal(
    proposal: dict[str, Any], row: dict[str, Any], anchor: int
) -> dict[str, Any]:
    require("error" not in proposal, f"anchor {anchor}: {proposal.get('error')}")
    require(int(proposal["frame_idx"]) == int(row["frame_idx"]), "frame changed")
    require(
        int(proposal["goal_start_frame"]) == int(row["goal_start_frame"]),
        "goal boundary changed",
    )
    require(
        int(proposal["candidate_ceiling"]) == int(row["candidate_ceiling"]),
        "candidate ceiling changed",
    )
    require(
        int(proposal["candidate_count"]) == int(row["eligible_anchor_count"]),
        "candidate count changed",
    )
    require(int(proposal["anchor"]) == int(anchor), "forced anchor not honored")
    require(int(proposal["forced_anchor"]) == int(anchor), "forced flag missing")
    aux_pose = list(map(float, proposal["aux_pose"]))
    bearing = bearing_from_aux_pose(aux_pose)
    return {
        "anchor": int(anchor),
        "aux_pose": aux_pose,
        "bearing_deg": bearing,
        "selected_anchor_score": float(proposal["selected_anchor_score"]),
        "forced_anchor_score": float(proposal["forced_anchor_score"]),
        "raw_score": float(proposal["raw_score"]),
        "retrieved_anchor": int(proposal["retrieved_anchor"]),
        "current_goal_cos": float(proposal["current_goal_cos"]),
        "goal_rel_yaw": (
            None
            if proposal.get("goal_rel_yaw") is None
            else float(proposal["goal_rel_yaw"])
        ),
    }


def replay_unit(
    row: dict[str, Any],
    *,
    input_root: Path,
    server_base: str,
) -> dict[str, Any]:
    episode_dir = input_root / row["local_episode_dir"]
    required = [episode_dir / f"{index}.jpg" for index in range(row["frame_idx"] + 1)]
    required.append(episode_dir / "_goal.jpg")
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"{row['identity']}: missing inputs {missing[:3]}")
    start = time.monotonic()
    reset = http_json_post(
        f"{server_base}/navigator_reset",
        json_payload={
            "camera_height": 0.5,
            "seed": int(row["seed"]),
            "episode_len": int(row["frame_idx"]) + 600,
        },
    )
    require(reset.get("retrieval") == "raw", "server is not raw retrieval")
    for frame_index in range(int(row["frame_idx"])):
        frame = (episode_dir / f"{frame_index}.jpg").read_bytes()
        response = http_json_post(
            f"{server_base}/memory_step",
            files={"image": ("image.jpg", frame)},
        )
        require(
            int(response["frame_idx"]) == frame_index,
            f"{row['identity']}: replay index mismatch",
        )
    current = (episode_dir / f"{row['frame_idx']}.jpg").read_bytes()
    goal = (episode_dir / "_goal.jpg").read_bytes()
    common_data = {
        "candidate_ceiling_override": str(row["candidate_ceiling"]),
    }
    factual_raw = http_json_post(
        f"{server_base}/posegoal_step",
        files={"image": ("image.jpg", current), "goal": ("goal.jpg", goal)},
        data={**common_data, "forced_anchor": str(row["factual_anchor"])},
    )
    factual = validate_proposal(factual_raw, row, int(row["factual_anchor"]))
    controls = []
    for anchor in row["counterfactual_anchors"]:
        proposal = http_json_post(
            f"{server_base}/posegoal_query",
            files={"goal": ("goal.jpg", goal)},
            data={**common_data, "forced_anchor": str(anchor)},
        )
        controls.append(validate_proposal(proposal, row, int(anchor)))
    return {
        "schema_version": UNIT_SCHEMA,
        "identity": row["identity"],
        "scene": row["scene"],
        "episode": row["episode"],
        "frame_idx": int(row["frame_idx"]),
        "factual_anchor": int(row["factual_anchor"]),
        "counterfactual_anchors": list(map(int, row["counterfactual_anchors"])),
        "factual": factual,
        "counterfactuals": controls,
        "local_vs_logged_factual": {
            "bearing_error_deg": angular_error_deg(
                factual["bearing_deg"], row["logged_bearing_deg"]
            ),
            "aux_pose_l2": float(
                np.linalg.norm(
                    np.asarray(factual["aux_pose"], dtype=np.float64)
                    - np.asarray(row["logged_aux_pose"], dtype=np.float64)
                )
            ),
            "raw_score_abs_error": abs(
                factual["raw_score"] - float(row["logged_raw_score"])
            ),
        },
        "input_sha256": {
            "goal": sha256_file(episode_dir / "_goal.jpg"),
            "current": sha256_file(episode_dir / f"{row['frame_idx']}.jpg"),
            "history_chain": sha256_bytes(
                "\n".join(sha256_file(path) for path in required[:-2]).encode()
            ),
        },
        "elapsed_seconds": float(time.monotonic() - start),
    }


def run_replay(args: argparse.Namespace) -> None:
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "bad manifest")
    manifest_sha = sha256_bytes(manifest_bytes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipt = args.output_root / "manifest_receipt.json"
    receipt_payload = {
        "schema_version": "raw_novel_forced_anchor_manifest_receipt_v1_20260816",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "server_base": args.server_base.rstrip("/"),
    }
    if receipt.exists():
        require(json.loads(receipt.read_text()) == receipt_payload, "receipt changed")
    else:
        receipt.write_bytes(canonical_json_bytes(receipt_payload))
    for index, row in enumerate(manifest["records"]):
        output = args.output_root / f"{index:03d}_{row['unit']}.json"
        if output.exists():
            payload = json.loads(output.read_text())
            require(payload.get("schema_version") == UNIT_SCHEMA, f"bad {output}")
            require(payload.get("identity") == row["identity"], f"wrong {output}")
            print(f"[skip {index + 1}/{len(manifest['records'])}] {row['identity']}")
            continue
        print(f"[run {index + 1}/{len(manifest['records'])}] {row['identity']}", flush=True)
        payload = replay_unit(
            row,
            input_root=args.input_root,
            server_base=args.server_base.rstrip("/"),
        )
        output.write_bytes(canonical_json_bytes(payload))
        print(
            f"  factual={payload['factual']['bearing_deg']:.2f}deg "
            f"controls={len(payload['counterfactuals'])} "
            f"elapsed={payload['elapsed_seconds']:.1f}s",
            flush=True,
        )


def scalar(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, seed: int, resamples: int
) -> dict[str, Any]:
    scenes = sorted({row["scene"] for row in rows})
    grouped = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        sample = [row for scene in chosen for row in grouped[str(scene)]]
        values[index] = float(np.mean([row["advantage_deg"] for row in sample]))
    observed = float(np.mean([row["advantage_deg"] for row in rows]))
    return {
        "scene_clusters": len(scenes),
        "resamples": int(resamples),
        "seed": int(seed),
        "observed_mean_advantage_deg": observed,
        "ci_95_deg": np.quantile(values, [0.025, 0.975]).tolist(),
        "probability_bootstrap_mean_le_zero": float(np.mean(values <= 0.0)),
    }


def summarize_target(
    records: list[dict[str, Any]], target_field: str, *, seed: int, resamples: int
) -> dict[str, Any]:
    rows = []
    for row in records:
        target = float(row[target_field])
        factual_error = angular_error_deg(row["result"]["factual"]["bearing_deg"], target)
        control_errors = [
            angular_error_deg(item["bearing_deg"], target)
            for item in row["result"]["counterfactuals"]
        ]
        control_mean = float(np.mean(control_errors))
        rows.append(
            {
                "identity": row["identity"],
                "scene": row["scene"],
                "target_deg": target,
                "factual_error_deg": factual_error,
                "counterfactual_error_mean_deg": control_mean,
                "advantage_deg": control_mean - factual_error,
                "factual_le_30_deg": factual_error <= 30.0,
                "counterfactual_fraction_le_30_deg": float(
                    np.mean(np.asarray(control_errors) <= 30.0)
                ),
                "counterfactual_errors_deg": control_errors,
            }
        )
    advantages = [row["advantage_deg"] for row in rows]
    return {
        "n": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "factual_error_deg": scalar([row["factual_error_deg"] for row in rows]),
        "counterfactual_expected_error_deg": scalar(
            [row["counterfactual_error_mean_deg"] for row in rows]
        ),
        "factual_advantage_deg": scalar(advantages),
        "factual_count_le_30_deg": int(sum(row["factual_le_30_deg"] for row in rows)),
        "counterfactual_expected_count_le_30_deg": float(
            sum(row["counterfactual_fraction_le_30_deg"] for row in rows)
        ),
        "scenes_with_positive_mean_advantage": int(
            sum(
                np.mean(
                    [row["advantage_deg"] for row in rows if row["scene"] == scene]
                )
                > 0.0
                for scene in {row["scene"] for row in rows}
            )
        ),
        "cluster_bootstrap": cluster_bootstrap(
            rows, seed=seed, resamples=resamples
        ),
        "records": rows,
    }


def summarize(args: argparse.Namespace) -> None:
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "bad manifest")
    joined = []
    for index, row in enumerate(manifest["records"]):
        result_path = args.result_root / f"{index:03d}_{row['unit']}.json"
        require(result_path.is_file(), f"missing {result_path}")
        result = json.loads(result_path.read_text())
        require(result.get("schema_version") == UNIT_SCHEMA, f"bad {result_path}")
        require(result.get("identity") == row["identity"], "result identity changed")
        require(
            result["counterfactual_anchors"] == row["counterfactual_anchors"],
            "counterfactual anchor set changed",
        )
        joined.append({**row, "result": result})
    local_shifts = [
        row["result"]["local_vs_logged_factual"]["bearing_error_deg"]
        for row in joined
    ]
    shortest = summarize_target(
        joined,
        "shortest_path_target_deg",
        seed=args.seed,
        resamples=args.resamples,
    )
    direct = summarize_target(
        joined,
        "direct_goal_target_deg",
        seed=args.seed + 1,
        resamples=args.resamples,
    )
    primary_ci = shortest["cluster_bootstrap"]["ci_95_deg"]
    decision = (
        "dino_visual_context_advantage_supported_continue_goal_shuffle"
        if primary_ci[0] > 0.0
        else "dino_visual_context_advantage_not_supported_stop_novel_dino_branch"
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": (
            "proposal-only consumed-data attribution; no Habitat rollout and "
            "no final14 access"
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "result_root": str(args.result_root.resolve()),
        "records": len(joined),
        "scene_clusters": len({row["scene"] for row in joined}),
        "local_vs_logged_factual_bearing_error_deg": scalar(local_shifts),
        "primary_target": "shortest_path_first_segment",
        "shortest_path": shortest,
        "direct_goal": direct,
        "predeclared_stop_rule": (
            "continue only if scene-cluster 95% CI lower bound for factual "
            "DINO-anchor advantage is greater than zero degrees"
        ),
        "decision": decision,
        "final14_accessed": False,
        "habitat_rollout": False,
    }
    encoded = canonical_json_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "decision": decision,
                "local_reproduction": report[
                    "local_vs_logged_factual_bearing_error_deg"
                ],
                "shortest_path": {
                    key: shortest[key]
                    for key in (
                        "n",
                        "scenes",
                        "factual_error_deg",
                        "counterfactual_expected_error_deg",
                        "factual_advantage_deg",
                        "factual_count_le_30_deg",
                        "counterfactual_expected_count_le_30_deg",
                        "cluster_bootstrap",
                    )
                },
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--cohort", default="phase2")
    prepare.add_argument("--cohort-root", type=Path, required=True)
    prepare.add_argument("--direction-report", type=Path, required=True)
    prepare.add_argument("--remote-buffer-root", type=Path, required=True)
    prepare.add_argument("--protocol", default="natural_direction")
    prepare.add_argument("--buffer-slot-offset", type=int, default=64)
    prepare.add_argument("--anchor-samples", type=int, default=12)
    prepare.add_argument("--seed", type=int, default=20260816)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=prepare_manifest)

    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--input-root", type=Path, required=True)
    run.add_argument("--server-base", default="http://127.0.0.1:21220")
    run.add_argument("--output-root", type=Path, required=True)
    run.set_defaults(func=run_replay)

    summary = sub.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--result-root", type=Path, required=True)
    summary.add_argument("--resamples", type=int, default=100000)
    summary.add_argument("--seed", type=int, default=20260816)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
