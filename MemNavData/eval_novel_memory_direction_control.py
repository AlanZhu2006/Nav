#!/usr/bin/env python3
"""Four-arm Novel direction control on a sealed consumed role-pair set.

The production role-pair evaluator remains the source of Habitat rollout and
pairing semantics.  This wrapper changes only the sidecar replay source or the
raw adapter bearing, then appends an explicit intervention ledger.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import eval_2leg_habitat as base
import eval_shared_online_role_pairs as role_eval
from novel_memory_direction_control import (
    ARMS,
    RandomizedBearingAdapter,
    SCHEMA_VERSION,
    aggregate_hash,
    load_online_source,
    replay_deranged_sidecar,
    sha256_file,
    validate_control_manifest,
)


RESULT_SCHEMA = "novel_memory_direction_closed_loop_v1_20260816"
MANIFEST_PATH = Path(os.environ.get("NOVEL_CONTROL_MANIFEST", ""))
EXPECTED_MANIFEST_SHA = os.environ.get(
    "EXPECTED_NOVEL_CONTROL_MANIFEST_SHA", ""
)
CONTROL_ARM = os.environ.get("NOVEL_CONTROL_ARM", "")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_manifest() -> dict[str, Any]:
    require(CONTROL_ARM in ARMS, "NOVEL_CONTROL_ARM is missing or invalid")
    require(MANIFEST_PATH.is_file(), "NOVEL_CONTROL_MANIFEST is missing")
    require(
        len(EXPECTED_MANIFEST_SHA) == 64
        and sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA,
        "Novel-control manifest changed",
    )
    payload = json.loads(MANIFEST_PATH.read_text())
    validate_control_manifest(payload)
    require(
        payload["evaluation_stage"] == "consumed_development_mechanism_only",
        "Novel control attempted to claim a different evaluation stage",
    )
    return payload


MANIFEST = load_manifest()
ROWS = {
    (str(row["scene"]), str(row["episode"])): row
    for row in MANIFEST["episodes"]
}


def selected_identity() -> tuple[str, str]:
    episodes = [
        item.strip() for item in str(base.args.episode_ids).split(",")
        if item.strip()
    ]
    require(len(episodes) == 1, "control evaluator requires exactly one episode")
    identity = (str(base.SCENE_IDENTITY), episodes[0])
    require(identity in ROWS, "selected episode is outside control manifest")
    return identity


IDENTITY = selected_identity()
ROW = ROWS[IDENTITY]
require(
    sha256_file(Path(ROW["role_pairs_path"])) == ROW["role_pairs_sha256"],
    "selected role-pair sidecar changed",
)

_role_payload = json.loads(Path(ROW["role_pairs_path"]).read_text())
_stored_queries = [
    query
    for pair in _role_payload["pairs"]
    for query in pair["queries"]
    if str(query["query_id"]) == str(ROW["query_id"])
]
require(len(_stored_queries) == 1, "control query identity is ambiguous")
QUERY_FLOOR_POSITION = base.np.asarray(
    _stored_queries[0]["floor_position"], dtype=base.np.float64
)
require(base.args.role_pair_query_role == "novel", "control must run Novel only")
require(
    base.args.role_pair_scope == "consumed_integration",
    "control result must retain consumed-development scope",
)


_original_resolve_arm = role_eval.resolve_arm
_original_replay_prefix = role_eval.replay_prefix
_original_adapter = base.adapt_revisit_pointgoal
_original_run_policy_leg = base.run_policy_leg
_randomizer: RandomizedBearingAdapter | None = None
_final_geodesic_m: list[float] = []


def resolve_control_arm() -> tuple[str, str | None]:
    original_arm, backend = _original_resolve_arm()
    if CONTROL_ARM == "native":
        require(original_arm == "native", "native control uses the wrong backend")
    else:
        require(
            original_arm == "raw_fixed_bearing",
            "direction controls require the raw-fixed production path",
        )
    return CONTROL_ARM, backend


def _factual_replay_receipt(frozen: dict[str, Any], replay: dict[str, Any]) -> None:
    trace = frozen["trace"]
    pose_hashes = [str(pose["jpg_sha256"]) for pose in trace["poses"]]
    plan_steps = {int(plan["step"]) for plan in trace["plans"]}
    fifo_hashes = [
        str(pose["jpg_sha256"])
        for pose in trace["poses"]
        if int(pose["step"]) in plan_steps
    ]
    replay.update({
        "causal_control_schema_version": SCHEMA_VERSION,
        "factual_fifo_scene": IDENTITY[0],
        "factual_fifo_episode": IDENTITY[1],
        "factual_fifo_frames": len(trace["poses"]),
        "factual_fifo_decision_sha256": aggregate_hash(fifo_hashes),
        "sidecar_scene": IDENTITY[0] if CONTROL_ARM != "native" else None,
        "sidecar_episode": IDENTITY[1] if CONTROL_ARM != "native" else None,
        "sidecar_memory_frames": (
            len(trace["poses"]) if CONTROL_ARM != "native" else 0
        ),
        "sidecar_memory_sha256": (
            aggregate_hash(pose_hashes) if CONTROL_ARM != "native" else None
        ),
        "sidecar_is_deranged": False,
    })


def replay_control_prefix(frozen: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if CONTROL_ARM != "raw_deranged_history":
        leg, replay = _original_replay_prefix(frozen)
        _factual_replay_receipt(frozen, replay)
        return leg, replay

    donor = load_online_source(ROW["donor"])
    factual = dict(frozen)
    factual.update(scene=IDENTITY[0], episode=IDENTITY[1])
    replay = replay_deranged_sidecar(
        factual,
        donor,
        memory_step=base.srv_memory,
        navdp_replay_step=base.srv_navdp_memory_replay,
    )
    trace = frozen["trace"]
    leg = {
        "reached": True,
        "path_len": float(trace["path_len"]),
        "steps": int(trace["steps"]),
        "plans": trace["plans"],
        "memory_trace": replay["memory_trace"],
        "rollout_trace": trace["poses"],
        "end_pos": base.np.asarray(trace["end_position"], dtype=base.np.float64),
        "end_psi": float(trace["end_yaw"]),
    }
    return leg, replay


def run_control_policy_leg(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Attach a scoring-only final geodesic without changing the controller."""

    leg = _original_run_policy_leg(*args, **kwargs)
    require(len(args) >= 2, "policy wrapper lost its pathfinder argument")
    pathfinder = args[1]
    ok, distance, _ = base.geodesic(
        pathfinder,
        base.np.asarray(leg["end_pos"], dtype=base.np.float64),
        QUERY_FLOOR_POSITION,
    )
    require(ok and base.np.isfinite(distance), "final query geodesic failed")
    leg["final_goal_geodesic_m"] = float(distance)
    _final_geodesic_m.append(float(distance))
    return leg


def augment_outputs() -> None:
    output = Path(base.args.out)
    require(len(_final_geodesic_m) == 1, "control arm must emit one final geodesic")
    plans_paths = sorted(output.glob("*_plans.json"))
    require(len(plans_paths) == 1, "control arm must emit exactly one Novel plan file")
    path = plans_paths[0]
    payload = json.loads(path.read_text())
    query_leg = payload["query_leg"]
    ledger = list(_randomizer.ledger) if _randomizer is not None else []
    if CONTROL_ARM == "raw_randomized_bearing":
        require(len(ledger) == len(query_leg), "random ledger/plan count differs")
        for plan, audit in zip(query_leg, ledger):
            require(
                bool(plan.get("revisit_adapter_takeover"))
                == bool(audit["randomized_takeover"]),
                "randomization changed or misreported proposal availability",
            )
            if audit["randomized_takeover"]:
                require(
                    float(plan["memory_pointgoal_fixed_radius_m"]) == 2.5,
                    "randomized controller radius changed",
                )
    else:
        require(not ledger, "non-random arm unexpectedly emitted a transform ledger")
    payload["novel_causal_control"] = {
        "schema_version": SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA,
        "arm": CONTROL_ARM,
        "evaluation_stage": MANIFEST["evaluation_stage"],
        "confirmation_claim_allowed": False,
        "method_or_threshold_selection_allowed": False,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "identity": list(IDENTITY),
        "donor_identity": (
            [ROW["donor"]["scene"], ROW["donor"]["episode"]]
            if CONTROL_ARM == "raw_deranged_history" else None
        ),
        "factual_history_receipts": {
            "role_pairs_sha256": ROW["role_pairs_sha256"],
            "online_a_receipt_sha256": ROW["online_a_receipt_sha256"],
            "online_a_trace_sha256": ROW["online_a_trace_sha256"],
        },
        "sidecar_history_receipts": (
            {
                "online_a_receipt_sha256": ROW["donor"][
                    "online_a_receipt_sha256"
                ],
                "online_a_trace_sha256": ROW["donor"][
                    "online_a_trace_sha256"
                ],
            }
            if CONTROL_ARM == "raw_deranged_history"
            else (
                {
                    "online_a_receipt_sha256": ROW[
                        "online_a_receipt_sha256"
                    ],
                    "online_a_trace_sha256": ROW[
                        "online_a_trace_sha256"
                    ],
                }
                if CONTROL_ARM != "native" else None
            )
        ),
        "randomized_bearing_ledger": ledger,
        "final_goal_geodesic_m": _final_geodesic_m[0],
        "final_geodesic_is_evaluator_side_only": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary["arm"] == CONTROL_ARM, "summary arm label changed")
    summary["novel_causal_control"] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "evaluation_stage": MANIFEST["evaluation_stage"],
        "confirmation_claim_allowed": False,
        "transform_ledger_entries": len(ledger),
        "final_goal_geodesic_m": _final_geodesic_m[0],
        "final_geodesic_is_evaluator_side_only": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    metric_path = output / "metric.csv"
    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, "control arm must emit one metric row")
    require(
        "final_goal_geodesic_m" not in rows[0],
        "base evaluator unexpectedly owns the control geodesic field",
    )
    rows[0]["final_goal_geodesic_m"] = _final_geodesic_m[0]
    with metric_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global _randomizer
    role_eval.RESULT_SCHEMA = RESULT_SCHEMA
    role_eval.resolve_arm = resolve_control_arm
    role_eval.replay_prefix = replay_control_prefix
    base.run_policy_leg = run_control_policy_leg
    if CONTROL_ARM == "raw_randomized_bearing":
        _randomizer = RandomizedBearingAdapter(
            original_adapter=_original_adapter,
            global_seed=int(MANIFEST["global_seed"]),
            scene=IDENTITY[0],
            episode=IDENTITY[1],
            query_id=str(ROW["query_id"]),
        )
        base.adapt_revisit_pointgoal = _randomizer
    try:
        role_eval.main()
        augment_outputs()
    finally:
        base.adapt_revisit_pointgoal = _original_adapter
        base.run_policy_leg = _original_run_policy_leg


if __name__ == "__main__":
    main()
