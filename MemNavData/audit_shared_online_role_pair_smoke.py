#!/usr/bin/env python3
"""Fail-closed audit of the consumed role-pair three-arm integration smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


BASE_ARMS = ("native", "raw_direct", "certified")
FOUR_ARM_ORDER = (
    "native",
    "raw_direct",
    "raw_fixed_bearing",
    "certified",
)
FINAL14_LEARNED_ORDER = (
    "native",
    "raw_fixed_bearing",
    "geometry_fixed",
    "certified",
    "learned_pi3x_spatial",
)
ROLES = ("novel", "revisit")
FORBIDDEN_RUNTIME_FIELDS = {
    "analysis_role",
    "max_online_a_covis",
    "covis_curve",
    "geodesic_from_a_end_m",
    "initial_path_bearing_rad",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_metrics(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, f"{path}: expected exactly two role queries")
    indexed = {row["analysis_role"]: row for row in rows}
    require(set(indexed) == set(ROLES), f"{path}: role coverage changed")
    return indexed


def as_int(row: dict, field: str) -> int:
    return int(row[field])


def as_float(row: dict, field: str) -> float:
    return float(row[field])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict:
    contract = json.loads((root / "run_contract.json").read_text())
    require(
        contract.get("schema_version")
        == "shared_online_role_pair_local_smoke_v1_20260814",
        "run contract schema changed",
    )
    require(
        contract.get("scope") in {
            "consumed-scene integration only; no SR claim",
            "Replica cross-dataset integration only; no SR claim",
        },
        "smoke scope changed",
    )
    require(contract.get("role_visible_to_runtime") is False,
            "run contract exposed role to runtime")
    declared_arms = tuple(contract.get("arms") or ())
    require(
        declared_arms in (BASE_ARMS, FOUR_ARM_ORDER, FINAL14_LEARNED_ORDER),
        "arm set changed",
    )
    scene_roots = sorted(path for path in (root / "scenes").iterdir() if path.is_dir())
    require(bool(scene_roots), "smoke contains no scene outputs")
    rows = []
    for scene_root in scene_roots:
        metrics = {}
        summaries = {}
        for arm in declared_arms:
            arm_root = scene_root / arm
            summaries[arm] = json.loads((arm_root / "summary.json").read_text())
            metrics[arm] = read_metrics(arm_root / "metric.csv")
            summary = summaries[arm]
            require(summary["arm"] == arm, f"{scene_root}/{arm}: arm mismatch")
            require(summary["runtime_role_visibility"] == "none",
                    f"{scene_root}/{arm}: role visibility changed")
            require(summary["shared_A_all_hashes_ok"] is True,
                    f"{scene_root}/{arm}: online-A hashes failed")
            require(summary["shared_A_total_diffusion_samples"] == 0,
                    f"{scene_root}/{arm}: replay sampled diffusion")
            require(summary["runtime_failure_plans"] == 0,
                    f"{scene_root}/{arm}: runtime failure occurred")
            if arm == "learned_pi3x_spatial":
                learned_contract = summary["memnav_server_info"].get(
                    "learned_pi3x_relocalization"
                )
                declared_learned = contract.get("learned_pi3x") or {}
                require(
                    isinstance(learned_contract, dict)
                    and learned_contract.get("enabled") is True,
                    f"{scene_root}/{arm}: learned server contract missing",
                )
                require(
                    learned_contract.get("model_sha256")
                    == declared_learned.get("model_sha256")
                    and learned_contract.get("proof_manifest_sha256")
                    == declared_learned.get("proof_manifest_sha256"),
                    f"{scene_root}/{arm}: learned artifact identity changed",
                )
        identities = {
            arm: {
                role: (
                    metrics[arm][role]["scene"],
                    metrics[arm][role]["episode"],
                    metrics[arm][role]["pair_id"],
                    metrics[arm][role]["query_id"],
                    metrics[arm][role]["seed"],
                    metrics[arm][role]["geodesic_m"],
                )
                for role in ROLES
            }
            for arm in declared_arms
        }
        reference_identity = identities[declared_arms[0]]
        require(
            all(identity == reference_identity for identity in identities.values()),
            f"{scene_root}: arm inputs are not paired",
        )

        plans = {}
        for arm in declared_arms:
            plans[arm] = {}
            for role in ROLES:
                metric = metrics[arm][role]
                path = (
                    scene_root
                    / arm
                    / f"{metric['episode']}_{metric['query_id']}_plans.json"
                )
                payload = json.loads(path.read_text())
                runtime_fields = set(payload["query_runtime_fields"])
                require(
                    not runtime_fields.intersection(FORBIDDEN_RUNTIME_FIELDS),
                    f"{path}: runtime projection leaked analysis fields",
                )
                require(payload["analysis_role_not_forwarded"] is True,
                        f"{path}: role-hiding receipt failed")
                plans[arm][role] = payload

        novel_native = metrics["native"]["novel"]
        novel_certified = metrics["certified"]["novel"]
        require(as_int(novel_certified, "router_active_plans") == 0,
                f"{scene_root}: certified falsely activated on Novel")
        require(as_int(novel_certified, "certificate_accept_plans") == 0,
                f"{scene_root}: certificate falsely accepted Novel")
        require(as_int(novel_certified, "adapter_takeover_plans") == 0,
                f"{scene_root}: adapter falsely took over Novel")
        require(
            plans["native"]["novel"]["rollout_traces"]
            == plans["certified"]["novel"]["rollout_traces"],
            f"{scene_root}: certified Novel fallback changed physical rollout",
        )
        for field in ("reached", "steps", "path_len_m", "final_goal_dist_m"):
            require(
                novel_native[field] == novel_certified[field],
                f"{scene_root}: certified Novel fallback changed {field}",
            )

        revisit_certified = metrics["certified"]["revisit"]
        require(as_int(revisit_certified, "certificate_accept_plans") > 0,
                f"{scene_root}: certified Revisit positive control did not accept")
        require(as_int(revisit_certified, "adapter_takeover_plans") > 0,
                f"{scene_root}: certified Revisit positive control did not take over")
        for role in ROLES:
            if "raw_direct" in declared_arms:
                require(
                    as_int(
                        metrics["raw_direct"][role],
                        "adapter_takeover_plans",
                    )
                    > 0,
                    f"{scene_root}: raw-direct did not cover {role}",
                )
            if "raw_fixed_bearing" in declared_arms:
                require(
                    as_int(
                        metrics["raw_fixed_bearing"][role],
                        "adapter_takeover_plans",
                    )
                    > 0,
                    f"{scene_root}: raw fixed-bearing did not cover {role}",
                )
        learned_fallback_roles = []
        learned_bearing_errors = {role: [] for role in ROLES}
        if "learned_pi3x_spatial" in declared_arms:
            for role in ROLES:
                learned_metric = metrics["learned_pi3x_spatial"][role]
                require(
                    as_int(
                        learned_metric,
                        "learned_pi3x_initial_inference_plans",
                    )
                    == 1,
                    f"{scene_root}/{role}: learned first-query lifecycle changed",
                )
                learned_accepts = as_int(
                    learned_metric, "learned_pi3x_accept_plans"
                )
                learned_takeovers = as_int(
                    learned_metric, "adapter_takeover_plans"
                )
                require(
                    learned_accepts == learned_takeovers,
                    f"{scene_root}/{role}: learned accept/takeover mismatch",
                )
                learned_query_plans = plans["learned_pi3x_spatial"][role][
                    "query_leg"
                ]
                accepted_plans = [
                    plan for plan in learned_query_plans
                    if plan.get("learned_pi3x_relocalization_accepted") is True
                ]
                require(
                    len(accepted_plans) == learned_accepts,
                    f"{scene_root}/{role}: learned accepted-plan count differs",
                )
                for plan in accepted_plans:
                    value = plan.get(
                        "learned_pi3x_evaluation_gt_bearing_error_deg"
                    )
                    require(
                        value is not None and math.isfinite(float(value)),
                        f"{scene_root}/{role}: accepted learned bearing lacks "
                        "an evaluation-only GT error",
                    )
                    error = float(value)
                    require(
                        0.0 <= error <= 180.0,
                        f"{scene_root}/{role}: invalid learned bearing error",
                    )
                    learned_bearing_errors[role].append(error)
                if learned_accepts == 0:
                    learned_fallback_roles.append(role)
                    native_payload = plans["native"][role]
                    learned_payload = plans["learned_pi3x_spatial"][role]
                    require(
                        native_payload["rollout_traces"]["query"]
                        == learned_payload["rollout_traces"]["query"],
                        f"{scene_root}/{role}: learned fallback rollout changed",
                    )
                    native_query_plans = native_payload["query_leg"]
                    learned_query_plans = learned_payload["query_leg"]
                    require(
                        len(native_query_plans) == len(learned_query_plans),
                        f"{scene_root}/{role}: learned fallback plan count changed",
                    )
                    equality_keys = (
                        "step",
                        "requested_diffusion_seed",
                        "diffusion_seed",
                        "server_selected_idx",
                        "trajectory_candidate_count",
                        "selected_trajectory_sha256",
                    )
                    require(
                        all(
                            native_plan.get(key) == learned_plan.get(key)
                            for native_plan, learned_plan in zip(
                                native_query_plans, learned_query_plans
                            )
                            for key in equality_keys
                        ),
                        f"{scene_root}/{role}: learned fallback plan changed",
                    )
                    for field in (
                        "reached",
                        "steps",
                        "path_len_m",
                        "final_goal_dist_m",
                        "termination_reason",
                    ):
                        require(
                            metrics["native"][role][field]
                            == learned_metric[field],
                            f"{scene_root}/{role}: learned fallback changed {field}",
                        )
        rows.append({
            "scene": novel_native["scene"],
            "episode": novel_native["episode"],
            "novel_certified_exact_fallback": True,
            "novel_certified_accept_plans": as_int(
                novel_certified, "certificate_accept_plans"
            ),
            "revisit_certified_accept_plans": as_int(
                revisit_certified, "certificate_accept_plans"
            ),
            "raw_direct_novel_takeover_plans": (
                as_int(
                    metrics["raw_direct"]["novel"],
                    "adapter_takeover_plans",
                )
                if "raw_direct" in declared_arms else None
            ),
            "raw_direct_revisit_takeover_plans": (
                as_int(
                    metrics["raw_direct"]["revisit"],
                    "adapter_takeover_plans",
                )
                if "raw_direct" in declared_arms else None
            ),
            "raw_fixed_bearing_novel_takeover_plans": (
                as_int(
                    metrics["raw_fixed_bearing"]["novel"],
                    "adapter_takeover_plans",
                )
                if "raw_fixed_bearing" in declared_arms else None
            ),
            "raw_fixed_bearing_revisit_takeover_plans": (
                as_int(
                    metrics["raw_fixed_bearing"]["revisit"],
                    "adapter_takeover_plans",
                )
                if "raw_fixed_bearing" in declared_arms else None
            ),
            "native_novel_final_m": as_float(novel_native, "final_goal_dist_m"),
            "raw_direct_novel_final_m": (
                as_float(
                    metrics["raw_direct"]["novel"], "final_goal_dist_m"
                )
                if "raw_direct" in declared_arms else None
            ),
            "native_revisit_final_m": as_float(
                metrics["native"]["revisit"], "final_goal_dist_m"
            ),
            "raw_direct_revisit_final_m": (
                as_float(
                    metrics["raw_direct"]["revisit"], "final_goal_dist_m"
                )
                if "raw_direct" in declared_arms else None
            ),
            "certified_revisit_final_m": as_float(
                revisit_certified, "final_goal_dist_m"
            ),
            "raw_fixed_bearing_novel_final_m": (
                as_float(
                    metrics["raw_fixed_bearing"]["novel"],
                    "final_goal_dist_m",
                )
                if "raw_fixed_bearing" in declared_arms else None
            ),
            "raw_fixed_bearing_revisit_final_m": (
                as_float(
                    metrics["raw_fixed_bearing"]["revisit"],
                    "final_goal_dist_m",
                )
                if "raw_fixed_bearing" in declared_arms else None
            ),
            "learned_pi3x_accept_plans": (
                {
                    role: as_int(
                        metrics["learned_pi3x_spatial"][role],
                        "learned_pi3x_accept_plans",
                    )
                    for role in ROLES
                }
                if "learned_pi3x_spatial" in declared_arms else None
            ),
            "learned_pi3x_exact_fallback_roles": learned_fallback_roles,
            "learned_pi3x_accepted_bearing_errors_deg": (
                learned_bearing_errors
                if "learned_pi3x_spatial" in declared_arms else None
            ),
        })
    if "learned_pi3x_spatial" in declared_arms:
        learned_accept_queries = sum(
            row["learned_pi3x_accept_plans"][role] > 0
            for row in rows
            for role in ROLES
        )
        learned_fallback_queries = sum(
            len(row["learned_pi3x_exact_fallback_roles"]) for row in rows
        )
        require(
            learned_accept_queries > 0,
            "learned dry-run did not exercise the acceptance path",
        )
        require(
            learned_fallback_queries > 0,
            "learned dry-run did not exercise exact native fallback",
        )
        learned_bearing_errors = [
            error
            for row in rows
            for role in ROLES
            for error in row[
                "learned_pi3x_accepted_bearing_errors_deg"
            ][role]
        ]
        require(
            learned_bearing_errors,
            "learned dry-run did not produce an accepted bearing score",
        )
        learned_errors_over_90 = sum(
            error > 90.0 for error in learned_bearing_errors
        )
        require(
            learned_errors_over_90 == 0,
            "learned dry-run accepted a bearing with error above 90 degrees",
        )
    else:
        learned_accept_queries = None
        learned_fallback_queries = None
        learned_bearing_errors = []
        learned_errors_over_90 = None
    return {
        "ok": True,
        "auditor_sha256": sha256_file(Path(__file__)),
        "source_run_contract_sha256": sha256_file(
            root / "run_contract.json"
        ),
        "source_inputs_receipt_sha256": sha256_file(
            root / "source_inputs.sha256"
        ),
        "scope": contract["scope"],
        "benchmark_manifest_sha256": contract[
            "benchmark_manifest_sha256"
        ],
        "selected_indices": contract["selected_indices"],
        "selected_identities": contract["selected_identities"],
        "max_steps": int(contract["max_steps"]),
        "scenes": len(rows),
        "queries_per_arm": 2 * len(rows),
        "arms": list(declared_arms),
        "runtime_role_visibility": "none",
        "online_a_hashes_verified": True,
        "online_a_diffusion_samples": 0,
        "runtime_failure_plans": 0,
        "novel_certified_exact_fallback_scenes": sum(
            row["novel_certified_exact_fallback"] for row in rows
        ),
        "novel_certified_accept_plans": sum(
            row["novel_certified_accept_plans"] for row in rows
        ),
        "revisit_certified_accept_plans": sum(
            row["revisit_certified_accept_plans"] for row in rows
        ),
        "raw_direct_takeover_plans": (
            {
                role: sum(
                    row[f"raw_direct_{role}_takeover_plans"] for row in rows
                )
                for role in ROLES
            }
            if "raw_direct" in declared_arms else None
        ),
        "raw_fixed_bearing_takeover_plans": (
            {
                role: sum(
                    row[f"raw_fixed_bearing_{role}_takeover_plans"]
                    for row in rows
                )
                for role in ROLES
            }
            if "raw_fixed_bearing" in declared_arms else None
        ),
        "learned_pi3x_accept_queries": learned_accept_queries,
        "learned_pi3x_exact_fallback_queries": learned_fallback_queries,
        "learned_pi3x_accepted_bearing_error_count": (
            len(learned_bearing_errors)
            if "learned_pi3x_spatial" in declared_arms else None
        ),
        "learned_pi3x_accepted_bearing_error_max_deg": (
            max(learned_bearing_errors)
            if learned_bearing_errors else None
        ),
        "learned_pi3x_accepted_bearing_errors_over_90_deg": (
            learned_errors_over_90
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
