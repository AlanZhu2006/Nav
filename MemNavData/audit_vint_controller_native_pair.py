#!/usr/bin/env python3
"""Audit one mixed-role ViNT-native versus ViNT+CEC authority pair.

Both arms execute the same frozen ViNT.  ``forced_reject_native`` runs the
full role-free proof pipeline but withholds every takeover, so its control
input is always the unchanged original ImageGoal.  ``grant`` changes that
ImageGoal only when CEC emits a valid proof-bound historical anchor.  The
auditor checks causal starts and first-decision identity, then permits the two
closed-loop trajectories (and therefore later proof streams) to diverge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from MemNavData.cec_handoff_contract import verify_handoff_packet_envelope


SCHEMA = "vint_controller_native_pair_audit_v1_20260828"
HUB_SCHEMA = "cec_controller_portability_hub_v2"
SHA256 = re.compile(r"[0-9a-f]{64}")
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(root: Path, scope: str) -> dict[str, Any]:
    require(root.is_dir(), f"{scope}: arm root is missing")
    result = root / "result"
    summary_path = result / "summary.json"
    metric_path = result / "metric.csv"
    compute_path = root / "compute_identity.json"
    health_path = root / "hub_health.json"
    require(all(path.is_file() for path in (
        summary_path, metric_path, compute_path, health_path,
    )), f"{scope}: arm receipt is incomplete")
    summary = json.loads(summary_path.read_text())
    compute = json.loads(compute_path.read_text())
    health = json.loads(health_path.read_text())
    with metric_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2, f"{scope}: expected one Novel and one Revisit")
    by_query = {str(row["query_id"]): row for row in rows}
    require(len(by_query) == 2, f"{scope}: duplicate query identity")

    payloads: dict[str, dict[str, Any]] = {}
    payload_paths: dict[str, Path] = {}
    for path in sorted(result.glob("*_plans.json")):
        payload = json.loads(path.read_text())
        runtime_fields = payload.get("query_runtime_fields")
        require(isinstance(runtime_fields, list)
                and "query_id" in runtime_fields,
                f"{scope}: runtime query receipt is incomplete")
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{scope}: analysis role may have reached runtime")
        require(not ({"analysis_role", "role", "query_role"}
                     & set(runtime_fields)),
                f"{scope}: privileged role field reached runtime")
        stem_matches = [query_id for query_id in by_query if query_id in path.stem]
        require(len(stem_matches) == 1,
                f"{scope}: cannot bind plan file to query: {path.name}")
        query_id = stem_matches[0]
        require(query_id not in payloads,
                f"{scope}: duplicate plan file for {query_id}")
        payloads[query_id] = payload
        payload_paths[query_id] = path
    require(set(payloads) == set(by_query),
            f"{scope}: plan/metric query set differs")
    return {
        "root": root,
        "summary_path": summary_path,
        "metric_path": metric_path,
        "compute_path": compute_path,
        "health_path": health_path,
        "summary": summary,
        "compute": compute,
        "health": health,
        "rows": by_query,
        "payloads": payloads,
        "payload_paths": payload_paths,
    }


def _process_identity(compute: dict[str, Any], key: str) -> Any:
    value = compute.get(key)
    if value is None:
        return None
    require(isinstance(value, dict), f"invalid {key} process receipt")
    return int(value["pid"]), int(value["process_start_ticks"])


def _finite_nonnegative(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    require(math.isfinite(value) and value >= 0.0,
            f"invalid non-negative metric {field}")
    return value


def _validate_plan(plan: dict[str, Any], scope: str) -> None:
    require(plan.get("cec_portability_schema") == HUB_SCHEMA,
            f"{scope}: hub schema changed")
    require(plan.get("role_label_visible") is False,
            f"{scope}: runtime role label leaked")
    require(plan.get("metric_depth_sensor_consumed") is False
            and plan.get("metric_depth_sensor_consumed_by_policy") is False,
            f"{scope}: simulator metric depth reached the policy")
    require(plan.get("cec_accept_controller") == "vint"
            and plan.get("cec_reject_controller") == "vint",
            f"{scope}: CEC did not keep ViNT on both branches")
    require(plan.get("cec_reject_policy") == "controller_native_exact",
            f"{scope}: reject policy is not controller-native exact")
    receipt = plan.get("cec_controller_portability_receipt")
    require(isinstance(receipt, dict)
            and receipt.get("controller") == "vint"
            and receipt.get("endpoint") == "imagegoal_step"
            and receipt.get("reject_policy") == "controller_native_exact"
            and receipt.get("fallback_controller") == "vint",
            f"{scope}: ViNT portability receipt is incomplete")


def audit_pair(root: Path) -> dict[str, Any]:
    contract_path = root / "authority_pair_contract.json"
    require(contract_path.is_file(), "authority-pair contract is missing")
    contract = json.loads(contract_path.read_text())
    require(contract.get("schema_version")
            == "cec_authority_pair_contract_v2_20260828",
            "authority-pair contract schema changed")
    require(contract.get("controller") == "vint"
            and contract.get("reject_policy") == "controller_native_exact",
            "authority-pair treatment is not ViNT controller-native CEC")
    require(contract.get("runtime_role_visibility") == "none",
            "authority-pair contract exposes runtime role")
    order = contract.get("authority_order")
    require(order in (["grant", "forced_reject_native"],
                      ["forced_reject_native", "grant"]),
            "authority order is invalid")

    grant = _load_arm(root / "grant", "grant")
    forced = _load_arm(
        root / "forced_reject_native", "forced_reject_native")
    for scope, arm in (("grant", grant), ("forced", forced)):
        summary = arm["summary"]
        require(summary.get("server_backend") == "cec_portability"
                and summary.get("runtime_role_visibility") == "none",
                f"{scope}: backend or role-visibility contract changed")
        require(summary.get("queries") == 2
                and summary.get("role_counts")
                == {"novel": 1, "revisit": 1},
                f"{scope}: mixed-role population changed")
        require(arm["health"].get("reject_policy")
                == "controller_native_exact"
                and arm["health"].get("reject_controller") == "vint",
                f"{scope}: live hub is not controller-native")
        roles = {str(row["analysis_role"]) for row in arm["rows"].values()}
        require(roles == set(ROLES), f"{scope}: role balance changed")
        for query_id, row in arm["rows"].items():
            require(row.get("shared_A_hashes_ok") == "1"
                    and row.get("shared_A_diffusion_samples") == "0",
                    f"{scope}/{query_id}: causal replay changed")
            require(row.get("metric_depth_sensor_consumed_any") == "0"
                    and row.get("runtime_failure_plans") == "0",
                    f"{scope}/{query_id}: sensor/runtime contract failed")
            plans = arm["payloads"][query_id].get("query_leg")
            require(isinstance(plans, list) and plans,
                    f"{scope}/{query_id}: query has no controller decisions")
            for plan in plans:
                _validate_plan(plan, f"{scope}/{query_id}")

    require(set(grant["rows"]) == set(forced["rows"]),
            "paired query identities differ")
    require(grant["compute"].get("host") == forced["compute"].get("host")
            and grant["compute"].get("gpu_uuid")
            == forced["compute"].get("gpu_uuid"),
            "paired arms did not share host/GPU")
    for key in ("memnav", "navdp", "accepted_controller", "controller_proxy"):
        require(_process_identity(grant["compute"], key)
                == _process_identity(forced["compute"], key),
                f"paired arms did not share {key}")

    query_results: list[dict[str, Any]] = []
    for query_id in sorted(grant["rows"]):
        grant_row = grant["rows"][query_id]
        forced_row = forced["rows"][query_id]
        for field in (
            "scene", "episode", "pair_id", "query_id", "analysis_role",
            "seed", "shared_A_frames", "shared_A_decision_frames",
            "geodesic_m",
        ):
            require(grant_row.get(field) == forced_row.get(field),
                    f"{query_id}: paired start differs in {field}")
        grant_payload = grant["payloads"][query_id]
        forced_payload = forced["payloads"][query_id]
        require(grant_payload.get("replay") == forced_payload.get("replay")
                and grant_payload.get("legA") == forced_payload.get("legA"),
                f"{query_id}: causal prefix differs across arms")

        grant_plans = grant_payload["query_leg"]
        forced_plans = forced_payload["query_leg"]
        first_grant, first_forced = grant_plans[0], forced_plans[0]
        proof = first_grant.get("cec_proof_sha256")
        require(isinstance(proof, str) and SHA256.fullmatch(proof) is not None
                and first_forced.get("cec_proof_sha256") == proof,
                f"{query_id}: first proof differs before treatment")
        for field in (
            "cec_frame_idx", "cec_goal_sha256", "cec_goal_start_frame",
            "cec_selected_anchor", "cec_shadow_takeover", "cec_reason",
        ):
            require(first_grant.get(field) == first_forced.get(field),
                    f"{query_id}: first decision differs in {field}")
        require(first_grant.get("cec_takeover")
                is first_grant.get("cec_shadow_takeover"),
                f"{query_id}: grant did not follow its proof")
        require(first_forced.get("cec_takeover") is False,
                f"{query_id}: baseline granted control")
        require(all(plan.get("cec_takeover") is False
                    and plan.get("cec_forced_reject_native") is True
                    for plan in forced_plans),
                f"{query_id}: baseline did not withhold every takeover")

        packet_verified = False
        packet_sha = None
        grant_packet = first_grant.get("cec_handoff_packet")
        forced_packet = first_forced.get("cec_handoff_packet")
        require((grant_packet is None) == (forced_packet is None),
                f"{query_id}: first packet presence differs")
        require((grant_packet is not None)
                == bool(first_grant.get("cec_shadow_takeover")),
                f"{query_id}: packet presence does not follow first proof")
        if grant_packet is not None:
            require(grant_packet == forced_packet,
                    f"{query_id}: first handoff packet differs")
            verify_handoff_packet_envelope(grant_packet)
            packet_sha = str(grant_packet["packet_sha256"])
            require(grant_packet.get("proof_sha256") == proof,
                    f"{query_id}: packet/proof binding differs")
            require(first_grant.get("cec_handoff_packet_sha256") == packet_sha
                    and first_forced.get("cec_handoff_packet_sha256")
                    == packet_sha,
                    f"{query_id}: packet digest receipt differs")
            packet_verified = True

        grant_takeovers = sum(
            plan.get("cec_takeover") is True for plan in grant_plans)
        exact_fallback_trace_match = None
        if grant_takeovers == 0:
            exact_fallback_trace_match = bool(
                grant_payload.get("rollout_traces", {}).get("query")
                == forced_payload.get("rollout_traces", {}).get("query")
                and grant_payload.get("query_result")
                == forced_payload.get("query_result"))
            require(exact_fallback_trace_match,
                    f"{query_id}: all-reject execution was not exact fallback")

        initial = _finite_nonnegative(grant_row, "geodesic_m")
        grant_final = _finite_nonnegative(grant_row, "final_goal_dist_m")
        forced_final = _finite_nonnegative(forced_row, "final_goal_dist_m")
        grant_success = int(grant_row["reached"])
        forced_success = int(forced_row["reached"])
        query_results.append({
            "scene": grant_row["scene"],
            "episode": grant_row["episode"],
            "pair_id": grant_row["pair_id"],
            "query_id": query_id,
            "analysis_role": grant_row["analysis_role"],
            "first_proof_sha256": proof,
            "first_shadow_takeover": bool(
                first_grant.get("cec_shadow_takeover")),
            "first_anchor": first_grant.get("cec_selected_anchor"),
            "first_packet_verified": packet_verified,
            "first_packet_sha256": packet_sha,
            "grant_takeover_plans": int(grant_takeovers),
            "forced_takeover_plans": 0,
            "grant_success": grant_success,
            "native_success": forced_success,
            "paired_gain": int(grant_success == 1 and forced_success == 0),
            "paired_loss": int(grant_success == 0 and forced_success == 1),
            "initial_geodesic_m": initial,
            "grant_final_distance_m": grant_final,
            "native_final_distance_m": forced_final,
            "grant_path_len_m": _finite_nonnegative(grant_row, "path_len_m"),
            "native_path_len_m": _finite_nonnegative(
                forced_row, "path_len_m"),
            "grant_steps": int(grant_row["steps"]),
            "native_steps": int(forced_row["steps"]),
            "exact_fallback_trace_match": exact_fallback_trace_match,
            "post_divergence_proof_equality_required": False,
            "files": {
                "grant_plan_sha256": sha256_file(
                    grant["payload_paths"][query_id]),
                "native_plan_sha256": sha256_file(
                    forced["payload_paths"][query_id]),
            },
        })

    return {
        "schema_version": SCHEMA,
        "verified": True,
        "controller": "vint",
        "reject_policy": "controller_native_exact",
        "scene": str(contract["scene"]),
        "episode": str(contract["episode"]),
        "authority_order": order,
        "same_process_pair": True,
        "runtime_role_visibility": "none",
        "query_count": 2,
        "query_results": query_results,
        "files": {
            "contract_sha256": sha256_file(contract_path),
            "grant_summary_sha256": sha256_file(grant["summary_path"]),
            "native_summary_sha256": sha256_file(forced["summary_path"]),
            "grant_metric_sha256": sha256_file(grant["metric_path"]),
            "native_metric_sha256": sha256_file(forced["metric_path"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_pair(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "verified": True,
        "queries": result["query_count"],
        "gain": sum(row["paired_gain"] for row in result["query_results"]),
        "loss": sum(row["paired_loss"] for row in result["query_results"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
