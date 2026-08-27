#!/usr/bin/env python3
"""Audit one same-process grant/forced-reject controller pair.

The two arms must share the exact causal query start and loaded processes.  A
granted arm may diverge after its first certified action; requiring later proof
streams to remain byte-identical would therefore be a causal error.  This
auditor binds only the first decision, then scores the resulting paired
rollouts without pretending their later observations are interchangeable.
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

from MemNavData.cec_handoff_contract import (
    verify_handoff_packet_envelope,
)


SCHEMA = "cec_authority_pair_audit_v1_20260827"
HUB_SCHEMA = "cec_controller_portability_hub_v2"
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arm(root: Path, scope: str) -> dict[str, Any]:
    require(root.is_dir(), f"{scope}: arm root is missing")
    result = root / "result"
    summary_path = result / "summary.json"
    metric_path = result / "metric.csv"
    compute_path = root / "compute_identity.json"
    require(summary_path.is_file() and metric_path.is_file()
            and compute_path.is_file(), f"{scope}: incomplete arm")
    summary = json.loads(summary_path.read_text())
    compute = json.loads(compute_path.read_text())
    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"{scope}: accepted-set pair expects one query")
    row = rows[0]
    matches = sorted(result.glob("*_plans.json"))
    require(len(matches) == 1, f"{scope}: expected one query plan file")
    payload = json.loads(matches[0].read_text())
    plans = payload.get("query_leg")
    require(isinstance(plans, list) and plans,
            f"{scope}: query has no decisions")
    return {
        "root": root,
        "summary_path": summary_path,
        "metric_path": metric_path,
        "plan_path": matches[0],
        "summary": summary,
        "compute": compute,
        "row": row,
        "payload": payload,
        "plans": plans,
    }


def _process_identity(compute: dict[str, Any], key: str) -> Any:
    value = compute.get(key)
    if value is None:
        return None
    require(isinstance(value, dict), f"invalid {key} process receipt")
    return (int(value["pid"]), int(value["process_start_ticks"]))


def audit_pair(
    root: Path,
    expected_controller: str | None = None,
    query_manifest: Path | None = None,
) -> dict[str, Any]:
    grant = _load_arm(root / "grant", "grant")
    forced = _load_arm(root / "forced_reject_native", "forced_reject_native")

    for scope, arm in (("grant", grant), ("forced", forced)):
        summary = arm["summary"]
        row = arm["row"]
        require(summary.get("server_backend") == "cec_portability",
                f"{scope}: wrong backend")
        require(summary.get("runtime_role_visibility") == "none",
                f"{scope}: runtime role label leaked")
        require(summary.get("queries") == 1,
                f"{scope}: accepted-set query count changed")
        require(row.get("analysis_role") == "revisit",
                f"{scope}: frozen accepted-set identity changed")
        require(row.get("shared_A_hashes_ok") == "1"
                and row.get("shared_A_diffusion_samples") == "0",
                f"{scope}: causal A replay changed")
        require(row.get("metric_depth_sensor_consumed_any") == "0",
                f"{scope}: simulator depth entered policy")
        require(row.get("runtime_failure_plans") == "0",
                f"{scope}: runtime failure was hidden")
        for plan in arm["plans"]:
            require(plan.get("cec_portability_schema") == HUB_SCHEMA,
                    f"{scope}: hub schema changed")
            require(plan.get("role_label_visible") is False,
                    f"{scope}: plan exposed a role label")
            require(plan.get("metric_depth_sensor_consumed") is False,
                    f"{scope}: plan sensor contract changed")

    controller = str(grant["plans"][0].get("cec_accept_controller"))
    require(controller and controller != "None", "grant controller is missing")
    if expected_controller is not None:
        require(controller == expected_controller,
                "grant controller differs from the frozen arm")
    require(all(plan.get("cec_accept_controller") == controller
                for plan in grant["plans"] + forced["plans"]),
            "controller identity changed within the pair")

    # The physical query start and every upstream process are shared.  Hubs are
    # intentionally separate because one is immutable grant and one immutable
    # forced reject; both point at the same live processes.
    require(grant["compute"]["host"] == forced["compute"]["host"]
            and grant["compute"]["gpu_uuid"] == forced["compute"]["gpu_uuid"],
            "paired arms did not run on the same host/GPU")
    for key in ("memnav", "navdp", "accepted_controller", "controller_proxy"):
        require(_process_identity(grant["compute"], key)
                == _process_identity(forced["compute"], key),
                f"paired arms did not share the {key} process")

    for field in ("scene", "episode", "pair_id", "query_id", "seed",
                  "shared_A_frames", "shared_A_decision_frames", "geodesic_m"):
        require(grant["row"].get(field) == forced["row"].get(field),
                f"paired query start differs in {field}")
    require(grant["payload"].get("replay") == forced["payload"].get("replay"),
            "paired causal replay receipts differ")
    require(grant["payload"].get("legA") == forced["payload"].get("legA"),
            "paired online-A plans differ")

    first_grant = grant["plans"][0]
    first_forced = forced["plans"][0]
    proof = first_grant.get("cec_proof_sha256")
    require(isinstance(proof, str) and SHA256.fullmatch(proof) is not None,
            "first grant proof digest is invalid")
    require(first_forced.get("cec_proof_sha256") == proof,
            "first CEC proof differs before authority treatment")
    for field in ("cec_frame_idx", "cec_goal_sha256", "cec_goal_start_frame",
                  "cec_selected_anchor"):
        require(first_grant.get(field) == first_forced.get(field),
                f"first certified handoff differs in {field}")
    require(first_grant.get("cec_shadow_takeover") is True
            and first_grant.get("cec_takeover") is True
            and first_grant.get("cec_action_state") == "takeover",
            "grant arm did not execute the first certified handoff")
    require(first_forced.get("cec_shadow_takeover") is True
            and first_forced.get("cec_takeover") is False
            and first_forced.get("cec_action_state") == "forced_reject",
            "forced arm did not withhold the same first handoff")
    require(all(plan.get("cec_takeover") is False for plan in forced["plans"]),
            "forced-reject arm granted a later takeover")
    require(all(plan.get("cec_forced_reject_native") is True
                for plan in forced["plans"]),
            "forced-reject receipt is incomplete")

    grant_packet = first_grant.get("cec_handoff_packet")
    forced_packet = first_forced.get("cec_handoff_packet")
    require((grant_packet is None) == (forced_packet is None),
            "paired handoff packet presence differs")
    handoff_packet_verified = False
    packet_sha256 = None
    source_manifest_match = None
    if grant_packet is not None:
        grant_public = verify_handoff_packet_envelope(grant_packet)
        forced_public = verify_handoff_packet_envelope(forced_packet)
        require(grant_packet == forced_packet,
                "paired arms did not consume the same handoff packet")
        require(grant_public == forced_public,
                "paired handoff public proofs differ")
        packet_sha256 = str(grant_packet["packet_sha256"])
        require(first_grant.get("cec_handoff_packet_sha256") == packet_sha256
                and first_forced.get("cec_handoff_packet_sha256")
                == packet_sha256,
                "handoff packet receipt digest differs")
        require(grant_packet["proof_sha256"] == proof,
                "handoff packet does not bind the first CEC proof")
        require(grant_packet["goal_rgb_sha256"]
                == first_grant.get("cec_goal_sha256"),
                "handoff packet does not bind the runtime goal")
        handoff_packet_verified = True

    if query_manifest is not None:
        require(query_manifest.is_file(), "accepted query manifest is missing")
        manifest = json.loads(query_manifest.read_text())
        require(manifest.get("schema_version")
                == "cec_first_decision_accepted_population_v1_20260827",
                "accepted query manifest schema changed")
        matches = [entry for entry in manifest.get("queries", [])
                   if (str(entry.get("scene")) == str(grant["row"]["scene"])
                       and str(entry.get("episode"))
                       == str(grant["row"]["episode"])
                       and str(entry.get("query_id"))
                       == str(grant["row"]["query_id"]))]
        require(len(matches) == 1,
                "paired query is not uniquely bound in accepted manifest")
        entry = matches[0]
        require(not ({"analysis_role", "role", "query_role"} & set(entry)),
                "accepted runtime manifest leaked a role")
        require(entry.get("goal_rgb_sha256")
                == first_grant.get("cec_goal_sha256"),
                "accepted manifest goal binding differs")
        require(entry.get("first_proof_sha256") == proof,
                "live first proof differs from the frozen accepted proof")
        require(int(entry.get("selected_anchor"))
                == int(first_grant.get("cec_selected_anchor")),
                "live first anchor differs from the frozen accepted anchor")
        if grant_packet is not None:
            require(entry.get("causal_history_sha256")
                    == grant_packet.get("causal_history_sha256"),
                    "handoff packet causal history differs from manifest")
            require(entry.get("selected_anchor_image_sha256")
                    == grant_packet.get("anchor_jpeg_sha256"),
                    "handoff anchor bytes differ from manifest")
        source_manifest_match = True

    grant_success = int(grant["row"]["reached"])
    forced_success = int(forced["row"]["reached"])
    initial_distance = float(grant["row"]["geodesic_m"])
    grant_final = float(grant["row"]["final_goal_dist_m"])
    forced_final = float(forced["row"]["final_goal_dist_m"])
    require(all(math.isfinite(value) and value >= 0.0 for value in
                (initial_distance, grant_final, forced_final)),
            "paired distance metric is invalid")

    output = {
        "schema_version": SCHEMA,
        "verified": True,
        "controller": controller,
        "scene": grant["row"]["scene"],
        "episode": grant["row"]["episode"],
        "query_id": grant["row"]["query_id"],
        "same_process_pair": True,
        "first_handoff_proof_sha256": proof,
        "first_handoff_anchor": first_grant.get("cec_selected_anchor"),
        "handoff_packet_verified": handoff_packet_verified,
        "handoff_packet_sha256": packet_sha256,
        "source_accepted_manifest_match": source_manifest_match,
        "grant_success": grant_success,
        "forced_reject_success": forced_success,
        "paired_gain": int(grant_success == 1 and forced_success == 0),
        "paired_loss": int(grant_success == 0 and forced_success == 1),
        "initial_geodesic_m": initial_distance,
        "grant_final_distance_m": grant_final,
        "forced_reject_final_distance_m": forced_final,
        "grant_progress_m": initial_distance - grant_final,
        "forced_reject_progress_m": initial_distance - forced_final,
        "grant_plans": len(grant["plans"]),
        "forced_reject_plans": len(forced["plans"]),
        "post_handoff_proof_equality_required": False,
        "files": {
            "grant_summary_sha256": sha256_file(grant["summary_path"]),
            "forced_summary_sha256": sha256_file(forced["summary_path"]),
            "grant_plan_sha256": sha256_file(grant["plan_path"]),
            "forced_plan_sha256": sha256_file(forced["plan_path"]),
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controller")
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = audit_pair(
        args.root.resolve(), args.controller,
        None if args.query_manifest is None else args.query_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verified": True,
        "controller": output["controller"],
        "gain": output["paired_gain"],
        "loss": output["paired_loss"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
