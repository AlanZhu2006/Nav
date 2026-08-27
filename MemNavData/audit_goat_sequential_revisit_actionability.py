#!/usr/bin/env python3
"""Post-hoc actionability audit for a frozen GOAT sequential-Revisit run.

This audit never changes the confirmatory population or method.  It answers a
strictly mechanistic question left out of the preregistered summary: did a
certificate acceptance ever produce an executable non-STOP intervention?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "goat_sequential_revisit_actionability_audit_v1_20260815"
RESULT_NAME = "goat_sequential_revisit_pilot.json"
SUBTASK_STOP_ID = 6


def _digest(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _accepted(record: Mapping[str, Any]) -> bool:
    certificate = record.get("certificate")
    return bool(
        isinstance(certificate, Mapping)
        and certificate.get("ok") is True
        and certificate.get("accepted") is True
        and certificate.get("pointgoal_units")
        == "lingbot_raw_direction_only"
    )


def _positions_match(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= 1e-6 for a, b in zip(left, right)
    )


def _verify_sidecar(result_path: pathlib.Path) -> None:
    sidecar = result_path.with_suffix(result_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("missing result SHA sidecar: {}".format(sidecar))
    fields = sidecar.read_text().strip().split()
    if not fields or fields[0] != _digest(result_path):
        raise RuntimeError("result SHA sidecar mismatch: {}".format(result_path))


def audit(manifest_path: pathlib.Path, run_root: pathlib.Path,
          summary_path: pathlib.Path) -> Dict[str, Any]:
    manifest_path = pathlib.Path(manifest_path)
    run_root = pathlib.Path(run_root)
    summary_path = pathlib.Path(summary_path)
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest_sha = _digest(manifest_path)
    if summary.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("summary refers to another manifest")

    accepted_events = []
    total_plans = 0
    override_events = 0
    override_episodes = 0
    first_override_episodes = 0
    exact_cross_arm_episodes = 0
    verified_sidecars = 0

    for entry in manifest["episodes"]:
        index = int(entry["index"])
        path = run_root / "episodes" / "{:03d}".format(index) / RESULT_NAME
        if not path.is_file():
            raise RuntimeError("missing raw result: {}".format(path))
        _verify_sidecar(path)
        verified_sidecars += 1
        raw = json.loads(path.read_text())
        if raw.get("complete") is not True:
            raise RuntimeError("raw result is incomplete: {}".format(path))
        if raw.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("raw manifest digest mismatch")
        if raw.get("manifest_entry") != entry:
            raise RuntimeError("raw manifest entry mismatch")
        pair = raw.get("pairs", [])
        if len(pair) != 1:
            raise RuntimeError("raw result does not contain one pair")
        pair = pair[0]
        native = pair["native"]
        cec = pair["cec"]
        target = int(entry["target_subtask_index"])
        total_plans += int(cec.get("navdp_plan_count", 0))
        first_override_episodes += int(
            cec.get("first_override_step") is not None)

        episode_override_events = 0
        for record in cec.get("records", []):
            official = int(record["official_action_id"])
            executed = int(record["executed_action_id"])
            if official != executed:
                override_events += 1
                episode_override_events += 1
            if _accepted(record):
                accepted_events.append({
                    "index": index,
                    "scene_id": str(pair["scene_id"]),
                    "episode_id": str(pair["episode_id"]),
                    "step": int(record["step"]),
                    "subtask_index": int(record["subtask_before"]),
                    "is_target_subtask": (
                        int(record["subtask_before"]) == target),
                    "official_action_id": official,
                    "official_action": str(record["official_action"]),
                    "executed_action_id": executed,
                    "executed_action": str(record["executed_action"]),
                    "action_source": str(record["action_source"]),
                    "navdp_plan_present": record.get("navdp_plan") is not None,
                })
        override_episodes += int(episode_override_events > 0)

        native_records = native.get("records", [])
        cec_records = cec.get("records", [])
        exact = len(native_records) == len(cec_records)
        if exact:
            for left, right in zip(native_records, cec_records):
                if (int(left["executed_action_id"])
                        != int(right["executed_action_id"])
                        or not _positions_match(
                            left["position_before"], right["position_before"])):
                    exact = False
                    break
        exact_cross_arm_episodes += int(exact)

    n = len(manifest["episodes"])
    stop_accepts = sum(
        int(event["official_action_id"] == SUBTASK_STOP_ID)
        for event in accepted_events)
    actionable_accepts = len(accepted_events) - stop_accepts
    target_accepts = sum(
        int(event["is_target_subtask"]) for event in accepted_events)
    no_op = bool(
        n > 0
        and exact_cross_arm_episodes == n
        and total_plans == 0
        and override_events == 0
        and first_override_episodes == 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "audit_source_sha256": _digest(pathlib.Path(__file__).resolve()),
        "posthoc_intervention_audit_not_preregistered": True,
        "method_or_threshold_selection_allowed": False,
        "manifest_sha256": manifest_sha,
        "summary_sha256": _digest(summary_path),
        "raw_result_count": n,
        "result_sha_sidecars_verified": verified_sidecars,
        "certificate_accept_events": len(accepted_events),
        "target_certificate_accept_events": target_accepts,
        "pre_target_certificate_accept_events": (
            len(accepted_events) - target_accepts),
        "accept_events_on_official_subtask_stop": stop_accepts,
        "actionable_non_stop_accept_events": actionable_accepts,
        "navdp_plan_count": total_plans,
        "executed_override_events": override_events,
        "executed_override_episodes": override_episodes,
        "first_override_episodes": first_override_episodes,
        "cross_arm_action_pose_exact_episodes": exact_cross_arm_episodes,
        "behaviorally_identical_episodes": exact_cross_arm_episodes,
        "all_accepts_non_actionable_under_stop_contract": bool(
            accepted_events and stop_accepts == len(accepted_events)),
        "formal_effect_is_degenerate_noop": no_op,
        "formal_null_classification": (
            "degenerate_noop_no_executed_intervention" if no_op
            else "nondegenerate_intervention_observed"),
        "accepted_events": accepted_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite {}".format(args.output))
    payload = audit(args.manifest, args.run_root, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
