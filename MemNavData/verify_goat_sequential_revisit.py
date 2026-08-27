#!/usr/bin/env python3
"""Independent raw-result verifier for GOAT sequential-Revisit evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any, Dict, Mapping


SCHEMA_VERSION = "goat_sequential_revisit_verification_v1_20260815"
RESULT_NAME = "goat_sequential_revisit_pilot.json"


def _digest(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _combination(n: int, k: int) -> int:
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def _mcnemar(gain: int, loss: int) -> float:
    total = gain + loss
    if total == 0:
        return 1.0
    lower = min(gain, loss)
    probability = sum(_combination(total, k) for k in range(lower + 1))
    return min(1.0, 2.0 * probability / float(2 ** total))


def _accepted(record: Mapping[str, Any]) -> bool:
    certificate = record.get("certificate")
    return bool(isinstance(certificate, Mapping)
                and certificate.get("ok") is True
                and certificate.get("accepted") is True)


def verify(manifest_path: pathlib.Path, run_root: pathlib.Path,
           summary_path: pathlib.Path) -> Dict[str, Any]:
    manifest_path = pathlib.Path(manifest_path)
    run_root = pathlib.Path(run_root)
    summary_path = pathlib.Path(summary_path)
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest_hash = _digest(manifest_path)
    if summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("summary refers to another manifest")

    native_success = 0
    cec_success = 0
    gains = 0
    losses = 0
    native_entered = 0
    cec_entered = 0
    both_entered = 0
    target_accept_events = 0
    target_accept_episodes = 0
    pretarget_accept_events = 0
    pretarget_accept_episodes = 0
    checkpoint_hashes = set()
    goat_commits = set()
    for entry in manifest["episodes"]:
        index = int(entry["index"])
        result_path = (run_root / "episodes" / "{:03d}".format(index)
                       / RESULT_NAME)
        if not result_path.is_file():
            raise RuntimeError("missing raw result {}".format(result_path))
        raw = json.loads(result_path.read_text())
        if raw.get("complete") is not True:
            raise RuntimeError("raw result is incomplete")
        if raw.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("raw manifest digest mismatch")
        if raw.get("manifest_entry") != entry:
            raise RuntimeError("raw manifest entry mismatch")
        if raw.get("role_label_read_by_controller") is not False:
            raise RuntimeError("controller received evaluator role metadata")
        if raw.get(
                "official_goat_subtask_stop_preserved_exactly") is not True:
            raise RuntimeError("official SUBTASK_STOP contract missing")
        if not str(raw.get("policy_device", "")).startswith("cuda"):
            raise RuntimeError("raw result used a non-CUDA GOAT policy")
        provenance = raw.get("runtime_provenance", {})
        if manifest.get("evaluation_stage") == \
                "formal_targeted_external_evaluation":
            if not all(provenance.get(key) for key in (
                    "slurm_job_id", "slurm_array_job_id",
                    "slurm_array_task_id", "cuda_visible_devices")):
                raise RuntimeError("raw Slurm provenance is incomplete")
            if int(provenance["slurm_array_task_id"]) != index:
                raise RuntimeError("raw Slurm array index mismatch")
        pairs = raw.get("pairs", [])
        if len(pairs) != 1:
            raise RuntimeError("raw result does not contain one pair")
        pair = pairs[0]
        if (str(pair["scene_id"]), str(pair["episode_id"])) != (
                str(entry["scene_id"]), str(entry["episode_id"])):
            raise RuntimeError("raw identity mismatch")
        if pair.get("executed_arm_order") != entry.get("arm_order"):
            raise RuntimeError("raw arm order mismatch")
        if pair["prefix_audit"].get(
                "prefix_paired_before_first_override") is not True:
            raise RuntimeError("raw paired-prefix audit failed")
        native = pair["native"]
        cec = pair["cec"]
        if any(
                int(record.get("official_action_id", -1)) == 6
                and int(record.get("executed_action_id", -1)) != 6
                for record in cec.get("records", [])):
            raise RuntimeError("CEC replaced official SUBTASK_STOP")
        nvalue = float(native["target_success"])
        cvalue = float(cec["target_success"])
        if nvalue not in (0.0, 1.0) or cvalue not in (0.0, 1.0):
            raise RuntimeError("target success is not binary")
        native_success += int(nvalue)
        cec_success += int(cvalue)
        gains += int(nvalue == 0.0 and cvalue == 1.0)
        losses += int(nvalue == 1.0 and cvalue == 0.0)
        nentered = any(
            int(record["subtask_before"]) == int(entry["target_subtask_index"])
            for record in native.get("records", []))
        centered = any(
            int(record["subtask_before"]) == int(entry["target_subtask_index"])
            for record in cec.get("records", []))
        if bool(native.get("target_entered")) != nentered:
            raise RuntimeError("native target-entered receipt mismatch")
        if bool(cec.get("target_entered")) != centered:
            raise RuntimeError("CEC target-entered receipt mismatch")
        native_entered += int(nentered)
        cec_entered += int(centered)
        both_entered += int(nentered and centered)
        target = int(entry["target_subtask_index"])
        target_events = sum(
            int(_accepted(record)) for record in cec.get("records", [])
            if int(record["subtask_before"]) == target)
        pretarget_events = sum(
            int(_accepted(record)) for record in cec.get("records", [])
            if int(record["subtask_before"]) < target)
        target_accept_events += target_events
        target_accept_episodes += int(target_events > 0)
        pretarget_accept_events += pretarget_events
        pretarget_accept_episodes += int(pretarget_events > 0)
        checkpoint_hashes.add(str(raw["checkpoint_sha256"]))
        goat_commits.add(str(raw["goat_commit"]))

    primary = summary["primary_intention_to_treat"]
    expected_primary = {
        "n": len(manifest["episodes"]),
        "native_successes": native_success,
        "cec_successes": cec_success,
        "paired_gains": gains,
        "paired_losses": losses,
    }
    for key, expected in expected_primary.items():
        if int(primary[key]) != expected:
            raise RuntimeError("summary primary {} mismatch".format(key))
    expected_p = _mcnemar(gains, losses)
    if abs(float(primary["exact_mcnemar_two_sided_p"]) - expected_p) > 1e-15:
        raise RuntimeError("summary McNemar p-value mismatch")
    constructibility = summary["constructibility"]
    for key, expected in (
            ("native_target_entered", native_entered),
            ("cec_target_entered", cec_entered),
            ("both_target_entered", both_entered)):
        if int(constructibility[key]) != expected:
            raise RuntimeError("summary constructibility {} mismatch".format(key))
    safety = summary["safety_and_coverage"]
    for key, expected in (
            ("target_certificate_accept_events", target_accept_events),
            ("target_certificate_accept_episodes", target_accept_episodes),
            ("pre_target_nonrecurrent_accept_events", pretarget_accept_events),
            ("pre_target_nonrecurrent_accept_episodes", pretarget_accept_episodes)):
        if int(safety[key]) != expected:
            raise RuntimeError("summary safety {} mismatch".format(key))
    if len(checkpoint_hashes) != 1 or len(goat_commits) != 1:
        raise RuntimeError("raw dependency identities are inconsistent")
    if summary["audit"]["checkpoint_sha256"] not in checkpoint_hashes:
        raise RuntimeError("summary checkpoint receipt mismatch")
    if summary["audit"]["goat_commit"] not in goat_commits:
        raise RuntimeError("summary GOAT commit receipt mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "manifest_sha256": manifest_hash,
        "summary_sha256": _digest(summary_path),
        "raw_result_count": len(manifest["episodes"]),
        "independent_primary_recompute": dict(
            expected_primary,
            exact_mcnemar_two_sided_p=expected_p,
        ),
        "independent_constructibility_recompute": {
            "native_target_entered": native_entered,
            "cec_target_entered": cec_entered,
            "both_target_entered": both_entered,
        },
        "independent_safety_recompute": {
            "target_certificate_accept_events": target_accept_events,
            "target_certificate_accept_episodes": target_accept_episodes,
            "pre_target_nonrecurrent_accept_events": pretarget_accept_events,
            "pre_target_nonrecurrent_accept_episodes": pretarget_accept_episodes,
        },
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
    payload = verify(args.manifest, args.run_root, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
