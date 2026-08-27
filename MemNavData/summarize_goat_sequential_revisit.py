#!/usr/bin/env python3
"""Strictly summarize a frozen paired GOAT sequential-Revisit evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import pathlib
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = "goat_sequential_revisit_summary_v1_20260815"
RESULT_NAME = "goat_sequential_revisit_pilot.json"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _choose(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for offset in range(1, k + 1):
        value = value * (n - k + offset) // offset
    return value


def exact_mcnemar_two_sided(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(_choose(discordant, value)
               for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * float(tail) / float(2 ** discordant))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction)
                 + ordered[upper] * fraction)


def scene_cluster_interval(
        rows: Sequence[Mapping[str, Any]], repetitions: int = 20000,
        seed: int = 20260815) -> List[float]:
    clusters: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["scene_id"])].append(
            float(row["cec_target_success"])
            - float(row["native_target_success"]))
    names = sorted(clusters)
    if not names:
        raise ValueError("no scene clusters")
    rng = random.Random(seed)
    draws = []
    for _ in range(int(repetitions)):
        sample = [names[rng.randrange(len(names))] for _ in names]
        values = [value for name in sample for value in clusters[name]]
        draws.append(100.0 * sum(values) / len(values))
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _success_vector(arm: Mapping[str, Any]) -> Sequence[Any]:
    success = arm.get("metrics", {}).get("success", {})
    values = success.get("subtask_success", []) if isinstance(
        success, Mapping) else []
    return values if isinstance(values, list) else []


def _certificate_accepted(record: Mapping[str, Any]) -> bool:
    certificate = record.get("certificate")
    return bool(isinstance(certificate, Mapping)
                and certificate.get("ok") is True
                and certificate.get("accepted") is True)


def _same_trajectory(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = left.get("records", [])
    second = right.get("records", [])
    if len(first) != len(second):
        return False
    for lhs, rhs in zip(first, second):
        if lhs.get("executed_action_id") != rhs.get("executed_action_id"):
            return False
        lpos = lhs.get("position_before")
        rpos = rhs.get("position_before")
        if not isinstance(lpos, list) or not isinstance(rpos, list):
            return False
        if len(lpos) != len(rpos):
            return False
        if any(abs(float(a) - float(b)) > 1e-6
               for a, b in zip(lpos, rpos)):
            return False
    return True


def _pair_row(entry: Mapping[str, Any], pair: Mapping[str, Any]) -> Dict[str, Any]:
    native = pair["native"]
    cec = pair["cec"]
    target = int(entry["target_subtask_index"])
    target_records = [record for record in cec.get("records", [])
                      if int(record["subtask_before"]) == target]
    pre_target_records = [record for record in cec.get("records", [])
                          if int(record["subtask_before"]) < target]
    target_probes = [record for record in target_records
                     if isinstance(record.get("certificate"), Mapping)]
    candidate_probes = [record for record in target_probes
                        if record["certificate"].get("probe_candidates")]
    target_accepts = sum(
        int(_certificate_accepted(record)) for record in target_records)
    pre_target_accepts = sum(
        int(_certificate_accepted(record)) for record in pre_target_records)
    rejection_reasons = Counter()
    for record in target_probes:
        if not _certificate_accepted(record):
            rejection_reasons[str(
                record["certificate"].get("reason", "missing_reason"))] += 1
    prior_modalities = sorted({
        str(item["modality"])
        for item in entry["prior_instance_subtasks"]
    })
    native_successes = _success_vector(native)
    cec_successes = _success_vector(cec)
    prior_indices = [int(item["subtask_index"])
                     for item in entry["prior_instance_subtasks"]]
    native_prior_successes = [
        float(native_successes[index]) if index < len(native_successes) else 0.0
        for index in prior_indices]
    cec_prior_successes = [
        float(cec_successes[index]) if index < len(cec_successes) else 0.0
        for index in prior_indices]
    no_accept = int(cec.get("certificate_accept_count", 0)) == 0
    all_official_stops_preserved = all(
        int(record.get("official_action_id", -1)) != 6
        or int(record.get("executed_action_id", -1)) == 6
        for record in cec.get("records", []))
    return {
        "index": int(entry["index"]),
        "scene_id": str(entry["scene_id"]),
        "episode_id": str(entry["episode_id"]),
        "target_subtask_index": target,
        "target_instance_id": str(entry["target_instance_id"]),
        "prior_modalities": prior_modalities,
        "native_target_entered": bool(native.get("target_entered")),
        "cec_target_entered": bool(cec.get("target_entered")),
        "both_target_entered": bool(
            native.get("target_entered") and cec.get("target_entered")),
        "native_complete_through_target": bool(
            native.get("complete_through_target")),
        "cec_complete_through_target": bool(cec.get("complete_through_target")),
        "native_target_success": float(native["target_success"]),
        "cec_target_success": float(cec["target_success"]),
        "native_prior_instance_successes": native_prior_successes,
        "cec_prior_instance_successes": cec_prior_successes,
        "native_steps": int(native["steps"]),
        "cec_steps": int(cec["steps"]),
        "native_termination_reason": str(native.get("termination_reason")),
        "cec_termination_reason": str(cec.get("termination_reason")),
        "target_probe_count": len(target_probes),
        "target_candidate_probe_count": len(candidate_probes),
        "target_certificate_accept_count": target_accepts,
        "pre_target_nonrecurrent_accept_count": pre_target_accepts,
        "cec_certificate_accept_count": int(
            cec.get("certificate_accept_count", 0)),
        "cec_navdp_plan_count": int(cec.get("navdp_plan_count", 0)),
        "first_override_step": cec.get("first_override_step"),
        "prefix_paired": bool(pair["prefix_audit"][
            "prefix_paired_before_first_override"]),
        "all_official_subtask_stops_preserved": all_official_stops_preserved,
        "exact_fallback_when_no_accept": (
            _same_trajectory(native, cec) if no_accept else None),
        "target_rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def paired_effect(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    native = sum(float(row["native_target_success"]) for row in rows)
    cec = sum(float(row["cec_target_success"]) for row in rows)
    gains = sum(int(not row["native_target_success"]
                    and row["cec_target_success"]) for row in rows)
    losses = sum(int(row["native_target_success"]
                     and not row["cec_target_success"]) for row in rows)
    return {
        "n": len(rows),
        "native_successes": int(native),
        "cec_successes": int(cec),
        "native_success_rate": native / len(rows),
        "cec_success_rate": cec / len(rows),
        "risk_difference_pp": 100.0 * (cec - native) / len(rows),
        "paired_gains": gains,
        "paired_losses": losses,
        "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(gains, losses),
        "scene_cluster_bootstrap_95ci_pp": scene_cluster_interval(rows),
    }


def summarize(manifest_path: pathlib.Path, run_root: pathlib.Path) -> Dict[str, Any]:
    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = _sha256_file(manifest_path)
    entries = manifest["episodes"]
    expected_by_index = {int(entry["index"]): entry for entry in entries}
    files = sorted((pathlib.Path(run_root) / "episodes").glob(
        "*/" + RESULT_NAME))
    if len(files) != len(entries):
        raise RuntimeError("expected {} results, found {}".format(
            len(entries), len(files)))
    by_index = {}
    checkpoint_hashes = set()
    goat_commits = set()
    for path in files:
        payload = json.loads(path.read_text())
        if payload.get("complete") is not True:
            raise RuntimeError("incomplete result {}".format(path))
        if payload.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("manifest hash mismatch in {}".format(path))
        if payload.get("role_label_read_by_controller") is not False:
            raise RuntimeError("controller role-label audit failed")
        if payload.get("official_goat_stop_authority_retained") is not True:
            raise RuntimeError("official STOP authority was not retained")
        if payload.get(
                "official_goat_subtask_stop_preserved_exactly") is not True:
            raise RuntimeError("official SUBTASK_STOP preservation audit failed")
        if payload.get(
                "official_policy_uses_released_stochastic_eval_semantics") is not True:
            raise RuntimeError("official stochastic policy semantics changed")
        if not str(payload.get("policy_device", "")).startswith("cuda"):
            raise RuntimeError("non-CUDA policy result is not formal")
        if manifest.get("evaluation_stage") == "formal_targeted_external_evaluation":
            provenance = payload.get("runtime_provenance", {})
            if not all(provenance.get(key) for key in (
                    "slurm_job_id", "slurm_array_job_id",
                    "slurm_array_task_id", "cuda_visible_devices")):
                raise RuntimeError("formal Slurm runtime provenance is incomplete")
            if int(provenance["slurm_array_task_id"]) != int(
                    payload["manifest_entry"]["index"]):
                raise RuntimeError("Slurm array index does not match manifest")
        if int(payload.get("max_steps", -1)) != int(
                manifest["analysis_contract"]["maximum_steps_per_arm"]):
            raise RuntimeError("max-step contract mismatch")
        if len(payload.get("pairs", [])) != 1:
            raise RuntimeError("each array result must contain exactly one pair")
        entry = payload.get("manifest_entry")
        index = int(entry["index"])
        if index not in expected_by_index or entry != expected_by_index[index]:
            raise RuntimeError("manifest entry mismatch at index {}".format(index))
        if bool(payload.get("paper_claim_authorized")) != bool(
                manifest.get("paper_claim_authorized")):
            raise RuntimeError("paper-claim scope mismatch")
        if index in by_index:
            raise RuntimeError("duplicate manifest index {}".format(index))
        by_index[index] = payload["pairs"][0]
        checkpoint_hashes.add(str(payload["checkpoint_sha256"]))
        goat_commits.add(str(payload["goat_commit"]))
    if sorted(by_index) != list(range(len(entries))):
        raise RuntimeError("result indices are not complete and contiguous")
    if len(checkpoint_hashes) != 1 or len(goat_commits) != 1:
        raise RuntimeError("checkpoint or GOAT commit changed across tasks")

    rows = []
    for entry in entries:
        index = int(entry["index"])
        pair = by_index[index]
        if (str(pair["scene_id"]), str(pair["episode_id"])) != (
                str(entry["scene_id"]), str(entry["episode_id"])):
            raise RuntimeError("episode identity mismatch at index {}".format(index))
        if pair.get("executed_arm_order") != entry.get("arm_order"):
            raise RuntimeError("arm order mismatch at index {}".format(index))
        row = _pair_row(entry, pair)
        if not row["prefix_paired"]:
            raise RuntimeError("pre-override pairing failed at index {}".format(index))
        if not row["all_official_subtask_stops_preserved"]:
            raise RuntimeError("official SUBTASK_STOP changed at index {}".format(index))
        if row["exact_fallback_when_no_accept"] is False:
            raise RuntimeError("certificate reject was not exact fallback at {}".format(
                index))
        rows.append(row)

    both_entered = sum(int(row["both_target_entered"]) for row in rows)
    minimum = manifest["analysis_contract"][
        "minimum_mechanistic_coverage_for_interpretation"]
    rejection_reasons = Counter()
    for row in rows:
        rejection_reasons.update(row["target_rejection_reasons"])
    strata = {}
    for label, predicate in (
            ("description_to_image",
             lambda row: "description" in row["prior_modalities"]),
            ("image_to_image",
             lambda row: "image" in row["prior_modalities"])):
        subset = [row for row in rows if predicate(row)]
        if subset:
            strata[label] = paired_effect(subset)
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "manifest_sha256": manifest_sha,
        "manifest_schema_version": manifest.get("schema_version"),
        "evaluation_stage": manifest.get("evaluation_stage"),
        "is_full_goat_benchmark_score": False,
        "paper_claim_authorized": bool(manifest.get("paper_claim_authorized")),
        "audit": {
            "expected_results": len(entries),
            "complete_results": len(rows),
            "distinct_scenes": len({row["scene_id"] for row in rows}),
            "all_prefixes_paired": all(row["prefix_paired"] for row in rows),
            "native_first_results": sum(
                int(entry["arm_order"][0] == "native") for entry in entries),
            "cec_first_results": sum(
                int(entry["arm_order"][0] == "cec") for entry in entries),
            "all_official_subtask_stops_preserved": all(
                row["all_official_subtask_stops_preserved"] for row in rows),
            "all_no_accepts_are_exact_fallback": all(
                row["exact_fallback_when_no_accept"] is not False
                for row in rows),
            "goat_commit": next(iter(goat_commits)),
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
        },
        "primary_intention_to_treat": paired_effect(rows),
        "constructibility": {
            "native_target_entered": sum(
                int(row["native_target_entered"]) for row in rows),
            "cec_target_entered": sum(
                int(row["cec_target_entered"]) for row in rows),
            "both_target_entered": both_entered,
            "both_target_entered_distinct_scenes": len({
                row["scene_id"] for row in rows if row["both_target_entered"]}),
            "mechanistic_coverage_gate_passed": bool(
                both_entered >= int(minimum["paired_episodes_entering_target"])
                and len({row["scene_id"] for row in rows
                         if row["both_target_entered"]})
                >= int(minimum["distinct_scenes"])),
            "target_candidate_supported_episodes": sum(
                int(row["target_candidate_probe_count"] > 0) for row in rows),
            "target_certificate_accepted_episodes": sum(
                int(row["target_certificate_accept_count"] > 0) for row in rows),
        },
        "safety_and_coverage": {
            "pre_target_nonrecurrent_accept_events": sum(
                row["pre_target_nonrecurrent_accept_count"] for row in rows),
            "pre_target_nonrecurrent_accept_episodes": sum(
                int(row["pre_target_nonrecurrent_accept_count"] > 0)
                for row in rows),
            "target_certificate_accept_events": sum(
                row["target_certificate_accept_count"] for row in rows),
            "target_certificate_accept_episodes": sum(
                int(row["target_certificate_accept_count"] > 0) for row in rows),
            "target_rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "descriptive_strata": strata,
        "episodes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _atomic_json(args.output, summarize(args.manifest, args.run_root))


if __name__ == "__main__":
    main()
