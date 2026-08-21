#!/usr/bin/env python3
"""Audit exact fail-closed behavior in a strict-v4 mixed-role stream."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "certified_mixed_role_safety_gate_audit_v1_20260813"
ROLE_SEQUENCE = ["initial_imagegoal", "novel", "revisit"]
EXPECTED_ARMS = {
    "known_c_reference": {
        "server_backend": "hybrid_pose",
        "hybrid_route": "phase",
        "policy_backends": {"A": "navdp", "B": "navdp", "C": "navdp_mix"},
    },
    "certified": {
        "server_backend": "hybrid_pose",
        "hybrid_route": "certified_relocalization",
        "policy_backends": {
            "A": "navdp_auto",
            "B": "navdp_auto",
            "C": "navdp_auto",
        },
    },
}
PREFIX_METRICS = (
    "reached_{leg}",
    "steps_{leg}",
    "len_{leg}",
    "final_dist_{leg}",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_metric(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"metric must contain one row: {path}")
    return rows[0]


def boolean(value: Any) -> bool:
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False", ""):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def numeric_equal(first: str, second: str, tolerance: float = 1e-12) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def certificate_lifecycle(plans: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_cached") is not None
    ]
    uncached = [
        plan for plan in requests
        if plan.get("certified_relocalization_cached") is False
    ]
    reasons = Counter(
        str(plan.get("certified_relocalization_reason")) for plan in uncached
    )
    return {
        "plans": len(plans),
        "requests": len(requests),
        "uncached": len(uncached),
        "accepted_requests": sum(
            plan.get("certified_relocalization_accepted") is True
            for plan in requests
        ),
        "accepted_uncached": sum(
            plan.get("certified_relocalization_accepted") is True
            for plan in uncached
        ),
        "takeovers": sum(
            plan.get("revisit_adapter_takeover") is True for plan in plans
        ),
        "runtime_failures": sum(
            plan.get("certified_relocalization_reason")
            == "certificate_endpoint_failure"
            for plan in requests
        ),
        "uncached_reasons": dict(sorted(reasons.items())),
    }


def exact_prefix(reference: dict[str, Any], certified: dict[str, Any], leg: str,
                 reference_metric: dict[str, str],
                 certified_metric: dict[str, str]) -> dict[str, Any]:
    fields: dict[str, bool] = {}
    for template in PREFIX_METRICS:
        field = template.format(leg=leg)
        fields[field] = numeric_equal(reference_metric[field], certified_metric[field])
    fields["rollout_trace"] = (
        reference["rollout_traces"][f"leg{leg}"]
        == certified["rollout_traces"][f"leg{leg}"]
    )
    fields["memory_trace"] = (
        reference["memory_traces"][f"leg{leg}"]
        == certified["memory_traces"][f"leg{leg}"]
    )
    return {"all_exact": all(fields.values()), "fields": fields}


def _scene_name(scene_dir: Path) -> str:
    parts = scene_dir.name.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else scene_dir.name


def audit(run_root: Path, *, require_positive_control: bool = True) -> dict[str, Any]:
    contract = read_json(run_root / "run_contract.json")
    require(contract.get("protocol") ==
            "certified_mixed_role_safety_gate_v1_20260813",
            "run contract protocol changed")
    require(contract.get("deterministic_plan_seeds") is True,
            "run did not bind deterministic plan seeds")
    require(contract.get("blind_data_read") is False,
            "run contract does not explicitly forbid blind data")

    scenes_root = run_root / "scenes"
    require(scenes_root.is_dir(), "missing scenes directory")
    scene_dirs = sorted(path for path in scenes_root.iterdir() if path.is_dir())
    require(bool(scene_dirs), "no scene outputs found")

    records = []
    total_reasons: Counter[str] = Counter()
    positive_controls = 0
    positive_control_successes = 0

    for scene_dir in scene_dirs:
        arms: dict[str, dict[str, Any]] = {}
        for arm, expected in EXPECTED_ARMS.items():
            arm_root = scene_dir / arm
            require(arm_root.is_dir(), f"missing {scene_dir.name}/{arm}")
            summary = read_json(arm_root / "summary.json")
            metric = read_metric(arm_root / "metric.csv")
            plan_paths = list(arm_root.glob("episode_*_plans.json"))
            require(len(plan_paths) == 1,
                    f"expected one plan file: {scene_dir.name}/{arm}")
            plans = read_json(plan_paths[0])
            require(summary.get("episodes") == 1,
                    f"episode count changed: {scene_dir.name}/{arm}")
            require(summary.get("server_backend") == expected["server_backend"],
                    f"server backend changed: {scene_dir.name}/{arm}")
            require(summary.get("hybrid_route") == expected["hybrid_route"],
                    f"route changed: {scene_dir.name}/{arm}")
            require(summary.get("policy_backends") == expected["policy_backends"],
                    f"policy backends changed: {scene_dir.name}/{arm}")
            require(summary.get("role_labels") ==
                    {"A": ROLE_SEQUENCE[0], "B": ROLE_SEQUENCE[1],
                     "C": ROLE_SEQUENCE[2]},
                    f"role labels changed: {scene_dir.name}/{arm}")
            require(summary.get("multigoal_contract") ==
                    "multileg_v4_role_paired_20260812",
                    f"data protocol changed: {scene_dir.name}/{arm}")
            require(summary.get("contract_valid_episodes") == 1,
                    f"data contract failed: {scene_dir.name}/{arm}")
            require(metric.get("multigoal_contract_ok") == "1",
                    f"metric contract failed: {scene_dir.name}/{arm}")
            require(json.loads(metric["role_sequence"]) == ROLE_SEQUENCE,
                    f"metric role sequence changed: {scene_dir.name}/{arm}")
            arms[arm] = {"summary": summary, "metric": metric, "plans": plans}

        reference = arms["known_c_reference"]
        certified = arms["certified"]
        require(reference["metric"]["episode"] == certified["metric"]["episode"],
                f"episode mismatch: {scene_dir.name}")
        require(reference["metric"]["seed"] == certified["metric"]["seed"],
                f"seed mismatch: {scene_dir.name}")

        prefix = {
            leg: exact_prefix(
                reference["plans"], certified["plans"], leg,
                reference["metric"], certified["metric"],
            ) for leg in ("A", "B")
        }
        lifecycle = {
            leg: certificate_lifecycle(certified["plans"][f"leg{leg}"])
            for leg in ("A", "B", "C")
        }
        for leg in ("A", "B", "C"):
            total_reasons.update(lifecycle[leg]["uncached_reasons"])
            require(lifecycle[leg]["runtime_failures"] == 0,
                    f"certificate runtime failure: {scene_dir.name}/leg{leg}")
        for leg in ("A", "B"):
            require(prefix[leg]["all_exact"],
                    f"native fallback prefix differs: {scene_dir.name}/leg{leg}: "
                    f"{prefix[leg]['fields']}")
            require(lifecycle[leg]["accepted_requests"] == 0,
                    f"false certificate accept: {scene_dir.name}/leg{leg}")
            require(lifecycle[leg]["takeovers"] == 0,
                    f"false adapter takeover: {scene_dir.name}/leg{leg}")
            require(int(certified["metric"][f"router_active_plans_{leg}"]) == 0,
                    f"false router activation: {scene_dir.name}/leg{leg}")

        eligible = (
            boolean(reference["metric"]["reached_A"])
            and boolean(reference["metric"]["reached_B"])
        )
        positive = False
        if eligible:
            positive_controls += 1
            positive = (
                lifecycle["C"]["uncached"] >= 1
                and lifecycle["C"]["accepted_uncached"] >= 1
                and lifecycle["C"]["takeovers"] >= 1
            )
            require(positive,
                    f"Revisit C positive control did not activate: {scene_dir.name}")
            if boolean(certified["metric"]["reached_C"]):
                positive_control_successes += 1

        novel_leg_executed = {
            leg: lifecycle[leg]["plans"] > 0 for leg in ("A", "B")
        }
        records.append({
            "scene": _scene_name(scene_dir),
            "episode": certified["metric"]["episode"],
            "seed": int(certified["metric"]["seed"]),
            "reference_success": {
                leg: boolean(reference["metric"][f"reached_{leg}"])
                for leg in ("A", "B", "C")
            },
            "certified_success": {
                leg: boolean(certified["metric"][f"reached_{leg}"])
                for leg in ("A", "B", "C")
            },
            "prefix_exact": prefix,
            "novel_leg_executed": novel_leg_executed,
            "certificate": lifecycle,
            "revisit_positive_control_eligible": eligible,
            "revisit_positive_control_activated": positive,
        })

    if require_positive_control:
        require(positive_controls >= 1,
                "selected scenes contain no A/B-success Revisit C control")
        require(positive_control_successes >= 1,
                "no activated Revisit C positive control completed navigation")

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "strict-v4 implementation/causal safety gate; not an SR estimate",
        "audit_ok": True,
        "run_root": str(run_root.resolve()),
        "scenes": len(records),
        "novel_legs_audited": sum(
            record["novel_leg_executed"][leg]
            for record in records for leg in ("A", "B")
        ),
        "all_novel_prefixes_exact": all(
            record["prefix_exact"][leg]["all_exact"]
            for record in records for leg in ("A", "B")
            if record["novel_leg_executed"][leg]
        ),
        "novel_certificate_accepts": sum(
            record["certificate"][leg]["accepted_requests"]
            for record in records for leg in ("A", "B")
        ),
        "novel_adapter_takeovers": sum(
            record["certificate"][leg]["takeovers"]
            for record in records for leg in ("A", "B")
        ),
        "certificate_runtime_failures": sum(
            record["certificate"][leg]["runtime_failures"]
            for record in records for leg in ("A", "B", "C")
        ),
        "revisit_positive_controls": positive_controls,
        "revisit_positive_control_successes": positive_control_successes,
        "independent_certificate_reasons": dict(sorted(total_reasons.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-no-positive-control", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_root,
        require_positive_control=not args.allow_no_positive_control,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
