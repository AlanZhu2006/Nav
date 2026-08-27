#!/usr/bin/env python3
"""Seal the first-decision CEC-accepted subset of a consumed role-pair run.

The selector reads no navigation success, final distance or metric CSV.  It
uses only the deployment-visible first certificate decision.  Analysis roles
are retained exclusively in the separate audit receipt and never enter the
runtime query manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from MemNavData.cec_handoff_contract import public_cec_proof
from MemNavData.controller_portability_contract import (
    CEC_POINTGOAL_UNITS,
    cec_proof_sha256,
)


MANIFEST_SCHEMA = "cec_first_decision_accepted_population_v1_20260827"
AUDIT_SCHEMA = "cec_first_decision_accepted_population_audit_v1_20260827"
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "analysis_role", "role", "goal_role", "query_role", "is_revisit",
    "is_novel", "oracle_pose", "gt_pose", "ground_truth_pose",
    "habitat_pose",
})
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_anchor(plan: dict[str, Any]) -> int:
    value = plan.get("router_selected_anchor", plan.get("anchor"))
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "accepted first decision lacks an integer anchor")
    return value


def _direction(plan: dict[str, Any]) -> list[float]:
    value = plan.get("memory_unbounded_pointgoal", plan.get("aux_pose"))
    require(isinstance(value, list) and len(value) == 2,
            "accepted first decision lacks a scale-free direction")
    direction = [float(value[0]), float(value[1])]
    require(all(abs(item) < float("inf") for item in direction)
            and abs(direction[0]) + abs(direction[1]) > 1e-9,
            "accepted first direction is invalid")
    return direction


def _proof(plan: dict[str, Any], anchor_sha256: str) -> dict[str, Any]:
    certificate = plan.get("certified_relocalization_certificate")
    require(isinstance(certificate, dict), "certificate receipt is missing")
    proof = {
        "certified_relocalization_schema_version": certificate.get(
            "schema_version"),
        "frame_idx": plan.get("frame_idx"),
        "ok": plan.get("certified_relocalization_ok"),
        "accepted": plan.get("certified_relocalization_accepted"),
        "reason": plan.get("certified_relocalization_reason"),
        "selected_anchor": _selected_anchor(plan),
        "selected_anchor_image_sha256": anchor_sha256,
        "direction_vector": _direction(plan),
        "pointgoal_units": plan.get(
            "certified_relocalization_pointgoal_units"),
        "certificate": certificate,
    }
    require(proof["pointgoal_units"] == CEC_POINTGOAL_UNITS,
            "first proof pointgoal units changed")
    return public_cec_proof(proof)


def build(
    run_root: Path,
    *,
    expected_queries: int,
    expected_accepted: int,
    expected_accepted_scenes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = run_root.resolve()
    benchmark_root = run_root / "benchmarks/natural_direction"
    manifest_path = benchmark_root / "manifest.json"
    require(manifest_path.is_file(), "source benchmark manifest is missing")
    benchmark = json.loads(manifest_path.read_text())
    episodes = benchmark.get("episodes")
    require(isinstance(episodes, list) and episodes,
            "source benchmark population is empty")

    selected = []
    decision_counts = {"accepted": 0, "rejected": 0}
    role_counts = {
        "accepted": {"novel": 0, "revisit": 0},
        "rejected": {"novel": 0, "revisit": 0},
    }
    total = 0
    for history_index, item in enumerate(episodes):
        scene = str(item["scene"])
        episode = str(item["episode"])
        source_episode = Path(item["online_a_episode"])
        trace_path = source_episode / "online_a_trace.json"
        require(trace_path.is_file(), "online-A trace is missing")
        require(sha256_file(trace_path) == item["online_a_trace_sha256"],
                "online-A trace binding changed")
        role_pair_path = benchmark_root / scene / episode / "role_pairs.json"
        require(role_pair_path.is_file(), "role-pair benchmark is missing")
        role_pair = json.loads(role_pair_path.read_text())
        queries = [query for pair in role_pair["pairs"]
                   for query in pair["queries"]]
        label = f"{history_index:03d}_{scene}_{episode}"
        result_root = (run_root / "evaluation/natural_direction" / label
                       / "mono_cec")
        for query in queries:
            total += 1
            query_id = str(query["query_id"])
            role = str(query["analysis_role"])
            require(role in {"novel", "revisit"}, "unknown audit role")
            plan_path = result_root / f"{episode}_{query_id}_plans.json"
            require(plan_path.is_file(), f"missing CEC plan {plan_path}")
            payload = json.loads(plan_path.read_text())
            runtime_fields = set(payload.get("query_runtime_fields", []))
            require(not runtime_fields.intersection(FORBIDDEN_RUNTIME_FIELDS),
                    "analysis role or oracle field reached the policy")
            plans = payload.get("query_leg")
            require(isinstance(plans, list) and plans,
                    "query has no CEC decisions")
            first = plans[0]
            require(first.get("certified_relocalization_cached") is False,
                    "first certificate decision was unexpectedly cached")
            accepted = first.get("certified_relocalization_accepted") is True
            require(all((plan.get("certified_relocalization_accepted") is True)
                        == accepted for plan in plans),
                    "certificate decision changed after the first decision")
            state = "accepted" if accepted else "rejected"
            decision_counts[state] += 1
            role_counts[state][role] += 1
            if not accepted:
                continue

            anchor = _selected_anchor(first)
            anchor_path = source_episode / "rgb" / f"{anchor:06d}.jpg"
            require(anchor_path.is_file(), "certified history anchor is missing")
            anchor_sha = sha256_file(anchor_path)
            proof = _proof(first, anchor_sha)
            goal_path = benchmark_root / scene / episode / query["goal_rgb"]
            require(goal_path.is_file()
                    and sha256_file(goal_path) == query["goal_rgb_sha256"],
                    "accepted query goal binding changed")
            selected.append({
                "history_index": history_index,
                "scene": scene,
                "episode": episode,
                "pair_id": str(query_id).rsplit("_", 1)[0],
                "query_id": query_id,
                "goal_rgb_sha256": str(query["goal_rgb_sha256"]),
                "causal_history_sha256": str(item["online_a_trace_sha256"]),
                "first_decision_frame_idx": int(first["frame_idx"]),
                "first_proof_sha256": cec_proof_sha256(proof),
                "public_proof": proof,
                "selected_anchor": anchor,
                "selected_anchor_image_sha256": anchor_sha,
                "anchor_jpeg_path": str(anchor_path.resolve()),
                "direction_vector": proof["direction_vector"],
                "source_plan_sha256": sha256_file(plan_path),
            })

    require(total == expected_queries,
            f"query count {total} != frozen {expected_queries}")
    require(len(selected) == expected_accepted,
            f"accepted count {len(selected)} != frozen {expected_accepted}")
    accepted_scenes = len({entry["scene"] for entry in selected})
    require(accepted_scenes == expected_accepted_scenes,
            "accepted scene-cluster count changed")
    identities = {(entry["scene"], entry["episode"], entry["query_id"])
                  for entry in selected}
    require(len(identities) == len(selected), "accepted query identity duplicated")

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "scope": "consumed_external_scene_controller_portability_ablation",
        "selection_rule": "first_deployment_visible_cec_decision_accepted",
        "source_benchmark_manifest_sha256": sha256_file(manifest_path),
        "source_navigation_outcomes_read": False,
        "runtime_role_labels_present": False,
        "queries": selected,
    }
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "verified": True,
        "source_run_root": str(run_root),
        "source_benchmark_manifest_sha256": sha256_file(manifest_path),
        "source_plan_receipts_only": True,
        "navigation_metric_files_read": False,
        "query_count": total,
        "decision_counts": decision_counts,
        "accepted_scene_clusters": accepted_scenes,
        "analysis_role_counts_not_in_runtime_manifest": role_counts,
        "accepted_population_equals_revisit_posthoc": (
            role_counts["accepted"] == {"novel": 0, "revisit": len(selected)}),
        "manifest_sha256": hashlib.sha256(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
    }
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=56)
    parser.add_argument("--expected-accepted", type=int, default=28)
    parser.add_argument("--expected-accepted-scenes", type=int, default=21)
    args = parser.parse_args()
    manifest, audit = build(
        args.run_root,
        expected_queries=args.expected_queries,
        expected_accepted=args.expected_accepted,
        expected_accepted_scenes=args.expected_accepted_scenes,
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(manifest_text)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verified": True,
        "queries": audit["query_count"],
        "accepted": audit["decision_counts"]["accepted"],
        "accepted_scenes": audit["accepted_scene_clusters"],
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
