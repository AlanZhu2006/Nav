#!/usr/bin/env python3
"""Build the conference Table-II waterfall from independently sealed inputs.

The policy verifier intentionally reports only the Leg-3 treatment estimand.
This second, post-seal verifier joins it to the factual Goal-A and Goal-B
receipts without pretending that the result-blind B-candidate expansion is an
unconditional three-leg joint rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_table2_meeting_result_verification_v1_20260830"
POLICY_SCHEMA = "hm3d_table2_leg3_navdp_pair_verification_v1_20260829"
UNION_SCHEMA = "hm3d_fullmono_lifelong_population_union_v1_20260830"
UNION_VERIFY_SCHEMA = (
    "hm3d_fullmono_lifelong_population_union_verification_v1_20260830"
)
SOURCE_VERIFIERS = {
    "original_v4": "independent_natural_v4_population_verification.json",
    "natural_b_expansion": (
        "independent_natural_b_expansion_population_verification.json"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing input: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA sidecar: {sidecar}")
    digest = sha256(path)
    fields = sidecar.read_text().split()
    # GNU sha256sum records exactly the spelling passed by the producer.  Our
    # sealed artifacts legitimately contain both ``file.json`` and an
    # absolute ``/run/root/file.json`` in the second column.  Bind the digest
    # and basename while rejecting a sidecar for any other artifact.
    require(
        len(fields) == 2
        and fields[0] == digest
        and Path(fields[1]).name == path.name,
        f"invalid SHA sidecar: {sidecar}",
    )
    return json.loads(path.read_text()), digest


def base_episode(value: str) -> str:
    marker = "__natural_b_"
    return value.split(marker, 1)[0] if marker in value else value


def verify(
    *,
    parent_manifest_path: Path,
    source_union_root: Path,
    construction_verification_path: Path,
    policy_verification_path: Path,
) -> dict[str, Any]:
    parent_manifest = json.loads(parent_manifest_path.read_text())
    require(
        parent_manifest.get("schema_version")
        == "hm3d_fresh_fullmono_parent_manifest_v1_20260820",
        "Goal-A parent manifest schema changed",
    )
    parent_root = parent_manifest_path.parent.parent
    parent_verification_path = (
        parent_root / "hm3d_fullmono_mixed_role_independent_verification.json"
    )
    parent_verification, parent_verification_sha = load_verified(
        parent_verification_path
    )
    require(
        parent_verification.get("verified") is True
        and parent_verification.get("authorized") is True,
        "Goal-A independent verifier did not authorize the source",
    )
    parent_episodes = {
        (str(row["scene"]), str(row["episode"]))
        for row in parent_manifest["episodes"]
    }
    goal_a_attempts = int(parent_verification["goal_a_sources"])
    goal_a_successes = int(parent_verification["goal_a_successes"])
    require(
        goal_a_attempts == len(parent_episodes)
        == int(parent_manifest["episode_count"]),
        "Goal-A denominator changed",
    )
    require(0 < goal_a_successes <= goal_a_attempts, "bad Goal-A successes")

    union_population_path = source_union_root / "population/population.json"
    union_population, union_population_sha = load_verified(union_population_path)
    require(
        union_population.get("schema_version") == UNION_SCHEMA
        and union_population.get("selection_reads_C_B2_C2_navigation_outcomes")
        is False,
        "source union is not the result-blind Table-II population",
    )
    union_verification_path = (
        source_union_root / "independent_population_union_verification.json"
    )
    union_verification, union_verification_sha = load_verified(
        union_verification_path
    )
    require(
        union_verification.get("schema_version") == UNION_VERIFY_SCHEMA
        and union_verification.get("verified") is True
        and union_verification.get("result_blind") is True
        and union_verification.get("leg3_query_navigation_outcomes_read")
        is False,
        "population-union independent verification failed",
    )
    require(
        union_verification.get("population_sha256") == union_population_sha,
        "union verifier and population hash differ",
    )

    source_rows = union_population.get("source_populations")
    require(
        isinstance(source_rows, list)
        and {str(row["name"]) for row in source_rows}
        == set(SOURCE_VERIFIERS),
        "Table-II source populations changed",
    )
    factual_b_rollouts = 0
    factual_b_successes = 0
    supported_prefixes = 0
    materialized_a_counts: set[int] = set()
    b_candidate_a_identities: set[tuple[str, str]] = set()
    source_breakdown: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    parent_manifest_sha = sha256(parent_manifest_path)
    for source in source_rows:
        name = str(source["name"])
        root = Path(source["run_root"])
        source_population_path = root / "population/population.json"
        source_population, source_population_sha = load_verified(
            source_population_path
        )
        require(
            source_population_sha == source["population_sha256"]
            and int(source_population["supported_population"])
            == int(source["supported_histories"]),
            f"{name}: population binding changed",
        )
        source_verification_path = root / SOURCE_VERIFIERS[name]
        source_verification, source_verification_sha = load_verified(
            source_verification_path
        )
        require(
            source_verification.get("verified") is True
            and source_verification.get("query_navigation_outcomes_read")
            is False,
            f"{name}: factual-B independent verifier failed",
        )
        b_rollouts = int(source_verification["factual_B_rollouts"])
        b_successes = int(source_verification["factual_B_successes"])
        source_supported = int(source_verification["supported_population"])
        require(
            b_rollouts == int(source_population["intention_to_collect_B"])
            and source_supported == int(source_population["supported_population"])
            and 0 <= source_supported <= b_successes <= b_rollouts,
            f"{name}: factual-B waterfall changed",
        )
        ab_receipt_path = root / "ab_population/population_receipt.json"
        require(ab_receipt_path.is_file(), f"{name}: AB receipt missing")
        require(
            sha256(ab_receipt_path)
            == str(source_population["AB_population_receipt_sha256"]),
            f"{name}: AB receipt hash changed",
        )
        ab_receipt = json.loads(ab_receipt_path.read_text())
        require(
            ab_receipt.get("navigation_outcome_selection") is False
            and ab_receipt.get("query_policy_outcomes_read") is False
            and ab_receipt.get("parent_manifest_sha256") == parent_manifest_sha,
            f"{name}: AB source provenance changed",
        )
        materialized_a_counts.add(int(ab_receipt["source_materialized_A_histories"]))
        candidate_rows = ab_receipt["benchmark_audit"]["rows"]
        require(len(candidate_rows) == b_rollouts, f"{name}: B candidate rows changed")
        for row in candidate_rows:
            identity = (str(row["scene"]), base_episode(str(row["episode"])))
            require(identity in parent_episodes, f"{name}: B candidate escaped Goal A")
            b_candidate_a_identities.add(identity)
        factual_b_rollouts += b_rollouts
        factual_b_successes += b_successes
        supported_prefixes += source_supported
        source_hashes[name] = source_verification_sha
        source_breakdown[name] = {
            "factual_B_rollouts": b_rollouts,
            "factual_B_successes": b_successes,
            "supported_AB_prefixes": source_supported,
            "population_sha256": source_population_sha,
            "independent_verification_sha256": source_verification_sha,
        }

    require(len(materialized_a_counts) == 1, "source A population changed")
    eligible_a_histories = materialized_a_counts.pop()
    require(
        len(b_candidate_a_identities) == eligible_a_histories
        and eligible_a_histories <= goal_a_successes,
        "B candidates do not reproduce the eligible successful-A set",
    )
    require(
        factual_b_rollouts == int(union_population["intention_to_collect_B"])
        and supported_prefixes == int(union_population["supported_population"]),
        "source waterfall does not reproduce the union",
    )

    construction, construction_sha = load_verified(
        construction_verification_path
    )
    require(
        construction.get("verified") is True
        and construction.get("construction_only") is True
        and construction.get("formal_policy_evaluation_authorized") is True
        and construction.get("policy_outcomes_read") is False,
        "Leg-3 construction verifier did not pass",
    )
    policy, policy_sha = load_verified(policy_verification_path)
    require(
        policy.get("schema_version") == POLICY_SCHEMA
        and policy.get("verified") is True
        and policy.get("authorized") is True
        and policy.get("dataset") == "HM3D_TABLE2",
        "final policy verification did not pass",
    )
    require(
        policy.get("construction_verification_sha256") == construction_sha
        and int(policy["histories"]) == int(construction["histories"]),
        "policy result is not bound to the verified Leg-3 population",
    )
    require(
        policy.get("unconditional_three_leg_joint_sr_reported") is False,
        "policy result incorrectly reports unconditional joint SR",
    )
    recomputed = policy.get("recomputed")
    require(
        isinstance(recomputed, dict)
        and set(recomputed) == {"novel", "revisit", "all"},
        "policy verifier lacks role-stratified raw recount",
    )
    leg3_histories = int(policy["histories"])
    require(
        leg3_histories == int(policy["factual_prefix_waterfall"]
                              ["leg3_constructible_histories"])
        and leg3_histories <= supported_prefixes,
        "Leg-3 constructible denominator changed",
    )

    return {
        "schema_version": SCHEMA,
        "verified": True,
        "scope": "conference Table-II factual-prefix waterfall and conditional Leg-3 effect",
        "runtime_role_visibility": "none",
        "methods_share_factual_A_and_B_prefixes": True,
        "leg1_novel": {
            "denominator": "executed actual-mono Goal-A source rollouts",
            "attempts": goal_a_attempts,
            "successes": goal_a_successes,
            "sr": goal_a_successes / goal_a_attempts,
            "controller_depth": "causal_monocular",
        },
        "leg2_novel": {
            "denominator": "result-blind factual-B candidate rollouts from successful Goal-A prefixes",
            "eligible_unique_A_histories": eligible_a_histories,
            "attempts": factual_b_rollouts,
            "successes": factual_b_successes,
            "sr": factual_b_successes / factual_b_rollouts,
            "repeated_A_prefixes_across_distinct_B_candidates": True,
            "supported_AB_prefixes": supported_prefixes,
            "source_breakdown": source_breakdown,
        },
        "leg3_conditional": {
            "estimand": "C_given_successful_supported_factual_A_and_B",
            "histories": leg3_histories,
            "scene_clusters": int(policy["scene_clusters"]),
            "novel": recomputed["novel"],
            "revisit": recomputed["revisit"],
            "balanced_all": recomputed["all"],
        },
        "reporting_boundary": {
            "unconditional_three_leg_joint_sr_reported": False,
            "selected_prefix_joint_equals_conditional_C": True,
            "reason": (
                "The powered B stage contains multiple result-blind B candidates "
                "for some successful A prefixes; multiplying stage rates would "
                "not define one intention-to-treat three-leg cohort."
            ),
        },
        "receipts": {
            "parent_manifest_sha256": parent_manifest_sha,
            "goal_A_independent_verification_sha256": parent_verification_sha,
            "source_population_union_sha256": union_population_sha,
            "source_population_union_verification_sha256": union_verification_sha,
            "source_factual_B_verification_sha256": source_hashes,
            "leg3_construction_verification_sha256": construction_sha,
            "leg3_policy_independent_verification_sha256": policy_sha,
        },
        "fallback_completion_used": False,
        "threshold_relaxation_used": False,
        "partial_policy_outcomes_used": False,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--source-union-root", type=Path, required=True)
    parser.add_argument("--construction-verification", type=Path, required=True)
    parser.add_argument("--policy-verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(
        parent_manifest_path=args.parent_manifest.resolve(),
        source_union_root=args.source_union_root.resolve(),
        construction_verification_path=args.construction_verification.resolve(),
        policy_verification_path=args.policy_verification.resolve(),
    )
    write_exclusive(args.out.resolve(), payload)
    print(json.dumps({
        "verified": True,
        "goal_A": [payload["leg1_novel"]["successes"],
                   payload["leg1_novel"]["attempts"]],
        "goal_B": [payload["leg2_novel"]["successes"],
                   payload["leg2_novel"]["attempts"]],
        "leg3_histories": payload["leg3_conditional"]["histories"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
