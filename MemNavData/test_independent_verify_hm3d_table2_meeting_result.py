from __future__ import annotations

import json
from pathlib import Path

import pytest

from MemNavData.independent_verify_hm3d_table2_meeting_result import (
    sha256,
    verify,
)


def write_json(path: Path, payload: dict, *, sidecar: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if sidecar:
        path.with_name(path.name + ".sha256").write_text(
            f"{sha256(path)}  {path.name}\n"
        )


def fixture(tmp_path: Path):
    parent_root = tmp_path / "parent"
    parent_manifest = parent_root / "sealed_inputs/parent_manifest.json"
    parent_rows = [
        {"scene": "s0", "episode": "episode_0000"},
        {"scene": "s0", "episode": "episode_0001"},
        {"scene": "s1", "episode": "episode_0000"},
        {"scene": "s1", "episode": "episode_0001"},
    ]
    write_json(parent_manifest, {
        "schema_version": "hm3d_fresh_fullmono_parent_manifest_v1_20260820",
        "episode_count": 4,
        "episodes": parent_rows,
    }, sidecar=False)
    write_json(
        parent_root / "hm3d_fullmono_mixed_role_independent_verification.json",
        {"verified": True, "authorized": True,
         "goal_a_sources": 4, "goal_a_successes": 3},
    )
    parent_sha = sha256(parent_manifest)

    union_root = tmp_path / "union"
    source_specs = [
        ("original_v4", "independent_natural_v4_population_verification.json",
         [("s0", "episode_0000__natural_b_00"),
          ("s0", "episode_0001__natural_b_00")], 1, 1),
        ("natural_b_expansion",
         "independent_natural_b_expansion_population_verification.json",
         [("s1", "episode_0000__natural_b_04")], 1, 1),
    ]
    union_sources = []
    for name, verifier_name, candidates, successes, supported in source_specs:
        root = tmp_path / name
        ab_receipt = {
            "source_materialized_A_histories": 3,
            "navigation_outcome_selection": False,
            "query_policy_outcomes_read": False,
            "parent_manifest_sha256": parent_sha,
            "benchmark_audit": {
                "rows": [
                    {"scene": scene, "episode": episode}
                    for scene, episode in candidates
                ]
            },
        }
        write_json(root / "ab_population/population_receipt.json",
                   ab_receipt, sidecar=False)
        population = {
            "intention_to_collect_B": len(candidates),
            "supported_population": supported,
            "AB_population_receipt_sha256": sha256(
                root / "ab_population/population_receipt.json"),
        }
        write_json(root / "population/population.json", population)
        write_json(root / verifier_name, {
            "verified": True,
            "query_navigation_outcomes_read": False,
            "factual_B_rollouts": len(candidates),
            "factual_B_successes": successes,
            "supported_population": supported,
        })
        union_sources.append({
            "name": name,
            "run_root": str(root),
            "population_sha256": sha256(root / "population/population.json"),
            "supported_histories": supported,
        })
    union_population = {
        "schema_version": "hm3d_fullmono_lifelong_population_union_v1_20260830",
        "selection_reads_C_B2_C2_navigation_outcomes": False,
        "intention_to_collect_B": 3,
        "supported_population": 2,
        "source_populations": union_sources,
    }
    write_json(union_root / "population/population.json", union_population)
    write_json(union_root / "independent_population_union_verification.json", {
        "schema_version": (
            "hm3d_fullmono_lifelong_population_union_verification_v1_20260830"
        ),
        "verified": True,
        "result_blind": True,
        "leg3_query_navigation_outcomes_read": False,
        "population_sha256": sha256(union_root / "population/population.json"),
    })

    construction = tmp_path / "construction.json"
    write_json(construction, {
        "verified": True, "construction_only": True,
        "formal_policy_evaluation_authorized": True,
        "policy_outcomes_read": False, "histories": 2,
    })
    role = {
        "n": 2, "native_success": 0, "cec_success": 1,
        "native_sr": 0.0, "cec_sr": 0.5,
    }
    policy = tmp_path / "policy.json"
    write_json(policy, {
        "schema_version": (
            "hm3d_table2_leg3_navdp_pair_verification_v1_20260829"
        ),
        "verified": True, "authorized": True, "dataset": "HM3D_TABLE2",
        "construction_verification_sha256": sha256(construction),
        "histories": 2, "scene_clusters": 2,
        "unconditional_three_leg_joint_sr_reported": False,
        "factual_prefix_waterfall": {"leg3_constructible_histories": 2},
        "recomputed": {"novel": role, "revisit": role, "all": role},
    })
    return parent_manifest, union_root, construction, policy


def test_builds_complete_factual_waterfall(tmp_path: Path) -> None:
    parent, union, construction, policy = fixture(tmp_path)
    result = verify(
        parent_manifest_path=parent,
        source_union_root=union,
        construction_verification_path=construction,
        policy_verification_path=policy,
    )
    assert result["verified"] is True
    assert result["leg1_novel"]["successes"] == 3
    assert result["leg1_novel"]["attempts"] == 4
    assert result["leg2_novel"]["eligible_materialized_A_histories"] == 3
    assert result["leg2_novel"]["candidate_covered_unique_A_histories"] == 3
    assert result["leg2_novel"]["successes"] == 2
    assert result["leg2_novel"]["attempts"] == 3
    assert result["leg2_novel"]["supported_AB_prefixes"] == 2
    assert result["leg3_conditional"]["histories"] == 2
    assert result["reporting_boundary"][
        "unconditional_three_leg_joint_sr_reported"
    ] is False


def test_accepts_absolute_path_sha256_sidecars(tmp_path: Path) -> None:
    parent, union, construction, policy = fixture(tmp_path)
    sidecar = policy.with_name(policy.name + ".sha256")
    sidecar.write_text(f"{sha256(policy)}  {policy.resolve()}\n")
    result = verify(
        parent_manifest_path=parent,
        source_union_root=union,
        construction_verification_path=construction,
        policy_verification_path=policy,
    )
    assert result["verified"] is True


def test_accepts_scene_keyed_parent_episode_manifest(tmp_path: Path) -> None:
    parent, union, construction, policy = fixture(tmp_path)
    payload = json.loads(parent.read_text())
    grouped: dict[str, list[dict]] = {}
    for row in payload["episodes"]:
        grouped.setdefault(row["scene"], []).append({"episode": row["episode"]})
    payload["episodes"] = grouped
    write_json(parent, payload, sidecar=False)
    parent_sha = sha256(parent)
    for source_name in ("original_v4", "natural_b_expansion"):
        receipt = tmp_path / source_name / "ab_population/population_receipt.json"
        source = json.loads(receipt.read_text())
        source["parent_manifest_sha256"] = parent_sha
        write_json(receipt, source, sidecar=False)
        population = tmp_path / source_name / "population/population.json"
        pop = json.loads(population.read_text())
        pop["AB_population_receipt_sha256"] = sha256(receipt)
        write_json(population, pop)
        union_population = union / "population/population.json"
        merged = json.loads(union_population.read_text())
        for row in merged["source_populations"]:
            if row["name"] == source_name:
                row["population_sha256"] = sha256(population)
        write_json(union_population, merged)
    union_population = union / "population/population.json"
    union_verification = union / "independent_population_union_verification.json"
    verification = json.loads(union_verification.read_text())
    verification["population_sha256"] = sha256(union_population)
    write_json(union_verification, verification)
    result = verify(
        parent_manifest_path=parent,
        source_union_root=union,
        construction_verification_path=construction,
        policy_verification_path=policy,
    )
    assert result["leg1_novel"]["attempts"] == 4


def test_rejects_b_candidate_outside_goal_a(tmp_path: Path) -> None:
    parent, union, construction, policy = fixture(tmp_path)
    source = tmp_path / "natural_b_expansion"
    receipt = source / "ab_population/population_receipt.json"
    payload = json.loads(receipt.read_text())
    payload["benchmark_audit"]["rows"][0]["episode"] = "episode_9999__natural_b_04"
    write_json(receipt, payload, sidecar=False)
    population = source / "population/population.json"
    pop = json.loads(population.read_text())
    pop["AB_population_receipt_sha256"] = sha256(receipt)
    write_json(population, pop)
    union_population = union / "population/population.json"
    merged = json.loads(union_population.read_text())
    for row in merged["source_populations"]:
        if row["name"] == "natural_b_expansion":
            row["population_sha256"] = sha256(population)
    write_json(union_population, merged)
    union_verification = union / "independent_population_union_verification.json"
    verification = json.loads(union_verification.read_text())
    verification["population_sha256"] = sha256(union_population)
    write_json(union_verification, verification)
    with pytest.raises(RuntimeError, match="escaped Goal A"):
        verify(
            parent_manifest_path=parent,
            source_union_root=union,
            construction_verification_path=construction,
            policy_verification_path=policy,
        )


def test_reports_candidate_coverage_below_materialized_a(tmp_path: Path) -> None:
    parent, union, construction, policy = fixture(tmp_path)
    source = tmp_path / "natural_b_expansion"
    receipt = source / "ab_population/population_receipt.json"
    payload = json.loads(receipt.read_text())
    payload["benchmark_audit"]["rows"][0] = {
        "scene": "s0", "episode": "episode_0000__natural_b_04"
    }
    write_json(receipt, payload, sidecar=False)
    population = source / "population/population.json"
    pop = json.loads(population.read_text())
    pop["AB_population_receipt_sha256"] = sha256(receipt)
    write_json(population, pop)
    union_population = union / "population/population.json"
    merged = json.loads(union_population.read_text())
    for row in merged["source_populations"]:
        if row["name"] == "natural_b_expansion":
            row["population_sha256"] = sha256(population)
    write_json(union_population, merged)
    union_verification = union / "independent_population_union_verification.json"
    verification = json.loads(union_verification.read_text())
    verification["population_sha256"] = sha256(union_population)
    write_json(union_verification, verification)
    result = verify(
        parent_manifest_path=parent,
        source_union_root=union,
        construction_verification_path=construction,
        policy_verification_path=policy,
    )
    assert result["leg2_novel"]["eligible_materialized_A_histories"] == 3
    assert result["leg2_novel"]["candidate_covered_unique_A_histories"] == 2
