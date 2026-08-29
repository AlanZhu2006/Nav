import json

import pytest

from MemNavData.audit_hm3d_table2_leg3_constructibility import (
    FRAGMENT_SCHEMA,
    NOVEL_ATTRITION,
    POPULATION_SCHEMA,
    REVISIT_ATTRITION,
    STRATA,
    VERIFICATION_SCHEMA,
    audit,
    sha256_file,
)
from MemNavData.independent_verify_hm3d_table2_leg3_constructibility_audit import (
    verify,
)


def _diagnostic(attempts, support, local):
    return {
        "attempts": attempts,
        "deterministic_local_proposals": 63,
        "deterministic_local_attempts": local,
        "uniform_random_attempts": attempts - local,
        "duplicate_position_rejects": 0,
        "floor_or_clearance_rejects": attempts - support,
        "non_navigable_rejects": 0,
        "floor_mismatch_rejects": attempts - support,
        "clearance_rejects": 0,
        "candidate_separation_rejects": 0,
        "geodesic_rejects": 0,
        "unreachable_rejects": 0,
        "a_to_b_outside_band_rejects": 0,
        "direction_stratum_rejects": 0,
        "paired_separation_rejects": 0,
        "paired_unreachable_rejects": 0,
        "paired_below_minimum_rejects": 0,
        "paired_above_maximum_rejects": 0,
        "support_rejects": support,
    }


def _fixture(tmp_path):
    paths = []
    rows = []
    support_totals = {"front": 881, "side": 2847, "rear": 2932}
    diag_rank = 0
    for index in range(22):
        eligible = index < 8
        reason = None
        natural = {}
        revisit = {
            "grid_attempts": 100,
            "fully_scored": 20,
            "support_rejects": 10,
        }
        selected_stratum = "front" if eligible else None
        if 8 <= index <= 20:
            reason = NOVEL_ATTRITION
            for stratum in STRATA:
                support = support_totals[stratum] if index == 8 else 0
                local = 86 if diag_rank == 0 else 60
                diag_rank += 1
                natural[stratum] = _diagnostic(5000, support, local)
        elif index == 21:
            reason = REVISIT_ATTRITION
            revisit = {
                "grid_attempts": 3456,
                "fully_scored": 1256,
                "support_rejects": 1256,
            }
        else:
            natural["front"] = _diagnostic(1, 0, 1)
        row = {
            "schema_version": FRAGMENT_SCHEMA,
            "status": "complete",
            "population_index": index,
            "scene": f"scene_{index % 6 if eligible else index}",
            "episode": f"episode_{index:04d}",
            "protocol_sha256": "a" * 64,
            "leg3_query_policy_outcomes_read": False,
            "old_goal_C_navigation_outcomes_read": False,
            "eligible": eligible,
            "attrition_reason": reason,
            "selected_stratum": selected_stratum,
            "selected_revisit_segment": "A" if eligible else None,
            "combined_prefix_steps": 200 + index,
            "natural_diagnostics": natural,
            "revisit_diagnostics": revisit,
        }
        path = tmp_path / f"completion_{index:03d}.json"
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
        paths.append(path)
        rows.append(row)

    population = {
        "schema_version": POPULATION_SCHEMA,
        "navigation_outcomes_generated": False,
        "query_outcomes_read_for_selection": False,
        "old_goal_C_outcomes_read_for_construction": False,
        "factual_AB_successful_prefixes": 22,
        "factual_AB_scene_clusters": 15,
        "leg3_constructible_histories": 8,
        "leg3_scene_clusters": 6,
        "formal_policy_evaluation_authorized": False,
        "power_gate": {
            "target_met": False,
            "direction_strata": {"front": 4, "side": 0, "rear": 4},
        },
        "construction_inputs": [
            {
                "population_index": index,
                "completion_sha256": sha256_file(path),
                "eligible": rows[index]["eligible"],
            }
            for index, path in enumerate(paths)
        ],
    }
    construction = {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
    }
    return paths, rows, population, construction


def test_audit_reconstructs_the_exhaustive_support_bottleneck(tmp_path):
    paths, rows, population, construction = _fixture(tmp_path)
    result = audit(
        list(zip(paths, rows)), population, construction,
        source_uri="sealed://fixture",
    )
    novel = result["novel_attrition_audit"]
    assert novel["attempts"] == 195_000
    assert novel["deterministic_local_attempts"] == 2366
    assert novel["deterministic_seeded_uniform_attempts"] == 192_634
    assert novel["candidates_reaching_final_support_check"] == 6660
    assert novel["by_direction_stratum"]["side"][
        "candidates_reaching_final_support_check"] == 2847
    conclusion = result["audit_conclusion"]
    assert conclusion["simple_attempt_budget_failure"] is False
    assert conclusion["side_direction_sampler_failure"] is False
    assert conclusion["binary_role_pair_constructibility_failure"] is True

    verified = verify(paths, population, construction, result)
    assert verified["verified"] is True
    assert verified["recomputed"]["final_support_checks"] == 6660


def test_audit_rejects_a_completion_changed_after_population_seal(tmp_path):
    paths, rows, population, construction = _fixture(tmp_path)
    changed = dict(rows[0])
    changed["combined_prefix_steps"] += 1
    paths[0].write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        audit(
            [(path, json.loads(path.read_text())) for path in paths],
            population,
            construction,
            source_uri="sealed://fixture",
        )
