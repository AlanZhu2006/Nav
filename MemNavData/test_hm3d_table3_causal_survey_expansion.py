from __future__ import annotations

import hashlib
import json
from pathlib import Path

import finalize_hm3d_table3_causal_survey_merged_population as merged_finalizer
from finalize_hm3d_table3_causal_survey_merged_population import select_powered
from freeze_hm3d_table3_causal_survey_expansion_plan import (
    candidate_identity,
    freeze,
    sha256_file,
)
from independent_verify_hm3d_table3_causal_survey_expansion_plan import verify
from independent_verify_hm3d_table3_causal_survey_merged_population import (
    independent_select,
)
import independent_verify_hm3d_table3_causal_survey_merged_population as merged_verifier


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PROTOCOL = ROOT / "MemNavData" / (
    "hm3d_table3_causal_survey_expansion_selection_protocol_20260831.json"
)
CONSTRUCTOR = ROOT / "MemNavData" / (
    "construct_hm3d_table3_causal_survey_role_pair.py"
)
MERGED_FINALIZER = ROOT / "MemNavData" / (
    "finalize_hm3d_table3_causal_survey_merged_population.py"
)
MERGED_VERIFIER = ROOT / "MemNavData" / (
    "independent_verify_hm3d_table3_causal_survey_merged_population.py"
)
EXPANSION_CONSTRUCTION_SUBMITTER = ROOT / "MemNavData" / (
    "submit_hm3d_table3_causal_survey_expansion_construction_hpc.sh"
)


def write_json(path: Path, payload: dict, *, sidecar: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if sidecar:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(
            f"{digest}  {path.name}\n"
        )


def geometry(seed: int) -> dict:
    return {
        "query_start": [float(seed), 0.0, 0.0],
        "first_goal": [float(seed), 0.0, 10.0],
        "second_goal": [float(seed), 0.0, -10.0],
        "query_start_sample": seed,
        "first_goal_sample": seed + 100,
        "second_goal_sample": seed + 200,
        "first_goal_geodesic_m": 10.0,
        "second_goal_geodesic_m": 10.0,
        "goal_distance_mismatch": 0.0,
        "goal_to_goal_geodesic_m": 20.0,
        "initial_bearing_separation_deg": 180.0,
        "ranking": [0.0, 0.0, -180.0, seed, seed + 100, seed + 200],
    }


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    bins = ("0_to_20_m", "20_to_30_m", "30_to_50_m")
    parent_path = tmp_path / "parent.json"
    scenes = [f"scene_{index}" for index in range(10)]
    parent = {
        "scenes": scenes,
        "assets": {scene: {"glb_path": f"/{scene}.glb"}
                   for scene in scenes},
    }
    write_json(parent_path, parent)

    base_plan_path = tmp_path / "base_plan.json"
    episodes = []
    for index, bin_name in enumerate((bins[0], bins[0], bins[1], bins[1],
                                      bins[2], bins[2])):
        scene = scenes[index]
        row_geometry = geometry(index)
        episodes.append({
            "history_index": index,
            "bin_name": bin_name,
            "scene": scene,
            "scene_index": index,
            "candidate_identity_sha256": candidate_identity(
                scene, bin_name, row_geometry),
            "capacity_geometry": row_geometry,
            "asset": parent["assets"][scene],
        })
    write_json(base_plan_path, {"episodes": episodes})

    base_protocol_path = tmp_path / "base_protocol.json"
    base_protocol = {
        "schema_version": "hm3d_table3_causal_survey_protocol_v1_20260830",
        "source_candidate_plan": {
            "sha256": sha256_file(base_plan_path), "candidate_count": 6,
        },
        "history": {"minimum_frames": 72},
        "query_construction": {"novel_max_history_covis_exclusive": 0.10},
        "length_definition": {"bins_m": [{"name": name} for name in bins]},
        "runtime": {"arms": ["mono_native", "mono_cec"]},
        "population_gate": {
            "minimum_histories_per_bin": 2,
            "minimum_scene_clusters_per_bin": 2,
            "maximum_selected_histories_per_scene_per_bin": 2,
        },
        "guards": {
            "query_policy_outcomes_read_before_population_seal": False,
            "partial_results_prohibited": True,
        },
    }
    write_json(base_protocol_path, base_protocol)

    base_run = tmp_path / "base_run"
    for index, candidate in enumerate(episodes):
        constructed = index != 5
        fragment = {
            "schema_version": "hm3d_table3_causal_survey_fragment_v1_20260830",
            "history_index": index,
            "scene": candidate["scene"],
            "bin_name": candidate["bin_name"],
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "source_candidate_plan_sha256": sha256_file(base_plan_path),
            "protocol_sha256": sha256_file(base_protocol_path),
            "query_policy_outcomes_read": False,
            "constructed": constructed,
            "status": "constructed" if constructed else "geometry_ineligible",
        }
        if not constructed:
            fragment["reason"] = "synthetic support failure"
        write_json(base_run / "construction_fragments" / f"{index:03d}"
                   / "completion.json", fragment, sidecar=True)

    capacity_root = tmp_path / "capacity"
    fragment_ledgers = []
    capacity_rows = {
        scenes[0]: {bins[0]: [geometry(20)], bins[1]: [], bins[2]: []},
        scenes[4]: {bins[0]: [], bins[1]: [], bins[2]: [geometry(4)]},
        scenes[6]: {bins[0]: [], bins[1]: [], bins[2]: [geometry(30)]},
        scenes[7]: {bins[0]: [], bins[1]: [], bins[2]: [geometry(31)]},
        scenes[8]: {bins[0]: [], bins[1]: [], bins[2]: [geometry(32)]},
    }
    for index, (scene, triads) in enumerate(capacity_rows.items()):
        path = capacity_root / "scenes" / f"{index:02d}" / "capacity.json"
        write_json(path, {
            "scene": scene,
            "candidate_triads": triads,
            "query_policy_outcomes_read": False,
            "navigation_outcomes_read": False,
        })
        fragment_ledgers.append({
            "scene": scene, "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    summary_path = capacity_root / "formal" / "capacity_summary.json"
    write_json(summary_path, {
        "schema_version": "hm3d_table3_navmesh_capacity_summary_v1_20260830",
        "scene_fragments": fragment_ledgers,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
    })
    capacity_verify_path = (
        capacity_root / "formal" / "independent_capacity_verification.json"
    )
    write_json(capacity_verify_path, {
        "schema_version": "hm3d_table3_navmesh_capacity_verification_v1_20260830",
        "verified": True,
        "all_geometry_capacity_gates_passed": True,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
    })

    selection_path = tmp_path / "selection.json"
    selection = {
        "schema_version": "hm3d_table3_causal_survey_expansion_selection_v1_20260831",
        "frozen_at": "test",
        "scope": "test append-only expansion",
        "base": {
            "candidate_plan": str(base_plan_path.resolve()),
            "candidate_plan_sha256": sha256_file(base_plan_path),
            "candidate_count": 6,
            "construction_run_root": str(base_run.resolve()),
            "construction_protocol": str(base_protocol_path.resolve()),
            "construction_protocol_sha256": sha256_file(base_protocol_path),
        },
        "capacity_source": {
            "run_root": str(capacity_root.resolve()),
            "summary": "formal/capacity_summary.json",
            "summary_sha256": sha256_file(summary_path),
            "independent_verification": (
                "formal/independent_capacity_verification.json"),
            "independent_verification_sha256": sha256_file(
                capacity_verify_path),
        },
        "parent": {
            "manifest": str(parent_path.resolve()),
            "manifest_sha256": sha256_file(parent_path),
            "expected_scenes": len(scenes),
        },
        "selection": {
            "bin_order": list(bins),
            "global_history_index_offset": 6,
        },
        "guards": {
            "all_125_base_completion_receipts_required": True,
            "query_policy_evaluation_authorized": False,
            "threshold_relaxation": False,
            "partial_population_allowed": False,
            "fallback_completion_allowed": False,
        },
    }
    write_json(selection_path, selection)
    return selection_path, tmp_path / "expansion_plan.json", tmp_path / "construction.json"


def test_expansion_freeze_appends_only_unused_candidates_in_deficient_bins(
    tmp_path: Path,
) -> None:
    selection, plan_path, construction = make_fixture(tmp_path)
    plan, _ = freeze(
        selection_protocol_path=selection,
        out_plan=plan_path,
        out_construction_protocol=construction,
    )
    assert plan["deficient_bins"] == ["30_to_50_m"]
    assert plan["candidate_count"] == 3
    assert {row["bin_name"] for row in plan["episodes"]} == {"30_to_50_m"}
    assert [row["history_index"] for row in plan["episodes"]] == [6, 7, 8]
    assert plan["base_candidates_deleted_or_replaced"] is False
    verification = verify(
        selection_protocol_path=selection,
        plan_path=plan_path,
        construction_protocol_path=construction,
    )
    assert verification["verified"] is True
    assert verification["candidate_count"] == 3
    assert verification["query_policy_outcomes_read"] is False


def test_merge_selection_retains_base_order_and_meets_full_gate() -> None:
    base = [
        {"scene": f"base_{index:02d}", "identity": f"b{index}"}
        for index in range(12)
    ]
    expansion = [
        {"scene": f"exp_{index:02d}", "identity": f"e{index}"}
        for index in range(8)
    ]
    rows = base + expansion
    selected = select_powered(
        rows, histories=16, scenes=10, maximum_per_scene=2,
    )
    independently_selected = independent_select(
        rows, histories=16, scenes=10, maximum_per_scene=2,
    )
    assert [row["identity"] for row in selected] == [
        row["identity"] for row in independently_selected
    ]
    assert sum(row["identity"].startswith("b") for row in selected) == 12
    assert len({row["scene"] for row in selected}) == 16


def test_formal_expansion_contract_has_no_relaxation_or_deletion() -> None:
    payload = json.loads(SELECTION_PROTOCOL.read_text())
    assert payload["selection"]["delete_or_replace_base_candidates"] is False
    assert payload["selection"]["read_query_policy_outcomes"] is False
    assert payload["selection"]["read_navigation_policy_outcomes"] is False
    assert payload["guards"]["threshold_relaxation"] is False
    assert payload["guards"]["partial_population_allowed"] is False
    assert payload["guards"]["fallback_completion_allowed"] is False


def test_expansion_constructor_and_merge_gate_are_explicit() -> None:
    constructor = CONSTRUCTOR.read_text()
    finalizer = MERGED_FINALIZER.read_text()
    verifier = MERGED_VERIFIER.read_text()
    assert 'index_group.add_argument("--plan-index"' in constructor
    assert 'fragment["plan_index"] = plan_index' in constructor
    assert '"base_candidates_deleted_or_replaced": False' in finalizer
    assert '== 48' in verifier
    assert '"0_to_20_m": 16' in verifier
    assert '"20_to_30_m": 16' in verifier
    assert '"30_to_50_m": 16' in verifier


def test_expansion_execution_submits_every_verified_candidate_and_no_fallback() -> None:
    submitter = EXPANSION_CONSTRUCTION_SUBMITTER.read_text()
    assert 'array="0-$((candidate_count - 1))%4"' in submitter
    assert "verified expansion is empty; original population must be used" in submitter
    assert "--dependency='afterok:${construct_job}'" in submitter
    assert "--dependency='afterok:${finalize_job}'" in submitter
    assert "'base_candidates_deleted_or_replaced':False" in submitter
    assert "'query_policy_jobs_submitted':False" in submitter
    assert "'fallback_completion_allowed':False" in submitter
    assert "bundle_selftest.sh" in submitter
    assert "EXPECTED_SSH_USER" in submitter


def test_merged_finalizer_and_independent_verifier_preserve_all_sources(
    tmp_path: Path, monkeypatch,
) -> None:
    bins = ("0_to_20_m", "20_to_30_m", "30_to_50_m")
    common = {
        "history": {"minimum_frames": 72},
        "query_construction": {
            "novel_max_history_covis_exclusive": 0.10,
            "revisit_min_history_covis_inclusive": 0.55,
            "maximum_role_distance_mismatch_m": 2.0,
            "minimum_initial_bearing_separation_deg": 60.0,
        },
        "length_definition": {"bins_m": [
            {"name": bins[0], "lower_inclusive": 2.0, "upper": 20.0,
             "upper_inclusive": False},
            {"name": bins[1], "lower_inclusive": 20.0, "upper": 30.0,
             "upper_inclusive": False},
            {"name": bins[2], "lower_inclusive": 30.0, "upper": 50.0,
             "upper_inclusive": True},
        ]},
        "runtime": {"arms": ["mono_native", "mono_cec"]},
        "population_gate": {
            "minimum_histories_per_bin": 16,
            "minimum_scene_clusters_per_bin": 10,
            "maximum_selected_histories_per_scene_per_bin": 2,
        },
    }
    base_plan_path = tmp_path / "base_plan.json"
    base_episodes = []
    for bin_index, bin_name in enumerate(bins):
        for within in range(16):
            history_index = bin_index * 16 + within
            scene = f"scene_{bin_index}_{within:02d}"
            base_episodes.append({
                "history_index": history_index,
                "scene": scene,
                "scene_index": history_index,
                "bin_name": bin_name,
                "candidate_identity_sha256": hashlib.sha256(
                    f"{bin_name}:{scene}".encode()).hexdigest(),
            })
    write_json(base_plan_path, {"episodes": base_episodes})
    expansion_plan_path = tmp_path / "expansion_plan.json"
    write_json(expansion_plan_path, {
        "schema_version": "hm3d_table3_causal_survey_expansion_plan_v1_20260831",
        "episodes": [], "base_candidates_deleted_or_replaced": False,
        "query_policy_outcomes_read": False,
        "navigation_policy_outcomes_read": False,
    })
    base_protocol_path = tmp_path / "base_protocol.json"
    expansion_protocol_path = tmp_path / "expansion_protocol.json"
    write_json(base_protocol_path, {
        "schema_version": "hm3d_table3_causal_survey_protocol_v1_20260830",
        "source_candidate_plan": {
            "sha256": sha256_file(base_plan_path),
            "candidate_count": len(base_episodes),
        },
        **common,
    })
    write_json(expansion_protocol_path, {
        "schema_version": "hm3d_table3_causal_survey_protocol_v1_20260830",
        "source_candidate_plan": {
            "sha256": sha256_file(expansion_plan_path), "candidate_count": 0,
        },
        **common,
    })
    expansion_verification_path = tmp_path / "expansion_verification.json"
    write_json(expansion_verification_path, {
        "schema_version": (
            "hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831"),
        "verified": True,
        "plan_sha256": sha256_file(expansion_plan_path),
        "construction_protocol_sha256": sha256_file(expansion_protocol_path),
        "base_candidates_deleted_or_replaced": False,
        "query_policy_outcomes_read": False,
        "navigation_policy_outcomes_read": False,
        "query_policy_evaluation_authorized": False,
    })

    base_run, expansion_run = tmp_path / "base", tmp_path / "expansion"
    for plan_index, candidate in enumerate(base_episodes):
        source = (base_run / "role_pair_candidates" / candidate["scene"]
                  / f"survey_{plan_index:03d}")
        payload = {
            "scene": candidate["scene"],
            "episode": f"episode_table3_survey_{plan_index:03d}",
            "bin_name": candidate["bin_name"],
            "history_index": candidate["history_index"],
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        }
        write_json(source / "role_pairs.json", payload)
        fragment = {
            "schema_version": "hm3d_table3_causal_survey_fragment_v1_20260830",
            "history_index": candidate["history_index"],
            "scene": candidate["scene"],
            "bin_name": candidate["bin_name"],
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "source_candidate_plan_sha256": sha256_file(base_plan_path),
            "protocol_sha256": sha256_file(base_protocol_path),
            "query_policy_outcomes_read": False,
            "constructed": True,
            "status": "constructed",
            "role_pair_candidate": str(source.resolve()),
            "role_pairs_sha256": sha256_file(source / "role_pairs.json"),
        }
        write_json(base_run / "construction_fragments" / f"{plan_index:03d}"
                   / "completion.json", fragment, sidecar=True)

    monkeypatch.setattr(merged_finalizer, "validate_manifest", lambda value: value)
    population_root = tmp_path / "population"
    result = merged_finalizer.finalize(
        base_run_root=base_run,
        base_plan_path=base_plan_path,
        base_protocol_path=base_protocol_path,
        expansion_run_root=expansion_run,
        expansion_plan_path=expansion_plan_path,
        expansion_protocol_path=expansion_protocol_path,
        expansion_verification_path=expansion_verification_path,
        out=population_root,
    )
    assert result["histories"] == 48
    assert result["base_candidates_preserved"] == 48
    assert result["base_candidates_deleted_or_replaced"] is False

    manifest = json.loads(
        (population_root / "role_pairs" / "manifest.json").read_text()
    )
    manifest_sha = sha256_file(population_root / "role_pairs" / "manifest.json")
    monkeypatch.setattr(merged_verifier, "audit", lambda _root: {
        "ok": True,
        "query_policy_outcomes_read": False,
        "online_history": "controlled_causal_rgb_geodesic_survey",
        "manifest_sha256": manifest_sha,
        "histories_by_bin": {name: 16 for name in bins},
        "scene_clusters_by_bin": {name: 16 for name in bins},
    })
    verification = merged_verifier.verify(
        population_root=population_root,
        base_run_root=base_run,
        base_plan_path=base_plan_path,
        base_protocol_path=base_protocol_path,
        expansion_run_root=expansion_run,
        expansion_plan_path=expansion_plan_path,
        expansion_protocol_path=expansion_protocol_path,
        expansion_plan_verification_path=expansion_verification_path,
    )
    assert len(manifest["episodes"]) == 48
    assert verification["verified"] is True
    assert verification["histories_by_bin"] == {name: 16 for name in bins}
    assert verification["base_candidates_preserved"] == 48
