import json
from pathlib import Path

import pytest

from MemNavData.aggregate_hm3d_table1_navdp_pair import aggregate, digest
from MemNavData.independent_verify_hm3d_table1_navdp_pair import verify
from MemNavData.test_hm3d_table1_navdp_pair import make_fixture, write_json


def make_table2_fixture(tmp_path):
    run, benchmark, construction_path = make_fixture(tmp_path)
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = item["scene"], item["episode"]
        prefix = tmp_path / "prefix" / scene / episode
        receipt = {
            "prefix_receipt_schema": (
                "hm3d_table2_actual_mono_ab_prefix_v1_20260829"
            ),
            "prefix_semantics": "actual_mono_Novel_A_then_Novel_B",
            "prefix_A_steps": 2,
            "prefix_B_steps": 3,
        }
        trace = {
            "poses": [{"step": value} for value in range(5)],
            "prefix_semantics": (
                "exact_actual_mono_A_then_B_observation_concat"
            ),
        }
        write_json(prefix / "receipt.json", receipt)
        write_json(prefix / "online_a_trace.json", trace)
        item["online_a_episode"] = str(prefix.resolve())
        item["table2_selected_revisit_segment"] = "A" if index == 0 else "B"
        root = (
            run / "evaluation/natural_direction"
            / f"{index:03d}_{scene}_{episode}"
        )
        completion_path = root / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion.update({
            "schema_version": "hm3d_table2_leg3_history_v1_20260829",
            "history_contract": "actual_ab",
            "shared_history_policy": (
                "actual_mono_navdp_novel_A_then_novel_B_rgb_replay"
            ),
            "prefix_A_steps": 2,
            "prefix_B_steps": 3,
            "online_a_steps": 5,
        })
        write_json(completion_path, completion)
        (root / "completion.json.sha256").write_text(
            digest(completion_path) + "  completion.json\n"
        )
    write_json(manifest_path, manifest)
    manifest_sha = digest(manifest_path)
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = item["scene"], item["episode"]
        root = (
            run / "evaluation/natural_direction"
            / f"{index:03d}_{scene}_{episode}"
        )
        completion_path = root / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["benchmark_manifest_sha256"] = manifest_sha
        write_json(completion_path, completion)
        (root / "completion.json.sha256").write_text(
            digest(completion_path) + "  completion.json\n"
        )
    population_path = benchmark.parent / "population_receipt.json"
    write_json(population_path, {
        "source_A_attempts": 99,
        "factual_AB_successful_prefixes": 22,
        "factual_AB_scene_clusters": 15,
        "leg3_constructible_histories": 2,
        "leg3_scene_clusters": 2,
    })
    construction = json.loads(construction_path.read_text())
    construction.update({
        "schema_version": (
            "hm3d_table2_leg3_mixed_role_construction_"
            "verification_v1_20260829"
        ),
        "benchmark_manifest_sha256": manifest_sha,
        "population_receipt_sha256": digest(population_path),
    })
    write_json(construction_path, construction)
    return run, benchmark, construction_path


def test_table2_aggregate_and_independent_verifier(tmp_path):
    run, benchmark, construction = make_table2_fixture(tmp_path)
    summary = aggregate(
        run,
        benchmark,
        construction,
        claim_scope="conference_table2_hm3d_conditional_leg3",
        dataset="HM3D_TABLE2",
        bootstrap_samples=1000,
    )
    assert summary["conditional_on_factual_AB_success"] is True
    assert summary["unconditional_three_leg_joint_sr_reported"] is False
    assert summary["factual_prefix_waterfall"] == {
        "source_A_successful_histories_entering_B": 99,
        "factual_AB_successful_prefixes": 22,
        "factual_AB_scene_clusters": 15,
        "leg3_constructible_histories": 2,
        "leg3_scene_clusters": 2,
    }
    assert summary["revisit_source_segment_counts"] == {"A": 1, "B": 1}
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    checked = verify(
        run, benchmark, construction, summary_path, dataset="HM3D_TABLE2"
    )
    assert checked["verified"] is True
    assert checked["unconditional_three_leg_joint_sr_reported"] is False


def test_table2_verifier_rejects_prefix_tampering(tmp_path):
    run, benchmark, construction = make_table2_fixture(tmp_path)
    summary = aggregate(
        run,
        benchmark,
        construction,
        claim_scope="conference_table2_hm3d_conditional_leg3",
        dataset="HM3D_TABLE2",
        bootstrap_samples=10,
    )
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    item = json.loads((benchmark / "manifest.json").read_text())["episodes"][0]
    prefix = Path(item["online_a_episode"])
    receipt = json.loads((prefix / "receipt.json").read_text())
    receipt["prefix_B_steps"] = 4
    write_json(prefix / "receipt.json", receipt)
    with pytest.raises(RuntimeError, match="raw A/B prefix"):
        verify(
            run,
            benchmark,
            construction,
            summary_path,
            dataset="HM3D_TABLE2",
        )
