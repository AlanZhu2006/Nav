from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from MemNavData.independent_verify_hm3d_table2_stage_spl import (
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


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent_root = tmp_path / "parent"
    parent_manifest = parent_root / "sealed_inputs/parent_manifest.json"
    write_json(parent_manifest, {
        "schema_version": "hm3d_fresh_fullmono_parent_manifest_v1_20260820",
        "episode_count": 2,
        "scenes": ["s0", "s1"],
        "episodes": {
            "s0": [{"episode": "episode_0000"}],
            "s1": [{"episode": "episode_0001"}],
        },
    }, sidecar=False)
    parent_verifier = (
        parent_root / "hm3d_fullmono_mixed_role_independent_verification.json"
    )
    write_json(parent_verifier, {
        "verified": True, "authorized": True,
        "goal_a_sources": 2, "goal_a_successes": 1,
    })
    for scene_index, (scene, episode, reached, geo, path) in enumerate([
        ("s0", "episode_0000", 1, 2.0, 4.0),
        ("s1", "episode_0001", 0, 3.0, 5.0),
    ]):
        scene_root = parent_root / "goal_a/scenes" / f"{scene_index:02d}_{scene}"
        episode_root = scene_root / episode
        trace = episode_root / f"{episode}_leg1_trace.json"
        write_json(trace, {
            "source_scene": scene, "episode": episode,
            "reached": bool(reached), "path_len": path,
        }, sidecar=False)
        trace_sha = sha256(trace)
        write_csv(episode_root / "metric.csv", {
            "episode": episode, "reached_A": reached,
            "leg1_trace_sha256": trace_sha, "geo_A": geo,
            "len_A": path,
            "spl_A": reached * geo / max(geo, path),
        })
        write_json(scene_root / "completion.json", {
            "schema_version": "hm3d_fullmono_goal_a_scene_v1_20260820",
            "status": "complete",
            "records": [{
                "episode": episode, "reached_a": reached,
                "trace_path": str(trace), "trace_sha256": trace_sha,
            }],
        })

    union_root = tmp_path / "union"
    source_verifier_hashes: dict[str, str] = {}
    union_sources = []
    source_specs = [
        ("original_v4", "independent_natural_v4_population_verification.json",
         "s0", "episode_0000__natural_b_00", 1, 3.0, 3.0),
        ("natural_b_expansion",
         "independent_natural_b_expansion_population_verification.json",
         "s1", "episode_0001__natural_b_04", 0, 4.0, 6.0),
    ]
    for name, verifier_name, scene, episode, reached, geo, path in source_specs:
        root = tmp_path / name
        manifest = root / "ab_population/role_pairs/manifest.json"
        write_json(manifest, {"episodes": [{"scene": scene, "episode": episode}]})
        verifier = root / verifier_name
        write_json(verifier, {
            "verified": True, "query_navigation_outcomes_read": False,
            "factual_B_rollouts": 1, "factual_B_successes": reached,
        })
        source_verifier_hashes[name] = sha256(verifier)
        output = root / "factual_b" / f"000_{scene}_{episode}"
        trace = output / f"{episode}_legB_trace.json"
        write_json(trace, {"reached": bool(reached), "path_len": path},
                   sidecar=False)
        metric = output / "result/metric.csv"
        final_distance = 0.5 if reached else 2.5
        write_csv(metric, {
            "scene": scene, "episode": episode, "analysis_role": "novel",
            "reached": reached, "geodesic_m": geo, "path_len_m": path,
            "final_goal_dist_m": final_distance,
        })
        write_json(output / "completion.json", {
            "schema_version": (
                "hm3d_fullmono_lifelong_b_collection_v1_20260824"
            ),
            "status": "complete", "history_index": 0,
            "scene": scene, "episode": episode,
            "reached_B": bool(reached), "path_len_B_m": path,
            "final_goal_dist_B_m": final_distance,
            "B_trace_path": str(trace), "B_trace_sha256": sha256(trace),
            "result_metric_sha256": sha256(metric),
        })
        population = root / "population/population.json"
        write_json(population, {"supported_population": reached})
        union_sources.append({
            "name": name, "run_root": str(root),
            "population_sha256": sha256(population),
            "supported_histories": reached,
        })
    union = union_root / "population/population.json"
    write_json(union, {
        "schema_version": (
            "hm3d_fullmono_lifelong_population_union_v1_20260830"
        ),
        "source_populations": union_sources,
    })

    meeting = tmp_path / "meeting.json"
    write_json(meeting, {
        "schema_version": (
            "hm3d_table2_meeting_result_verification_v1_20260830"
        ),
        "verified": True,
        "leg1_novel": {
            "denominator": "actual A", "attempts": 2, "successes": 1,
        },
        "leg2_novel": {
            "denominator": "factual B", "attempts": 2, "successes": 1,
        },
        "receipts": {
            "parent_manifest_sha256": sha256(parent_manifest),
            "goal_A_independent_verification_sha256": sha256(parent_verifier),
            "source_population_union_sha256": sha256(union),
            "source_factual_B_verification_sha256": source_verifier_hashes,
        },
    })
    return meeting, parent_manifest, union_root


def test_recomputes_factual_stage_spl(tmp_path: Path) -> None:
    meeting, parent, union = make_fixture(tmp_path)
    result = verify(
        meeting_verification_path=meeting,
        parent_manifest_path=parent,
        source_union_root=union,
    )
    assert result["verified"] is True
    assert result["leg1_novel"]["attempts"] == 2
    assert result["leg1_novel"]["spl"] == pytest.approx(0.25)
    assert result["leg2_novel"]["attempts"] == 2
    assert result["leg2_novel"]["spl"] == pytest.approx(0.5)
    assert result["selection_or_policy_execution_performed"] is False


def test_rejects_changed_factual_b_metric(tmp_path: Path) -> None:
    meeting, parent, union = make_fixture(tmp_path)
    metric = next((tmp_path / "original_v4/factual_b").glob("*/result/metric.csv"))
    metric.write_text(metric.read_text().replace("3.0,3.0", "3.0,4.0"))
    with pytest.raises(RuntimeError, match="factual-B metric changed"):
        verify(
            meeting_verification_path=meeting,
            parent_manifest_path=parent,
            source_union_root=union,
        )
