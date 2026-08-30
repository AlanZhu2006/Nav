from __future__ import annotations

import json
from pathlib import Path

from independent_verify_hm3d_fullmono_lifelong_population_union import verify
from merge_hm3d_fullmono_lifelong_populations import merge
from hm3d_fullmono_lifelong import sha256_file


def sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )


def source(root: Path, *, scene: str, episode: str) -> dict:
    benchmark_root = root / "population/benchmark" / scene / episode
    benchmark_root.mkdir(parents=True)
    benchmark = benchmark_root / "benchmark.json"
    benchmark.write_text(json.dumps({"scene": scene, "episode": episode}) + "\n")
    role_root = root / "ab_population/role_pairs" / scene / episode
    role_root.mkdir(parents=True)
    (role_root / "role_pairs.json").write_text(json.dumps({
        "scene": scene, "episode": episode, "pairs": [],
    }) + "\n")
    population = {
        "schema_version": "fixture",
        "intention_to_collect_B": 2,
        "selection_reads_C_B2_C2_navigation_outcomes": False,
        "runtime_role_visibility": "none",
        "accepted": [{
            "population_index": 0,
            "source_AB_history_index": 0,
            "scene": scene,
            "episode": episode,
            "benchmark": f"benchmark/{scene}/{episode}/benchmark.json",
            "benchmark_sha256": sha256_file(benchmark),
            "B_goal_strong_support": True,
        }],
    }
    population_path = root / "population/population.json"
    population_path.write_text(json.dumps(population, sort_keys=True) + "\n")
    sidecar(population_path)
    (root / "population/SEALED").write_text("sealed\n")
    files = sorted(
        path for path in (root / "population").rglob("*")
        if path.is_file() and path.name not in {"POPULATION_FILES.sha256", "SEALED"}
    )
    ledger = root / "population/POPULATION_FILES.sha256"
    ledger.write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(root / 'population')}\n"
        for path in files
    ))
    return {
        "population_sha256": sha256_file(population_path),
        "ledger_sha256": sha256_file(ledger),
    }


def test_exact_union_and_independent_verification(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    expansion_root = tmp_path / "expansion"
    original = source(original_root, scene="scene_a", episode="episode_a")
    expansion = source(
        expansion_root, scene="scene_b", episode="episode_b__natural_b_04"
    )
    expansion_verification = (
        expansion_root
        / "independent_natural_b_expansion_population_verification.json"
    )
    expansion_verification.write_text(json.dumps({
        "verified": True,
        "query_navigation_outcomes_read": False,
        "factual_C_B2_C2_executed": False,
        "population_sha256": expansion["population_sha256"],
        "supported_population": 1,
    }) + "\n")
    sidecar(expansion_verification)

    expansion_protocol = tmp_path / "expansion_protocol.json"
    expansion_protocol.write_text(json.dumps({
        "population_union": {
            "original_run_root": str(original_root),
            "original_population_sha256": original["population_sha256"],
            "original_population_file_ledger_sha256": original["ledger_sha256"],
            "original_supported_histories": 1,
            "minimum_target_histories": 2,
            "minimum_target_scene_clusters": 2,
        }
    }) + "\n")
    base_protocol = Path(__file__).parent / (
        "hm3d_table2_leg3_mixed_role_protocol_20260829.json"
    )
    out = tmp_path / "union"
    result = merge(
        expansion_protocol_path=expansion_protocol,
        expansion_run=expansion_root,
        base_table2_protocol_path=base_protocol,
        out=out,
    )
    assert result["union_supported_histories"] == 2
    assert result["target_met"] is True
    audited = verify(out)
    assert audited["verified"] is True
    assert audited["source_supported_histories"] == {
        "original_v4": 1, "natural_b_expansion": 1,
    }
    assert audited["union_supported_histories"] == 2
    assert audited["union_scene_clusters"] == 2
