import hashlib
import json
from pathlib import Path

import pytest

from MemNavData.build_cec_accepted_query_manifest import build
from MemNavData.controller_portability_contract import CEC_POINTGOAL_UNITS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def make_run(tmp_path: Path, *, flip=False) -> Path:
    run = tmp_path / "run"
    bench = run / "benchmarks/natural_direction"
    source = run / "source/scene0/episode_0000"
    (source / "rgb").mkdir(parents=True)
    (source / "rgb/000003.jpg").write_bytes(b"anchor")
    write_json(source / "online_a_trace.json", {"reached": True, "poses": []})
    item = {
        "scene": "scene0", "episode": "episode_0000",
        "online_a_episode": str(source),
        "online_a_trace_sha256": sha(source / "online_a_trace.json"),
    }
    write_json(bench / "manifest.json", {"episodes": [item]})
    role_pair = {
        "pairs": [{"pair_id": "pair_00", "queries": [
            {"query_id": "pair_00_novel", "analysis_role": "novel",
             "goal_rgb": "pair_00/novel/goal.jpg"},
            {"query_id": "pair_00_revisit", "analysis_role": "revisit",
             "goal_rgb": "pair_00/revisit/goal.jpg"},
        ]}],
    }
    for query in role_pair["pairs"][0]["queries"]:
        goal = bench / "scene0/episode_0000" / query["goal_rgb"]
        goal.parent.mkdir(parents=True, exist_ok=True)
        goal.write_bytes(query["query_id"].encode())
        query["goal_rgb_sha256"] = sha(goal)
    write_json(bench / "scene0/episode_0000/role_pairs.json", role_pair)

    result = run / (
        "evaluation/natural_direction/000_scene0_episode_0000/mono_cec")
    certificate = {
        "schema_version": 3, "accepted": True,
        "checks": {"minimum_inliers": True},
    }
    accepted = {
        "frame_idx": 40, "router_selected_anchor": 3,
        "memory_unbounded_pointgoal": [1.0, 0.2],
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "certified_relocalization_cached": False,
        "certified_relocalization_reason": "certificate_accepted",
        "certified_relocalization_pointgoal_units": CEC_POINTGOAL_UNITS,
        "certified_relocalization_certificate": certificate,
    }
    rejected = {
        "frame_idx": 40,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": False,
        "certified_relocalization_cached": False,
        "certified_relocalization_reason": "no_certificate",
    }
    write_json(result / "episode_0000_pair_00_novel_plans.json", {
        "query_runtime_fields": ["goal_rgb"],
        "query_leg": [rejected],
    })
    revisit_plans = [accepted]
    if flip:
        revisit_plans.append({**accepted,
                              "certified_relocalization_accepted": False,
                              "certified_relocalization_cached": True})
    write_json(result / "episode_0000_pair_00_revisit_plans.json", {
        "query_runtime_fields": ["goal_rgb"],
        "query_leg": revisit_plans,
    })
    return run


def test_builds_role_free_first_accept_population(tmp_path):
    manifest, audit = build(
        make_run(tmp_path), expected_queries=2, expected_accepted=1,
        expected_accepted_scenes=1)
    assert len(manifest["queries"]) == 1
    entry = manifest["queries"][0]
    assert entry["query_id"] == "pair_00_revisit"
    assert "analysis_role" not in entry
    assert entry["selected_anchor"] == 3
    assert audit["accepted_population_equals_revisit_posthoc"] is True
    assert audit["navigation_metric_files_read"] is False


def test_decision_flip_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="decision changed"):
        build(make_run(tmp_path, flip=True), expected_queries=2,
              expected_accepted=1, expected_accepted_scenes=1)
