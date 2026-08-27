import gzip
import json

import pytest

from MemNavData.build_goat_sequential_revisit_manifest import build_manifest


def _write_shard(root, scene, episodes):
    content = root / "content"
    content.mkdir(parents=True, exist_ok=True)
    path = content / (scene + ".json.gz")
    with gzip.open(str(path), "wt") as handle:
        json.dump({"episodes": episodes, "goals": {}}, handle)


def _episode(scene, episode_id, tasks):
    return {
        "episode_id": str(episode_id),
        "scene_id": "hm3d/val//00001-{0}/{0}.basis.glb".format(scene),
        "tasks": tasks,
    }


def test_manifest_uses_description_or_image_history_and_earliest_target(tmp_path):
    _write_shard(tmp_path, "sceneA", [
        _episode("sceneA", 0, [
            ["chair", "object", None],
            ["lamp", "image", "lamp_1", 2],
            ["lamp", "image", "lamp_1", 4],
        ]),
        _episode("sceneA", 1, [
            ["rug", "description", "rug_7"],
            ["rug", "image", "rug_7", 3],
        ]),
    ])
    manifest = build_manifest(tmp_path)
    assert manifest["source_population"] == {
        "content_scenes": 1,
        "episodes": 2,
        "image_goals": 3,
        "exact_repeated_image_targets": 2,
        "episodes_with_exact_recurrence": 2,
        "scenes_with_exact_recurrence": 1,
        "selected_scenes": 1,
    }
    selected = manifest["episodes"][0]
    assert selected["arm_order"] == ["native", "cec"]
    assert selected["episode_id"] == "1"
    assert selected["target_subtask_index"] == 1
    assert selected["prior_instance_subtasks"] == [{
        "subtask_index": 0,
        "modality": "description",
        "instance_id": "rug_7",
    }]


def test_manifest_exclusion_and_dataset_receipt_are_deterministic(tmp_path):
    tasks = [
        ["lamp", "image", "lamp_1", 2],
        ["lamp", "image", "lamp_1", 4],
    ]
    _write_shard(tmp_path, "sceneA", [_episode("sceneA", 0, tasks)])
    _write_shard(tmp_path, "sceneB", [_episode("sceneB", 0, tasks)])
    first = build_manifest(tmp_path, excluded_scenes=["sceneA"])
    second = build_manifest(tmp_path, excluded_scenes=["sceneA"])
    assert first == second
    assert [item["scene_id"] for item in first["episodes"]] == ["sceneB"]
    assert first["episodes"][0]["arm_order"] == ["native", "cec"]
    assert len(first["dataset_receipt"]["sorted_content_receipts_sha256"]) == 64


def test_include_and_exclude_cannot_overlap(tmp_path):
    with pytest.raises(ValueError):
        build_manifest(
            tmp_path, excluded_scenes=["sceneA"], included_scenes=["sceneA"])
