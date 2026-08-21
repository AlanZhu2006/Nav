import json
import pathlib

import pytest

from MemNavData.goat_sequential_revisit_pilot import (
    ACTION_IDS,
    _enforce_official_stop_authority,
    _manifest_request,
    _validate_manifest_target,
    first_repeated_image_subtask,
    prior_instance_subtasks,
)


def test_finds_first_repeated_image_instance_without_role_label():
    tasks = [
        ["chair", "object", None],
        ["mirror", "image", "mirror_30", 27],
        ["rug", "description", "rug_1"],
        ["mirror", "image", "mirror_30", 21],
    ]
    assert first_repeated_image_subtask(tasks) == 3


def test_object_repetition_does_not_define_image_revisit():
    tasks = [
        ["mirror", "object", None],
        ["mirror", "image", "mirror_30", 27],
        ["mirror", "image", "mirror_30", 21],
    ]
    assert first_repeated_image_subtask(tasks) == 2


def test_description_to_image_exact_instance_is_recurrence():
    tasks = [
        ["rug", "description", "rug_7"],
        ["chair", "object", None],
        ["rug", "image", "rug_7", 11],
    ]
    assert first_repeated_image_subtask(tasks) == 2
    assert prior_instance_subtasks(tasks, 2) == [{
        "subtask_index": 0,
        "modality": "description",
        "instance_id": "rug_7",
    }]


def test_episode_without_repeated_image_is_rejected():
    with pytest.raises(ValueError):
        first_repeated_image_subtask([
            ["mirror", "image", "mirror_30", 27],
            ["rug", "image", "rug_1", 5],
        ])


def test_frozen_manifest_entry_is_evaluator_only_and_validated(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {
        "method_or_threshold_selection_allowed": False,
        "controller_reads_target_metadata": False,
        "purpose": "test",
        "episodes": [{
            "index": 0,
            "arm_order": ["native", "cec"],
            "scene_id": "sceneA",
            "episode_id": "7",
            "target_subtask_index": 2,
            "target_instance_id": "rug_7",
            "prior_instance_subtasks": [{
                "subtask_index": 0,
                "modality": "description",
                "instance_id": "rug_7",
            }],
        }],
    }
    path.write_text(json.dumps(payload))
    identity, loaded, entry = _manifest_request(path, 0)
    assert identity == ("sceneA", "7")
    assert loaded == payload

    class Episode:
        tasks = [
            ["rug", "description", "rug_7"],
            ["chair", "object", None],
            ["rug", "image", "rug_7", 5],
        ]

    _validate_manifest_target(Episode(), entry)


def test_manifest_rejects_controller_role_metadata(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "method_or_threshold_selection_allowed": False,
        "controller_reads_target_metadata": True,
        "episodes": [{"scene_id": "sceneA", "episode_id": "7"}],
    }))
    with pytest.raises(RuntimeError, match="role-free"):
        _manifest_request(path, 0)


def test_cec_cannot_delay_or_replace_official_subtask_stop():
    stop = ACTION_IDS["subtask_stop"]
    assert _enforce_official_stop_authority(stop, stop) == stop
    with pytest.raises(RuntimeError, match="SUBTASK_STOP"):
        _enforce_official_stop_authority(stop, ACTION_IDS["turn_left"])
    assert _enforce_official_stop_authority(
        ACTION_IDS["move_forward"], ACTION_IDS["turn_left"]
    ) == ACTION_IDS["turn_left"]


def test_manifest_target_rejects_invalid_arm_order():
    class Episode:
        tasks = [
            ["rug", "description", "rug_7"],
            ["rug", "image", "rug_7", 5],
        ]

    entry = {
        "target_subtask_index": 1,
        "target_instance_id": "rug_7",
        "prior_instance_subtasks": [{
            "subtask_index": 0,
            "modality": "description",
            "instance_id": "rug_7",
        }],
        "arm_order": ["native", "native"],
    }
    with pytest.raises(RuntimeError, match="arm order"):
        _validate_manifest_target(Episode(), entry)


def test_goat_torch_does_not_inherit_new_allocator_option():
    script = pathlib.Path(__file__).with_name(
        "slurm_goat_sequential_revisit_eval.sbatch").read_text()
    assert "export PYTORCH_CUDA_ALLOC_CONF" not in script
    assert script.count(
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") == 2
    assert '"${BASE_SIF}" env -u PYTORCH_CUDA_ALLOC_CONF' in script
