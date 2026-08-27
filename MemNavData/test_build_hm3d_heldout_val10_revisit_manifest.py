#!/usr/bin/env python3

import json
from pathlib import Path

from MemNavData.build_hm3d_heldout_val10_revisit_manifest import (
    validate_protocol,
    validate_parent_protocol,
    validate_selection_audit,
)


def _protocol() -> dict:
    path = Path(__file__).with_name(
        "hm3d_heldout_val10_revisit_protocol_20260816.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _attrition_protocol() -> tuple[Path, dict]:
    path = Path(__file__).with_name(
        "hm3d_heldout_val10_revisit_attrition_protocol_20260816.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_frozen_protocol_is_valid_and_scene_balanced() -> None:
    protocol = _protocol()
    scenes = validate_protocol(protocol)
    assert len(scenes) == 10
    assert [row["index"] for row in scenes] == list(range(10))
    assert len({row["scene_id"] for row in scenes}) == 10
    path = Path(__file__).with_name(
        "hm3d_heldout_val10_revisit_protocol_20260816.json")
    _, audit = validate_selection_audit(protocol, path)
    assert audit["consumed_scene_count"] == 36
    assert audit["selected_overlap_with_consumed"] == []


def test_protocol_rejects_mp3d_guard_removal() -> None:
    protocol = _protocol()
    protocol["frozen_guards"]["no_mp3d_evaluation"] = False
    try:
        validate_protocol(protocol)
    except RuntimeError as error:
        assert "no_mp3d_evaluation" in str(error)
    else:
        raise AssertionError("missing no-MP3D guard was accepted")


def test_attrition_protocol_preserves_parent_and_original_indices() -> None:
    path, protocol = _attrition_protocol()
    scenes = validate_protocol(protocol)
    parent_path, parent, parent_sha = validate_parent_protocol(protocol, path)
    assert parent_path.name == "hm3d_heldout_val10_revisit_protocol_20260816.json"
    assert parent_sha == protocol["parent_protocol"]["sha256"]
    assert parent["scenes"] == scenes
    attrition = protocol["construction_attrition"]
    assert attrition["constructible_scene_indices"] == list(range(8)) + [9]
    assert attrition["failed_scenes"][0]["index"] == 8
    assert attrition["navigation_outcomes_read"] is False


def test_attrition_protocol_rejects_scene_replacement() -> None:
    _path, protocol = _attrition_protocol()
    protocol["construction_attrition"]["scene_replacement"] = True
    try:
        validate_protocol(protocol)
    except RuntimeError as error:
        assert "outcome blind" in str(error)
    else:
        raise AssertionError("scene replacement was accepted")
