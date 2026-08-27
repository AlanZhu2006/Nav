import pandas as pd
import pytest

from MemNavData.build_train40_certificate_challenge_manifest import (
    build_manifest,
    session_universe_sha256,
)


def inventory():
    rows = []
    for scene in ("scene_b", "scene_a"):
        for index in range(2):
            for _candidate in range(3):
                rows.append({
                    "session_id": f"train/{scene}/episode/{index}",
                    "scene": scene,
                    "kind": "manifest_causal_goal_localization_train",
                    "split_role": "train",
                    "teacher_covis": 0.99 if index else 0.0,
                })
    return pd.DataFrame(rows)


def test_complete_universe_is_sorted_and_label_independent():
    first = inventory()
    second = first.copy()
    second["teacher_covis"] = 1.0 - second["teacher_covis"]
    kwargs = dict(
        evidence_sha256="a" * 64,
        teacher_sha256="b" * 64,
        expected_sessions=4,
        expected_scenes=2,
    )
    manifest = build_manifest(first, **kwargs)
    changed_labels = build_manifest(second, **kwargs)
    assert manifest == changed_labels
    assert manifest["sessions"] == sorted(manifest["sessions"])
    assert manifest["session_universe_sha256"] == session_universe_sha256(
        manifest["sessions"])
    assert manifest["selection_uses_labels"] is False


def test_wrong_universe_fails_closed():
    with pytest.raises(RuntimeError, match="session universe changed"):
        build_manifest(
            inventory(), evidence_sha256="a" * 64,
            teacher_sha256="b" * 64,
            expected_sessions=5, expected_scenes=2)
