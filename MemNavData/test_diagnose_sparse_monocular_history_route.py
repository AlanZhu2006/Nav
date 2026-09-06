import pytest

from MemNavData.diagnose_sparse_monocular_history_route import (
    fixed_keyframe_indices,
)


def test_fixed_keyframe_schedule_after_metric_receipt():
    assert fixed_keyframe_indices(
        anchor=67, frame_count=100, frame_stride=8,
    ) == [67, 72, 80, 88, 96, 99]


def test_fixed_keyframe_schedule_includes_frame40_bridge():
    assert fixed_keyframe_indices(
        anchor=26, frame_count=67, frame_stride=8,
    ) == [26, 40, 48, 56, 64, 66]


def test_fixed_keyframe_schedule_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        fixed_keyframe_indices(anchor=5, frame_count=6, frame_stride=8)
    with pytest.raises(ValueError):
        fixed_keyframe_indices(anchor=1, frame_count=10, frame_stride=1)
