from __future__ import annotations

from MemNavData.analyze_hm3d_action_coordinate_compass_pair import (
    contrast,
    mcnemar,
)
from MemNavData.run_hm3d_action_coordinate_compass_pair import (
    ARMS,
    rotated_arm_order,
)


def test_two_arm_rotation_is_balanced() -> None:
    assert len(ARMS) == 2
    assert rotated_arm_order(0) == ARMS
    assert rotated_arm_order(1) == ARMS[1:] + ARMS[:1]


def test_exact_mcnemar_and_cluster_receipt() -> None:
    rows = [
        {"scene": "a", "left": 1, "right": 0},
        {"scene": "a", "left": 1, "right": 0},
        {"scene": "b", "left": 0, "right": 1},
        {"scene": "c", "left": 1, "right": 1},
    ]
    result = contrast(rows, left="left", right="right", seed=9)
    assert result["paired_gains"] == 2
    assert result["paired_losses"] == 1
    assert result["scene_clusters"] == 3
    assert result["mcnemar_exact_two_sided_p"] == mcnemar(2, 1)
