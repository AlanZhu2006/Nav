import json
import math

from MemNavData.audit_vint_cec_direction_consumption import (
    audit_file,
    wrap_deg,
)


def test_wrap_deg() -> None:
    assert wrap_deg(181.0) == -179.0
    assert wrap_deg(-181.0) == 179.0


def test_audit_detects_discarded_u_turn_bearing(tmp_path) -> None:
    result = tmp_path / "formal/evaluation/000_scene_episode/vint/grant/result"
    result.mkdir(parents=True)
    path = result / "episode_pair_revisit_plans.json"
    plans = [
        {
            "step": 0,
            "cec_takeover": True,
            "evaluation_gt_goal_distance_m": 2.0,
            "cec_handoff_packet": {
                "public_proof": {
                    "accepted": True,
                    "pointgoal_units": "lingbot_raw_direction_only",
                    "direction_vector": [-1.0, 0.0],
                }
            },
        },
        {"step": 8, "evaluation_gt_goal_distance_m": 2.3},
    ]
    rollout = [
        {"step": index, "x": 0.0, "z": -0.1 * index, "yaw": 0.0}
        for index in range(9)
    ]
    path.write_text(json.dumps({
        "query_leg": plans,
        "rollout_traces": {"query": rollout},
    }))

    row = audit_file(path, exec_horizon=8)
    assert math.isclose(row["cec_bearing_abs_deg"], 180.0)
    assert math.isclose(row["executed_heading_deg"], 0.0)
    assert math.isclose(row["bearing_execution_error_deg"], 180.0)
    assert row["moved_away"] is True
