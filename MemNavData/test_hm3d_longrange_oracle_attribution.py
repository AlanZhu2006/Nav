from MemNavData.longrange_oracle_attribution import ORACLE_ARMS
from MemNavData.run_hm3d_longrange_oracle_attribution import (
    audit_oracle_plans,
    rotated_arm_order,
)


def test_four_arm_rotation_is_balanced():
    assert len(ORACLE_ARMS) == 4
    for index in range(4):
        assert rotated_arm_order(index) == (
            ORACLE_ARMS[index:] + ORACLE_ARMS[:index])


def _oracle_plan(arm):
    plan = {
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "action_coordinate_oracle_schema_version": (
            "longrange_oracle_attribution_v1_20260903"),
        "action_coordinate_oracle_arm": arm,
        "action_coordinate_oracle_evaluator_pose_consumed": True,
        "action_coordinate_oracle_goal_position_consumed": True,
        "action_coordinate_oracle_read_only_resample": True,
        "action_coordinate_oracle_pointgoal": [2.5, 0.0],
        "action_coordinate_oracle_controller": (
            "pure_pointgoal" if arm == "oracle_geodesic_point"
            else "mixed_image_pointgoal"),
    }
    if arm == "oracle_route_mixed":
        plan.update(
            action_coordinate_oracle_progress_m=2.0,
            action_coordinate_oracle_cross_track_m=0.5,
        )
    else:
        plan["action_coordinate_oracle_current_geodesic_m"] = 10.0
    return plan


def test_oracle_receipt_audits_mixed_and_pure_controllers():
    for arm in ORACLE_ARMS[1:]:
        result = audit_oracle_plans(arm, [_oracle_plan(arm)])
        assert result["evaluator_pose_consumed"] is True
        assert result["fixed_controller_radius_m"] == 2.5
