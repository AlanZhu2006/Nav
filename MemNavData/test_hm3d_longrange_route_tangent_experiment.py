import math
from pathlib import Path
import subprocess
import sys

import pytest

from MemNavData.hm3d_longrange_route_tangent_experiment import (
    ARMS,
    audit_route_tangent,
    initial_proof,
    rotated_arm_order,
)


def _accepted_plan(**updates):
    plan = {
        "certified_relocalization_accepted": True,
        "local_tangent_status": "active",
        "local_tangent_evaluator_pose_consumed": False,
        "local_tangent_habitat_path_consumed": False,
        "local_tangent_executor_odometry_consumed": False,
        "local_tangent_metric_depth_sensor_consumed": False,
        "local_tangent_distance_regime_present": False,
        "local_tangent_stuck_trigger_present": False,
        "local_tangent_endpoint_fallback_available": False,
        "local_tangent_native_fallback_available": False,
        "local_tangent_tangent_baseline_m": 0.3,
        "local_tangent_controller_radius_m": 2.5,
        "local_tangent_unit_bearing": [1.0, 0.0],
        "local_tangent_controller_pointgoal": [2.5, 0.0],
        "local_tangent_history_edge_receipt_sha256": "a" * 64,
        "local_tangent_target_anchor": 8,
        "local_tangent_pre_metric_anchor_bridge": False,
        "router_selected_anchor": 8,
        "router_candidate_order_dino": [8, 16],
        "router_candidate_order_used": [8, 16],
        "certified_relocalization_certificate": {"accepted": True},
        "certified_relocalization_authority": {"accepted": True},
        "certified_relocalization_pnp": {"status": "ok"},
        "certified_relocalization_selected_proposal_source": "geometry",
    }
    plan.update(updates)
    return plan


def test_arm_order_rotates_without_changing_membership():
    for index in range(6):
        assert set(rotated_arm_order(index)) == set(ARMS)
        assert rotated_arm_order(index)[0] == ARMS[index % len(ARMS)]


def test_route_tangent_audit_accepts_total_scale_free_readout():
    result = audit_route_tangent([_accepted_plan()])
    assert result == {
        "certificate_accept_plans": 1,
        "active_route_plans": 1,
        "geometry_stream_stop_plans": 0,
        "history_route_receipt_hashes": 1,
        "canonical_anchor_boundary_verified": True,
    }


def test_route_tangent_audit_accepts_atomic_terminal_geometry_stop():
    stop = _accepted_plan(
        geometry_stream_stop=True,
        geometry_stream_controller_called=False,
        geometry_stream_native_fallback_executed=False,
        geometry_stream_endpoint_fallback_executed=False,
        local_tangent_status="geometry_failure",
    )
    # Runtime stop packets intentionally omit successful-route receipts.
    stop.pop("local_tangent_pre_metric_anchor_bridge")
    result = audit_route_tangent([stop])
    assert result["geometry_stream_stop_plans"] == 1
    assert result["active_route_plans"] == 0
    assert result["canonical_anchor_boundary_verified"] is True


def test_route_tangent_audit_rejects_stop_with_forbidden_bridge():
    with pytest.raises(RuntimeError, match="forbidden pre-metric bridge"):
        audit_route_tangent([_accepted_plan(
            geometry_stream_stop=True,
            geometry_stream_controller_called=False,
            geometry_stream_native_fallback_executed=False,
            geometry_stream_endpoint_fallback_executed=False,
            local_tangent_status="geometry_failure",
            local_tangent_pre_metric_anchor_bridge=True,
        )])


def test_route_tangent_rejects_hidden_fallback():
    with pytest.raises(RuntimeError, match="forbidden branch"):
        audit_route_tangent([_accepted_plan(
            local_tangent_native_fallback_available=True,
        )])


def test_initial_proof_excludes_guidance_specific_fields():
    first = _accepted_plan(local_tangent_unit_bearing=[1.0, 0.0])
    second = _accepted_plan(local_tangent_unit_bearing=[0.0, 1.0])
    assert initial_proof([first]) == initial_proof([second])
    assert math.isclose(
        math.hypot(*first["local_tangent_controller_pointgoal"]), 2.5)


def test_experiment_contract_import_does_not_load_opencv():
    """The Habitat evaluator must not inherit the geometry runtime stack."""

    repository = Path(__file__).resolve().parents[1]
    script = "\n".join((
        "import sys",
        f"sys.path.insert(0, {str(repository)!r})",
        "class RejectOpenCV:",
        "    def find_spec(self, fullname, path=None, target=None):",
        "        if fullname == 'cv2' or fullname.startswith('cv2.'):",
        "            raise RuntimeError('OpenCV imported by lightweight contract')",
        "        return None",
        "sys.meta_path.insert(0, RejectOpenCV())",
        "import MemNavData.hm3d_longrange_route_tangent_experiment",
    ))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
