import types
import hashlib

import numpy as np
from PIL import Image
import pytest
import torch

from MemNavData.monocular_adjacent_motion import PlanarMotionReceipt
from MemNavData.route_alignment_contract import (
    build_route_alignment_packet,
    route_alignment_executor_source,
)
from NavDP.baselines.memnav.policy_agent import (
    MemNavAgent,
    certified_route_edge_motion_model,
)


def motion(forward: float, left: float = 0.0, yaw: float = 0.0):
    return PlanarMotionReceipt(
        forward_m=forward,
        left_m=left,
        yaw_rad=yaw,
        vertical_m=0.0,
    )


def pose9(x=0.0, y=0.0, z=0.0):
    return torch.tensor(
        [x, y, z, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    )


def fake_route_agent():
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.n = 65
    agent.certified_relocalization_matcher = object()
    agent.certified_route_depth_cache_stride = 8
    agent.certified_route_motion_model = "fundamental_then_pnp"
    agent.certified_route_motion_unfiltered_shadow = False
    agent._certified_route_depth_cached_anchors = {
        8, 16, 24, 32, 40, 48, 56, 64,
    }
    agent._certified_route_live_depth_cache = {}
    agent._certified_monocular_route_tangents = {}

    def terminal(self, target_anchor, goal_pose9):
        del target_anchor, goal_pose9
        return motion(0.2), {
            "edge_kind": "cec_terminal_geometric_witness",
            "motion": motion(0.2).audit_dict(),
        }

    def edge(self, **kwargs):
        value = -0.5 if kwargs["edge_kind"] == (
            "live_query_adjacent_sample") else 1.0
        receipt = {
            "edge_kind": kwargs["edge_kind"],
            "from_frame": kwargs["reference_frame"],
            "to_frame": kwargs["query_frame"],
            "motion": motion(value).audit_dict(),
        }
        return motion(value), receipt

    def current_depth(self):
        return np.ones((2, 2)), np.ones((2, 2)), "unit_test_depth"

    agent._certified_terminal_route_motion = types.MethodType(terminal, agent)
    agent._certified_visual_route_edge = types.MethodType(edge, agent)
    agent._materialize_current_route_depth = types.MethodType(
        current_depth, agent)
    return agent


def test_dense_query_model_changes_only_live_query_estimator():
    assert certified_route_edge_motion_model(
        "direct_pnp_dense_query", "historical_adjacent_sample",
    ) == "fundamental_then_pnp"
    assert certified_route_edge_motion_model(
        "direct_pnp_dense_query", "history_to_query_bridge",
    ) == "fundamental_then_pnp"
    assert certified_route_edge_motion_model(
        "direct_pnp_dense_query", "live_query_adjacent_sample",
    ) == "direct_pnp"
    assert certified_route_edge_motion_model(
        "direct_pnp", "historical_adjacent_sample",
    ) == "direct_pnp"


def test_dense_query_model_applies_stage_specific_prefilter(
        tmp_path, monkeypatch):
    for frame in (40, 41):
        Image.new("RGB", (32, 24), color=(frame, frame, frame)).save(
            tmp_path / f"{frame}.jpg")
    observed_thresholds = []

    def estimate(**kwargs):
        observed_thresholds.append(kwargs["epipolar_threshold_px"])
        return {
            "pnp": {
                "status": "ok",
                "inliers": 20,
                "reprojection_rmse_px": 0.5,
            },
            "local_motion_validity": {"accepted": True},
            "motion": motion(0.1).audit_dict(),
            "matches": 20,
            "primary_motion_model": (
                "fundamental_then_pnp"
                if kwargs["epipolar_threshold_px"] is not None
                else "direct_pnp"),
        }

    monkeypatch.setattr(
        "MemNavData.monocular_adjacent_motion.estimate_adjacent_motion",
        estimate,
    )
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.rgb_dir = str(tmp_path)
    agent.camera_intrinsic = np.asarray([
        [20.0, 0.0, 16.0],
        [0.0, 20.0, 12.0],
        [0.0, 0.0, 1.0],
    ])
    agent.lb = types.SimpleNamespace(patch_size=14)
    agent.certified_relocalization_matcher = object()
    agent.certified_route_motion_model = "direct_pnp_dense_query"
    agent.certified_route_motion_unfiltered_shadow = False
    agent._certified_metric_route_depth = types.MethodType(
        lambda self, frame, target_anchor: (
            np.ones((24, 32)), {"frame_index": frame}),
        agent,
    )

    history, history_receipt = agent._certified_visual_route_edge(
        reference_frame=40,
        query_frame=41,
        target_anchor=40,
        edge_kind="historical_adjacent_sample",
    )
    live, live_receipt = agent._certified_visual_route_edge(
        reference_frame=40,
        query_frame=41,
        target_anchor=40,
        edge_kind="live_query_adjacent_sample",
    )

    assert history.forward_m == pytest.approx(0.1)
    assert live.forward_m == pytest.approx(0.1)
    assert observed_thresholds[0] is not None
    assert observed_thresholds[1] is None
    assert history_receipt[
        "resolved_edge_motion_model"] == "fundamental_then_pnp"
    assert live_receipt["resolved_edge_motion_model"] == "direct_pnp"


def test_policy_builds_one_visual_route_and_reuses_it_continuously():
    agent = fake_route_agent()
    direction, first = agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )

    np.testing.assert_allclose(direction, [-1.0, 0.0], atol=1e-12)
    assert first["local_tangent_status"] == "active"
    assert first["local_tangent_history_nodes"] == [40, 48, 56, 64]
    assert first["local_tangent_history_edge_count"] == 4
    assert first["local_tangent_evaluator_pose_consumed"] is False
    assert first["local_tangent_executor_odometry_consumed"] is False
    assert first["local_tangent_native_fallback_available"] is False
    assert np.linalg.norm(first[
        "local_tangent_controller_pointgoal"]) == pytest.approx(2.5)

    # One later planning frame advances the same route state from visual
    # motion; it must not rebuild or switch to an endpoint controller.
    agent.n = 73
    agent._certified_route_depth_cached_anchors.add(72)
    direction, second = agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )
    assert second["local_tangent_query_edge_count"] == 1
    assert second["local_tangent_projected_progress_m"] == pytest.approx(0.5)
    np.testing.assert_allclose(direction, [-1.0, 0.0], atol=1e-12)


def _route_alignment_packet(direction=(-1.0, 0.0)):
    proof = {
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 40,
        "selected_anchor_image_sha256": "a" * 64,
        "certificate": {"accepted": True},
    }
    return build_route_alignment_packet(
        authority_proof=proof,
        current_frame=64,
        goal_start_frame=64,
        target_anchor=40,
        current_rgb_sha256="b" * 64,
        goal_image_sha256="c" * 64,
        anchor_image_sha256="a" * 64,
        history_edge_receipt_sha256="d" * 64,
        unit_bearing=direction,
        tangent_baseline_m=0.30,
        controller_radius_m=2.5,
    )


def _control_receipt(frame, yaw, packet_sha):
    return {
        "frame_index": frame,
        "executed_translation_m": 0.0,
        "executed_yaw_rad": yaw,
        "contract": "frame_bound_realized_executor_motion_v1",
        "local_se2": {
            "contract": "frame_bound_local_se2_v1",
            "executed_forward_m": 0.0,
            "executed_left_m": 0.0,
            "executed_yaw_rad": yaw,
            "source": route_alignment_executor_source(packet_sha),
        },
    }


def test_policy_consumes_atomic_route_yaw_without_visual_pnp():
    agent = fake_route_agent()
    agent.executor_motion_receipts = [None] * agent.n
    agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )
    route = agent._certified_monocular_route_tangents["goal"]
    packet = _route_alignment_packet()
    route["alignment_packet"] = packet

    yaw = np.deg2rad(30.0)
    agent.executor_motion_receipts.extend([
        _control_receipt(frame, yaw, packet["packet_sha256"])
        for frame in range(65, 71)
    ])
    agent.n = 71
    direction, diagnostics = (
        agent._certified_monocular_route_tangent_direction(
            goal_key="goal",
            target_anchor=40,
            goal_start_frame=64,
            goal_pose9=np.asarray(pose9()),
        )
    )

    np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-12)
    assert diagnostics["local_tangent_query_edge_count"] == 0
    assert diagnostics["local_tangent_update_edge_count"] == 6
    assert diagnostics["local_tangent_alignment_control_edge_count"] == 6
    assert diagnostics[
        "local_tangent_alignment_executor_receipt_consumed"] is True


def test_policy_issues_alignment_packet_from_accepted_route_proof(tmp_path):
    agent = fake_route_agent()
    agent.S = 8
    agent.rgb_dir = str(tmp_path)
    anchor = b"anchor-jpeg"
    (tmp_path / "40.jpg").write_bytes(anchor)
    agent._last_frame_jpg_sha256 = "b" * 64
    anchor_sha = hashlib.sha256(anchor).hexdigest()
    direction, diagnostics = (
        agent._certified_monocular_route_tangent_direction(
            goal_key="goal",
            target_anchor=40,
            goal_start_frame=64,
            goal_pose9=np.asarray(pose9()),
            authority_proof={
                "ok": True,
                "accepted": True,
                "reason": "certificate_accepted",
                "selected_anchor": 40,
                "selected_anchor_image_sha256": anchor_sha,
                "certificate": {"accepted": True},
            },
            goal_image_sha256="c" * 64,
        )
    )
    np.testing.assert_allclose(direction, [-1.0, 0.0], atol=1e-12)
    packet = diagnostics["local_tangent_alignment_packet"]
    assert packet["target_anchor"] == 40
    assert packet["current_rgb_sha256"] == "b" * 64
    assert diagnostics["local_tangent_alignment_packet_sha256"] == packet[
        "packet_sha256"]


def test_policy_rejects_partial_route_alignment_receipt_sequence():
    agent = fake_route_agent()
    agent.executor_motion_receipts = [None] * agent.n
    agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )
    route = agent._certified_monocular_route_tangents["goal"]
    packet = _route_alignment_packet()
    route["alignment_packet"] = packet
    agent.executor_motion_receipts.append(_control_receipt(
        65, np.deg2rad(30.0), packet["packet_sha256"]))
    agent.n = 66

    direction, diagnostics = (
        agent._certified_monocular_route_tangent_direction(
            goal_key="goal",
            target_anchor=40,
            goal_start_frame=64,
            goal_pose9=np.asarray(pose9()),
        )
    )
    assert direction is None
    assert diagnostics["local_tangent_status"] == "geometry_failure"
    assert "before its proof-bound turn was complete" in diagnostics[
        "local_tangent_error"]


def test_dense_query_motion_consumes_every_frame_bound_depth_receipt():
    agent = fake_route_agent()
    agent.certified_route_motion_model = "direct_pnp_dense_query"
    agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )

    agent.n = 73
    agent._certified_route_live_depth_cache = {
        frame: (np.ones((2, 2)), np.ones((2, 2)))
        for frame in range(65, 73)
    }
    _direction, diagnostics = (
        agent._certified_monocular_route_tangent_direction(
            goal_key="goal",
            target_anchor=40,
            goal_start_frame=64,
            goal_pose9=np.asarray(pose9()),
        )
    )

    assert diagnostics["local_tangent_update_edge_count"] == 8
    assert diagnostics["local_tangent_query_edge_count"] == 8
    assert agent._certified_route_live_depth_cache == {}


def test_policy_fails_closed_when_visual_route_breaks():
    agent = fake_route_agent()

    def broken(self, **kwargs):
        del kwargs
        raise RuntimeError("adjacent visual motion rejected: minimum_inliers")

    agent._certified_visual_route_edge = types.MethodType(broken, agent)
    direction, output = agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )
    assert direction is None
    assert output["local_tangent_status"] == "geometry_failure"
    assert "minimum_inliers" in output["local_tangent_error"]
    assert output["local_tangent_native_fallback_available"] is False


def test_policy_supports_early_cec_anchor_with_canonical_depth():
    agent = fake_route_agent()
    direction, output = agent._certified_monocular_route_tangent_direction(
        goal_key="early-goal",
        target_anchor=9,
        goal_start_frame=64,
        goal_pose9=np.asarray(pose9()),
    )
    assert output["local_tangent_status"] == "active"
    assert output["local_tangent_pre_metric_anchor_bridge"] is False
    assert output["local_tangent_history_nodes"][:2] == [9, 16]
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_canonical_cec_depth_boundary_is_lingbot_scale_block_not_frame40():
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.S = 8
    agent.n = 40
    with pytest.raises(ValueError, match=r"outside \[8, 39\]"):
        agent._certified_reference_depth_impl(7)


def test_terminal_goal_offset_is_derived_from_the_cec_pose_witness():
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.cam_pose = [pose9() for _ in range(41)]
    agent._first40_local_pose_metric_scale = types.MethodType(
        lambda self: {
            "available": True,
            "metric_scale_m_per_raw": 0.5,
            "scale_receipt_sha256": "a" * 64,
        },
        agent,
    )
    goal = np.asarray(pose9(z=2.0))
    goal_to_anchor, receipt = agent._certified_terminal_route_motion(40, goal)

    # Anchor->goal is +1 m forward after metricization, so the chronological
    # goal->anchor edge is its exact inverse.
    assert goal_to_anchor.forward_m == pytest.approx(-1.0)
    assert goal_to_anchor.left_m == pytest.approx(0.0)
    assert goal_to_anchor.yaw_rad == pytest.approx(0.0)
    assert receipt["edge_kind"] == "cec_terminal_geometric_witness"
