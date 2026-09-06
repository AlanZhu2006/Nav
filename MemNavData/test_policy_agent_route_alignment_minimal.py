import hashlib
import types

import numpy as np
import pytest
import torch

from MemNavData.monocular_adjacent_motion import PlanarMotionReceipt
from MemNavData.route_alignment_contract import (
    route_alignment_executor_source,
)
from NavDP.baselines.memnav.policy_agent import MemNavAgent


def _motion(forward: float, left: float = 0.0, yaw: float = 0.0):
    return PlanarMotionReceipt(
        forward_m=forward,
        left_m=left,
        yaw_rad=yaw,
        vertical_m=0.0,
    )


def _pose9():
    return torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    )


def _agent(tmp_path):
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.S = 8
    agent.n = 65
    agent.rgb_dir = str(tmp_path)
    agent.certified_relocalization_matcher = object()
    agent.certified_route_depth_cache_stride = 8
    agent.certified_route_motion_model = "fundamental_then_pnp"
    agent.certified_route_motion_unfiltered_shadow = False
    agent._certified_route_live_depth_cache = {}
    agent._certified_route_depth_cached_anchors = {
        8, 16, 24, 32, 40, 48, 56, 64,
    }
    agent._certified_monocular_route_tangents = {}
    agent.executor_motion_receipts = [None] * agent.n
    agent._last_frame_jpg_sha256 = "b" * 64
    anchor = b"immutable-anchor-jpeg"
    (tmp_path / "40.jpg").write_bytes(anchor)

    def terminal(self, target_anchor, goal_pose9):
        del target_anchor, goal_pose9
        value = _motion(0.2)
        return value, {
            "edge_kind": "cec_terminal_geometric_witness",
            "motion": value.audit_dict(),
        }

    def edge(self, **kwargs):
        value = _motion(1.0)
        return value, {
            "edge_kind": kwargs["edge_kind"],
            "from_frame": kwargs["reference_frame"],
            "to_frame": kwargs["query_frame"],
            "motion": value.audit_dict(),
        }

    def current_depth(self):
        return np.ones((2, 2)), np.ones((2, 2)), "unit_test_depth"

    agent._certified_terminal_route_motion = types.MethodType(terminal, agent)
    agent._certified_visual_route_edge = types.MethodType(edge, agent)
    agent._materialize_current_route_depth = types.MethodType(
        current_depth, agent)
    return agent, hashlib.sha256(anchor).hexdigest()


def _proof(anchor_sha):
    return {
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 40,
        "selected_anchor_image_sha256": anchor_sha,
        "certificate": {"accepted": True},
    }


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


def _initialize(agent, anchor_sha):
    return agent._certified_monocular_route_tangent_direction(
        goal_key="goal",
        target_anchor=40,
        goal_start_frame=64,
        goal_pose9=np.asarray(_pose9()),
        authority_proof=_proof(anchor_sha),
        goal_image_sha256="c" * 64,
    )


def test_minimal_overlay_issues_proof_bound_packet(tmp_path):
    agent, anchor_sha = _agent(tmp_path)
    direction, diagnostics = _initialize(agent, anchor_sha)

    np.testing.assert_allclose(direction, [-1.0, 0.0], atol=1e-12)
    packet = diagnostics["local_tangent_alignment_packet"]
    assert packet["target_anchor"] == 40
    assert packet["anchor_image_sha256"] == anchor_sha
    assert packet["current_rgb_sha256"] == "b" * 64
    assert diagnostics["local_tangent_alignment_packet_sha256"] == packet[
        "packet_sha256"]


def test_minimal_overlay_consumes_yaw_without_visual_edge(tmp_path):
    agent, anchor_sha = _agent(tmp_path)
    _direction, first = _initialize(agent, anchor_sha)
    packet = first["local_tangent_alignment_packet"]
    original_history_edges = first["local_tangent_history_edge_count"]

    yaw = np.deg2rad(30.0)
    agent.executor_motion_receipts.extend([
        _control_receipt(frame, yaw, packet["packet_sha256"])
        for frame in range(65, 71)
    ])
    agent.n = 71
    visual_calls = []

    def forbidden_visual_edge(self, **kwargs):
        visual_calls.append(kwargs)
        raise AssertionError("turn frames entered visual PnP")

    agent._certified_visual_route_edge = types.MethodType(
        forbidden_visual_edge, agent)
    direction, diagnostics = _initialize(agent, anchor_sha)

    np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-12)
    assert visual_calls == []
    assert diagnostics["local_tangent_history_edge_count"] == original_history_edges
    assert diagnostics["local_tangent_query_edge_count"] == 0
    assert diagnostics["local_tangent_update_edge_count"] == 6
    assert diagnostics["local_tangent_alignment_control_edge_count"] == 6
    assert diagnostics[
        "local_tangent_alignment_executor_receipt_consumed"] is True
    assert diagnostics["local_tangent_projected_progress_m"] == pytest.approx(0.0)


def test_minimal_overlay_rejects_partial_or_mixed_control_batch(tmp_path):
    agent, anchor_sha = _agent(tmp_path)
    _direction, first = _initialize(agent, anchor_sha)
    packet = first["local_tangent_alignment_packet"]
    agent.executor_motion_receipts.append(_control_receipt(
        65, np.deg2rad(30.0), packet["packet_sha256"]))
    agent.n = 66

    direction, diagnostics = _initialize(agent, anchor_sha)
    assert direction is None
    assert diagnostics["local_tangent_status"] == "geometry_failure"
    assert "before its proof-bound turn was complete" in diagnostics[
        "local_tangent_error"]
