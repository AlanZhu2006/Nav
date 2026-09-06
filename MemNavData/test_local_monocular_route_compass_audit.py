import math

import numpy as np

from MemNavData.monocular_adjacent_motion import (
    integrate_planar_motion,
    PlanarMotionReceipt,
)
from MemNavData.run_local_monocular_route_compass_audit import (
    angle_between_degrees,
    history_route_contract,
    identity_motion,
    route_from_chronological,
)


def test_route_reconstruction_inverts_chronological_local_edges():
    motions = [
        PlanarMotionReceipt(1.0, 0.0, math.pi / 2.0, 0.0),
        PlanarMotionReceipt(1.0, 0.0, 0.0, 0.0),
    ]
    route = route_from_chronological(motions)
    assert route.shape == (3, 2)
    np.testing.assert_allclose(route[0], [0.0, 0.0])
    position = np.zeros(2)
    yaw = 0.0
    for motion in motions:
        position, yaw = integrate_planar_motion(position, yaw, motion)
    assert np.linalg.norm(route[-1]) == np.linalg.norm(position)


def test_angle_between_and_identity_receipt():
    assert angle_between_degrees(
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 90.0
    identity = identity_motion()
    assert identity.translation_m == 0.0
    assert identity.yaw_rad == 0.0


def test_history_route_contract_bridges_a_pre_receipt_anchor():
    assert history_route_contract(anchor=30, frame_count=100) == {
        "route_start_frame": 40,
        "pre_scale_anchor_bridge": True,
        "expected_edge_count": 60,
    }


def test_history_route_contract_starts_at_a_post_receipt_anchor():
    assert history_route_contract(anchor=67, frame_count=100) == {
        "route_start_frame": 67,
        "pre_scale_anchor_bridge": False,
        "expected_edge_count": 32,
    }
