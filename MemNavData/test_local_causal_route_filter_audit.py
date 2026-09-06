import numpy as np

from MemNavData.episodic_route_filter import (
    MonotoneRouteFilter,
    derive_route_transition,
)


def test_turn_then_translation_uses_action_conditioned_transition():
    positions = np.stack([
        0.02 * np.arange(100, dtype=np.float64),
        np.zeros(100, dtype=np.float64),
    ], axis=1)
    transition = derive_route_transition(
        positions, nominal_motion_m=0.30, maximum_search_frames=32)
    tracker = MonotoneRouteFilter(100, transition)
    flat = np.zeros(100, dtype=np.float64)
    for _ in range(6):
        assert tracker.update(flat, translated=False).state_index == 0
    assert tracker.update(flat, translated=True).state_index == 15
