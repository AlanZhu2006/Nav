import pytest

from MemNavData.gem_bearing_attribution import InitialTurnIntervention
from MemNavData.revisit_bearing_adapter import adapt_revisit_pointgoal


def decision(point, accepted=True):
    return adapt_revisit_pointgoal(mode="verified_bearing_v1", router_active=accepted,
                                  pointgoal=point, pointgoal_units="lingbot_raw_direction_only")


def test_initial_turn_preserved_and_native_begins_after_completion():
    state = InitialTurnIntervention("initial_turn_only")
    rear = decision((-1., .1))
    assert state.adapt(rear) is rear
    assert not state.native and state.phase == "turning"
    state.completed(37)
    assert state.native and state.cutoff_action == 38
    with pytest.raises(RuntimeError, match="Sparse control"):
        state.adapt(rear)
    with pytest.raises(RuntimeError, match="another guided turn"):
        state.completed(80)


def test_full_gem_remains_unchanged_after_one_or_more_turns():
    state = InitialTurnIntervention("full_gem")
    rear = decision((-1., .1))
    assert state.adapt(rear) is rear
    state.completed(37)
    assert not state.native
    front = decision((1., 0.))
    assert state.adapt(front) is front
    assert state.adapt(rear) is rear
    state.completed(100)
    assert state.completed_turns == 2 and state.cutoff_action == 38


@pytest.mark.parametrize("cue", [decision((1., .2)), decision((-1., 0.), accepted=False)])
def test_no_initial_turn_does_not_execute_a_free_mixed_segment(cue):
    state = InitialTurnIntervention("initial_turn_only")
    result = state.adapt(cue)
    assert state.native and state.cutoff_action == 0
    assert not result.takeover and result.controller_pointgoal is None
    assert result.controller_distance_m is None
    assert result.raw_pointgoal == cue.raw_pointgoal
    assert result.audit_dict()["revisit_adapter_takeover"] is False


def test_invalid_arm_and_unissued_turn_are_rejected():
    with pytest.raises(ValueError):
        InitialTurnIntervention("conditional_fallback")
    state = InitialTurnIntervention("initial_turn_only")
    with pytest.raises(RuntimeError, match="not authorized"):
        state.completed(30)
