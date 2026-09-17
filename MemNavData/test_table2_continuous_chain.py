from copy import deepcopy

import numpy as np
import pytest

from MemNavData.table2_continuous_chain import run_chain, summarize_chains


def executor(outcomes, *, length=2, offset=0):
    calls = []

    def execute(index, goal, position, yaw):
        calls.append((index, goal, position.copy(), yaw))
        first = dict(step=0, x=position[0], y=position[1], z=position[2], yaw=yaw)
        return dict(reached=outcomes[index], steps=length, rollout_trace=[first],
                    end_pos=position + [index + 1 + offset, 0, 0], end_psi=yaw + .1,
                    termination_reason="position_success" if outcomes[index] else "budget",
                    memory_trace=[dict(frame_idx=length * index + j) for j in range(length)])

    return execute, calls


@pytest.mark.parametrize("outcomes,attempts,cumulative", [
    ([False, True, True], 1, [0, 0, 0]),
    ([True, False, True], 2, [1, 0, 0]),
    ([True, True, False], 3, [1, 1, 0]),
    ([True, True, True], 3, [1, 1, 1]),
])
def test_own_state_staircase(outcomes, attempts, cumulative):
    execute, calls = executor(outcomes)
    result = run_chain([0, 0, 0], 0, [dict(goal=i) for i in range(3)], execute)
    assert len(calls) == attempts
    assert result["cumulative_success"] == cumulative
    assert result["goals_completed"] == sum(cumulative)
    for index in range(1, len(calls)):
        np.testing.assert_allclose(calls[index][2], result["legs"][index-1]["end_position"])
        assert calls[index][3] == result["legs"][index-1]["end_yaw"]
    for leg in result["legs"][attempts:]:
        assert leg["attempted"] is False and leg["reached"] is None


def test_two_arms_keep_different_histories_and_survival():
    native, ncalls = executor([True, False, True])
    gem, gcalls = executor([True, True, True], offset=10)
    n = run_chain([0, 0, 0], 0, [1, 2, 3], native)
    g = run_chain([0, 0, 0], 0, [1, 2, 3], gem)
    assert n["joint_success"] is False and g["joint_success"] is True
    assert ncalls[1][2][0] == 1 and gcalls[1][2][0] == 11
    assert gcalls[2][2][0] == 23


@pytest.mark.parametrize("field", ["position", "yaw", "memory"])
def test_rejects_transplant_or_reset(field):
    original, _ = executor([True] * 3)

    def invalid(index, *args):
        leg = original(index, *args)
        if index == 1:
            if field == "position":
                leg["rollout_trace"][0]["x"] = -99
            elif field == "yaw":
                leg["rollout_trace"][0]["yaw"] = 3
            else:
                leg["memory_trace"] = [dict(frame_idx=0)]
        return leg

    with pytest.raises((ValueError, AssertionError)):
        run_chain([0, 0, 0], 0, [1, 2, 3], invalid)


def test_no_goal_mutation_or_replacement():
    goals = [dict(value=i) for i in range(3)]
    original_goals = deepcopy(goals)
    execute, calls = executor([True] * 3)

    def mutating(index, goal, *args):
        result = execute(index, deepcopy(goal), *args)
        goal["value"] = -1
        return result

    run_chain([0, 0, 0], 0, goals, mutating)
    assert goals == original_goals
    assert [call[1] for call in calls] == original_goals


def test_infrastructure_failure_not_converted_to_navigation_failure():
    def broken(*args):
        raise ConnectionError("server disconnected")

    with pytest.raises(ConnectionError):
        run_chain([0, 0, 0], 0, [1, 2, 3], broken)


def test_fixed_denominator_and_conditional_counts():
    outcomes = ([False, True, True], [True, False, True],
                [True, True, False], [True, True, True])
    chains = [run_chain([0, 0, 0], 0, [1, 2, 3], executor(o)[0]) for o in outcomes]
    result = summarize_chains(chains)
    assert result["initial_n"] == 4
    assert [r["success"] for r in result["stages"]] == [3, 2, 1]
    assert [r["attempted"] for r in result["stages"]] == [4, 3, 2]
    assert [r["cumulative_sr"] for r in result["stages"]] == [.75, .5, .25]
    assert [r["conditional_sr"] for r in result["stages"]] == [.75, 2/3, .5]
    assert result["joint_sr"] == .25
    assert result["mean_completed_goals"] == 1.5
