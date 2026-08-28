import pytest

from MemNavData.navdp_replay_contract import (
    NavDPReplayContractError,
    validate_replay_queue_growth,
)


def response(length: int, size: int = 8) -> dict:
    return {
        "diffusion_sampled": False,
        "memory_size": size,
        "queue_lengths": [length],
    }


def test_empty_fifo_growth():
    audit = validate_replay_queue_growth(
        [response(1), response(2), response(3)],
        expected_initial_length=1,
    )
    assert audit == {
        "appends": 3,
        "first_post_append_length": 1,
        "final_length": 3,
        "memory_size": 8,
    }


def test_goal_b_can_start_from_full_goal_a_fifo():
    audit = validate_replay_queue_growth([response(8) for _ in range(6)])
    assert audit["appends"] == 6
    assert audit["first_post_append_length"] == 8
    assert audit["final_length"] == 8


def test_partially_filled_fifo_saturates_at_memory_bound():
    validate_replay_queue_growth(
        [response(value) for value in (6, 7, 8, 8, 8)])


@pytest.mark.parametrize(
    "responses",
    [
        [],
        [response(3), response(5)],
        [response(1), response(2, size=7)],
        [{"memory_size": 8, "queue_lengths": []}],
        [{"memory_size": 8, "queue_lengths": [0]}],
    ],
)
def test_invalid_growth_fails_closed(responses):
    with pytest.raises(NavDPReplayContractError):
        validate_replay_queue_growth(responses)


def test_reset_expectation_is_enforced():
    with pytest.raises(NavDPReplayContractError, match="expected FIFO"):
        validate_replay_queue_growth(
            [response(8), response(8)], expected_initial_length=1)
