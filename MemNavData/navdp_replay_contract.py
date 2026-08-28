"""Pure validation for restoring decision frames to NavDP's bounded FIFO."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


class NavDPReplayContractError(RuntimeError):
    """A replay response cannot represent one append to a bounded FIFO."""


def validate_replay_queue_growth(
    responses: Sequence[Mapping[str, Any]],
    *,
    expected_initial_length: Optional[int] = None,
) -> dict[str, int]:
    """Validate monotone one-frame FIFO growth without assuming an empty FIFO.

    Goal-B is replayed after Goal-A, so its first decision frame may enter an
    already-full FIFO.  The old audit compared the final queue length with the
    number of Goal-B plans alone and incorrectly rejected valid short Goal-B
    traces.  This contract instead anchors on the first observed post-append
    length and checks every subsequent append up to the frozen memory bound.

    ``expected_initial_length=1`` remains available for callers that have just
    issued an audited short-memory reset (for example sealed-C replay).
    """

    if not responses:
        raise NavDPReplayContractError("replay produced no controller responses")
    lengths: list[int] = []
    memory_size: int | None = None
    for ordinal, response in enumerate(responses):
        if not isinstance(response, Mapping):
            raise NavDPReplayContractError(
                f"replay response {ordinal} is not a mapping")
        size = response.get("memory_size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise NavDPReplayContractError(
                f"replay response {ordinal} has invalid memory_size")
        if memory_size is None:
            memory_size = size
        elif size != memory_size:
            raise NavDPReplayContractError("NavDP memory_size changed during replay")
        queue_lengths = response.get("queue_lengths")
        if (not isinstance(queue_lengths, list) or len(queue_lengths) != 1
                or isinstance(queue_lengths[0], bool)
                or not isinstance(queue_lengths[0], int)):
            raise NavDPReplayContractError(
                f"replay response {ordinal} has invalid queue_lengths")
        length = queue_lengths[0]
        if not 1 <= length <= size:
            raise NavDPReplayContractError(
                f"replay response {ordinal} queue length is out of bounds")
        lengths.append(length)

    assert memory_size is not None
    initial = lengths[0]
    if (expected_initial_length is not None
            and initial != expected_initial_length):
        raise NavDPReplayContractError(
            "first replay append did not start from the expected FIFO state")
    expected = [
        min(initial + ordinal, memory_size)
        for ordinal in range(len(lengths))
    ]
    if lengths != expected:
        raise NavDPReplayContractError(
            f"NavDP replay queue growth changed: observed={lengths} "
            f"expected={expected}")
    return {
        "appends": len(lengths),
        "first_post_append_length": initial,
        "final_length": lengths[-1],
        "memory_size": memory_size,
    }


__all__ = ["NavDPReplayContractError", "validate_replay_queue_growth"]
