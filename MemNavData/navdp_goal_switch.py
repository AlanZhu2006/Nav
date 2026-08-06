"""Causal controls for NavDP state at an ImageGoal switch.

LingBot/MemNav owns the long-term episode memory.  NavDP separately keeps a
bounded FIFO of recent decision observations.  These helpers reset only that
NavDP FIFO, so a goal-switch ablation cannot accidentally erase the long-term
memory that the revisit leg is intended to test.
"""

from __future__ import annotations

from typing import Callable


RESET_MODES = ("carry", "before_b", "every_goal")


def should_reset_before_leg(mode: str, leg_index: int) -> bool:
    """Return whether the NavDP FIFO is reset before the requested leg.

    Leg indices are zero based: A=0, B=1, C=2.  ``before_b`` deliberately
    isolates the Novel A->B transition; ``every_goal`` is the broader follow-up
    ablation and resets before both B and C.
    """
    if mode not in RESET_MODES:
        raise ValueError(f"unknown NavDP goal-switch mode: {mode!r}")
    if leg_index < 1:
        return False
    return mode == "every_goal" or (mode == "before_b" and leg_index == 1)


def navdp_server_base(
    server_backend: str,
    base_url: str,
    novel_base_url: str | None,
) -> str:
    """Select the server that owns the frozen NavDP local controller."""
    if server_backend == "navdp":
        return base_url
    if server_backend in ("hybrid_oracle", "hybrid_pose"):
        if novel_base_url is None:
            raise ValueError(f"{server_backend} requires a NavDP novel server")
        return novel_base_url
    raise ValueError(
        "short-memory-only reset is unavailable for a standalone MemNav server"
    )


def reset_navdp_short_memory(
    post: Callable,
    server_backend: str,
    base_url: str,
    novel_base_url: str | None,
    env_id: int = 0,
) -> dict:
    """Clear only NavDP's recent-observation FIFO through its HTTP endpoint."""
    navdp_base = navdp_server_base(server_backend, base_url, novel_base_url)
    response = post(
        f"{navdp_base}/navigator_reset_env",
        json={"env_id": int(env_id)},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("algo") != "navdp":
        raise RuntimeError(
            "short-memory reset reached a non-NavDP server: "
            f"{payload.get('algo')!r}"
        )
    return payload
