"""Own-state, failure-terminated episode lifecycle, independent of a controller.

Goal construction and role annotation are deliberately outside this module.
It cannot replay a prefix, reset a server, pick a replacement goal, or fill a
failed method's history from another method. Run the supplied fixed goals in
order and keep the initial episode denominator.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


def run_chain(start_position, start_yaw, goals, execute, observe=None):
    """Execute ``execute(index, goal, own_position, own_yaw)`` until failure.

    ``execute`` uses the existing run_policy_leg return contract. The caller
    initializes its servers once, before calling this function. ``observe``
    records full leg output before the next leg; exceptions are infrastructure
    errors and propagate rather than becoming measured navigation failures.
    """
    goals = deepcopy(list(goals))
    if not goals:
        raise ValueError("A continuous episode needs at least one goal")
    position = np.asarray(start_position, dtype=float).copy()
    yaw = float(start_yaw)
    if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError("Invalid initial physical state")
    records, alive, completed = [], True, 0
    previous_memory_index = None
    for index, goal in enumerate(goals):
        if not alive:
            records.append(dict(leg_index=index, attempted=False, reached=None,
                                status="not_attempted_after_previous_failure"))
            continue
        initial_position, initial_yaw = position.copy(), yaw
        leg = execute(index, deepcopy(goal), position.copy(), yaw)
        trace = leg["rollout_trace"]
        if trace:
            first = trace[0]
            np.testing.assert_allclose([first[k] for k in "xyz"], initial_position,
                                       atol=1e-8, rtol=0,
                                       err_msg="Leg did not inherit its own endpoint")
            delta = math.atan2(math.sin(first["yaw"] - initial_yaw),
                               math.cos(first["yaw"] - initial_yaw))
            if abs(delta) > 1e-8:
                raise ValueError("Leg did not inherit its own yaw")
        memory = [int(row["frame_idx"]) for row in leg["memory_trace"]
                  if row.get("frame_idx") is not None]
        if memory:
            if any(b != a + 1 for a, b in zip(memory, memory[1:])):
                raise ValueError("Noncontiguous memory within a leg")
            if previous_memory_index is not None and memory[0] != previous_memory_index + 1:
                raise ValueError("Memory was reset, replayed, or skipped at a goal switch")
            previous_memory_index = memory[-1]
        position = np.asarray(leg["end_pos"], dtype=float).copy()
        yaw = float(leg["end_psi"])
        if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
            raise ValueError("Invalid executed endpoint")
        reached = bool(leg["reached"])
        record = dict(leg_index=index, attempted=True, reached=reached,
                      status="success" if reached else "navigation_failure",
                      start_position=initial_position.tolist(), start_yaw=initial_yaw,
                      end_position=position.tolist(), end_yaw=yaw,
                      steps=int(leg["steps"]),
                      first_memory_index=memory[0] if memory else None,
                      last_memory_index=memory[-1] if memory else None,
                      termination_reason=leg.get("termination_reason"))
        if observe is not None:
            observe(index, deepcopy(goal), leg, deepcopy(record))
        records.append(record)
        completed += int(reached)
        alive = reached
    return dict(legs=records, goals_completed=completed,
                cumulative_success=[int(completed >= i + 1) for i in range(len(goals))],
                joint_success=bool(alive), initial_episode_denominator=1)


def summarize_chains(chains):
    """Aggregate an initial cohort without reselecting successful prefixes."""
    chains = list(chains)
    if not chains:
        raise ValueError("No completed episode records")
    stages = len(chains[0]["legs"])
    if any(len(c["legs"]) != stages for c in chains):
        raise ValueError("Different sequence lengths need separate summaries")
    rows = []
    for index in range(stages):
        legs = [c["legs"][index] for c in chains]
        attempted = sum(bool(r["attempted"]) for r in legs)
        successes = sum(r["reached"] is True for r in legs)
        rows.append(dict(leg_index=index, initial_n=len(chains), attempted=attempted,
                         success=successes, not_attempted=len(chains) - attempted,
                         cumulative_sr=successes / len(chains),
                         conditional_sr=successes / attempted if attempted else None))
    return dict(initial_n=len(chains), stages=rows,
                joint_sr=sum(c["joint_success"] for c in chains) / len(chains),
                mean_completed_goals=sum(c["goals_completed"] for c in chains) / len(chains))
