"""Pure geometry contract for the controlled Table-III causal survey."""

from __future__ import annotations

import math

import numpy as np


def wrap(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def densify(points: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    output = []
    for first, second in zip(points[:-1], points[1:]):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        segment = second - first
        length = float(np.linalg.norm(segment))
        count = max(1, int(math.ceil(length / step_m)))
        output.extend(first + segment * (index / count) for index in range(count))
    output.append(np.asarray(points[-1], dtype=np.float64))
    return output


def yaw_facing(delta_xz: np.ndarray) -> float:
    dx, dz = np.asarray(delta_xz, dtype=np.float64)
    return float(np.arctan2(-dx, -dz))


def survey_frames(
    points: list[np.ndarray], *, step_m: float, maximum_yaw_step_deg: float,
) -> list[tuple[np.ndarray, float]]:
    dense = [np.asarray(point, dtype=np.float64)
             for point in densify(points, step_m)]
    compact = [dense[0]]
    for point in dense[1:]:
        if float(np.linalg.norm(point - compact[-1])) > 1e-6:
            compact.append(point)
    if len(compact) < 2:
        raise ValueError("survey geodesic has no motion")
    maximum_yaw_step = math.radians(maximum_yaw_step_deg)
    first_yaw = yaw_facing((compact[1] - compact[0])[[0, 2]])
    frames: list[tuple[np.ndarray, float]] = [(compact[0], first_yaw)]
    current_yaw = first_yaw
    for previous, point in zip(compact[:-1], compact[1:]):
        desired = yaw_facing((point - previous)[[0, 2]])
        delta = wrap(desired - current_yaw)
        turns = int(math.ceil(abs(delta) / maximum_yaw_step))
        for turn in range(1, turns + 1):
            yaw = wrap(current_yaw + delta * turn / turns)
            frames.append((previous.copy(), yaw))
        current_yaw = desired
        frames.append((point.copy(), current_yaw))
    for (first_position, first_yaw), (second_position, second_yaw) in zip(
        frames[:-1], frames[1:]
    ):
        if float(np.linalg.norm(second_position - first_position)) > step_m + 1e-5:
            raise ValueError("survey translation step changed")
        if abs(math.degrees(wrap(second_yaw - first_yaw))) > maximum_yaw_step_deg + 1e-6:
            raise ValueError("survey yaw step changed")
    return frames


__all__ = ["survey_frames", "wrap"]
