"""Reverse a recorded pose stream into short, metric memory subgoals.

The image-goal localizer identifies an old memory node.  Driving directly to
that node can cut through walls when the recorded path bends.  This module
resamples the *already traversed* pose chain in reverse order, so a local
controller can follow short point goals until it reaches the localized node.

The implementation is deliberately model agnostic and contains no Habitat
ground truth.  It uses only LingBot's causal camera translations and the same
per-episode metric scale used by the pose-goal controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reverse_metric_nodes(
        translations: np.ndarray, *, start_index: int, anchor_index: int,
        metric_scale: float, spacing_m: float) -> tuple[int, ...]:
    """Return temporally ordered nodes from ``start`` back to ``anchor``.

    Consecutive LingBot translations are accumulated along the recorded path,
    rather than replacing the path with one long Euclidean chord.  The anchor
    is always the last node, including when the remaining path is shorter than
    ``spacing_m``.
    """
    translations = np.asarray(translations, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] < 3:
        raise ValueError("translations must have shape [frames, >=3]")
    if not np.isfinite(translations).all():
        raise ValueError("translations must be finite")
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        raise ValueError("metric_scale must be finite and positive")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if not 0 <= anchor_index <= start_index < len(translations):
        raise ValueError(
            "indices must satisfy 0 <= anchor_index <= start_index < frames")
    if anchor_index == start_index:
        return ()

    # LingBot's planar motion lives in native x-z.  Metric scale converts the
    # canonical translation gauge into metres for this episode.
    planar = translations[:, (0, 2)]
    nodes: list[int] = []
    accumulated_m = 0.0
    previous = int(start_index)
    for index in range(start_index - 1, anchor_index - 1, -1):
        accumulated_m += float(metric_scale) * float(np.linalg.norm(
            planar[previous] - planar[index]))
        previous = index
        if accumulated_m >= spacing_m or index == anchor_index:
            nodes.append(index)
            accumulated_m = 0.0
    if not nodes or nodes[-1] != anchor_index:
        nodes.append(anchor_index)
    return tuple(nodes)


@dataclass
class ReverseRouteProgress:
    """Monotone cursor over a frozen reverse-memory route."""

    anchor_index: int
    start_index: int
    nodes: tuple[int, ...]
    cursor: int = 0

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.nodes)

    @property
    def current_node(self) -> int | None:
        return None if self.complete else int(self.nodes[self.cursor])

    def accept_distance(self, distance_m: float, arrival_m: float) -> bool:
        """Advance exactly one node when the current subgoal is reached."""
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("distance_m must be finite and non-negative")
        if not np.isfinite(arrival_m) or arrival_m <= 0.0:
            raise ValueError("arrival_m must be finite and positive")
        if not self.complete and distance_m <= arrival_m:
            self.cursor += 1
            return True
        return False
