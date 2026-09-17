"""Scoring-only pose-neighbour audit of frozen visual-memory observations.

Position/yaw proximity is an observation-coverage proxy, not co-visibility or
localizability. Evaluator poses must never select runtime memory references.
This module loads no learned model and sends no requests to a robot or service.
"""
from __future__ import annotations

import numpy as np


def coverage_panel(history, query, eligible, candidates, *, radii, yaws):
    """Audit the complete eligible set and a previously frozen shortlist.

    Poses are [floor x, floor y, floor z, yaw radians] in one scene frame.
    Full 3-D distances avoid accidentally treating different floors as nearby.
    """
    history = np.asarray(history, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if history.ndim != 2 or history.shape[1] != 4 or query.shape != (4,):
        raise ValueError('Expected [N,4] history and [4] query poses')
    if not np.isfinite(history).all() or not np.isfinite(query).all():
        raise ValueError('Pose coverage requires finite evaluator poses')
    if (eligible.ndim != 1 or candidates.ndim != 1 or len(np.unique(eligible)) != len(eligible)
            or len(np.unique(candidates)) != len(candidates)
            or np.any(eligible < 0) or np.any(eligible >= len(history))
            or not np.isin(candidates, eligible).all()):
        raise ValueError('Candidate identities must belong to the complete eligible set')
    radii, yaws = list(radii), list(yaws)
    if (not radii or not yaws or any(not np.isfinite(r) or r <= 0 for r in radii)
            or any(not np.isfinite(a) or not 0 < a <= 180 for a in yaws)):
        raise ValueError('Invalid reporting radii or yaw angles')
    distance = np.linalg.norm(history[eligible, :3] - query[:3], axis=1)
    dyaw = history[eligible, 3] - query[3]
    yaw = np.degrees(np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))))
    selected = np.isin(eligible, candidates)
    nearest = None
    if len(eligible):
        index = int(np.argmin(distance))
        nearest = dict(frame=int(eligible[index]), distance_3d_m=float(distance[index]),
            yaw_difference_deg=float(yaw[index]))
    panel = {}
    for radius in radii:
        near = distance <= radius
        panel[str(radius)] = dict(nearby_history_frames=int(near.sum()),
            nearby_shortlist_frames=int((near & selected).sum()),
            minimum_nearby_yaw_deg=float(yaw[near].min()) if near.any() else None,
            angles={str(angle): dict(history_frames=int((near & (yaw <= angle)).sum()),
                shortlist_frames=int((near & (yaw <= angle) & selected).sum()))
                for angle in yaws})
    return dict(eligible_frames=len(eligible), shortlist_frames=len(candidates),
        nearest=nearest, by_radius=panel)
