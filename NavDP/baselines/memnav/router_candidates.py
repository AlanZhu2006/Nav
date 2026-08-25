"""Deterministic temporal diversification for visual retrieval candidates."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def causal_goal_support_indices(
    frame_count: int,
    *,
    candidate_frame_idx: int,
    stride: int,
    min_frame_gap: int,
) -> list[int]:
    """Sample only history that causally precedes a prospective goal.

    ``candidate_frame_idx`` is the number of memory frames that had already
    been committed when the candidate-only image was captured.  Freezing this
    boundary prevents both adjacent-video self matches at capture time and
    future frames recorded after capture from contaminating a later rescore.
    The exact causal ceiling is always included even when it is off-stride.
    """
    frame_count = int(frame_count)
    candidate_frame_idx = int(candidate_frame_idx)
    stride = int(stride)
    min_frame_gap = int(min_frame_gap)
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if candidate_frame_idx < 0 or candidate_frame_idx > frame_count:
        raise ValueError(
            "candidate_frame_idx must be within the committed history, got "
            f"{candidate_frame_idx} for {frame_count} frames"
        )
    if stride < 1 or min_frame_gap < 1:
        raise ValueError("stride and min_frame_gap must be positive")

    ceiling = min(frame_count - 1, candidate_frame_idx - min_frame_gap)
    if ceiling < 0:
        return []
    indices = list(range(0, ceiling + 1, stride))
    if indices[-1] != ceiling:
        indices.append(ceiling)
    return indices


def temporal_nms_candidates(
    scores: Sequence[float],
    eligible: Sequence[bool],
    *,
    top_k: int,
    min_frame_gap: int,
) -> list[dict]:
    """Rank by score, suppressing candidates from the same temporal cluster.

    Adjacent video frames are near duplicates.  A raw top-K can therefore be
    filled by one wrong location even when a correct loop-closure frame has a
    competitive score.  Ties are resolved by the lower frame index, matching
    ``torch.argmax`` on the chronological memory tensor.
    """
    if len(scores) != len(eligible):
        raise ValueError("scores and eligible must have the same length")
    if top_k < 1 or min_frame_gap < 1:
        raise ValueError("top_k and min_frame_gap must be positive")

    ranked = sorted(
        (
            (index, float(score))
            for index, (score, keep) in enumerate(zip(scores, eligible))
            if bool(keep) and math.isfinite(float(score))
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected: list[dict] = []
    selected_frames: list[int] = []
    for frame, score in ranked:
        if all(abs(frame - other) >= min_frame_gap
               for other in selected_frames):
            selected.append({"anchor": int(frame), "score": float(score)})
            selected_frames.append(frame)
            if len(selected) == top_k:
                break
    return selected
