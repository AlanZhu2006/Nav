"""Deterministic temporal diversification for visual retrieval candidates."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


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
