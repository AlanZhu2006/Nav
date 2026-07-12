from typing import NamedTuple

import numpy as np


class RetrievalLabel(NamedTuple):
    pos_mask: np.ndarray
    neg_mask: np.ndarray
    candidate_mask: np.ndarray
    null_pos: bool


def build_retrieval_label(curve, kind, pos_hi, pos_lo, anchor_margin):
    """Build one retrieval label from generator metadata.

    ``kind`` is the semantic ground truth. A revisit without a strong positive is
    skipped instead of being relabeled as novel. This preserves the configured
    positive threshold and keeps the null class reserved for genuinely novel goals.
    """
    curve = np.asarray(curve, dtype=np.float32)
    if curve.ndim != 1 or curve.size == 0:
        return None, 'invalid_curve'
    if not np.isfinite(curve).all() or np.any((curve < 0.0) | (curve > 1.0)):
        return None, 'invalid_curve'
    if not 0.0 <= pos_lo < pos_hi <= 1.0:
        return None, 'invalid_thresholds'
    if anchor_margin < 0:
        return None, 'invalid_anchor_margin'

    candidate_mask = np.arange(curve.size) >= int(anchor_margin)
    if not candidate_mask.any():
        return None, 'no_valid_candidates'

    pos_mask = candidate_mask & (curve >= float(pos_hi))
    neg_mask = candidate_mask & (curve <= float(pos_lo))

    if kind == 'revisit':
        if not pos_mask.any():
            return None, 'weak_revisit'
        null_pos = False
    elif kind == 'novel':
        if pos_mask.any():
            return None, 'novel_has_positive'
        null_pos = True
    else:
        return None, 'unknown_goal_kind'

    return RetrievalLabel(pos_mask, neg_mask, candidate_mask, null_pos), None
