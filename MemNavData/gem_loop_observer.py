"""Experimental causal loop evidence; no writes to GEM geometry or goal state.

MASt3R-SLAM provides a reference retriever and pair verifier. The caller owns
the native LingBot archive. A failed verifier never invokes another verifier.
"""
from dataclasses import dataclass

import numpy as np


class CausalKeyframePool:
    def __init__(self, reference_frames, *, exclude_recent=64):
        refs = tuple(reference_frames)
        if any(int(x) != x or x < 0 for x in refs) or tuple(sorted(set(refs))) != refs:
            raise ValueError('Reference identities must be sorted, unique nonnegative integers')
        if int(exclude_recent) != exclude_recent or exclude_recent < 1:
            raise ValueError('A positive causal exclusion is required')
        self.references = refs
        self.gap = int(exclude_recent)
        self.current = -1
        self.indexed = []

    def advance(self, current):
        if int(current) != current or current <= self.current:
            raise ValueError('Queries must advance strictly in observed time')
        self.current = int(current)
        start = len(self.indexed)
        while len(self.indexed) < len(self.references):
            frame = self.references[len(self.indexed)]
            if frame > self.current-self.gap:
                break
            self.indexed.append(frame)
        return self.indexed[start:]

    def resolve(self, database_ids):
        ids = list(database_ids)
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated retrieval identity')
        if any(int(i) != i or not 0 <= i < len(self.indexed) for i in ids):
            raise ValueError('Retrieved identity is outside the causal database')
        return [self.indexed[i] for i in ids]


@dataclass(frozen=True)
class PairRaster:
    """Official MASt3R resize/crop inverse, followed by native LingBot pad."""
    height: int
    width: int
    scale_x: float
    scale_y: float
    crop_x: float
    crop_y: float
    raw_width: int
    raw_height: int

    def pixels(self, indices):
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Pixel addresses must be integer indices')
        if np.any(indices < 0) or np.any(indices >= self.height*self.width):
            raise ValueError('Pixel address outside decoded raster')
        return np.stack((indices % self.width, indices // self.width), -1).astype(float)

    def raw(self, pixels):
        return (np.asarray(pixels)+[self.crop_x, self.crop_y])*[self.scale_x, self.scale_y]

    def lingbot(self, pixels):
        from MemNavData.lingbot_pnp_localization import map_raw_points_to_lingbot_pad
        return map_raw_points_to_lingbot_pad(self.raw(pixels), raw_height=self.raw_height,
            raw_width=self.raw_width, target_height=518, target_width=518, patch_size=14)


def official_loop_support(raw, *, confidence_threshold, minimum_fraction):
    """Same nonconsecutive-edge decision as official FactorGraph.add_factors.

    Raw output has one pair, in the official mast3r_match_symmetric order.
    Matching validity and descriptor confidence are distinct from depth_conf.
    """
    import torch
    idx_i2j, idx_j2i, valid_j, valid_i, Qii, Qjj, Qji, Qij = raw
    if idx_i2j.shape[0] != 1 or idx_j2i.shape[0] != 1:
        raise ValueError('One candidate pair per evidence record is required')
    batch = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[:, None].repeat(1, idx_i2j.shape[1])
    Qj = torch.sqrt(Qii[batch, idx_i2j]*Qji)
    Qi = torch.sqrt(Qjj[batch, idx_j2i]*Qij)
    keep_j = valid_j & (Qj > confidence_threshold)
    keep_i = valid_i & (Qi > confidence_threshold)
    fraction_j = keep_j.sum(dim=(1, 2))/(keep_j.shape[1]*keep_j.shape[2])
    fraction_i = keep_i.sum(dim=(1, 2))/(keep_i.shape[1]*keep_i.shape[2])
    accepted = bool((torch.minimum(fraction_i, fraction_j) >= minimum_fraction).item())
    return dict(accepted=accepted, reference_fraction=float(fraction_i.item()),
        current_fraction=float(fraction_j.item()), reference_valid=keep_i,
        current_valid=keep_j, reference_Q=Qi, current_Q=Qj)


def sample_current_grid(support, *, limit=2048):
    """Uniform deterministic subsample in current-pixel order, without scores.

    Only native LingBot/PnP transfer is capped; official verification uses the
    full dense set. This cap is a new adapter convention, not upstream SLAM.
    """
    mask = np.asarray(support, dtype=bool).reshape(-1)
    ids = np.flatnonzero(mask)
    if limit < 1 or int(limit) != limit:
        raise ValueError('Positive point budget required')
    if len(ids) <= limit:
        return ids
    return ids[np.linspace(0, len(ids)-1, limit, dtype=np.int64)]
