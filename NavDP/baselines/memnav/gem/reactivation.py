"""Observation-only LingBot packet re-estimation for episodic memory.

This module estimates geometry; it does not match goal images, change SP/LG,
or decide whether an image goal is accepted. Integration is explicit. The
SDPA state lease shares model weights and restores the live stream on errors.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PacketRequest:
    reference: int
    current: int
    historical_frames: tuple
    recent_frames: tuple
    similarity: float

    @property
    def frames(self):
        return self.historical_frames + self.recent_frames

    def __post_init__(self):
        if not self.historical_frames or not self.recent_frames:
            raise ValueError("Both observed packets must be nonempty")
        if tuple(sorted(set(self.frames))) != self.frames:
            raise ValueError("Packets must be chronological and disjoint")
        if self.reference not in self.historical_frames:
            raise ValueError("Reference must belong to the historical packet")
        if self.current != self.recent_frames[-1]:
            raise ValueError("Current must be the last observed frame")
        if self.frames[0] < 0 or not np.isfinite(self.similarity):
            raise ValueError("Invalid frame identity or retrieval score")


def retrieve_packet(descriptors, current, *, packet_size=8, exclude_recent=64):
    """One causal top-1 candidate, with ties resolved by oldest frame identity.

    The interface has no goal, pose, ground truth, or alternative-candidate
    input. The excluded suffix is a temporal separation, not an acceptance
    threshold. Low retrieval scores remain real requests for evaluation.
    """
    keys = np.asarray(descriptors, dtype=np.float32)
    if packet_size < 1 or exclude_recent < 2 * packet_size:
        raise ValueError("Temporal separation must keep the two packets disjoint")
    if keys.ndim != 2 or not 0 <= current < len(keys):
        raise ValueError("Current frame is missing from descriptor history")
    ceiling = current - exclude_recent
    if ceiling < packet_size - 1:
        return None
    eligible = keys[:ceiling + 1]
    query = keys[current]
    norms = np.linalg.norm(eligible, axis=1)
    query_norm = np.linalg.norm(query)
    if (not np.isfinite(eligible).all() or not np.isfinite(query).all()
            or query_norm == 0 or np.any(norms == 0)):
        raise ValueError("Descriptor history contains invalid keys")
    scores = eligible @ query / (norms * query_norm)
    reference = int(np.argmax(scores))
    start = max(0, reference - packet_size + 1)
    historical = tuple(range(start, start + packet_size))
    recent = tuple(range(current - packet_size + 1, current + 1))
    return PacketRequest(reference, current, historical, recent,
                         float(scores[reference]))


@contextmanager
def isolated_stream(model):
    """Borrow LingBot without copying the SDPA or GEM paged working state.

    Detach containers *before* clean_kv_cache, which mutates the aggregator
    dictionary. Restore the exact original containers and temporal counters.
    The caller serializes all access to this model, including ordinary writes.
    External hooks owned by a caller must be restored by that caller.
    """
    agg, camera = model.aggregator, model.camera_head
    paged = getattr(agg, 'gem_paged_memory', False)
    if (not agg.use_sdpa and not paged) or (agg.use_sdpa and agg.kv_cache_manager is not None):
        raise ValueError("State isolation requires native SDPA or GEM paged memory")
    agg_names = ("kv_cache", "total_frames_processed", "_cached_pos3d", "kv_cache_manager")
    cam_names = ("kv_cache", "frame_idx", "pos_cache")
    saved_agg = {key: getattr(agg, key) for key in agg_names}
    saved_camera = {key: getattr(camera, key) for key in cam_names}
    # SDPA indexes each layer directly, so preserve its initialized key schema.
    # Values are detached; clean_kv_cache sets the skip flag to False.
    agg.kv_cache = {key: None for key in saved_agg['kv_cache']}
    agg.kv_cache_manager = None
    camera.kv_cache = None
    camera.pos_cache = None
    try:
        model.clean_kv_cache()
        yield
    finally:
        try:
            model.clean_kv_cache()
        finally:
            for key, value in saved_agg.items():
                setattr(agg, key, value)
            for key, value in saved_camera.items():
                setattr(camera, key, value)


class LingBotReactivation:
    """Estimate a short historical/recent packet using one frozen backbone."""

    def __init__(self, model, *, num_scale_frames=8):
        self.model = model
        self.num_scale_frames = int(num_scale_frames)

    def estimate(self, images, request):
        import torch

        if len(images) != len(request.frames):
            raise ValueError("RGB packet does not match the requested frame identities")
        if len(request.historical_frames) != self.num_scale_frames:
            raise ValueError("Historical packet must match scale initialization")
        with isolated_stream(self.model), torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16):
            predictions = self.model.inference_streaming(
                images, num_scale_frames=self.num_scale_frames,
                keyframe_interval=1, output_device=torch.device("cpu"))
        return predictions


def fit_similarity(source, target):
    """Fit a proper Sim(3) to known same-pixel 3-D correspondences.

    Invalid or degenerate observations raise, with no replacement transform.
    Residuals diagnose consistency, not accuracy or independent confidence.
    """
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("At least three paired 3-D points are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite paired geometry")
    xc, yc = x - x.mean(0), y - y.mean(0)
    variance = np.mean(np.sum(xc * xc, axis=1))
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    if variance <= np.finfo(float).eps or singular[1] <= singular[0] * 1e-10:
        raise ValueError("Degenerate similarity alignment")
    sign = np.array([1., 1., np.linalg.det(u @ vt)])
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / variance)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid similarity scale")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = y.mean(0) - scale * rotation @ x.mean(0)
    residuals = np.linalg.norm(x @ transform[:3, :3].T + transform[:3, 3] - y, axis=1)
    return transform, dict(scale=scale, count=len(x),
        rms=float(np.sqrt(np.mean(residuals ** 2))),
        normalized_rms=float(np.sqrt(np.mean(residuals ** 2) / variance)),
        singular_values=singular.tolist())
