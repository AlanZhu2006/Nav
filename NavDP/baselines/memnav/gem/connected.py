"""Causal LingBot episode memory connected by identical observed cameras.

One frozen model writes a bounded episode: eight observed initialization views
and eight subsequent committed views. All intervening RGB frames are predicted.
Before the next episode, its eight shared
views are re-expressed together. Camera identity determines rotation/position
alignment, and same-pixel predicted depth determines scale. There is no image
association, target input, geometric acceptance threshold, or alternative
writer. This candidate must be evaluated separately from long-gap retrieval.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .bindings import immutable_similarity


def pose_matrices(poses):
    poses = np.asarray(poses, dtype=np.float64)
    matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:7]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def align_same_cameras(old_poses, new_poses, old_depth, new_depth, valid_pixels):
    """Map a new local episode into the preceding episode's coordinates.

    Depth is camera-z in both predictions of the *same RGB pixel*. Scale is
    fit per shared frame, then averaged with one vote per frame. No scale
    ratio is inferred from a possibly zero camera translation. Rotation uses
    camera orientations, so depth distortion cannot translate/rotate a known
    camera center as it can in unconstrained point-cloud alignment.
    """
    old, new = np.asarray(old_poses), np.asarray(new_poses)
    old_z, new_z = np.asarray(old_depth), np.asarray(new_depth)
    if old.shape != new.shape or old.ndim != 3 or old.shape[1:] != (4, 4):
        raise ValueError('Shared cameras must have paired 4x4 poses')
    if old_z.shape != new_z.shape or len(old_z) != len(old):
        raise ValueError('Shared depth must have paired frame and pixel identities')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Nonfinite shared camera geometry')
    relative_rotations = old[:, :3, :3] @ new[:, :3, :3].transpose(0, 2, 1)
    u, _, vt = np.linalg.svd(relative_rotations.sum(0))
    sign = np.array([1., 1., np.linalg.det(u @ vt)])
    rotation = (u * sign) @ vt
    scales, counts, residuals = [], [], []
    for a, b in zip(old_z, new_z):
        valid = np.asarray(valid_pixels) & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if not np.any(valid):
            raise ValueError('A shared frame has no finite positive paired depth')
        aa, bb = a[valid].astype(np.float64), b[valid].astype(np.float64)
        scale = float(aa @ bb / (bb @ bb))
        scales.append(scale)
        counts.append(int(valid.sum()))
        residuals.append(float(np.linalg.norm(aa-scale*bb)/np.linalg.norm(aa)))
    scale = float(np.mean(scales))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Shared geometry has no valid positive scale')
    translation = np.mean(old[:, :3, 3] - scale * new[:, :3, 3] @ rotation.T, axis=0)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    transform = immutable_similarity(transform)
    position_residual = old[:, :3, 3] - (scale * new[:, :3, 3] @ rotation.T + translation)
    rotation_cos = (np.einsum('nij,ij->n', relative_rotations, rotation) - 1) / 2
    return transform, dict(scale=scale, per_frame_scale=scales, valid_pixels=counts,
        scale_log_std=float(np.std(np.log(scales))),
        normalized_depth_residual=residuals,
        camera_rotation_residual_deg=np.degrees(np.arccos(np.clip(rotation_cos, -1, 1))).tolist(),
        camera_position_rms=float(np.sqrt(np.mean(np.sum(position_residual**2, axis=1)))))


@dataclass(frozen=True)
class GeometryWrite:
    frames: tuple
    local_pose9: np.ndarray
    world_pose9: np.ndarray
    world_scale: float
    predictions: dict
    episode: int


class ConnectedEpisodeMemory:
    """One live LingBot cache, immutable first-observation geometry, no goals.

    Caller owns RGB/depth archives and existing SP/LG readout. ``write`` returns
    predictions for the new observations only; replayed overlap predictions
    are used for a coordinate relation and never overwrite historical pixels.
    Shared views and all model state are bounded by the episode length.
    """

    def __init__(self, model, sample_uv, valid_pixels, *, overlap=8, stride=8,
                 keyframe_interval=7, transport='camera_ols', bounded=True):
        if overlap != 8 or stride != 8:
            raise ValueError('This sealed candidate uses eight shared and eight new views')
        self.model = model
        if not model.aggregator.use_sdpa and (
                bounded or not getattr(model.aggregator, 'gem_paged_memory', False)):
            raise ValueError('Memory requires native SDPA or the explicit GEM paged writer')
        self.uv = np.asarray(sample_uv, dtype=int)
        self.valid_pixels = np.asarray(valid_pixels, dtype=bool)
        if self.uv.ndim != 2 or self.uv.shape[1] != 2 or self.valid_pixels.shape != (len(self.uv),):
            raise ValueError('Invalid sampled pixel identities')
        self.overlap, self.stride = overlap, stride
        if keyframe_interval < 1:
            raise ValueError('Keyframe interval must be positive')
        self.keyframe_interval = int(keyframe_interval)
        if transport not in ('camera_ols', 'reciprocal'):
            raise ValueError('Unknown coordinate transport')
        self.transport = transport
        self.bounded = bool(bounded)
        self.failed = None
        self.images = deque(maxlen=overlap)
        self.geometry = deque(maxlen=overlap)
        self.frame_count = 0
        self.episode = 0
        self.new_in_episode = 0
        self.world_from_episode = np.eye(4)
        self.relations = []
        self.last_overlap = None
        model.clean_kv_cache()

    def _forward(self, images, count, *, commit=True):
        import torch

        self.model._set_skip_append(not commit)
        torch.compiler.cudagraph_mark_step_begin()
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                predictions = self.model(images.unsqueeze(0).to(next(self.model.parameters()).device),
                    num_frame_for_scale=self.overlap, num_frame_per_block=count, causal_inference=True)
        finally:
            self.model._set_skip_append(False)
        return predictions

    def _sample(self, predictions):
        poses = predictions['pose_enc'][0].float().cpu().numpy()
        x, y = self.uv[:, 0], self.uv[:, 1]
        depth = predictions['depth'][0, :, y, x, 0].float().cpu().numpy()
        return poses, depth

    def _rebase(self):
        import torch

        assert len(self.images) == len(self.geometry) == self.overlap
        old_pose = np.stack([row[1] for row in self.geometry])
        old_depth = np.stack([row[2] for row in self.geometry])
        frames = [row[0] for row in self.geometry]
        self.model.clean_kv_cache()
        predictions = self._forward(torch.stack(list(self.images)), self.overlap)
        new_pose9, new_depth = self._sample(predictions)
        new_pose = pose_matrices(new_pose9)
        if self.transport == 'reciprocal':
            from .reciprocal import reciprocal_camera_transport
            transport = reciprocal_camera_transport
        else:
            transport = align_same_cameras
        old_from_new, audit = transport(old_pose, new_pose, old_depth, new_depth, self.valid_pixels)
        self.last_overlap = dict(shared_frames=np.asarray(frames), old_poses=old_pose,
            new_poses=new_pose, old_pose9=np.stack([row[3] for row in self.geometry]),
            new_pose9=new_pose9, old_depth=old_depth, new_depth=new_depth)
        self.world_from_episode = immutable_similarity(self.world_from_episode @ old_from_new)
        self.episode += 1
        self.new_in_episode = 0
        self.relations.append(dict(episode=self.episode, published_before_frame=self.frame_count,
            shared_frames=frames, old_from_new=old_from_new.tolist(),
            world_from_episode=self.world_from_episode.tolist(), **audit))
        # The rolling working geometry is expressed in the new episode; the
        # caller's immutable first-write records are not touched.
        self.geometry = deque(zip(frames, new_pose, new_depth, new_pose9), maxlen=self.overlap)
        del predictions

    def write(self, frame, image):
        if self.failed is not None:
            raise RuntimeError('The geometry stream failed; reset the episode before continuing')
        if frame != self.frame_count:
            raise ValueError('Only the next observed RGB frame can be written')
        if tuple(image.shape) != (3, 518, 518):
            raise ValueError('Use the frozen LingBot 518 pad preprocessing')
        try:
            return self._write(frame, image)
        except BaseException as error:
            # Neural cache writes cannot in general be rolled back after a
            # failed kernel. Never continue from a partially consumed frame.
            self.failed = f'{type(error).__name__}: {error}'
            raise

    def _write(self, frame, image):
        import torch

        commit = frame < self.overlap or (frame-self.overlap) % self.keyframe_interval == 0
        if self.bounded and self.frame_count >= self.overlap and commit and self.new_in_episode == self.stride:
            self._rebase()
        if not self.bounded and self.model.aggregator.total_frames_processed >= 320:
            raise RuntimeError('Native comparison reached the 320 committed-view protocol limit')
        if commit:
            self.images.append(image.detach().cpu())
        self.frame_count += 1
        if self.frame_count < self.overlap:
            return None
        if self.frame_count == self.overlap:
            frames = tuple(range(self.overlap))
            predictions = self._forward(torch.stack(list(self.images)), self.overlap)
        else:
            frames = (frame,)
            predictions = self._forward(image.unsqueeze(0), 1, commit=commit)
            self.new_in_episode += int(commit)
        local_pose9, depth = self._sample(predictions)
        local_poses = pose_matrices(local_pose9)
        world_poses = self.world_from_episode @ local_poses
        scale = float(np.cbrt(np.linalg.det(self.world_from_episode[:3, :3])))
        world_pose9 = local_pose9.astype(np.float64).copy()
        world_pose9[:, :3] = world_poses[:, :3, 3]
        world_pose9[:, 3:7] = Rotation.from_matrix(world_poses[:, :3, :3]/scale).as_quat()
        if commit:
            for identity, pose, z, pose9 in zip(frames, local_poses, depth, local_pose9):
                self.geometry.append((identity, pose, z, pose9))
        if self.bounded:
            assert self.model.aggregator.total_frames_processed <= self.overlap+self.stride
        return GeometryWrite(frames, local_pose9, world_pose9, scale, predictions, self.episode)
