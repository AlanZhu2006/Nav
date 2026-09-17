"""Lossless, write-once CPU depth records, resolved in the memory coordinate frame."""
from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np


class FrameDepthArchive(Mapping):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = {}
        self.array_bytes = 0
        self.disk_bytes = 0

    def append(self, frame, depth, confidence, world_scale):
        if isinstance(frame, bool) or int(frame) != frame or frame != len(self.records):
            raise ValueError('Depth records must be appended in frame order')
        depth = np.asarray(depth, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        scale = float(world_scale)
        if depth.ndim != 2 or depth.shape != confidence.shape:
            raise ValueError('Depth and confidence must describe the same pixels')
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Depth requires a finite positive coordinate scale')
        path = self.directory / f'{frame:06d}.npz'
        # Opening exclusively prevents a retry from overwriting first-write
        # evidence, including after a process failure during serialization.
        with path.open('xb') as stream:
            np.savez_compressed(stream, frame=np.array(frame), depth=depth,
                confidence=confidence, world_scale=np.array(scale))
        content = path.read_bytes()
        self.records[frame] = dict(sha256=hashlib.sha256(content).hexdigest(),
            shape=depth.shape, scale=scale, bytes=len(content))
        self.array_bytes += depth.nbytes + confidence.nbytes
        self.disk_bytes += len(content)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, frame):
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise KeyError(frame)
        record = self.records[frame]
        path = self.directory / f'{frame:06d}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'Historical depth record {frame} changed')
        with np.load(path, allow_pickle=False) as a:
            if int(a['frame']) != frame or float(a['world_scale']) != record['scale']:
                raise RuntimeError('Depth record identity or coordinate scale changed')
            # Confidence is dimensionless. Both the selected reference pose
            # and the unprojected depth use the same first-episode units.
            return a['depth'] * np.float32(record['scale']), a['confidence'].copy()
