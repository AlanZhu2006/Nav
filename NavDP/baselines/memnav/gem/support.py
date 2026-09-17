"""Archive exactly the depth samples accessible to a frozen point detector.

The four bilinear neighbours of every historical SuperPoint keypoint are
retained, together with the actual global minimum-confidence pixel. With the
existing zero-quantile depth filter this preserves all lifted correspondences
for any future matching subset. Unstored pixels are explicitly unknown.

This experimental storage primitive does not estimate or revise geometry,
change SuperPoint/LightGlue, select goals, or shorten the neural context.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path
import time
from weakref import ref

import numpy as np


def pixel_support(points_xy, shape):
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Finite detector coordinates with shape[N,2] are required')
    height, width = map(int, shape)
    if min(height,width) < 2:
        raise ValueError('The original depth raster is required')
    x0 = np.clip(np.floor(points[:,0]).astype(np.int64),0,width-1)
    y0 = np.clip(np.floor(points[:,1]).astype(np.int64),0,height-1)
    x1, y1 = np.minimum(x0+1,width-1), np.minimum(y0+1,height-1)
    return np.unique(np.concatenate((y0*width+x0,y0*width+x1,y1*width+x0,y1*width+x1)))


def encode(depth, confidence, points_xy):
    depth, confidence = np.asarray(depth,dtype=np.float32), np.asarray(confidence,dtype=np.float32)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError('Depth and confidence must share the original2D raster')
    indices = pixel_support(points_xy,depth.shape)
    finite = np.flatnonzero(np.isfinite(confidence.ravel()))
    minimum_index = -1
    if len(finite):
        # Preserve the actual statistic consumed by confidence_quantile=0.
        # Its pixel is retained at its original location with its real depth.
        minimum_index = int(finite[np.argmin(confidence.ravel()[finite])])
        indices = np.union1d(indices,[minimum_index])
    return dict(shape=np.asarray(depth.shape,dtype=np.int64),
        indices=indices.astype(np.uint32),depth=depth.ravel()[indices].copy(),
        confidence=confidence.ravel()[indices].copy(),
        minimum_confidence_index=np.array(minimum_index,dtype=np.int64),
        detector_points=np.array(len(points_xy),dtype=np.int64))


def decode(record, world_scale=1.):
    shape = tuple(map(int,record['shape']))
    depth = np.full(shape,np.nan,dtype=np.float32)
    confidence = np.full(shape,np.nan,dtype=np.float32)
    indices = record['indices'].astype(np.int64)
    depth.ravel()[indices] = record['depth']
    confidence.ravel()[indices] = record['confidence']
    return depth*np.float32(world_scale), confidence


def validate_points(record, points_xy):
    requested = pixel_support(points_xy,record['shape'])
    stored = record['indices'].astype(np.int64)
    if not np.isin(requested,stored,assume_unique=True).all():
        raise ValueError('The matcher requested geometry outside the frozen detector support')


class FrameSupportArchive(Mapping):
    """Persistent geometry; at most one reconstructed historical raster cached."""

    def __init__(self,directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.records = {}
        self.count = 0
        self.array_bytes = 0
        self.disk_bytes = 0
        self._cached_frame = None
        self._cached_record = None

    def append(self,frame,depth,confidence,world_scale,points_xy,*,rgb_sha256):
        if isinstance(frame,bool) or int(frame)!=frame or frame!=len(self.records):
            raise ValueError('Historical observations must append in order')
        scale = float(world_scale)
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('A positive original coordinate scale is required')
        if len(rgb_sha256)!=64 or any(c not in '0123456789abcdef' for c in rgb_sha256):
            raise ValueError('The original RGB identity is required')
        record = encode(depth,confidence,points_xy)
        path = self.directory / f'{frame:06d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream,**record,frame=np.array(frame),world_scale=np.array(scale),
                rgb_sha256=np.array(rgb_sha256))
        self.records[frame] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shape=tuple(map(int,record['shape'])),scale=scale,bytes=path.stat().st_size,
            support_pixels=len(record['indices']),detector_points=len(points_xy),rgb_sha256=rgb_sha256)
        self.array_bytes += sum(record[k].nbytes for k in ('indices','depth','confidence'))
        self.disk_bytes += path.stat().st_size
        self.count += 1

    @classmethod
    def open(cls,directory,records):
        archive = object.__new__(cls)
        archive.directory = Path(directory)
        archive.records = {int(k):v for k,v in records.items()}
        if sorted(archive.records)!=list(range(len(archive.records))):
            raise ValueError('The saved archive is not a complete observed prefix')
        archive.count = len(archive.records)
        archive.array_bytes = sum(v['support_pixels']*12 for v in archive.records.values())
        archive.disk_bytes = sum(v['bytes'] for v in archive.records.values())
        archive._cached_frame = None
        archive._cached_record = None
        return archive

    def _record(self,frame):
        if isinstance(frame,bool) or not isinstance(frame,(int,np.integer)) or not 0<=frame<self.count:
            raise KeyError(frame)
        if self._cached_frame != frame:
            path = self.directory / f'{frame:06d}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.records[frame]['sha256']:
                raise RuntimeError('Historical geometric support changed')
            with np.load(path,allow_pickle=False) as source:
                self._cached_record = {k:source[k] for k in source.files}
            record = self._cached_record
            if int(record['frame'])!=frame or str(record['rgb_sha256'])!=self.records[frame]['rgb_sha256']:
                raise RuntimeError('Historical observation identity changed')
            self._cached_frame = frame
        return self._cached_record

    def validate_matches(self,frame,points_xy):
        validate_points(self._record(frame),points_xy)

    def __getitem__(self,frame):
        record = self._record(frame)
        return decode(record,record['world_scale'])

    def __iter__(self):
        return iter(range(self.count))

    def __len__(self):
        return self.count


class OnlineSupportArchive(FrameSupportArchive):
    """Write detector-accessible geometry from each actual observed RGB.

    Uses the readout's existing frozen detector without filling or evicting its
    feature cache. Only the depth samples persist; descriptors are released.
    The current-frame dense readout remains owned by EpisodicGEM.
    """

    def __init__(self, directory, *, rgb_directory, matcher, patch_size):
        from MemNavData.lingbot_pnp_localization import LightGluePointMatcher, SiftPnPConfig

        if not isinstance(matcher, LightGluePointMatcher):
            raise TypeError('Detector support requires the original SuperPoint/LightGlue provider')
        if SiftPnPConfig().depth_confidence_quantile != 0.:
            raise ValueError('Detector support preserves the original zero-quantile confidence filter')
        if matcher.extractor.training or matcher.matcher.training:
            raise ValueError('Detector support requires frozen evaluation-mode models')
        super().__init__(directory)
        self.rgb_directory = Path(rgb_directory).resolve()
        self._matcher = ref(matcher)
        self.patch_size = int(patch_size)
        self.detector_milliseconds = 0.
        self.storage_milliseconds = 0.
        self.validated_match_calls = 0

    def append(self, frame, depth, confidence, world_scale):
        from MemNavData.lingbot_pnp_localization import map_raw_points_to_lingbot_pad

        matcher = self._matcher()
        if matcher is None:
            raise RuntimeError('The frozen detector has been released')
        path = self.rgb_directory / f'{frame}.jpg'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        began = time.perf_counter()
        # Calling _reference_features here would populate the query cache with
        # write-time descriptors and change its cost/state. Extraction alone
        # leaves both reference and goal caches untouched.
        rgb = matcher.load_image(str(path)).to(matcher.device)
        features = matcher._extract(rgb)
        points = matcher.rbd(features)['keypoints'].detach().cpu().numpy()
        mapped = map_raw_points_to_lingbot_pad(points,
            raw_height=int(rgb.shape[-2]), raw_width=int(rgb.shape[-1]),
            target_height=int(depth.shape[-2]), target_width=int(depth.shape[-1]),
            patch_size=self.patch_size)
        self.detector_milliseconds += 1000*(time.perf_counter()-began)
        del features, rgb
        began = time.perf_counter()
        super().append(frame, depth, confidence, world_scale, mapped, rgb_sha256=digest)
        self.storage_milliseconds += 1000*(time.perf_counter()-began)

    def validate_reference(self, frame, path, matched_points, *, confidence_quantile):
        if confidence_quantile != 0.:
            raise ValueError('The confidence statistic differs from the archived readout contract')
        path = Path(path).resolve()
        if path != self.rgb_directory / f'{frame}.jpg':
            raise ValueError('Matched reference is outside this observed episode')
        record = self._record(frame)
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(record['rgb_sha256']):
            raise RuntimeError('The matched RGB differs from the archived observation')
        validate_points(record, matched_points)
        self.validated_match_calls += 1
