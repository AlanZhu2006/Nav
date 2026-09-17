"""Bounded LingBot writing and persistent evidence under the existing GEM API.

Only the memory writer and storage change. SparseReadout still owns the
original DINO shortlist, SuperPoint/LightGlue, PnP, certificate and bearing.
No learned-policy cache or second geometric model is maintained.
"""
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .archive import FrameDepthArchive
from .connected import ConnectedEpisodeMemory
from .descriptor_cache import DescriptorReadCache
from .memory import GeometricEpisodicMemory
from .reactivation import isolated_stream


class EpisodicGEM(GeometricEpisodicMemory):
    def __init__(self, backend, *, bounded=True, geometry_storage='dense', kv_storage='native'):
        super().__init__(backend)
        if geometry_storage not in ('dense', 'detector_support'):
            raise ValueError('Unknown historical geometry storage')
        if kv_storage not in ('native', 'reader_precision', 'paged_bf16', 'int8_storage', 'lossless_bf16'):
            raise ValueError('Unknown working KV storage')
        if kv_storage != 'native' and bounded:
            raise ValueError('Optimized KV storage requires the native interval7 writer')
        self.bounded = bool(bounded)
        self.geometry_storage = geometry_storage
        self.kv_storage = kv_storage
        self.reader_precision_receipt = None
        self.paged_memory_receipt = None
        self.int8_memory_receipt = None
        self.lossless_memory_receipt = None
        self.stream = None
        self.failure = None
        self.image_size = None
        self.write_timings = []
        self.descriptor_read_cache = DescriptorReadCache()

    def reset(self, camera_height=0.5, seed=None, episode_len=None, camera_intrinsic=None):
        backend = self.backend
        if self.kv_storage != 'native':
            self.failure = 'Working-memory reset did not complete; reset is required'
            self.reader_precision_receipt = None
            self.paged_memory_receipt = None
            self.int8_memory_receipt = None
            self.lossless_memory_receipt = None
        if backend.S != 8:
            raise ValueError('Episodic GEM requires the native eight-view initialization')
        if backend.certified_reference_depth_source != 'online_history':
            raise ValueError('Episodic GEM reads its own observed historical depth')
        if backend.certified_eager_depth_cache or backend.depth_observation_only:
            raise ValueError('Episodic GEM owns one complete geometry stream')
        if self.kv_storage != 'native' and (
                backend.W != 64 or backend.lb.model.kv_cache_sliding_window != 64):
            raise ValueError('Optimized KV storage requires the verified W64 context')
        super().reset(camera_height, seed, episode_len, camera_intrinsic)
        self.descriptor_read_cache.reset()
        backend.lb.model.aggregator.to(dtype=torch.bfloat16)
        # The native GCT forward calls the FP32 camera wrapper. Do not replace
        # it by the direct autocast camera-head call in the legacy writer.
        backend.lb.model.camera_head.float()
        if self.kv_storage == 'reader_precision':
            from .read_precision import enable_reader_precision
            self.reader_precision_receipt = enable_reader_precision(backend.lb.model)
        elif self.kv_storage == 'paged_bf16':
            from .paged import enable_paged_memory
            self.paged_memory_receipt = enable_paged_memory(backend.lb.model)
        elif self.kv_storage == 'int8_storage':
            from .int8 import enable_int8_storage
            self.int8_memory_receipt = enable_int8_storage(backend.lb.model)
        elif self.kv_storage == 'lossless_bf16':
            from .lossless import enable_lossless_storage
            self.lossless_memory_receipt = enable_lossless_storage(backend.lb.model)
        self.stream = None
        self.failure = None
        self.image_size = None
        self.write_timings = []
        if self.geometry_storage == 'detector_support':
            from .support import OnlineSupportArchive
            self.online_depths = OnlineSupportArchive(Path(self.rgb_dir) / 'geometry',
                rgb_directory=self.rgb_dir, matcher=backend.certified_relocalization_matcher,
                patch_size=backend.lb.patch_size)
        else:
            self.online_depths = FrameDepthArchive(Path(self.rgb_dir) / 'geometry')

    def _healthy(self):
        if self.failure is not None:
            raise RuntimeError('Episodic GEM failed; reset is required: ' + self.failure)

    def write(self, jpg_bytes, *, executed_translation_m=None, executed_yaw_rad=None,
              executed_forward_m=None, executed_left_m=None, executor_local_se2_source=None):
        self._healthy()
        # This RGB memory does not consume executor odometry. The HTTP layer
        # can retain its own execution receipts for evaluation.
        if any(x is not None for x in (executed_translation_m, executed_yaw_rad,
                executed_forward_m, executed_left_m, executor_local_se2_source)):
            raise ValueError('Episodic GEM accepts RGB observations without executor motion')
        backend = self.backend
        frame = self.frame_count
        started = time.perf_counter()
        self.current_relative_depth = None
        self.current_depth_frame = None
        try:
            path = Path(self.rgb_dir) / f'{frame}.jpg'
            with path.open('xb') as stream:
                stream.write(jpg_bytes)
            with Image.open(path) as rgb:
                size = rgb.size
            if self.image_size is not None and self.image_size != size:
                raise ValueError('Camera raster changed within the geometry episode')
            image = backend.lb.load_images([str(path)])[0]
            if self.stream is None:
                self.image_size = size
                yy, xx = np.meshgrid(np.arange(8, 518, 16), np.arange(8, 518, 16), indexing='ij')
                uv = np.stack([xx.ravel(), yy.ravel()], -1)
                factor = 518 / max(size)
                nw, nh = [round(v * factor / 14) * 14 for v in size]
                left, top = (518-nw)//2, (518-nh)//2
                valid = ((uv[:, 0] >= left) & (uv[:, 0] < left+nw)
                    & (uv[:, 1] >= top) & (uv[:, 1] < top+nh))
                self.stream = ConnectedEpisodeMemory(backend.lb.model, uv, valid,
                    transport='reciprocal', bounded=self.bounded)
            result = self.stream.write(frame, image)
            if result is not None:
                descriptors = self.pop_descriptors(len(result.frames))
                depth = result.predictions['depth'][0, ..., 0].float().cpu().numpy()
                confidence = result.predictions['depth_conf'][0].float().cpu().numpy()
                for j, identity in enumerate(result.frames):
                    self.online_depths.append(identity, depth[j], confidence[j], result.world_scale)
                self.descriptors.extend(descriptors)
                self.poses.extend(torch.from_numpy(p.copy()) for p in result.world_pose9)
                self.online_depth_frames.update(result.frames)
                self.current_relative_depth = torch.from_numpy(
                    depth[-1].copy() * np.float32(result.world_scale))
                self.current_depth_frame = frame
                self.anchor_frames = [row[0] for row in self.stream.geometry]
                self.camera_frames = self.anchor_frames.copy()
                backend._dino_out[0] = None
            self.current_rgb_sha256 = hashlib.sha256(jpg_bytes).hexdigest()
            self.frame_count += 1
            backend.executor_motion_receipts.append(None)
            if self.frame_count == 40:
                self.freeze_metric_scale()
            self.write_timings.append(dict(frame=frame, milliseconds=1000*(time.perf_counter()-started),
                episode=self.stream.episode, live_committed=backend.lb.agg.total_frames_processed))
            return frame
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            self.current_relative_depth = None
            self.current_depth_frame = None
            raise

    def freeze_metric_scale(self):
        from MemNavData.monocular_depth_runtime import compute_first40_scale_receipt

        if self.frame_count != 40 or self.metric_scale_receipt is not None:
            raise RuntimeError('Metric scale must be frozen once at observation 40')
        backend = self.backend
        saved_descriptor_hook = backend._dino_out[0]
        tick = time.perf_counter()
        try:
            with isolated_stream(backend.lb.model):
                self.metric_scale_receipt = compute_first40_scale_receipt(backend.lb, self.rgb_dir,
                    torch.stack(self.poses).float().numpy(), backend.camera_height)
        finally:
            backend._dino_out[0] = saved_descriptor_hook
            self.metric_scale_freeze_ms = 1000*(time.perf_counter()-tick)

    def read_online_depth(self, anchor):
        self._healthy()
        depth, confidence = self.online_depths[anchor]
        self.last_reference_read = dict(enabled=True, anchor=int(anchor), cache_hit=True,
            cache_source='episodic_first_observation', replayed_frames=0,
            cached_anchors=len(self.online_depths), cache_bytes=self.online_depths.array_bytes,
            disk_bytes=self.online_depths.disk_bytes)
        return depth, confidence

    def read_dense(self):
        self._healthy()
        return super().read_dense()

    def read_replayed_depth(self, anchor):
        self._healthy()
        raise RuntimeError('Episodic GEM reads its archived depth; canonical replay is incompatible')

    def read_sparse(self, goal_jpg_bytes, candidates, **kwargs):
        self._healthy()
        if kwargs.get('reference_depth_source') not in (None, 'online_history'):
            raise ValueError('Episodic GEM requires its archived online-history depth')
        if (kwargs.get('graph_rescue', False) or kwargs.get('allow_learned_rescue', False)
                or kwargs.get('guidance_mode', 'endpoint_bearing') != 'endpoint_bearing'
                or kwargs.get('route_start_anchor') is not None):
            raise ValueError('Episodic GEM is evaluated with the unchanged endpoint-bearing readout')
        try:
            result = super().read_sparse(goal_jpg_bytes, candidates, **kwargs)
            if (result.get('pnp', {}).get('status') == 'runtime_exception'
                    or any(r.get('error') for r in result.get('ranked_candidates', []))):
                raise RuntimeError('Sparse memory evidence could not be read or localized')
            return result
        except BaseException as error:
            # An inference/archive failure is not a geometric abstention.
            # The frozen navigation harness already enforces this boundary;
            # enforce it at the public memory API as well.
            self.failure = f'{type(error).__name__}: {error}'
            raise

    def retrieve(self, *args, **kwargs):
        self._healthy()
        return super().retrieve(*args, **kwargs)

    def retrieval_descriptors(self, descriptors):
        return self.descriptor_read_cache.read(descriptors, self.backend.device)

    def planning_cache(self):
        raise RuntimeError('Episodic GEM supplies geometric readouts; learned-policy KV caches are absent')

    def status(self):
        result = dict(module='connected_reciprocal' if self.bounded else 'native_interval7',
            frames=self.frame_count, descriptor_frames=len(self.descriptors), pose_frames=len(self.poses),
            episodes=0 if self.stream is None else self.stream.episode+1,
            live_committed=0 if self.stream is None else self.backend.lb.agg.total_frames_processed,
            online_depth_frames=len(self.online_depths), online_depth_bytes=self.online_depths.array_bytes,
            depth_archive_bytes=self.online_depths.disk_bytes, cached_goals=len(self.certificates),
            historical_depth_source='online_history', failure=self.failure)
        result['descriptor_read_cache'] = self.descriptor_read_cache.statistics()
        if self.geometry_storage == 'detector_support':
            result.update(geometry_storage=self.geometry_storage,
                support_detector_milliseconds=self.online_depths.detector_milliseconds,
                support_storage_milliseconds=self.online_depths.storage_milliseconds,
                support_validated_match_calls=self.online_depths.validated_match_calls)
        if self.kv_storage == 'reader_precision':
            result.update(kv_storage=self.kv_storage,
                reader_precision_receipt=self.reader_precision_receipt)
        elif self.kv_storage == 'paged_bf16':
            manager = self.backend.lb.agg.kv_cache_manager
            result.update(kv_storage=self.kv_storage,
                paged_memory_receipt=self.paged_memory_receipt,
                paged_working_memory=None if manager is None else manager.statistics())
        elif self.kv_storage == 'int8_storage':
            result.update(kv_storage=self.kv_storage,
                int8_memory_receipt=self.int8_memory_receipt,
                int8_working_memory=self.backend.lb.agg.int8_statistics())
        elif self.kv_storage == 'lossless_bf16':
            result.update(kv_storage=self.kv_storage,
                lossless_memory_receipt=self.lossless_memory_receipt,
                lossless_working_memory=self.backend.lb.agg.lossless_statistics())
        return result
