"""Current monocular depth and the two existing historical-depth providers."""

import hashlib
import os
import time

import numpy as np
import torch


class DenseReadout:
    @torch.no_grad()
    def read_dense(self):
        """Return current raw LingBot depth under the frozen MDTEC contract."""
        backend = self.backend

        started = time.perf_counter()
        if self.frame_count < 1 or self.current_rgb_sha256 is None:
            raise RuntimeError("monocular depth requires one streamed RGB frame")
        from PIL import Image
        from MemNavData.monocular_depth_runtime import (
            ACTIVE_FROM_FRAME_INDEX,
            build_monocular_depth_payload,
        )

        frame_index = self.frame_count - 1
        current_path = os.path.join(self.rgb_dir, f"{frame_index}.jpg")
        with Image.open(current_path) as current_image:
            source_width, source_height = current_image.size

        relative_depth = None
        depth_shape = (source_height, source_width)
        cache_hit = False
        prediction_runtime_ms = 0.0
        if frame_index >= ACTIVE_FROM_FRAME_INDEX:
            if self.metric_scale_receipt is None:
                raise RuntimeError(
                    "first-40 scale receipt missing after activation frame"
                )
            if self.metric_scale_receipt["scale_valid"] is True:
                cache_hit = (
                    self.current_depth_frame == frame_index
                    and self.current_relative_depth is not None
                )
                if cache_hit:
                    relative_depth = (
                        self.current_relative_depth.cpu().numpy()
                    )
                else:
                    if self.current_aggregation is None or self.patch_start_index is None:
                        raise RuntimeError("LingBot current-frame depth state is absent")
                    prediction_started = time.perf_counter()
                    prediction = backend.lb.model._predict_depth(
                        self.current_aggregation,
                        self.window_images[-1][None, None].to(backend.device),
                        self.patch_start_index,
                    )
                    relative_depth = prediction[
                        "depth"][0, -1, ..., 0].float().cpu().numpy()
                    prediction_runtime_ms = 1000.0 * (
                        time.perf_counter() - prediction_started
                    )
                depth_shape = tuple(int(value) for value in relative_depth.shape)

        payload = build_monocular_depth_payload(
            relative_depth=relative_depth,
            depth_shape=depth_shape,
            image_sha256_value=self.current_rgb_sha256,
            frame_index=frame_index,
            scale_receipt=self.metric_scale_receipt,
        )
        payload["first40_scale_freeze_ms"] = self.metric_scale_freeze_ms
        payload["stream_observation_count"] = int(self.frame_count)
        payload["depth_prediction_cache_hit"] = bool(cache_hit)
        payload["depth_prediction_runtime_ms"] = float(
            prediction_runtime_ms
        )
        payload["depth_materialization_runtime_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        return payload

    @torch.no_grad()
    def read_replayed_depth(self, anchor):
        """Return exact causal LingBot depth with an anchor-result cache.

        Candidate ranking is image-only, so this expensive full replay happens
        for at most one history frame per goal.  The first request for an anchor
        remains the independently confirmed full replay.  Later goals selecting
        that same immutable frame reuse its final arrays exactly; no intermediate
        transformer state is approximated.
        """
        backend = self.backend
        anchor = int(anchor)
        cached = self.replay_depths.get(anchor)
        if cached is not None:
            depth, confidence = cached
            self.last_reference_read = {
                "enabled": True,
                "anchor": anchor,
                "cache_hit": True,
                "cache_source": (
                    "eager_dense_writer"
                    if anchor in backend._certified_eager_depth_cached_anchors
                    else "prior_selected_anchor"),
                "replayed_frames": 0,
                "cached_anchors": len(
                    self.replay_depths),
                "cache_bytes": int(sum(
                    array.nbytes for pair in
                    self.replay_depths.values()
                    for array in pair)),
            }
            return depth.copy(), confidence.copy()
        depth, confidence = backend._certified_reference_depth_impl(anchor)
        self.replay_depths[anchor] = (
            depth.copy(), confidence.copy())
        self.last_reference_read = {
            "enabled": True,
            "anchor": anchor,
            "cache_hit": False,
            "replayed_frames": anchor - backend.S + 1,
            "cached_anchors": len(self.replay_depths),
            "cache_bytes": int(sum(
                array.nbytes for pair in
                self.replay_depths.values()
                for array in pair)),
        }
        return depth, confidence

    @torch.no_grad()
    def read_online_depth(self, anchor):
        """Return one write-once sparse depth observation without replay.

        This cache is generated by the same causal short-range LingBot stream
        that already supplies NavDP depth. The online_history configuration
        also uses it for the initial target certificate; legacy route tracking
        uses the same storage. Neither launches a second dense KV stream.
        """
        backend = self.backend

        anchor = int(anchor)
        cached = self.online_depths.get(anchor)
        if cached is None:
            raise RuntimeError(
                f"route depth is unavailable for historical frame {anchor}")
        depth, confidence = cached
        self.last_reference_read = {
            "enabled": True,
            "anchor": anchor,
            "cache_hit": True,
            "cache_source": "sparse_causal_route_writer",
            "replayed_frames": 0,
            "cached_anchors": len(
                self.online_depths),
            "cache_bytes": int(sum(
                array.nbytes for pair in
                self.online_depths.values()
                for array in pair)),
        }
        return depth.copy(), confidence.copy()

    @torch.no_grad()
    def replay_depth(self, anchor):
        """Original exact replay from the frozen scale block through anchor."""
        backend = self.backend
        anchor = int(anchor)
        if anchor < backend.S or anchor >= self.frame_count:
            raise ValueError(
                f"certified anchor {anchor} outside [{backend.S}, {self.frame_count - 1}]")
        snap = backend._snapshot()
        try:
            cache = backend._live_cache()
            indices = cache.get("anchor_frame_indices")
            if indices is None:
                backend.lb._inject(
                    cache["scale_k"], cache["scale_v"],
                    cache["anchor_k"], cache["anchor_v"],
                    n_hist=0, total_frames=backend.S)
            else:
                backend.lb._inject(
                    cache["scale_k"], cache["scale_v"],
                    cache["anchor_k"], cache["anchor_v"],
                    anchor_frame_indices=indices, raw_start=backend.S)
            final_agg = final_psi = final_image = None
            # Bound host memory while retaining exact sequential inference.
            for chunk_start in range(backend.S, anchor + 1, 16):
                chunk_end = min(anchor + 1, chunk_start + 16)
                paths = [
                    os.path.join(self.rgb_dir, f"{index}.jpg")
                    for index in range(chunk_start, chunk_end)
                ]
                if not all(os.path.isfile(path) for path in paths):
                    missing = next(path for path in paths
                                   if not os.path.isfile(path))
                    raise FileNotFoundError(missing)
                images = backend.lb.load_images(paths)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for offset in range(len(images)):
                        final_image = images[offset:offset + 1][None].to(
                            backend.device)
                        final_agg, final_psi = (
                            backend.lb.model._aggregate_features(
                                final_image,
                                num_frame_for_scale=backend.S,
                                num_frame_per_block=1))
            if final_agg is None or final_image is None:
                raise RuntimeError("certified anchor replay produced no frame")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = backend.lb.model._predict_depth(
                    final_agg, final_image, final_psi)
            depth = prediction["depth"][0, -1, ..., 0].float().cpu().numpy()
            confidence = prediction["depth_conf"][0, -1].float().cpu().numpy()
            return depth, confidence
        finally:
            backend._restore(snap)
