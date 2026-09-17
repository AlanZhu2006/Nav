"""Causal RGB writer and LingBot working state, extracted without changing inference."""

import hashlib
import os
import shutil
import time

import numpy as np
import torch


FLOW_TIERS = [(702, 20.0), (877, 25.0), (1075, 30.0), (1506, 40.0), (2048, 50.0)]
FLOW_GAP = 30


def flow_threshold_for_length(n_frames):
    """The existing per-episode keyframe schedule used by precompute."""
    for upper, threshold in FLOW_TIERS:
        if n_frames <= upper:
            return threshold
    return 60.0


class CausalWriter:
    def reset(self, camera_height=0.5, seed=None, episode_len=None,
              camera_intrinsic=None):
        # Reset diffusion randomness per episode so terminal-mode A/B runs have
        # an identical navigation prefix.  This does not force deterministic
        # CUDA kernels; it controls the explicit torch.randn DDPM start noise.
        backend = self.backend
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        backend._episode_counter += 1
        self.rgb_dir = os.path.join(backend.buffer_root, f"ep_{backend._episode_counter:04d}")
        shutil.rmtree(self.rgb_dir, ignore_errors=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        backend.camera_height = float(camera_height)
        backend.camera_intrinsic = None
        if camera_intrinsic is not None:
            intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError("camera_intrinsic must be a finite 3x3 matrix")
            backend.camera_intrinsic = intrinsic
        if backend.flow_gate == "off":
            backend.flow_threshold = 0.0
        elif backend.flow_gate == "auto":
            backend.flow_threshold = (
                flow_threshold_for_length(int(episode_len))
                if episode_len else FLOW_TIERS[0][1]
            )
        else:
            backend.flow_threshold = float(backend.flow_gate)
        backend.flow_gap = FLOW_GAP
        backend._last_episode_len = episode_len
        backend._last_seed = seed
        self.last_keyframe_pose = None
        self.last_keyframe_index = backend.S - 1
        self.anchor_frames = []
        self.camera_frames = []
        self.frame_count = 0                       # frames streamed so far
        self.pending_images = []               # preprocessed frames waiting for the scale block
        self.window_images = []           # last W preprocessed frames (cpu), for window_forward
        self.descriptors = []               # per-frame [1024] fp32 cpu
        self.anchor_k = []               # per phase-2 frame [L,H,6,d] bf16 gpu
        self.anchor_v = []
        self.camera_k = []                  # per-frame [NI,TD,H,d] bf16 gpu (stacked lazily)
        self.camera_v = []
        self.poses = []               # per-frame [9] fp32
        # Optional frame-bound executor receipts. Entry i describes realized
        # motion from causal RGB i-1 to i. They are internal action/odometry
        # receipts, not evaluator poses or an additional exteroceptive stream.
        backend.executor_motion_receipts = []
        self.scale_k = None              # [L,H,S,P,d] bf16 gpu
        self.scale_v = None
        backend._metric_scale = None        # lazy ground-anchored scale
        self.goal_embeddings_and_poses = {}            # (goal_md5, anchor) -> goal_pose; goal_md5 -> goal_cls
        backend._anchor_state = {}          # goal_md5 -> dict(m, score): sticky-anchor ratchet
        self.goal_start_frames = {}      # goal_md5 -> first frame queried for this goal
        backend._graph_routes = {}          # goal_md5 -> frozen reverse-memory route + cursor
        # A goal image may reappear after intervening goals in a lifelong
        # episode.  Long-term RGB/map state survives that switch, whereas every
        # goal-conditioned proposal/proof cache belongs to one contiguous goal
        # session.  Otherwise an A->B->A sequence would reuse A's original
        # candidate ceiling and could never consume observations acquired while
        # first pursuing A.
        self.active_goal_key = None
        self.goal_session_index = 0
        self.goal_session_started = False
        # SIFT/essential verification is a deterministic function of the goal,
        # immutable history image, and per-episode intrinsic.  Cache both
        # positive and negative results so temporal confirmation checks anchor
        # stability instead of recomputing the identical image pair.
        backend._retrieval_verification_cache = {}
        # Read-only learned-ranking results are frozen per goal and exact
        # shortlist.  DINO scores can vary by a few floating-point bits across
        # otherwise identical GPU queries, so the expensive deterministic
        # geometry is cached separately by its actual immutable inputs.  The
        # cheap model rank is still recomputed from each request's exact DINO
        # scores whenever the exact-result cache misses.
        backend._phase_b_rank_cache = {}
        backend._phase_b_scale_cache = {}
        backend._phase_b_geometry_cache = {}
        # One immutable absolute goal pose (or one immutable abstention) per
        # goal.  Accepted poses are converted to a fresh current-relative
        # PointGoal on each request, so localization is paid once rather than
        # once per navigation replan.
        self.certificates = {}
        # Dense reference depth depends only on the immutable history and the
        # selected anchor, not on the goal.  Different lifelong goals often
        # retrieve the same place, so retain exact final depth/confidence while
        # keeping the first request identical to the confirmed full replay.
        self.replay_depths = {}
        # Sparse CPU-only local-observation cache.  It is intentionally
        # separate from the exact canonical certificate depth cache above, so
        # enabling route tracking cannot alter initial CEC authorization.
        self.online_depths = {}
        self.online_depth_frames = set()
        # The dense-query motion model retains at most one controller interval
        # of write-time depth.  These frame-bound values let the visual route
        # consume the same observation cadence as the executor without turning
        # every query frame into a permanent long-term-memory node.
        backend._certified_route_live_depth_cache = {}
        self.last_reference_read = None
        backend._certified_dense_stream_snapshot = None
        backend._certified_eager_depth_error = None
        backend._certified_eager_depth_runtime_ms = []
        backend._certified_eager_depth_cached_anchors = set()
        # One frozen initial Pi3X proposal decision per goal.  An initial
        # reject is sticky; an accepted anchor is fixed while its current-to-
        # goal bearing is recomputed from causal RGB at each later replan.
        backend._pi3x_relocalization_cache = {}
        # Optional, default-off rescue routes are separate from the historical
        # always-on reverse graph.  A route is created only after an external
        # progress monitor declares the direct certified bearing stuck.
        backend._certified_graph_routes = {}
        # Development-only long-range readout.  Each accepted goal owns one
        # continuous tail-to-anchor path and monotone progress scalar.  It is
        # separate from the dormant discrete stuck-rescue graph above and is
        # never consulted by canonical endpoint-bearing CEC.
        backend._certified_path_field_routes = {}
        # Scale-free long-return challenger. CEC authorizes the historical
        # route once; subsequent route state is advanced only by frame-bound
        # executor translation/yaw receipts.
        backend._certified_action_coordinate_routes = {}
        # Development successor to the scalar action coordinate.  It rebuilds
        # a local 2-D executor route from the same receipts and advances route
        # state by monotone geometric projection, never by travelled distance.
        backend._certified_se2_route_compasses = {}
        # Deployable long-range route readout.  Both the historical route and
        # live state are reconstructed from causal RGB, height-scaled LingBot
        # depth, and adjacent LightGlue/PnP motion.  It never consumes the
        # optional executor/simulator receipts above.
        backend._certified_monocular_route_tangents = {}
        # The candidate set is fixed at the first causal query for a goal.
        # An empty set is a real, cacheable abstention (for example after a
        # very short Novel leg), not permission to admit later goal-session
        # frames or repeatedly pay localization cost.
        self.shortlists = {}
        # GOAT semantic arrival is intentionally independent of the Revisit
        # controller's pooled-scale fallback.  This cache holds the one strict
        # first-64-frame estimate (or its fail-closed unavailability receipt).
        backend._arrival_metric_scale_result = None
        # MDTEC short-horizon readout.  This is deliberately separate from
        # ``_metric_scale``: the latter is a legacy lazy helper whose evidence
        # grows with the whole stream, whereas the monocular NavDP interface
        # must freeze exactly RGB observations 0..39 and never update again.
        self.metric_scale_receipt = None
        self.metric_scale_freeze_ms = None
        self.current_rgb_sha256 = None
        # The flow gate already predicts depth for every post-warmup frame.
        # Retain only that newest immutable tensor so the MDTEC transaction
        # can serialize it without running the same depth head a second time.
        self.current_depth_frame = None
        self.current_relative_depth = None
        # tower-1 live capture: the current frame's post-GCT tokens + agg list from the
        # CONTINUOUS stream. Training used window_forward's cold-cache recompute only
        # because samples load from disk; at eval the live stream supersedes it.
        self.current_tokens = None         # [1, P, 2C] current frame post-GCT tokens
        self.current_aggregation = None            # list of [1,1,P,2C] (selected layers, current frame)
        self.patch_start_index = None                 # patch_start_idx from the scale block
        backend.lb.model.clean_kv_cache()
        backend.lb.model.camera_head.clean_kv_cache()

    def pop_descriptors(self, n_frames):
        backend = self.backend
        out = backend._dino_out[0]
        cls = out["x_norm_clstoken"].reshape(n_frames, -1).float().cpu()
        return [cls[i] for i in range(n_frames)]

    def capture_anchor(self):
        backend = self.backend
        kv = backend.lb.agg.kv_cache
        ak = torch.stack([kv[f"k_{i}"][0, :, -1, :backend.psi].to(torch.bfloat16)
                          for i in range(backend.L_depth)])
        av = torch.stack([kv[f"v_{i}"][0, :, -1, :backend.psi].to(torch.bfloat16)
                          for i in range(backend.L_depth)])
        return ak, av                                   # [L,H,6,d]

    def capture_camera(self, n_new):
        backend = self.backend
        ch = backend.lb.model.camera_head
        NI, TD = ch.num_iterations, ch.trunk_depth
        ks, vs = [], []
        for it in range(NI):
            d = ch.kv_cache[it]
            ks.append(torch.stack([d[f"k_{bl}"][0, :, -n_new:, 0] for bl in range(TD)], 0))
            vs.append(torch.stack([d[f"v_{bl}"][0, :, -n_new:, 0] for bl in range(TD)], 0))
        # ks: list[NI] of [TD, H, n_new, d] -> [n_new, NI, TD, H, d]
        k = torch.stack(ks, 0).permute(3, 0, 1, 2, 4).to(torch.bfloat16)
        v = torch.stack(vs, 0).permute(3, 0, 1, 2, 4).to(torch.bfloat16)
        return [k[i] for i in range(n_new)], [v[i] for i in range(n_new)]

    def write(
            self, jpg_bytes, *, executed_translation_m=None,
            executed_yaw_rad=None, executed_forward_m=None,
            executed_left_m=None, executor_local_se2_source=None):
        """Ingest one RGB frame (jpg bytes). Returns the frame index."""
        backend = self.backend
        idx = self.frame_count
        # A failed or partial append must never expose the preceding frame's
        # depth under the new RGB SHA/frame binding.
        self.current_depth_frame = None
        self.current_relative_depth = None
        if executed_translation_m is None and executed_yaw_rad is None:
            if (executed_forward_m is not None
                    or executed_left_m is not None
                    or executor_local_se2_source is not None):
                raise ValueError(
                    "local SE(2) receipt requires the legacy motion binding")
            executor_receipt = None
        elif executed_translation_m is None or executed_yaw_rad is None:
            raise ValueError(
                "executor translation and yaw receipts must be supplied "
                "together")
        else:
            translation = float(executed_translation_m)
            yaw = float(executed_yaw_rad)
            if (not np.isfinite(translation) or translation < 0.0
                    or not np.isfinite(yaw)):
                raise ValueError(
                    "executor receipt must contain finite non-negative "
                    "translation and finite yaw")
            if idx == 0 and (translation > 1e-9 or abs(yaw) > 1e-9):
                raise ValueError("the first causal frame must have zero motion")
            executor_receipt = {
                "frame_index": int(idx),
                "executed_translation_m": translation,
                "executed_yaw_rad": yaw,
                "contract": "frame_bound_realized_executor_motion_v1",
            }
            local_absent = (
                executed_forward_m is None
                and executed_left_m is None
                and executor_local_se2_source is None)
            if not local_absent:
                if (executed_forward_m is None
                        or executed_left_m is None
                        or executor_local_se2_source is None):
                    raise ValueError(
                        "executor forward and left receipts must be supplied "
                        "together")
                forward = float(executed_forward_m)
                left = float(executed_left_m)
                if (not isinstance(executor_local_se2_source, str)
                        or not executor_local_se2_source.strip()):
                    raise ValueError(
                        "executor local SE(2) source must be explicit")
                if not np.isfinite(forward) or not np.isfinite(left):
                    raise ValueError(
                        "executor local SE(2) receipt must be finite")
                if idx == 0 and (abs(forward) > 1e-9 or abs(left) > 1e-9):
                    raise ValueError(
                        "the first causal frame must have zero local motion")
                executor_receipt["local_se2"] = {
                    "contract": "frame_bound_local_se2_v1",
                    "executed_forward_m": forward,
                    "executed_left_m": left,
                    "executed_yaw_rad": yaw,
                    "source": executor_local_se2_source.strip(),
                }
        # Fail closed at the LingBot RoPE position cap.  Past max_frame_num
        # the streaming 3D-RoPE table slices silently truncate: the temporal
        # frequency components come back EMPTY (verified 2026-08-22), so
        # positions past the cap are malformed with no error at this layer.
        # The flow gate rolls back total_frames_processed for dropped frames
        # (the gatecurr-era interval fix), so the binding counter is the
        # aggregator's committed-position count, NOT the raw frame index --
        # a gated 2500-step episode consumes only a few hundred positions.
        agg_mod_cap = getattr(backend.lb, "agg", None)
        cap = getattr(agg_mod_cap, "max_frame_num", None)
        if cap is not None and int(
                getattr(agg_mod_cap, "total_frames_processed", 0)
        ) >= int(cap):
            raise RuntimeError(
                "memory stream RoPE position cap reached: "
                f"total_frames_processed would exceed max_frame_num={int(cap)} "
                "and positions past the cap are silently malformed. Raise "
                "MEMNAV_MAX_FRAME_NUM, or check that flow gating is enabled "
                "with the episode's true total length.")
        backend.executor_motion_receipts.append(executor_receipt)
        self.current_rgb_sha256 = hashlib.sha256(jpg_bytes).hexdigest()
        path = os.path.join(self.rgb_dir, f"{idx}.jpg")
        with open(path, "wb") as f:
            f.write(jpg_bytes)
        img = backend.lb.load_images([path])[0]            # [3,518,518] pad-518 (cpu)
        self.window_images.append(img)
        if len(self.window_images) > backend.W:
            self.window_images.pop(0)

        ch = backend.lb.model.camera_head
        if idx < backend.S - 1:
            self.pending_images.append(img)
            self.frame_count += 1
            return idx
        if idx == backend.S - 1:
            # scale block: first S frames as ONE bidirectional block
            self.pending_images.append(img)
            blk = torch.stack(self.pending_images, 0)[None].to(backend.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, psi = backend.lb.model._aggregate_features(
                    blk, num_frame_for_scale=backend.S, num_frame_per_block=backend.S)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=backend.S, num_frame_for_scale=backend.S)
            self.patch_start_index = psi
            self.current_tokens = agg[-1][:, -1]
            self.current_aggregation = [layer[:, -1:] for layer in agg]
            self.descriptors.extend(backend._pop_cls(backend.S))
            if not getattr(backend, "depth_observation_only", False):
                kv = backend.lb.agg.kv_cache
                self.scale_k = torch.stack([
                    kv[f"k_{i}"][0, :, :backend.S].to(torch.bfloat16)
                    for i in range(backend.L_depth)
                ]).contiguous()
                self.scale_v = torch.stack([
                    kv[f"v_{i}"][0, :, :backend.S].to(torch.bfloat16)
                    for i in range(backend.L_depth)
                ]).contiguous()
            pose = pl[-1][0].float()                    # [S,9]
            self.poses.extend([pose[i].cpu() for i in range(backend.S)])
            if not getattr(backend, "depth_observation_only", False):
                ck, cv = backend._read_cam_newest(backend.S)
                self.camera_k.extend(ck); self.camera_v.extend(cv)
            self.last_keyframe_pose = pl[-1][:, -1:].float()
            self.last_keyframe_index = backend.S - 1
            self.camera_frames = list(range(backend.S))
            self.pending_images = []
            if backend.certified_eager_depth_cache:
                # Scale inference is common to sparse navigation and dense
                # certificate streams.  Hold its immutable reference snapshot
                # as the exact starting point for the first post-scale frame.
                backend._certified_dense_stream_snapshot = backend._snapshot()
        else:
            # The live stream always evaluates the newest frame. When the flow
            # policy rejects it, roll back only the stored KV append; dense cls,
            # pose, and current-state tokens remain aligned to raw frame indices.
            agg_mod = backend.lb.agg
            gate_on = backend.flow_threshold > 0
            if gate_on:
                saved_kv = dict(agg_mod.kv_cache)
                saved_cam = [dict(layer) for layer in ch.kv_cache]
                saved_total = int(agg_mod.total_frames_processed)
            prediction = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, _ = backend.lb.model._aggregate_features(
                    img[None, None].to(backend.device),
                    num_frame_for_scale=backend.S, num_frame_per_block=1)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=1, num_frame_for_scale=backend.S)
            self.current_tokens = agg[-1][:, -1]
            self.current_aggregation = [layer for layer in agg]
            self.descriptors.extend(backend._pop_cls(1))
            self.poses.append(pl[-1][0].float()[-1].cpu())

            if gate_on:
                cur_pose = pl[-1][:, -1:].float()
                if idx == backend.S:
                    is_keyframe = True
                else:
                    from lingbot_map.models.gct_stream_window import _compute_flow_magnitude
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = backend.lb.model._predict_depth(
                            agg, img[None, None].to(backend.device), self.patch_start_index)
                        depth = prediction["depth"].float()
                    flow = _compute_flow_magnitude(
                        cur_pose, self.last_keyframe_pose, depth, tuple(depth.shape[2:4])
                    )
                    is_keyframe = (
                        flow > backend.flow_threshold
                        or (idx - self.last_keyframe_index) >= backend.flow_gap
                    )
            else:
                is_keyframe = True

            if is_keyframe:
                if gate_on:
                    self.last_keyframe_pose = cur_pose
                    self.last_keyframe_index = idx
                if not getattr(backend, "depth_observation_only", False):
                    ak, av = backend._read_anchor_newest()
                    self.anchor_k.append(ak); self.anchor_v.append(av)
                    self.anchor_frames.append(idx)
                    ck, cv = backend._read_cam_newest(1)
                    self.camera_k.extend(ck); self.camera_v.extend(cv)
                    self.camera_frames.append(idx)
            else:
                agg_mod.kv_cache.clear()
                agg_mod.kv_cache.update(saved_kv)
                ch.kv_cache = saved_cam
                agg_mod.total_frames_processed = saved_total
            dense_query_due = (
                backend.certified_route_motion_model ==
                "direct_pnp_dense_query"
                and bool(backend._certified_monocular_route_tangents)
            )
            route_cache_due = (
                backend.certified_route_depth_cache_stride > 0
                and idx % backend.certified_route_depth_cache_stride == 0
            )
            if route_cache_due or dense_query_due:
                if prediction is None:
                    with torch.no_grad(), torch.autocast(
                            "cuda", dtype=torch.bfloat16):
                        prediction = backend.lb.model._predict_depth(
                            agg, img[None, None].to(backend.device), self.patch_start_index)
                route_depth = prediction[
                    "depth"][0, -1, ..., 0].float().cpu().numpy()
                route_confidence = prediction[
                    "depth_conf"][0, -1].float().cpu().numpy()
                if route_cache_due:
                    self.online_depths[idx] = (
                        route_depth.copy(), route_confidence.copy())
                    self.online_depth_frames.add(idx)
                if dense_query_due and not route_cache_due:
                    backend._certified_route_live_depth_cache[idx] = (
                        route_depth.copy(), route_confidence.copy())
                    # Formal execution replans every eight actions.  Keep two
                    # intervals defensively; older dense values have already
                    # been consumed and are not long-term memory.
                    oldest = idx - 16
                    for stale in tuple(
                            backend._certified_route_live_depth_cache):
                        if stale < oldest:
                            del backend._certified_route_live_depth_cache[stale]
            if prediction is not None:
                self.current_relative_depth = prediction[
                    "depth"][0, -1, ..., 0].float().detach()
                self.current_depth_frame = int(idx)
            if (backend.certified_eager_depth_cache
                    and backend._certified_dense_stream_snapshot is not None):
                backend._update_certified_eager_depth(idx, img)
        self.frame_count += 1
        if self.frame_count == 40:
            backend._freeze_first40_scale()
        return idx

    @torch.no_grad()
    def freeze_metric_scale(self):
        """Freeze the sole causal RGB-only scale receipt for this episode.

        LingBot's scale routine replays the prefix and clears its KV caches, so
        the live map stream is restored exactly afterwards.  Any failure is a
        frozen invalid receipt; it can only yield zero depth, never a pooled or
        oracle fallback.
        """
        backend = self.backend

        if self.metric_scale_receipt is not None:
            raise RuntimeError("first-40 metric scale was already frozen")
        if self.frame_count != 40 or len(self.poses) != 40:
            raise RuntimeError(
                "first-40 scale freeze requires exactly 40 live observations"
            )
        from MemNavData.monocular_depth_runtime import (
            compute_first40_scale_receipt,
            failed_first40_scale_receipt,
        )

        snapshot = backend._snapshot()
        saved_dino_output = backend._dino_out[0]
        started = time.perf_counter()
        try:
            self.metric_scale_receipt = compute_first40_scale_receipt(
                backend.lb,
                self.rgb_dir,
                torch.stack(self.poses, 0).float().cpu().numpy(),
                backend.camera_height,
            )
        except Exception as error:
            self.metric_scale_receipt = failed_first40_scale_receipt(
                backend.camera_height,
                f"{type(error).__name__}: {error}",
            )
        finally:
            backend._restore(snapshot, empty_cuda_cache=False)
            backend._dino_out[0] = saved_dino_output
            self.metric_scale_freeze_ms = (
                1000.0 * (time.perf_counter() - started)
            )

    def snapshot_stream(self):
        # Snapshot by REFERENCE, not clone: plan-time ops (window_forward /
        # goal_append_warm / camera_pose) start with clean_kv_cache + _inject,
        # which REPLACE dict entries — they never mutate the existing KV tensors
        # in place. Holding references keeps the old tensors alive at zero copy
        # cost (a full clone of the 32-frame window KV is ~5.5 GB and OOMs).
        backend = self.backend
        agg = backend.lb.agg
        ch = backend.lb.model.camera_head
        return dict(
            kv=dict(agg.kv_cache),
            total=int(agg.total_frames_processed),
            cam=list(ch.kv_cache) if ch.kv_cache is not None else None,
            cam_idx=int(getattr(ch, "frame_idx", 0)),
        )

    def restore_stream(self, snap, *, empty_cuda_cache=True):
        backend = self.backend
        agg = backend.lb.agg
        ch = backend.lb.model.camera_head
        backend.lb.model.clean_kv_cache()
        agg.kv_cache.update(snap["kv"])
        agg.total_frames_processed = snap["total"]
        ch.kv_cache = snap["cam"]
        ch.frame_idx = snap["cam_idx"]
        if empty_cuda_cache:
            torch.cuda.empty_cache()

    def planning_cache(self):
        """The in-memory equivalent of MemNavNet._load_cache's dict."""
        backend = self.backend
        if getattr(backend, "depth_observation_only", False):
            raise RuntimeError(
                "depth-observation-only mode does not materialize planning "
                "caches")
        n_anchor = len(self.anchor_k)
        if n_anchor > 0:
            ak = torch.stack(self.anchor_k, 2)          # [L,H,N,6,d]
            av = torch.stack(self.anchor_v, 2)
        else:
            L, H, d = self.scale_k.shape[0], self.scale_k.shape[1], self.scale_k.shape[-1]
            ak = self.scale_k.new_zeros((L, H, 0, backend.psi, d))
            av = self.scale_k.new_zeros((L, H, 0, backend.psi, d))
        cache = dict(
            scale_k=self.scale_k, scale_v=self.scale_v,
            anchor_k=ak, anchor_v=av,
            cam_k=torch.stack(self.camera_k, 0), cam_v=torch.stack(self.camera_v, 0),
            cam_pose_enc=torch.stack(self.poses, 0).to(backend.device),
            ground_h_est=None,
        )
        if backend.flow_threshold > 0:
            cache["anchor_frame_indices"] = torch.as_tensor(
                self.anchor_frames, dtype=torch.long
            )
            cache["cam_frame_indices"] = torch.as_tensor(
                self.camera_frames, dtype=torch.long
            )
        return cache
