"""MemNav live closed-loop agent.

Wraps the trained `MemNavPolicy` (InternNav, branch main) for frame-by-frame
inference: the agent ingests RGB frames one at a time (building the same
LingBot KV caches the training precompute writes to disk, incrementally and
in-memory), and on request plans a trajectory toward a goal image.

Design rule: every op is the TRAINING op. The live capture stream mirrors
`scripts/dataset_converters/precompute_lingbot_features.py:extract_trajectory`
step by step (scale block -> per-frame stream, write-once capture of the
newest cache slot); planning mirrors `MemNavNet.encode_memory` + the DDPM
reverse loop implied by its scheduler config. Where training reads an npz,
we read the identical in-memory dict.

Plan-time LingBot ops (window_forward / goal_append_warm / camera_pose)
destroy the streaming KV state, so plan() snapshots the aggregator +
camera-head caches and restores them afterwards — the capture stream stays
continuous, which is what makes the captured cam_pose_enc match precompute's.

Run inside the `memnav` conda env. Requires InternNav on sys.path (the
server adds it).
"""

import os
import shutil

import numpy as np
import torch

try:  # package import in tests; script-local import in memnav_server.py
    from .pose_alignment import lingbot_relative_yaw
    from .router_candidates import temporal_nms_candidates
except ImportError:  # pragma: no cover - exercised by the live script entrypoint
    from pose_alignment import lingbot_relative_yaw
    from router_candidates import temporal_nms_candidates


FLOW_TIERS = [(702, 20.0), (877, 25.0), (1075, 30.0), (1506, 40.0), (2048, 50.0)]
FLOW_GAP = 30


def flow_threshold_for_length(n_frames):
    """Match the length-tiered sparse-keyframe policy used by precompute."""
    for upper, threshold in FLOW_TIERS:
        if n_frames <= upper:
            return threshold
    return 60.0


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
class MemNavAgent:
    def __init__(self, checkpoint, internnav_root, device="cuda:0",
                 exclude_recent=83, num_samples=16, buffer_root=None,
                 gate_skip_below=0.0, retrieval_mode="raw", anchor_switch_margin=0.01,
                 flow_gate="auto", retrieval_candidate_top_k=32,
                 retrieval_candidate_min_gap=16):
        # auto: training's per-episode tier; off: legacy dense capture; otherwise
        # parse a fixed pixel-flow threshold.
        self.flow_gate = flow_gate
        import sys
        if internnav_root not in sys.path:
            sys.path.insert(0, internnav_root)
        from internnav.model.basemodel.memnav.memnav_policy import (   # noqa: E402
            MemNavModelConfig, MemNavPolicy)
        from scripts.train.configs.memnav import memnav_exp_cfg        # noqa: E402

        self.policy = MemNavPolicy.from_pretrained(
            checkpoint, config=MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump()))
        self.policy.eval()
        self.core = self.policy.core
        self.lb = self.core.lingbot
        self.device = self.core.device
        self.S = self.lb.num_scale                      # 8
        self.W = self.lb.window                         # 32 (mp3d geometry)
        self.amargin = self.S + self.W - 1              # 39
        self.exclude_recent = int(exclude_recent)       # dataset default 83
        self.num_samples = int(num_samples)
        self.gate_skip_below = float(gate_skip_below)
        # "raw": match by RAW dino-cls cosine (frozen features; measured corr +0.29 with
        # GT covis, top-5 all in the GT neighborhood) instead of the trained projection
        # (measured corr -0.75 at ckpt-1500, top-5 all covis~0). Gate stays the trained
        # head's (decoder conditioning must match training).
        self.retrieval_mode = retrieval_mode
        # anchor hysteresis: per-frame scores are STATIC (goal fixed, cls write-once),
        # so the argmax moves only when a NEW candidate beats the incumbent. Switch
        # only on a clear win to keep novel-goal anchors sticky (no wasted warms).
        self.anchor_switch_margin = float(anchor_switch_margin)
        self.retrieval_candidate_top_k = int(retrieval_candidate_top_k)
        self.retrieval_candidate_min_gap = int(retrieval_candidate_min_gap)
        if (self.retrieval_candidate_top_k < 1
                or self.retrieval_candidate_min_gap < 1):
            raise ValueError("retrieval candidate top-k and gap must be positive")
        self.L_depth = self.lb.depth                    # aggregator layers
        self.psi = self.lb.num_special                  # 6 special tokens

        self.buffer_root = buffer_root or "/tmp/memnav_agent_buffer"
        os.makedirs(self.buffer_root, exist_ok=True)
        self._episode_counter = -1

        # dino-cls capture hook on the aggregator's patch_embed — same values the
        # precompute stores (`x_norm_clstoken` of the streaming forward itself).
        self._dino_out = [None]

        def _hook(_m, _i, out):
            self._dino_out[0] = out
        self.lb.agg.patch_embed.register_forward_hook(_hook)

        self.reset(camera_height=0.5)

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, camera_height=0.5, seed=None, episode_len=None,
              camera_intrinsic=None):
        # Reset diffusion randomness per episode so terminal-mode A/B runs have
        # an identical navigation prefix.  This does not force deterministic
        # CUDA kernels; it controls the explicit torch.randn DDPM start noise.
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self._episode_counter += 1
        self.rgb_dir = os.path.join(self.buffer_root, f"ep_{self._episode_counter:04d}")
        shutil.rmtree(self.rgb_dir, ignore_errors=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        self.camera_height = float(camera_height)
        self.camera_intrinsic = None
        if camera_intrinsic is not None:
            intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError("camera_intrinsic must be a finite 3x3 matrix")
            self.camera_intrinsic = intrinsic
        if self.flow_gate == "off":
            self.flow_threshold = 0.0
        elif self.flow_gate == "auto":
            self.flow_threshold = (
                flow_threshold_for_length(int(episode_len))
                if episode_len else FLOW_TIERS[0][1]
            )
        else:
            self.flow_threshold = float(self.flow_gate)
        self.flow_gap = FLOW_GAP
        self._last_episode_len = episode_len
        self._last_seed = seed
        self._last_kf_pose = None
        self._last_kf_idx = self.S - 1
        self.anchor_frame_indices = []
        self.cam_frame_indices = []
        self.n = 0                       # frames streamed so far
        self._pending = []               # preprocessed frames waiting for the scale block
        self._window_imgs = []           # last W preprocessed frames (cpu), for window_forward
        self.dino_cls = []               # per-frame [1024] fp32 cpu
        self.anchor_k = []               # per phase-2 frame [L,H,6,d] bf16 gpu
        self.anchor_v = []
        self.cam_k = []                  # per-frame [NI,TD,H,d] bf16 gpu (stacked lazily)
        self.cam_v = []
        self.cam_pose = []               # per-frame [9] fp32
        self.scale_k = None              # [L,H,S,P,d] bf16 gpu
        self.scale_v = None
        self._metric_scale = None        # lazy ground-anchored scale
        self._goal_cache = {}            # (goal_md5, anchor) -> goal_pose; goal_md5 -> goal_cls
        self._anchor_state = {}          # goal_md5 -> dict(m, score): sticky-anchor ratchet
        self._goal_start_frame = {}      # goal_md5 -> first frame queried for this goal
        # SIFT/essential verification is a deterministic function of the goal,
        # immutable history image, and per-episode intrinsic.  Cache both
        # positive and negative results so temporal confirmation checks anchor
        # stability instead of recomputing the identical image pair.
        self._retrieval_verification_cache = {}
        # tower-1 live capture: the current frame's post-GCT tokens + agg list from the
        # CONTINUOUS stream. Training used window_forward's cold-cache recompute only
        # because samples load from disk; at eval the live stream supersedes it.
        self._last_tokens = None         # [1, P, 2C] current frame post-GCT tokens
        self._last_agg = None            # list of [1,1,P,2C] (selected layers, current frame)
        self._psi = None                 # patch_start_idx from the scale block
        self.lb.model.clean_kv_cache()
        self.lb.model.camera_head.clean_kv_cache()

    @torch.no_grad()
    def image_goal_similarity(self, image_jpg_bytes, goal_jpg_bytes):
        """Stateless raw-DINO cosine for terminal visual verification.

        Unlike ``plan()``, this neither appends the image to streaming memory
        nor runs retrieval.  Terminal-motion frames therefore cannot produce a
        trivial near-self retrieval match at the verification step.
        """
        import hashlib

        current_path = os.path.join(self.rgb_dir, "_verify_current.jpg")
        with open(current_path, "wb") as f:
            f.write(image_jpg_bytes)
        current_img = self.lb.load_images([current_path])[0][None].to(self.device)
        current_cls = self.lb.dino(current_img)["cls"]

        gkey = hashlib.md5(goal_jpg_bytes).hexdigest()
        goal_cls = self._goal_cache.get(("cls", gkey))
        if goal_cls is None:
            goal_path = os.path.join(self.rgb_dir, "_verify_goal.jpg")
            with open(goal_path, "wb") as f:
                f.write(goal_jpg_bytes)
            goal_img = self.lb.load_images([goal_path])[0][None].to(self.device)
            goal_cls = self.lb.dino(goal_img)["cls"]
            self._goal_cache[("cls", gkey)] = goal_cls

        return float(torch.nn.functional.cosine_similarity(
            current_cls, goal_cls, dim=-1)[0].item())

    def verify_retrieval_overlap(self, goal_jpg_bytes, anchor):
        """CPU SIFT/epipolar verification for one retrieved history frame.

        DINO supplies a high-recall candidate. This second stage rejects a
        similar-looking but geometrically unrelated corridor before the
        memory route is allowed to latch. Streaming model state is untouched.
        """
        import hashlib
        import time

        started = time.perf_counter()
        anchor = int(anchor)
        empty = dict(matches=0, inliers=0, inlier_ratio=0.0, anchor=anchor)
        if anchor < 0 or anchor >= self.n:
            return dict(
                empty, error=f"anchor {anchor} outside [0, {self.n - 1}]",
                cached=False,
                verification_ms=1000.0 * (time.perf_counter() - started))
        anchor_path = os.path.join(self.rgb_dir, f"{anchor}.jpg")
        if not os.path.isfile(anchor_path):
            return dict(
                empty, error=f"anchor image missing: {anchor_path}",
                cached=False,
                verification_ms=1000.0 * (time.perf_counter() - started))

        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        cache_key = (goal_key, anchor)
        cached = self._retrieval_verification_cache.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["cached"] = True
            result["verification_ms"] = 1000.0 * (
                time.perf_counter() - started)
            return result

        def finish(result):
            result = dict(result)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            result.update(
                cached=False,
                verification_ms=elapsed_ms,
                uncached_verification_ms=elapsed_ms,
            )
            self._retrieval_verification_cache[cache_key] = dict(result)
            return result

        try:
            import cv2
            anchor_gray = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
            goal_gray = cv2.imdecode(
                np.frombuffer(goal_jpg_bytes, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if anchor_gray is None or goal_gray is None:
                return finish(dict(empty, error="image decode failed"))
            sift = cv2.SIFT_create(nfeatures=4000)
            anchor_kp, anchor_desc = sift.detectAndCompute(anchor_gray, None)
            goal_kp, goal_desc = sift.detectAndCompute(goal_gray, None)
            if anchor_desc is None or goal_desc is None:
                return finish(dict(
                    empty, error="insufficient image features"))
            pairs = cv2.BFMatcher().knnMatch(anchor_desc, goal_desc, k=2)
            good = [pair[0] for pair in pairs
                    if len(pair) == 2
                    and pair[0].distance < 0.75 * pair[1].distance]
            matches = len(good)
            base = dict(matches=matches, inliers=0, inlier_ratio=0.0,
                        anchor=anchor)
            if matches < 8:
                return finish(dict(
                    base, error="too few ratio-test matches"))
            anchor_pts = np.float32([anchor_kp[m.queryIdx].pt for m in good])
            goal_pts = np.float32([goal_kp[m.trainIdx].pt for m in good])
            inliers = 0
            if self.camera_intrinsic is not None:
                essential, ransac_mask = cv2.findEssentialMat(
                    anchor_pts, goal_pts, self.camera_intrinsic,
                    cv2.RANSAC, 0.999, 1.5)
                if essential is not None:
                    essential = np.asarray(essential, dtype=np.float64)
                    if essential.shape == (3, 3):
                        candidates = [essential]
                    elif (essential.ndim == 2 and essential.shape[1] == 3
                          and essential.shape[0] % 3 == 0):
                        candidates = [essential[i:i + 3]
                                      for i in range(0, essential.shape[0], 3)]
                    else:
                        candidates = []
                    for candidate in candidates:
                        mask = (None if ransac_mask is None
                                else ransac_mask.copy())
                        try:
                            recovered, _rotation, _translation, _pose_mask = (
                                cv2.recoverPose(
                                    candidate, anchor_pts, goal_pts,
                                    self.camera_intrinsic, mask=mask))
                            inliers = max(inliers, int(recovered))
                        except cv2.error:
                            continue
            else:
                _fundamental, mask = cv2.findFundamentalMat(
                    anchor_pts, goal_pts, cv2.FM_RANSAC, 1.5, 0.999)
                if mask is not None:
                    inliers = int(np.asarray(mask).astype(bool).sum())
            return finish(dict(
                error=None,
                matches=matches,
                inliers=inliers,
                inlier_ratio=float(inliers / matches),
                anchor=anchor,
            ))
        except Exception as exc:
            return finish(dict(
                empty, error=f"overlap verification failed: {exc}"))

    # ------------------------------------------------------------------ #
    # capture-stream internals (mirrors precompute extract_trajectory)
    # ------------------------------------------------------------------ #
    def _pop_cls(self, n_frames):
        out = self._dino_out[0]
        cls = out["x_norm_clstoken"].reshape(n_frames, -1).float().cpu()
        return [cls[i] for i in range(n_frames)]

    def _read_anchor_newest(self):
        kv = self.lb.agg.kv_cache
        ak = torch.stack([kv[f"k_{i}"][0, :, -1, :self.psi].to(torch.bfloat16)
                          for i in range(self.L_depth)])
        av = torch.stack([kv[f"v_{i}"][0, :, -1, :self.psi].to(torch.bfloat16)
                          for i in range(self.L_depth)])
        return ak, av                                   # [L,H,6,d]

    def _read_cam_newest(self, n_new):
        ch = self.lb.model.camera_head
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

    def add_frame(self, jpg_bytes):
        """Ingest one RGB frame (jpg bytes). Returns the frame index."""
        idx = self.n
        path = os.path.join(self.rgb_dir, f"{idx}.jpg")
        with open(path, "wb") as f:
            f.write(jpg_bytes)
        img = self.lb.load_images([path])[0]            # [3,518,518] pad-518 (cpu)
        self._window_imgs.append(img)
        if len(self._window_imgs) > self.W:
            self._window_imgs.pop(0)

        ch = self.lb.model.camera_head
        if idx < self.S - 1:
            self._pending.append(img)
            self.n += 1
            return idx
        if idx == self.S - 1:
            # scale block: first S frames as ONE bidirectional block
            self._pending.append(img)
            blk = torch.stack(self._pending, 0)[None].to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, psi = self.lb.model._aggregate_features(
                    blk, num_frame_for_scale=self.S, num_frame_per_block=self.S)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=self.S, num_frame_for_scale=self.S)
            self._psi = psi
            self._last_tokens = agg[-1][:, -1]
            self._last_agg = [layer[:, -1:] for layer in agg]
            self.dino_cls.extend(self._pop_cls(self.S))
            kv = self.lb.agg.kv_cache
            self.scale_k = torch.stack([kv[f"k_{i}"][0, :, :self.S].to(torch.bfloat16)
                                        for i in range(self.L_depth)]).contiguous()
            self.scale_v = torch.stack([kv[f"v_{i}"][0, :, :self.S].to(torch.bfloat16)
                                        for i in range(self.L_depth)]).contiguous()
            pose = pl[-1][0].float()                    # [S,9]
            self.cam_pose.extend([pose[i].cpu() for i in range(self.S)])
            ck, cv = self._read_cam_newest(self.S)
            self.cam_k.extend(ck); self.cam_v.extend(cv)
            self._last_kf_pose = pl[-1][:, -1:].float()
            self._last_kf_idx = self.S - 1
            self.cam_frame_indices = list(range(self.S))
            self._pending = []
        else:
            # The live stream always evaluates the newest frame. When the flow
            # policy rejects it, roll back only the stored KV append; dense cls,
            # pose, and current-state tokens remain aligned to raw frame indices.
            agg_mod = self.lb.agg
            gate_on = self.flow_threshold > 0
            if gate_on:
                saved_kv = dict(agg_mod.kv_cache)
                saved_cam = [dict(layer) for layer in ch.kv_cache]
                saved_total = int(agg_mod.total_frames_processed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, _ = self.lb.model._aggregate_features(
                    img[None, None].to(self.device),
                    num_frame_for_scale=self.S, num_frame_per_block=1)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=1, num_frame_for_scale=self.S)
            self._last_tokens = agg[-1][:, -1]
            self._last_agg = [layer for layer in agg]
            self.dino_cls.extend(self._pop_cls(1))
            self.cam_pose.append(pl[-1][0].float()[-1].cpu())

            if gate_on:
                cur_pose = pl[-1][:, -1:].float()
                if idx == self.S:
                    is_keyframe = True
                else:
                    from lingbot_map.models.gct_stream_window import _compute_flow_magnitude
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        depth = self.lb.model._predict_depth(
                            agg, img[None, None].to(self.device), self._psi
                        )["depth"].float()
                    flow = _compute_flow_magnitude(
                        cur_pose, self._last_kf_pose, depth, tuple(depth.shape[2:4])
                    )
                    is_keyframe = (
                        flow > self.flow_threshold
                        or (idx - self._last_kf_idx) >= self.flow_gap
                    )
            else:
                is_keyframe = True

            if is_keyframe:
                if gate_on:
                    self._last_kf_pose = cur_pose
                    self._last_kf_idx = idx
                ak, av = self._read_anchor_newest()
                self.anchor_k.append(ak); self.anchor_v.append(av)
                self.anchor_frame_indices.append(idx)
                ck, cv = self._read_cam_newest(1)
                self.cam_k.extend(ck); self.cam_v.extend(cv)
                self.cam_frame_indices.append(idx)
            else:
                agg_mod.kv_cache.clear()
                agg_mod.kv_cache.update(saved_kv)
                ch.kv_cache = saved_cam
                agg_mod.total_frames_processed = saved_total
        self.n += 1
        return idx

    # ------------------------------------------------------------------ #
    # stream-state snapshot (plan ops destroy the KV caches)
    # ------------------------------------------------------------------ #
    def _snapshot(self):
        # Snapshot by REFERENCE, not clone: plan-time ops (window_forward /
        # goal_append_warm / camera_pose) start with clean_kv_cache + _inject,
        # which REPLACE dict entries — they never mutate the existing KV tensors
        # in place. Holding references keeps the old tensors alive at zero copy
        # cost (a full clone of the 32-frame window KV is ~5.5 GB and OOMs).
        agg = self.lb.agg
        ch = self.lb.model.camera_head
        return dict(
            kv=dict(agg.kv_cache),
            total=int(agg.total_frames_processed),
            cam=list(ch.kv_cache) if ch.kv_cache is not None else None,
            cam_idx=int(getattr(ch, "frame_idx", 0)),
        )

    def _restore(self, snap):
        agg = self.lb.agg
        ch = self.lb.model.camera_head
        self.lb.model.clean_kv_cache()
        agg.kv_cache.update(snap["kv"])
        agg.total_frames_processed = snap["total"]
        ch.kv_cache = snap["cam"]
        ch.frame_idx = snap["cam_idx"]
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def _live_cache(self):
        """The in-memory equivalent of MemNavNet._load_cache's dict."""
        n_anchor = len(self.anchor_k)
        if n_anchor > 0:
            ak = torch.stack(self.anchor_k, 2)          # [L,H,N,6,d]
            av = torch.stack(self.anchor_v, 2)
        else:
            L, H, d = self.scale_k.shape[0], self.scale_k.shape[1], self.scale_k.shape[-1]
            ak = self.scale_k.new_zeros((L, H, 0, self.psi, d))
            av = self.scale_k.new_zeros((L, H, 0, self.psi, d))
        cache = dict(
            scale_k=self.scale_k, scale_v=self.scale_v,
            anchor_k=ak, anchor_v=av,
            cam_k=torch.stack(self.cam_k, 0), cam_v=torch.stack(self.cam_v, 0),
            cam_pose_enc=torch.stack(self.cam_pose, 0).to(self.device),
            ground_h_est=None,
        )
        if self.flow_threshold > 0:
            cache["anchor_frame_indices"] = torch.as_tensor(
                self.anchor_frame_indices, dtype=torch.long
            )
            cache["cam_frame_indices"] = torch.as_tensor(
                self.cam_frame_indices, dtype=torch.long
            )
        return cache

    def _get_metric_scale(self):
        if self._metric_scale is None and self.n >= self.S:
            cam_pose = torch.stack(self.cam_pose, 0)
            s = self.lb.get_metric_scale(self.rgb_dir, cam_pose, self.camera_height)
            from internnav.model.basemodel.memnav.memnav_policy import RevisitMerge
            self._metric_scale = s if s is not None else RevisitMerge._SCALE
        return self._metric_scale

    @torch.no_grad()
    def plan(self, goal_jpg_bytes, forced_anchor=None, forced_gate=None,
             pose_only=False, retrieval_only=False):
        """Plan toward a goal image. Returns dict with metre-space waypoints in the
        current camera planar frame (x forward, y left, theta CCW).

        ``forced_anchor`` and ``forced_gate`` are evaluation-only oracle
        interventions. They change
        only the history frame used by the revisit tower; retrieval logits,
        gate, current-state tower, novel tower, and diffusion decoding remain
        unchanged. The default ``None`` preserves deployment behavior.

        ``pose_only`` stops after retrieval and LingBot relative-pose recovery.
        It is used by the hybrid controller, where the frozen NavDP decoder
        consumes this metric point-goal; running MemNav's second diffusion
        decoder in that path would only waste compute.

        ``retrieval_only`` stops before goal-pose recovery. It is the cheap
        first stage of a reliability router: Novel goals can be rejected from
        memory using DINO/retrieval scores without allocating goal-append pose
        caches that will never control the robot.
        """
        k = self.n - 1
        lo = self.amargin
        import hashlib
        gkey = hashlib.md5(goal_jpg_bytes).hexdigest()
        goal_start_frame = self._goal_start_frame.setdefault(gkey, k)
        candidate_ceiling = goal_start_frame - 1
        if k < self.S + self.W:
            out = dict(
                error=f"need >= {self.S + self.W + 1} frames, have {self.n}")
            if pose_only or retrieval_only:
                # posegoal_step has already appended this observation. Return
                # its index even before retrieval becomes legal so the caller's
                # online frame-to-pose trace remains complete.
                out.update(
                    frame_idx=k, candidate_count=0,
                    goal_start_frame=goal_start_frame,
                    candidate_ceiling=candidate_ceiling,
                )
            return out

        gpath = os.path.join(self.rgb_dir, "_goal.jpg")
        with open(gpath, "wb") as f:
            f.write(goal_jpg_bytes)
        goal_img = self.lb.load_images([gpath])[0]      # [3,518,518]

        snap = self._snapshot()
        try:
            cache = self._live_cache()
            dev = self.device
            goal_t = goal_img[None].to(dev)

            # retrieval over candidates E(k) = [amargin .. k - exclude_recent]
            if ("cls", gkey) not in self._goal_cache:
                self._goal_cache[("cls", gkey)] = self.lb.dino(goal_t)["cls"]
            goal_cls = self._goal_cache[("cls", gkey)]                   # [1,1024]
            mem_cls = torch.stack(self.dino_cls, 0)[None].to(dev)        # [1,k+1,1024]
            current_goal_cos = float(torch.nn.functional.cosine_similarity(
                goal_cls, mem_cls[:, k], dim=-1)[0].item())
            cand = torch.zeros(1, k + 1, dtype=torch.bool, device=dev)
            # Let frames near the goal-session boundary become eligible after
            # exclude_recent time has elapsed, but never admit observations
            # collected while pursuing this same goal. Without this ceiling a
            # long revisit eventually retrieves its own recent return path.
            hi = min(k - self.exclude_recent, candidate_ceiling)
            if hi >= lo:
                cand[0, lo:hi + 1] = True
            candidate_count = int(cand.sum().item())
            match_idx, gate_logit, _ = self.core.retrieval(goal_cls, mem_cls, cand)
            gate = torch.sigmoid(gate_logit)     # trained gate: decoder soft-bias, as in training
            predicted_gate = float(gate.item())
            if forced_gate is not None:
                forced_gate = float(forced_gate)
                if not 0.0 <= forced_gate <= 1.0:
                    return dict(error=f"forced gate {forced_gate} outside [0, 1]")
                gate = torch.full_like(gate, forced_gate)
            if (pose_only or retrieval_only) and not cand.any():
                return dict(
                    error=(f"no eligible retrieval frame in [{lo}, {hi}] "
                           f"at current frame {k}"),
                    gate=float(gate.item()),
                    predicted_gate=predicted_gate,
                    forced_gate=(float(forced_gate)
                                 if forced_gate is not None else None),
                    current_goal_cos=current_goal_cos,
                    candidate_count=candidate_count,
                    goal_start_frame=goal_start_frame,
                    candidate_ceiling=candidate_ceiling,
                    frame_idx=k,
                )

            raw_score = None
            retrieval_second_score = None
            retrieval_margin = None
            visual_score = None
            visual_second_score = None
            visual_margin = None
            visual_anchor = None
            visual_candidates = []
            raw_cos = None
            if cand.any():
                import torch.nn.functional as Fnn
                visual_cos = Fnn.cosine_similarity(
                    goal_cls.unsqueeze(1), mem_cls, dim=-1)[0]
                if self.retrieval_mode == "raw":
                    raw_cos = visual_cos
                else:
                    goal_proj = Fnn.normalize(
                        self.core.retrieval.proj_goal(goal_cls), dim=-1
                    )
                    memory_proj = Fnn.normalize(
                        self.core.retrieval.proj_mem(mem_cls), dim=-1
                    )
                    raw_cos = (goal_proj.unsqueeze(1) * memory_proj).sum(-1)[0]
                raw_cos = raw_cos.masked_fill(~cand[0], -1.0)
                cand_best = int(raw_cos.argmax().item())
                raw_score = float(raw_cos[cand_best].item())
                raw_top = torch.topk(
                    raw_cos[cand[0]], k=min(2, candidate_count)).values
                if candidate_count > 1:
                    retrieval_second_score = float(raw_top[1].item())
                    retrieval_margin = raw_score - retrieval_second_score

                visual_valid = visual_cos[cand[0]]
                visual_indices = torch.nonzero(
                    cand[0], as_tuple=False).flatten()
                visual_anchor = int(
                    visual_indices[visual_valid.argmax()].item())
                visual_top = torch.topk(
                    visual_valid, k=min(2, candidate_count)).values
                visual_score = float(visual_top[0].item())
                if candidate_count > 1:
                    visual_second_score = float(visual_top[1].item())
                    visual_margin = visual_score - visual_second_score
                visual_candidates = temporal_nms_candidates(
                    visual_cos.detach().float().cpu().tolist(),
                    cand[0].detach().cpu().tolist(),
                    top_k=self.retrieval_candidate_top_k,
                    min_frame_gap=self.retrieval_candidate_min_gap,
                )
                st = self._anchor_state.get(gkey)
                # ratchet: keep the incumbent unless the new best clearly beats it
                if st is not None and raw_score <= st["score"] + self.anchor_switch_margin:
                    match = st["m"]
                else:
                    match = cand_best
                    self._anchor_state[gkey] = dict(m=cand_best, score=raw_score)
                match_idx = torch.tensor([match], device=dev)
            retrieved_anchor = int(match_idx.clamp(lo, k - 1).item())
            forced_anchor_score = None
            if forced_anchor is not None:
                forced_anchor = int(forced_anchor)
                if forced_anchor < lo or forced_anchor > hi:
                    return dict(
                        error=(f"forced anchor {forced_anchor} outside eligible "
                               f"range [{lo}, {hi}]"),
                        gate=float(gate.item()),
                        predicted_gate=predicted_gate,
                        current_goal_cos=current_goal_cos,
                        candidate_count=candidate_count,
                        raw_score=raw_score,
                        retrieval_second_score=retrieval_second_score,
                        retrieval_margin=retrieval_margin,
                        visual_score=visual_score,
                        visual_anchor=visual_anchor,
                        visual_second_score=visual_second_score,
                        visual_margin=visual_margin,
                        goal_start_frame=goal_start_frame,
                        candidate_ceiling=candidate_ceiling,
                        frame_idx=k,
                    )
                match_idx = torch.tensor([forced_anchor], device=dev)
                if raw_cos is None:
                    import torch.nn.functional as Fnn
                    raw_cos = Fnn.cosine_similarity(
                        goal_cls.unsqueeze(1), mem_cls, dim=-1)[0]
                forced_anchor_score = float(raw_cos[forced_anchor].item())
            anchor = int(match_idx.clamp(lo, k - 1).item())
            selected_anchor_score = (float(raw_cos[anchor].item())
                                     if raw_cos is not None else None)
            anchor_gap = k - anchor

            if retrieval_only:
                return dict(
                    gate=float(gate.item()),
                    predicted_gate=predicted_gate,
                    forced_gate=(float(forced_gate)
                                 if forced_gate is not None else None),
                    match_idx=int(match_idx.item()),
                    raw_score=raw_score,
                    retrieval_second_score=retrieval_second_score,
                    retrieval_margin=retrieval_margin,
                    visual_score=visual_score,
                    visual_anchor=visual_anchor,
                    visual_second_score=visual_second_score,
                    visual_margin=visual_margin,
                    visual_candidates=visual_candidates,
                    visual_candidate_min_gap=self.retrieval_candidate_min_gap,
                    selected_anchor_score=selected_anchor_score,
                    candidate_count=candidate_count,
                    anchor_gap=anchor_gap,
                    goal_start_frame=goal_start_frame,
                    candidate_ceiling=candidate_ceiling,
                    retrieved_anchor=retrieved_anchor,
                    forced_anchor=(int(forced_anchor)
                                   if forced_anchor is not None else None),
                    forced_anchor_score=forced_anchor_score,
                    anchor=anchor,
                    aux_pose=None,
                    goal_rel_yaw=None,
                    current_goal_cos=current_goal_cos,
                    frame_idx=k,
                )

            # poses: current from the continuous capture stream; goal via warm re-insert.
            # goal_pose depends only on (goal image, anchor, caches[<=anchor]) and the
            # captured caches are write-once, so this cache is EXACT — recompute only
            # when retrieval moves the anchor (saves the 64-frame warm ~10s per plan).
            cur_pose = cache["cam_pose_enc"][k][None]                    # [1,9]
            pkey = ("pose", gkey, anchor)
            # Gate-conditioned tower 2: only pay the goal-insert when retrieval says
            # revisit (or the pose is already cached). When skipped, the revisit
            # readout is zeroed — NOTE this is a deliberate eval-time deviation:
            # training always computes revisit tokens and lets the soft gate
            # (log(gate) attention bias) mask them; zeroing changes what low-gate
            # steps condition on. Revisit if warm-arm results look off.
            goal_pose = self._goal_cache.get(pkey)
            if goal_pose is None and (pose_only
                                      or float(gate.item()) >= self.gate_skip_below):
                # warm all the way back to the scale block (n_hist=0). Injected
                # compressed-history frames poison the camera head's goal pose
                # (measured: n_hist=34 -> 34° yaw err, 100 -> 169°; 0 -> ~0°
                # — Aiden_eval/memnav_eval/FINDINGS.md §2.5/§3). Cost is once
                # per (goal, m) via the cache, not per step.
                warm_full = max(self.core.goal_warm, anchor - self.S + 1)
                _, goal_agg = self.lb.goal_append_warm(
                    goal_img, cache, anchor, self.rgb_dir, warm_full, return_agg=True)
                goal_pose = self.lb.camera_pose(
                    cache["cam_k"], cache["cam_v"], anchor + 1, goal_agg,
                    cam_frame_indices=cache.get("cam_frame_indices"))[-1][None]
                self._goal_cache[pkey] = goal_pose

            mscale = torch.tensor([self._get_metric_scale()], device=dev, dtype=torch.float32)
            if goal_pose is not None:
                revisit, aux_pose, relative_rotation = self.core.build_revisit(
                    cur_pose.to(dev), goal_pose.to(dev), mscale)
                goal_rel_yaw = lingbot_relative_yaw(
                    relative_rotation[0].float().cpu().numpy())
            else:
                revisit = torch.zeros((1, self.core.n_rev, self.core.action_head.in_features), device=dev)
                aux_pose = torch.zeros((1, 2), device=dev)
                goal_rel_yaw = None
            if pose_only:
                return dict(
                    gate=float(gate.item()),
                    predicted_gate=predicted_gate,
                    forced_gate=(float(forced_gate)
                                 if forced_gate is not None else None),
                    match_idx=int(match_idx.item()),
                    raw_score=raw_score,
                    retrieval_second_score=retrieval_second_score,
                    retrieval_margin=retrieval_margin,
                    visual_score=visual_score,
                    visual_anchor=visual_anchor,
                    visual_second_score=visual_second_score,
                    visual_margin=visual_margin,
                    selected_anchor_score=selected_anchor_score,
                    candidate_count=candidate_count,
                    anchor_gap=anchor_gap,
                    goal_start_frame=goal_start_frame,
                    candidate_ceiling=candidate_ceiling,
                    retrieved_anchor=retrieved_anchor,
                    forced_anchor=(int(forced_anchor)
                                   if forced_anchor is not None else None),
                    forced_anchor_score=forced_anchor_score,
                    anchor=anchor,
                    aux_pose=aux_pose[0].float().cpu().tolist(),
                    goal_rel_yaw=goal_rel_yaw,
                    current_goal_cos=current_goal_cos,
                    frame_idx=k,
                )

            # Current-state depth features and both diffusion towers are not
            # needed by the pose-only hybrid path. Keep their original order
            # for the legacy MemNav planner, but pay this cost only when its
            # waypoint decoder will actually consume them.
            cur_t = self._last_tokens                                    # [1,P,2C]
            cur_img = self._window_imgs[-1]
            dfeat = self.lb.depth_feature(
                self._last_agg, cur_img[None][None], self._psi)[None]
            current_state = self.core.build_current_state(cur_t, dfeat)
            novel = self.core.novel(cur_img[None].to(dev), goal_t)

            # DDPM reverse loop (no critic in this model)
            N = self.num_samples
            cs = current_state.expand(N, -1, -1)
            rv = revisit.expand(N, -1, -1)
            nv = novel.expand(N, -1, -1)
            gt = gate.expand(N)
            sched = self.core.noise_scheduler
            naction = torch.randn((N, self.core.predict_size, 3), device=dev)
            sched.set_timesteps(sched.config.num_train_timesteps)
            for t in sched.timesteps:
                eps = self.core.predict_noise(naction, t[None].to(dev), cs, rv, nv, gt)
                naction = sched.step(eps, t, naction).prev_sample

            # decode: normalized deltas / 4 -> metres; cumsum -> waypoints
            deltas = (naction / 4.0).float().cpu().numpy()               # [N,24,3]
            paths = np.cumsum(deltas, axis=1)

            # geometric trajectory selection (MEMNAV_COLLISION_SELECT, default on):
            # this model has no critic — NavDP's collision score is instead computed
            # geometrically from the CURRENT view's predicted depth (the map-scale
            # depth head, made metric by the ground scale). One extra _predict_depth
            # on tokens we already hold (_last_agg); pure head, no KV mutation.
            # Any failure (or =0) falls back to the endpoint medoid.
            values, pick = [0.0] * N, None
            if os.environ.get("MEMNAV_COLLISION_SELECT", "1") != "0":
                try:
                    from internnav.model.basemodel.memnav.collision_check import (
                        obstacle_points_from_depth, score_trajectories,
                        select_trajectory)
                    from internnav.model.basemodel.memnav.lingbot_stream import (
                        GROUND_BIAS_CORRECTION)
                    pred = self.lb.model._predict_depth(
                        self._last_agg, cur_img[None][None].to(dev), self._psi)
                    d_cur = pred["depth"][0, -1, ..., 0].float()
                    c_cur = pred["depth_conf"][0, -1].float()
                    fov_v, fov_h = float(cur_pose[0, 7]), float(cur_pose[0, 8])
                    ms = float(mscale.item())
                    # h_est back out of the scale (scale = 1.15*cam_h/h_est) —
                    # exact whenever the ground-scale clamp didn't bind (>95% eps)
                    h_est = GROUND_BIAS_CORRECTION * self.camera_height / ms
                    obs = obstacle_points_from_depth(
                        d_cur, c_cur, fov_v, fov_h, h_est, ms)
                    paths_t = torch.as_tensor(paths, device=obs.device)
                    # Optional diagnostic/controller alignment: one predicted
                    # waypoint represents ``pred_digit=4`` source frames, while
                    # the paired Habitat client replans every 8 frames.  Scoring
                    # all 24 waypoints therefore judges hazards far beyond the
                    # portion that will actually execute before the next plan.
                    # Zero preserves the historical full-horizon behaviour.
                    score_horizon = int(os.environ.get(
                        "MEMNAV_COLLISION_HORIZON_WAYPOINTS", "0"))
                    score_paths = paths_t
                    if score_horizon > 0:
                        score_paths = paths_t[:, :min(score_horizon, paths_t.shape[1])]
                    scores, _ = score_trajectories(score_paths, obs, fov_h)
                    pick = select_trajectory(score_paths, scores)
                    values = scores.float().cpu().tolist()
                except Exception as e:
                    print(f"[MemNavAgent] collision select failed ({e}); "
                          f"falling back to medoid", flush=True)
            if pick is None:
                ends = paths[:, -1, :2]
                pick = int(np.argmin(np.linalg.norm(ends - ends.mean(0), axis=1)))
            return dict(
                trajectory=paths[pick].tolist(),
                all_trajectory=paths.tolist(),
                all_values=values,
                gate=float(gate.item()),
                predicted_gate=predicted_gate,
                forced_gate=(float(forced_gate)
                             if forced_gate is not None else None),
                match_idx=int(match_idx.item()),
                raw_score=raw_score,
                retrieval_second_score=retrieval_second_score,
                retrieval_margin=retrieval_margin,
                visual_score=visual_score,
                visual_anchor=visual_anchor,
                visual_second_score=visual_second_score,
                visual_margin=visual_margin,
                selected_anchor_score=selected_anchor_score,
                candidate_count=candidate_count,
                anchor_gap=anchor_gap,
                goal_start_frame=goal_start_frame,
                candidate_ceiling=candidate_ceiling,
                retrieved_anchor=retrieved_anchor,
                forced_anchor=(int(forced_anchor)
                               if forced_anchor is not None else None),
                forced_anchor_score=forced_anchor_score,
                anchor=anchor,
                aux_pose=aux_pose[0].float().cpu().tolist(),
                goal_rel_yaw=goal_rel_yaw,
                current_goal_cos=current_goal_cos,
                frame_idx=k,
            )
        finally:
            self._restore(snap)
