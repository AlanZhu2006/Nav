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

import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import torch

try:  # package import in tests; script-local import in memnav_server.py
    from .pose_alignment import lingbot_relative_yaw
    from .reverse_memory_graph import (
        ReverseRouteProgress,
        metric_nodes_between,
        reverse_metric_nodes,
    )
    from .router_candidates import (
        causal_goal_support_indices,
        temporal_nms_candidates,
    )
except ImportError:  # pragma: no cover - exercised by the live script entrypoint
    from pose_alignment import lingbot_relative_yaw
    from reverse_memory_graph import (
        ReverseRouteProgress,
        metric_nodes_between,
        reverse_metric_nodes,
    )
    from router_candidates import (
        causal_goal_support_indices,
        temporal_nms_candidates,
    )


FLOW_TIERS = [(702, 20.0), (877, 25.0), (1075, 30.0), (1506, 40.0), (2048, 50.0)]
FLOW_GAP = 30
CERTIFIED_GUIDANCE_MODES = (
    "endpoint_bearing",
    "episodic_path_field",
    "action_coordinate_compass",
    "se2_route_compass",
    "monocular_route_tangent",
)
CERTIFIED_ROUTE_MOTION_MODELS = (
    "fundamental_then_pnp",
    "direct_pnp",
    "direct_pnp_dense_query",
)


def certified_route_edge_motion_model(route_motion_model, edge_kind):
    """Resolve the frozen estimator used by one route edge.

    ``direct_pnp_dense_query`` is deliberately stage-specific. The sparse
    historical tape keeps the already-validated Fundamental-MAGSAC -> PnP
    estimator; only consecutive live-query observations use direct PnP. This
    prevents a query-cadence attribution from silently changing how the
    historical route itself is constructed.
    """

    model = str(route_motion_model)
    kind = str(edge_kind)
    if model not in CERTIFIED_ROUTE_MOTION_MODELS:
        raise ValueError(f"unknown certified route motion model: {model}")
    if model == "direct_pnp":
        return "direct_pnp"
    if (model == "direct_pnp_dense_query"
            and kind == "live_query_adjacent_sample"):
        return "direct_pnp"
    return "fundamental_then_pnp"


def flow_threshold_for_length(n_frames):
    """Match the length-tiered sparse-keyframe policy used by precompute."""
    for upper, threshold in FLOW_TIERS:
        if n_frames <= upper:
            return threshold
    return 60.0


def effective_candidate_ceiling(
        goal_start_frame, candidate_ceiling_override=None):
    """Return a fail-closed causal retrieval ceiling.

    The default remains the frame immediately before the current goal began.
    An evaluation may tighten (never widen) that ceiling to distinguish old
    memory from observations collected on an intervening goal leg.
    """
    default = int(goal_start_frame) - 1
    if candidate_ceiling_override is None:
        return default
    override = int(candidate_ceiling_override)
    if override < 0 or override > default:
        raise ValueError(
            "candidate ceiling override must be non-negative and may not "
            f"exceed the causal default {default}, got {override}"
        )
    return override


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
class MemNavAgent:
    def __init__(self, checkpoint, internnav_root, device="cuda:0",
                 exclude_recent=83, num_samples=16, buffer_root=None,
                 gate_skip_below=0.0, retrieval_mode="raw", anchor_switch_margin=0.01,
                 flow_gate="auto", retrieval_candidate_top_k=32,
                 retrieval_candidate_min_gap=16,
                 graph_subgoal_spacing_m=0.0,
                 graph_subgoal_arrival_m=0.60,
                 phase_b_ranker=None,
                 certified_relocalization_matcher=None,
                 certified_counterfactual_audit=False,
                 certified_eager_depth_cache=False,
                 certified_route_depth_cache_stride=0,
                 certified_route_motion_model="fundamental_then_pnp",
                 certified_route_motion_unfiltered_shadow=False,
                 cdec_pairwise_ranker=None,
                 pi3x_online_relocalizer=None,
                 depth_observation_only=False):
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
        # Long-stream mechanism audits need the exact causal LingBot depth
        # state, but never call MemNav's later goal-insertion/planning path.
        # In this mode we retain the model's live KV state while omitting only
        # the duplicate, write-only planning-cache snapshots below.  Planning
        # fails closed so an incomplete cache can never reach a controller.
        self.depth_observation_only = bool(depth_observation_only)
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
        self.graph_subgoal_spacing_m = float(graph_subgoal_spacing_m)
        self.graph_subgoal_arrival_m = float(graph_subgoal_arrival_m)
        if (not np.isfinite(self.graph_subgoal_spacing_m)
                or self.graph_subgoal_spacing_m < 0.0):
            raise ValueError("graph subgoal spacing must be finite and non-negative")
        if (not np.isfinite(self.graph_subgoal_arrival_m)
                or self.graph_subgoal_arrival_m <= 0.0):
            raise ValueError("graph subgoal arrival radius must be finite and positive")
        # Experimental P0 component. It is intentionally injected by the
        # server so the base MemNav agent has no training-stack dependency.
        # The ranker can only order candidates; activation remains outside the
        # agent in the evaluator's existing SIFT/RANSAC gate.
        self.phase_b_ranker = phase_b_ranker
        # Optional frozen SuperPoint+LightGlue provider.  It is used only by
        # the explicit certified-relocalization endpoint; the historical
        # SIFT/RANSAC and MemNav paths remain byte-for-byte selectable.
        self.certified_relocalization_matcher = (
            certified_relocalization_matcher)
        # Read-only mechanism audit.  When enabled, the endpoint applies the
        # exact same PnP/certificate to hypotheses in canonical DINO order as
        # well as to the deployed geometry proposal.  The semantic-first audit
        # stops at its first accepted hypothesis.  Its result is logged but
        # never has action authority, so the formal controller is unchanged.
        self.certified_counterfactual_audit = bool(
            certified_counterfactual_audit)
        # Optional low-query-latency deployment mode.  It maintains a second,
        # exact dense LingBot stream alongside the flow-gated navigation stream
        # and materializes causal depth per history frame.  The default remains
        # lazy replay because eager mode trades ~one extra frozen-front-end pass
        # per observation and bounded CPU/GPU memory for sub-millisecond depth
        # lookup at a new goal.
        self.certified_eager_depth_cache = bool(
            certified_eager_depth_cache)
        self.certified_route_depth_cache_stride = int(
            certified_route_depth_cache_stride)
        if self.certified_route_depth_cache_stride < 0:
            raise ValueError(
                "certified route depth cache stride must be non-negative")
        self.certified_route_motion_model = str(
            certified_route_motion_model)
        if self.certified_route_motion_model not in (
                CERTIFIED_ROUTE_MOTION_MODELS):
            raise ValueError(
                "certified route motion model must be fundamental_then_pnp "
                "direct_pnp, or direct_pnp_dense_query")
        self.certified_route_motion_unfiltered_shadow = bool(
            certified_route_motion_unfiltered_shadow)
        if (self.certified_route_motion_unfiltered_shadow
                and self.certified_route_motion_model !=
                "fundamental_then_pnp"):
            raise ValueError(
                "unfiltered route-motion shadow requires the fundamental "
                "primary model")
        # Optional factorized CDEC proposal.  It has no activation authority:
        # the geometry proposal is always tried first, accepted geometry can
        # never be overridden, and the learned proposal reaches PnP only after
        # the first certificate rejects.
        self.cdec_pairwise_ranker = cdec_pairwise_ranker
        # Optional frozen learned relocalizer.  It shares only the causal DINO
        # shortlist with Certified Episodic Compass; it consumes no
        # LightGlue/PnP/LingBot-depth/certificate evidence.
        self.pi3x_online_relocalizer = pi3x_online_relocalizer
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
        # Optional frame-bound executor receipts. Entry i describes realized
        # motion from causal RGB i-1 to i. They are internal action/odometry
        # receipts, not evaluator poses or an additional exteroceptive stream.
        self.executor_motion_receipts = []
        self.scale_k = None              # [L,H,S,P,d] bf16 gpu
        self.scale_v = None
        self._metric_scale = None        # lazy ground-anchored scale
        self._goal_cache = {}            # (goal_md5, anchor) -> goal_pose; goal_md5 -> goal_cls
        self._anchor_state = {}          # goal_md5 -> dict(m, score): sticky-anchor ratchet
        self._goal_start_frame = {}      # goal_md5 -> first frame queried for this goal
        self._graph_routes = {}          # goal_md5 -> frozen reverse-memory route + cursor
        # A goal image may reappear after intervening goals in a lifelong
        # episode.  Long-term RGB/map state survives that switch, whereas every
        # goal-conditioned proposal/proof cache belongs to one contiguous goal
        # session.  Otherwise an A->B->A sequence would reuse A's original
        # candidate ceiling and could never consume observations acquired while
        # first pursuing A.
        self._active_goal_key = None
        self._goal_session_index = 0
        self._last_goal_session_started = False
        # SIFT/essential verification is a deterministic function of the goal,
        # immutable history image, and per-episode intrinsic.  Cache both
        # positive and negative results so temporal confirmation checks anchor
        # stability instead of recomputing the identical image pair.
        self._retrieval_verification_cache = {}
        # Read-only learned-ranking results are frozen per goal and exact
        # shortlist.  DINO scores can vary by a few floating-point bits across
        # otherwise identical GPU queries, so the expensive deterministic
        # geometry is cached separately by its actual immutable inputs.  The
        # cheap model rank is still recomputed from each request's exact DINO
        # scores whenever the exact-result cache misses.
        self._phase_b_rank_cache = {}
        self._phase_b_scale_cache = {}
        self._phase_b_geometry_cache = {}
        # One immutable absolute goal pose (or one immutable abstention) per
        # goal.  Accepted poses are converted to a fresh current-relative
        # PointGoal on each request, so localization is paid once rather than
        # once per navigation replan.
        self._certified_relocalization_cache = {}
        # Dense reference depth depends only on the immutable history and the
        # selected anchor, not on the goal.  Different lifelong goals often
        # retrieve the same place, so retain exact final depth/confidence while
        # keeping the first request identical to the confirmed full replay.
        self._certified_reference_depth_cache = {}
        # Sparse CPU-only local-observation cache.  It is intentionally
        # separate from the exact canonical certificate depth cache above, so
        # enabling route tracking cannot alter initial CEC authorization.
        self._certified_route_reference_depth_cache = {}
        self._certified_route_depth_cached_anchors = set()
        # The dense-query motion model retains at most one controller interval
        # of write-time depth.  These frame-bound values let the visual route
        # consume the same observation cadence as the executor without turning
        # every query frame into a permanent long-term-memory node.
        self._certified_route_live_depth_cache = {}
        self._certified_dense_replay_last_stats = None
        self._certified_dense_stream_snapshot = None
        self._certified_eager_depth_error = None
        self._certified_eager_depth_runtime_ms = []
        self._certified_eager_depth_cached_anchors = set()
        # One frozen initial Pi3X proposal decision per goal.  An initial
        # reject is sticky; an accepted anchor is fixed while its current-to-
        # goal bearing is recomputed from causal RGB at each later replan.
        self._pi3x_relocalization_cache = {}
        # Optional, default-off rescue routes are separate from the historical
        # always-on reverse graph.  A route is created only after an external
        # progress monitor declares the direct certified bearing stuck.
        self._certified_graph_routes = {}
        # Development-only long-range readout.  Each accepted goal owns one
        # continuous tail-to-anchor path and monotone progress scalar.  It is
        # separate from the dormant discrete stuck-rescue graph above and is
        # never consulted by canonical endpoint-bearing CEC.
        self._certified_path_field_routes = {}
        # Scale-free long-return challenger. CEC authorizes the historical
        # route once; subsequent route state is advanced only by frame-bound
        # executor translation/yaw receipts.
        self._certified_action_coordinate_routes = {}
        # Development successor to the scalar action coordinate.  It rebuilds
        # a local 2-D executor route from the same receipts and advances route
        # state by monotone geometric projection, never by travelled distance.
        self._certified_se2_route_compasses = {}
        # Deployable long-range route readout.  Both the historical route and
        # live state are reconstructed from causal RGB, height-scaled LingBot
        # depth, and adjacent LightGlue/PnP motion.  It never consumes the
        # optional executor/simulator receipts above.
        self._certified_monocular_route_tangents = {}
        # The candidate set is fixed at the first causal query for a goal.
        # An empty set is a real, cacheable abstention (for example after a
        # very short Novel leg), not permission to admit later goal-session
        # frames or repeatedly pay localization cost.
        self._certified_candidate_cache = {}
        # GOAT semantic arrival is intentionally independent of the Revisit
        # controller's pooled-scale fallback.  This cache holds the one strict
        # first-64-frame estimate (or its fail-closed unavailability receipt).
        self._arrival_metric_scale_result = None
        # MDTEC short-horizon readout.  This is deliberately separate from
        # ``_metric_scale``: the latter is a legacy lazy helper whose evidence
        # grows with the whole stream, whereas the monocular NavDP interface
        # must freeze exactly RGB observations 0..39 and never update again.
        self._first40_scale_receipt = None
        self._first40_scale_freeze_ms = None
        self._last_frame_jpg_sha256 = None
        # The flow gate already predicts depth for every post-warmup frame.
        # Retain only that newest immutable tensor so the MDTEC transaction
        # can serialize it without running the same depth head a second time.
        self._last_stream_depth_frame_index = None
        self._last_stream_relative_depth = None
        # tower-1 live capture: the current frame's post-GCT tokens + agg list from the
        # CONTINUOUS stream. Training used window_forward's cold-cache recompute only
        # because samples load from disk; at eval the live stream supersedes it.
        self._last_tokens = None         # [1, P, 2C] current frame post-GCT tokens
        self._last_agg = None            # list of [1,1,P,2C] (selected layers, current frame)
        self._psi = None                 # patch_start_idx from the scale block
        self.lb.model.clean_kv_cache()
        self.lb.model.camera_head.clean_kv_cache()

    @staticmethod
    def _cache_key_contains_goal(cache_key, goal_key):
        if cache_key == goal_key:
            return True
        return (
            isinstance(cache_key, tuple)
            and any(item == goal_key for item in cache_key)
        )

    def _clear_goal_conditioned_state(self, goal_key):
        """Forget one goal session without touching causal visual history.

        Anchor depth is intentionally retained because it is a property of an
        immutable history frame, not of a query.  Every cache listed below is
        query-bound through a goal hash, candidate ceiling, or sticky action
        decision and must be recomputed if the same image becomes a later goal.
        """
        goal_starts = getattr(self, "_goal_start_frame", None)
        if goal_starts is not None:
            goal_starts.pop(goal_key, None)
        cache_names = (
            "_goal_cache", "_anchor_state", "_graph_routes",
            "_retrieval_verification_cache", "_phase_b_rank_cache",
            "_phase_b_scale_cache", "_phase_b_geometry_cache",
            "_certified_relocalization_cache", "_pi3x_relocalization_cache",
            "_certified_graph_routes", "_certified_path_field_routes",
            "_certified_action_coordinate_routes",
            "_certified_se2_route_compasses",
            "_certified_monocular_route_tangents",
            "_certified_candidate_cache",
        )
        for name in cache_names:
            mapping = getattr(self, name, None)
            if mapping is None:
                continue
            stale = [
                key for key in mapping
                if self._cache_key_contains_goal(key, goal_key)
            ]
            for key in stale:
                del mapping[key]
        # Dense query depths are transient evidence for the active goal.  The
        # fixed-stride causal tape remains in the persistent route cache.
        self._certified_route_live_depth_cache.clear()

    def _begin_goal_session(self, goal_key):
        """Open a contiguous goal session while preserving lifelong memory."""
        goal_key = str(goal_key)
        active_goal_key = getattr(self, "_active_goal_key", None)
        switched = goal_key != active_goal_key
        if switched:
            if active_goal_key is not None:
                self._clear_goal_conditioned_state(active_goal_key)
            # Defensive cleanup makes a repeated A->B->A query independent of
            # any legacy A cache that predates this lifecycle contract.
            self._clear_goal_conditioned_state(goal_key)
            self._active_goal_key = goal_key
            self._goal_session_index = int(
                getattr(self, "_goal_session_index", 0)) + 1
        self._last_goal_session_started = bool(switched)
        return switched

    def goal_session_status(self):
        return {
            "goal_session_index": int(getattr(
                self, "_goal_session_index", 0)),
            "goal_session_started": bool(getattr(
                self, "_last_goal_session_started", False)),
            "long_term_memory_preserved": True,
        }

    def replay_goal_session(self, goal_jpg_bytes, expected_start_frame):
        """Restore a frozen query boundary without appending or planning.

        Paired lifelong evaluations replay an already executed C trajectory
        before branching at B2.  Replaying RGB frames alone is insufficient:
        the original C query also opened a goal session at the first C frame.
        This method restores only that lifecycle boundary.  It deliberately
        performs no retrieval, certificate inference, or controller action.
        """
        import hashlib

        expected_start_frame = int(expected_start_frame)
        if expected_start_frame != int(self.n):
            raise ValueError(
                "replayed goal session does not start at the current frame")
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        switched = self._begin_goal_session(goal_key)
        if not switched:
            raise ValueError("replayed goal session did not switch goals")
        goal_start_frame = self._goal_start_frame.setdefault(
            goal_key, expected_start_frame)
        if int(goal_start_frame) != expected_start_frame:
            raise ValueError("replayed goal-session boundary changed")
        return {
            **self.goal_session_status(),
            "goal_start_frame": int(goal_start_frame),
            "candidate_ceiling": int(goal_start_frame) - 1,
            "frame_count": int(self.n),
            "diffusion_sampled": False,
            "memory_appended": False,
        }

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

    @torch.no_grad()
    def goal_candidate_support(
            self, goal_jpg_bytes, stride=8, candidate_frame_idx=None,
            min_frame_gap=None):
        """Score one prospective goal against cached causal history.

        This is the online counterpart of the real-world candidate scoring
        script.  It encodes the goal once, compares it with the already-cached
        per-frame DINO CLS vectors, and geometrically verifies the best sampled
        anchor.  It does not append a frame, begin a goal session, alter the
        candidate ceiling, or advance LingBot state.
        """
        import hashlib
        import time

        started = time.perf_counter()
        stride = max(1, int(stride))
        if candidate_frame_idx is None:
            candidate_frame_idx = self.n
        candidate_frame_idx = int(candidate_frame_idx)
        if min_frame_gap is None:
            min_frame_gap = self.retrieval_candidate_min_gap
        min_frame_gap = max(1, int(min_frame_gap))
        dense_cls = getattr(self, "dino_cls", ())
        if self.n < self.S or len(dense_cls) != self.n:
            return {
                "ok": False,
                "reason": "history_dino_not_ready",
                "frames_total": int(self.n),
                "dino_frames_ready": int(len(dense_cls)),
                "state_mutated": False,
            }

        try:
            indices = causal_goal_support_indices(
                self.n,
                candidate_frame_idx=candidate_frame_idx,
                stride=stride,
                min_frame_gap=min_frame_gap,
            )
        except ValueError as error:
            return {
                "ok": False,
                "reason": "invalid_causal_support_window",
                "detail": str(error),
                "frames_total": int(self.n),
                "state_mutated": False,
            }
        if not indices:
            return {
                "ok": False,
                "reason": "insufficient_temporal_support",
                "frames_total": int(self.n),
                "candidate_frame_idx": candidate_frame_idx,
                "min_frame_gap": min_frame_gap,
                "state_mutated": False,
            }
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        goal_cls = self._goal_cache.get(("cls", goal_key))
        if goal_cls is None:
            goal_path = os.path.join(
                self.rgb_dir, "_candidate_support_goal_{}.jpg".format(goal_key))
            with open(goal_path, "wb") as handle:
                handle.write(goal_jpg_bytes)
            goal_image = self.lb.load_images([goal_path])[0][None].to(
                self.device)
            goal_cls = self.lb.dino(goal_image)["cls"]
            self._goal_cache[("cls", goal_key)] = goal_cls

        memory_cls = torch.stack(
            [dense_cls[index] for index in indices], 0
        ).to(self.device)
        cosine = torch.nn.functional.cosine_similarity(
            goal_cls.expand(memory_cls.shape[0], -1), memory_cls, dim=-1)
        best_offset = int(torch.argmax(cosine).item())
        best_anchor = int(indices[best_offset])
        best_cos = float(cosine[best_offset].item())
        overlap = self.verify_retrieval_overlap(goal_jpg_bytes, best_anchor)
        return {
            "ok": True,
            "max_cos": best_cos,
            "argmax_idx": best_anchor,
            "frames_swept": len(indices),
            "frames_total": int(self.n),
            "stride": stride,
            "candidate_frame_idx": candidate_frame_idx,
            "min_frame_gap": min_frame_gap,
            "eligible_anchor_ceiling": int(indices[-1]),
            "geometry": overlap,
            "geometry_backend": "sift_fundamental_ransac",
            "state_mutated": False,
            "goal_session_index": int(getattr(
                self, "_goal_session_index", 0)),
            "scoring_ms": 1000.0 * (time.perf_counter() - started),
        }

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

    def phase_b_status(self):
        if self.phase_b_ranker is None:
            return {
                "enabled": False,
                "activation_semantics": "geometry_gate_unchanged",
            }
        return self.phase_b_ranker.status()

    @torch.no_grad()
    def rank_retrieval_candidates(self, goal_jpg_bytes, candidates):
        """Rank one already-frozen DINO shortlist without mutating memory.

        Candidate activation is deliberately absent from this method. The
        caller must run the same geometric verifier and confirmation latch as
        the DINO-order baseline. Candidate-validity and no-match outputs are
        returned only as diagnostics.
        """
        import copy
        import hashlib
        import time

        started = time.perf_counter()

        def failure(message):
            return {
                "ok": False,
                "error": str(message),
                "cached": False,
                "ranking_ms": 1000.0 * (time.perf_counter() - started),
                "activation_uses_model_score": False,
            }

        if self.phase_b_ranker is None:
            return failure("Phase-B ranker is not enabled on this server")
        if not isinstance(candidates, list) or not candidates:
            return failure("candidates must be a non-empty JSON list")

        parsed = []
        seen = set()
        for input_rank, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                return failure(f"candidate {input_rank} is not an object")
            try:
                anchor = int(item["anchor"])
                score = float(item["score"])
            except (KeyError, TypeError, ValueError) as error:
                return failure(f"candidate {input_rank} is malformed: {error}")
            if not np.isfinite(score):
                return failure(f"candidate {input_rank} score is non-finite")
            if anchor in seen:
                return failure(f"candidate anchor is duplicated: {anchor}")
            seen.add(anchor)
            parsed.append({
                "anchor": anchor,
                "score": score,
                "dino_rank": input_rank,
            })

        try:
            from MemNavData.phase_b_runtime import RUNTIME_CONFIG
        except ModuleNotFoundError as error:
            return failure(f"Phase-B runtime import failed: {error}")
        if len(parsed) > RUNTIME_CONFIG.candidate_top_k:
            return failure(
                f"candidate shortlist exceeds top-{RUNTIME_CONFIG.candidate_top_k}")
        anchors = [item["anchor"] for item in parsed]
        if any(
            abs(left - right) < RUNTIME_CONFIG.candidate_min_gap
            for index, left in enumerate(anchors)
            for right in anchors[index + 1:]
        ):
            return failure(
                "candidate shortlist violates the audited temporal gap"
            )

        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        goal_start_frame = self._goal_start_frame.get(goal_key)
        if goal_start_frame is None:
            return failure(
                "goal must be registered by retrieval_probe_step before ranking"
            )
        candidate_ceiling = int(goal_start_frame) - 1
        for item in parsed:
            anchor = item["anchor"]
            if not self.amargin <= anchor <= candidate_ceiling:
                return failure(
                    f"candidate {anchor} is outside causal retrieval range "
                    f"[{self.amargin}, {candidate_ceiling}]"
                )
            if anchor >= self.n:
                return failure(f"candidate {anchor} is outside live history")
            if not os.path.isfile(os.path.join(self.rgb_dir, f"{anchor}.jpg")):
                return failure(f"candidate image is missing: {anchor}.jpg")

        signature = tuple(
            (item["anchor"], float(item["score"])) for item in parsed
        )
        cache_key = (goal_key, int(goal_start_frame), signature)
        cached = self._phase_b_rank_cache.get(cache_key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result["cached"] = True
            result["ranking_ms"] = 1000.0 * (time.perf_counter() - started)
            return result

        snapshot = self._snapshot()
        try:
            from pathlib import Path
            from MemNavData.phase_b_runtime import (
                append_goal_geometry,
                external_causal_metric_scale,
                measurement_feature_row,
            )

            live_cache = self._live_cache()
            scale_key = (goal_key, int(goal_start_frame))
            cached_scale = self._phase_b_scale_cache.get(scale_key)
            scale_cached = cached_scale is not None
            if cached_scale is None:
                metric_scale, scale_quality = external_causal_metric_scale(
                    self.lb,
                    Path(self.rgb_dir),
                    live_cache["cam_pose_enc"],
                    self.camera_height,
                    int(goal_start_frame),
                )
                self._phase_b_scale_cache[scale_key] = (
                    float(metric_scale), copy.deepcopy(scale_quality))
            else:
                metric_scale, scale_quality = copy.deepcopy(cached_scale)

            geometry_keys = [
                (goal_key, int(goal_start_frame), item["anchor"])
                for item in parsed
            ]
            missing_geometry = [
                key for key in geometry_keys
                if key not in self._phase_b_geometry_cache
            ]
            goal_image = None
            if missing_geometry:
                goal_path = os.path.join(
                    self.rgb_dir, f"_phase_b_goal_{goal_key}.jpg")
                with open(goal_path, "wb") as handle:
                    handle.write(goal_jpg_bytes)
                goal_image = self.lb.load_images([goal_path])[0].to(self.device)

            feature_rows = []
            measurements = []
            geometry_cache_hits = 0
            geometry_cache_misses = 0
            for item, geometry_key in zip(parsed, geometry_keys):
                cached_measurement = self._phase_b_geometry_cache.get(
                    geometry_key)
                if cached_measurement is None:
                    measurement = append_goal_geometry(
                        self.lb,
                        live_cache,
                        Path(self.rgb_dir),
                        goal_image,
                        item["anchor"],
                    )
                    self._phase_b_geometry_cache[geometry_key] = (
                        copy.deepcopy(measurement))
                    geometry_cache_misses += 1
                else:
                    measurement = copy.deepcopy(cached_measurement)
                    geometry_cache_hits += 1
                row = measurement_feature_row(
                    measurement,
                    dino_cosine=item["score"],
                    metric_scale=metric_scale,
                    scale_quality=scale_quality,
                )
                feature_rows.append(row)
                predicted_xy = np.asarray(
                    row["predicted_relative_xy_m"], dtype=np.float64)
                measurements.append({
                    "anchor": item["anchor"],
                    "dino_rank": item["dino_rank"],
                    "dino_cosine": item["score"],
                    "predicted_relative_xy_m": predicted_xy.tolist(),
                    "cloud_overlap_f1": float(
                        measurement["cloud_overlap_f1"]),
                    "anchor_goal_distance_norm": float(
                        row["anchor_goal_distance_norm_center"]),
                    "goal_refine_translation_norm": float(
                        row["goal_refine_translation_norm_median"]),
                    "goal_refine_rotation_deg": float(
                        row["goal_refine_rotation_deg_median"]),
                })

            ranked = self.phase_b_ranker.rank(feature_rows)
            order = ranked["order"]
            if sorted(order) != list(range(len(parsed))):
                raise RuntimeError("Phase-B ranker returned a non-permutation")
            ranked_candidates = []
            for learned_rank, index in enumerate(order, start=1):
                item = dict(parsed[index])
                item.update({
                    "learned_rank": learned_rank,
                    "rank_probability": float(
                        ranked["rank_probability"][index]),
                    "candidate_validity_diagnostic": float(
                        ranked["candidate_validity"][index]),
                })
                ranked_candidates.append(item)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            result = {
                "ok": True,
                "error": None,
                "cached": False,
                "ranking_ms": elapsed_ms,
                "uncached_ranking_ms": elapsed_ms,
                "ranked_candidates": ranked_candidates,
                "candidate_measurements": measurements,
                "metric_scale_m_per_raw": float(metric_scale),
                "metric_scale_source": (
                    "external_causal_first_prefix_v1"),
                **{key: float(value)
                   for key, value in scale_quality.items()},
                "no_match_probability_diagnostic": float(
                    ranked["no_match_probability_diagnostic"]),
                "activation_uses_model_score": False,
                "checkpoint_sha256": (
                    self.phase_b_ranker.checkpoint_sha256),
                "goal_start_frame": int(goal_start_frame),
                "candidate_ceiling": candidate_ceiling,
                "scale_cached": bool(scale_cached),
                "geometry_cache_hit_count": int(geometry_cache_hits),
                "geometry_cache_miss_count": int(geometry_cache_misses),
            }
            self._phase_b_rank_cache[cache_key] = copy.deepcopy(result)
            return result
        except Exception as error:
            return failure(
                f"Phase-B feature/ranking failure "
                f"({type(error).__name__}): {error}")
        finally:
            self._restore(snapshot)

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

    def add_frame(
            self, jpg_bytes, *, executed_translation_m=None,
            executed_yaw_rad=None, executed_forward_m=None,
            executed_left_m=None, executor_local_se2_source=None):
        """Ingest one RGB frame (jpg bytes). Returns the frame index."""
        idx = self.n
        # A failed or partial append must never expose the preceding frame's
        # depth under the new RGB SHA/frame binding.
        self._last_stream_depth_frame_index = None
        self._last_stream_relative_depth = None
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
        agg_mod_cap = getattr(self.lb, "agg", None)
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
        self.executor_motion_receipts.append(executor_receipt)
        self._last_frame_jpg_sha256 = hashlib.sha256(jpg_bytes).hexdigest()
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
            if not getattr(self, "depth_observation_only", False):
                kv = self.lb.agg.kv_cache
                self.scale_k = torch.stack([
                    kv[f"k_{i}"][0, :, :self.S].to(torch.bfloat16)
                    for i in range(self.L_depth)
                ]).contiguous()
                self.scale_v = torch.stack([
                    kv[f"v_{i}"][0, :, :self.S].to(torch.bfloat16)
                    for i in range(self.L_depth)
                ]).contiguous()
            pose = pl[-1][0].float()                    # [S,9]
            self.cam_pose.extend([pose[i].cpu() for i in range(self.S)])
            if not getattr(self, "depth_observation_only", False):
                ck, cv = self._read_cam_newest(self.S)
                self.cam_k.extend(ck); self.cam_v.extend(cv)
            self._last_kf_pose = pl[-1][:, -1:].float()
            self._last_kf_idx = self.S - 1
            self.cam_frame_indices = list(range(self.S))
            self._pending = []
            if self.certified_eager_depth_cache:
                # Scale inference is common to sparse navigation and dense
                # certificate streams.  Hold its immutable reference snapshot
                # as the exact starting point for the first post-scale frame.
                self._certified_dense_stream_snapshot = self._snapshot()
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
            prediction = None
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
                        prediction = self.lb.model._predict_depth(
                            agg, img[None, None].to(self.device), self._psi)
                        depth = prediction["depth"].float()
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
                if not getattr(self, "depth_observation_only", False):
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
            dense_query_due = (
                self.certified_route_motion_model ==
                "direct_pnp_dense_query"
                and bool(self._certified_monocular_route_tangents)
            )
            route_cache_due = (
                self.certified_route_depth_cache_stride > 0
                and idx % self.certified_route_depth_cache_stride == 0
            )
            if route_cache_due or dense_query_due:
                if prediction is None:
                    with torch.no_grad(), torch.autocast(
                            "cuda", dtype=torch.bfloat16):
                        prediction = self.lb.model._predict_depth(
                            agg, img[None, None].to(self.device), self._psi)
                route_depth = prediction[
                    "depth"][0, -1, ..., 0].float().cpu().numpy()
                route_confidence = prediction[
                    "depth_conf"][0, -1].float().cpu().numpy()
                if route_cache_due:
                    self._certified_route_reference_depth_cache[idx] = (
                        route_depth.copy(), route_confidence.copy())
                    self._certified_route_depth_cached_anchors.add(idx)
                if dense_query_due and not route_cache_due:
                    self._certified_route_live_depth_cache[idx] = (
                        route_depth.copy(), route_confidence.copy())
                    # Formal execution replans every eight actions.  Keep two
                    # intervals defensively; older dense values have already
                    # been consumed and are not long-term memory.
                    oldest = idx - 16
                    for stale in tuple(
                            self._certified_route_live_depth_cache):
                        if stale < oldest:
                            del self._certified_route_live_depth_cache[stale]
            if prediction is not None:
                self._last_stream_relative_depth = prediction[
                    "depth"][0, -1, ..., 0].float().detach()
                self._last_stream_depth_frame_index = int(idx)
            if (self.certified_eager_depth_cache
                    and self._certified_dense_stream_snapshot is not None):
                self._update_certified_eager_depth(idx, img)
        self.n += 1
        if self.n == 40:
            self._freeze_first40_scale()
        return idx

    @torch.no_grad()
    def _freeze_first40_scale(self):
        """Freeze the sole causal RGB-only scale receipt for this episode.

        LingBot's scale routine replays the prefix and clears its KV caches, so
        the live map stream is restored exactly afterwards.  Any failure is a
        frozen invalid receipt; it can only yield zero depth, never a pooled or
        oracle fallback.
        """

        if self._first40_scale_receipt is not None:
            raise RuntimeError("first-40 metric scale was already frozen")
        if self.n != 40 or len(self.cam_pose) != 40:
            raise RuntimeError(
                "first-40 scale freeze requires exactly 40 live observations"
            )
        from MemNavData.monocular_depth_runtime import (
            compute_first40_scale_receipt,
            failed_first40_scale_receipt,
        )

        snapshot = self._snapshot()
        saved_dino_output = self._dino_out[0]
        started = time.perf_counter()
        try:
            self._first40_scale_receipt = compute_first40_scale_receipt(
                self.lb,
                self.rgb_dir,
                torch.stack(self.cam_pose, 0).float().cpu().numpy(),
                self.camera_height,
            )
        except Exception as error:
            self._first40_scale_receipt = failed_first40_scale_receipt(
                self.camera_height,
                f"{type(error).__name__}: {error}",
            )
        finally:
            self._restore(snapshot, empty_cuda_cache=False)
            self._dino_out[0] = saved_dino_output
            self._first40_scale_freeze_ms = (
                1000.0 * (time.perf_counter() - started)
            )

    @torch.no_grad()
    def monocular_depth_observation(self):
        """Return current raw LingBot depth under the frozen MDTEC contract."""

        started = time.perf_counter()
        if self.n < 1 or self._last_frame_jpg_sha256 is None:
            raise RuntimeError("monocular depth requires one streamed RGB frame")
        from PIL import Image
        from MemNavData.monocular_depth_runtime import (
            ACTIVE_FROM_FRAME_INDEX,
            build_monocular_depth_payload,
        )

        frame_index = self.n - 1
        current_path = os.path.join(self.rgb_dir, f"{frame_index}.jpg")
        with Image.open(current_path) as current_image:
            source_width, source_height = current_image.size

        relative_depth = None
        depth_shape = (source_height, source_width)
        cache_hit = False
        prediction_runtime_ms = 0.0
        if frame_index >= ACTIVE_FROM_FRAME_INDEX:
            if self._first40_scale_receipt is None:
                raise RuntimeError(
                    "first-40 scale receipt missing after activation frame"
                )
            if self._first40_scale_receipt["scale_valid"] is True:
                if self._last_agg is None or self._psi is None:
                    raise RuntimeError("LingBot current-frame depth state is absent")
                cache_hit = (
                    self._last_stream_depth_frame_index == frame_index
                    and self._last_stream_relative_depth is not None
                )
                if cache_hit:
                    relative_depth = (
                        self._last_stream_relative_depth.cpu().numpy()
                    )
                else:
                    prediction_started = time.perf_counter()
                    prediction = self.lb.model._predict_depth(
                        self._last_agg,
                        self._window_imgs[-1][None, None].to(self.device),
                        self._psi,
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
            image_sha256_value=self._last_frame_jpg_sha256,
            frame_index=frame_index,
            scale_receipt=self._first40_scale_receipt,
        )
        payload["first40_scale_freeze_ms"] = self._first40_scale_freeze_ms
        payload["stream_observation_count"] = int(self.n)
        payload["depth_prediction_cache_hit"] = bool(cache_hit)
        payload["depth_prediction_runtime_ms"] = float(
            prediction_runtime_ms
        )
        payload["depth_materialization_runtime_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        return payload

    def monocular_depth_status(self):
        """Small JSON status without materializing the current depth map."""

        from MemNavData.monocular_depth_runtime import (
            ACTIVE_FROM_FRAME_INDEX,
            DEPTH_CONTRACT,
            SCALE_CONTRACT,
        )

        return {
            "enabled": True,
            "depth_contract": DEPTH_CONTRACT,
            "scale_evidence_contract": SCALE_CONTRACT,
            "active_from_frame_index": ACTIVE_FROM_FRAME_INDEX,
            "stream_observation_count": int(self.n),
            "latest_frame_index": None if self.n == 0 else int(self.n - 1),
            "first40_scale_frozen": self._first40_scale_receipt is not None,
            "first40_scale_valid": (
                None
                if self._first40_scale_receipt is None
                else bool(self._first40_scale_receipt["scale_valid"])
            ),
            "first40_scale_freeze_ms": self._first40_scale_freeze_ms,
            # This is the same immutable causal receipt already consumed by
            # the dense-depth sidecar.  Exposing its validated compact form
            # lets an experimental long-range adapter reuse one gauge instead
            # of estimating a second scale or reading simulator depth.
            "longrange_metric_scale": (
                self._first40_local_pose_metric_scale()),
            "metric_depth_sensor_consumed": False,
        }

    def causal_pose_trace_receipt(self):
        """Expose the model-internal causal pose stream for mechanism audits.

        The values are LingBot predictions already held by the runtime; no
        simulator state, wheel odometry, goal role, or future observation is
        consumed.  Navigation never calls this read-only endpoint.
        """

        poses = [
            pose.float().cpu().numpy().astype(np.float64).tolist()
            for pose in self.cam_pose
        ]
        return {
            "schema_version": "lingbot_causal_pose_trace_v1_20260902",
            "pose9": poses,
            "pose_count": len(poses),
            "stream_observation_count": int(self.n),
            "runtime_evaluator_pose_visible": False,
            "metric_depth_sensor_consumed": False,
            "longrange_metric_scale": (
                self._first40_local_pose_metric_scale()),
        }

    def causal_metric_scale_receipt(self):
        """Expose the two causal scale receipts without advancing the stream.

        This read-only mechanism endpoint is deliberately separate from the
        navigation path.  It lets a scale audit compare the immutable MDTEC
        first-40 receipt with the independently specified first-64
        ground-plane receipt while preserving the live LingBot KV state.
        Neither value is selected here and neither authorizes control.
        """

        return {
            "schema_version": "lingbot_causal_metric_scale_v1_20260902",
            "stream_observation_count": int(self.n),
            "first40": self._first40_local_pose_metric_scale(),
            "first64": self._strict_arrival_metric_scale_preserving_stream(),
            "runtime_evaluator_pose_visible": False,
            "metric_depth_sensor_consumed": False,
        }

    @torch.no_grad()
    def causal_visual_similarity_receipt(self, history_count):
        """Return query-to-history DINO cosine for a sequence audit."""

        boundary = int(history_count)
        if not 1 <= boundary < len(self.dino_cls):
            raise ValueError("history_count must split the causal DINO stream")
        descriptors = torch.stack(self.dino_cls, dim=0).float().to(self.device)
        history = torch.nn.functional.normalize(
            descriptors[:boundary], dim=-1)
        query = torch.nn.functional.normalize(
            descriptors[boundary:], dim=-1)
        similarity = query @ history.T
        return {
            "schema_version": "causal_dino_similarity_v1_20260902",
            "history_count": boundary,
            "query_count": int(query.shape[0]),
            "descriptor_dimension": int(query.shape[1]),
            "similarity": similarity.cpu().tolist(),
            "runtime_evaluator_pose_visible": False,
            "metric_depth_sensor_consumed": False,
        }

    @torch.no_grad()
    def _update_certified_eager_depth(self, raw_index, image):
        """Advance the optional exact dense stream and cache one frame depth.

        The primary flow-gated navigation stream is snapshotted and restored
        without cloning.  Any eager-path failure disables further eager work;
        already materialized arrays remain valid and a missing anchor falls
        back to the unchanged lazy full replay.
        """
        import time

        raw_index = int(raw_index)
        live_snapshot = self._snapshot()
        saved_dino_output = self._dino_out[0]
        started = time.perf_counter()
        next_dense_snapshot = None
        depth = confidence = None
        try:
            self._restore(
                self._certified_dense_stream_snapshot,
                empty_cuda_cache=False)
            frame = image[None, None].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                agg, psi = self.lb.model._aggregate_features(
                    frame,
                    num_frame_for_scale=self.S,
                    num_frame_per_block=1)
                prediction = self.lb.model._predict_depth(
                    agg, frame, psi)
            depth = prediction[
                "depth"][0, -1, ..., 0].float().cpu().numpy()
            confidence = prediction[
                "depth_conf"][0, -1].float().cpu().numpy()
            next_dense_snapshot = self._snapshot()
        except Exception as exception:
            self._certified_eager_depth_error = (
                f"{type(exception).__name__}: {exception}")
            self._certified_dense_stream_snapshot = None
        finally:
            self._restore(live_snapshot, empty_cuda_cache=False)
            self._dino_out[0] = saved_dino_output
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        self._certified_eager_depth_runtime_ms.append(elapsed_ms)
        if next_dense_snapshot is not None:
            self._certified_dense_stream_snapshot = next_dense_snapshot
            self._certified_reference_depth_cache[raw_index] = (
                depth, confidence)
            self._certified_eager_depth_cached_anchors.add(raw_index)

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

    def _restore(self, snap, *, empty_cuda_cache=True):
        agg = self.lb.agg
        ch = self.lb.model.camera_head
        self.lb.model.clean_kv_cache()
        agg.kv_cache.update(snap["kv"])
        agg.total_frames_processed = snap["total"]
        ch.kv_cache = snap["cam"]
        ch.frame_idx = snap["cam_idx"]
        if empty_cuda_cache:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def _live_cache(self):
        """The in-memory equivalent of MemNavNet._load_cache's dict."""
        if getattr(self, "depth_observation_only", False):
            raise RuntimeError(
                "depth-observation-only mode does not materialize planning "
                "caches")
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

    def _get_metric_scale_preserving_stream(self):
        """Lazily estimate metric scale without consuming the live KV stream.

        LingBot's metric-scale routine calls ``clean_kv_cache`` internally.
        That is safe during one-shot/offline planning, but a late graph-rescue
        request can otherwise erase the camera-head cache used by the next
        online ``add_frame`` call.  The existing reference snapshot contract
        is sufficient because the scale routine replaces/clears cache entries
        rather than mutating the retained tensors in place.
        """
        if self._metric_scale is not None or self.n < self.S:
            return self._get_metric_scale()
        snapshot = self._snapshot()
        try:
            return self._get_metric_scale()
        finally:
            self._restore(snapshot)

    def certified_relocalization_status(self):
        """Advertise the exact optional runtime contract."""
        from MemNavData.certified_relocalization_runtime import (
            runtime_contract,
        )

        learned = getattr(self, "cdec_pairwise_ranker", None)
        return {
            "enabled": self.certified_relocalization_matcher is not None,
            "runtime_contract": runtime_contract(),
            "counterfactual_dino_top1_audit": bool(getattr(
                self, "certified_counterfactual_audit", False)),
            "eager_depth_cache": bool(getattr(
                self, "certified_eager_depth_cache", False)),
            "eager_depth_cache_error": getattr(
                self, "_certified_eager_depth_error", None),
            "eager_depth_cached_frames": len(getattr(
                self, "_certified_eager_depth_cached_anchors", ())),
            "route_depth_cache_stride": int(getattr(
                self, "certified_route_depth_cache_stride", 0)),
            "route_depth_cached_frames": len(getattr(
                self, "_certified_route_depth_cached_anchors", ())),
            "route_live_depth_cached_frames": len(getattr(
                self, "_certified_route_live_depth_cache", {})),
            "route_motion_model": getattr(
                self, "certified_route_motion_model",
                "fundamental_then_pnp"),
            "route_motion_unfiltered_shadow": bool(getattr(
                self, "certified_route_motion_unfiltered_shadow", False)),
            "route_localization_horizon": (
                "all_remaining_authorized_route"),
            "route_control_horizon_m": 2.5,
            "route_tangent_baseline_m": 0.30,
            "supported_guidance_modes": list(CERTIFIED_GUIDANCE_MODES),
            "action_coordinate_receipt_contract": (
                "frame_bound_realized_executor_motion_v1"),
            "se2_route_receipt_contract": "frame_bound_local_se2_v1",
            "monocular_route_tangent_contract": (
                "causal_rgb_height_scaled_adjacent_pnp_v1"),
            "learned_rescue_proposal": (
                learned.status() if learned is not None else {"enabled": False}),
        }

    def learned_pi3x_relocalization_status(self):
        """Advertise the optional frozen learned relocalizer contract."""
        runtime = getattr(self, "pi3x_online_relocalizer", None)
        if runtime is None:
            return {"enabled": False}
        return runtime.status()

    def _certified_anchor_image_record(self, anchor):
        """Read one immutable causal RGB anchor and bind it to a digest."""
        if isinstance(anchor, bool) or not isinstance(anchor, (int, np.integer)):
            raise ValueError("certified anchor must be an integer")
        anchor = int(anchor)
        if anchor < int(self.S) or anchor >= int(self.n):
            raise ValueError(
                f"certified anchor {anchor} outside [{self.S}, {self.n - 1}]")
        path = os.path.join(self.rgb_dir, f"{anchor}.jpg")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as stream:
            image = stream.read()
        if not image:
            raise ValueError("certified anchor image is empty")
        return {
            "anchor": anchor,
            "image": image,
            "sha256": hashlib.sha256(image).hexdigest(),
        }

    def certified_anchor_image(
            self, goal_jpg_bytes, selected_anchor, *, expected_sha256):
        """Return only the history JPEG authorized by a cached CEC proof.

        Goal bytes select the immutable certificate cache entry. A rejected,
        absent, or mismatched proof cannot become a generic memory-image read.
        """
        if (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or expected_sha256 != expected_sha256.lower()):
            raise ValueError("expected anchor SHA-256 is invalid")
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise ValueError("expected anchor SHA-256 is invalid") from exc
        if isinstance(selected_anchor, bool) or not isinstance(
                selected_anchor, (int, np.integer)):
            raise ValueError("selected anchor must be an integer")
        selected_anchor = int(selected_anchor)
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        cached = self._certified_relocalization_cache.get(goal_key)
        if not isinstance(cached, dict):
            raise ValueError("no cached CEC proof for this goal")
        result = cached.get("result")
        if not isinstance(result, dict) or result.get("accepted") is not True:
            raise ValueError("CEC proof did not authorize a history anchor")
        if result.get("selected_anchor") != selected_anchor:
            raise ValueError("requested anchor differs from the certified anchor")
        goal_start = self._goal_start_frame.get(goal_key)
        if goal_start is None or not int(self.S) <= selected_anchor < int(goal_start):
            raise ValueError("certified anchor violates the causal goal boundary")
        record = self._certified_anchor_image_record(selected_anchor)
        if record["sha256"] != expected_sha256:
            raise ValueError("certified anchor image digest changed")
        if result.get("selected_anchor_image_sha256") != expected_sha256:
            raise ValueError("anchor digest is not bound to the cached CEC proof")
        return record

    def _has_frozen_visual_relocalizer(self):
        """Whether any endpoint consumes the shared frozen DINO top-8."""
        return (
            getattr(self, "certified_relocalization_matcher", None) is not None
            or getattr(self, "pi3x_online_relocalizer", None) is not None
        )

    def certified_arrival_status(self):
        """Advertise the frozen GOAT semantic-arrival boundary."""

        from MemNavData.goat_certified_arrival_contract import (
            contract_receipt,
        )

        return {
            "enabled": self.certified_relocalization_matcher is not None,
            "contract": contract_receipt(),
            "scale_policy": "strict_first_64_frames_no_pooled_fallback",
            "simulator_depth_consumed": False,
        }

    def _strict_arrival_metric_scale_preserving_stream(self):
        """Return the frozen first-64-frame scale or fail closed.

        ``_get_metric_scale`` belongs to the navigation controller and may use
        a pooled constant when floor recovery fails.  Semantic STOP cannot use
        that fallback: the train audit required a valid causal scale record for
        every accepted prediction.  This method mirrors that path exactly and
        restores the online LingBot caches after the destructive scale pass.
        """

        from MemNavData.goat_certified_arrival_contract import (
            CAUSAL_SCALE_CONFIG,
            MINIMUM_CAUSAL_STREAM_FRAMES,
        )

        if self._arrival_metric_scale_result is not None:
            return dict(self._arrival_metric_scale_result)
        if self.n < MINIMUM_CAUSAL_STREAM_FRAMES:
            return {
                "available": False,
                "reason": "causal_scale_prefix_incomplete",
                "frame_count": int(self.n),
            }
        snapshot = self._snapshot()
        try:
            poses = torch.stack(self.cam_pose, 0)
            paths = [
                os.path.join(self.rgb_dir, f"{index}.jpg")
                for index in range(MINIMUM_CAUSAL_STREAM_FRAMES)
            ]
            if not all(os.path.isfile(path) for path in paths):
                raise RuntimeError("strict causal scale RGB prefix is incomplete")
            scale, debug = self.lb.compute_metric_scale(
                paths,
                poses[:MINIMUM_CAUSAL_STREAM_FRAMES],
                camera_height_m=float(self.camera_height),
                conf_quantile=float(
                    CAUSAL_SCALE_CONFIG["confidence_quantile"]),
                pixel_stride=int(CAUSAL_SCALE_CONFIG["pixel_stride"]),
                nbins=int(CAUSAL_SCALE_CONFIG["histogram_bins"]),
                n_frames=MINIMUM_CAUSAL_STREAM_FRAMES,
                peak_thresh=float(CAUSAL_SCALE_CONFIG["peak_threshold"]),
                bias_correction=float(CAUSAL_SCALE_CONFIG["bias_correction"]),
                scale_range=(
                    float(CAUSAL_SCALE_CONFIG["minimum_scale"]),
                    float(CAUSAL_SCALE_CONFIG["maximum_scale"]),
                ),
                return_debug=True,
            )
            if scale is None or not isinstance(debug, dict):
                raise RuntimeError("strict causal metric scale is unavailable")
            scale = float(scale)
            ground_h = float(debug.get("h_est"))
            h_iqr = float(debug.get("h_iqr"))
            valid_frames = int(debug.get("n_valid"))
            total_frames = int(debug.get("n_frames"))
            if (not np.isfinite(scale) or scale <= 0.0
                    or not np.isfinite(ground_h) or ground_h <= 0.0
                    or not np.isfinite(h_iqr) or h_iqr < 0.0
                    or total_frames != MINIMUM_CAUSAL_STREAM_FRAMES
                    or not max(3, total_frames // 8)
                    <= valid_frames <= total_frames):
                raise RuntimeError("strict causal scale quality contract failed")
            unclamped = (
                float(CAUSAL_SCALE_CONFIG["bias_correction"])
                * float(self.camera_height) / ground_h
            )
            quality = {
                "external_scale_valid_frame_ratio": (
                    valid_frames / total_frames),
                "external_scale_relative_h_iqr": h_iqr / ground_h,
                "external_scale_clamped": float(not np.isclose(
                    scale, unclamped, rtol=1e-6, atol=1e-6)),
            }
            result = {
                "available": True,
                "reason": "strict_causal_scale_available",
                "frame_count": MINIMUM_CAUSAL_STREAM_FRAMES,
                "metric_scale_m_per_raw": float(scale),
                "quality": {
                    str(key): float(value)
                    for key, value in quality.items()
                },
            }
        except Exception as error:
            # Floor visibility is a deployment condition, not permission to
            # substitute a pooled value.  Cache this deterministic failure so
            # repeated zero proposals do not repeatedly destroy/replay state.
            result = {
                "available": False,
                "reason": "strict_causal_scale_unavailable",
                "frame_count": MINIMUM_CAUSAL_STREAM_FRAMES,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            self._restore(snapshot)
        self._arrival_metric_scale_result = dict(result)
        return result

    def _first40_local_pose_metric_scale(self):
        """Reuse the already-frozen MDTEC scale for robot-local control.

        The full-mono runtime freezes this receipt exactly once after causal
        frames 0..39 and already uses it for every NavDP depth observation.
        Replaying LingBot over a second 64-frame prefix inside every fresh
        real-world episode adds roughly a minute of avoidable startup latency.
        This path validates and exposes the existing immutable receipt; it
        never computes a pooled/oracle fallback and does not alter the strict
        GOAT ``/arrival_query`` policy.
        """

        from MemNavData.monocular_depth_runtime import (
            PREFIX_FRAMES,
            canonical_sha256,
            validate_first40_scale_receipt,
        )

        receipt = self._first40_scale_receipt
        if receipt is None:
            return {
                "available": False,
                "reason": "mdtec_first40_scale_unavailable",
                "frame_count": int(self.n),
            }
        try:
            validate_first40_scale_receipt(receipt)
            if receipt["scale_valid"] is not True:
                raise RuntimeError("frozen first-40 scale is invalid")
            scale = float(receipt["scale_hat"])
            result = {
                "available": True,
                "reason": "mdtec_first40_causal_scale_available",
                "frame_count": int(PREFIX_FRAMES),
                "metric_scale_m_per_raw": scale,
                "scale_receipt_sha256": canonical_sha256(receipt),
                "scale_evidence_contract": str(
                    receipt["scale_evidence_contract"]),
                "quality": {
                    "valid_frame_ratio": float(
                        receipt["valid_frame_ratio"]),
                    "relative_floor_iqr": float(
                        receipt["relative_floor_iqr"]),
                    "scale_clamped": float(
                        bool(receipt["scale_clamped"])),
                },
            }
        except Exception as error:
            result = {
                "available": False,
                "reason": "mdtec_first40_scale_invalid",
                "frame_count": int(PREFIX_FRAMES),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        return result

    @torch.no_grad()
    def certify_current_image_goal_arrival(
            self, goal_jpg_bytes, goal_camera_intrinsic=None,
            metric_scale_policy="strict_first64"):
        """Measure current-to-goal distance without mutating the live stream.

        This endpoint supplies geometry evidence only.  It never sees the
        native NavDP decision and therefore cannot authorize GOAT
        ``SUBTASK_STOP`` by itself; the runner combines its receipt with the
        typed native-zero proposal through ``goat_certified_arrival_contract``.
        """

        import hashlib
        import json
        import time

        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_EPIPOLAR_THRESHOLD_PX,
            certificate_decision,
            fundamental_can_reach_certificate,
            fundamental_support,
            scale_free_relative_xy,
        )
        from MemNavData.goat_certified_arrival_contract import (
            MINIMUM_CAUSAL_STREAM_FRAMES,
            contract_receipt,
        )
        from MemNavData.lingbot_pnp_localization import (
            SiftPnPConfig,
            correspondence_pnp_localize,
            map_raw_intrinsic_to_lingbot_pad,
        )

        started = time.monotonic()
        metric_scale_policy = str(metric_scale_policy)
        if metric_scale_policy not in {"strict_first64", "mdtec_first40"}:
            return {
                "schema_version": (
                    "lingbot_pnp_online_arrival_evidence_v2_20260818"),
                "status": "invalid_metric_scale_policy",
                "metric_scale_policy": metric_scale_policy,
                "certificate_accepted": False,
                "metric_scale_available": False,
                "predicted_distance_m": None,
                "simulator_depth_consumed": False,
            }
        if goal_camera_intrinsic is None:
            raw_goal_intrinsic = None
        else:
            try:
                raw_goal_intrinsic = np.asarray(
                    goal_camera_intrinsic, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raw_goal_intrinsic = np.empty((0, 0), dtype=np.float64)
        base = {
            "schema_version": "lingbot_pnp_online_arrival_evidence_v2_20260818",
            "frame_count": int(self.n),
            "frame_index": int(self.n - 1),
            "contract": contract_receipt(),
            "certificate_accepted": False,
            "metric_scale_available": False,
            "predicted_distance_m": None,
            "simulator_depth_consumed": False,
            "metric_scale_policy": metric_scale_policy,
            "goal_camera_calibration": (
                "explicit_distinct_intrinsic"
                if raw_goal_intrinsic is not None
                else "legacy_shared_history_intrinsic"
            ),
        }
        if (raw_goal_intrinsic is not None
                and (raw_goal_intrinsic.shape != (3, 3)
                     or not np.isfinite(raw_goal_intrinsic).all()
                     or raw_goal_intrinsic[0, 0] <= 0.0
                     or raw_goal_intrinsic[1, 1] <= 0.0)):
            return {**base, "status": "invalid_goal_camera_intrinsic"}
        if self.certified_relocalization_matcher is None:
            return {**base, "status": "matcher_disabled"}
        minimum_frames = (
            40 if metric_scale_policy == "mdtec_first40"
            else MINIMUM_CAUSAL_STREAM_FRAMES
        )
        if self.n < minimum_frames:
            return {**base, "status": "causal_scale_prefix_incomplete"}
        if self._last_agg is None or self._psi is None or not self.cam_pose:
            return {**base, "status": "live_lingbot_state_unavailable"}

        current_path = os.path.join(self.rgb_dir, f"{self.n - 1}.jpg")
        if not os.path.isfile(current_path):
            raise FileNotFoundError(current_path)
        goal_key = hashlib.sha256(goal_jpg_bytes).hexdigest()
        goal_path = os.path.join(
            self.rgb_dir, f"_arrival_goal_{goal_key}.jpg")
        if not os.path.isfile(goal_path):
            with open(goal_path, "wb") as handle:
                handle.write(goal_jpg_bytes)

        matched = self.certified_relocalization_matcher.match_paths(
            current_path,
            goal_path,
            target_height=int(self.lb.img_size),
            target_width=int(self.lb.img_size),
            patch_size=int(self.lb.patch_size),
        )
        support = fundamental_support(
            matched["reference_raw_points"],
            matched["query_raw_points"],
            matched["scores"],
            tuple(matched["reference_raw_hw"]),
            tuple(matched["query_raw_hw"]),
            threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        )
        possible, precheck_reason = fundamental_can_reach_certificate(support)
        support_json = {
            str(key): (value.item() if isinstance(value, np.generic) else value)
            for key, value in support.items()
        }
        if not possible:
            return {
                **base,
                **support_json,
                "status": str(precheck_reason),
                "precheck_passed": False,
                "runtime_s": float(time.monotonic() - started),
            }

        current_image = self._window_imgs[-1][None, None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.lb.model._predict_depth(
                self._last_agg, current_image, self._psi)
        depth = prediction["depth"][0, -1, ..., 0].float().cpu().numpy()
        confidence = prediction["depth_conf"][0, -1].float().cpu().numpy()
        current_pose = self.cam_pose[-1].float().cpu().numpy()
        query_intrinsic = None
        if raw_goal_intrinsic is not None:
            query_height, query_width = (
                int(value) for value in matched["query_raw_hw"])
            query_intrinsic = map_raw_intrinsic_to_lingbot_pad(
                raw_goal_intrinsic,
                raw_height=query_height,
                raw_width=query_width,
                target_height=int(depth.shape[-2]),
                target_width=int(depth.shape[-1]),
                patch_size=int(self.lb.patch_size),
            )
        pnp = correspondence_pnp_localize(
            matched["reference_points"],
            matched["query_points"],
            depth,
            confidence,
            current_pose,
            config=SiftPnPConfig(),
            match_scores=matched["scores"],
            epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX,
            query_intrinsic=query_intrinsic,
        )
        certificate = certificate_decision(pnp)
        result = {
            **base,
            **support_json,
            "status": str(pnp.get("status", precheck_reason)),
            "precheck_passed": True,
            "pnp_matches": int(pnp.get("matches", 0)),
            "pnp_epipolar_inliers": int(pnp.get("epipolar_inliers", 0)),
            "pnp_depth_valid_matches": int(
                pnp.get("depth_valid_matches", 0)),
            "pnp_inliers": int(pnp.get("inliers", 0)),
            "pnp_inlier_ratio": float(pnp.get("inlier_ratio", 0.0)),
            "pnp_reprojection_rmse_px": (
                float(pnp["reprojection_rmse_px"])
                if pnp.get("reprojection_rmse_px") is not None else None),
            "pnp_reference_inlier_coverage": (
                float(pnp["reference_inlier_coverage"])
                if pnp.get("reference_inlier_coverage") is not None else None),
            "pnp_query_inlier_coverage": (
                float(pnp["query_inlier_coverage"])
                if pnp.get("query_inlier_coverage") is not None else None),
            "certificate_accepted": bool(certificate["accepted"]),
            "certificate_reason": str(certificate["reason"]),
        }
        if not certificate["accepted"] or "pose9" not in pnp:
            result["runtime_s"] = float(time.monotonic() - started)
            return result

        # A certified two-view pose supports relative direction even when the
        # monocular metric scale is unavailable or poorly calibrated.  Expose
        # that scale-free quantity before consulting either scale policy.  In
        # particular, real-world local control must not promote a geometrically
        # valid bearing into centimetre-level arrival authority merely because
        # a causal floor-scale receipt exists.
        scale_free_xy = np.asarray(
            scale_free_relative_xy(
                current_pose, np.asarray(pnp["pose9"], dtype=np.float64)),
            dtype=np.float64,
        )
        result["predicted_scale_free_relative_xy"] = scale_free_xy.tolist()
        result["scale_free_direction_available"] = bool(
            np.isfinite(scale_free_xy).all()
            and float(np.linalg.norm(scale_free_xy)) > 1e-8
        )
        result.update(self._certified_view_alignment(pnp["pose9"]))

        if metric_scale_policy == "mdtec_first40":
            scale = self._first40_local_pose_metric_scale()
        else:
            scale = self._strict_arrival_metric_scale_preserving_stream()
        result["metric_scale"] = scale
        result["metric_scale_available"] = bool(scale["available"])
        if not scale["available"]:
            result["runtime_s"] = float(time.monotonic() - started)
            return result

        relative_xy = float(scale["metric_scale_m_per_raw"]) * scale_free_xy
        result["predicted_relative_xy_m"] = relative_xy.tolist()
        result["predicted_distance_m"] = float(np.linalg.norm(relative_xy))
        result["pnp_pose9"] = np.asarray(
            pnp["pose9"], dtype=np.float64).tolist()
        result["runtime_s"] = float(time.monotonic() - started)
        return result

    @torch.no_grad()
    def _certified_reference_depth_legacy(self, anchor):
        """Original full replay, retained as the equivalence oracle."""
        return self._certified_reference_depth_impl(anchor)

    @torch.no_grad()
    def _certified_reference_depth(self, anchor):
        """Return exact causal LingBot depth with an anchor-result cache.

        Candidate ranking is image-only, so this expensive full replay happens
        for at most one history frame per goal.  The first request for an anchor
        remains the independently confirmed full replay.  Later goals selecting
        that same immutable frame reuse its final arrays exactly; no intermediate
        transformer state is approximated.
        """
        anchor = int(anchor)
        cached = self._certified_reference_depth_cache.get(anchor)
        if cached is not None:
            depth, confidence = cached
            self._certified_dense_replay_last_stats = {
                "enabled": True,
                "anchor": anchor,
                "cache_hit": True,
                "cache_source": (
                    "eager_dense_writer"
                    if anchor in self._certified_eager_depth_cached_anchors
                    else "prior_selected_anchor"),
                "replayed_frames": 0,
                "cached_anchors": len(
                    self._certified_reference_depth_cache),
                "cache_bytes": int(sum(
                    array.nbytes for pair in
                    self._certified_reference_depth_cache.values()
                    for array in pair)),
            }
            return depth.copy(), confidence.copy()
        depth, confidence = self._certified_reference_depth_impl(anchor)
        self._certified_reference_depth_cache[anchor] = (
            depth.copy(), confidence.copy())
        self._certified_dense_replay_last_stats = {
            "enabled": True,
            "anchor": anchor,
            "cache_hit": False,
            "replayed_frames": anchor - self.S + 1,
            "cached_anchors": len(self._certified_reference_depth_cache),
            "cache_bytes": int(sum(
                array.nbytes for pair in
                self._certified_reference_depth_cache.values()
                for array in pair)),
        }
        return depth, confidence

    @torch.no_grad()
    def _certified_route_reference_depth(self, anchor):
        """Return one write-once sparse depth observation without replay.

        This cache is generated by the same causal short-range LingBot stream
        that already supplies NavDP depth.  It never enters the initial target
        certificate and never launches a second dense KV stream.
        """

        anchor = int(anchor)
        cached = self._certified_route_reference_depth_cache.get(anchor)
        if cached is None:
            raise RuntimeError(
                f"route depth is unavailable for historical frame {anchor}")
        depth, confidence = cached
        self._certified_dense_replay_last_stats = {
            "enabled": True,
            "anchor": anchor,
            "cache_hit": True,
            "cache_source": "sparse_causal_route_writer",
            "replayed_frames": 0,
            "cached_anchors": len(
                self._certified_route_reference_depth_cache),
            "cache_bytes": int(sum(
                array.nbytes for pair in
                self._certified_route_reference_depth_cache.values()
                for array in pair)),
        }
        return depth.copy(), confidence.copy()

    @torch.no_grad()
    def _materialize_current_route_depth(self):
        """Write the current LingBot relative depth into the sparse route cache.

        Route tracking plans less frequently than the RGB stream is written.
        The fixed-stride writer covers intermediate frames; this method binds
        the exact current planning frame so the next interval always begins
        from a depth-bearing reference.  It runs the already-loaded frozen
        depth head and does not mutate the LingBot causal stream.
        """

        frame = int(self.n - 1)
        cached = self._certified_route_reference_depth_cache.get(frame)
        if cached is not None:
            depth, confidence = cached
            return depth.copy(), confidence.copy(), "sparse_causal_route_writer"
        live_cached = self._certified_route_live_depth_cache.pop(frame, None)
        if live_cached is not None:
            depth, confidence = live_cached
            self._certified_route_reference_depth_cache[frame] = (
                depth.copy(), confidence.copy())
            self._certified_route_depth_cached_anchors.add(frame)
            return depth.copy(), confidence.copy(), "dense_query_writer"
        if frame < 40:
            raise RuntimeError(
                "current route depth predates the first-40 metric receipt")
        if self._last_agg is None or self._psi is None or not self._window_imgs:
            raise RuntimeError("current LingBot depth state is unavailable")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.lb.model._predict_depth(
                self._last_agg,
                self._window_imgs[-1][None, None].to(self.device),
                self._psi,
            )
        depth = prediction[
            "depth"][0, -1, ..., 0].float().cpu().numpy()
        confidence = prediction[
            "depth_conf"][0, -1].float().cpu().numpy()
        if (depth.ndim != 2 or confidence.shape != depth.shape
                or not np.isfinite(depth).all()
                or not np.isfinite(confidence).all()
                or np.any(depth < 0.0)):
            raise RuntimeError("current LingBot route depth is malformed")
        self._certified_route_reference_depth_cache[frame] = (
            depth.copy(), confidence.copy())
        self._certified_route_depth_cached_anchors.add(frame)
        return depth.copy(), confidence.copy(), "current_plan_frame_writer"

    @torch.no_grad()
    def _certified_metric_route_depth(self, frame, *, target_anchor):
        """Return frame-bound route depth in the immutable first-40 gauge."""

        frame = int(frame)
        target = int(target_anchor)
        scale_receipt = self._first40_local_pose_metric_scale()
        if scale_receipt.get("available") is not True:
            raise RuntimeError(str(scale_receipt.get(
                "reason", "first40 metric scale unavailable")))
        scale = float(scale_receipt["metric_scale_m_per_raw"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("first-40 route scale is invalid")

        cached = self._certified_route_reference_depth_cache.get(frame)
        if cached is not None:
            raw_depth, _confidence = cached
            source = "sparse_causal_route_writer"
        elif frame in self._certified_route_live_depth_cache:
            raw_depth, _confidence = self._certified_route_live_depth_cache[
                frame]
            source = "dense_query_writer"
        elif frame == target:
            raw_depth, _confidence = self._certified_reference_depth(frame)
            source = "cec_selected_anchor_depth"
        elif frame == int(self.n - 1):
            raw_depth, _confidence, source = (
                self._materialize_current_route_depth())
        else:
            raise RuntimeError(
                f"no causal route depth receipt for frame {frame}")
        metric = scale * np.asarray(raw_depth, dtype=np.float64)
        if (metric.ndim != 2 or not np.isfinite(metric).all()
                or np.any(metric < 0.0)):
            raise RuntimeError("metricized route depth is malformed")
        return metric, {
            "frame_index": frame,
            "source": source,
            "metric_scale_m_per_raw": scale,
            "scale_receipt_sha256": scale_receipt.get(
                "scale_receipt_sha256"),
        }

    @torch.no_grad()
    def _certified_visual_route_edge(
            self, *, reference_frame, query_frame, target_anchor,
            invert_estimate=False, edge_kind="adjacent_sample"):
        """Estimate and bind one causal RGB/depth local-motion edge."""

        from pathlib import Path
        from PIL import Image

        from MemNavData.monocular_adjacent_motion import (
            estimate_adjacent_motion,
            invert_planar_motion,
        )
        from MemNavData.certified_relocalization_contract import (
            CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        )
        from MemNavData.monocular_route_tangent_runtime import (
            accepted_planar_motion,
        )

        reference = int(reference_frame)
        query = int(query_frame)
        reference_path = Path(self.rgb_dir) / f"{reference}.jpg"
        query_path = Path(self.rgb_dir) / f"{query}.jpg"
        if not reference_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(
                reference_path if not reference_path.is_file() else query_path)
        if self.camera_intrinsic is None:
            raise RuntimeError("route motion requires camera intrinsics")
        with Image.open(reference_path) as image:
            raw_width, raw_height = image.size
        with Image.open(query_path) as image:
            query_size = image.size
        if query_size != (raw_width, raw_height):
            raise RuntimeError("route RGB resolution changed within an episode")
        metric_depth, depth_receipt = self._certified_metric_route_depth(
            reference, target_anchor=int(target_anchor))
        resolved_motion_model = certified_route_edge_motion_model(
            self.certified_route_motion_model, edge_kind)
        estimate = estimate_adjacent_motion(
            reference_path=reference_path,
            query_path=query_path,
            reference_metric_depth=metric_depth,
            raw_camera_intrinsic=self.camera_intrinsic,
            matcher=self.certified_relocalization_matcher,
            raw_height=int(raw_height),
            raw_width=int(raw_width),
            patch_size=int(self.lb.patch_size),
            epipolar_threshold_px=(
                CERTIFIED_EPIPOLAR_THRESHOLD_PX
                if resolved_motion_model == "fundamental_then_pnp"
                else None),
            run_unfiltered_pnp_shadow=(
                self.certified_route_motion_unfiltered_shadow),
        )
        pnp = estimate.get("pnp", {})
        reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        query_sha = hashlib.sha256(query_path.read_bytes()).hexdigest()
        edge_diagnostic = {
            "edge_kind": str(edge_kind),
            "match_reference_frame": reference,
            "match_query_frame": query,
            "match_reference_sha256": reference_sha,
            "match_query_sha256": query_sha,
            "depth_receipt": depth_receipt,
            "requested_route_motion_model": (
                self.certified_route_motion_model),
            "resolved_edge_motion_model": resolved_motion_model,
            "primary_motion_model": estimate.get("primary_motion_model"),
            "epipolar_threshold_px": estimate.get(
                "epipolar_threshold_px"),
            "primary_local_motion_validity": estimate.get(
                "local_motion_validity"),
            "primary_pnp": pnp,
            "unfiltered_pnp_shadow": estimate.get(
                "unfiltered_pnp_shadow"),
        }
        try:
            motion = accepted_planar_motion(estimate)
        except RuntimeError as error:
            # The shadow is diagnostic-only.  Bind it to the exception so the
            # typed geometry-stop packet can expose the same-edge comparison
            # without changing primary control or falling back to it.
            error.route_motion_diagnostic = edge_diagnostic
            raise
        if invert_estimate:
            motion = invert_planar_motion(motion)
        return motion, {
            "edge_kind": str(edge_kind),
            "from_frame": int(query if invert_estimate else reference),
            "to_frame": int(reference if invert_estimate else query),
            "match_reference_frame": reference,
            "match_query_frame": query,
            "match_reference_sha256": reference_sha,
            "match_query_sha256": query_sha,
            "estimate_inverted": bool(invert_estimate),
            "depth_receipt": depth_receipt,
            "requested_route_motion_model": (
                self.certified_route_motion_model),
            "resolved_edge_motion_model": resolved_motion_model,
            "matches": int(estimate.get("matches", 0)),
            "pnp_status": pnp.get("status"),
            "pnp_inliers": int(pnp.get("inliers", 0)),
            "pnp_reprojection_rmse_px": pnp.get(
                "reprojection_rmse_px"),
            "primary_motion_model": estimate.get("primary_motion_model"),
            "epipolar_threshold_px": estimate.get(
                "epipolar_threshold_px"),
            "unfiltered_pnp_shadow": estimate.get(
                "unfiltered_pnp_shadow"),
            "motion": motion.audit_dict(),
        }

    @torch.no_grad()
    def _certified_reference_depth_impl(self, anchor):
        """Original exact replay from the frozen scale block through anchor."""
        anchor = int(anchor)
        if anchor < self.S or anchor >= self.n:
            raise ValueError(
                f"certified anchor {anchor} outside [{self.S}, {self.n - 1}]")
        snap = self._snapshot()
        try:
            cache = self._live_cache()
            indices = cache.get("anchor_frame_indices")
            if indices is None:
                self.lb._inject(
                    cache["scale_k"], cache["scale_v"],
                    cache["anchor_k"], cache["anchor_v"],
                    n_hist=0, total_frames=self.S)
            else:
                self.lb._inject(
                    cache["scale_k"], cache["scale_v"],
                    cache["anchor_k"], cache["anchor_v"],
                    anchor_frame_indices=indices, raw_start=self.S)
            final_agg = final_psi = final_image = None
            # Bound host memory while retaining exact sequential inference.
            for chunk_start in range(self.S, anchor + 1, 16):
                chunk_end = min(anchor + 1, chunk_start + 16)
                paths = [
                    os.path.join(self.rgb_dir, f"{index}.jpg")
                    for index in range(chunk_start, chunk_end)
                ]
                if not all(os.path.isfile(path) for path in paths):
                    missing = next(path for path in paths
                                   if not os.path.isfile(path))
                    raise FileNotFoundError(missing)
                images = self.lb.load_images(paths)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for offset in range(len(images)):
                        final_image = images[offset:offset + 1][None].to(
                            self.device)
                        final_agg, final_psi = (
                            self.lb.model._aggregate_features(
                                final_image,
                                num_frame_for_scale=self.S,
                                num_frame_per_block=1))
            if final_agg is None or final_image is None:
                raise RuntimeError("certified anchor replay produced no frame")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.lb.model._predict_depth(
                    final_agg, final_image, final_psi)
            depth = prediction["depth"][0, -1, ..., 0].float().cpu().numpy()
            confidence = prediction["depth_conf"][0, -1].float().cpu().numpy()
            return depth, confidence
        finally:
            self._restore(snap)

    @torch.no_grad()
    def _certified_bearing_vector(self, goal_pose9):
        """Convert a certified pose to scale-free ``[forward, left]``.

        LingBot translation scale is monocular and is not certified by the v2
        image-geometry checks.  The relative direction is scale invariant, so
        the runtime boundary deliberately does not call ``_get_metric_scale``.
        ``verified_bearing_v1`` performs the only metric operation later: a
        frozen projection to its already validated 2.5 m controller radius.
        """
        from MemNavData.certified_relocalization_runtime import (
            scale_free_relative_xy,
        )

        pose = np.asarray(goal_pose9, dtype=np.float64)
        if pose.shape != (9,) or not np.isfinite(pose).all():
            raise ValueError("certified goal pose must be finite pose9")
        current_pose = self.cam_pose[-1].float().cpu().numpy()
        return scale_free_relative_xy(current_pose, pose)

    @torch.no_grad()
    def _certified_view_alignment(self, goal_pose9):
        """Return PnP-derived terminal view residuals, never a STOP decision."""
        from MemNavData.goat_terminal_alignment import (
            relative_optical_yaw_pitch_deg,
        )

        pose = np.asarray(goal_pose9, dtype=np.float64)
        if pose.shape != (9,) or not np.isfinite(pose).all():
            raise ValueError("certified goal pose must be finite pose9")
        current_pose = self.cam_pose[-1].float().cpu().numpy()
        yaw_right, pitch_up = relative_optical_yaw_pitch_deg(
            current_pose, pose)
        return {
            "terminal_yaw_right_deg": float(yaw_right),
            "terminal_pitch_up_deg": float(pitch_up),
            "terminal_alignment_source": (
                "certified_lingbot_current_to_pnp_goal_rotation"),
            "terminal_alignment_stop_authority": False,
        }

    @torch.no_grad()
    def _certified_path_field_direction(
            self, *, goal_key, target_anchor, goal_start_frame,
            goal_pose9, current_pose9_override=None,
            route_progress_hint_m=None):
        """Read one local bearing from an ordered causal LingBot path.

        This is the deployment-stage counterpart of the evaluator-route
        information audit.  It consumes only the live LingBot pose stream,
        the immutable first-40 scale receipt, the certificate-selected anchor,
        and the PnP terminal witness.  It never sees Habitat position, a
        shortest path, or a Novel/Revisit role.  Any malformed geometry is
        returned as an explicit receipt with no direction; the experimental
        evaluator treats that state as an arm failure rather than silently
        executing the endpoint chord.
        """
        from MemNavData.episodic_path_field import FrozenEpisodicPath

        diagnostics = {
            "episodic_path_field_requested": True,
            "episodic_path_field_status": "geometry_failure",
            "episodic_path_field_error_type": None,
            "episodic_path_field_error": None,
            "episodic_path_field_goal_start_frame": int(goal_start_frame),
            "episodic_path_field_target_anchor": int(target_anchor),
            "episodic_path_field_runtime_geometry": (
                "causal_lingbot_cam_pose_only"),
            "episodic_path_field_evaluator_pose_consumed": False,
            "episodic_path_field_habitat_path_consumed": False,
        }
        try:
            goal_start = int(goal_start_frame)
            target = int(target_anchor)
            if not self.cam_pose:
                raise RuntimeError("live LingBot pose stream is empty")
            if goal_start < 2 or goal_start > len(self.cam_pose):
                raise ValueError("goal start is outside the live pose stream")
            if target < 0 or target >= goal_start:
                raise ValueError("target anchor violates the causal prefix")

            terminal = np.asarray(goal_pose9, dtype=np.float64)
            if terminal.shape != (9,) or not np.isfinite(terminal).all():
                raise ValueError("PnP terminal witness must be finite pose9")
            receipt = self._first40_local_pose_metric_scale()
            if receipt.get("available") is not True:
                raise RuntimeError(str(receipt.get(
                    "reason", "first40 metric scale unavailable")))
            metric_scale = float(receipt["metric_scale_m_per_raw"])
            if not np.isfinite(metric_scale) or metric_scale <= 0.0:
                raise ValueError("first40 metric scale must be positive")

            route = self._certified_path_field_routes.get(goal_key)
            if route is None:
                translations = np.stack([
                    pose.float().cpu().numpy()[:3]
                    for pose in self.cam_pose[:goal_start]
                ], axis=0)
                path = FrozenEpisodicPath.from_history(
                    translations,
                    start_index=goal_start - 1,
                    anchor_index=target,
                    metric_scale_m_per_raw=metric_scale,
                    terminal_translation=terminal,
                )
                route = {
                    "path": path,
                    "goal_start_frame": goal_start,
                    "target_anchor": target,
                    "metric_scale_m_per_raw": metric_scale,
                    "scale_receipt_sha256": receipt.get(
                        "scale_receipt_sha256"),
                    "progress_m": 0.0,
                }
                self._certified_path_field_routes[goal_key] = route
            elif (int(route["goal_start_frame"]) != goal_start
                  or int(route["target_anchor"]) != target):
                raise RuntimeError("frozen episodic path contract changed")

            current = (
                self.cam_pose[-1].float().cpu().numpy()
                if current_pose9_override is None
                else np.asarray(current_pose9_override, dtype=np.float64)
            )
            if current.shape != (9,) or not np.isfinite(current).all():
                raise ValueError(
                    "path-field current pose must be a finite pose9")
            minimum_progress = float(route["progress_m"])
            if route_progress_hint_m is not None:
                hint = float(route_progress_hint_m)
                if not np.isfinite(hint):
                    raise ValueError("route progress hint must be finite")
                # A local visual witness observes route state; it grants no
                # new control authority. It may advance the monotone address
                # beyond one controller horizon because appearance addressing
                # and 2.5 m local control are distinct contracts.
                minimum_progress = max(minimum_progress, hint)
            guidance = route["path"].guidance(
                current,
                minimum_progress_m=minimum_progress,
            )
            if guidance.progress_m + 1e-9 < float(route["progress_m"]):
                raise RuntimeError("episodic path progress regressed")
            route["progress_m"] = float(guidance.progress_m)
            diagnostics.update(
                guidance.audit_dict(),
                episodic_path_field_status=(
                    "complete" if guidance.complete else "active"),
                episodic_path_field_metric_scale_m_per_raw=metric_scale,
                episodic_path_field_scale_receipt_sha256=route.get(
                    "scale_receipt_sha256"),
            )
            return (
                None if guidance.unit_bearing is None
                else [float(value) for value in guidance.unit_bearing],
                diagnostics,
            )
        except Exception as error:
            diagnostics.update(
                episodic_path_field_error_type=type(error).__name__,
                episodic_path_field_error=str(error),
            )
            return None, diagnostics

    @torch.no_grad()
    def _certified_action_coordinate_direction(
            self, *, goal_key, target_anchor, goal_start_frame):
        """Read a route bearing in the executor's continuous coordinate.

        Initial CEC proof fixes the historical anchor. The causal LingBot
        positions provide only route *shape*; their translation magnitude is
        never metricized. Realized executor translation parameterizes progress
        along the same historical route, while realized yaw rotates its tangent
        into the current body frame. Every valid receipt has one total output:
        there is no distance regime, visual update gate, endpoint controller,
        or native fallback after authorization.
        """

        from MemNavData.episodic_route_filter import (
            ActionCoordinateRouteCompass,
        )

        diagnostics = {
            "action_coordinate_compass_requested": True,
            "action_coordinate_compass_status": "geometry_failure",
            "action_coordinate_compass_error_type": None,
            "action_coordinate_compass_error": None,
            "action_coordinate_goal_start_frame": int(goal_start_frame),
            "action_coordinate_target_anchor": int(target_anchor),
            "action_coordinate_runtime_geometry": (
                "scale_free_causal_lingbot_route_plus_executor_receipts"),
            "action_coordinate_evaluator_pose_consumed": False,
            "action_coordinate_habitat_path_consumed": False,
            "action_coordinate_metric_scale_consumed": False,
            "action_coordinate_visual_gate_present_after_initialization": (
                False),
            "action_coordinate_distance_regime_present": False,
            "action_coordinate_endpoint_fallback_available": False,
            "action_coordinate_native_fallback_available": False,
        }
        try:
            goal_start = int(goal_start_frame)
            target = int(target_anchor)
            current_frame = int(self.n - 1)
            if (goal_start < 2 or goal_start > current_frame
                    or goal_start >= len(self.cam_pose)):
                raise ValueError(
                    "action-coordinate goal start is outside the pose stream")
            if target < 0 or target >= goal_start:
                raise ValueError(
                    "action-coordinate anchor violates the causal prefix")
            if len(self.executor_motion_receipts) != self.n:
                raise RuntimeError(
                    "executor receipts are not aligned with causal RGB")

            route = self._certified_action_coordinate_routes.get(goal_key)
            if route is None:
                route_indices = np.arange(
                    goal_start, target - 1, -1, dtype=np.int64)
                model_route = np.stack([
                    self.cam_pose[int(index)].float().cpu().numpy()[[0, 2]]
                    for index in route_indices
                ], axis=0)
                historical = []
                action_edges = []
                # Receipt i is motion from RGB i-1 to RGB i. Traversing the
                # observed route backwards therefore uses receipt i on edge
                # i -> i-1, without exposing either endpoint's global pose.
                for index in range(goal_start, target, -1):
                    receipt = self.executor_motion_receipts[index]
                    if receipt is None:
                        raise RuntimeError(
                            "authorized history has an unbound executor "
                            f"receipt at frame {index}")
                    if int(receipt["frame_index"]) != index:
                        raise RuntimeError(
                            "executor receipt frame binding changed")
                    historical.append(dict(receipt))
                    action_edges.append(float(
                        receipt["executed_translation_m"]))
                action_arc = np.concatenate([
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(np.asarray(action_edges, dtype=np.float64)),
                ])
                compass = ActionCoordinateRouteCompass(
                    model_route,
                    action_arc,
                    self.cam_pose[goal_start].float().cpu().numpy(),
                    lookahead_m=2.5,
                    controller_radius_m=2.5,
                )
                receipt_bytes = json.dumps(
                    historical, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
                route = {
                    "compass": compass,
                    "goal_start_frame": goal_start,
                    "target_anchor": target,
                    "last_consumed_frame": goal_start,
                    "history_receipt_sha256": hashlib.sha256(
                        receipt_bytes).hexdigest(),
                    "history_receipt_count": len(historical),
                }
                self._certified_action_coordinate_routes[goal_key] = route
            elif (int(route["goal_start_frame"]) != goal_start
                  or int(route["target_anchor"]) != target):
                raise RuntimeError(
                    "frozen action-coordinate route contract changed")

            last_frame = int(route["last_consumed_frame"])
            if current_frame < last_frame:
                raise RuntimeError("action-coordinate frame index regressed")
            pending = self.executor_motion_receipts[
                last_frame + 1:current_frame + 1]
            if any(receipt is None for receipt in pending):
                missing = last_frame + 1 + next(
                    index for index, receipt in enumerate(pending)
                    if receipt is None)
                raise RuntimeError(
                    "query has an unbound executor receipt at frame "
                    f"{missing}")
            translation = float(sum(
                float(receipt["executed_translation_m"])
                for receipt in pending))
            yaw = float(sum(
                float(receipt["executed_yaw_rad"])
                for receipt in pending))
            readout = route["compass"].advance(
                executed_translation_m=translation,
                executed_yaw_rad=yaw,
            )
            route["last_consumed_frame"] = current_frame
            diagnostics.update(
                # Retain the readout's compact internal receipt for direct
                # unit tests/debugging.  The explicitly prefixed duplicates
                # below are the stable HTTP/episode serialization contract.
                readout.audit_dict(),
                action_coordinate_compass_status=(
                    "terminal_tangent"
                    if readout.terminal_tangent_held else "active"),
                action_coordinate_schema_version=readout.schema_version,
                action_coordinate_progress_m=float(
                    readout.action_progress_m),
                action_coordinate_route_progress_fraction=float(
                    readout.route_progress_fraction),
                action_coordinate_route_state_index=int(
                    readout.route_state_index),
                action_coordinate_reference_action_arc_m=float(
                    readout.reference_action_arc_m),
                action_coordinate_terminal_tangent_held=bool(
                    readout.terminal_tangent_held),
                action_coordinate_cumulative_executor_yaw_rad=float(
                    readout.cumulative_executor_yaw_rad),
                action_coordinate_unit_bearing=[
                    float(value) for value in readout.unit_bearing],
                action_coordinate_controller_pointgoal=[
                    float(value) for value in readout.controller_pointgoal],
                action_coordinate_history_receipt_sha256=route[
                    "history_receipt_sha256"],
                action_coordinate_history_receipt_count=int(route[
                    "history_receipt_count"]),
                action_coordinate_last_consumed_frame=current_frame,
                action_coordinate_update_translation_m=translation,
                action_coordinate_update_yaw_rad=yaw,
                action_coordinate_route_extent_m=float(
                    route["compass"].route_action_extent_m),
            )
            return [float(value) for value in readout.unit_bearing], diagnostics
        except Exception as error:
            diagnostics.update(
                action_coordinate_compass_error_type=type(error).__name__,
                action_coordinate_compass_error=str(error),
            )
            return None, diagnostics

    @torch.no_grad()
    def _certified_se2_route_direction(
            self, *, goal_key, target_anchor, goal_start_frame):
        """Read a cross-track-correcting bearing in a local SE(2) route.

        CEC still supplies the one initial geometric proof and historical
        anchor.  After authorization, the historical route and live query pose
        are both reconstructed from the same frame-bound executor translation
        and yaw receipts.  Progress is the monotone orthogonal projection of
        the estimated live 2-D position onto that route; travelled distance is
        never equated with route progress.

        The motion contract is frame-bound local executor odometry.  The
        policy never reads a world-frame pose or target-relative displacement;
        the Habitat development arm currently forms this odometry proxy by
        differencing consecutive simulator poses, which is recorded explicitly
        in every receipt.  No LingBot translation scale, visual update gate,
        distance regime, endpoint controller, or native fallback is available
        after authorization.
        """

        from MemNavData.se2_projected_route_compass import (
            SE2_LOCAL_ODOMETRY_MODEL,
            SE2ProjectedRouteCompass,
        )

        diagnostics = {
            "se2_route_compass_requested": True,
            "se2_route_compass_status": "geometry_failure",
            "se2_route_compass_error_type": None,
            "se2_route_compass_error": None,
            "se2_route_goal_start_frame": int(goal_start_frame),
            "se2_route_target_anchor": int(target_anchor),
            "se2_route_runtime_geometry": (
                "local_executor_se2_route_plus_monotone_projection"),
            "se2_route_executor_motion_model": SE2_LOCAL_ODOMETRY_MODEL,
            "se2_route_evaluator_pose_consumed_as_odometry_proxy": True,
            "se2_route_world_pose_consumed_by_policy": False,
            "se2_route_executor_odometry_required": True,
            "se2_route_executor_odometry_source": None,
            "se2_route_habitat_path_consumed": False,
            "se2_route_lingbot_translation_consumed": False,
            "se2_route_metric_scale_consumed": False,
            "se2_route_visual_gate_present_after_initialization": False,
            "se2_route_distance_regime_present": False,
            "se2_route_endpoint_fallback_available": False,
            "se2_route_native_fallback_available": False,
        }
        try:
            goal_start = int(goal_start_frame)
            target = int(target_anchor)
            current_frame = int(self.n - 1)
            if goal_start < 2 or goal_start > current_frame:
                raise ValueError(
                    "SE(2) route goal start is outside the causal stream")
            if target < 0 or target >= goal_start:
                raise ValueError(
                    "SE(2) route anchor violates the causal prefix")
            if len(self.executor_motion_receipts) != self.n:
                raise RuntimeError(
                    "executor receipts are not aligned with causal RGB")

            routes = getattr(self, "_certified_se2_route_compasses", None)
            if routes is None:
                # Backward-compatible construction for tests and restored
                # agents created before this development mode existed.
                routes = {}
                self._certified_se2_route_compasses = routes
            route = routes.get(goal_key)
            if route is None:
                historical = []
                forwards = []
                lefts = []
                yaws = []
                odometry_sources = set()
                # Receipt i describes frame i-1 -> i.  Iterating from the
                # current query boundary towards the certified anchor supplies
                # newest-to-oldest edges to the exact inverse integrator.
                for index in range(goal_start, target, -1):
                    receipt = self.executor_motion_receipts[index]
                    if receipt is None:
                        raise RuntimeError(
                            "authorized history has an unbound executor "
                            f"receipt at frame {index}")
                    if int(receipt["frame_index"]) != index:
                        raise RuntimeError(
                            "executor receipt frame binding changed")
                    local = receipt.get("local_se2")
                    if (not isinstance(local, dict)
                            or local.get("contract")
                            != "frame_bound_local_se2_v1"):
                        raise RuntimeError(
                            "authorized history lacks a local SE(2) receipt at "
                            f"frame {index}")
                    forward = float(local["executed_forward_m"])
                    left = float(local["executed_left_m"])
                    yaw = float(local["executed_yaw_rad"])
                    source = local.get("source")
                    if not isinstance(source, str) or not source:
                        raise RuntimeError(
                            "authorized history lacks an explicit local "
                            f"SE(2) source at frame {index}")
                    odometry_sources.add(source)
                    historical.append(dict(receipt))
                    forwards.append(forward)
                    lefts.append(left)
                    yaws.append(yaw)
                compass = (
                    SE2ProjectedRouteCompass.from_reverse_local_se2_receipts(
                    forwards,
                    lefts,
                    yaws,
                    lookahead_m=2.5,
                    controller_radius_m=2.5,
                ))
                receipt_bytes = json.dumps(
                    historical, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
                route = {
                    "compass": compass,
                    "goal_start_frame": goal_start,
                    "target_anchor": target,
                    "last_consumed_frame": goal_start,
                    "history_receipt_sha256": hashlib.sha256(
                        receipt_bytes).hexdigest(),
                    "history_receipt_count": len(historical),
                    "query_receipt_count": 0,
                    "odometry_source": None,
                }
                if len(odometry_sources) != 1:
                    raise RuntimeError(
                        "authorized history mixes local SE(2) sources")
                route["odometry_source"] = next(iter(odometry_sources))
                routes[goal_key] = route
            elif (int(route["goal_start_frame"]) != goal_start
                  or int(route["target_anchor"]) != target):
                raise RuntimeError("frozen SE(2) route contract changed")

            last_frame = int(route["last_consumed_frame"])
            if current_frame < last_frame:
                raise RuntimeError("SE(2) route frame index regressed")
            pending = self.executor_motion_receipts[
                last_frame + 1:current_frame + 1]
            if any(receipt is None for receipt in pending):
                missing = last_frame + 1 + next(
                    index for index, receipt in enumerate(pending)
                    if receipt is None)
                raise RuntimeError(
                    "query has an unbound executor receipt at frame "
                    f"{missing}")

            # SE(2) increments do not commute.  Consume every bound frame in
            # order rather than summing a planning interval into one scalar.
            readout = None
            update_translation = 0.0
            update_forward = 0.0
            update_left = 0.0
            update_yaw = 0.0
            for offset, receipt in enumerate(pending, start=last_frame + 1):
                if int(receipt["frame_index"]) != offset:
                    raise RuntimeError(
                        "query executor receipt frame binding changed")
                local = receipt.get("local_se2")
                if (not isinstance(local, dict)
                        or local.get("contract")
                        != "frame_bound_local_se2_v1"):
                    raise RuntimeError(
                        "query lacks a local SE(2) receipt at frame "
                        f"{offset}")
                translation = float(receipt["executed_translation_m"])
                forward = float(local["executed_forward_m"])
                left = float(local["executed_left_m"])
                yaw = float(local["executed_yaw_rad"])
                if local.get("source") != route["odometry_source"]:
                    raise RuntimeError(
                        "query local SE(2) source differs from history")
                readout = route["compass"].advance_local_se2(
                    executed_forward_m=forward,
                    executed_left_m=left,
                    executed_yaw_rad=yaw,
                )
                update_translation += translation
                update_forward += forward
                update_left += left
                update_yaw += yaw
            if readout is None:
                readout = route["compass"].advance_local_se2(
                    executed_forward_m=0.0,
                    executed_left_m=0.0,
                    executed_yaw_rad=0.0,
                )

            route["last_consumed_frame"] = current_frame
            route["query_receipt_count"] += len(pending)
            diagnostics.update(
                readout.audit_dict(),
                se2_route_compass_status=(
                    "terminal_tangent"
                    if readout.terminal_tangent_held else "active"),
                se2_route_schema_version=readout.schema_version,
                se2_route_projected_progress_m=float(
                    readout.projected_progress_m),
                se2_route_progress_fraction=float(
                    readout.route_progress_fraction),
                se2_route_state_index=int(readout.route_state_index),
                se2_route_reference_arc_m=float(readout.reference_arc_m),
                se2_route_query_path_length_m=float(
                    readout.query_path_length_m),
                se2_route_cross_track_error_m=float(
                    readout.cross_track_error_m),
                se2_route_estimated_position=[
                    float(value) for value in readout.estimated_position],
                se2_route_estimated_yaw_rad=float(
                    readout.estimated_yaw_rad),
                se2_route_projected_position=[
                    float(value) for value in readout.projected_position],
                se2_route_reference_position=[
                    float(value) for value in readout.reference_position],
                se2_route_terminal_tangent_held=bool(
                    readout.terminal_tangent_held),
                se2_route_unit_bearing=[
                    float(value) for value in readout.unit_bearing],
                se2_route_controller_pointgoal=[
                    float(value) for value in readout.controller_pointgoal],
                se2_route_history_receipt_sha256=route[
                    "history_receipt_sha256"],
                se2_route_history_receipt_count=int(route[
                    "history_receipt_count"]),
                se2_route_query_receipt_count=int(route[
                    "query_receipt_count"]),
                se2_route_last_consumed_frame=current_frame,
                se2_route_update_translation_m=float(update_translation),
                se2_route_update_forward_sum_m=float(update_forward),
                se2_route_update_left_sum_m=float(update_left),
                se2_route_update_yaw_rad=float(update_yaw),
                se2_route_extent_m=float(route["compass"].route_extent_m),
                se2_route_executor_odometry_source=str(
                    route["odometry_source"]),
            )
            return [float(value) for value in readout.unit_bearing], diagnostics
        except Exception as error:
            diagnostics.update(
                se2_route_compass_error_type=type(error).__name__,
                se2_route_compass_error=str(error),
            )
            return None, diagnostics

    @torch.no_grad()
    def _certified_terminal_route_motion(self, target_anchor, goal_pose9):
        """Return the CEC-proven goal->anchor edge in the route gauge."""

        from MemNavData.lingbot_colored_registration import (
            quaternion_xyzw_to_matrix,
        )
        from MemNavData.monocular_adjacent_motion import (
            invert_planar_motion,
            PlanarMotionReceipt,
        )

        target = int(target_anchor)
        if target < 0 or target >= len(self.cam_pose):
            raise ValueError("terminal route anchor is outside the pose stream")
        anchor = self.cam_pose[target].float().cpu().numpy().astype(
            np.float64)
        goal = np.asarray(goal_pose9, dtype=np.float64)
        if (anchor.shape != (9,) or goal.shape != (9,)
                or not np.isfinite(anchor).all()
                or not np.isfinite(goal).all()):
            raise ValueError("terminal route witness must contain finite pose9")
        scale_receipt = self._first40_local_pose_metric_scale()
        if scale_receipt.get("available") is not True:
            raise RuntimeError(str(scale_receipt.get(
                "reason", "first40 metric scale unavailable")))
        scale = float(scale_receipt["metric_scale_m_per_raw"])
        anchor_rotation = quaternion_xyzw_to_matrix(anchor[3:7])
        goal_rotation = quaternion_xyzw_to_matrix(goal[3:7])
        relative = anchor_rotation.T @ (goal[:3] - anchor[:3])
        anchor_to_goal = PlanarMotionReceipt(
            forward_m=float(scale * relative[2]),
            left_m=float(-scale * relative[0]),
            yaw_rad=float(lingbot_relative_yaw(
                anchor_rotation.T @ goal_rotation)),
            vertical_m=float(scale * relative[1]),
        )
        goal_to_anchor = invert_planar_motion(anchor_to_goal)
        return goal_to_anchor, {
            "edge_kind": "cec_terminal_geometric_witness",
            "from_frame": "goal_image",
            "to_frame": target,
            "metric_scale_m_per_raw": scale,
            "scale_receipt_sha256": scale_receipt.get(
                "scale_receipt_sha256"),
            "motion": goal_to_anchor.audit_dict(),
        }

    @torch.no_grad()
    def _certified_monocular_route_tangent_direction(
            self, *, goal_key, target_anchor, goal_start_frame,
            goal_pose9, authority_proof=None,
            goal_image_sha256=None):
        """Read one continuous local tangent from a causal monocular route.

        CEC supplies the sole open-set authorization and target anchor.  The
        intervening history and every later query update are reconstructed
        from adjacent RGB correspondences and height-scaled LingBot depth.
        The route state advances by monotone projection under a cumulative
        visual-motion budget.  Its total readout is always the first feasible
        forward route tangent, normalized to the fixed NavDP controller radius.

        There is no distance classifier, update gate, endpoint controller,
        stuck trigger, or native fallback after CEC acceptance.  A broken
        visual-geometry chain is an explicit geometry-stream failure.
        """

        from MemNavData.monocular_adjacent_motion import (
            PlanarMotionReceipt,
        )
        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_MINIMUM_ANCHOR,
        )
        from MemNavData.monocular_route_tangent_runtime import (
            build_route_compass,
            edge_receipt_sha256,
            historical_route_schedule,
            query_update_schedule,
        )
        from MemNavData.route_alignment_contract import (
            build_route_alignment_packet,
            route_alignment_executor_source,
            verify_route_alignment_packet,
        )

        diagnostics = {
            "local_tangent_requested": True,
            "local_tangent_status": "geometry_failure",
            "local_tangent_error_type": None,
            "local_tangent_error": None,
            "local_tangent_goal_start_frame": int(goal_start_frame),
            "local_tangent_target_anchor": int(target_anchor),
            "local_tangent_runtime_geometry": (
                "causal_rgb_height_scaled_adjacent_pnp_route"),
            "local_tangent_route_motion_model": (
                self.certified_route_motion_model),
            "local_tangent_historical_motion_model": (
                certified_route_edge_motion_model(
                    self.certified_route_motion_model,
                    "historical_adjacent_sample")),
            "local_tangent_live_query_motion_model": (
                certified_route_edge_motion_model(
                    self.certified_route_motion_model,
                    "live_query_adjacent_sample")),
            "local_tangent_query_motion_sampling": (
                "per_action_dense"
                if self.certified_route_motion_model ==
                "direct_pnp_dense_query"
                else "fixed_stride_sparse"
            ),
            "local_tangent_route_depth_cache_stride": int(
                self.certified_route_depth_cache_stride),
            "local_tangent_unfiltered_pnp_shadow_enabled": bool(
                self.certified_route_motion_unfiltered_shadow),
            "local_tangent_evaluator_pose_consumed": False,
            "local_tangent_habitat_path_consumed": False,
            "local_tangent_executor_odometry_consumed": False,
            "local_tangent_metric_depth_sensor_consumed": False,
            "local_tangent_camera_height_scale_consumed": True,
            "local_tangent_distance_regime_present": False,
            "local_tangent_visual_authority_gate_after_initialization": False,
            "local_tangent_stuck_trigger_present": False,
            "local_tangent_endpoint_fallback_available": False,
            "local_tangent_native_fallback_available": False,
            "local_tangent_tangent_baseline_m": 0.30,
            "local_tangent_controller_radius_m": 2.5,
            "local_tangent_alignment_packet": None,
            "local_tangent_alignment_packet_sha256": None,
            "local_tangent_alignment_control_edge_count": 0,
            "local_tangent_alignment_executor_receipt_consumed": False,
        }
        try:
            goal_start = int(goal_start_frame)
            target = int(target_anchor)
            current_frame = int(self.n - 1)
            if goal_start < 40 or goal_start > current_frame:
                raise ValueError(
                    "monocular route goal start is outside the metric stream")
            if target < CERTIFIED_MINIMUM_ANCHOR or target >= goal_start:
                raise ValueError(
                    "monocular route anchor violates the CEC causal boundary")
            if self.certified_relocalization_matcher is None:
                raise RuntimeError("monocular route matcher is unavailable")
            if self.certified_route_depth_cache_stride != 8:
                raise RuntimeError(
                    "monocular route tangent requires frozen depth stride 8")

            routes = getattr(
                self, "_certified_monocular_route_tangents", None)
            if routes is None:
                routes = {}
                self._certified_monocular_route_tangents = routes
            route = routes.get(goal_key)
            if route is None:
                schedule = historical_route_schedule(
                    target_anchor=target,
                    goal_start_frame=goal_start,
                    cached_depth_frames=(
                        self._certified_route_depth_cached_anchors),
                )
                terminal_motion, terminal_receipt = (
                    self._certified_terminal_route_motion(
                        target, goal_pose9))
                motions: list[PlanarMotionReceipt] = [terminal_motion]
                edge_receipts = [terminal_receipt]
                for specification in schedule["edges_chronological"]:
                    motion, receipt = self._certified_visual_route_edge(
                        reference_frame=specification[
                            "depth_reference_frame"],
                        query_frame=specification["match_query_frame"],
                        target_anchor=target,
                        invert_estimate=bool(
                            specification["invert_estimate"]),
                        edge_kind=specification["edge_kind"],
                    )
                    motions.append(motion)
                    edge_receipts.append(receipt)
                compass = build_route_compass(
                    motions,
                    tangent_baseline_m=0.30,
                    controller_radius_m=2.5,
                )
                # Bind the exact first query observation as the reference for
                # the next visual-motion interval, even when it is not a
                # fixed-stride history cache frame.
                _depth, _confidence, current_depth_source = (
                    self._materialize_current_route_depth())
                route = {
                    "compass": compass,
                    "goal_start_frame": goal_start,
                    "target_anchor": target,
                    "last_consumed_frame": current_frame,
                    "history_schedule": schedule,
                    "history_edge_receipt_sha256": edge_receipt_sha256(
                        edge_receipts),
                    "history_edge_count": len(edge_receipts),
                    "history_node_count": len(
                        schedule["nodes_chronological"]) + 1,
                    "query_edge_count": 0,
                    "alignment_control_edge_count": 0,
                    "alignment_executor_receipt_consumed": False,
                    "alignment_packet": None,
                    "initial_current_depth_source": current_depth_source,
                }
                readout = compass.advance_local_se2(
                    executed_forward_m=0.0,
                    executed_left_m=0.0,
                    executed_yaw_rad=0.0,
                )
                if authority_proof is not None:
                    if goal_image_sha256 is None:
                        raise RuntimeError(
                            "route alignment lacks the goal-image binding")
                    anchor_record = self._certified_anchor_image_record(target)
                    route["alignment_packet"] = build_route_alignment_packet(
                        authority_proof=authority_proof,
                        current_frame=current_frame,
                        goal_start_frame=goal_start,
                        target_anchor=target,
                        current_rgb_sha256=self._last_frame_jpg_sha256,
                        goal_image_sha256=goal_image_sha256,
                        anchor_image_sha256=anchor_record["sha256"],
                        history_edge_receipt_sha256=route[
                            "history_edge_receipt_sha256"],
                        unit_bearing=readout.unit_bearing,
                        tangent_baseline_m=0.30,
                        controller_radius_m=2.5,
                    )
                routes[goal_key] = route
                update_receipts = []
            else:
                if (int(route["goal_start_frame"]) != goal_start
                        or int(route["target_anchor"]) != target):
                    raise RuntimeError(
                        "frozen monocular route contract changed")
                previous_frame = int(route["last_consumed_frame"])
                update_receipts = []
                readout = None
                executor_receipts = getattr(
                    self, "executor_motion_receipts", [])
                if executor_receipts and len(executor_receipts) != self.n:
                    raise RuntimeError(
                        "route executor receipt/frame binding changed")
                pending_executor = (
                    executor_receipts[previous_frame + 1:current_frame + 1]
                    if executor_receipts else [])
                source_prefix = route_alignment_executor_source(
                    "0" * 64)[:-64]
                control_flags = [
                    isinstance(receipt, dict)
                    and isinstance(receipt.get("local_se2"), dict)
                    and str(receipt["local_se2"].get("source", "")).startswith(
                        source_prefix)
                    for receipt in pending_executor
                ]
                if any(control_flags):
                    if (not pending_executor or not all(control_flags)
                            or route.get("alignment_packet") is None):
                        raise RuntimeError(
                            "atomic route alignment was mixed with visual motion")
                    packet = verify_route_alignment_packet(
                        route["alignment_packet"])
                    expected_source = route_alignment_executor_source(
                        packet["packet_sha256"])
                    total_yaw = 0.0
                    for frame_index, receipt in enumerate(
                            pending_executor, start=previous_frame + 1):
                        local = receipt["local_se2"]
                        translation = float(receipt[
                            "executed_translation_m"])
                        forward = float(local["executed_forward_m"])
                        left = float(local["executed_left_m"])
                        yaw = float(local["executed_yaw_rad"])
                        if (int(receipt["frame_index"]) != frame_index
                                or local.get("source") != expected_source
                                or abs(translation) > 1e-12
                                or abs(forward) > 1e-12
                                or abs(left) > 1e-12
                                or not math.isfinite(yaw)
                                or abs(yaw) > math.radians(30.0) + 1e-9):
                            raise RuntimeError(
                                "route alignment control edge is invalid")
                        readout = route["compass"].advance_local_se2(
                            executed_forward_m=0.0,
                            executed_left_m=0.0,
                            executed_yaw_rad=yaw,
                        )
                        total_yaw += yaw
                        update_receipts.append({
                            "edge_kind": "proof_bound_atomic_yaw_control",
                            "frame_index": frame_index,
                            "packet_sha256": packet["packet_sha256"],
                            "executed_yaw_rad": yaw,
                            "translation_m": 0.0,
                        })
                    turn_error = math.atan2(
                        math.sin(total_yaw - float(
                            packet["required_turn_rad"])),
                        math.cos(total_yaw - float(
                            packet["required_turn_rad"])),
                    )
                    if abs(turn_error) > 1e-8:
                        raise RuntimeError(
                            "atomic route alignment ended before its proof-"
                            "bound turn was complete")
                    route["alignment_control_edge_count"] += len(
                        pending_executor)
                    route["alignment_executor_receipt_consumed"] = True
                else:
                    if any(receipt is not None for receipt in pending_executor):
                        raise RuntimeError(
                            "monocular route received unauthorized executor "
                            "odometry")
                    endpoints = query_update_schedule(
                        previous_frame=previous_frame,
                        current_frame=current_frame,
                        cached_depth_frames=(set(
                            self._certified_route_depth_cached_anchors).union(
                                self._certified_route_live_depth_cache)),
                    )
                    reference = previous_frame
                    for endpoint in endpoints:
                        motion, receipt = self._certified_visual_route_edge(
                            reference_frame=reference,
                            query_frame=int(endpoint),
                            target_anchor=target,
                            edge_kind="live_query_adjacent_sample",
                        )
                        readout = route["compass"].advance_local_se2(
                            executed_forward_m=motion.forward_m,
                            executed_left_m=motion.left_m,
                            executed_yaw_rad=motion.yaw_rad,
                        )
                        update_receipts.append(receipt)
                        reference = int(endpoint)
                if readout is None:
                    readout = route["compass"].advance_local_se2(
                        executed_forward_m=0.0,
                        executed_left_m=0.0,
                        executed_yaw_rad=0.0,
                    )
                _depth, _confidence, _source = (
                    self._materialize_current_route_depth())
                self._certified_route_live_depth_cache.clear()
                route["last_consumed_frame"] = current_frame
                route["query_edge_count"] += sum(
                    receipt.get("edge_kind") != (
                        "proof_bound_atomic_yaw_control")
                    for receipt in update_receipts)

            diagnostics.update(
                readout.audit_dict(),
                local_tangent_status=(
                    "terminal_tangent"
                    if readout.terminal_tangent_held else "active"),
                local_tangent_schema_version=readout.schema_version,
                local_tangent_projected_progress_m=float(
                    readout.projected_progress_m),
                local_tangent_progress_fraction=float(
                    readout.route_progress_fraction),
                local_tangent_route_state_index=int(
                    readout.route_state_index),
                local_tangent_reference_arc_m=float(
                    readout.reference_arc_m),
                local_tangent_query_path_length_m=float(
                    readout.query_path_length_m),
                local_tangent_cross_track_error_m=float(
                    readout.cross_track_error_m),
                local_tangent_estimated_position=[
                    float(value) for value in readout.estimated_position],
                local_tangent_estimated_yaw_rad=float(
                    readout.estimated_yaw_rad),
                local_tangent_terminal_tangent_held=bool(
                    readout.terminal_tangent_held),
                local_tangent_unit_bearing=[
                    float(value) for value in readout.unit_bearing],
                local_tangent_controller_pointgoal=[
                    float(value) for value in readout.controller_pointgoal],
                local_tangent_route_extent_m=float(
                    route["compass"].route_extent_m),
                local_tangent_history_edge_receipt_sha256=route[
                    "history_edge_receipt_sha256"],
                local_tangent_history_edge_count=int(
                    route["history_edge_count"]),
                local_tangent_history_node_count=int(
                    route["history_node_count"]),
                local_tangent_history_nodes=list(route[
                    "history_schedule"]["nodes_chronological"]),
                local_tangent_pre_metric_anchor_bridge=bool(route[
                    "history_schedule"]["pre_metric_anchor_bridge"]),
                local_tangent_query_edge_count=int(
                    route["query_edge_count"]),
                local_tangent_alignment_packet=route.get(
                    "alignment_packet"),
                local_tangent_alignment_packet_sha256=(
                    None if route.get("alignment_packet") is None
                    else route["alignment_packet"]["packet_sha256"]),
                local_tangent_alignment_control_edge_count=int(route[
                    "alignment_control_edge_count"]),
                local_tangent_alignment_executor_receipt_consumed=bool(
                    route["alignment_executor_receipt_consumed"]),
                local_tangent_last_consumed_frame=int(
                    route["last_consumed_frame"]),
                local_tangent_update_edge_count=len(update_receipts),
                local_tangent_update_edge_receipts=update_receipts,
            )
            return [float(value) for value in readout.unit_bearing], diagnostics
        except Exception as error:
            diagnostics.update(
                local_tangent_error_type=type(error).__name__,
                local_tangent_error=str(error),
                local_tangent_failure_edge_diagnostic=getattr(
                    error, "route_motion_diagnostic", None),
            )
            return None, diagnostics

    @torch.no_grad()
    def certified_path_field_reanchor(self, target_goal_jpg_bytes):
        """Update one authorized route from a local visual pose witness.

        CEC proves the target once and freezes one tail-to-anchor route.  Every
        later call retrieves only among the not-yet-visited addresses of that
        same temporal route, recovers the current camera pose with the existing
        LightGlue/LingBot-depth PnP stack, and advances one monotone route
        coordinate. The output remains a 2.5 m arc-ahead bearing. PnP is a
        state observation here, not a second control-authority gate. A missing
        pose is a geometry-stream failure; this method never returns endpoint
        guidance or native fallback.
        """

        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_CANDIDATE_MIN_GAP,
            CERTIFIED_CANDIDATE_TOP_K,
            CERTIFIED_MINIMUM_ANCHOR,
            UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
        )

        started = time.perf_counter()
        target_key = hashlib.md5(target_goal_jpg_bytes).hexdigest()
        base = {
            "schema_version": "certified_route_coordinate_v2_20260902",
            "ok": True,
            "target_goal_key": target_key,
            "runtime_role_visible": False,
            "runtime_habitat_pose_visible": False,
            "endpoint_fallback_available": False,
            "native_fallback_available": False,
            "distance_gate_present": False,
            "local_observation_authority": False,
            "state_updated": False,
            "direction_vector": None,
            "aux_pose": None,
        }
        route = self._certified_path_field_routes.get(target_key)
        target_cache = self._certified_relocalization_cache.get(target_key)
        if route is None or target_cache is None:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "authorized_target_route_missing",
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }
        target_result = target_cache.get("result", {})
        if target_result.get("accepted") is not True:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "target_route_not_certificate_authorized",
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }
        path = route["path"]
        progress = float(route["progress_m"])
        source_indices = path.source_indices_at_or_after_progress(
            minimum_progress_m=progress)
        goal_start = int(route["goal_start_frame"])
        eligible_indices = tuple(
            index for index in source_indices
            if (CERTIFIED_MINIMUM_ANCHOR <= index < goal_start
                and index < len(self.dino_cls)
                and index in self._certified_route_depth_cached_anchors)
        )
        if not eligible_indices:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "route_control_window_has_no_visual_anchor",
                "path_progress_m": progress,
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }

        current_frame = int(self.n - 1)
        current_path = os.path.join(self.rgb_dir, f"{current_frame}.jpg")
        if not os.path.isfile(current_path):
            return {
                **base,
                "status": "geometry_failure",
                "reason": "current_rgb_missing",
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }
        with open(current_path, "rb") as stream:
            current_bytes = stream.read()
        if not current_bytes or len(self.dino_cls) != self.n:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "current_visual_state_incomplete",
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }

        current_cls = self.dino_cls[-1].to(self.device)
        memory_cls = torch.stack([
            self.dino_cls[index] for index in eligible_indices
        ], dim=0).to(self.device)
        cosine = torch.nn.functional.cosine_similarity(
            current_cls.expand(memory_cls.shape[0], -1),
            memory_cls,
            dim=-1,
        ).detach().float().cpu().tolist()
        score_by_index = dict(zip(eligible_indices, cosine))
        all_scores = [float("-inf")] * self.n
        eligible_mask = [False] * self.n
        for index, score in score_by_index.items():
            all_scores[index] = float(score)
            eligible_mask[index] = True
        candidates = temporal_nms_candidates(
            all_scores,
            eligible_mask,
            top_k=CERTIFIED_CANDIDATE_TOP_K,
            min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP,
        )
        if not candidates:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "route_visual_proposal_empty",
                "path_progress_m": progress,
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }

        local_key = hashlib.sha256(
            b"route-coordinate\0"
            + target_key.encode("ascii") + b"\0"
            + str(current_frame).encode("ascii") + b"\0"
            + current_bytes
        ).hexdigest()
        self._goal_start_frame[local_key] = goal_start
        try:
            witness = self.certified_relocalize(
                current_bytes,
                candidates,
                proposal_order="geometry_first",
                authority_policy=UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
                guidance_mode="endpoint_bearing",
                goal_key_override=local_key,
                reference_depth_source="route_sparse",
            )
        finally:
            # Local observations are frame-bound.  Keep immutable anchor-depth
            # arrays, but discard query-conditioned caches after each update.
            self._clear_goal_conditioned_state(local_key)
        pose9 = witness.get("pnp", {}).get("pose9")
        if witness.get("accepted") is not True or pose9 is None:
            return {
                **base,
                "status": "geometry_failure",
                "reason": "local_pnp_pose_unavailable",
                "local_witness": {
                    "reason": witness.get("reason"),
                    "selected_anchor": witness.get("selected_anchor"),
                    "candidate_count": len(candidates),
                    "current_frame": current_frame,
                    "eligible_route_anchor_count": len(eligible_indices),
                    "path_progress_m": progress,
                    "candidates": [dict(item) for item in candidates],
                    "proposal_attempts": witness.get("proposal_attempts"),
                },
                "path_progress_m": progress,
                "runtime_ms": 1000.0 * (time.perf_counter() - started),
            }

        direction, diagnostics = self._certified_path_field_direction(
            goal_key=target_key,
            target_anchor=int(route["target_anchor"]),
            goal_start_frame=goal_start,
            goal_pose9=target_cache["goal_pose9"],
            current_pose9_override=pose9,
            route_progress_hint_m=path.progress_for_source_index(
                int(witness["selected_anchor"])),
        )
        status = diagnostics.get("episodic_path_field_status")
        state_updated = status in ("active", "complete")
        return {
            **base,
            **diagnostics,
            "status": status,
            "reason": (
                "route_coordinate_updated"
                if state_updated else "path_field_geometry_failure"),
            "state_updated": state_updated,
            "direction_vector": direction,
            "aux_pose": direction,
            "pointgoal_units": "lingbot_raw_direction_only",
            "local_witness": {
                "selected_anchor": witness.get("selected_anchor"),
                "candidate_count": len(candidates),
                "pnp_status": witness.get("pnp", {}).get("status"),
                "strict_certificate": witness.get("certificate"),
                "state_observation_only": True,
            },
            "runtime_ms": 1000.0 * (time.perf_counter() - started),
        }

    @torch.no_grad()
    def _certified_graph_direction(
            self, *, goal_key, direct_bearing, target_anchor,
            goal_start_frame, route_start_anchor=None,
            graph_rescue=False):
        """Optionally replace a stalled direct chord by a history-graph node.

        This path is deliberately dormant until ``graph_rescue`` is requested
        by the caller's progress monitor.  It uses only causal LingBot poses
        and an explicitly supplied previous-goal anchor; Habitat state never
        enters the server.  If any route contract is invalid, execution fails
        closed to the already-certified direct bearing.
        """
        direct = [float(value) for value in direct_bearing]
        graph_spacing_m = float(getattr(
            self, "graph_subgoal_spacing_m", 0.0))
        diagnostics = {
            "certified_graph_enabled": graph_spacing_m > 0.0,
            "certified_graph_rescue_requested": bool(graph_rescue),
            "certified_graph_rescue_active": False,
            "certified_graph_route_start_contract": (
                int(route_start_anchor)
                if route_start_anchor is not None else None),
            "certified_graph_route_start_node": None,
            "certified_graph_target_anchor": int(target_anchor),
            "certified_graph_temporal_direction": None,
            "certified_graph_node": None,
            "certified_graph_cursor": None,
            "certified_graph_count": 0,
            "certified_graph_complete": False,
            "certified_graph_reason": "direct_bearing",
        }
        if not graph_rescue:
            return direct, diagnostics
        if graph_spacing_m <= 0.0:
            diagnostics["certified_graph_reason"] = (
                "graph_rescue_requested_but_disabled")
            return direct, diagnostics
        if not self.cam_pose:
            diagnostics["certified_graph_reason"] = "no_lingbot_pose"
            return direct, diagnostics

        target = int(target_anchor)
        ceiling = int(goal_start_frame) - 1
        start_contract = (
            ceiling if route_start_anchor is None
            else int(route_start_anchor))
        if (target < 0 or target > ceiling
                or start_contract < 0 or start_contract > ceiling):
            diagnostics["certified_graph_reason"] = "invalid_route_contract"
            return direct, diagnostics

        route_record = self._certified_graph_routes.get(goal_key)
        if route_record is None:
            translations = np.stack([
                pose.float().cpu().numpy()[:3]
                for pose in self.cam_pose[:goal_start_frame]
            ], axis=0)
            lo, hi = sorted((start_contract, target))
            current = self.cam_pose[-1].float().cpu().numpy()[:3]
            planar = translations[lo:hi + 1][:, (0, 2)]
            nearest = lo + int(np.argmin(np.linalg.norm(
                planar - current[[0, 2]], axis=1)))
            metric_scale_raw = self._get_metric_scale_preserving_stream()
            if metric_scale_raw is None:
                diagnostics["certified_graph_reason"] = (
                    "metric_scale_unavailable")
                return direct, diagnostics
            metric_scale = float(metric_scale_raw)
            if not np.isfinite(metric_scale) or metric_scale <= 0.0:
                diagnostics["certified_graph_reason"] = (
                    "metric_scale_invalid")
                return direct, diagnostics
            nodes = metric_nodes_between(
                translations,
                start_index=nearest,
                target_index=target,
                metric_scale=metric_scale,
                spacing_m=graph_spacing_m,
            )
            route_record = {
                "route_start_contract": start_contract,
                "route_start_node": nearest,
                "target_anchor": target,
                "metric_scale": metric_scale,
                "progress": ReverseRouteProgress(
                    anchor_index=target,
                    start_index=nearest,
                    nodes=nodes,
                ),
            }
            self._certified_graph_routes[goal_key] = route_record
        elif (int(route_record["route_start_contract"]) != start_contract
              or int(route_record["target_anchor"]) != target):
            diagnostics["certified_graph_reason"] = "route_contract_changed"
            return direct, diagnostics

        route = route_record["progress"]
        metric_scale = torch.tensor(
            [float(route_record["metric_scale"])], device=self.device,
            dtype=torch.float32)
        current_pose = self.cam_pose[-1][None]
        selected = None
        while not route.complete:
            node = int(route.current_node)
            node_pose = self.cam_pose[node][None]
            _, node_aux, _ = self.core.build_revisit(
                current_pose.to(self.device), node_pose.to(self.device),
                metric_scale)
            distance_m = float(torch.linalg.vector_norm(node_aux[0]).item())
            if route.accept_distance(
                    distance_m, self.graph_subgoal_arrival_m):
                continue
            selected = node_aux[0].detach().float().cpu().tolist()
            break

        start_node = int(route_record["route_start_node"])
        diagnostics.update(
            # A graph request can collapse to an empty route when the
            # current LingBot pose is already nearest to the target anchor.
            # In that case the returned command is the direct bearing, so do
            # not count it as an executed historical-subgoal intervention.
            certified_graph_rescue_active=not route.complete,
            certified_graph_route_start_contract=start_contract,
            certified_graph_route_start_node=start_node,
            certified_graph_temporal_direction=(
                "forward" if target > start_node
                else "reverse" if target < start_node else "same"),
            certified_graph_node=route.current_node,
            certified_graph_cursor=int(route.cursor),
            certified_graph_count=len(route.nodes),
            certified_graph_complete=route.complete,
            certified_graph_reason=(
                "route_complete_direct_bearing" if route.complete
                else "historical_subgoal"),
        )
        return (direct if route.complete else selected), diagnostics

    @torch.no_grad()
    def _cdec_pairwise_proposal(self, goal_path, canonical):
        """Rank a frozen shortlist and expose a non-authorizing posterior."""
        import time
        from pathlib import Path

        from MemNavData.cdec_pairwise_runtime import (
            pad_dino_image_batch,
            pool_dino_patch_tokens,
        )
        from MemNavData.certified_relocalization_runtime import (
            scale_free_relative_xy,
        )

        ranker = getattr(self, "cdec_pairwise_ranker", None)
        if ranker is None:
            raise RuntimeError("CDEC pairwise ranker is disabled")
        if not canonical:
            raise ValueError("CDEC proposal requires a non-empty shortlist")
        started = time.perf_counter()
        paths = [Path(goal_path)] + [
            Path(self.rgb_dir) / f"{int(item['anchor'])}.jpg"
            for item in canonical
        ]
        if not all(path.is_file() for path in paths):
            missing = next(path for path in paths if not path.is_file())
            raise FileNotFoundError(missing)
        images = self.lb.load_images([str(path) for path in paths])
        padded_images, real_count = pad_dino_image_batch(images)
        encoded = self.lb.dino(padded_images)
        pooled = pool_dino_patch_tokens(encoded["patch"][:real_count])
        current = self.cam_pose[-1].float().cpu().numpy()
        anchors = [int(item["anchor"]) for item in canonical]
        bearing_vectors = [
            scale_free_relative_xy(
                current, self.cam_pose[anchor].float().cpu().numpy())
            for anchor in anchors
        ]
        posterior_available = all(
            np.linalg.norm(np.asarray(vector, dtype=np.float64)) > 1e-12
            for vector in bearing_vectors)
        result = ranker.rank_pooled_tokens(
            pooled[0], pooled[1:],
            [float(item["score"]) for item in canonical], anchors,
            bearing_vectors=(bearing_vectors if posterior_available else None))
        result["direction_posterior_status"] = (
            "available" if posterior_available
            else "diagnostic_unavailable_zero_anchor_bearing")
        result["proposal_ms"] = 1000.0 * (time.perf_counter() - started)
        return result

    def certified_relocalize(
            self, goal_jpg_bytes, candidates, *, route_start_anchor=None,
            graph_rescue=False, allow_learned_rescue=False,
            proposal_order="geometry_first", goal_camera_intrinsic=None,
            authority_policy="strict_certificate",
            guidance_mode="endpoint_bearing", goal_key_override=None,
            reference_depth_source="canonical"):
        """Rank/localize/certify once; update only scale-free bearing later."""
        import hashlib
        import time
        from pathlib import Path

        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_CANDIDATE_TOP_K,
            CERTIFIED_EPIPOLAR_THRESHOLD_PX,
            CERTIFIED_MINIMUM_ANCHOR,
            STRICT_AUTHORITY_POLICY,
            SUPPORTED_AUTHORITY_POLICIES,
            UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
            fundamental_can_reach_certificate,
            fundamental_support,
            operational_authority_decision,
            rank_candidates,
            runtime_contract,
        )
        from MemNavData.lingbot_pnp_localization import (
            SiftPnPConfig,
            correspondence_pnp_localize,
            jsonable_pnp,
            map_raw_intrinsic_to_lingbot_pad,
        )

        started = time.perf_counter()
        frame_idx = self.n - 1
        proposal_order = str(proposal_order)
        authority_policy = str(authority_policy)
        guidance_mode = str(guidance_mode)
        reference_depth_source = str(reference_depth_source)
        if goal_camera_intrinsic is None:
            raw_goal_intrinsic = None
        else:
            try:
                raw_goal_intrinsic = np.asarray(
                    goal_camera_intrinsic, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raw_goal_intrinsic = np.empty((0, 0), dtype=np.float64)
            if (raw_goal_intrinsic.shape != (3, 3)
                    or not np.isfinite(raw_goal_intrinsic).all()
                    or raw_goal_intrinsic[0, 0] <= 0.0
                    or raw_goal_intrinsic[1, 1] <= 0.0):
                return {
                    "ok": False, "accepted": False,
                    "reason": "invalid_goal_camera_intrinsic",
                    "cached": False,
                    "relocalization_ms": 1000.0 * (
                        time.perf_counter() - started),
                }
        base = {
            "certified_relocalization_schema_version": (
                runtime_contract()["schema_version"]),
            "certified_relocalization_contract": runtime_contract(),
            "frame_idx": frame_idx,
            "aux_pose": None,
            "learned_rescue_requested": bool(allow_learned_rescue),
            "learned_rescue_available": (
                getattr(self, "cdec_pairwise_ranker", None) is not None),
            "proposal_order": proposal_order,
            "authority_policy": authority_policy,
            "guidance_mode": guidance_mode,
            "reference_depth_source": reference_depth_source,
            "goal_camera_calibration": (
                "explicit_distinct_intrinsic"
                if raw_goal_intrinsic is not None
                else "legacy_shared_history_intrinsic"
            ),
        }
        if proposal_order not in (
                "geometry_first", "dino_first_certified"):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_proposal_order", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if guidance_mode not in CERTIFIED_GUIDANCE_MODES:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_guidance_mode", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if reference_depth_source not in ("canonical", "route_sparse"):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_reference_depth_source", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (guidance_mode in (
                "episodic_path_field", "action_coordinate_compass",
                "se2_route_compass", "monocular_route_tangent")
                and (graph_rescue or route_start_anchor is not None)):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "path_field_incompatible_with_graph_rescue",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if authority_policy not in SUPPORTED_AUTHORITY_POLICIES:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_authority_policy", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (authority_policy == UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY
                and (proposal_order != "geometry_first"
                     or allow_learned_rescue)):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_authority_ablation_configuration",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (proposal_order == "dino_first_certified"
                and allow_learned_rescue):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "learned_rescue_incompatible_with_proposal_order",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if self.certified_relocalization_matcher is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "certified_relocalizer_disabled",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        goal_key = (
            hashlib.md5(goal_jpg_bytes).hexdigest()
            if goal_key_override is None else str(goal_key_override)
        )
        if (len(goal_key) != 64 and goal_key_override is not None) or any(
                character not in "0123456789abcdef" for character in goal_key):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_internal_goal_key", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        goal_start = self._goal_start_frame.get(goal_key)
        if goal_start is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "goal_not_probed_causally", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        try:
            if not isinstance(candidates, list):
                raise ValueError("candidates must be a list")
            if not 0 <= len(candidates) <= CERTIFIED_CANDIDATE_TOP_K:
                raise ValueError("candidate count outside frozen top-k contract")
            canonical = []
            seen = set()
            for dino_rank, item in enumerate(candidates, start=1):
                if not isinstance(item, dict):
                    raise ValueError("candidate is not an object")
                anchor = int(item["anchor"])
                score = float(item["score"])
                if (anchor in seen or anchor < CERTIFIED_MINIMUM_ANCHOR
                        or anchor >= int(goal_start)
                        or not np.isfinite(score)):
                    raise ValueError("candidate violates causal shortlist")
                seen.add(anchor)
                canonical.append({
                    "anchor": anchor,
                    "score": score,
                    "dino_rank": dino_rank,
                })
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_candidate_contract",
                "error": f"{type(error).__name__}: {error}",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        fingerprint = (
            ("learned_rescue_requested", bool(allow_learned_rescue)),
            ("proposal_order", proposal_order),
            ("authority_policy", authority_policy),
            ("reference_depth_source", reference_depth_source),
            ("goal_camera_intrinsic", (
                None if raw_goal_intrinsic is None
                else tuple(float(value) for value in raw_goal_intrinsic.flat)
            )),
            *tuple(
                (item["anchor"], float(item["score"]))
                for item in canonical
            ),
        )
        cached = self._certified_relocalization_cache.get(goal_key)
        if cached is not None:
            if cached["candidate_fingerprint"] != fingerprint:
                return {
                    **base, "ok": False, "accepted": False,
                    "reason": "candidate_contract_changed",
                    "cached": True,
                    "relocalization_ms": 1000.0 * (
                        time.perf_counter() - started),
                }
            result = dict(cached["result"])
            result.update(base, cached=True)
            if result.get("accepted"):
                direct_bearing = self._certified_bearing_vector(
                    cached["goal_pose9"])
                view_alignment = self._certified_view_alignment(
                    cached["goal_pose9"])
                if guidance_mode == "episodic_path_field":
                    tracked = self.certified_path_field_reanchor(
                        goal_jpg_bytes)
                    bearing_vector = tracked.get("direction_vector")
                    guidance_diagnostics = {
                        key: value for key, value in tracked.items()
                        if (key.startswith("episodic_path_field_")
                            or key.startswith("path_"))
                    }
                    guidance_diagnostics.update(
                        route_coordinate_state_updated=bool(
                            tracked.get("state_updated")),
                        route_coordinate_reason=tracked.get("reason"),
                        route_coordinate_local_witness=tracked.get(
                            "local_witness"),
                        route_coordinate_endpoint_fallback_available=(
                            tracked.get("endpoint_fallback_available")),
                        route_coordinate_native_fallback_available=(
                            tracked.get("native_fallback_available")),
                        route_coordinate_distance_gate_present=tracked.get(
                            "distance_gate_present"),
                    )
                    if tracked.get("state_updated") is not True:
                        guidance_diagnostics.update(
                            episodic_path_field_status="geometry_failure",
                            episodic_path_field_error_type=(
                                "RouteCoordinateObservationError"),
                            episodic_path_field_error=tracked.get("reason"),
                        )
                elif guidance_mode == "action_coordinate_compass":
                    bearing_vector, guidance_diagnostics = (
                        self._certified_action_coordinate_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                        ))
                elif guidance_mode == "se2_route_compass":
                    bearing_vector, guidance_diagnostics = (
                        self._certified_se2_route_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                        ))
                elif guidance_mode == "monocular_route_tangent":
                    bearing_vector, guidance_diagnostics = (
                        self._certified_monocular_route_tangent_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                            goal_pose9=np.asarray(
                                cached["goal_pose9"], dtype=np.float64),
                            authority_proof={
                                "ok": result.get("ok"),
                                "accepted": result.get("accepted"),
                                "reason": result.get("reason"),
                                "selected_anchor": result.get(
                                    "selected_anchor"),
                                "selected_anchor_image_sha256": result.get(
                                    "selected_anchor_image_sha256"),
                                "certificate": result.get("certificate"),
                                "authority": result.get("authority"),
                                "pnp": result.get("pnp"),
                            },
                            goal_image_sha256=hashlib.sha256(
                                goal_jpg_bytes).hexdigest(),
                        ))
                else:
                    bearing_vector, guidance_diagnostics = (
                        self._certified_graph_direction(
                            goal_key=goal_key,
                            direct_bearing=direct_bearing,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                            route_start_anchor=route_start_anchor,
                            graph_rescue=graph_rescue,
                        ))
                result.update(
                    aux_pose=bearing_vector,
                    direction_vector=bearing_vector,
                    pointgoal_units="lingbot_raw_direction_only",
                    metric_scale=None,
                    **view_alignment,
                    **guidance_diagnostics,
                )
            result["relocalization_ms"] = 1000.0 * (
                time.perf_counter() - started)
            return result

        if not canonical:
            uncached_ms = 1000.0 * (time.perf_counter() - started)
            result = {
                **base,
                "ok": True,
                "accepted": False,
                "reason": "no_causal_candidate",
                "authority": None,
                "certificate": None,
                "selected_anchor": None,
                "selected_dino_rank": None,
                "candidate_count": 0,
                "ranked_candidates": [],
                "pnp": {"status": "no_causal_candidate"},
                "cached": False,
                "uncached_relocalization_ms": uncached_ms,
                "relocalization_ms": uncached_ms,
            }
            cache_result = dict(result)
            cache_result.pop("frame_idx", None)
            self._certified_relocalization_cache[goal_key] = {
                "candidate_fingerprint": fingerprint,
                "goal_pose9": None,
                "result": cache_result,
            }
            return result

        goal_path = Path(self.rgb_dir) / f"_cert_goal_{goal_key}.jpg"
        goal_path.write_bytes(goal_jpg_bytes)
        evidence = []
        matched_by_anchor = {}
        for item in canonical:
            anchor = item["anchor"]
            try:
                matched = self.certified_relocalization_matcher.match_paths(
                    Path(self.rgb_dir) / f"{anchor}.jpg", goal_path,
                    target_height=518, target_width=518,
                    patch_size=int(self.lb.patch_size))
                support = fundamental_support(
                    matched["reference_raw_points"],
                    matched["query_raw_points"], matched["scores"],
                    tuple(matched["reference_raw_hw"]),
                    tuple(matched["query_raw_hw"]),
                    threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
                matched_by_anchor[anchor] = matched
                error = None
            except Exception as exception:  # one bad image must fail closed
                support = {
                    "lightglue_matches": 0,
                    "lightglue_score_median": 0.0,
                    "fundamental_inliers": 0,
                    "fundamental_inlier_ratio": 0.0,
                    "fundamental_query_grid_coverage": 0.0,
                    "fundamental_query_hull_coverage": 0.0,
                    "fundamental_reference_grid_coverage": 0.0,
                    "fundamental_reference_hull_coverage": 0.0,
                }
                error = f"{type(exception).__name__}: {exception}"
            evidence.append({
                **support,
                "anchor": anchor,
                "dino_cosine": item["score"],
                "dino_rank": item["dino_rank"],
                "error": error,
            })
        ranked = rank_candidates(evidence)
        evidence_by_anchor = {
            int(candidate["anchor"]): candidate for candidate in evidence}

        def attempt_proposal(selected, source):
            selected_anchor = int(selected["anchor"])
            if authority_policy == STRICT_AUTHORITY_POLICY:
                possible, precheck_reason = (
                    fundamental_can_reach_certificate(selected))
            else:
                # This is an algorithmic minimum for PnP, not an operational
                # certificate threshold.  The diagnostic arm still requires
                # local correspondences and a finite geometric pose witness.
                possible = bool(
                    int(selected.get("lightglue_matches", 0))
                    >= int(SiftPnPConfig().min_correspondences)
                    and selected_anchor in matched_by_anchor
                )
                precheck_reason = (
                    "pnp_attemptable"
                    if possible else "precheck_lightglue_matches"
                )
            pnp = {"status": precheck_reason}
            goal_pose9 = None
            reference_depth_cache = None
            if possible and selected_anchor in matched_by_anchor:
                try:
                    if reference_depth_source == "route_sparse":
                        depth, confidence = (
                            self._certified_route_reference_depth(
                                selected_anchor))
                    else:
                        depth, confidence = self._certified_reference_depth(
                            selected_anchor)
                    stats = getattr(
                        self, "_certified_dense_replay_last_stats", None)
                    reference_depth_cache = (
                        dict(stats) if stats is not None else {
                            "enabled": False,
                            "anchor": selected_anchor,
                            "cache_hit": False,
                            "cache_source": "legacy_full_replay",
                        })
                    matched = matched_by_anchor[selected_anchor]
                    reference_pose = (
                        self.cam_pose[selected_anchor].float().numpy())
                    query_intrinsic = None
                    if raw_goal_intrinsic is not None:
                        query_height, query_width = (
                            int(value) for value in matched["query_raw_hw"])
                        query_intrinsic = map_raw_intrinsic_to_lingbot_pad(
                            raw_goal_intrinsic,
                            raw_height=query_height,
                            raw_width=query_width,
                            target_height=int(depth.shape[-2]),
                            target_width=int(depth.shape[-1]),
                            patch_size=int(self.lb.patch_size),
                        )
                    pnp = correspondence_pnp_localize(
                        matched["reference_points"], matched["query_points"],
                        depth, confidence, reference_pose,
                        config=SiftPnPConfig(),
                        match_scores=matched["scores"],
                        epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX,
                        query_intrinsic=query_intrinsic)
                    pnp = jsonable_pnp(pnp)
                    if "pose9" in pnp:
                        candidate_pose9 = np.asarray(
                            pnp["pose9"], dtype=np.float64)
                        if (candidate_pose9.shape == (9,)
                                and np.isfinite(candidate_pose9).all()):
                            goal_pose9 = candidate_pose9
                except Exception as exception:
                    pnp = {
                        "status": "runtime_exception",
                        "error": f"{type(exception).__name__}: {exception}",
                    }
            authority = operational_authority_decision(
                pnp, policy=authority_policy)
            certificate = authority["strict_certificate"]
            accepted = bool(authority["accepted"])
            reason = (
                precheck_reason if not possible else authority["reason"])
            return {
                "source": source,
                "selected": selected,
                "selected_anchor": selected_anchor,
                "possible": possible,
                "precheck_reason": precheck_reason,
                "pnp": pnp,
                "certificate": certificate,
                "authority": authority,
                "accepted": accepted,
                "reason": reason,
                "goal_pose9": goal_pose9,
                "reference_depth_cache": reference_depth_cache,
            }

        def public_attempt(attempt):
            return {
                "source": attempt["source"],
                "selected_anchor": attempt["selected_anchor"],
                "selected_dino_rank": attempt["selected"].get("dino_rank"),
                "accepted": attempt["accepted"],
                "reason": attempt["reason"],
                "precheck_passed": attempt["possible"],
                "pnp_status": attempt["pnp"].get("status"),
                "certificate": attempt["certificate"],
                "authority": attempt["authority"],
                "reference_depth_cache": attempt[
                    "reference_depth_cache"],
            }

        geometry_attempt = None
        if proposal_order == "geometry_first":
            geometry_attempt = attempt_proposal(ranked[0], "geometry")
            final_attempt = geometry_attempt
            proposal_attempts = [public_attempt(geometry_attempt)]
        else:
            semantic_attempts = []
            accepted_semantic_attempt = None
            for canonical_item in canonical:
                semantic_attempt = attempt_proposal(
                    evidence_by_anchor[int(canonical_item["anchor"])],
                    "dino_first_certified",
                )
                semantic_attempts.append(semantic_attempt)
                if semantic_attempt["accepted"]:
                    accepted_semantic_attempt = semantic_attempt
                    break
            final_attempt = (
                accepted_semantic_attempt
                if accepted_semantic_attempt is not None
                else semantic_attempts[0]
            )
            proposal_attempts = [
                public_attempt(attempt) for attempt in semantic_attempts]
        counterfactual_audit = None
        counterfactual_dino_order_audit = None
        if (proposal_order == "geometry_first"
                and getattr(self, "certified_counterfactual_audit", False)):
            dino_order_attempts = []
            accepted_dino_order_attempt = None
            for canonical_item in canonical:
                dino_selected = evidence_by_anchor[
                    int(canonical_item["anchor"])]
                if (int(dino_selected["anchor"])
                        == int(geometry_attempt["selected_anchor"])):
                    public = {
                        **public_attempt(geometry_attempt),
                        "source": "dino_order_geometry_attempt_reuse",
                        "action_authority": False,
                    }
                else:
                    dino_attempt = attempt_proposal(
                        dino_selected, "dino_order_counterfactual")
                    public = {
                        **public_attempt(dino_attempt),
                        "action_authority": False,
                    }
                dino_order_attempts.append(public)
                if len(dino_order_attempts) == 1:
                    counterfactual_audit = {
                        **public,
                        "source": (
                            "dino_top1_same_anchor_reuse"
                            if public["source"]
                            == "dino_order_geometry_attempt_reuse"
                            else "dino_top1_counterfactual"),
                    }
                if public["accepted"]:
                    accepted_dino_order_attempt = public
                    break
            counterfactual_dino_order_audit = {
                "accepted": accepted_dino_order_attempt is not None,
                "selected_anchor": (
                    accepted_dino_order_attempt["selected_anchor"]
                    if accepted_dino_order_attempt is not None else None),
                "selected_dino_rank": (
                    accepted_dino_order_attempt["selected_dino_rank"]
                    if accepted_dino_order_attempt is not None else None),
                "attempt_count": len(dino_order_attempts),
                "attempts": dino_order_attempts,
                "action_authority": False,
            }
        learned_proposal = None
        ranker = getattr(self, "cdec_pairwise_ranker", None)
        if proposal_order != "geometry_first":
            learned_proposal = {
                "status": "not_applicable_semantic_first",
                "activation_authorized": False,
            }
        elif ranker is not None and not allow_learned_rescue:
            learned_proposal = {
                "status": "not_requested",
                "activation_authorized": False,
            }
        elif ranker is not None:
            if geometry_attempt["accepted"]:
                learned_proposal = {
                    "status": "not_evaluated_geometry_accepted",
                    "activation_authorized": False,
                }
            else:
                try:
                    learned_proposal = self._cdec_pairwise_proposal(
                        goal_path, canonical)
                    learned_anchor = int(
                        learned_proposal["selected_anchor"])
                    if learned_anchor == geometry_attempt["selected_anchor"]:
                        learned_proposal["status"] = (
                            "same_anchor_certificate_reused")
                        proposal_attempts.append({
                            **public_attempt(geometry_attempt),
                            "source": "learned_same_anchor_reuse",
                        })
                    else:
                        learned_selected = evidence_by_anchor.get(
                            learned_anchor)
                        if learned_selected is None:
                            raise RuntimeError(
                                "learned anchor escaped the frozen shortlist")
                        learned_attempt = attempt_proposal(
                            learned_selected, "learned_on_geometry_reject")
                        proposal_attempts.append(public_attempt(learned_attempt))
                        if learned_attempt["accepted"]:
                            final_attempt = learned_attempt
                        learned_proposal["status"] = (
                            "certificate_accepted"
                            if learned_attempt["accepted"]
                            else "certificate_rejected")
                except Exception as exception:
                    learned_proposal = {
                        "status": "runtime_exception_fail_closed",
                        "error": f"{type(exception).__name__}: {exception}",
                        "activation_authorized": False,
                    }

        selected = final_attempt["selected"]
        selected_anchor = final_attempt["selected_anchor"]
        possible = final_attempt["possible"]
        precheck_reason = final_attempt["precheck_reason"]
        pnp = final_attempt["pnp"]
        certificate = final_attempt["certificate"]
        accepted = final_attempt["accepted"]
        goal_pose9 = final_attempt["goal_pose9"]
        bearing_vector = None
        guidance_diagnostics = {}
        if accepted:
            direct_bearing = self._certified_bearing_vector(goal_pose9)
            view_alignment = self._certified_view_alignment(goal_pose9)
            if guidance_mode == "episodic_path_field":
                bearing_vector, guidance_diagnostics = (
                    self._certified_path_field_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        goal_pose9=goal_pose9,
                    ))
            elif guidance_mode == "action_coordinate_compass":
                bearing_vector, guidance_diagnostics = (
                    self._certified_action_coordinate_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                    ))
            elif guidance_mode == "se2_route_compass":
                bearing_vector, guidance_diagnostics = (
                    self._certified_se2_route_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                    ))
            elif guidance_mode == "monocular_route_tangent":
                route_anchor_record = self._certified_anchor_image_record(
                    selected_anchor)
                bearing_vector, guidance_diagnostics = (
                    self._certified_monocular_route_tangent_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        goal_pose9=goal_pose9,
                        authority_proof={
                            "ok": True,
                            "accepted": True,
                            "reason": final_attempt["reason"],
                            "selected_anchor": selected_anchor,
                            "selected_anchor_image_sha256": (
                                route_anchor_record["sha256"]),
                            "certificate": certificate,
                            "authority": final_attempt["authority"],
                            "pnp": pnp,
                        },
                        goal_image_sha256=hashlib.sha256(
                            goal_jpg_bytes).hexdigest(),
                    ))
            else:
                bearing_vector, guidance_diagnostics = (
                    self._certified_graph_direction(
                        goal_key=goal_key,
                        direct_bearing=direct_bearing,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        route_start_anchor=route_start_anchor,
                        graph_rescue=graph_rescue,
                    ))
        uncached_ms = 1000.0 * (time.perf_counter() - started)
        result = {
            **base,
            "ok": True,
            "accepted": accepted,
            "reason": final_attempt["reason"],
            "certificate": certificate,
            "authority": final_attempt["authority"],
            "selected_anchor": selected_anchor,
            "selected_dino_rank": selected.get("dino_rank"),
            "selected_proposal_source": final_attempt["source"],
            "proposal_attempts": proposal_attempts,
            "counterfactual_dino_top1_audit": counterfactual_audit,
            "counterfactual_dino_order_audit": (
                counterfactual_dino_order_audit),
            "learned_proposal": learned_proposal,
            "candidate_count": len(canonical),
            "ranked_candidates": ranked,
            "pnp": pnp,
            "reference_depth_cache": final_attempt[
                "reference_depth_cache"],
            "cached": False,
            "uncached_relocalization_ms": uncached_ms,
            "relocalization_ms": uncached_ms,
        }
        if accepted:
            anchor_record = self._certified_anchor_image_record(
                selected_anchor)
            result.update(
                aux_pose=bearing_vector,
                direction_vector=bearing_vector,
                pointgoal_units="lingbot_raw_direction_only",
                metric_scale=None,
                selected_anchor_image_sha256=anchor_record["sha256"],
                **view_alignment,
                **guidance_diagnostics,
            )
        cache_result = dict(result)
        cache_result.pop("frame_idx", None)
        for key in tuple(cache_result):
            if (key.startswith("episodic_path_field_")
                    or key.startswith("action_coordinate_")
                    or key.startswith("se2_route_")
                    or key.startswith("local_tangent_")):
                cache_result.pop(key)
        # Bearing is current-relative and must be recomputed after motion.
        cache_result["aux_pose"] = None
        self._certified_relocalization_cache[goal_key] = {
            "candidate_fingerprint": fingerprint,
            "goal_pose9": (goal_pose9.tolist() if accepted else None),
            "result": cache_result,
        }
        return result

    def learned_pi3x_relocalize(self, goal_jpg_bytes, candidates):
        """Run the frozen Pi3X proposal/proof and fail closed to native.

        The DINO top-8 is frozen on the first causal goal query.  Pi3X chooses
        and authorizes one anchor on that first call.  A first-call abstention
        is immutable for the goal; after an acceptance, later calls rerun only
        the selected anchor to update the direct current-to-goal bearing.
        """
        import hashlib
        import time
        from pathlib import Path

        from MemNavData.pi3x_online_relocalizer import (
            FROZEN_CANDIDATE_MIN_GAP,
            FROZEN_MINIMUM_ANCHOR,
            FROZEN_TOP_K,
        )

        started = time.perf_counter()
        frame_idx = int(self.n - 1)
        base = {
            "learned_pi3x_relocalization_schema_version": 1,
            "frame_idx": frame_idx,
            "aux_pose": None,
            "direction_vector": None,
            "pointgoal_units": None,
            "certificate_components_consumed": False,
            "simulator_pose_or_depth_consumed": False,
        }

        def elapsed_ms():
            return 1000.0 * (time.perf_counter() - started)

        runtime = getattr(self, "pi3x_online_relocalizer", None)
        if runtime is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "learned_pi3x_relocalizer_disabled",
                "cached": False, "relocalization_ms": elapsed_ms(),
            }
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        goal_start = self._goal_start_frame.get(goal_key)
        if goal_start is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "goal_not_probed_causally", "cached": False,
                "relocalization_ms": elapsed_ms(),
            }

        try:
            if not isinstance(candidates, list):
                raise ValueError("candidates must be a list")
            if not 0 <= len(candidates) <= FROZEN_TOP_K:
                raise ValueError("candidate count outside frozen top-k contract")
            canonical = []
            seen = set()
            for dino_rank, item in enumerate(candidates, start=1):
                if not isinstance(item, dict):
                    raise ValueError("candidate is not an object")
                anchor = int(item["anchor"])
                score = float(item["score"])
                if (anchor in seen or anchor < FROZEN_MINIMUM_ANCHOR
                        or anchor >= int(goal_start)
                        or not np.isfinite(score)):
                    raise ValueError("candidate violates causal shortlist")
                seen.add(anchor)
                canonical.append({
                    "anchor": anchor,
                    "score": score,
                    "dino_rank": dino_rank,
                })
            anchors = [item["anchor"] for item in canonical]
            if any(
                abs(left - right) < FROZEN_CANDIDATE_MIN_GAP
                for index, left in enumerate(anchors)
                for right in anchors[index + 1:]
            ):
                raise ValueError("candidate shortlist violates temporal gap")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_candidate_contract",
                "error": f"{type(error).__name__}: {error}",
                "cached": False, "relocalization_ms": elapsed_ms(),
            }

        fingerprint = tuple(
            (item["anchor"], float(item["score"]), item["dino_rank"])
            for item in canonical
        )
        cache = self._pi3x_relocalization_cache.get(goal_key)
        if cache is not None and cache["candidate_fingerprint"] != fingerprint:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "candidate_contract_changed", "cached": True,
                "initial_candidate_selection_cached": True,
                "relocalization_ms": elapsed_ms(),
            }
        if cache is not None and not cache["initial_accepted"]:
            result = dict(cache["result"])
            result.update(
                base, cached=True,
                initial_candidate_selection_cached=True,
                relocalization_ms=elapsed_ms(),
            )
            return result

        if not canonical:
            result = {
                **base, "ok": True, "accepted": False,
                "reason": "no_causal_candidate", "selected_anchor": None,
                "selected_dino_rank": None, "candidate_count": 0,
                "initial_candidate_count": 0, "ranked_candidates": [],
                "cached": False,
                "initial_candidate_selection_cached": False,
                "relocalization_ms": elapsed_ms(),
            }
            cached_result = dict(result)
            cached_result.pop("frame_idx", None)
            self._pi3x_relocalization_cache[goal_key] = {
                "candidate_fingerprint": fingerprint,
                "initial_accepted": False,
                "selected_candidate": None,
                "result": cached_result,
            }
            return result

        tracking = cache is not None
        runtime_candidates = (
            [dict(cache["selected_candidate"])] if tracking else canonical
        )
        goal_path = Path(self.rgb_dir) / f"_pi3x_goal_{goal_key}.jpg"
        if not goal_path.is_file():
            goal_path.write_bytes(goal_jpg_bytes)
        accepted_contract = False
        try:
            runtime_result = runtime.relocalize(
                rgb_dir=Path(self.rgb_dir),
                current_frame=frame_idx,
                candidates=runtime_candidates,
                goal_path=goal_path,
            )
            if not isinstance(runtime_result, dict):
                raise TypeError("runtime result must be an object")
            result = dict(runtime_result)
            accepted = bool(result.get("accepted", False))
            if accepted:
                anchor = int(result["selected_anchor"])
                rank = int(result["selected_dino_rank"])
                permitted = {
                    (item["anchor"], item["dino_rank"])
                    for item in runtime_candidates
                }
                vector = np.asarray(result.get("aux_pose"), dtype=np.float64)
                if ((anchor, rank) not in permitted
                        or result.get("pointgoal_units")
                        != "pi3x_current_camera_direction_only"
                        or vector.shape != (2,)
                        or not np.isfinite(vector).all()
                        or float(np.linalg.norm(vector)) <= 1e-12):
                    raise ValueError("accepted runtime result violates contract")
                accepted_contract = True
        except Exception as error:  # every learned-runtime failure is native
            result = {
                "ok": False, "accepted": False,
                "reason": "runtime_exception_fail_closed",
                "error": f"{type(error).__name__}: {error}",
                "selected_anchor": None, "selected_dino_rank": None,
                "candidate_count": len(runtime_candidates),
                "ranked_candidates": [],
            }

        result.update(
            base,
            cached=False,
            initial_candidate_selection_cached=tracking,
            initial_candidate_count=len(canonical),
            relocalization_ms=elapsed_ms(),
        )
        # ``base`` deliberately nulls action fields; restore them only after a
        # complete accepted-result contract check.
        if accepted_contract:
            result["aux_pose"] = list(runtime_result["aux_pose"])
            result["direction_vector"] = list(runtime_result["direction_vector"])
            result["pointgoal_units"] = runtime_result["pointgoal_units"]

        if not tracking:
            selected = None
            if result.get("accepted"):
                selected = next(
                    item for item in canonical
                    if (item["anchor"] == int(result["selected_anchor"])
                        and item["dino_rank"]
                        == int(result["selected_dino_rank"])))
            cached_result = dict(result)
            cached_result.pop("frame_idx", None)
            cached_result["aux_pose"] = None
            cached_result["direction_vector"] = None
            cached_result["pointgoal_units"] = None
            self._pi3x_relocalization_cache[goal_key] = {
                "candidate_fingerprint": fingerprint,
                "initial_accepted": bool(result.get("accepted")),
                "selected_candidate": (
                    None if selected is None else dict(selected)),
                "result": cached_result,
            }
        return result

    def _graph_conditioned_pose(
            self, *, goal_key, cache, current_pose,
            goal_aux_pose, anchor, goal_start_frame, metric_scale):
        """Replace a long direct point-goal with the next memory-graph node.

        The route is built once from the pre-goal pose chain and its cursor can
        only move toward the localized anchor.  Once every recorded node is
        reached, control returns to the image-conditioned goal pose for final
        alignment.  A zero spacing is an exact backward-compatible disable.
        """
        disabled = self.graph_subgoal_spacing_m <= 0.0
        diagnostics = dict(
            graph_subgoal_enabled=not disabled,
            graph_subgoal_node=None,
            graph_subgoal_cursor=None,
            graph_subgoal_count=0,
            graph_subgoal_complete=disabled,
            goal_aux_pose=goal_aux_pose[0].float().cpu().tolist(),
        )
        if disabled:
            return goal_aux_pose, diagnostics

        start_index = int(goal_start_frame) - 1
        anchor = int(anchor)
        route = self._graph_routes.get(goal_key)
        if (route is None or route.anchor_index != anchor
                or route.start_index != start_index):
            translations = (
                cache["cam_pose_enc"][:goal_start_frame, :3]
                .detach().float().cpu().numpy()
            )
            nodes = reverse_metric_nodes(
                translations,
                start_index=start_index,
                anchor_index=anchor,
                metric_scale=float(metric_scale.item()),
                spacing_m=self.graph_subgoal_spacing_m,
            )
            route = ReverseRouteProgress(
                anchor_index=anchor, start_index=start_index, nodes=nodes)
            self._graph_routes[goal_key] = route

        # A single replan can legitimately cross a very short residual node,
        # so consume all already-reached nodes before returning one target.
        selected = goal_aux_pose
        while not route.complete:
            node = int(route.current_node)
            node_pose = cache["cam_pose_enc"][node][None]
            _, node_aux, _ = self.core.build_revisit(
                current_pose.to(self.device), node_pose.to(self.device),
                metric_scale)
            distance_m = float(torch.linalg.vector_norm(node_aux[0]).item())
            if route.accept_distance(
                    distance_m, self.graph_subgoal_arrival_m):
                continue
            selected = node_aux
            break

        diagnostics.update(
            graph_subgoal_node=route.current_node,
            graph_subgoal_cursor=int(route.cursor),
            graph_subgoal_count=len(route.nodes),
            graph_subgoal_complete=route.complete,
        )
        return selected, diagnostics

    @torch.no_grad()
    def _certified_shortlist_before_decoder_warmup(
            self, goal_jpg_bytes, goal_key, frame_index,
            candidate_ceiling):
        """Build a certificate shortlist before the learned decoder is warm.

        The learned MemNav decoder requires ``S + W`` streamed frames, but
        Certified Episodic Compass does not consume that decoder. Its DINO
        shortlist and anchor-depth replay are valid once the eight-frame
        LingBot scale block has produced dense CLS features and camera poses.
        Keeping the decoder warm-up on this path silently rejects short GOAT
        revisits even when they have a valid causal history.
        """
        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_CANDIDATE_MIN_GAP,
            CERTIFIED_CANDIDATE_TOP_K,
            CERTIFIED_MINIMUM_ANCHOR,
        )

        frame_index = int(frame_index)
        candidate_ceiling = int(candidate_ceiling)
        if not self._has_frozen_visual_relocalizer():
            return [], None
        cache_key = (goal_key, candidate_ceiling)
        dense_cls = getattr(self, "dino_cls", ())
        if (frame_index < self.S - 1
                or len(dense_cls) != self.n
                or frame_index >= len(dense_cls)):
            frozen = self._certified_candidate_cache.setdefault(cache_key, [])
            return [dict(item) for item in frozen], None

        goal_path = os.path.join(
            self.rgb_dir, "_cert_shortlist_goal_{}.jpg".format(goal_key))
        if not os.path.isfile(goal_path):
            with open(goal_path, "wb") as handle:
                handle.write(goal_jpg_bytes)
        goal_cls = self._goal_cache.get(("cls", goal_key))
        if goal_cls is None:
            goal_image = self.lb.load_images([goal_path])[0][None].to(
                self.device)
            goal_cls = self.lb.dino(goal_image)["cls"]
            self._goal_cache[("cls", goal_key)] = goal_cls

        memory_cls = torch.stack(dense_cls, 0)[None].to(self.device)
        visual_cosine = torch.nn.functional.cosine_similarity(
            goal_cls.unsqueeze(1), memory_cls, dim=-1)[0]
        current_goal_cosine = float(visual_cosine[frame_index].item())
        cached = self._certified_candidate_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached], current_goal_cosine

        eligible = torch.zeros(
            frame_index + 1, dtype=torch.bool, device=self.device)
        high = min(frame_index - 1, candidate_ceiling)
        if high >= CERTIFIED_MINIMUM_ANCHOR:
            eligible[CERTIFIED_MINIMUM_ANCHOR:high + 1] = True
        candidates = temporal_nms_candidates(
            visual_cosine.detach().float().cpu().tolist(),
            eligible.detach().cpu().tolist(),
            top_k=CERTIFIED_CANDIDATE_TOP_K,
            min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP,
        )
        self._certified_candidate_cache[cache_key] = [
            dict(item) for item in candidates]
        return [dict(item) for item in candidates], current_goal_cosine

    @torch.no_grad()
    def plan(self, goal_jpg_bytes, forced_anchor=None, forced_gate=None,
             pose_only=False, retrieval_only=False,
             candidate_ceiling_override=None):
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

        ``candidate_ceiling_override`` is an evaluation-only tightening of
        the causal history boundary.  It cannot expose a frame newer than the
        default boundary and is used by the strict double-Revisit diagnostic
        to prevent C from relocalizing against its intervening B rollout.
        """
        if getattr(self, "depth_observation_only", False):
            raise RuntimeError(
                "planning is disabled in depth-observation-only mode")
        k = self.n - 1
        lo = self.amargin
        import hashlib
        gkey = hashlib.md5(goal_jpg_bytes).hexdigest()
        self._begin_goal_session(gkey)
        goal_start_frame = self._goal_start_frame.setdefault(gkey, k)
        candidate_ceiling = effective_candidate_ceiling(
            goal_start_frame, candidate_ceiling_override)
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
            if retrieval_only and self._has_frozen_visual_relocalizer():
                frozen, current_goal_cosine = (
                    self._certified_shortlist_before_decoder_warmup(
                        goal_jpg_bytes, gkey, k, candidate_ceiling))
                out["certified_visual_candidates"] = frozen
                out["visual_relocalization_candidates"] = frozen
                out["current_goal_cos"] = current_goal_cosine
                out["certified_shortlist_decoder_warmup_decoupled"] = True
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
            import torch.nn.functional as Fnn
            visual_cos_all = Fnn.cosine_similarity(
                goal_cls.unsqueeze(1), mem_cls, dim=-1)[0]
            current_goal_cos = float(visual_cos_all[k].item())
            certified_visual_candidates = []
            if self._has_frozen_visual_relocalizer():
                from MemNavData.certified_relocalization_runtime import (
                    CERTIFIED_CANDIDATE_MIN_GAP,
                    CERTIFIED_CANDIDATE_TOP_K,
                    CERTIFIED_MINIMUM_ANCHOR,
                )
                certified_cache_key = (gkey, candidate_ceiling)
                if certified_cache_key in self._certified_candidate_cache:
                    certified_visual_candidates = [
                        dict(item)
                        for item in self._certified_candidate_cache[
                            certified_cache_key]
                    ]
                else:
                    certified_eligible = torch.zeros(
                        k + 1, dtype=torch.bool, device=dev)
                    certified_hi = min(k - 1, candidate_ceiling)
                    if certified_hi >= CERTIFIED_MINIMUM_ANCHOR:
                        certified_eligible[
                            CERTIFIED_MINIMUM_ANCHOR:certified_hi + 1] = True
                    certified_visual_candidates = temporal_nms_candidates(
                        visual_cos_all.detach().float().cpu().tolist(),
                        certified_eligible.detach().cpu().tolist(),
                        top_k=CERTIFIED_CANDIDATE_TOP_K,
                        min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP,
                    )
                    self._certified_candidate_cache[certified_cache_key] = [
                        dict(item) for item in certified_visual_candidates
                    ]
            cand = torch.zeros(1, k + 1, dtype=torch.bool, device=dev)
            # Let frames near the goal-session boundary become eligible after
            # exclude_recent time has elapsed, but never admit observations
            # collected while pursuing this same goal. Without this ceiling a
            # long revisit eventually retrieves its own recent return path.
            hi = min(k - self.exclude_recent, candidate_ceiling)
            if hi >= lo:
                cand[0, lo:hi + 1] = True
            candidate_count = int(cand.sum().item())
            # RetrievalHead grew a fourth output (max_cos) after this server was
            # written; star-unpack so both the 3- and 4-value InternNav
            # revisions load.  Only the anchor index and the gate logit are
            # consumed here, and `--retrieval raw` scores candidates from raw
            # DINO cosine rather than the trained projection.
            match_idx, gate_logit, *_ = self.core.retrieval(
                goal_cls, mem_cls, cand)
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
                    certified_visual_candidates=certified_visual_candidates,
                    visual_relocalization_candidates=(
                        certified_visual_candidates),
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
                visual_cos = visual_cos_all
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
                    certified_visual_candidates=(
                        certified_visual_candidates),
                    visual_relocalization_candidates=(
                        certified_visual_candidates),
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
                aux_pose, graph_diagnostics = self._graph_conditioned_pose(
                    goal_key=gkey,
                    cache=cache,
                    current_pose=cur_pose,
                    goal_aux_pose=aux_pose,
                    anchor=anchor,
                    goal_start_frame=goal_start_frame,
                    metric_scale=mscale,
                )
            else:
                revisit = torch.zeros((1, self.core.n_rev, self.core.action_head.in_features), device=dev)
                aux_pose = torch.zeros((1, 2), device=dev)
                goal_rel_yaw = None
                graph_diagnostics = dict(
                    graph_subgoal_enabled=(self.graph_subgoal_spacing_m > 0.0),
                    graph_subgoal_node=None,
                    graph_subgoal_cursor=None,
                    graph_subgoal_count=0,
                    graph_subgoal_complete=False,
                    goal_aux_pose=None,
                )
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
                    **graph_diagnostics,
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
                **graph_diagnostics,
            )
        finally:
            self._restore(snap)
