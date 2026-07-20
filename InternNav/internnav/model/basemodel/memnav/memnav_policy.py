"""MemNav policy — trainable head over the frozen LingBotStream front-end.

Three goal pathways (see GL.md / memnav-project memory):
  (1) backbone current state      — frozen GCT (LingBotStream.window_forward)
  (2) revisit goal→history        — frozen GCT (LingBotStream.goal_append), visited goals
  (3) novel current→goal (DINO)   — TRAINABLE cross-attention, unseen goals
Retrieval confidence biases the decoder cross-attention toward (2) vs (3) (no multiply,
no goal_cls). NavDP DDPM decoder on top; NO critic (collision is geometric at eval).
Always goal-conditioned — no classifier-free "no-goal" branch (dropped: our goal-directed
two-leg episodes can require a genuine U-turn, so masking the goal out of the label gave
the unconditional branch contradictory supervision — same visual context, opposite action,
depending on whether that episode happened to reverse — worst exactly at the turn where a
CFG contrast would matter most; CFG guidance scale was never benchmarked for this model).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel

from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
from internnav.model.basemodel.memnav.retrieval_head import RetrievalHead
from internnav.model.basemodel.memnav.revisit_pose import GaugeInvariantRevisitPose
from internnav.model.basemodel.memnav.route_sketch import (
    build_residual_route_sketch,
)
from internnav.model.encoder.navdp_backbone import (
    LearnablePositionalEncoding,
    NavDP_ImageGoal_Backbone,
    SinusoidalPosEmb,
    TokenCompressor,
)


# --------------------------------------------------------------------------- #
# (3.novel) current DINO  →  goal DINO  cross-attention — trainable
# --------------------------------------------------------------------------- #
class NovelBranch(nn.Module):
    """Early-fusion goal↔current (NavDP_ImageGoal_Backbone design): 6-ch `concat(current, goal)`
    is **jointly** encoded by a trainable DINOv2-S (the 6-ch `patch_embed.proj` mixes the two
    images from layer 0 — true early fusion, the optical-flow-friendly inductive bias), → patch
    tokens → TokenCompressor → m_novel tokens. For unseen/overlapping goals; the diffusion reads
    the heading toward goal-matching content. (skips NavDP's mean-pool to keep spatial info.)
    """

    def __init__(self, dim=384, heads=8, out_tokens=4, image_size=224,
                 pretrained_checkpoint=None, device="cuda"):
        super().__init__()
        if not pretrained_checkpoint:
            raise ValueError(
                'MemNav novel branch requires a pretrained DINO/Depth-Anything '
                'checkpoint; set MEMNAV_DINO_WEIGHTS'
            )
        self.backbone = NavDP_ImageGoal_Backbone(
            image_size=image_size, embed_size=dim, device=device,
            checkpoint=pretrained_checkpoint,
        )
        self.backbone.project_layer = nn.Identity()              # unused (we skip NavDP's mean-pool)
        self.image_size = image_size
        self.register_buffer(
            'preprocess_mean',
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'preprocess_std',
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.proj = nn.Linear(384, dim)                          # DINOv2-S patch dim -> token_dim
        self.compress = TokenCompressor(dim, heads, out_tokens)

    def forward(self, cur_img, goal_img):
        """cur_img, goal_img: [B, 3, H, W] in [0,1] -> readout [B, out_tokens, dim]."""
        sz = (self.image_size, self.image_size)
        cur = F.interpolate(cur_img, size=sz, mode="bilinear", align_corners=False)
        goal = F.interpolate(goal_img, size=sz, mode="bilinear", align_corners=False)
        # The pretrained DINO trunk expects ImageNet-normalized RGB. Normalize the
        # two images independently before the six-channel early-fusion concat.
        cur = (cur.float() - self.preprocess_mean) / self.preprocess_std
        goal = (goal.float() - self.preprocess_mean) / self.preprocess_std
        six = torch.cat([cur, goal], dim=1)                      # [B, 6, H, W]  early fusion
        patch = self.backbone.imagegoal_encoder.get_intermediate_layers(six)[0]  # [B, N, 384] (no pool)
        return self.compress(self.proj(patch))                   # [B, out_tokens, dim]


# --------------------------------------------------------------------------- #
# (2.merge) Revisit: analytic relative pose -> decoder tokens + planar direction
# --------------------------------------------------------------------------- #
class RevisitMerge(nn.Module):
    """Turns the **current** and **goal** absolute camera poses (frozen camera head, map
    frame) into the goal's relative pose, analytically — NOT via independently-embedded
    absolute-pose tokens merged by attention. T_cur^-1 T_goal is BILINEAR in the two
    absolute poses (t_rel = R_cur^T(t_goal - t_cur) is a product of a rotation derived
    from cur_pose and a translation difference derived from both); a linear embed of each
    pose + attention-merge can only produce affine combinations of the two, and can never
    synthesize that cross term. So it's computed here in closed form (`_relative_pose`),
    same reasoning as VGGT/Pi3 supervising relative pose directly.

      - pose_encoder → a gauge-invariant planar bearing, stream-normalized bounded
        range, and an optional separately calibrated geometric reliability.  The raw long-range
        translation and endpoint camera rotation never enter the action token.
      - revisit_head  → revisit_readout (the diffusion goal slot). TRAINABLE: a plain
        Linear on the four-dimensional robust pose code — no attention needed for a
        single input feature vector.
      - aux_pose_head → translation direction in (x, y), not metric scale and not θ.
        The direction loss is invariant to LingBot's per-sequence canonical scale and,
        through ``rel_adapter``, shapes the same relative feature consumed by the
        diffusion revisit tokens. Raw (x,y) output is retained only for metric/drift
        diagnostics.
        θ (net heading change along the path from
        departure to arrival) is NOT a function of the two endpoint poses — it depends on
        the geodesic route's shape between them (obstacle layout), which two poses don't
        encode; that's the diffusion decoder's job (it sees current_state's depth/visual
        context), not RevisitMerge's. And the goal image's own rendered orientation is
        independent of the real arrival heading by construction of the data generator
        (MemNavData/generate_twoleg.py: "NO terminal orientation alignment... arrival
        heading is the natural approach heading"; goal_yaw = anchor's OWN heading +
        random jitter) — so there is no θ signal in (cur_pose, goal_pose) to extract even
        in principle.
        A global affine head cannot repair sequence-dependent monocular scale or
        accumulated VO drift, which is why metric x/y MSE is diagnostic rather than a
        training loss. ``rel_adapter`` is shared by the auxiliary direction head and
        ``revisit_head``, so this supervision is no longer an isolated sidecar.  The
        reliability head remains available as a diagnostic, but it is not allowed to
        attenuate conditioning unless explicitly enabled.  Local sparse-cache probes
        showed that it was nearly constant and failed on wrong anchors, while the
        semantic gate already rejected those anchors.
    """

    def __init__(self, dim=384, n_out=4, distance_unit_steps=32,
                 max_frame_num=4096, reliability_hidden=16,
                 reliability_init=0.95, condition_on_reliability=True):
        super().__init__()
        self.pose_encoder = GaugeInvariantRevisitPose(
            distance_unit_steps=distance_unit_steps,
            max_frame_num=max_frame_num,
            reliability_hidden=reliability_hidden,
            reliability_init=reliability_init,
            condition_on_reliability=condition_on_reliability,
        )
        pose_dim = 4
        self.rel_adapter = nn.Sequential(
            nn.Linear(pose_dim, pose_dim), nn.GELU(), nn.Linear(pose_dim, pose_dim)
        )
        # Residual identity at initialization lets both action and auxiliary
        # gradients reshape the robust representation without perturbing it at
        # step zero.  The dimension change intentionally prevents unsafe resume
        # from raw-t_rel checkpoints.
        nn.init.zeros_(self.rel_adapter[-1].weight)
        nn.init.zeros_(self.rel_adapter[-1].bias)
        self.revisit_head = nn.Linear(pose_dim, n_out * dim)
        self.n_out, self.dim = n_out, dim
        # The first two robust-code coordinates already are corrected NavDP bearing.
        self.aux_pose_head = nn.Linear(pose_dim, 2)
        with torch.no_grad():
            self.aux_pose_head.weight.zero_()
            self.aux_pose_head.weight[:, :2].copy_(torch.eye(2))
            self.aux_pose_head.bias.zero_()

    @staticmethod
    def _split_pose9(pose9):
        """9-d (absT[3], quaR[4] xyzw cam->world, FoV[2]) -> (t [...,3], unit-quat [...,4]).
        Drops FoV (constant intrinsic); normalizes the quaternion (head emits raw
        non-unit quat; magnitude is decoded away)."""
        return pose9[..., :3], F.normalize(pose9[..., 3:7], dim=-1)

    @staticmethod
    def _relative_pose(cur_pose9, goal_pose9):
        """Analytic T_cur^-1 @ T_goal, split (not recombined into a quaternion — nothing
        downstream needs the compact 4-d form, and mat_to_quat's branch-selection has
        known numerical rough edges near 180-deg rotations that a plain flattened
        rotation matrix avoids).
        quaR is cam->world (p_world = R @ p_cam), so T_cur^-1 expresses goal in cur's own
        local frame: t_rel = R_cur^T(t_goal - t_cur), R_rel = R_cur^T R_goal — the
        bilinear cross term a linear head can't reconstruct from (cur_pose, goal_pose)
        embedded independently. Lazy import: needs lingbot_repo on sys.path, which
        LingBotStream.__init__ guarantees has already run by the time this is called.
        """
        from lingbot_map.utils.rotation import quat_to_mat
        t_cur, q_cur = RevisitMerge._split_pose9(cur_pose9)
        t_goal, q_goal = RevisitMerge._split_pose9(goal_pose9)
        R_cur = quat_to_mat(q_cur)                                    # [B,3,3]
        R_goal = quat_to_mat(q_goal)                                  # [B,3,3]
        R_cur_T = R_cur.transpose(-1, -2)
        t_rel = (R_cur_T @ (t_goal - t_cur).unsqueeze(-1)).squeeze(-1)   # R_cur^T (t_goal - t_cur)
        R_rel = R_cur_T @ R_goal                                         # R_cur^T R_goal
        return t_rel, R_rel

    def forward(self, cur_pose, goal_pose, pose_context=None):
        """cur_pose, goal_pose: [B, 9] absolute camera poses (map frame)."""
        t_rel, R_rel = self._relative_pose(cur_pose, goal_pose)          # [B,3], [B,3,3]
        encoded = self.pose_encoder(t_rel, R_rel, pose_context)
        rel_feat = encoded['pose_code']                                  # [B,4]
        rel_feat = rel_feat + self.rel_adapter(rel_feat)
        aux_pose = self.aux_pose_head(rel_feat)                           # [B,2] direction proxy
        # Supervise the exact adapted range coordinate consumed by revisit_head.
        # This adds no checkpoint parameters: old checkpoints remain loadable and
        # the loss can be enabled independently by the trainer.
        aux_range_code = rel_feat[..., 2]
        revisit_readout = self.revisit_head(rel_feat).view(-1, self.n_out, self.dim)
        # R_rel returned too — not for any loss (no head/calibration needed for it, it's a
        # raw feature into revisit_head), just so the trainer can log a rotation-accuracy
        # diagnostic against GT (batch_goal_rel_rotation), same treatment as the gate/match
        # diagnostics already logged under no_grad in MemNavTrainer.compute_loss.
        return (
            revisit_readout,
            aux_pose,
            R_rel,
            encoded['raw_direction'],
            encoded['reliability'],
            encoded['reliability_features'],
            encoded['range_steps'],
            aux_range_code,
        )


# --------------------------------------------------------------------------- #
# MemNavNet — full policy: frozen encode loop + (trainable) gate/compress/decoder
# --------------------------------------------------------------------------- #
class MemNavNet(nn.Module):
    def __init__(self, lingbot_kwargs=None, dino_dim=1024, lingbot_dim=2048, depth_feat_dim=256,
                 token_dim=384, heads=8, m_rgbd=4, m_depth=4, m_revisit=4, m_novel=4,
                 predict_size=24, temporal_depth=8, num_diffusion_iters=10, goal_warm=64,
                 novel_backbone_weights=None, gate_center=0.94, gate_width=0.04,
                 gate_slope_init=1.6, gate_bias_init=0.0,
                 pose_scale_window=64, pose_reliability_hidden=16,
                 pose_reliability_init=0.95,
                 use_pose_reliability_conditioning=True,
                 use_route_sketch=False, route_horizons=(2, 8, 24),
                 require_versioned_cache=False,
                 device="cuda"):
        super().__init__()
        self.lingbot = LingBotStream(device=device, **(lingbot_kwargs or {}))
        self.window = self.lingbot.window
        self.num_scale = self.lingbot.num_scale
        self.device = device
        self.heads = heads
        self.predict_size = predict_size
        # goal_append_warm's live-recompute depth before streaming the goal — deeper than
        # `window` on purpose (see LingBotStream.goal_append_warm); validated against a
        # continuous-stream oracle in scripts/diag_lingbot_pose_accuracy.py.
        self.goal_warm = int(goal_warm)
        if self.goal_warm < 0:
            raise ValueError('goal_warm must be non-negative')
        self.require_versioned_cache = bool(require_versioned_cache)
        self.use_pose_reliability_conditioning = bool(
            use_pose_reliability_conditioning
        )
        self.pose_scale_window = int(pose_scale_window)
        if self.pose_scale_window < 2:
            raise ValueError('pose_scale_window must be at least two frames')

        # trainable heads
        self.retrieval = RetrievalHead(
            dino_dim=dino_dim,
            gate_center=gate_center,
            gate_width=gate_width,
            gate_slope_init=gate_slope_init,
            gate_bias_init=gate_bias_init,
        )
        self.novel = NovelBranch(
            dim=token_dim, heads=heads, out_tokens=m_novel,
            pretrained_checkpoint=novel_backbone_weights, device=device,
        )

        # current_state = two Perceiver branches (LoGoPlanner-style: perception + geometry)
        #   RGBD branch  : post-GCT window tokens (2C)        -> m_rgbd tokens
        #   depth branch : feature-only depth head (geometry) -> m_depth tokens
        self.proj_current = nn.Linear(lingbot_dim, token_dim)
        self.proj_depth = nn.Linear(depth_feat_dim, token_dim)
        self.compress_rgbd = TokenCompressor(token_dim, heads, m_rgbd)
        self.compress_depth = TokenCompressor(token_dim, heads, m_depth)
        # revisit: analytic relative pose from current + goal absolute camera poses (+ aux pose head)
        self.revisit_merge = RevisitMerge(
            token_dim,
            m_revisit,
            distance_unit_steps=self.window,
            max_frame_num=int((lingbot_kwargs or {}).get('max_frame_num', 4096)),
            reliability_hidden=pose_reliability_hidden,
            reliability_init=pose_reliability_init,
            condition_on_reliability=self.use_pose_reliability_conditioning,
        )

        # --- NavDP DDPM decoder (no critic) ---
        # memory layout: [ time(1) | current_state(n_cs) | revisit(n_rev) | novel(n_nov) ]
        self.n_cs, self.n_rev, self.n_nov = m_rgbd + m_depth, m_revisit, m_novel
        self.route_horizons = tuple(int(value) for value in route_horizons)
        self.route_sketch = (
            build_residual_route_sketch(token_dim, self.route_horizons)
            if bool(use_route_sketch) else None
        )
        if self.route_sketch is not None and len(self.route_horizons) > self.n_cs:
            raise ValueError(
                'route horizon count cannot exceed current-state token count'
            )
        if self.route_sketch is not None and any(
            value > self.predict_size for value in self.route_horizons
        ):
            raise ValueError(
                'route horizons cannot exceed the action prediction length'
            )
        self.mem_len = 1 + self.n_cs + self.n_rev + self.n_nov
        self.input_embed = nn.Linear(3, token_dim)            # noisy waypoints -> tokens
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, self.mem_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=4 * token_dim,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=temporal_depth)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        # (no critic — collision is checked geometrically from LingBot's point map at eval)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        tgt = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.register_buffer("tgt_mask",
                             tgt.float().masked_fill(tgt == 0, float("-inf")).masked_fill(tgt == 1, 0.0))

        # Keep the decoder routing identifiable: only the supervised per-sample gate
        # chooses revisit vs novel. A second learnable global branch bias can cancel
        # the gate and recreate an unsupervised shortcut.
        self.branch_bias = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.to(device)   # move trainable heads to device (lingbot.model already there)

    def build_current_state(self, current, depth_feat):
        """current [B,P,2C] (post-GCT), depth_feat [B,Pf,Cd] -> current_state [B, m_rgbd+m_depth, token_dim]."""
        rgbd = self.compress_rgbd(self.proj_current(current))    # [B, m_rgbd, token_dim]
        geom = self.compress_depth(self.proj_depth(depth_feat))  # [B, m_depth, token_dim]
        return torch.cat([rgbd, geom], dim=1)

    def build_revisit(self, cur_pose, goal_pose, pose_context=None):
        """cur_pose/goal_pose [B, 9] absolute camera poses (current frame + goal_append_warm)
        -> robust revisit tokens, auxiliary direction, and pose diagnostics."""
        return self.revisit_merge(cur_pose, goal_pose, pose_context)

    # ----- DDPM decoder ------------------------------------------------ #
    def _memory(self, current_state, revisit, novel, timestep):
        """[B, mem_len, D] = [time | current_state | revisit | novel] + pos embed."""
        B = current_state.shape[0]
        time_emb = self.time_emb(timestep.to(self.device)).unsqueeze(1).expand(B, 1, -1)
        mem = torch.cat([time_emb, current_state, revisit, novel], dim=1)
        return mem + self.cond_pos_embed(mem)

    def _gate_mask(self, gate):
        """Per-sample cross-attention bias [B*heads, predict_size, mem_len] — directs
        attention without scaling the readouts.
          revisit cols += log(gate), novel cols += log(1-gate)"""
        B = gate.shape[0]
        bias = gate.new_zeros(B, self.mem_len)
        rs, re = 1 + self.n_cs, 1 + self.n_cs + self.n_rev
        ns, ne = re, re + self.n_nov
        g = gate.clamp(1e-4, 1 - 1e-4)
        bias[:, rs:re] = torch.log(g).unsqueeze(1) + self.branch_bias[0]      # revisit
        bias[:, ns:ne] = torch.log(1 - g).unsqueeze(1) + self.branch_bias[1]  # novel
        bias = bias[:, None, None, :].expand(B, self.heads, self.predict_size, self.mem_len)
        return bias.reshape(B * self.heads, self.predict_size, self.mem_len)

    def predict_noise(self, noisy, timestep, current_state, revisit, novel, gate):
        a = self.input_embed(noisy)
        a = a + self.out_pos_embed(a)
        mem = self._memory(current_state, revisit, novel, timestep)
        out = self.decoder(tgt=a, memory=mem, tgt_mask=self.tgt_mask,
                           memory_mask=self._gate_mask(gate))
        return self.action_head(self.layernorm(out))

    def _effective_revisit_gate(self, revisit_gate, pose_reliability):
        if self.use_pose_reliability_conditioning:
            return revisit_gate * pose_reliability
        return revisit_gate

    def prepare_condition(self, batch):
        """Encode every goal-conditioned memory input once.

        Keeping this separate from the random training-noise draw lets offline
        diagnostics reuse the exact condition for a complete reverse-diffusion
        trajectory.  It does not change the training forward path.
        """
        dev = self.device
        enc = self.encode_memory(batch)
        current_state = self.build_current_state(enc["current"], enc["depth_feat"])
        (
            revisit,
            aux_pose,
            R_rel,
            raw_pose_direction,
            pose_reliability,
            pose_reliability_features,
            pose_range_steps,
            aux_range_code,
        ) = self.build_revisit(
            enc["cur_pose"], enc["goal_pose"], enc["pose_context"]
        )
        novel = self.novel(batch["batch_window_images"][:, -1].to(dev),   # current frame [B,3,H,W]
                           batch["batch_goal_image"].to(dev))             # goal frame
        # A pose-reliability multiplier is opt-in.  The sparse-cache diagnostic
        # found it nearly constant and unable to identify a wrong semantic anchor;
        # by default the calibrated semantic gate alone chooses revisit vs novel.
        effective_revisit_gate = self._effective_revisit_gate(
            enc['revisit_gate'], pose_reliability
        )
        route = None
        if self.route_sketch is not None:
            route = self.route_sketch(
                current_state, revisit, novel, effective_revisit_gate
            )
            current_state = route['current_state']
        condition = dict(
            current_state=current_state,
            revisit=revisit,
            novel=novel,
            aux_pose=aux_pose,
            R_rel=R_rel,
            raw_pose_direction=raw_pose_direction,
            pose_reliability=pose_reliability,
            pose_reliability_features=pose_reliability_features,
            pose_range_steps=pose_range_steps,
            aux_range_code=aux_range_code,
            ret_logits=enc['ret_logits'],
            revisit_gate=enc['revisit_gate'],
            effective_revisit_gate=effective_revisit_gate,
            gate_logit=enc['gate_logit'],
            gate_feature=enc['gate_feature'],
            match_idx=enc['match_idx'],
            anchor_idx=enc['anchor_idx'],
            anchor_teacher_forced=enc['anchor_teacher_forced'],
        )
        if route is not None:
            condition.update(
                route_direction=route['direction'],
                route_raw_direction_norm=route['raw_direction_norm'],
                route_curvature_gate=route['curvature_gate'],
                route_residual_scale=route['residual_scale'],
            )
        return condition

    def forward_with_condition(self, batch, condition):
        """Training-noise prediction from an already encoded condition."""
        dev = self.device
        labels = batch["batch_labels"].to(dev)          # [B, predict_size, 3]
        B = labels.shape[0]
        noise = batch.get('diagnostic_noise')
        if noise is None:
            noise = torch.randn_like(labels)
        else:
            noise = noise.to(dev)
        timesteps = batch.get('diagnostic_timesteps')
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=dev
            )
        else:
            timesteps = timesteps.to(dev)
        noisy = self.noise_scheduler.add_noise(labels, noise, timesteps)

        noise_pred = self.predict_noise(
            noisy,
            timesteps,
            condition["current_state"],
            condition["revisit"],
            condition["novel"],
            condition["effective_revisit_gate"],
        )
        result = dict(
            noise_pred=noise_pred, noise=noise, timesteps=timesteps,
            aux_pose=condition["aux_pose"], R_rel=condition["R_rel"],
            raw_pose_direction=condition["raw_pose_direction"],
            pose_reliability=condition["pose_reliability"],
            pose_reliability_features=condition["pose_reliability_features"],
            pose_range_steps=condition["pose_range_steps"],
            aux_range_code=condition["aux_range_code"],
            ret_logits=condition["ret_logits"],
            revisit_gate=condition["revisit_gate"],
            effective_revisit_gate=condition["effective_revisit_gate"],
            gate_logit=condition["gate_logit"],
            gate_feature=condition["gate_feature"],
            match_idx=condition["match_idx"], anchor_idx=condition["anchor_idx"],
            anchor_teacher_forced=condition["anchor_teacher_forced"],
            gate_effective_threshold=self.retrieval.effective_gate_threshold,
            gate_normalized_slope=self.retrieval.gate_slope,
        )
        for name in (
            'route_direction',
            'route_raw_direction_norm',
            'route_curvature_gate',
            'route_residual_scale',
        ):
            if name in condition:
                result[name] = condition[name]
        return result

    def forward(self, batch):
        return self.forward_with_condition(batch, self.prepare_condition(batch))

    @torch.no_grad()
    def sample_actions_from_condition(
        self,
        condition,
        initial_noise=None,
        generator=None,
        num_inference_steps=None,
    ):
        """Run the complete DDPM reverse process for a prepared condition.

        A caller can pass the same ``initial_noise`` and two generators with the
        same seed to obtain a paired correct-goal vs shuffled-goal comparison.
        DDPM injects variance at intermediate steps, so sharing only the initial
        noise would not be a controlled comparison.
        """
        current_state = condition['current_state']
        batch_size = current_state.shape[0]
        device = current_state.device
        shape = (batch_size, self.predict_size, 3)
        if initial_noise is None:
            sample = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f'initial_noise shape must be {shape}, got {tuple(initial_noise.shape)}'
                )
            sample = initial_noise.to(device=device).clone()

        steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )
        if not 1 <= steps <= self.noise_scheduler.config.num_train_timesteps:
            raise ValueError(
                f'num_inference_steps must be in [1, '
                f'{self.noise_scheduler.config.num_train_timesteps}], got {steps}'
            )
        try:
            self.noise_scheduler.set_timesteps(steps, device=device)
        except TypeError:
            # Compatibility with older diffusers; individual timesteps are moved
            # below before entering the model.
            self.noise_scheduler.set_timesteps(steps)

        for scheduler_timestep in self.noise_scheduler.timesteps:
            timestep = torch.as_tensor(
                scheduler_timestep, device=device, dtype=torch.long
            )
            batch_timestep = timestep.expand(batch_size)
            noise_pred = self.predict_noise(
                sample,
                batch_timestep,
                condition['current_state'],
                condition['revisit'],
                condition['novel'],
                condition['effective_revisit_gate'],
            )
            sample = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=sample,
                generator=generator,
            ).prev_sample
        return sample

    @torch.no_grad()
    def _load_cache(self, path, camera_path, rgb_dir):
        """Assemble the KV cache dict from disk. If the npz lacks
        ``scale_k/scale_v`` (--skip_scale precompute mode), compute it on the
        fly from the first ``num_scale`` RGB frames of ``rgb_dir`` — bf16 output,
        LRU-cached per trajectory inside LingBotStream."""
        with np.load(path) as c, np.load(camera_path) as cc:
            # Materialize once: validation and GPU conversion otherwise cause
            # np.load to reread the large ZIP_STORED KV arrays independently.
            aggregator = {name: c[name] for name in c.files}
            camera = {name: cc[name] for name in cc.files}
            layout = validate_cache_pair(
                aggregator,
                camera,
                expected_num_scale_frames=self.num_scale,
                expected_sliding_window=self.window,
                require_versioned=self.require_versioned_cache,
            )
            keys = set(aggregator)
            if "scale_k" in keys and "scale_v" in keys:
                sk, sv, ak, av = LingBotStream._cache_to_layered(
                    aggregator["scale_k"], aggregator["scale_v"],
                    aggregator["anchor_k"], aggregator["anchor_v"], self.device)
            else:
                sk, sv = self.lingbot.get_scale_kv(rgb_dir)
                ak = torch.as_tensor(aggregator["anchor_k"], device=self.device, dtype=torch.bfloat16)\
                    .permute(1, 2, 0, 3, 4).contiguous()
                av = torch.as_tensor(aggregator["anchor_v"], device=self.device, dtype=torch.bfloat16)\
                    .permute(1, 2, 0, 3, 4).contiguous()
            ck, cv = LingBotStream._cam_to_device(
                camera["cam_k"], camera["cam_v"], self.device
            )
            # cam_pose_enc [S,9]: the frozen camera head's own pose for every REAL
            # trajectory frame, captured during the continuous precompute stream.
            cam_pose_enc = torch.as_tensor(
                camera["cam_pose_enc"], device=self.device, dtype=torch.float32
            )
        # cur_pose reads cam_pose_enc directly instead of reconstructing it from a
        # cold-start window. goal_pose still needs a live camera_pose() call because
        # the goal image is newly inserted and has no trajectory entry.
        return dict(
            scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av,
            anchor_frame_indices=torch.as_tensor(
                layout.anchor_frame_indices, dtype=torch.long
            ),
            cam_k=ck, cam_v=cv,
            cam_frame_indices=torch.as_tensor(
                layout.cam_frame_indices, dtype=torch.long
            ),
            cam_pose_enc=cam_pose_enc,
            keyframe_interval=layout.keyframe_interval,
            cache_schema_version=layout.schema_version,
        )

    @torch.no_grad()
    def _pose_consistency_context(self, cam_pose_enc, k, anchor, goal_pose, semantic_score_z):
        """Observable, gauge-invariant cues for long-range pose reliability.

        No GT or future trajectory is used.  A uniform map rescaling cancels from
        every ratio; disagreement between the recent and prefix step scales is a
        direct signal of the time-varying drift seen in long 3-leg streams.
        """
        positions = cam_pose_enc[: k + 1, :3]
        planar_steps = torch.linalg.vector_norm(
            positions[1:, (0, 2)] - positions[:-1, (0, 2)], dim=-1
        )
        valid = torch.isfinite(planar_steps) & (planar_steps > 1e-6)
        if bool(valid.any()):
            prefix_scale = planar_steps[valid].median()
        else:
            prefix_scale = planar_steps.new_tensor(1.0)

        recent_start = max(0, len(planar_steps) - self.pose_scale_window)
        recent_steps = planar_steps[recent_start:]
        recent_valid = torch.isfinite(recent_steps) & (recent_steps > 1e-6)
        recent_scale = (
            recent_steps[recent_valid].median()
            if bool(recent_valid.any())
            else prefix_scale
        )
        prefix_scale = prefix_scale.clamp_min(1e-6)
        recent_scale = recent_scale.clamp_min(1e-6)
        step_scale_drift = (recent_scale / prefix_scale).log().abs()

        anchor_position = cam_pose_enc[anchor, :3]
        goal_anchor_raw = torch.linalg.vector_norm(
            (goal_pose[:3] - anchor_position)[[0, 2]]
        )
        return {
            'step_scale': prefix_scale,
            'step_scale_drift': step_scale_drift,
            'anchor_gap': prefix_scale.new_tensor(float(k - anchor)),
            'goal_anchor_steps': goal_anchor_raw / prefix_scale,
            'semantic_score_z': semantic_score_z.to(prefix_scale),
        }

    @staticmethod
    def _select_revisit_anchor(
        ret_logits,
        match_idx,
        pos_mask,
        *,
        training,
        force_oracle_positive=False,
        teacher_mask=None,
    ):
        """Choose a live or best-positive anchor and report actual TF rows."""
        if pos_mask is None:
            return match_idx, torch.zeros_like(match_idx, dtype=torch.bool)
        pos_mask = pos_mask.to(device=ret_logits.device, dtype=torch.bool)
        has_positive = pos_mask.any(-1)
        negative_inf = torch.finfo(ret_logits.dtype).min
        best_positive = ret_logits.masked_fill(
            ~pos_mask, negative_inf
        ).argmax(-1)
        if force_oracle_positive:
            request_teacher = torch.ones_like(has_positive)
        elif training:
            if teacher_mask is None:
                request_teacher = torch.ones_like(has_positive)
            else:
                request_teacher = torch.as_tensor(
                    teacher_mask,
                    device=ret_logits.device,
                    dtype=torch.bool,
                )
                if request_teacher.shape != has_positive.shape:
                    raise ValueError(
                        'anchor_teacher_forcing_mask must have shape '
                        f'{tuple(has_positive.shape)}, got '
                        f'{tuple(request_teacher.shape)}'
                    )
        else:
            request_teacher = torch.zeros_like(has_positive)
        teacher_forced = request_teacher & has_positive
        anchor = torch.where(teacher_forced, best_positive, match_idx)
        return anchor, teacher_forced

    def encode_memory(self, batch):
        """Frozen front-end orchestration. Retrieval (trainable, batched) picks the
        match index; a per-sample loop runs the frozen LingBot ops. Returns the
        readouts the trainable head consumes.
        """
        dev = self.device
        # goal_cls: real goal images (goal_{j}.jpg) have no cached CLS, so compute it
        # from the goal image via the frozen context-free DINO trunk (same space as the
        # cached per-frame dino_cls). Fall back to a provided batch_goal_cls (old path /
        # smoke tests where the goal is a trajectory frame).
        if batch.get("batch_goal_cls") is not None:
            goal_cls = batch["batch_goal_cls"].to(dev)
        else:
            goal_cls = self.lingbot.dino(batch["batch_goal_image"].to(dev))["cls"]  # [B, D']
        mem_cls = batch["batch_mem_cls"].to(dev)
        cand_mask = batch["batch_cand_mask"].to(dev)   # revisit candidates E(k) = [amargin..k-t]
        # (trainable) retrieval — match index + gate logit + ranking logits (over candidates)
        match_idx, gate_logit, ret_logits, gate_feature = self.retrieval(
            goal_cls, mem_cls, cand_mask
        )
        diagnostic_anchor_mode = batch.get(
            'diagnostic_retrieval_anchor_mode', 'projected'
        )
        if diagnostic_anchor_mode == 'raw':
            if self.training:
                raise RuntimeError(
                    'diagnostic raw retrieval anchor is evaluation-only'
                )
            match_idx, _ = self.retrieval.raw_match(
                goal_cls, mem_cls, cand_mask
            )
        elif diagnostic_anchor_mode != 'projected':
            raise ValueError(
                'diagnostic_retrieval_anchor_mode must be projected/raw, got '
                f'{diagnostic_anchor_mode!r}'
            )
        revisit_gate = torch.sigmoid(gate_logit)       # P(revisit) for the decoder soft-gate

        # goal_append anchor: training defaults to the legacy all-positive teacher
        # forcing, but the trainer may provide a per-row mask for scheduled exposure
        # to live retrieval anchors. Evaluation always uses match_idx unless an
        # explicit oracle-positive diagnostic is requested.
        pos_mask = batch.get("batch_pos_mask")
        force_oracle_positive = bool(batch.get('diagnostic_oracle_positive', False))
        teacher_mask = batch.get('anchor_teacher_forcing_mask')
        anchor, anchor_teacher_forced = self._select_revisit_anchor(
            ret_logits,
            match_idx,
            pos_mask,
            training=self.training,
            force_oracle_positive=force_oracle_positive,
            teacher_mask=teacher_mask,
        )

        B = len(batch["cache_paths"])
        lo = int(batch.get(
            'diagnostic_anchor_min_frame',
            self.num_scale + self.window - 1,
        ))
        if lo < self.num_scale:
            raise ValueError(
                f'anchor minimum {lo} precedes scale block {self.num_scale}'
            )
        cur_t, dfeat_t, curp, goalp = [], [], [], []
        pose_context_rows = []
        for b in range(B):
            k = int(batch["cur_steps"][b])
            rgb_dir = batch["rgb_dirs"][b]
            goal_img = batch["batch_goal_image"][b].to(dev)
            win_img = batch["batch_window_images"][b].to(dev)
            with torch.no_grad():
                cache = self._load_cache(
                    batch["cache_paths"][b], batch["camera_cache_paths"][b], rgb_dir
                )
                ck, cv = cache["cam_k"], cache["cam_v"]
                # (1) current state: post-GCT tokens + depth-head geometry + pose feature
                #  wt: window tokens [W, P, 2C], cur_agg: current frame's multi-layer agg, psi: patch_start_idx
                wt, cur_agg, psi = self.lingbot.window_forward(cache, win_img, k, return_multilayer=True)
                cur = wt[-1]                                                        # [P, 2C]
                dfeat = self.lingbot.depth_feature(cur_agg, win_img[-1:][None], psi)  # [Pf, Cd]
                # cur_pose: read the precomputed continuous-stream pose directly (exact,
                # no cold-start reconstruction) — k is always a real trajectory frame.
                cur_pose = cache["cam_pose_enc"][k]                                  # [9] current abs pose
                # (2) revisit: goal_append_warm at the anchor frame (clamped valid) -> goal abs pose.
                # Deep warm-recompute (self.goal_warm, not the nominal window W) before streaming
                # the goal — window_forward's cold start at the W boundary starves the goal's pose
                # estimate; goal_warm=64 empirically matches a true continuous-stream oracle (see
                # LingBotStream.goal_append_warm / scripts/diag_lingbot_pose_accuracy.py).
                m = int(anchor[b].clamp(lo, k - 1).item())
                _, goal_agg = self.lingbot.goal_append_warm(goal_img, cache, m, rgb_dir,
                                                            self.goal_warm, return_agg=True)
                goal_pose = self.lingbot.camera_pose(
                    ck, cv, m + 1, goal_agg, cache["cam_frame_indices"]
                )[-1]   # [9] goal abs pose
                semantic_score_z = (
                    (gate_feature[b] - self.retrieval.gate_center)
                    / self.retrieval.gate_width
                )
                pose_context_rows.append(
                    self._pose_consistency_context(
                        cache['cam_pose_enc'], k, m, goal_pose, semantic_score_z
                    )
                )
                # (3) novel branch runs on raw images (batched, in forward) — no live dino needed
            cur_t.append(cur); dfeat_t.append(dfeat); curp.append(cur_pose); goalp.append(goal_pose)

        pose_context = {
            name: torch.stack([row[name] for row in pose_context_rows])
            for name in pose_context_rows[0]
        }
        return dict(
            current=torch.stack(cur_t),      # [B, P, 2C]    post-GCT (RGBD branch)
            depth_feat=torch.stack(dfeat_t), # [B, Pf, Cd]   depth-head geometry
            cur_pose=torch.stack(curp),      # [B, 9]        current absolute camera pose (map frame)
            goal_pose=torch.stack(goalp),    # [B, 9]        goal absolute camera pose (map frame)
            pose_context=pose_context,
            match_idx=match_idx, anchor_idx=anchor, revisit_gate=revisit_gate,
            anchor_teacher_forced=anchor_teacher_forced,
            gate_logit=gate_logit, gate_feature=gate_feature, ret_logits=ret_logits,
        )


# --------------------------------------------------------------------------- #
# HF wrapper (for scripts/train/train.py registry: from_pretrained + config)
# --------------------------------------------------------------------------- #
class MemNavModelConfig(PretrainedConfig):
    model_type = 'memnav'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class MemNavPolicy(PreTrainedModel):
    config_class = MemNavModelConfig

    @staticmethod
    def _validate_checkpoint_incompatibility(incompatible, path):
        """Allow compact checkpoints to omit frozen LingBot tensors only."""
        missing_frozen = [
            key for key in incompatible.missing_keys if 'lingbot.' in key
        ]
        missing_trainable = [
            key for key in incompatible.missing_keys if 'lingbot.' not in key
        ]
        unexpected = list(incompatible.unexpected_keys)
        if missing_trainable or unexpected:
            raise ValueError(
                f"Unsafe MemNav checkpoint {path}: "
                f"missing non-LingBot={missing_trainable[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        return len(missing_frozen)

    @staticmethod
    def _validate_checkpoint_path(path):
        """Fail before model construction when an explicitly requested file is absent."""
        if path and len(str(path)) > 0 and not os.path.isfile(path):
            raise FileNotFoundError(
                f"Requested MemNav initialization checkpoint is not a file: {path}"
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, 'model_dump'):                  # pydantic ExpCfg -> wrap
            config = cls.config_class(model_cfg=config)
        path = pretrained_model_name_or_path
        cls._validate_checkpoint_path(path)
        model = cls(config)
        if path and len(str(path)) > 0:
            try:
                sd = torch.load(path, map_location='cpu', weights_only=True)
            except TypeError:
                sd = torch.load(path, map_location='cpu')
            sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
            sd = model.upgrade_checkpoint_state_dict(sd)
            inc = model.load_state_dict(sd, strict=False)
            missing_frozen = model._validate_checkpoint_incompatibility(inc, path)
            print(
                f"[memnav] loaded {path}: all non-LingBot tensors present; "
                f"omitted frozen LingBot tensors={missing_frozen}"
            )
        return model

    def __init__(self, config: MemNavModelConfig):
        super().__init__(config)
        il = config.model_cfg['il']
        # runtime LOCAL_RANK (set by torchrun) wins over the static config rank, so each
        # DDP rank builds the frozen LingBot + heads on its own GPU.
        local_rank = int(os.getenv('LOCAL_RANK', config.model_cfg.get('local_rank', 0)))
        self._device = torch.device(f"cuda:{local_rank}")
        # frozen-LingBot paths come from the config so HPC can override without code edits
        lingbot_kwargs = {}
        if il.get('lingbot_repo'):    lingbot_kwargs['lingbot_repo'] = il['lingbot_repo']
        if il.get('lingbot_weights'): lingbot_kwargs['weights'] = il['lingbot_weights']
        # memory-partition geometry — MUST match the precompute + dataset (mp3d: 32/8/4096).
        # LingBotStream sets kv_cache_sliding_window=window, so window here == the precompute
        # --kv_cache_sliding_window; max_frame_num sizes the 3D-RoPE table (long 3leg episodes).
        if il.get('window_size') is not None:   lingbot_kwargs['window'] = il['window_size']
        if il.get('num_scale') is not None:     lingbot_kwargs['num_scale'] = il['num_scale']
        if il.get('max_frame_num') is not None: lingbot_kwargs['max_frame_num'] = il['max_frame_num']
        novel_backbone_weights = il.get('novel_backbone_weights')
        if not novel_backbone_weights:
            raise ValueError(
                'model_cfg.il.novel_backbone_weights is required; '
                'set MEMNAV_DINO_WEIGHTS'
            )
        self.core = MemNavNet(
            token_dim=il['token_dim'], heads=il['heads'], predict_size=il['predict_size'],
            temporal_depth=il['temporal_depth'], num_diffusion_iters=il.get('num_diffusion_iters', 10),
            goal_warm=il.get('goal_warm', 64),
            novel_backbone_weights=novel_backbone_weights,
            gate_center=il.get('gate_center', 0.94),
            gate_width=il.get('gate_width', 0.04),
            gate_slope_init=il.get('gate_slope_init', 1.6),
            gate_bias_init=il.get('gate_bias_init', 0.0),
            pose_scale_window=il.get('pose_scale_window', 64),
            pose_reliability_hidden=il.get('pose_reliability_hidden', 16),
            pose_reliability_init=il.get('pose_reliability_init', 0.95),
            use_pose_reliability_conditioning=il.get(
                'use_pose_reliability_conditioning', True
            ),
            use_route_sketch=il.get('use_route_sketch', False),
            route_horizons=il.get('route_horizons', (2, 8, 24)),
            require_versioned_cache=il.get('require_versioned_cache', False),
            lingbot_kwargs=lingbot_kwargs or None, device=str(self._device),
        )

    def forward(self, batch):
        return self.core(batch)

    def prepare_condition(self, batch):
        return self.core.prepare_condition(batch)

    def forward_with_condition(self, batch, condition):
        return self.core.forward_with_condition(batch, condition)

    def sample_actions_from_condition(self, condition, **kwargs):
        return self.core.sample_actions_from_condition(condition, **kwargs)

    def upgrade_checkpoint_state_dict(self, state_dict):
        """Upgrade legacy gate tensors and optional zero-residual route state."""
        upgraded = self.core.retrieval.upgrade_legacy_state_dict(
            state_dict, prefix='core.retrieval.', copy=True
        )
        if self.core.route_sketch is not None:
            current = self.state_dict()
            prefix = 'core.route_sketch.'
            route_keys = {
                key: value for key, value in current.items()
                if key.startswith(prefix)
            }
            present = [key for key in upgraded if key.startswith(prefix)]
            if not present:
                # A genuinely legacy checkpoint has no route namespace at all.
                # Its newly constructed residual scale is exactly zero, so this
                # migration is behavior preserving.
                for key, value in route_keys.items():
                    upgraded[key] = value.detach().cpu().clone()
            else:
                # Do not repair a partially written new checkpoint.  The normal
                # strict missing-key check will reject it.  Validate horizon
                # semantics here because equal-length horizon sets have equal
                # tensor shapes and would otherwise load silently.
                horizon_key = f'{prefix}horizon_code'
                if horizon_key in upgraded:
                    incoming = upgraded[horizon_key].detach().cpu()
                    expected = route_keys[horizon_key].detach().cpu()
                    if (
                        incoming.shape != expected.shape
                        or not torch.allclose(incoming, expected, atol=0.0, rtol=0.0)
                    ):
                        raise ValueError(
                            'Route-sketch horizons do not match the checkpoint'
                        )
        return upgraded


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run encode_memory on a real batch (needs GPU + cache)")
    args = ap.parse_args()

    B, L, D, P = 4, 60, 1024, 1369
    # retrieval smoke: ranking logits over candidates + affine gate on max-cos
    rh = RetrievalHead()
    goal_cls = torch.randn(B, D)
    mem_cls = torch.randn(B, L, D)
    cand_mask = torch.ones(B, L, dtype=torch.bool)
    cand_mask[0, 40:] = False  # sample 0: fewer candidates
    cand_mask[1, :] = False    # sample 1: no candidate -> novel (gate floor)
    m, gate_logit, logits, gate_feature = rh(goal_cls, mem_cls, cand_mask)
    gate = torch.sigmoid(gate_logit)
    print(f"RetrievalHead: match_idx={m.tolist()} gate={[round(x,3) for x in gate.tolist()]} logits={tuple(logits.shape)}")
    # decoupled losses: InfoNCE (ranking) + BCE (gate)
    pos = torch.zeros(B, L, dtype=torch.bool); pos[[0, 2, 3], [12, 5, 33]] = True
    neg = cand_mask & ~pos
    rows = torch.tensor([0, 2, 3])
    floor = torch.finfo(logits.dtype).min
    lse_pn = logits.masked_fill(~(pos | neg), floor).logsumexp(-1)
    lse_p = logits.masked_fill(~pos, floor).logsumexp(-1)
    rank = (lse_pn[rows] - lse_p[rows]).mean()
    is_rev = torch.tensor([1.0, 0.0, 1.0, 1.0])
    gate_ce = F.binary_cross_entropy_with_logits(gate_logit, is_rev)
    print(f"  rank InfoNCE={rank.item():.3f}  gate BCE={gate_ce.item():.3f}  "
          f"grad ok={torch.autograd.grad(rank + gate_ce, rh.log_temp, retain_graph=True)[0] is not None}")

    # novel branch smoke (early fusion on raw images)
    dino_weights = os.environ.get('MEMNAV_DINO_WEIGHTS')
    nb = NovelBranch(pretrained_checkpoint=dino_weights, device="cuda").to("cuda")
    cur_img = torch.rand(B, 3, 518, 518, device="cuda")
    goal_img = torch.rand(B, 3, 518, 518, device="cuda")
    out = nb(cur_img, goal_img)
    print(f"NovelBranch: out={tuple(out.shape)} params={sum(p.numel() for p in nb.parameters())/1e6:.2f}M")

    if args.full:
        import sys
        sys.path.insert(0, "/home/asus/Research/Nav/InternNav")
        from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
        ds = MemNav_Dataset("/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data", predict_size=24)
        batch = memnav_collate_fn([ds[i] for i in range(2)])
        net = MemNavNet(novel_backbone_weights=dino_weights, device="cuda")
        out = net.encode_memory(batch)
        print("\nencode_memory readouts:")
        for key, v in out.items():
            if torch.is_tensor(v):
                print(f"  {key}: {tuple(v.shape)} {v.dtype} req_grad={v.requires_grad}")
        print(f"  cur_steps={batch['cur_steps']} goal_steps={batch['goal_steps']} match_idx={out['match_idx'].tolist()}")
        cs = net.build_current_state(out["current"], out["depth_feat"])
        nov = net.novel(batch["batch_window_images"][:, -1].to(net.device), batch["batch_goal_image"].to(net.device))
        rr, ap, _R_rel, *_pose_diag = net.build_revisit(
            out["cur_pose"], out["goal_pose"], out.get("pose_context")
        )
        print(f"  current_state (RGBD+depth Perceiver): {tuple(cs.shape)} req_grad={cs.requires_grad}")
        print(f"  novel readout: {tuple(nov.shape)} req_grad={nov.requires_grad}")
        print(f"  revisit_readout: {tuple(rr.shape)} | aux_pose: {tuple(ap.shape)} req_grad={rr.requires_grad}")

        fwd = net(batch)
        print("\nforward outputs:")
        for key, v in fwd.items():
            print(f"  {key}: {tuple(v.shape)} {v.dtype}")
        loss = ((fwd["noise_pred"] - fwd["noise"]).square().mean()
                + fwd["aux_pose"].square().mean())
        loss.backward()
        n_grad = sum(1 for p in net.parameters() if p.requires_grad and p.grad is not None)
        n_train = sum(1 for p in net.parameters() if p.requires_grad)
        print(f"  dummy loss={loss.item():.3f}; params w/ grad={n_grad}/{n_train} trainable")
