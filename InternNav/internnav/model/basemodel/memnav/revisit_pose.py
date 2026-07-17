"""Gauge-invariant planar pose code and geometric reliability for revisits.

Retrieval answers a semantic question (has the goal been seen before).  It does
not answer a different geometric question (is a long-range monocular pose still
trustworthy).  Keeping those probabilities separate lets the policy retain the
benefit of a correct memory match without forcing the diffusion decoder to use a
drifted global translation.

The encoder deliberately consumes relative pose *measurements*, not images or
ground truth.  Its reliability target is supplied by the trainer only during
supervision; inference uses the same observable consistency cues as training.
"""

import math

import torch
import torch.nn as nn


class GaugeInvariantRevisitPose(nn.Module):
    """Encode a LingBot relative pose for planar navigation.

    The navigation code is invariant to a uniform rescaling of LingBot's map:

    * planar bearing uses the corrected ``[z, -x]`` NavDP axis mapping;
    * range is divided by the stream's own robust step scale and compressed with
      ``asinh``;
    * 3-D endpoint rotation is retained for diagnostics but excluded from the
      action token because goal-render yaw is not the path's terminal heading.

    A small head predicts geometric reliability from observable cues.  It is
    initialized close to one so introducing it cannot silently disable the
    revisit branch before its calibration loss has learned anything.
    """

    CODE_VERSION = "gauge_invariant_bearing_reliability_v1"
    RELIABILITY_FEATURES = (
        "range_code",
        "anchor_gap_code",
        "step_scale_drift",
        "goal_anchor_range_code",
        "vertical_ratio",
        "rotation_tilt",
        "semantic_score_z",
    )

    def __init__(
        self,
        distance_unit_steps=32.0,
        max_frame_num=4096,
        reliability_hidden=16,
        reliability_init=0.95,
        condition_on_reliability=True,
    ):
        super().__init__()
        if distance_unit_steps <= 0:
            raise ValueError("distance_unit_steps must be positive")
        if max_frame_num <= 1:
            raise ValueError("max_frame_num must exceed one")
        if not 0.0 < reliability_init < 1.0:
            raise ValueError("reliability_init must be strictly between zero and one")

        self.distance_unit_steps = float(distance_unit_steps)
        self.max_frame_num = int(max_frame_num)
        self.condition_on_reliability = bool(condition_on_reliability)
        self.CODE_VERSION = (
            "gauge_invariant_bearing_reliability_v1"
            if self.condition_on_reliability
            else "gauge_invariant_bearing_diagnostic_reliability_v2"
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(len(self.RELIABILITY_FEATURES), reliability_hidden),
            nn.GELU(),
            nn.Linear(reliability_hidden, 1),
        )
        # Start as the old model (pose trusted) while allowing the supervised
        # quality target to calibrate both the intercept and individual cues.
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.constant_(
            self.reliability_head[-1].bias,
            math.log(reliability_init / (1.0 - reliability_init)),
        )

    @staticmethod
    def _context_value(context, name, like, default):
        if context is None or context.get(name) is None:
            return like.new_full(like.shape, float(default))
        value = context[name].to(device=like.device, dtype=like.dtype)
        if value.shape != like.shape:
            raise ValueError(
                f"pose context {name!r} must have shape {tuple(like.shape)}, "
                f"got {tuple(value.shape)}"
            )
        return value

    def forward(self, t_rel, R_rel, context=None):
        """Return the robust pose code and reliability diagnostics.

        Args:
            t_rel: ``[..., 3]`` goal translation in the current LingBot camera
                frame.
            R_rel: ``[..., 3, 3]`` endpoint camera rotation, used only to form a
                non-planarity cue and returned unchanged by the caller.
            context: Optional dictionary of one scalar per leading item:
                ``step_scale``, ``step_scale_drift``, ``anchor_gap``,
                ``goal_anchor_steps``, and ``semantic_score_z``.
        """
        if t_rel.shape[-1] != 3:
            raise ValueError(f"t_rel must end in 3 values, got {tuple(t_rel.shape)}")
        if R_rel.shape != t_rel.shape[:-1] + (3, 3):
            raise ValueError(
                f"R_rel shape must be {t_rel.shape[:-1] + (3, 3)}, got {tuple(R_rel.shape)}"
            )

        scalar_shape = t_rel.shape[:-1]
        scalar = t_rel[..., 0]
        step_scale = self._context_value(context, "step_scale", scalar, 1.0).clamp_min(1e-6)
        scale_drift = self._context_value(
            context, "step_scale_drift", scalar, 0.0
        ).clamp(0.0, 3.0)
        anchor_gap = self._context_value(context, "anchor_gap", scalar, 0.0).clamp_min(0.0)
        goal_anchor_steps = self._context_value(
            context, "goal_anchor_steps", scalar, 0.0
        ).clamp_min(0.0)
        semantic_score_z = self._context_value(
            context, "semantic_score_z", scalar, 0.0
        ).clamp(-5.0, 5.0)

        # LingBot local camera is OpenCV-like (+z forward, +x right).  NavDP's
        # planar action convention is +x forward, +y left.
        planar = torch.stack((t_rel[..., 2], -t_rel[..., 0]), dim=-1)
        raw_radius = torch.linalg.vector_norm(planar, dim=-1)
        raw_direction = planar / raw_radius.clamp_min(1e-6).unsqueeze(-1)

        range_steps = raw_radius / step_scale
        range_code = torch.asinh(range_steps / self.distance_unit_steps).clamp(max=5.0)
        anchor_gap_code = (
            torch.log1p(anchor_gap) / math.log1p(float(self.max_frame_num))
        ).clamp(max=1.5)
        goal_anchor_range_code = torch.asinh(
            goal_anchor_steps / self.distance_unit_steps
        ).clamp(max=5.0)

        # In an upright stream y is vertical and R_rel is nearly a yaw rotation.
        # These bounded residuals are useful failure cues but never enter the
        # navigation bearing itself.
        vertical_ratio = torch.tanh(
            t_rel[..., 1].abs() / raw_radius.clamp_min(1e-6)
        )
        tilt_sq = (
            R_rel[..., 0, 1].square()
            + R_rel[..., 2, 1].square()
            + R_rel[..., 1, 0].square()
            + R_rel[..., 1, 2].square()
        )
        rotation_tilt = torch.sqrt(0.5 * tilt_sq.clamp_min(0.0)).clamp(max=1.0)

        reliability_features = torch.stack(
            (
                range_code,
                anchor_gap_code,
                scale_drift,
                goal_anchor_range_code,
                vertical_ratio,
                rotation_tilt,
                semantic_score_z / 5.0,
            ),
            dim=-1,
        )
        if reliability_features.shape != scalar_shape + (len(self.RELIABILITY_FEATURES),):
            raise RuntimeError("unexpected reliability feature shape")
        reliability = torch.sigmoid(
            self.reliability_head(reliability_features).squeeze(-1)
        )

        # Reliability remains observable for diagnostics.  It only enters the
        # pose token (and the semantic/geometric AND gate in MemNavNet) when the
        # explicitly opt-in conditioning mode is enabled.
        conditioning_reliability = (
            reliability if self.condition_on_reliability else torch.ones_like(reliability)
        )
        pose_code = torch.cat(
            (
                raw_direction,
                range_code.unsqueeze(-1),
                conditioning_reliability.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "pose_code": pose_code,
            "raw_direction": raw_direction,
            "raw_radius": raw_radius,
            "range_steps": range_steps,
            "range_code": range_code,
            "reliability": reliability,
            "reliability_features": reliability_features,
        }
