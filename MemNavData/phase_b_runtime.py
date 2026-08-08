"""Deployment-time Phase-B candidate ranking and LingBot geometry features.

This module deliberately exposes ranking, not routing.  The learned model may
order a fixed causal DINO shortlist, but its candidate-validity and no-match
heads are diagnostics only.  The online controller must still use the frozen
SIFT/RANSAC verifier to decide whether memory is activated.

The geometry path mirrors ``diag_lingbot_goal_loop_closure.py`` with the exact
configuration used by the repaired Phase-B development artifact (top-8,
offset 0, full replay, stride 10, confidence quantile 0.5, 768 points, overlap
ratio 0.025).  It contains no teacher, Habitat pose, navmesh, or target input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from MemNavData.phase_b_feature_schema import (
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        EXTERNAL_SCALE_QUALITY_COLUMNS,
        FEATURE_DIMENSION,
        FEATURE_NAMES,
        METRIC_SCALE_SOURCES,
        SCALAR_INPUT_COLUMNS,
        validate_checkpoint_metadata,
    )
    from MemNavData.phase_b_model import LingBotNativeLocalizer
except ModuleNotFoundError:  # direct script invocation
    from phase_b_feature_schema import (  # type: ignore
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        EXTERNAL_SCALE_QUALITY_COLUMNS,
        FEATURE_DIMENSION,
        FEATURE_NAMES,
        METRIC_SCALE_SOURCES,
        SCALAR_INPUT_COLUMNS,
        validate_checkpoint_metadata,
    )
    from phase_b_model import LingBotNativeLocalizer  # type: ignore


class PhaseBRuntimeError(RuntimeError):
    """Raised when the experimental ranker cannot honor its frozen ABI."""


@dataclass(frozen=True)
class PhaseBRuntimeConfig:
    """Frozen feature-extraction settings for the v3 Phase-B checkpoint."""

    candidate_top_k: int = 8
    candidate_min_gap: int = 16
    full_replay: bool = True
    neighbor_offset: int = 0
    pixel_stride: int = 10
    confidence_quantile: float = 0.5
    max_points: int = 768
    overlap_ratio: float = 0.025
    scale_prefix_frame_cap: int = 64
    scale_confidence_quantile: float = 0.5
    scale_pixel_stride: int = 4
    scale_histogram_bins: int = 60
    scale_peak_threshold: float = 0.3
    scale_bias_correction: float = 1.15
    scale_min: float = 0.8
    scale_max: float = 6.0

    def validate(self) -> None:
        if (
            self.candidate_top_k != 8
            or self.candidate_min_gap != 16
            or self.full_replay is not True
            or self.neighbor_offset != 0
            or self.pixel_stride != 10
            or self.confidence_quantile != 0.5
            or self.max_points != 768
            or self.overlap_ratio != 0.025
            or self.scale_prefix_frame_cap != 64
            or self.scale_confidence_quantile != 0.5
            or self.scale_pixel_stride != 4
            or self.scale_histogram_bins != 60
            or self.scale_peak_threshold != 0.3
            or self.scale_bias_correction != 1.15
            or self.scale_min != 0.8
            or self.scale_max != 6.0
        ):
            raise PhaseBRuntimeError(
                "Phase-B runtime configuration differs from the audited v3 "
                "feature contract"
            )


RUNTIME_CONFIG = PhaseBRuntimeConfig()
RUNTIME_CONFIG.validate()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise PhaseBRuntimeError(f"{name} must be numeric, not Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PhaseBRuntimeError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise PhaseBRuntimeError(f"{name} must be finite")
    return result


def build_feature_vector(row: Mapping[str, object]) -> np.ndarray:
    """Construct one exact 20-D v3 vector from deployment-only values."""

    values: dict[str, float] = {}
    for name in SCALAR_INPUT_COLUMNS:
        if name not in row:
            raise PhaseBRuntimeError(f"Phase-B feature is missing: {name}")
        values[name] = _finite_float(row[name], name)

    predicted_xy = np.asarray(row.get("predicted_relative_xy_m"), dtype=np.float64)
    if predicted_xy.shape != (2,) or not np.isfinite(predicted_xy).all():
        raise PhaseBRuntimeError(
            "predicted_relative_xy_m must be a finite length-two vector"
        )
    values.update({
        "lingbot_predicted_forward_m": float(predicted_xy[0]),
        "lingbot_predicted_lateral_m": float(predicted_xy[1]),
        "lingbot_predicted_distance_m": float(np.linalg.norm(predicted_xy)),
    })

    source = str(row.get("metric_scale_source", ""))
    for category in METRIC_SCALE_SOURCES:
        values[f"metric_scale_source={category}"] = float(source == category)
    values["metric_scale_source=other"] = float(
        source not in METRIC_SCALE_SOURCES
    )

    for name in EXTERNAL_SCALE_QUALITY_COLUMNS:
        if name not in row:
            raise PhaseBRuntimeError(f"Phase-B feature is missing: {name}")
        values[name] = _finite_float(row[name], name)
    valid_ratio = values["external_scale_valid_frame_ratio"]
    relative_iqr = values["external_scale_relative_h_iqr"]
    clamped = values["external_scale_clamped"]
    if not 0.0 < valid_ratio <= 1.0:
        raise PhaseBRuntimeError("external scale valid-frame ratio is invalid")
    if relative_iqr < 0.0:
        raise PhaseBRuntimeError("external scale relative h-IQR is invalid")
    if clamped not in (0.0, 1.0):
        raise PhaseBRuntimeError("external scale clamp flag is invalid")

    try:
        vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    except KeyError as error:
        raise PhaseBRuntimeError(
            f"runtime did not construct ABI feature {error.args[0]}"
        ) from error
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        raise PhaseBRuntimeError("constructed Phase-B feature vector is invalid")
    return vector


class PhaseBEnsembleRanker:
    """Strict CPU loader for the experimental Phase-B ensemble checkpoint."""

    def __init__(self, checkpoint: Path, *, allow_unapproved: bool = False):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file() or self.checkpoint.is_symlink():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        try:
            artifact = torch.load(
                self.checkpoint, map_location="cpu", weights_only=False
            )
        except TypeError:  # older torch without weights_only
            artifact = torch.load(self.checkpoint, map_location="cpu")
        if not isinstance(artifact, Mapping):
            raise PhaseBRuntimeError("Phase-B checkpoint is not a mapping")
        validate_checkpoint_metadata(
            artifact, require_deployment_input_contract=True
        )
        self.deployment_approved = artifact.get("deployment_approved") is True
        self.allow_unapproved = bool(allow_unapproved)
        if not self.deployment_approved and not self.allow_unapproved:
            raise PhaseBRuntimeError(
                "Phase-B checkpoint is not deployment-approved; P0 experiments "
                "must pass the explicit allow_unapproved flag"
            )

        config = artifact.get("config")
        states = artifact.get("states")
        if not isinstance(config, Mapping):
            raise PhaseBRuntimeError("Phase-B checkpoint config is missing")
        if (
            not isinstance(states, Sequence)
            or isinstance(states, (str, bytes))
            or not states
        ):
            raise PhaseBRuntimeError("Phase-B checkpoint ensemble is empty")
        hidden_dim = int(config.get("hidden_dim", 64))
        if hidden_dim < 1:
            raise PhaseBRuntimeError("Phase-B hidden dimension is invalid")
        self.models = []
        for index, state in enumerate(states):
            if not isinstance(state, Mapping):
                raise PhaseBRuntimeError(
                    f"Phase-B ensemble state {index} is malformed"
                )
            model = LingBotNativeLocalizer(
                int(artifact["input_dim"]), hidden_dim=hidden_dim, dropout=0.0
            )
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise PhaseBRuntimeError(
                    f"Phase-B ensemble state {index} violates the model ABI"
                ) from error
            model.eval()
            self.models.append(model)

        self.mean = torch.as_tensor(
            artifact["normalization_mean"], dtype=torch.float32
        )
        self.scale = torch.as_tensor(
            artifact["normalization_scale"], dtype=torch.float32
        )
        if self.mean.shape != (FEATURE_DIMENSION,) or self.scale.shape != (
            FEATURE_DIMENSION,
        ):
            raise PhaseBRuntimeError("Phase-B normalization shape is invalid")
        self.model_kind = str(artifact["model_kind"])
        self.feature_schema_version = str(artifact["feature_schema_version"])
        self.input_contract_approved = (
            artifact.get("deployment_input_contract_approved") is True
        )

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "ensemble_members": len(self.models),
            "deployment_approved": self.deployment_approved,
            "deployment_input_contract_approved": self.input_contract_approved,
            "allow_unapproved": self.allow_unapproved,
            "feature_schema_version": self.feature_schema_version,
            "activation_semantics": "diagnostic_only_geometry_gate_unchanged",
            "runtime_config": asdict(RUNTIME_CONFIG),
        }

    @torch.no_grad()
    def rank(self, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not rows:
            raise PhaseBRuntimeError("Phase-B cannot rank an empty shortlist")
        if len(rows) > RUNTIME_CONFIG.candidate_top_k:
            raise PhaseBRuntimeError(
                f"Phase-B shortlist exceeds top-{RUNTIME_CONFIG.candidate_top_k}"
            )
        raw = np.stack([build_feature_vector(row) for row in rows])
        features = (torch.from_numpy(raw) - self.mean) / self.scale
        features = features.unsqueeze(0)
        mask = torch.ones((1, len(rows)), dtype=torch.bool)
        rank_sum = torch.zeros(len(rows), dtype=torch.float64)
        validity_sum = torch.zeros(len(rows), dtype=torch.float64)
        no_match_sum = 0.0
        for model in self.models:
            logits, no_match_logit, _residual, _log_variance = model(
                features, mask
            )
            rank_sum += torch.softmax(logits[0], dim=-1).double()
            validity_sum += torch.sigmoid(logits[0]).double()
            no_match_sum += float(torch.sigmoid(no_match_logit)[0].item())
        rank_probability = (rank_sum / len(self.models)).numpy()
        candidate_validity = (validity_sum / len(self.models)).numpy()
        order = np.argsort(-rank_probability, kind="stable")
        return {
            "order": [int(index) for index in order],
            "rank_probability": rank_probability.tolist(),
            "candidate_validity": candidate_validity.tolist(),
            "no_match_probability_diagnostic": no_match_sum / len(self.models),
            "activation_uses_model_score": False,
        }


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise PhaseBRuntimeError("quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise PhaseBRuntimeError("quaternion is non-finite or degenerate")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lingbot_relative_prediction(
    anchor_pose9: np.ndarray, goal_pose9: np.ndarray, metric_scale: float
) -> np.ndarray:
    """Decode LingBot translation as NavDP [forward, left] metres."""

    anchor = np.asarray(anchor_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    if anchor.shape != (9,) or goal.shape != (9,):
        raise PhaseBRuntimeError("LingBot pose encodings must have shape (9,)")
    metric_scale = _finite_float(metric_scale, "metric scale")
    if metric_scale <= 0.0:
        raise PhaseBRuntimeError("metric scale must be positive")
    anchor_rotation = quaternion_xyzw_to_matrix(anchor[3:7])
    translation = anchor_rotation.T @ (goal[:3] - anchor[:3])
    return metric_scale * np.array(
        [translation[2], -translation[0]], dtype=np.float64
    )


def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> float:
    q1 = torch.nn.functional.normalize(q1.float(), dim=-1)
    q2 = torch.nn.functional.normalize(q2.float(), dim=-1)
    cosine = torch.sum(q1 * q2).abs().clamp(0.0, 1.0)
    return float(2.0 * torch.acos(cosine))


@torch.no_grad()
def world_cloud(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    pose9: torch.Tensor,
    *,
    pixel_stride: int,
    confidence_quantile: float,
    max_points: int,
) -> tuple[torch.Tensor, float]:
    from lingbot_map.utils.rotation import quat_to_mat

    depth = depth.float()
    confidence = confidence.float()
    pose9 = pose9.float()
    height, width = depth.shape
    d = depth[::pixel_stride, ::pixel_stride]
    c = confidence[::pixel_stride, ::pixel_stride]
    ys = torch.arange(
        0, height, pixel_stride, device=depth.device, dtype=torch.float32
    )
    xs = torch.arange(
        0, width, pixel_stride, device=depth.device, dtype=torch.float32
    )
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    fy = (height / 2.0) / torch.tan(pose9[7] / 2.0)
    fx = (width / 2.0) / torch.tan(pose9[8] / 2.0)
    cam_x = (x - width / 2.0) * d / fx
    cam_y = (y - height / 2.0) * d / fy
    points = torch.stack([cam_x, cam_y, d], dim=-1)
    threshold = torch.quantile(c.reshape(-1), confidence_quantile)
    valid = (
        torch.isfinite(d)
        & (d > 1e-6)
        & torch.isfinite(c)
        & (c >= threshold)
    )
    points = points[valid]
    if points.shape[0] > max_points:
        indices = torch.linspace(
            0, points.shape[0] - 1, max_points, device=points.device
        ).round().long()
        points = points[indices]
    rotation = quat_to_mat(
        torch.nn.functional.normalize(pose9[3:7], dim=-1)
    )
    points = points @ rotation.transpose(0, 1) + pose9[:3]
    confidence_mean = float(c[valid].mean()) if valid.any() else float("nan")
    return points, confidence_mean


@torch.no_grad()
def symmetric_cloud_overlap(
    first: torch.Tensor, second: torch.Tensor, threshold: float
) -> tuple[float, float, float]:
    if not len(first) or not len(second):
        return float("nan"), float("nan"), float("nan")
    distance = torch.cdist(first, second)
    forward = float((distance.min(dim=1).values <= threshold).float().mean())
    backward = float((distance.min(dim=0).values <= threshold).float().mean())
    harmonic = 2.0 * forward * backward / max(forward + backward, 1e-12)
    return forward, backward, harmonic


@torch.no_grad()
def append_goal_geometry(
    lb,
    cache: Mapping[str, object],
    rgb_dir: Path,
    goal_image: torch.Tensor,
    anchor: int,
    *,
    config: PhaseBRuntimeConfig = RUNTIME_CONFIG,
) -> dict[str, object]:
    """Full-replay offset-0 geometry measurement for one causal candidate."""

    config.validate()
    scale = int(lb.num_scale)
    if not config.full_replay:
        raise PhaseBRuntimeError("Phase-B v3 requires full candidate replay")
    start = scale
    anchor_indices = cache.get("anchor_frame_indices")
    if anchor_indices is None:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            n_hist=0, total_frames=start,
        )
    else:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            anchor_frame_indices=anchor_indices, raw_start=start,
        )

    paths = [Path(rgb_dir) / f"{index}.jpg" for index in range(start, anchor + 1)]
    if not paths or not all(path.is_file() for path in paths):
        missing = next((path for path in paths if not path.is_file()), Path(rgb_dir))
        raise PhaseBRuntimeError(f"candidate replay frame is missing: {missing}")
    warm_images = lb.load_images([str(path) for path in paths]).to(lb.device)
    candidate_agg = candidate_psi = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(len(warm_images)):
            candidate_agg, candidate_psi = lb.model._aggregate_features(
                warm_images[index:index + 1][None],
                num_frame_for_scale=scale,
                num_frame_per_block=1,
            )
        candidate_depth = lb.model._predict_depth(
            candidate_agg, warm_images[-1:][None], candidate_psi
        )
        goal_agg, goal_psi = lb.model._aggregate_features(
            goal_image[None, None].to(lb.device),
            num_frame_for_scale=scale,
            num_frame_per_block=1,
        )
        goal_depth = lb.model._predict_depth(
            goal_agg, goal_image[None, None].to(lb.device), goal_psi
        )

    lb._inject_camera(
        cache["cam_k"], cache["cam_v"], anchor + 1,
        cache.get("cam_frame_indices"),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refinement = lb.model.camera_head(
            goal_agg,
            causal_inference=True,
            num_frame_per_block=1,
            num_frame_for_scale=scale,
        )
    poses = [item[0, -1].float() for item in refinement]
    if not poses:
        raise PhaseBRuntimeError("LingBot camera head returned no goal pose")
    goal_pose = poses[-1]
    anchor_pose = cache["cam_pose_enc"][anchor]

    candidate_d = candidate_depth["depth"][0, -1, ..., 0].float()
    candidate_c = candidate_depth["depth_conf"][0, -1].float()
    goal_d = goal_depth["depth"][0, -1, ..., 0].float()
    goal_c = goal_depth["depth_conf"][0, -1].float()
    candidate_cloud, candidate_confidence = world_cloud(
        candidate_d,
        candidate_c,
        anchor_pose,
        pixel_stride=config.pixel_stride,
        confidence_quantile=config.confidence_quantile,
        max_points=config.max_points,
    )
    goal_cloud, goal_confidence = world_cloud(
        goal_d,
        goal_c,
        goal_pose,
        pixel_stride=config.pixel_stride,
        confidence_quantile=config.confidence_quantile,
        max_points=config.max_points,
    )
    positive_depth = torch.cat([
        candidate_d[candidate_d > 1e-6].reshape(-1),
        goal_d[goal_d > 1e-6].reshape(-1),
    ])
    if not len(positive_depth):
        raise PhaseBRuntimeError("LingBot returned no positive candidate/goal depth")
    depth_scale = float(torch.median(positive_depth))
    overlap_threshold = max(1e-4, config.overlap_ratio * depth_scale)
    overlap_forward, overlap_backward, overlap_f1 = symmetric_cloud_overlap(
        candidate_cloud, goal_cloud, overlap_threshold
    )

    if len(poses) < 2:
        raise PhaseBRuntimeError(
            "LingBot camera head cannot produce refinement-consistency features"
        )
    refine_translation = float((poses[-1][:3] - poses[-2][:3]).norm())
    refine_rotation = math.degrees(
        quaternion_angle(poses[-1][3:7], poses[-2][3:7])
    )
    result = {
        "anchor": int(anchor),
        "anchor_pose": anchor_pose.detach().cpu().numpy(),
        "goal_pose": goal_pose.detach().cpu().numpy(),
        "anchor_goal_distance_raw": float(
            (goal_pose[:3] - anchor_pose[:3]).norm()
        ),
        "goal_refine_translation_raw": refine_translation,
        "goal_refine_rotation_deg": refine_rotation,
        "candidate_depth_confidence": candidate_confidence,
        "goal_depth_confidence": goal_confidence,
        "cloud_overlap_candidate_to_goal": overlap_forward,
        "cloud_overlap_goal_to_candidate": overlap_backward,
        "cloud_overlap_f1": overlap_f1,
        "overlap_threshold_raw": overlap_threshold,
        "depth_scale_raw": depth_scale,
    }
    for name, value in result.items():
        if name not in ("anchor", "anchor_pose", "goal_pose"):
            _finite_float(value, name)
    return result


@torch.no_grad()
def external_causal_metric_scale(
    lb,
    rgb_dir: Path,
    cam_pose_enc: torch.Tensor,
    camera_height_m: float,
    causal_prefix_end_exclusive: int,
    *,
    config: PhaseBRuntimeConfig = RUNTIME_CONFIG,
) -> tuple[float, dict[str, float]]:
    """Recompute the audited first-prefix scale and its three quality inputs."""

    config.validate()
    prefix_count = min(
        config.scale_prefix_frame_cap,
        int(causal_prefix_end_exclusive),
        int(len(cam_pose_enc)),
    )
    if prefix_count < int(lb.num_scale):
        raise PhaseBRuntimeError(
            "causal goal prefix does not contain the complete scale block"
        )
    paths = [Path(rgb_dir) / f"{index}.jpg" for index in range(prefix_count)]
    if not all(path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise PhaseBRuntimeError(f"causal scale frame is missing: {missing}")
    scale, debug = lb.compute_metric_scale(
        [str(path) for path in paths],
        cam_pose_enc[:prefix_count],
        camera_height_m=float(camera_height_m),
        conf_quantile=config.scale_confidence_quantile,
        pixel_stride=config.scale_pixel_stride,
        nbins=config.scale_histogram_bins,
        n_frames=prefix_count,
        peak_thresh=config.scale_peak_threshold,
        bias_correction=config.scale_bias_correction,
        scale_range=(config.scale_min, config.scale_max),
        return_debug=True,
    )
    if scale is None or not isinstance(debug, Mapping):
        raise PhaseBRuntimeError("external-causal metric scale is unavailable")
    scale = _finite_float(scale, "metric_scale_m_per_raw")
    ground_h = _finite_float(debug.get("h_est"), "external scale h_est")
    h_iqr = _finite_float(debug.get("h_iqr"), "external scale h_iqr")
    try:
        n_frames = int(debug.get("n_frames"))
        n_valid = int(debug.get("n_valid"))
    except (TypeError, ValueError) as error:
        raise PhaseBRuntimeError(
            "external scale debug frame support is malformed"
        ) from error
    if n_frames != prefix_count or not max(3, n_frames // 8) <= n_valid <= n_frames:
        raise PhaseBRuntimeError(
            "external scale has insufficient or inconsistent frame support"
        )
    unclamped = config.scale_bias_correction * float(camera_height_m) / ground_h
    quality = {
        "external_scale_valid_frame_ratio": n_valid / n_frames,
        "external_scale_relative_h_iqr": h_iqr / ground_h,
        "external_scale_clamped": float(
            not math.isclose(scale, unclamped, rel_tol=1e-6, abs_tol=1e-6)
        ),
    }
    return scale, quality


def measurement_feature_row(
    measurement: Mapping[str, object],
    *,
    dino_cosine: float,
    metric_scale: float,
    scale_quality: Mapping[str, float],
) -> dict[str, object]:
    """Convert one offset-0 measurement into the exact checkpoint inputs."""

    depth_scale = _finite_float(measurement["depth_scale_raw"], "depth scale")
    if depth_scale <= 0.0:
        raise PhaseBRuntimeError("depth scale must be positive")
    goal_pose = np.asarray(measurement["goal_pose"], dtype=np.float64)
    anchor_pose = np.asarray(measurement["anchor_pose"], dtype=np.float64)
    predicted_xy = lingbot_relative_prediction(
        anchor_pose, goal_pose, metric_scale
    )
    row: dict[str, object] = {
        "dino_cosine": _finite_float(dino_cosine, "dino cosine"),
        "metric_scale_m_per_raw": _finite_float(metric_scale, "metric scale"),
        "metric_scale_source": EXTERNAL_CAUSAL_SCALE_SOURCE,
        "depth_scale_raw": depth_scale,
        "cloud_overlap_f1_center": measurement["cloud_overlap_f1"],
        "anchor_goal_distance_norm_center": (
            _finite_float(
                measurement["anchor_goal_distance_raw"],
                "anchor-goal distance",
            )
            / depth_scale
        ),
        "goal_refine_translation_norm_median": (
            _finite_float(
                measurement["goal_refine_translation_raw"],
                "goal refine translation",
            )
            / depth_scale
        ),
        "goal_refine_rotation_deg_median": measurement[
            "goal_refine_rotation_deg"
        ],
        "goal_depth_confidence_mean": measurement["goal_depth_confidence"],
        "candidate_depth_confidence_mean": measurement[
            "candidate_depth_confidence"
        ],
        "predicted_relative_xy_m": predicted_xy,
        **scale_quality,
    }
    # Build once here so feature failures occur next to their geometry source.
    build_feature_vector(row)
    return row


__all__ = [
    "PhaseBEnsembleRanker",
    "PhaseBRuntimeConfig",
    "PhaseBRuntimeError",
    "RUNTIME_CONFIG",
    "append_goal_geometry",
    "build_feature_vector",
    "external_causal_metric_scale",
    "lingbot_relative_prediction",
    "measurement_feature_row",
    "sha256_file",
    "symmetric_cloud_overlap",
    "world_cloud",
]
