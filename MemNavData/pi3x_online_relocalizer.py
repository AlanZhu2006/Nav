"""Online Pi3X b16 relocalization with learned spatial authorization.

This module consumes only the current RGB, a causal on-disk RGB history, a
frozen DINO shortlist, and the ImageGoal.  It deliberately has no simulator
pose, depth, role label, LightGlue/PnP feature or atomic-certificate input.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from MemNavData.pi3x_spatial_proof_runtime import (
    Pi3XSpatialProofEnsemble,
)


FROZEN_BRIDGE_FRAMES = 16
FROZEN_ANCHOR_OFFSETS = (-8, 0, 8)
FROZEN_WIDTH = 224
FROZEN_HEIGHT = 126
FROZEN_POINT_STRIDE = 3
FROZEN_CONFIDENCE_QUANTILE = 0.5
FROZEN_TOP_K = 8
FROZEN_CANDIDATE_MIN_GAP = 4
FROZEN_MINIMUM_ANCHOR = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_bridge_frames(
    anchor: int,
    current_frame: int,
    *,
    bridge_count: int = FROZEN_BRIDGE_FRAMES,
    anchor_offsets: tuple[int, ...] = FROZEN_ANCHOR_OFFSETS,
) -> tuple[list[int], list[int]]:
    """Return the exact b16 current-to-anchor causal bridge."""
    upper = current_frame - 1
    if not 0 <= anchor <= upper:
        raise ValueError("anchor must precede the current frame")
    if bridge_count < 2:
        raise ValueError("bridge_count must be at least two")
    bridge = {
        int(round(value))
        for value in np.linspace(anchor, upper, num=bridge_count)
    }
    support = {
        min(max(anchor + offset, 0), upper) for offset in anchor_offsets
    }
    support.add(anchor)
    bridge.update(support)
    return sorted(bridge, reverse=True), sorted(support)


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (FROZEN_WIDTH, FROZEN_HEIGHT), Image.Resampling.LANCZOS
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _inference_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "float32":
        return None
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if requested != "auto":
        raise ValueError(f"unsupported inference dtype {requested!r}")
    if device.type != "cuda":
        return None
    return (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )


def _filtered_points(
    points: np.ndarray,
    confidence: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    points = points[::FROZEN_POINT_STRIDE, ::FROZEN_POINT_STRIDE].reshape(-1, 3)
    confidence = confidence[
        ::FROZEN_POINT_STRIDE, ::FROZEN_POINT_STRIDE
    ].reshape(-1)
    finite = np.isfinite(points).all(1) & np.isfinite(confidence)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float64)
    cutoff = float(np.quantile(
        confidence[finite], FROZEN_CONFIDENCE_QUANTILE
    ))
    distance = np.linalg.norm(points - center[None], axis=1)
    mask = finite & (confidence >= cutoff)
    mask &= (distance >= 0.15) & (distance <= 10.0)
    return points[mask]


def _directed_nn(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        return np.asarray([math.inf], dtype=np.float64)
    return cKDTree(target).query(source, k=1, workers=-1)[0]


def _best_view_overlap_20cm(
    goal_points: np.ndarray, history_points: Iterable[np.ndarray]
) -> float:
    best = 0.0
    for history in history_points:
        goal_to_history = _directed_nn(goal_points, history)
        history_to_goal = _directed_nn(history, goal_points)
        precision = float(np.mean(goal_to_history <= 0.20))
        recall = float(np.mean(history_to_goal <= 0.20))
        if precision + recall > 0.0:
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def _scale_free_bearing(current_c2w: np.ndarray,
                        goal_c2w: np.ndarray) -> np.ndarray:
    relative = current_c2w[:3, :3].T @ (
        goal_c2w[:3, 3] - current_c2w[:3, 3]
    )
    return np.asarray([relative[2], -relative[0]], dtype=np.float64)


def _spatial_geometry(
    predicted_poses: np.ndarray,
    predicted_points: np.ndarray,
    local_points: np.ndarray,
    confidence: np.ndarray,
    *,
    patch_size: int,
) -> dict[str, np.ndarray | float]:
    poses = np.asarray(predicted_poses, dtype=np.float64)
    world = np.asarray(predicted_points, dtype=np.float64)
    local = np.asarray(local_points, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or world.shape != local.shape or world.ndim != 4
            or world.shape[-1] != 3 or conf.shape != world.shape[:-1]):
        raise ValueError("invalid Pi3X spatial output shapes")
    height, width = world.shape[1:3]
    if height % patch_size or width % patch_size:
        raise ValueError("Pi3X output is not divisible by its patch size")
    center = patch_size // 2
    world = world[:, center::patch_size, center::patch_size]
    local = local[:, center::patch_size, center::patch_size]
    conf = conf[:, center::patch_size, center::patch_size]
    current_from_world = np.linalg.inv(poses[0])
    homogeneous = np.concatenate([
        world, np.ones((*world.shape[:-1], 1), dtype=np.float64)
    ], axis=-1)
    world_in_current = np.einsum(
        "ij,nhwj->nhwi", current_from_world, homogeneous
    )[..., :3]
    poses_in_current = np.einsum("ij,njk->nik", current_from_world, poses)
    current_depth = local[0, ..., 2]
    positive = np.isfinite(current_depth) & (current_depth > 1e-6)
    high_confidence = positive & (conf[0] >= np.median(conf[0]))
    scale_values = current_depth[high_confidence]
    if not len(scale_values):
        scale_values = current_depth[positive]
    if not len(scale_values):
        raise ValueError("Pi3X current view has no positive finite depth")
    scale = float(np.median(scale_values))
    if not math.isfinite(scale) or scale <= 1e-6:
        raise ValueError("invalid Pi3X spatial normalization scale")
    world_in_current /= scale
    local /= scale
    poses_in_current[:, :3, 3] /= scale
    result: dict[str, np.ndarray | float] = {
        "world_points_in_current": world_in_current.astype(np.float16),
        "local_points": local.astype(np.float16),
        "confidence": conf.astype(np.float16),
        "poses_in_current": poses_in_current[:, :3].astype(np.float16),
        "normalization_scale": scale,
    }
    if any(
        not np.isfinite(value).all()
        for value in result.values() if isinstance(value, np.ndarray)
    ):
        raise ValueError("Pi3X spatial evidence became non-finite")
    return result


def pack_candidate_evidence(
    evidence: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not evidence:
        raise ValueError("cannot pack an empty candidate set")
    maximum_views = max(len(item["roles"]) for item in evidence)
    descriptor_dim = int(np.asarray(evidence[0]["descriptors"]).shape[-1])
    point_shape = tuple(np.asarray(
        evidence[0]["world_points_in_current"]
    ).shape[1:])
    if len(point_shape) != 3 or point_shape[-1] != 3:
        raise ValueError("invalid point-grid shape")
    rows = len(evidence)
    descriptors = np.zeros(
        (rows, maximum_views, descriptor_dim), dtype=np.float16
    )
    roles = np.full((rows, maximum_views), -1, dtype=np.int8)
    age = np.zeros((rows, maximum_views), dtype=np.float32)
    valid = np.zeros((rows, maximum_views), dtype=np.bool_)
    world = np.zeros((rows, maximum_views, *point_shape), dtype=np.float16)
    local = np.zeros_like(world)
    confidence = np.zeros(
        (rows, maximum_views, *point_shape[:-1], 1), dtype=np.float16
    )
    poses = np.zeros((rows, maximum_views, 3, 4), dtype=np.float16)
    for index, item in enumerate(evidence):
        count = len(item["roles"])
        item_descriptors = np.asarray(item["descriptors"], dtype=np.float16)
        if item_descriptors.shape != (count, descriptor_dim):
            raise ValueError("candidate descriptor shape differs")
        item_world = np.asarray(
            item["world_points_in_current"], dtype=np.float16
        )
        if item_world.shape != (count, *point_shape):
            raise ValueError("candidate point-grid shape differs")
        descriptors[index, :count] = item_descriptors
        roles[index, :count] = np.asarray(item["roles"], dtype=np.int8)
        age[index, :count] = np.asarray(item["relative_age"], dtype=np.float32)
        valid[index, :count] = True
        world[index, :count] = item_world
        local[index, :count] = np.asarray(item["local_points"], dtype=np.float16)
        confidence[index, :count, ..., 0] = np.asarray(
            item["confidence"], dtype=np.float16
        )
        poses[index, :count] = np.asarray(
            item["poses_in_current"], dtype=np.float16
        )
    return {
        "descriptors": descriptors,
        "roles": roles,
        "relative_age": age,
        "valid": valid,
        "world_points_in_current": world,
        "local_points": local,
        "confidence": confidence,
        "poses_in_current": poses,
    }


class Pi3XOnlineRelocalizer:
    """Frozen Pi3X proposal/bearing plus frozen learned spatial proof."""

    def __init__(
        self,
        *,
        pi3_root: Path,
        snapshot: Path,
        expected_model_sha256: str,
        proof_manifest: Path,
        device: str = "cuda:0",
        inference_dtype: str = "auto",
    ) -> None:
        self.pi3_root = Path(pi3_root).resolve()
        self.snapshot = Path(snapshot).resolve()
        self.model_path = self.snapshot / "model.safetensors"
        self.expected_model_sha256 = str(expected_model_sha256)
        if not self.pi3_root.is_dir() or not self.model_path.is_file():
            raise FileNotFoundError("Pi3 source or model snapshot is missing")
        if _sha256(self.model_path) != self.expected_model_sha256:
            raise ValueError("Pi3X model SHA mismatch")
        if str(self.pi3_root) not in sys.path:
            sys.path.insert(0, str(self.pi3_root))
        from pi3.models.pi3x import Pi3X

        self.device = torch.device(device)
        self.model = Pi3X.from_pretrained(
            str(self.snapshot), local_files_only=True
        ).eval()
        self.model.disable_multimodal()
        self.model = self.model.to(self.device)
        self.dtype = _inference_dtype(self.device, inference_dtype)
        self.proof = Pi3XSpatialProofEnsemble(
            Path(proof_manifest), device=str(self.device)
        )
        self.proof_manifest = Path(proof_manifest).resolve()
        self._captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs) -> None:
            if not inputs:
                raise RuntimeError("Pi3X camera decoder received no hidden input")
            self._captured["hidden"] = inputs[0].detach()

        self._descriptor_hook = self.model.camera_decoder.register_forward_pre_hook(
            capture
        )
        if int(self.model.patch_size) != 14:
            raise ValueError("frozen Pi3X patch size changed")

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": 1,
            "method": "dino_top8_pi3x_b16_spatial_proof_v1",
            "model_sha256": self.expected_model_sha256,
            "proof_manifest": str(self.proof_manifest),
            "proof_manifest_sha256": _sha256(self.proof_manifest),
            "candidate_top_k": FROZEN_TOP_K,
            "candidate_min_gap": FROZEN_CANDIDATE_MIN_GAP,
            "minimum_anchor": FROZEN_MINIMUM_ANCHOR,
            "candidate_lifecycle": "frozen_at_first_goal_query",
            "empty_candidate_semantics": "cached_native_abstention",
            "accepted_anchor_lifecycle": "fixed_after_initial_authorization",
            "bearing_update": "causal_selected_anchor_reinfer_each_replan",
            "bridge_frames": FROZEN_BRIDGE_FRAMES,
            "anchor_offsets": list(FROZEN_ANCHOR_OFFSETS),
            "spatial_proof_member_count": len(self.proof.models),
            "spatial_proof_consensus_required": self.proof.consensus,
            "output": "scale_free_relative_bearing",
            "pointgoal_units": "pi3x_current_camera_direction_only",
            "controller_adapter": "verified_bearing_v1_fixed_2.5m",
            "fallback": "native_imagegoal",
            "certificate_components_consumed": False,
            "simulator_pose_or_depth_consumed": False,
        }

    def _infer_candidate(
        self,
        *,
        rgb_dir: Path,
        current_frame: int,
        anchor: int,
        goal_path: Path,
    ) -> dict[str, Any]:
        frames, support_frames = causal_bridge_frames(anchor, current_frame)
        paths = [
            rgb_dir / f"{current_frame}.jpg",
            *[rgb_dir / f"{frame}.jpg" for frame in frames],
            goal_path,
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Pi3X causal views are missing: {missing}")
        images = torch.stack([_load_rgb(path) for path in paths])[None].to(
            self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        autocast = (
            contextlib.nullcontext()
            if self.dtype is None
            else torch.autocast(
                device_type=self.device.type, dtype=self.dtype
            )
        )
        self._captured.clear()
        with torch.inference_mode(), autocast:
            prediction = self.model(imgs=images)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        hidden = self._captured.pop("hidden", None)
        expected_views = len(paths)
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != expected_views:
            raise RuntimeError("Pi3X hidden-view capture failed")
        registers = int(self.model.patch_start_idx)
        if registers <= 0 or hidden.shape[1] <= registers:
            raise RuntimeError("Pi3X hidden state lacks register/patch tokens")
        descriptors = hidden[:, :registers].mean(dim=1).float().cpu().numpy()
        poses = prediction["camera_poses"][0].float().cpu().numpy()
        points = prediction["points"][0].float().cpu().numpy()
        local = prediction["local_points"][0].float().cpu().numpy()
        confidence = torch.sigmoid(
            prediction["conf"][0, ..., 0]
        ).float().cpu().numpy()
        denominator = max(current_frame - anchor, 1)
        roles = [0, *(2 if frame == anchor else 1 for frame in frames), 3]
        relative_age = [
            0.0,
            *((current_frame - frame) / denominator for frame in frames),
            -1.0,
        ]
        spatial = _spatial_geometry(
            poses, points, local, confidence,
            patch_size=int(self.model.patch_size),
        )
        filtered = [
            _filtered_points(points[index], confidence[index], poses[index, :3, 3])
            for index in range(expected_views)
        ]
        support_indices = [1 + frames.index(frame) for frame in support_frames]
        overlap = _best_view_overlap_20cm(
            filtered[-1], [filtered[index] for index in support_indices]
        )
        bearing = _scale_free_bearing(poses[0], poses[-1])
        del prediction, images
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "anchor": anchor,
            "history_frames": frames,
            "support_frames": support_frames,
            "descriptors": descriptors.astype(np.float16),
            "roles": roles,
            "relative_age": relative_age,
            "bearing": bearing,
            "overlap": float(overlap),
            "inference_ms": elapsed_ms,
            **spatial,
        }

    @staticmethod
    def _canonical_candidates(
        candidates: Sequence[dict[str, Any]], current_frame: int
    ) -> list[dict[str, Any]]:
        if not isinstance(candidates, (list, tuple)):
            raise ValueError("candidates must be a sequence")
        if not 1 <= len(candidates) <= FROZEN_TOP_K:
            raise ValueError("candidate count violates frozen top-k")
        canonical = []
        seen = set()
        for fallback_rank, candidate in enumerate(candidates, start=1):
            anchor = int(candidate["anchor"])
            score = float(candidate["score"])
            dino_rank = int(candidate.get("dino_rank", fallback_rank))
            if (anchor in seen or anchor < FROZEN_MINIMUM_ANCHOR
                    or anchor >= current_frame or not math.isfinite(score)
                    or not 1 <= dino_rank <= FROZEN_TOP_K):
                raise ValueError("candidate violates causal contract")
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
            raise ValueError("candidate shortlist violates frozen temporal gap")
        return canonical

    def relocalize(
        self,
        *,
        rgb_dir: Path,
        current_frame: int,
        candidates: Sequence[dict[str, Any]],
        goal_path: Path,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        canonical = self._canonical_candidates(candidates, current_frame)
        evidence = [
            self._infer_candidate(
                rgb_dir=Path(rgb_dir), current_frame=int(current_frame),
                anchor=item["anchor"], goal_path=Path(goal_path),
            )
            for item in canonical
        ]
        packed = pack_candidate_evidence(evidence)
        decision = self.proof.decide(
            overlaps=[item["overlap"] for item in evidence],
            bearings_forward_left=[item["bearing"] for item in evidence],
            **packed,
        )
        selected_index = decision.selected_candidate
        selected = None if selected_index is None else canonical[selected_index]
        selected_evidence = (
            None if selected_index is None else evidence[selected_index]
        )
        public_candidates = [
            {
                **item,
                "pi3x_overlap": evidence[index]["overlap"],
                "pi3x_scale_free_bearing_forward_left": (
                    evidence[index]["bearing"].tolist()),
                "pi3x_inference_ms": evidence[index]["inference_ms"],
                "history_frame_count": len(evidence[index]["history_frames"]),
                "selected": index == selected_index,
            }
            for index, item in enumerate(canonical)
        ]
        accepted = decision.status == "accepted"
        return {
            "ok": decision.status != "error",
            "accepted": accepted,
            "reason": decision.reason,
            "selected_anchor": (
                None if selected is None else selected["anchor"]
            ),
            "selected_dino_rank": (
                None if selected is None else selected["dino_rank"]
            ),
            "aux_pose": (
                list(decision.scale_free_bearing_forward_left)
                if accepted else None
            ),
            "direction_vector": (
                list(decision.scale_free_bearing_forward_left)
                if accepted else None
            ),
            "pointgoal_units": (
                "pi3x_current_camera_direction_only" if accepted else None
            ),
            "member_scores": list(decision.member_scores),
            "member_votes": decision.member_votes,
            "consensus_required": decision.consensus_required,
            "selected_overlap": (
                None if selected_evidence is None
                else selected_evidence["overlap"]
            ),
            "candidate_count": len(canonical),
            "ranked_candidates": public_candidates,
            "relocalization_ms": 1000.0 * (time.perf_counter() - started),
            "cached": False,
        }
