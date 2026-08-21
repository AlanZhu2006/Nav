"""Fail-closed runtime for the frozen Pi3X spatial-proof ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from MemNavData.pi3x_spatial_reliability_model import (
    Pi3XSpatialReliabilityHead,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialProofDecision:
    status: str
    selected_candidate: int | None
    member_scores: tuple[float, ...]
    member_votes: int
    consensus_required: int
    scale_free_bearing_forward_left: tuple[float, float] | None
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class Pi3XSpatialProofEnsemble:
    """Select by Pi3X overlap, then authorize with learned spatial evidence."""

    def __init__(self, manifest_path: Path, *, device: str = "cpu") -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported deployment manifest")
        authorization = manifest["authorization"]
        self.consensus = int(authorization["consensus_numerator"])
        denominator = int(authorization["consensus_denominator"])
        members = manifest["members"]
        if len(members) != denominator or not 1 <= self.consensus <= denominator:
            raise ValueError("invalid deployment consensus")
        config = manifest["model"]
        self.device = torch.device(device)
        self.models = []
        self.thresholds = []
        for expected_member, member in enumerate(members):
            if int(member["member"]) != expected_member:
                raise ValueError("deployment members are not ordered")
            checkpoint_path = self.manifest_path.parent / member["checkpoint"]
            if _sha256(checkpoint_path) != member["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            checkpoint_config = checkpoint["model_config"]
            expected_config = {
                "descriptor_dim": int(config["descriptor_dim"]),
                "model_dim": int(config["model_dim"]),
                "layers": int(config["layers"]),
                "heads": int(config["heads"]),
            }
            if checkpoint_config != expected_config:
                raise ValueError("checkpoint model config differs from manifest")
            if not math.isclose(
                float(checkpoint["threshold"]), float(member["threshold"]),
                rel_tol=0.0, abs_tol=0.0,
            ):
                raise ValueError("checkpoint threshold differs from manifest")
            model = Pi3XSpatialReliabilityHead(
                expected_config["descriptor_dim"],
                model_dim=expected_config["model_dim"],
                layers=expected_config["layers"],
                heads=expected_config["heads"],
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.models.append(model.to(self.device).eval())
            self.thresholds.append(float(member["threshold"]))

    @staticmethod
    def _validate_inputs(
        overlaps: np.ndarray,
        bearings: np.ndarray,
        descriptors: np.ndarray,
        roles: np.ndarray,
        relative_age: np.ndarray,
        valid: np.ndarray,
        world_points: np.ndarray,
        local_points: np.ndarray,
        confidence: np.ndarray,
        poses: np.ndarray,
    ) -> None:
        candidates = len(overlaps)
        if candidates == 0 or bearings.shape != (candidates, 2):
            raise ValueError("candidate overlaps/bearings have invalid shape")
        views = descriptors.shape[1] if descriptors.ndim == 3 else -1
        if roles.shape != (candidates, views):
            raise ValueError("roles have invalid shape")
        if relative_age.shape != roles.shape or valid.shape != roles.shape:
            raise ValueError("view metadata shapes differ")
        if world_points.ndim != 5 or world_points.shape[:2] != (candidates, views):
            raise ValueError("world point grid has invalid shape")
        if local_points.shape != world_points.shape or world_points.shape[-1] != 3:
            raise ValueError("local/world point grids differ")
        if confidence.shape != (*world_points.shape[:-1], 1):
            raise ValueError("confidence grid has invalid shape")
        if poses.shape != (candidates, views, 3, 4):
            raise ValueError("relative poses have invalid shape")
        numeric = (
            overlaps, bearings, descriptors, relative_age, world_points,
            local_points, confidence, poses,
        )
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError("runtime evidence contains non-finite values")

    @torch.inference_mode()
    def decide(
        self,
        *,
        overlaps: Sequence[float],
        bearings_forward_left: Sequence[Sequence[float]],
        descriptors: np.ndarray,
        roles: np.ndarray,
        relative_age: np.ndarray,
        valid: np.ndarray,
        world_points_in_current: np.ndarray,
        local_points: np.ndarray,
        confidence: np.ndarray,
        poses_in_current: np.ndarray,
    ) -> SpatialProofDecision:
        try:
            overlap_array = np.asarray(overlaps, dtype=np.float32)
            bearing_array = np.asarray(bearings_forward_left, dtype=np.float32)
            arrays = tuple(np.asarray(value) for value in (
                descriptors, roles, relative_age, valid,
                world_points_in_current, local_points, confidence,
                poses_in_current,
            ))
            self._validate_inputs(overlap_array, bearing_array, *arrays)
            selected = int(np.argmax(overlap_array))
            chosen = slice(selected, selected + 1)
            tensors = (
                torch.as_tensor(arrays[0][chosen], device=self.device, dtype=torch.float32),
                torch.as_tensor(arrays[1][chosen], device=self.device, dtype=torch.long),
                torch.as_tensor(arrays[2][chosen], device=self.device, dtype=torch.float32),
                torch.as_tensor(arrays[3][chosen], device=self.device, dtype=torch.bool),
                torch.as_tensor(arrays[4][chosen], device=self.device, dtype=torch.float32),
                torch.as_tensor(arrays[5][chosen], device=self.device, dtype=torch.float32),
                torch.as_tensor(arrays[6][chosen], device=self.device, dtype=torch.float32),
                torch.as_tensor(arrays[7][chosen], device=self.device, dtype=torch.float32),
            )
            scores = tuple(float(torch.sigmoid(model(*tensors)[0])[0]) for model in self.models)
            votes = sum(score >= threshold for score, threshold in zip(scores, self.thresholds))
            if votes < self.consensus:
                return SpatialProofDecision(
                    "abstain", selected, scores, votes, self.consensus, None,
                    "learned_spatial_proof_below_consensus_native_fallback",
                )
            bearing = bearing_array[selected].astype(np.float64)
            norm = float(np.linalg.norm(bearing))
            if not math.isfinite(norm) or norm <= 1e-9:
                return SpatialProofDecision(
                    "error", selected, scores, votes, self.consensus, None,
                    "invalid_selected_bearing_native_fallback",
                )
            unit = bearing / norm
            return SpatialProofDecision(
                "accepted", selected, scores, votes, self.consensus,
                (float(unit[0]), float(unit[1])),
                "learned_spatial_proof_accepted",
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return SpatialProofDecision(
                "error", None, (), 0, self.consensus, None,
                f"invalid_spatial_evidence_native_fallback:{type(error).__name__}",
            )
