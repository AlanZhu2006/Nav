"""MicKey shadow adapter for the unified learned-relocalizer contract.

The Niantic reference repository remains an immutable external dependency.
This adapter loads its model lazily, records raw relative poses, and never
authorizes a navigation action.  Labels are not accepted by the inference
API; evaluation joins them after predictions have been frozen.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from MemNavData.learned_relocalizer_contract import LearnedPairPrediction


MICKEY_OFFICIAL_COMMIT = "2391be8a35491e7b43481c069f5dab65030839b9"


def scaled_intrinsic(
    intrinsic: Sequence[Sequence[float]],
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> np.ndarray:
    """Scale a 3x3 intrinsic from ``(width, height)`` to a new image size."""

    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    source_width, source_height = map(int, source_size)
    destination_width, destination_height = map(int, destination_size)
    if min(source_width, source_height,
           destination_width, destination_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale_x = destination_width / source_width
    scale_y = destination_height / source_height
    output = matrix.copy()
    output[0] *= scale_x
    output[1] *= scale_y
    output[2] = [0.0, 0.0, 1.0]
    return output.astype(np.float32)


def read_rgb_tensor(path: Path, destination_size: tuple[int, int], torch):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_size = (int(image.shape[1]), int(image.shape[0]))
    if source_size != destination_size:
        image = cv2.resize(image, destination_size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    return tensor, source_size


@contextmanager
def _local_dino_loader(torch, dino_weights: Path):
    """Prevent hidden network access while the official model is constructed."""

    original = torch.hub.load_state_dict_from_url
    state = torch.load(
        dino_weights, map_location="cpu", weights_only=True)

    def load_local(*_args, **_kwargs):
        return state

    torch.hub.load_state_dict_from_url = load_local
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original


class MicKeyShadowAdapter:
    """Thin, model-neutral wrapper around the frozen official MicKey model."""

    def __init__(
        self,
        *,
        repository: Path,
        python_dependencies: Path,
        config: Path,
        checkpoint: Path,
        dino_weights: Path,
        device: str = "cuda:0",
        resize: tuple[int, int] = (476, 266),
    ) -> None:
        for name, path in (("repository", repository),
                           ("python_dependencies", python_dependencies),
                           ("config", config),
                           ("checkpoint", checkpoint),
                           ("dino_weights", dino_weights)):
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        if min(map(int, resize)) <= 0 or any(int(value) % 14 for value in resize):
            raise ValueError("MicKey resize must be positive and divisible by 14")

        self.repository = Path(repository).resolve()
        self.resize = tuple(map(int, resize))
        self.model_id = f"mickey@{MICKEY_OFFICIAL_COMMIT[:8]}"

        for path in (str(Path(python_dependencies).resolve()),
                     str(self.repository)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import torch
        from config.default import cfg
        from lib.models.MicKey.compute_pose import MickeyRelativePose

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        cfg.defrost()
        cfg.merge_from_file(str(Path(config).resolve()))
        with _local_dino_loader(torch, Path(dino_weights).resolve()):
            model = MickeyRelativePose(cfg)
        # PyTorch >=2.6 defaults to weights_only=True, while the official
        # Lightning checkpoint contains ordinary configuration objects.
        checkpoint_value = torch.load(
            Path(checkpoint).resolve(), map_location="cpu", weights_only=False)
        model.on_load_checkpoint(checkpoint_value)
        model.load_state_dict(checkpoint_value["state_dict"])
        model = model.to(self.device).eval()
        model.is_eval_model(True)
        self.model = model
        self.maximum_solver_support = float(
            model.e2e_Procrustes.num_samples_matches)

    def infer(
        self,
        reference_paths: Sequence[Path],
        query_paths: Sequence[Path],
        reference_intrinsics: Sequence[Sequence[Sequence[float]]],
        query_intrinsics: Sequence[Sequence[Sequence[float]]],
        *,
        seed: int,
    ) -> list[LearnedPairPrediction]:
        if not (len(reference_paths) == len(query_paths)
                == len(reference_intrinsics) == len(query_intrinsics)):
            raise ValueError("MicKey batch inputs must have equal lengths")
        if not reference_paths:
            return []
        torch = self.torch
        references = []
        queries = []
        scaled_references = []
        scaled_queries = []
        for reference_path, query_path, reference_k, query_k in zip(
                reference_paths, query_paths,
                reference_intrinsics, query_intrinsics):
            reference, reference_size = read_rgb_tensor(
                Path(reference_path), self.resize, torch)
            query, query_size = read_rgb_tensor(
                Path(query_path), self.resize, torch)
            references.append(reference)
            queries.append(query)
            scaled_references.append(scaled_intrinsic(
                reference_k, reference_size, self.resize))
            scaled_queries.append(scaled_intrinsic(
                query_k, query_size, self.resize))

        data = {
            "image0": torch.stack(references).to(self.device),
            "image1": torch.stack(queries).to(self.device),
            "K_color0": torch.from_numpy(
                np.stack(scaled_references)).to(self.device),
            "K_color1": torch.from_numpy(
                np.stack(scaled_queries)).to(self.device),
        }
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            self.model(data, return_inliers=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        latency_ms = 1000.0 * (time.perf_counter() - started) / len(references)

        rotations = data["R"].detach().float().cpu().numpy()
        translations = data["t"].detach().float().cpu().numpy().reshape(-1, 3)
        supports = data["inliers"].detach().float().cpu().numpy().reshape(-1)
        outputs: list[LearnedPairPrediction] = []
        for rotation, translation, support in zip(
                rotations, translations, supports):
            valid = (np.isfinite(rotation).all()
                     and np.isfinite(translation).all()
                     and np.isfinite(support)
                     and np.linalg.norm(translation) > 1e-9
                     and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                     and math.isclose(
                         float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3))
            if valid:
                prediction = LearnedPairPrediction(
                    model_id=self.model_id,
                    status="ok",
                    rotation_reference_to_query=rotation,
                    translation_reference_to_query_m=translation,
                    support_score=float(np.clip(
                        support / self.maximum_solver_support, 0.0, 1.0)),
                    solver_support=float(support),
                    latency_ms=latency_ms,
                    reason="pose_estimated",
                )
            else:
                prediction = LearnedPairPrediction(
                    model_id=self.model_id,
                    status="abstain",
                    rotation_reference_to_query=None,
                    translation_reference_to_query_m=None,
                    support_score=(
                        float(support) if np.isfinite(support) else None),
                    solver_support=(
                        float(support) if np.isfinite(support) else None),
                    latency_ms=latency_ms,
                    reason="invalid_or_degenerate_pose",
                )
            outputs.append(prediction.validated())
        return outputs


__all__ = [
    "MICKEY_OFFICIAL_COMMIT",
    "MicKeyShadowAdapter",
    "read_rgb_tensor",
    "scaled_intrinsic",
]
