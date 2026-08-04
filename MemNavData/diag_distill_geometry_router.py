#!/usr/bin/env python3
"""Distill the geometric memory verifier into a selective DINO-CLS router.

This is an offline, scene-disjoint diagnostic.  It never changes the live
router or navigation policy.  The learned head may bypass geometry only in
empirically zero-error confidence tails; uncertain examples retain the current
SIFT/essential-matrix verifier.

The experiment deliberately extracts CLS tokens from the *exact* DINOv2-L
trunk and weights embedded in the deployed LingBot checkpoint.  It constructs
two goal sessions per episode:

* revisit_b: the rendered B goal against its own pre-switch history;
* paired_swap_probe: the paired episode's A image against a populated
  trajectory. Geometry, rather than the source name, decides whether this is
  actually a revisit; same-scene paired goals can legitimately overlap.

Model selection and threshold calibration are always scene-disjoint. Reporting
can use leave-one-scene-out folds or an explicit fixed train/test scene split.
Every exported diagnostic model is stamped ``deployment_approved=false``;
closed-loop A/B is still required even if an offline split looks promising.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from MemNavData.reliability_router import (
        LinearReliabilityRouter,
        calibrate_zero_error_thresholds,
        selective_decisions,
        symmetric_relation_features,
    )
except ModuleNotFoundError:  # direct ``python MemNavData/...py`` invocation
    from reliability_router import (  # type: ignore
        LinearReliabilityRouter,
        calibrate_zero_error_thresholds,
        selective_decisions,
        symmetric_relation_features,
    )


DEFAULT_WEIGHT_SHA = (
    "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409")
FEATURE_VERSION = "exact_lingbot_dinov2l_cls_symmetric_v1"


@dataclass(frozen=True)
class Episode:
    scene: str
    name: str
    root: Path
    rgb_dir: Path
    switch: int
    n_frames: int
    intrinsic: np.ndarray


@dataclass(frozen=True)
class Session:
    session_id: str
    scene: str
    episode: str
    kind: str
    query: Path
    candidates: Tuple[Path, ...]
    intrinsic: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def numbered_jpgs(rgb_dir: Path, stop: Optional[int] = None,
                  stride: int = 1) -> Tuple[Path, ...]:
    paths = sorted(
        (path for path in rgb_dir.glob("*.jpg") if path.stem.isdigit()),
        key=lambda path: int(path.stem))
    if stop is not None:
        paths = [path for path in paths if int(path.stem) < stop]
    paths = paths[::stride]
    if not paths:
        raise ValueError(f"no candidate frames found in {rgb_dir}")
    return tuple(path.resolve() for path in paths)


def load_episodes(root: Path) -> List[Episode]:
    import pandas as pd

    episodes = []
    for scene_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for episode_root in sorted(
                path for path in scene_root.iterdir() if path.is_dir()):
            meta_path = episode_root / "meta" / "gen_meta.json"
            parquet_path = (episode_root / "data" / "chunk-000" /
                            "episode_000000.parquet")
            rgb_dir = (episode_root / "videos" / "chunk-000" /
                       "observation.images.rgb")
            goal_path = episode_root / "goal_1.jpg"
            for required in (meta_path, parquet_path, rgb_dir, goal_path):
                if not required.exists():
                    raise FileNotFoundError(required)
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            rows = pd.read_parquet(parquet_path, columns=[
                "observation.camera_intrinsic"])
            raw = rows.iloc[0]["observation.camera_intrinsic"]
            intrinsic = np.stack([
                np.asarray(row, dtype=np.float64) for row in raw])
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError(f"invalid intrinsic in {parquet_path}")
            episodes.append(Episode(
                scene=scene_root.name,
                name=episode_root.name,
                root=episode_root.resolve(),
                rgb_dir=rgb_dir.resolve(),
                switch=int(meta["switch_idx"]),
                n_frames=int(meta["n_frames"]),
                intrinsic=intrinsic,
            ))
    if not episodes:
        raise ValueError(f"no episodes found under {root}")
    return episodes


def build_sessions(episodes: Sequence[Episode], stride: int) -> List[Session]:
    by_scene: Dict[str, List[Episode]] = {}
    for episode in episodes:
        by_scene.setdefault(episode.scene, []).append(episode)
    sessions = []
    for scene, scene_episodes in sorted(by_scene.items()):
        scene_episodes = sorted(scene_episodes, key=lambda item: item.name)
        if len(scene_episodes) % 2:
            raise ValueError(
                f"scene {scene} needs adjacent episode pairs, got "
                f"{len(scene_episodes)} episodes")
        for index, episode in enumerate(scene_episodes):
            partner_index = index + 1 if index % 2 == 0 else index - 1
            partner = scene_episodes[partner_index]
            partner_query = partner.rgb_dir / f"{partner.switch - 1}.jpg"
            if not partner_query.is_file():
                raise FileNotFoundError(partner_query)
            sessions.append(Session(
                session_id=f"{scene}/{episode.name}/revisit_b",
                scene=scene,
                episode=episode.name,
                kind="revisit_b",
                query=(episode.root / "goal_1.jpg").resolve(),
                candidates=numbered_jpgs(
                    episode.rgb_dir, stop=episode.switch, stride=stride),
                intrinsic=episode.intrinsic,
            ))
            sessions.append(Session(
                session_id=f"{scene}/{episode.name}/paired_swap_probe",
                scene=scene,
                episode=episode.name,
                kind="paired_swap_probe",
                query=partner_query.resolve(),
                candidates=numbered_jpgs(
                    episode.rgb_dir, stop=episode.n_frames, stride=stride),
                intrinsic=episode.intrinsic,
            ))
    return sessions


def load_exact_dino_embeddings(paths: Sequence[Path], lingbot_repo: Path,
                               weights: Path, device: str,
                               batch_size: int) -> Tuple[np.ndarray, float]:
    import torch

    sys.path.insert(0, str(lingbot_repo))
    from lingbot_map.layers.vision_transformer import vit_large
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    raw_state = torch.load(weights, map_location="cpu", weights_only=False)
    if (isinstance(raw_state, dict) and "model" in raw_state
            and isinstance(raw_state["model"], dict)):
        state = raw_state["model"]
    else:
        state = raw_state
    prefix = "aggregator.patch_embed."
    patch_state = {
        key[len(prefix):]: value for key, value in state.items()
        if key.startswith(prefix)}
    if len(patch_state) != 344:
        raise RuntimeError(
            f"expected 344 DINO tensors in LingBot checkpoint, got "
            f"{len(patch_state)}")
    del raw_state, state
    gc.collect()

    model = vit_large(
        img_size=518,
        patch_size=14,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    )
    model.load_state_dict(patch_state, strict=True)
    del patch_state
    gc.collect()
    model = model.to(device).eval()
    torch_device = torch.device(device)
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=torch_device).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=torch_device).view(1, 3, 1, 1)
    use_cuda = torch_device.type == "cuda"

    outputs = []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        batch_paths = [str(path) for path in paths[start:start + batch_size]]
        images = load_and_preprocess_images(
            batch_paths, mode="pad", image_size=518, patch_size=14)
        images = images.to(torch_device, non_blocking=use_cuda)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_cuda else torch.autocast("cpu", enabled=False))
        with torch.inference_mode(), autocast:
            encoded = model.forward_features((images - mean) / std)
        outputs.append(encoded["x_norm_clstoken"].float().cpu().numpy())
        del images, encoded
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(
                f"[dino] {min(start + batch_size, len(paths))}/{len(paths)}",
                flush=True)
            gc.collect()
    elapsed = time.perf_counter() - started
    embeddings = np.concatenate(outputs, axis=0)
    if embeddings.shape != (len(paths), 1024):
        raise RuntimeError(f"unexpected exact DINO shape {embeddings.shape}")
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return embeddings, elapsed


def embedding_cache(paths: Sequence[Path], cache_path: Path,
                    lingbot_repo: Path, weights: Path, weight_sha: str,
                    device: str, batch_size: int
                    ) -> Tuple[np.ndarray, dict]:
    path_strings = np.asarray([str(path) for path in paths])
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = cached["paths"].astype(str)
        cached_sha = str(cached["weight_sha"].item())
        if np.array_equal(cached_paths, path_strings) and cached_sha == weight_sha:
            embeddings = cached["embeddings"].astype(np.float32)
            return embeddings, {
                "cache_hit": True,
                "seconds": 0.0,
                "images": len(paths),
            }
        raise RuntimeError(
            f"embedding cache identity mismatch: remove {cache_path} explicitly")
    embeddings, seconds = load_exact_dino_embeddings(
        paths, lingbot_repo, weights, device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        paths=path_strings,
        embeddings=embeddings.astype(np.float16),
        weight_sha=np.asarray(weight_sha),
        feature_version=np.asarray(FEATURE_VERSION),
    )
    return embeddings, {
        "cache_hit": False,
        "seconds": seconds,
        "images": len(paths),
    }


def sift_description(path: Path, sift) -> Tuple[Sequence, Optional[np.ndarray]]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to decode {path}")
    return sift.detectAndCompute(image, None)


def geometric_teacher(query_features, candidate_path: Path,
                      intrinsic: np.ndarray, sift, matcher) -> dict:
    import cv2

    query_keypoints, query_desc = query_features
    candidate_keypoints, candidate_desc = sift_description(
        candidate_path, sift)
    base = dict(matches=0, inliers=0, inlier_ratio=0.0, error=None)
    if query_desc is None or candidate_desc is None:
        return dict(base, error="insufficient image features")
    pairs = matcher.knnMatch(candidate_desc, query_desc, k=2)
    good = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    matches = len(good)
    if matches < 8:
        return dict(base, matches=matches, error="too few ratio-test matches")
    candidate_pts = np.float32([
        candidate_keypoints[item.queryIdx].pt for item in good])
    query_pts = np.float32([
        query_keypoints[item.trainIdx].pt for item in good])
    essential, ransac_mask = cv2.findEssentialMat(
        candidate_pts, query_pts, intrinsic,
        cv2.RANSAC, 0.999, 1.5)
    best = 0
    if essential is not None:
        essential = np.asarray(essential, dtype=np.float64)
        if essential.shape == (3, 3):
            candidates = [essential]
        elif (essential.ndim == 2 and essential.shape[1] == 3
              and essential.shape[0] % 3 == 0):
            candidates = [essential[index:index + 3]
                          for index in range(0, essential.shape[0], 3)]
        else:
            candidates = []
        for candidate in candidates:
            mask = None if ransac_mask is None else ransac_mask.copy()
            try:
                recovered = cv2.recoverPose(
                    candidate, candidate_pts, query_pts, intrinsic,
                    mask=mask)[0]
                best = max(best, int(recovered))
            except cv2.error:
                continue
    ratio = float(best / matches)
    return dict(
        matches=matches,
        inliers=best,
        inlier_ratio=ratio,
        error=None if essential is not None else "essential matrix unavailable",
    )


def teacher_rows(sessions: Sequence[Session]) -> Tuple[List[dict], dict]:
    import cv2

    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher()
    rows = []
    started = time.perf_counter()
    for session_index, session in enumerate(sessions):
        query_features = sift_description(session.query, sift)
        for candidate in session.candidates:
            geometry = geometric_teacher(
                query_features, candidate, session.intrinsic, sift, matcher)
            passed = bool(
                geometry["matches"] >= 20
                and geometry["inliers"] >= 12
                and geometry["inlier_ratio"] >= 0.50)
            rows.append(dict(
                session_id=session.session_id,
                scene=session.scene,
                episode=session.episode,
                kind=session.kind,
                query_path=str(session.query),
                candidate_path=str(candidate),
                candidate_frame=int(candidate.stem),
                teacher_pass=int(passed),
                **geometry,
            ))
        print(
            f"[teacher] {session_index + 1}/{len(sessions)} "
            f"{session.session_id}", flush=True)
    elapsed = time.perf_counter() - started
    return rows, {
        "seconds": elapsed,
        "pairs": len(rows),
        "milliseconds_per_pair": 1000.0 * elapsed / max(len(rows), 1),
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    negative = labels == 0
    positive = labels == 1
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy_at_0.5": float(accuracy_score(labels, predicted)),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(labels, predicted)),
        "false_positive_rate_at_0.5": float(
            np.mean(predicted[negative])) if np.any(negative) else None,
        "false_negative_rate_at_0.5": float(
            np.mean(~predicted[positive])) if np.any(positive) else None,
    }


def cascade_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=np.int8)
    accepted = decisions == 1
    rejected = decisions == -1
    deferred = decisions == 0
    return {
        "examples": int(labels.size),
        "automatic_accept": int(accepted.sum()),
        "automatic_reject": int(rejected.sum()),
        "teacher_calls": int(deferred.sum()),
        "teacher_call_rate": float(deferred.mean()),
        "automatic_coverage": float(np.mean(~deferred)),
        "false_accept": int(np.sum(accepted & (labels == 0))),
        "false_reject": int(np.sum(rejected & (labels == 1))),
        "cascade_disagreements_with_teacher": int(
            np.sum((accepted & (labels == 0)) | (rejected & (labels == 1)))),
    }


def finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def threshold_metrics(thresholds) -> dict:
    return {
        "reject_max": finite_or_none(thresholds.reject_max),
        "accept_min": finite_or_none(thresholds.accept_min),
        "reject_calibration_count": thresholds.reject_calibration_count,
        "accept_calibration_count": thresholds.accept_calibration_count,
    }


def evaluate_scene_disjoint(features: np.ndarray, cosine: np.ndarray,
                            labels: np.ndarray, scenes: np.ndarray,
                            rows: Sequence[dict], min_calibration: int,
                            seed: int,
                            heldout_scenes: Optional[Sequence[str]] = None
                            ) -> Tuple[dict, object, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def relation_pipeline():
        # Strong L2 regularization is intentional: the exact DINO relation has
        # 2049 dimensions while calibration is grouped by a small number of
        # scenes.  The geometric teacher remains the fallback for ambiguity.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.01, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=seed),
        )

    def cosine_pipeline():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=seed),
        )

    def inner_scene_oof(indices: np.ndarray, inputs: np.ndarray,
                        model_factory) -> np.ndarray:
        """Predictions for threshold calibration from unseen inner scenes."""
        inner_groups = scenes[indices]
        if len(np.unique(inner_groups)) < 3:
            raise ValueError("nested calibration requires at least three scenes")
        probabilities = np.full(indices.shape, np.nan, dtype=np.float64)
        inner_logo = LeaveOneGroupOut()
        for inner_train, inner_test in inner_logo.split(
                inputs[indices], labels[indices], groups=inner_groups):
            model = model_factory()
            model.fit(inputs[indices[inner_train]], labels[indices[inner_train]])
            probabilities[inner_test] = model.predict_proba(
                inputs[indices[inner_test]])[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("inner scene-disjoint calibration is incomplete")
        return probabilities

    relation_oof = np.full(labels.shape, np.nan, dtype=np.float64)
    cosine_oof = np.full(labels.shape, np.nan, dtype=np.float64)
    relation_decision_oof = np.zeros(labels.shape, dtype=np.int8)
    cosine_decision_oof = np.zeros(labels.shape, dtype=np.int8)
    relation_min1_decision_oof = np.zeros(labels.shape, dtype=np.int8)
    cosine_min1_decision_oof = np.zeros(labels.shape, dtype=np.int8)
    fold_reports = []
    if heldout_scenes:
        heldout_set = set(heldout_scenes)
        known_scenes = set(scenes.tolist())
        missing = heldout_set - known_scenes
        if missing:
            raise ValueError(
                f"held-out scenes absent from data: {sorted(missing)}")
        evaluation_mask = np.isin(scenes, sorted(heldout_set))
        test_index = np.flatnonzero(evaluation_mask)
        train_index = np.flatnonzero(~evaluation_mask)
        if not train_index.size or not test_index.size:
            raise ValueError("fixed holdout needs non-empty train and test sets")
        outer_splits = [(train_index, test_index)]
        final_train_index = train_index
        protocol = "fixed_scene_holdout"
    else:
        logo = LeaveOneGroupOut()
        outer_splits = list(logo.split(features, labels, groups=scenes))
        evaluation_mask = np.ones(labels.shape, dtype=bool)
        final_train_index = np.arange(labels.size)
        protocol = "leave_one_scene_out"

    for train_index, test_index in outer_splits:
        heldout = ",".join(sorted(np.unique(scenes[test_index]).tolist()))
        relation_model = relation_pipeline()
        cosine_model = cosine_pipeline()
        relation_calibration_probability = inner_scene_oof(
            train_index, features, relation_pipeline)
        cosine_calibration_probability = inner_scene_oof(
            train_index, cosine[:, None], cosine_pipeline)
        relation_thresholds = calibrate_zero_error_thresholds(
            labels[train_index], relation_calibration_probability,
            min_samples=min_calibration)
        cosine_thresholds = calibrate_zero_error_thresholds(
            labels[train_index], cosine_calibration_probability,
            min_samples=min_calibration)
        relation_min1_thresholds = calibrate_zero_error_thresholds(
            labels[train_index], relation_calibration_probability,
            min_samples=1)
        cosine_min1_thresholds = calibrate_zero_error_thresholds(
            labels[train_index], cosine_calibration_probability,
            min_samples=1)
        relation_model.fit(features[train_index], labels[train_index])
        cosine_model.fit(cosine[train_index, None], labels[train_index])
        test_probability = relation_model.predict_proba(
            features[test_index])[:, 1]
        test_decisions = selective_decisions(
            test_probability, relation_thresholds)
        test_cosine_probability = cosine_model.predict_proba(
            cosine[test_index, None])[:, 1]
        test_cosine_decisions = selective_decisions(
            test_cosine_probability, cosine_thresholds)
        test_relation_min1_decisions = selective_decisions(
            test_probability, relation_min1_thresholds)
        test_cosine_min1_decisions = selective_decisions(
            test_cosine_probability, cosine_min1_thresholds)
        relation_oof[test_index] = test_probability
        cosine_oof[test_index] = test_cosine_probability
        relation_decision_oof[test_index] = test_decisions
        cosine_decision_oof[test_index] = test_cosine_decisions
        relation_min1_decision_oof[test_index] = (
            test_relation_min1_decisions)
        cosine_min1_decision_oof[test_index] = test_cosine_min1_decisions
        fold_reports.append({
            "heldout_scene": heldout,
            "train_examples": int(train_index.size),
            "test_examples": int(test_index.size),
            "test_positive": int(labels[test_index].sum()),
            "reject_max": finite_or_none(relation_thresholds.reject_max),
            "accept_min": finite_or_none(relation_thresholds.accept_min),
            "reject_calibration_count": (
                relation_thresholds.reject_calibration_count),
            "accept_calibration_count": (
                relation_thresholds.accept_calibration_count),
            "relation": binary_metrics(labels[test_index], test_probability),
            "cascade": cascade_metrics(labels[test_index], test_decisions),
            "cosine_only": binary_metrics(
                labels[test_index], test_cosine_probability),
            "cosine_only_thresholds": {
                "reject_max": finite_or_none(cosine_thresholds.reject_max),
                "accept_min": finite_or_none(cosine_thresholds.accept_min),
                "reject_calibration_count": (
                    cosine_thresholds.reject_calibration_count),
                "accept_calibration_count": (
                    cosine_thresholds.accept_calibration_count),
            },
            "cosine_only_cascade": cascade_metrics(
                labels[test_index], test_cosine_decisions),
            "min1_tail_capacity": {
                "relation_thresholds": threshold_metrics(
                    relation_min1_thresholds),
                "relation_cascade": cascade_metrics(
                    labels[test_index], test_relation_min1_decisions),
                "cosine_only_thresholds": threshold_metrics(
                    cosine_min1_thresholds),
                "cosine_only_cascade": cascade_metrics(
                    labels[test_index], test_cosine_min1_decisions),
            },
        })
    if (not np.isfinite(relation_oof[evaluation_mask]).all()
            or not np.isfinite(cosine_oof[evaluation_mask]).all()):
        raise RuntimeError("scene-disjoint predictions are incomplete")

    session_rows = []
    by_session: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        if evaluation_mask[index]:
            by_session.setdefault(row["session_id"], []).append(index)
    for session_id, indices in sorted(by_session.items()):
        index = indices[int(np.argmax(cosine[indices]))]
        row = rows[index]
        session_rows.append({
            "session_id": session_id,
            "scene": row["scene"],
            "episode": row["episode"],
            "kind": row["kind"],
            "candidate_frame": row["candidate_frame"],
            "dino_cosine": float(cosine[index]),
            "teacher_pass": int(labels[index]),
            "relation_probability": float(relation_oof[index]),
            "selective_decision": int(relation_decision_oof[index]),
            "cosine_probability": float(cosine_oof[index]),
            "cosine_selective_decision": int(
                cosine_decision_oof[index]),
            "relation_min1_selective_decision": int(
                relation_min1_decision_oof[index]),
            "cosine_min1_selective_decision": int(
                cosine_min1_decision_oof[index]),
            "geometry_matches": int(row["matches"]),
            "geometry_inliers": int(row["inliers"]),
            "geometry_inlier_ratio": float(row["inlier_ratio"]),
        })
    session_labels = np.asarray(
        [row["teacher_pass"] for row in session_rows], dtype=np.int64)
    session_probability = np.asarray(
        [row["relation_probability"] for row in session_rows],
        dtype=np.float64)
    session_decisions = np.asarray(
        [row["selective_decision"] for row in session_rows], dtype=np.int8)
    session_cosine_probability = np.asarray(
        [row["cosine_probability"] for row in session_rows],
        dtype=np.float64)
    session_cosine_decisions = np.asarray(
        [row["cosine_selective_decision"] for row in session_rows],
        dtype=np.int8)
    session_relation_min1_decisions = np.asarray(
        [row["relation_min1_selective_decision"] for row in session_rows],
        dtype=np.int8)
    session_cosine_min1_decisions = np.asarray(
        [row["cosine_min1_selective_decision"] for row in session_rows],
        dtype=np.int8)
    session_expected = np.asarray(
        [row["kind"] == "revisit_b" for row in session_rows], dtype=bool)

    report = {
        "protocol": protocol,
        "train_scenes": sorted(np.unique(scenes[final_train_index]).tolist()),
        "test_scenes": sorted(np.unique(scenes[evaluation_mask]).tolist()),
        "pair_relation": binary_metrics(
            labels[evaluation_mask], relation_oof[evaluation_mask]),
        "pair_cosine_only": binary_metrics(
            labels[evaluation_mask], cosine_oof[evaluation_mask]),
        "pair_cascade": cascade_metrics(
            labels[evaluation_mask],
            relation_decision_oof[evaluation_mask]),
        "pair_cosine_only_cascade": cascade_metrics(
            labels[evaluation_mask], cosine_decision_oof[evaluation_mask]),
        "pair_min1_tail_capacity": {
            "relation": cascade_metrics(
                labels[evaluation_mask],
                relation_min1_decision_oof[evaluation_mask]),
            "cosine_only": cascade_metrics(
                labels[evaluation_mask],
                cosine_min1_decision_oof[evaluation_mask]),
        },
        "pair_visual_floor_0.88": {
            "predicted_positive": int(np.sum(
                (cosine >= 0.88) & evaluation_mask)),
            "false_positive": int(np.sum(
                (cosine >= 0.88) & (labels == 0) & evaluation_mask)),
            "false_negative": int(np.sum(
                (cosine < 0.88) & (labels == 1) & evaluation_mask)),
        },
        "top1_sessions": {
            "count": len(session_rows),
            "teacher_positive": int(session_labels.sum()),
            "expected_revisit_count": int(session_expected.sum()),
            "revisit_top1_teacher_positive": int(
                np.sum(session_labels[session_expected])),
            "paired_swap_top1_teacher_positive": int(
                np.sum(session_labels[~session_expected])),
            "classifier": binary_metrics(
                session_labels, session_probability),
            "cascade": cascade_metrics(session_labels, session_decisions),
            "cosine_only_classifier": binary_metrics(
                session_labels, session_cosine_probability),
            "cosine_only_cascade": cascade_metrics(
                session_labels, session_cosine_decisions),
            "min1_tail_capacity": {
                "relation": cascade_metrics(
                    session_labels, session_relation_min1_decisions),
                "cosine_only": cascade_metrics(
                    session_labels, session_cosine_min1_decisions),
            },
        },
        "folds": fold_reports,
        "session_rows": session_rows,
    }

    final_relation = relation_pipeline()
    final_relation.fit(
        features[final_train_index], labels[final_train_index])
    # Deployment thresholds must likewise come from examples not used to fit
    # their probability.  The full-data model is exported only as a diagnostic;
    # its thresholds are calibrated from the outer scene-disjoint predictions.
    final_calibration_probability = (
        inner_scene_oof(final_train_index, features, relation_pipeline)
        if heldout_scenes else relation_oof)
    final_thresholds = calibrate_zero_error_thresholds(
        labels[final_train_index], final_calibration_probability,
        min_samples=min_calibration)
    scaler = final_relation.named_steps["standardscaler"]
    logistic = final_relation.named_steps["logisticregression"]
    portable = LinearReliabilityRouter(
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=logistic.coef_[0],
        intercept=float(logistic.intercept_[0]),
        thresholds=final_thresholds,
        feature_version=FEATURE_VERSION,
    )
    final_cosine_calibration_probability = (
        inner_scene_oof(final_train_index, cosine[:, None], cosine_pipeline)
        if heldout_scenes else cosine_oof)
    final_cosine_thresholds = calibrate_zero_error_thresholds(
        labels[final_train_index], final_cosine_calibration_probability,
        min_samples=min_calibration)
    report["final_cosine_only_thresholds"] = threshold_metrics(
        final_cosine_thresholds)
    comparison_index = final_train_index[:32]
    np.testing.assert_allclose(
        portable.predict_proba_from_features(features[comparison_index]),
        final_relation.predict_proba(features[comparison_index])[:, 1],
        rtol=1e-10, atol=1e-10)
    return report, portable, final_relation


def write_pair_csv(path: Path, rows: Sequence[dict], cosine: np.ndarray,
                   probabilities: np.ndarray, decisions: np.ndarray) -> None:
    fieldnames = list(rows[0]) + [
        "dino_cosine", "oof_relation_probability", "oof_selective_decision"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score, probability, decision in zip(
                rows, cosine, probabilities, decisions):
            writer.writerow(dict(
                row,
                dino_cosine=float(score),
                oof_relation_probability=float(probability),
                oof_selective_decision=int(decision),
            ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-root", type=Path, action="append", required=True,
        help="repeat to combine disjoint episode roots")
    parser.add_argument(
        "--heldout-scene", action="append", default=[],
        help="repeat for a fixed train-scene to heldout-scene evaluation")
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--min-calibration", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--expected-weight-sha", default=DEFAULT_WEIGHT_SHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.candidate_stride < 1:
        raise ValueError("batch size and candidate stride must be positive")
    for required in (*args.episode_root, args.lingbot_repo, args.weights):
        if not required.exists():
            raise FileNotFoundError(required)
    weight_sha = sha256(args.weights)
    if args.expected_weight_sha and weight_sha != args.expected_weight_sha:
        raise RuntimeError(
            f"LingBot SHA mismatch: expected {args.expected_weight_sha}, "
            f"got {weight_sha}")
    dependency_commit = git_value(args.lingbot_repo, "rev-parse", "HEAD")
    if dependency_commit != "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2":
        raise RuntimeError(
            "LingBot dependency commit mismatch: expected "
            "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2, got "
            f"{dependency_commit}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for episode_root in args.episode_root:
        episodes.extend(load_episodes(episode_root.resolve()))
    identities = [(episode.scene, episode.name) for episode in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate scene/episode identities across episode roots")
    sessions = build_sessions(episodes, args.candidate_stride)
    all_paths = sorted({
        path for session in sessions
        for path in (session.query, *session.candidates)})
    print(
        f"[dataset] episodes={len(episodes)} sessions={len(sessions)} "
        f"unique_images={len(all_paths)}", flush=True)

    embeddings, dino_timing = embedding_cache(
        all_paths,
        args.out_dir / "exact_dino_cls.npz",
        args.lingbot_repo.resolve(),
        args.weights.resolve(),
        weight_sha,
        args.device,
        args.batch_size,
    )
    rows, teacher_timing = teacher_rows(sessions)
    path_index = {str(path): index for index, path in enumerate(all_paths)}
    goal_embeddings = embeddings[[path_index[row["query_path"]] for row in rows]]
    memory_embeddings = embeddings[[
        path_index[row["candidate_path"]] for row in rows]]
    features = symmetric_relation_features(goal_embeddings, memory_embeddings)
    cosine = features[:, -1].astype(np.float64)
    labels = np.asarray([row["teacher_pass"] for row in rows], dtype=np.int64)
    scenes = np.asarray([row["scene"] for row in rows])
    if len(np.unique(scenes)) < 3:
        raise ValueError("scene-disjoint evaluation requires at least three scenes")
    if len(np.unique(labels)) != 2:
        raise ValueError("geometry teacher produced only one class")

    report, portable, _final_model = evaluate_scene_disjoint(
        features, cosine, labels, scenes, rows,
        args.min_calibration, args.seed, args.heldout_scene)

    with open(args.out_dir / "teacher_pairs.csv", "w", newline="",
              encoding="utf-8") as handle:
        fieldnames = list(rows[0]) + ["dino_cosine"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score in zip(rows, cosine):
            writer.writerow(dict(row, dino_cosine=float(score)))

    with open(args.out_dir / "top1_sessions.csv", "w", newline="",
              encoding="utf-8") as handle:
        session_rows = report["session_rows"]
        writer = csv.DictWriter(handle, fieldnames=list(session_rows[0]))
        writer.writeheader()
        writer.writerows(session_rows)

    # Benchmark only the incremental portable head. Exact DINO CLS is already
    # produced by the deployed retrieval path and is therefore not new router
    # latency. SIFT timing above includes the current implementation's image
    # decoding, detection, matching, and essential-matrix verification.
    repetitions = 5000
    started = time.perf_counter()
    for _ in range(repetitions):
        portable.predict_proba_from_features(features[:1])
    head_us = 1e6 * (time.perf_counter() - started) / repetitions

    manifest = {
        "deployment_approved": False,
        "reason": (
            f"local feasibility diagnostic on "
            f"{len({episode.scene for episode in episodes})} scenes / "
            f"{len(episodes)} episodes; "
            "requires larger scene-disjoint calibration and closed-loop A/B"),
        "feature_version": FEATURE_VERSION,
        "created_at_unix": time.time(),
        "repo_root": str(Path(__file__).resolve().parents[1]),
        "repo_commit": git_value(
            Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "lingbot_repo": str(args.lingbot_repo.resolve()),
        "lingbot_commit": dependency_commit,
        "lingbot_weights": str(args.weights.resolve()),
        "lingbot_weight_sha256": weight_sha,
        "episode_roots": [str(path.resolve()) for path in args.episode_root],
        "heldout_scenes": list(args.heldout_scene),
        "episodes": len(episodes),
        "scenes": len({episode.scene for episode in episodes}),
        "sessions": len(sessions),
        "candidate_stride": args.candidate_stride,
        "pairs": len(rows),
        "teacher_positive": int(labels.sum()),
        "teacher_negative": int((labels == 0).sum()),
        "dino_timing": dino_timing,
        "geometry_teacher_timing": teacher_timing,
        "portable_head_microseconds_per_pair": head_us,
        "evaluation": report,
        "final_diagnostic_thresholds": {
            "reject_max": finite_or_none(portable.thresholds.reject_max),
            "accept_min": finite_or_none(portable.thresholds.accept_min),
            "reject_calibration_count": (
                portable.thresholds.reject_calibration_count),
            "accept_calibration_count": (
                portable.thresholds.accept_calibration_count),
        },
    }
    with open(args.out_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    portable.save(args.out_dir / "diagnostic_router_not_for_deployment.json")
    print(json.dumps({
        "deployment_approved": False,
        "pairs": len(rows),
        "positive": int(labels.sum()),
        "pair_relation": report["pair_relation"],
        "pair_cosine_only": report["pair_cosine_only"],
        "pair_cascade": report["pair_cascade"],
        "pair_cosine_only_cascade": report["pair_cosine_only_cascade"],
        "pair_min1_tail_capacity": report["pair_min1_tail_capacity"],
        "top1_sessions": report["top1_sessions"],
        "teacher_ms_per_pair": teacher_timing["milliseconds_per_pair"],
        "head_us_per_pair": head_us,
        "out_dir": str(args.out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
