#!/usr/bin/env python3
"""Train-only open-set localization audit with SuperPoint + LightGlue.

The candidate generator is deliberately unchanged: candidates are ordered by
the existing DINO cosine.  For each selected history frame, this script asks
whether a modern local matcher produces spatially distributed epipolar support
for the image goal.  It reports both candidate-level co-visibility separation
and session-level Novel/Revisit existence separation.  No decision threshold
is fitted and no navigation rollout is performed here.

This inexpensive image-only audit precedes the LingBot 2D--3D localization
test.  A matcher that cannot reject no-match sessions in 2-D is not allowed to
consume long LingBot/Habitat evaluation time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = {
    "session_id", "scene", "episode", "kind", "query_path",
    "candidate_path", "candidate_frame", "dino_cosine", "teacher_covis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument(
        "--feature-root", type=Path,
        help=("optional LingBot-cache availability filter; omit when the "
              "teacher/manifest already defines the runnable universe"))
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kind", default="cross_episode_train")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--candidate-min-gap", type=int, default=4)
    parser.add_argument(
        "--minimum-anchor", type=int, default=0,
        help=("exclude history frames before the downstream LingBot causal "
              "initialization boundary; formal runs use num_scale=8"))
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.2)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold-px", type=float, default=1.5)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path.resolve()}", "-C",
             str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        bundled = path / "BUNDLED_COMMIT"
        if not bundled.is_file():
            raise RuntimeError(
                "LightGlue provenance requires a Git checkout or "
                "BUNDLED_COMMIT")
        commit = bundled.read_text(encoding="utf-8").strip()
        if len(commit) != 40 or any(char not in "0123456789abcdef"
                                    for char in commit):
            raise RuntimeError("invalid LightGlue BUNDLED_COMMIT")
        return commit


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cache_available(feature_root: Path, scene: str, episode: str) -> bool:
    pattern = f"*/{scene}/{episode}/videos/chunk-000/lingbot_cache.npz"
    return any(feature_root.glob(pattern))


def candidate_family(path: str | Path) -> str:
    return next((
        part for part in Path(path).parts if part.startswith("mp3d_")), "")


def cache_maximum_anchor(
        feature_root: Path, scene: str, episode: str,
        candidate_path: str | Path) -> int | None:
    """Return the last center anchor executable by a legacy camera cache."""
    family = candidate_family(candidate_path)
    matches = []
    if family:
        direct = (feature_root / family / scene / episode / "videos"
                  / "chunk-000" / "lingbot_cam_cache.npz")
        if direct.is_file():
            matches = [direct]
    if not matches:
        pattern = f"*/{scene}/{episode}/videos/chunk-000/lingbot_cam_cache.npz"
        matches = sorted(feature_root.glob(pattern))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"ambiguous camera caches for {scene}/{episode}: {matches}")
    with np.load(matches[0]) as cache:
        if "cam_pose_enc" not in cache:
            raise RuntimeError(f"camera cache lacks cam_pose_enc: {matches[0]}")
        # The online append path needs anchor+1 to exist.
        return int(np.asarray(cache["cam_pose_enc"]).shape[0]) - 2


def label_from_covis(value: float, positive: float, negative: float) -> int:
    if value >= positive:
        return 1
    if value <= negative:
        return 0
    return -1


def temporal_topk(group: pd.DataFrame, top_k: int,
                  minimum_gap: int) -> pd.DataFrame:
    """DINO-ordered top-K with the deployment temporal diversity rule."""
    ordered = group.sort_values(
        ["dino_cosine", "candidate_frame"], ascending=[False, True])
    selected = []
    for index, row in ordered.iterrows():
        frame = int(row["candidate_frame"])
        if all(abs(frame - previous) >= minimum_gap for previous in selected):
            selected.append(frame)
            yield index
            if len(selected) == top_k:
                return


def select_rows(frame: pd.DataFrame, *, feature_root: Path | None, kind: str,
                scenes: Iterable[str], sessions: Iterable[str], top_k: int,
                minimum_gap: int, minimum_anchor: int, positive: float,
                negative: float) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    selected = frame.loc[frame["kind"].eq(kind)].copy()
    scenes = tuple(scenes)
    sessions = tuple(sessions)
    if scenes:
        selected = selected.loc[selected["scene"].isin(scenes)]
    if sessions:
        selected = selected.loc[selected["session_id"].isin(sessions)]
    selected = selected.loc[
        selected["candidate_frame"].ge(minimum_anchor)
        & selected["teacher_covis"].notna()
        & selected["dino_cosine"].notna()].copy()
    if feature_root is not None:
        availability = {}
        for row in selected.itertuples():
            family = candidate_family(row.candidate_path)
            key = str(row.scene), str(row.episode), family
            if key not in availability:
                availability[key] = cache_maximum_anchor(
                    feature_root, str(row.scene), str(row.episode),
                    str(row.candidate_path))
        selected = selected.loc[
            [
                availability[(
                    str(row.scene), str(row.episode),
                    candidate_family(row.candidate_path))] is not None
                and int(row.candidate_frame) <= int(
                    availability[(
                        str(row.scene), str(row.episode),
                        candidate_family(row.candidate_path))])
                for row in selected.itertuples()
            ]
        ].copy()
    if selected.empty:
        raise RuntimeError("no teacher sessions have a local LingBot cache")
    session_max = selected.groupby("session_id")["teacher_covis"].max()
    session_label = session_max.map(
        lambda value: label_from_covis(value, positive, negative))
    selected_indices = []
    for _session_id, group in selected.groupby("session_id", sort=True):
        selected_indices.extend(temporal_topk(group, top_k, minimum_gap))
    selected = selected.loc[selected_indices].copy()
    selected["candidate_label"] = selected["teacher_covis"].map(
        lambda value: label_from_covis(value, positive, negative))
    selected["session_label"] = selected["session_id"].map(session_label)
    selected["session_max_covis"] = selected["session_id"].map(session_max)
    selected["dino_rank"] = selected.groupby("session_id")[
        "dino_cosine"].rank(method="first", ascending=False).astype(int)
    return selected.sort_values(
        ["session_id", "dino_rank", "candidate_frame"])


def hull_coverage(points: np.ndarray, height: int, width: int) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / max(float(height * width), 1.0))


def grid_coverage(points: np.ndarray, height: int, width: int,
                  grid_size: int = 4) -> float:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return 0.0
    x = np.clip((points[:, 0] / width * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip((points[:, 1] / height * grid_size).astype(int), 0, grid_size - 1)
    return float(len(set(zip(x.tolist(), y.tolist()))) / (grid_size ** 2))


def geometric_support(points0: np.ndarray, points1: np.ndarray,
                      scores: np.ndarray, shape0: tuple[int, int],
                      shape1: tuple[int, int], threshold: float) -> dict:
    """Measure epipolar and planar support without fitting a classifier."""
    points0 = np.asarray(points0, dtype=np.float32)
    points1 = np.asarray(points1, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    count = len(points0)
    result = {
        "lightglue_matches": int(count),
        "lightglue_score_mean": float(scores.mean()) if count else 0.0,
        "lightglue_score_median": float(np.median(scores)) if count else 0.0,
        "fundamental_inliers": 0,
        "fundamental_inlier_ratio": 0.0,
        "fundamental_query_hull_coverage": 0.0,
        "fundamental_reference_hull_coverage": 0.0,
        "fundamental_query_grid_coverage": 0.0,
        "fundamental_reference_grid_coverage": 0.0,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
    }
    if count < 8:
        return result
    cv2.setRNGSeed(0)
    try:
        _fundamental, mask = cv2.findFundamentalMat(
            points0, points1, cv2.USAC_MAGSAC, threshold, 0.999, 10000)
    except cv2.error:
        # Repeated or exactly degenerate keypoints occasionally make OpenCV's
        # USAC initializer assert.  Degeneracy is no geometric support and
        # must fail closed rather than abort a long audit.
        mask = None
    if mask is not None:
        inliers = np.asarray(mask).reshape(-1).astype(bool)
        result.update({
            "fundamental_inliers": int(inliers.sum()),
            "fundamental_inlier_ratio": float(inliers.mean()),
            "fundamental_reference_hull_coverage": hull_coverage(
                points0[inliers], *shape0),
            "fundamental_query_hull_coverage": hull_coverage(
                points1[inliers], *shape1),
            "fundamental_reference_grid_coverage": grid_coverage(
                points0[inliers], *shape0),
            "fundamental_query_grid_coverage": grid_coverage(
                points1[inliers], *shape1),
        })
    try:
        _homography, homography_mask = cv2.findHomography(
            points0, points1, cv2.USAC_MAGSAC, 3.0, None, 10000, 0.999)
    except cv2.error:
        homography_mask = None
    if homography_mask is not None:
        homography_inliers = int(np.asarray(homography_mask).sum())
        result.update({
            "homography_inliers": homography_inliers,
            "homography_inlier_ratio": float(homography_inliers / count),
        })
    return result


def metric_summary(labels: np.ndarray, values: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    valid = (labels >= 0) & np.isfinite(values)
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"n": int(valid.sum()), "roc_auc": None, "ap": None}
    return {
        "n": int(valid.sum()),
        "roc_auc": float(roc_auc_score(labels[valid], values[valid])),
        "ap": float(average_precision_score(labels[valid], values[valid])),
    }


def feature_auc(frame: pd.DataFrame, label_column: str) -> dict:
    features = (
        "dino_cosine", "matches", "inliers", "inlier_ratio",
        "lightglue_matches", "lightglue_score_mean",
        "lightglue_score_median", "fundamental_inliers",
        "fundamental_inlier_ratio", "fundamental_query_hull_coverage",
        "fundamental_reference_hull_coverage",
        "fundamental_query_grid_coverage",
        "fundamental_reference_grid_coverage", "homography_inliers",
    )
    return {
        feature: metric_summary(frame[label_column], frame[feature])
        for feature in features if feature in frame
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if (args.top_k < 1 or args.candidate_min_gap < 1
            or args.minimum_anchor < 0 or args.print_every < 1):
        raise ValueError("top-k and candidate gap must be positive")
    if not args.negative_threshold < args.positive_threshold:
        raise ValueError("negative threshold must be below positive threshold")
    if args.dependency_root:
        sys.path.insert(0, str(args.dependency_root.resolve()))
    sys.path.insert(0, str(args.lightglue_repo.resolve()))
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd

    teacher_sha = sha256_file(args.teacher_csv)
    source = pd.read_csv(args.teacher_csv)
    selected = select_rows(
        source, feature_root=args.feature_root, kind=args.kind,
        scenes=args.scene, sessions=args.session, top_k=args.top_k,
        minimum_gap=args.candidate_min_gap,
        minimum_anchor=args.minimum_anchor,
        positive=args.positive_threshold, negative=args.negative_threshold)
    for column in ("query_path", "candidate_path"):
        missing = [path for path in selected[column] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {column}: {missing[0]}")

    device = torch.device(args.device)
    torch.manual_seed(0)
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    def features(path: str):
        image = load_image(path).to(device)
        with torch.inference_mode():
            extracted = extractor.extract(image)
        return image, extracted

    rows = []
    started = time.time()
    current_query_path = None
    current_query = None
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        candidate_image, candidate_features = features(str(row.candidate_path))
        if str(row.query_path) != current_query_path:
            current_query = features(str(row.query_path))
            current_query_path = str(row.query_path)
        assert current_query is not None
        query_image, query_features = current_query
        with torch.inference_mode():
            matches = matcher({
                "image0": candidate_features,
                "image1": query_features,
            })
        feature0, feature1, matches = [
            rbd(value) for value in
            (candidate_features, query_features, matches)
        ]
        indices = matches["matches"]
        points0 = feature0["keypoints"][indices[:, 0]].detach().cpu().numpy()
        points1 = feature1["keypoints"][indices[:, 1]].detach().cpu().numpy()
        scores = matches["scores"].detach().cpu().numpy()
        geometry = geometric_support(
            points0, points1, scores,
            tuple(candidate_image.shape[-2:]), tuple(query_image.shape[-2:]),
            args.ransac_threshold_px)
        rows.append({
            **row._asdict(),
            **geometry,
        })
        if index == 1 or index == len(selected) or index % args.print_every == 0:
            print(
                f"[{index}/{len(selected)}] {row.session_id} "
                f"rank={row.dino_rank} covis={row.teacher_covis:.3f} "
                f"matches={geometry['lightglue_matches']} "
                f"F={geometry['fundamental_inliers']}", flush=True)

    result = pd.DataFrame(rows)
    csv_path = args.out_dir / "lightglue_open_set_rows.csv"
    report_path = args.out_dir / "lightglue_open_set_report.json"
    atomic_csv(csv_path, result)
    session_features = [
        "dino_cosine", "matches", "inliers", "inlier_ratio",
        "lightglue_matches", "lightglue_score_mean",
        "lightglue_score_median", "fundamental_inliers",
        "fundamental_inlier_ratio", "fundamental_query_hull_coverage",
        "fundamental_reference_hull_coverage",
        "fundamental_query_grid_coverage",
        "fundamental_reference_grid_coverage", "homography_inliers",
    ]
    aggregation = {
        feature: "max" for feature in session_features if feature in result
    }
    aggregation.update({
        "session_label": "first", "session_max_covis": "first",
        "scene": "first",
    })
    by_session = result.groupby("session_id", as_index=False).agg(aggregation)
    positive_sessions = result.loc[
        result["session_label"].eq(1), "session_id"].unique()
    retrieved_positive_sessions = result.loc[
        result["candidate_label"].eq(1), "session_id"].unique()
    weights = {
        name: str(path)
        for name, path in {
            "superpoint": Path(torch.hub.get_dir()) / "checkpoints"
            / "superpoint_v1.pth",
            "lightglue": Path(torch.hub.get_dir()) / "checkpoints"
            / "superpoint_lightglue_v0-1_arxiv.pth",
        }.items() if path.is_file()
    }
    report = {
        "status": "train_only_image_localization_audit_not_deployment",
        "n_candidates": int(len(result)),
        "n_sessions": int(result["session_id"].nunique()),
        "candidate_labels": {
            str(int(label)): int(count) for label, count in
            result["candidate_label"].value_counts().sort_index().items()
        },
        "session_labels": {
            str(int(label)): int(count) for label, count in
            by_session["session_label"].value_counts().sort_index().items()
        },
        "positive_session_candidate_recall": (
            len(set(positive_sessions) & set(retrieved_positive_sessions))
            / len(positive_sessions) if len(positive_sessions) else None),
        "candidate_level_auc": feature_auc(result, "candidate_label"),
        "session_level_max_auc": feature_auc(by_session, "session_label"),
        "config": {
            "kind": args.kind,
            "scenes": args.scene,
            "sessions": args.session,
            "top_k": args.top_k,
            "candidate_min_gap": args.candidate_min_gap,
            "minimum_anchor": args.minimum_anchor,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "max_keypoints": args.max_keypoints,
            "ransac_threshold_px": args.ransac_threshold_px,
            "print_every": args.print_every,
            "matcher": "SuperPoint2048+LightGlue_official_defaults",
            "decision_threshold_fitted": False,
            "candidate_bound_source": (
                "legacy_feature_camera_cache_anchor_plus_one"
                if args.feature_root else "teacher_contract_only"),
        },
        "provenance": {
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": teacher_sha,
            "feature_root": (
                str(args.feature_root.resolve()) if args.feature_root else None),
            "lightglue_repo": str(args.lightglue_repo.resolve()),
            "lightglue_commit": git_commit(args.lightglue_repo),
            "rows_csv_sha256": sha256_file(csv_path),
            "weights": {
                name: {"path": path, "sha256": sha256_file(Path(path))}
                for name, path in weights.items()
            },
            "elapsed_seconds": time.time() - started,
        },
        "rows_csv": str(csv_path.resolve()),
        "limitations": [
            "only train-role local-cache sessions are used",
            "teacher co-visibility labels are evaluation-only",
            "no threshold is selected and no closed-loop claim is made",
            "session score is a fixed max over unchanged DINO top-K candidates",
        ],
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
