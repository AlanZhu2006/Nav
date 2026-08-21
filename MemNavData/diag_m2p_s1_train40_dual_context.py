#!/usr/bin/env python3
"""Train40 zero-training gate for dual-context frozen-GCT goal queries.

This diagnostic does *not* introduce another retrieval method.  Its local arm
is deliberately the already-established raw-DINO-direct mechanism: the frozen
DINO top-1 history anchor from the pinned train40 table is followed by a
read-only frozen-GCT goal query.  The only new measurement is an independent
query of the same goal after the complete causal decision prefix.

Each source 3-leg trajectory is streamed exactly once.  Read-only goal queries
are scheduled at (a) the frozen DINO top-1 anchor and (b) decision_frame - 1.
The two goal poses therefore share weights and the same causal map but have
different context.  We test whether their disagreement adds information about
controller-relevant actionability (positive session and local bearing error at
most 30 degrees) beyond DINO cosine.  No LightGlue, PnP, RANSAC, model fitting,
closed loop, development scene, or blind scene is used.

The script is intentionally a representation gate.  A good result authorizes
only a scene-OOF selective head on train40; it is not a deployable certificate
or an SR result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

try:
    from MemNavData.diag_m2p_s1_gct_query import (
        DEFAULT_LINGBOT_REPO,
        _atomic_json,
        _build_model,
        _finite_json,
        _lingbot_relative_direction,
        _matrix,
        _navdp_ground_truth_relative,
        _read_only_query,
        _resolve_generated_mount,
        direction_error_degrees,
        require,
        signed_bearing_degrees,
    )
except ModuleNotFoundError:
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO,
        _atomic_json,
        _build_model,
        _finite_json,
        _lingbot_relative_direction,
        _matrix,
        _navdp_ground_truth_relative,
        _read_only_query,
        _resolve_generated_mount,
        direction_error_degrees,
        require,
        signed_bearing_degrees,
    )


DEFAULT_SELECTION = Path(
    ".diagnostics/certificate_distilled_compass_20260813/"
    "static_top8_480_lightglue_open_set_rows.csv")
DEFAULT_SPLIT = Path("MemNavData/router_multiscene_split_20260805.json")
DEFAULT_OUT = Path(".diagnostics/m2p_s1_train40_dual_context_20260813")

REQUIRED_COLUMNS = {
    "session_id", "split_role", "scene", "episode", "kind",
    "goal_role", "goal_variant", "decision_frame", "query_path",
    "candidate_frame", "candidate_path", "dino_cosine", "dino_rank",
    "candidate_label", "session_label", "session_max_covis", "no_future",
}


@dataclass(frozen=True)
class Session:
    session_id: str
    scene: str
    episode: str
    kind: str
    goal_role: str
    goal_variant: str
    decision_frame: int
    query_path: Path
    candidate_frame: int
    candidate_path: Path
    dino_cosine: float
    dino_rank: int
    candidate_label: int
    session_label: int
    session_max_covis: float

    @property
    def episode_root(self) -> Path:
        path = self.candidate_path
        for parent in path.parents:
            if parent.name == "videos":
                return parent.parent
        raise ValueError(f"cannot locate episode root for {path}")

    @property
    def rgb_dir(self) -> Path:
        return self.candidate_path.parent


@dataclass(frozen=True)
class EpisodePoseData:
    actions: np.ndarray
    base_extrinsic: np.ndarray
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def episode_root_from_image(path: Path) -> Path:
    path = Path(path)
    if path.name == "goal_image.jpg" or (
            path.stem.startswith("goal_") and path.suffix.lower() == ".jpg"):
        return path.parent
    for parent in path.parents:
        if parent.name == "videos":
            return parent.parent
    raise ValueError(f"cannot locate episode root for {path}")


def load_episode_pose_data(root: Path) -> EpisodePoseData:
    parquet = root / "data/chunk-000/episode_000000.parquet"
    metadata_path = root / "meta/gen_meta.json"
    require(parquet.is_file(), f"missing pose parquet: {parquet}")
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    frame = pd.read_parquet(
        parquet, columns=["action", "observation.camera_extrinsic"])
    require(not frame.empty, f"empty pose parquet: {parquet}")
    actions = np.stack([_matrix(value, "action") for value in frame["action"]])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mount = _resolve_generated_mount(
        _matrix(frame.iloc[0]["observation.camera_extrinsic"],
                "camera extrinsic"),
        str(metadata.get("frame_convention", "")))
    return EpisodePoseData(actions, mount, metadata)


_HABITAT_TO_DATA_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def _yaw_habitat_to_data_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    habitat = np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)
    return _HABITAT_TO_DATA_ROTATION @ habitat


def query_camera_to_world(path: Path,
                          cache: dict[Path, EpisodePoseData]) -> np.ndarray:
    root = episode_root_from_image(path).resolve()
    if root not in cache:
        cache[root] = load_episode_pose_data(root)
    episode = cache[root]
    if path.name == "goal_image.jpg":
        goal_index = 0
    elif path.stem.startswith("goal_") and path.suffix.lower() == ".jpg":
        goal_index = int(path.stem.split("_", 1)[1]) - 1
    else:
        frame_index = int(path.stem)
        require(0 <= frame_index < len(episode.actions),
                f"query frame outside trajectory: {path}")
        return episode.actions[frame_index].copy()
    goals = episode.metadata.get("goals", [])
    require(0 <= goal_index < len(goals), f"goal absent from {root}: {path}")
    goal = goals[goal_index]
    position = np.asarray(goal.get("pos"), dtype=np.float64)
    require(position.shape == (3,) and bool(np.isfinite(position).all()),
            f"invalid goal position in {root}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _yaw_habitat_to_data_rotation(
        float(goal.get("yaw_habitat", 0.0)))
    result[:3, 3] = position
    return result


def _consistent(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates().tolist()
    require(len(values) == 1,
            f"session {group.iloc[0]['session_id']} changes {column}: {values}")
    return values[0]


def _boolean_series(values: pd.Series, *, label: str) -> pd.Series:
    """Parse booleans without treating the string ``False`` as truthy."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    parsed = normalized.map({
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    })
    require(not parsed.isna().any(), f"{label} contains non-boolean values")
    return parsed.astype(bool)


def load_sessions(args: argparse.Namespace) -> list[Session]:
    table = pd.read_csv(args.selection_csv)
    missing = REQUIRED_COLUMNS - set(table.columns)
    require(not missing, f"selection CSV missing columns: {sorted(missing)}")
    table = table.loc[table["split_role"].astype(str) == args.allowed_role].copy()
    require(not table.empty, f"no rows for split role {args.allowed_role}")
    require(_boolean_series(table["no_future"], label="no_future").all(),
            "selection contains future rows")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    allowed_scenes = set(map(str, split[args.allowed_role]))
    observed_scenes = set(map(str, table["scene"].unique()))
    require(observed_scenes == allowed_scenes,
            "selection scene universe differs from frozen split: "
            f"missing={sorted(allowed_scenes - observed_scenes)} "
            f"extra={sorted(observed_scenes - allowed_scenes)}")

    sessions = []
    for session_id, group in table.groupby("session_id", sort=True):
        for column in (
                "scene", "episode", "kind", "goal_role", "goal_variant",
                "decision_frame", "query_path", "session_label",
                "session_max_covis"):
            _consistent(group, column)
        ordered = group.sort_values(
            ["dino_rank", "candidate_frame"], kind="stable")
        chosen = ordered.iloc[0]
        require(int(chosen["dino_rank"]) == 1,
                f"session lacks one-indexed DINO top-1: {session_id}")
        sessions.append(Session(
            session_id=str(session_id),
            scene=str(chosen["scene"]),
            episode=str(chosen["episode"]),
            kind=str(chosen["kind"]),
            goal_role=str(chosen["goal_role"]),
            goal_variant=str(chosen["goal_variant"]),
            decision_frame=int(chosen["decision_frame"]),
            query_path=Path(str(chosen["query_path"])),
            candidate_frame=int(chosen["candidate_frame"]),
            candidate_path=Path(str(chosen["candidate_path"])),
            dino_cosine=float(chosen["dino_cosine"]),
            dino_rank=int(chosen["dino_rank"]),
            candidate_label=int(chosen["candidate_label"]),
            session_label=int(chosen["session_label"]),
            session_max_covis=float(chosen["session_max_covis"]),
        ))

    if args.scene:
        wanted = set(args.scene)
        sessions = [row for row in sessions if row.scene in wanted]
        found = {row.scene for row in sessions}
        require(found == wanted, f"unknown scene(s): {sorted(wanted - found)}")
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    if args.max_episodes:
        keys = sorted({(row.scene, row.episode) for row in sessions})
        selected = set(keys[:args.max_episodes])
        sessions = [row for row in sessions
                    if (row.scene, row.episode) in selected]
    require(bool(sessions), "no sessions selected")
    return sessions


def validate_sessions(sessions: list[Session], *, num_scale: int) -> None:
    for row in sessions:
        require(row.query_path.is_file(), f"missing query: {row.query_path}")
        require(row.candidate_path.is_file(),
                f"missing candidate: {row.candidate_path}")
        require(int(row.candidate_path.stem) == row.candidate_frame,
                f"candidate filename/frame mismatch: {row.session_id}")
        require(num_scale <= row.candidate_frame < row.decision_frame,
                f"invalid causal anchor for {row.session_id}")
        require((row.rgb_dir / "0.jpg").is_file(),
                f"history has no frame zero: {row.rgb_dir}")
        require((row.rgb_dir / f"{row.decision_frame - 1}.jpg").is_file(),
                f"history prefix is incomplete: {row.session_id}")


def _stream_with_scheduled_queries(
        model: torch.nn.Module, images: torch.Tensor,
        scheduled: dict[int, list[tuple[str, str, torch.Tensor]]],
        *, num_scale: int, device: str, label: str,
        ) -> dict[tuple[str, str], dict[str, Any]]:
    """Stream once and query goals at frozen anchor/decision frames."""
    require(len(images) >= num_scale, f"{label}: prefix too short")
    require(all(num_scale <= index < len(images) for index in scheduled),
            f"{label}: scheduled query lies outside streamable frames")
    model.clean_kv_cache()
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16):
        scale = images[:num_scale][None].to(device)
        aggregated, _ = model._aggregate_features(
            scale, num_frame_for_scale=num_scale,
            num_frame_per_block=num_scale)
        poses = model.camera_head(
            aggregated, causal_inference=True,
            num_frame_per_block=num_scale,
            num_frame_for_scale=num_scale)
        del scale, aggregated, poses

        for frame_index in range(num_scale, len(images)):
            frame = images[frame_index:frame_index + 1][None].to(device)
            aggregated, _ = model._aggregate_features(
                frame, num_frame_for_scale=num_scale,
                num_frame_per_block=1)
            poses = model.camera_head(
                aggregated, causal_inference=True,
                num_frame_per_block=1,
                num_frame_for_scale=num_scale)
            reference_pose = poses[-1][0, -1].float().cpu().numpy()
            del frame, aggregated, poses
            for session_id, context, goal_image in scheduled.get(
                    frame_index, []):
                goal_pose, identity, audit = _read_only_query(
                    model, goal_image, num_scale=num_scale, device=device,
                    label=f"{context}:{session_id}")
                outputs[(session_id, context)] = {
                    "goal_pose": goal_pose,
                    "reference_pose": reference_pose.copy(),
                    "query_state_identity": bool(identity),
                    "query_audit": audit,
                }
            if ((frame_index + 1) % 50 == 0
                    or frame_index + 1 == len(images)):
                print(f"[{label}] {frame_index + 1}/{len(images)} frames "
                      f"queries={len(outputs)} elapsed="
                      f"{time.monotonic() - started:.1f}s", flush=True)
    require(len(outputs) == sum(len(value) for value in scheduled.values()),
            f"{label}: not all scheduled queries executed")
    return outputs


def circular_difference_degrees(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def cdf(errors: Iterable[float], threshold: float) -> dict[str, Any]:
    values = [float(value) for value in errors if math.isfinite(float(value))]
    hits = sum(value <= threshold for value in values)
    return {"hits": hits, "total": len(values),
            "rate": hits / len(values) if values else None}


def binary_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    pairs = [(int(label), float(score))
             for label, score in zip(labels, scores)
             if int(label) in (0, 1) and math.isfinite(float(score))]
    positive = [score for label, score in pairs if label == 1]
    negative = [score for label, score in pairs if label == 0]
    if not positive or not negative:
        return float("nan")
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive for neg in negative)
    return wins / (len(positive) * len(negative))


def bootstrap_auc_delta(
        rows: list[dict[str, Any]], *, label_key: str,
        candidate_key: str, baseline_key: str,
        resamples: int, seed: int) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[label_key] in (0, 1):
            by_scene[str(row["scene"])].append(row)
    scenes = sorted(by_scene)
    require(len(scenes) >= 2, "cluster bootstrap needs at least two scenes")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        sample = [row for scene in sampled for row in by_scene[str(scene)]]
        labels = [int(row[label_key]) for row in sample]
        candidate = binary_auc(labels, [row[candidate_key] for row in sample])
        baseline = binary_auc(labels, [row[baseline_key] for row in sample])
        if math.isfinite(candidate) and math.isfinite(baseline):
            deltas.append(candidate - baseline)
    require(bool(deltas), "all cluster bootstrap samples were degenerate")
    return {
        "resamples_requested": resamples,
        "resamples_valid": len(deltas),
        "median": float(np.median(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
    }


def bootstrap_positive_cdf30_delta(
        rows: list[dict[str, Any]], *, resamples: int,
        seed: int) -> dict[str, Any]:
    """Scene-cluster bootstrap of global minus DINO-local CDF@30."""
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["session_label"]) == 1:
            by_scene[str(row["scene"])].append(row)
    scenes = sorted(by_scene)
    require(len(scenes) >= 2, "positive CDF bootstrap needs two scenes")

    def delta(sample: list[dict[str, Any]]) -> float:
        global_rate = np.mean([
            float(row["global_direction_error_deg"] <= 30.0)
            for row in sample])
        local_rate = np.mean([
            float(row["local_direction_error_deg"] <= 30.0)
            for row in sample])
        return float(global_rate - local_rate)

    observed = [row for scene in scenes for row in by_scene[scene]]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        sample = [row for scene in sampled for row in by_scene[str(scene)]]
        deltas.append(delta(sample))
    return {
        "point_delta": delta(observed),
        "resamples": resamples,
        "median": float(np.median(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    positives = [row for row in rows if row["session_label"] == 1]
    strict_negatives = [row for row in rows if row["session_label"] == 0]
    ambiguous = [row for row in rows if row["session_label"] not in (0, 1)]
    for row in rows:
        row["existence_target"] = (
            int(row["session_label"]) if row["session_label"] in (0, 1)
            else -1)
        # Auxiliary target for the established DINO-local mechanism.  This is
        # deliberately not the primary candidate-free M2P capability target.
        row["local_direct_actionability_target"] = (
            1 if row["session_label"] == 1
            and row["local_direction_error_deg"] <= 30.0
            else 0 if row["session_label"] in (0, 1) else -1)
        row["agreement_score"] = -float(
            row["global_local_bearing_disagreement_deg"])
        row["translation_consistency_score"] = -float(
            row["global_local_translation_divergence_normalized"])

    def aucs(label_key: str) -> dict[str, float]:
        selected = [row for row in rows if row[label_key] in (0, 1)]
        labels = [int(row[label_key]) for row in selected]
        return {
            "dino_cosine": binary_auc(
                labels, [row["dino_cosine"] for row in selected]),
            "dual_context_bearing_agreement": binary_auc(
                labels, [row["agreement_score"] for row in selected]),
            "dual_context_translation_consistency": binary_auc(
                labels, [row["translation_consistency_score"]
                         for row in selected]),
        }

    local_actionability_aucs = aucs("local_direct_actionability_target")
    existence_aucs = aucs("existence_target")
    actionability_bootstrap = bootstrap_auc_delta(
        rows, label_key="local_direct_actionability_target",
        candidate_key="agreement_score", baseline_key="dino_cosine",
        resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    direction_bootstrap = bootstrap_positive_cdf30_delta(
        rows, resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 1)
    positive_local = [row["local_direction_error_deg"] for row in positives]
    positive_global = [row["global_direction_error_deg"] for row in positives]
    negative_global = [
        row["global_direction_error_deg"] for row in strict_negatives]
    summary = {
        "sessions": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "episodes": len({(row["scene"], row["episode"]) for row in rows}),
        "positive_sessions": len(positives),
        "strict_no_match_sessions": len(strict_negatives),
        "ambiguous_sessions": len(ambiguous),
        "local_raw_dino_gct_positive_bearing": {
            "median_direction_error_deg": float(np.median(positive_local)),
            "cdf_le_15": cdf(positive_local, 15.0),
            "cdf_le_30": cdf(positive_local, 30.0),
            "cdf_le_45": cdf(positive_local, 45.0),
        },
        "global_candidate_free_positive_bearing": {
            "median_direction_error_deg": float(np.median(positive_global)),
            "cdf_le_15": cdf(positive_global, 15.0),
            "cdf_le_30": cdf(positive_global, 30.0),
            "cdf_le_45": cdf(positive_global, 45.0),
        },
        # Strict no-match has no historical support.  Privileged GT direction
        # is reported only to reveal accidental Novel-direction capability or
        # hallucination; it is not used as the memory-support label.
        "global_candidate_free_strict_no_match_privileged_bearing": {
            "median_direction_error_deg": float(np.median(negative_global)),
            "cdf_le_15": cdf(negative_global, 15.0),
            "cdf_le_30": cdf(negative_global, 30.0),
            "cdf_le_45": cdf(negative_global, 45.0),
        },
        "positive_global_minus_local_cdf30_scene_bootstrap": (
            direction_bootstrap),
        "existence_auc": existence_aucs,
        "local_direct_actionability_auc": local_actionability_aucs,
        "local_actionability_agreement_minus_dino_scene_bootstrap": (
            actionability_bootstrap),
        "all_query_state_identity": all(
            bool(row["all_query_state_identity"]) for row in rows),
        "scope": (
            "train40_zero_training_representation_gate_not_closed_loop_or_sr"
        ),
    }
    summary["candidate_free_observability_gate"] = {
        "global_positive_bearing_cdf30_at_least_0p80": (
            summary["global_candidate_free_positive_bearing"]["cdf_le_30"][
                "rate"] >= 0.80),
        "all_query_state_identity": summary["all_query_state_identity"],
    }
    summary["candidate_free_observability_gate"]["passed"] = all(
        summary["candidate_free_observability_gate"][key]
        for key in (
            "global_positive_bearing_cdf30_at_least_0p80",
            "all_query_state_identity",
        ))
    summary["dino_free_replacement_gate"] = {
        "global_minus_local_cdf30_at_least_minus_0p05": (
            direction_bootstrap["point_delta"] >= -0.05),
        "cluster_ci_lower_above_minus_0p10": (
            direction_bootstrap["ci95"][0] > -0.10),
    }
    summary["dino_free_replacement_gate"]["passed"] = all(
        summary["dino_free_replacement_gate"][key]
        for key in (
            "global_minus_local_cdf30_at_least_minus_0p05",
            "cluster_ci_lower_above_minus_0p10",
        ))
    summary["raw_unsupported_signal_gate"] = {
        "agreement_local_actionability_auc_gain_at_least_0p03": (
            local_actionability_aucs["dual_context_bearing_agreement"]
            - local_actionability_aucs["dino_cosine"] >= 0.03),
        "agreement_auc_gain_cluster_ci_lower_above_zero": (
            actionability_bootstrap["ci95"][0] > 0.0),
    }
    summary["raw_unsupported_signal_gate"]["passed"] = all(
        summary["raw_unsupported_signal_gate"][key]
        for key in (
            "agreement_local_actionability_auc_gain_at_least_0p03",
            "agreement_auc_gain_cluster_ci_lower_above_zero",
        ))
    summary["authorization"] = {
        "low_capacity_scene_oof_probe": (
            summary["candidate_free_observability_gate"]["passed"]),
        "dino_free_adapter_design": (
            summary["candidate_free_observability_gate"]["passed"]
            and summary["dino_free_replacement_gate"]["passed"]),
        # This frozen-feature diagnostic can never by itself authorize a long
        # train.  A scene-OOF low-capacity probe must add information first.
        "selective_m2p_long_training": False,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path,
                        default=DEFAULT_SELECTION)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--allowed-role", default="train")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--smoke-mode", action="store_true",
        help=("collect a partial contract/timing sample without evaluating "
              "or authorizing any representation gate"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--lingbot-repo", type=Path,
                        default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-iterations", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--expected-selection-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--expected-weight-sha256")
    parser.add_argument("--expected-lingbot-revision")
    args = parser.parse_args()
    if args.weights is None:
        args.weights = args.lingbot_repo / "weights/lingbot-map-long.pt"
    for path in (args.selection_csv, args.split_manifest, args.weights):
        require(path.is_file(), f"missing required input: {path}")
    require(args.lingbot_repo.is_dir(), "LingBot repository is missing")
    require(args.max_sessions >= 0 and args.max_episodes >= 0,
            "sample limits must be non-negative")
    if args.max_sessions or args.max_episodes or args.scene:
        require(args.smoke_mode or args.preflight_only,
                "partial selections require --smoke-mode")
    require(args.bootstrap_resamples >= 1, "bootstrap count must be positive")
    if not args.preflight_only:
        require(args.device.startswith("cuda") and torch.cuda.is_available(),
                "this diagnostic requires CUDA")
    pins = (
        (args.selection_csv, args.expected_selection_sha256),
        (args.split_manifest, args.expected_split_sha256),
        (args.weights, args.expected_weight_sha256),
    )
    for path, expected in pins:
        if expected:
            require(sha256_file(path) == expected,
                    f"SHA256 mismatch for {path}")
    revision = git_revision(args.lingbot_repo)
    if args.expected_lingbot_revision:
        require(revision == args.expected_lingbot_revision,
                "LingBot revision mismatch")
    return args


def main() -> None:
    args = parse_args()
    sessions = load_sessions(args)
    validate_sessions(sessions, num_scale=args.num_scale)
    episode_groups: dict[tuple[str, str], list[Session]] = defaultdict(list)
    for row in sessions:
        episode_groups[(row.scene, row.episode)].append(row)
    preflight = {
        "status": "preflight_passed",
        "sessions": len(sessions),
        "scenes": len({row.scene for row in sessions}),
        "episodes": len(episode_groups),
        "session_labels": {
            str(label): sum(row.session_label == label for row in sessions)
            for label in sorted({row.session_label for row in sessions})
        },
        "selection_csv_sha256": sha256_file(args.selection_csv),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "weights_sha256": sha256_file(args.weights),
        "lingbot_revision": git_revision(args.lingbot_repo),
        "development_or_blind_read": False,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    args.out.mkdir(parents=True, exist_ok=False)
    model = _build_model(args)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    pose_cache: dict[Path, EpisodePoseData] = {}
    rows: list[dict[str, Any]] = []
    ordered_groups = sorted(episode_groups.items())
    for episode_index, ((scene, episode), group) in enumerate(
            ordered_groups, start=1):
        group = sorted(group, key=lambda row: row.session_id)
        root = group[0].episode_root.resolve()
        rgb_dir = group[0].rgb_dir.resolve()
        require(all(row.episode_root.resolve() == root for row in group),
                f"episode group crosses roots: {scene}/{episode}")
        maximum_decision = max(row.decision_frame for row in group)
        rgb_paths = [rgb_dir / f"{index}.jpg"
                     for index in range(maximum_decision)]
        require(all(path.is_file() for path in rgb_paths),
                f"incomplete history for {scene}/{episode}")
        print(f"[episode {episode_index}/{len(ordered_groups)}] "
              f"{scene}/{episode} frames={maximum_decision} "
              f"sessions={len(group)}", flush=True)
        images = load_and_preprocess_images(
            [os.fspath(path) for path in rgb_paths], mode="pad",
            image_size=args.image_size, patch_size=args.patch_size)
        scheduled: dict[int, list[tuple[str, str, torch.Tensor]]] = defaultdict(list)
        goal_images: dict[str, torch.Tensor] = {}
        for session in group:
            goal = load_and_preprocess_images(
                [os.fspath(session.query_path)], mode="pad",
                image_size=args.image_size, patch_size=args.patch_size)[0]
            goal_images[session.session_id] = goal
            scheduled[session.candidate_frame].append(
                (session.session_id, "local_dino_top1", goal))
            scheduled[session.decision_frame - 1].append(
                (session.session_id, "global_full_prefix", goal))
        outputs = _stream_with_scheduled_queries(
            model, images, scheduled, num_scale=args.num_scale,
            device=args.device, label=f"{scene}/{episode}")
        source_pose = load_episode_pose_data(root)
        pose_cache[root] = source_pose
        for session in group:
            local = outputs[(session.session_id, "local_dino_top1")]
            global_query = outputs[(session.session_id, "global_full_prefix")]
            current_pose = global_query["reference_pose"]
            local_direction = _lingbot_relative_direction(
                current_pose, local["goal_pose"])
            global_direction = _lingbot_relative_direction(
                current_pose, global_query["goal_pose"])
            query_pose = query_camera_to_world(session.query_path, pose_cache)
            gt = _navdp_ground_truth_relative(
                source_pose.actions[session.decision_frame - 1], query_pose,
                source_pose.base_extrinsic)
            local_bearing = signed_bearing_degrees(local_direction)
            global_bearing = signed_bearing_degrees(global_direction)
            local_norm = float(np.linalg.norm(local_direction))
            global_norm = float(np.linalg.norm(global_direction))
            goal_translation_l2 = float(np.linalg.norm(
                local["goal_pose"][:3] - global_query["goal_pose"][:3]))
            normalization = max(0.5 * (local_norm + global_norm), 1e-6)
            row = {
                "session_id": session.session_id,
                "scene": session.scene,
                "episode": session.episode,
                "kind": session.kind,
                "goal_role": session.goal_role,
                "goal_variant": session.goal_variant,
                "decision_frame": session.decision_frame,
                "query_path": os.fspath(session.query_path),
                "dino_anchor": session.candidate_frame,
                "dino_cosine": session.dino_cosine,
                "dino_rank": session.dino_rank,
                "candidate_label": session.candidate_label,
                "session_label": session.session_label,
                "session_max_covis": session.session_max_covis,
                "gt_direction": gt.tolist(),
                "gt_bearing_deg": signed_bearing_degrees(gt),
                "local_direction": local_direction.tolist(),
                "local_raw_norm": local_norm,
                "local_bearing_deg": local_bearing,
                "local_direction_error_deg": direction_error_degrees(
                    local_direction, gt),
                "global_direction": global_direction.tolist(),
                "global_raw_norm": global_norm,
                "global_bearing_deg": global_bearing,
                "global_direction_error_deg": direction_error_degrees(
                    global_direction, gt),
                "global_local_bearing_disagreement_deg": (
                    circular_difference_degrees(
                        global_bearing, local_bearing)),
                "global_local_goal_translation_l2": goal_translation_l2,
                "global_local_translation_divergence_normalized": (
                    goal_translation_l2 / normalization),
                "all_query_state_identity": bool(
                    local["query_state_identity"]
                    and global_query["query_state_identity"]),
                "query_audits": {
                    "local": local["query_audit"],
                    "global": global_query["query_audit"],
                },
            }
            rows.append(row)
        _atomic_json(args.out / "partial_rows.json", _finite_json(rows))
        del images, goal_images, outputs
        model.clean_kv_cache()
        torch.cuda.empty_cache()

    if args.smoke_mode:
        summary = {
            "scope": "partial_contract_timing_smoke_no_effectiveness_gate",
            "sessions": len(rows),
            "scenes": len({row["scene"] for row in rows}),
            "episodes": len({(row["scene"], row["episode"])
                             for row in rows}),
            "all_query_state_identity": all(
                bool(row["all_query_state_identity"]) for row in rows),
            "queries_executed": 2 * len(rows),
            "authorization": {
                "full_train40_collection": False,
                "low_capacity_scene_oof_probe": False,
                "selective_m2p_long_training": False,
            },
        }
    else:
        summary = summarize(rows, args)
    report = {
        "schema": "m2p_s1_train40_dual_context_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "selection_csv": os.fspath(args.selection_csv.resolve()),
            "selection_csv_sha256": sha256_file(args.selection_csv),
            "split_manifest": os.fspath(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "allowed_role": args.allowed_role,
            "lingbot_repo": os.fspath(args.lingbot_repo.resolve()),
            "lingbot_revision": git_revision(args.lingbot_repo),
            "weights": os.fspath(args.weights.resolve()),
            "weights_sha256": sha256_file(args.weights),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "num_scale": args.num_scale,
            "window": args.window,
            "max_frame_num": args.max_frame_num,
            "camera_iterations": args.camera_iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "development_or_blind_read": False,
            "closed_loop": False,
            "model_training": False,
            "smoke_mode": bool(args.smoke_mode),
        },
        "preflight": preflight,
        "summary": summary,
        "rows": rows,
    }
    strict = _finite_json(report)
    _atomic_json(args.out / "report.json", strict)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "query_audits"}
        for row in rows
    ]).to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(strict["summary"], indent=2, sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
