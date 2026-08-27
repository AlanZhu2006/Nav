#!/usr/bin/env python3
"""Train-only audit of NavDP's candidate consensus as an arrival signal.

NavDP zeroes every sampled trajectory whose predicted endpoint is shorter
than 0.5 m before critic ranking.  A selected zero trajectory is therefore a
*proposal* for stopping, not proof that an ImageGoal has been reached.  This
script measures whether the complete sampled set supplies a safer signal:
near a goal, many independent diffusion samples should contract; a spurious
short candidate far from the goal should not necessarily receive consensus.

The collector is deliberately offline and diagnostic.  It reads only the
frozen train40 expert streams, constructs distance-stratified states from the
active goal segment, and queries an unchanged NavDP server.  Ground-truth
distance is used only after inference for analysis.  No threshold selected by
this script is automatically authorized for GOAT or paper evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests


SCHEMA_VERSION = "navdp_arrival_consensus_train_audit_v1"
DEFAULT_SELECTION = Path(
    ".diagnostics/certificate_distilled_compass_20260813/"
    "static_top8_480_lightglue_open_set_rows.csv"
)
DEFAULT_SPLIT = Path("MemNavData/router_multiscene_split_20260805.json")
DEFAULT_OUT = Path(".diagnostics/navdp_arrival_consensus_train40_20260815")

# These bands are fixed before inference.  One deterministic state per
# available band and goal prevents long trajectories from dominating counts.
DISTANCE_BANDS = (
    ("arrived_025", 0.0, 0.25),
    ("near_miss_050", 0.25, 0.50),
    ("near_100", 0.50, 1.00),
    ("mid_200", 1.00, 2.00),
    ("far_400", 2.00, 4.00),
    ("very_far", 4.00, math.inf),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    with tempfile.NamedTemporaryFile(
            dir=str(path.parent), prefix=path.name + ".", delete=False) as tmp:
        tmp.write(payload)
        temporary = Path(tmp.name)
    os.replace(str(temporary), str(path))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=str(path.parent), prefix=path.name + ".", suffix=".csv",
            mode="w", encoding="utf-8", newline="", delete=False) as tmp:
        frame.to_csv(tmp, index=False)
        temporary = Path(tmp.name)
    os.replace(str(temporary), str(path))


def matrix(value: object, label: str) -> np.ndarray:
    try:
        result = np.stack([np.asarray(row, dtype=np.float64) for row in value])
    except Exception as error:
        raise ValueError(f"{label} is not a matrix") from error
    require(result.shape == (4, 4), f"{label} has shape {result.shape}")
    require(bool(np.isfinite(result).all()), f"{label} is non-finite")
    return result


def episode_root_from_image(path: Path) -> Path:
    path = Path(path)
    for parent in path.parents:
        if parent.name == "videos":
            return parent.parent
    raise ValueError(f"cannot resolve episode root from {path}")


@dataclass(frozen=True)
class EpisodeSource:
    scene: str
    episode: str
    root: Path


@dataclass(frozen=True)
class GoalSpec:
    goal_index: int
    name: str
    image: Path
    position_xy: np.ndarray
    segment_start: int
    segment_end: int


@dataclass(frozen=True)
class StateSpec:
    scene: str
    episode: str
    goal_index: int
    goal_name: str
    goal_image: Path
    frame_index: int
    segment_start: int
    distance_band: str
    euclidean_distance_m: float

    @property
    def state_id(self) -> str:
        return (
            f"{self.scene}/{self.episode}/goal_{self.goal_index}/"
            f"{self.distance_band}/frame_{self.frame_index:06d}"
        )


def load_episode_sources(selection_csv: Path, split_manifest: Path,
                         allowed_role: str) -> list[EpisodeSource]:
    table = pd.read_csv(selection_csv)
    required = {
        "split_role", "scene", "episode", "candidate_path", "no_future",
    }
    missing = required - set(table.columns)
    require(not missing, f"selection is missing columns: {sorted(missing)}")
    table = table.loc[table["split_role"].astype(str) == allowed_role].copy()
    require(not table.empty, f"no selection rows for role {allowed_role}")

    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    allowed_scenes = set(map(str, split[allowed_role]))
    observed_scenes = set(map(str, table["scene"].unique()))
    require(
        observed_scenes == allowed_scenes,
        "selection/split scene mismatch: missing={} extra={}".format(
            sorted(allowed_scenes - observed_scenes),
            sorted(observed_scenes - allowed_scenes),
        ),
    )

    normalized_no_future = table["no_future"].astype(str).str.lower()
    require(
        bool(normalized_no_future.isin({"true", "1"}).all()),
        "selection contains future candidates",
    )

    sources: list[EpisodeSource] = []
    for (scene, episode), group in table.groupby(
            ["scene", "episode"], sort=True):
        roots = {
            episode_root_from_image(Path(raw)).resolve()
            for raw in group["candidate_path"].astype(str)
        }
        require(
            len(roots) == 1,
            f"{scene}/{episode} resolves to multiple episode roots: {roots}",
        )
        sources.append(EpisodeSource(str(scene), str(episode), roots.pop()))
    require(
        len(sources) == 2 * len(allowed_scenes),
        f"expected two train episodes per scene, found {len(sources)}",
    )
    return sources


def load_episode(source: EpisodeSource) -> tuple[np.ndarray, np.ndarray, dict]:
    root = source.root
    metadata_path = root / "meta" / "gen_meta.json"
    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    require(parquet_path.is_file(), f"missing parquet: {parquet_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(
        parquet_path,
        columns=["action", "observation.camera_intrinsic"],
    )
    require(not frame.empty, f"empty trajectory: {parquet_path}")
    poses = np.stack([
        matrix(value, f"{source.scene}/{source.episode} action")
        for value in frame["action"]
    ])
    intrinsic_raw = frame.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack([
        np.asarray(row, dtype=np.float64) for row in intrinsic_raw
    ])
    require(intrinsic.shape == (3, 3), "camera intrinsic is not 3x3")
    require(bool(np.isfinite(intrinsic).all()), "camera intrinsic is non-finite")
    require(
        int(metadata.get("n_frames", len(frame))) == len(frame),
        f"metadata frame count differs for {source.root}",
    )
    return poses, intrinsic, metadata


def episode_goals(source: EpisodeSource, metadata: Mapping[str, Any],
                  frame_count: int) -> list[GoalSpec]:
    goals = metadata.get("goals")
    switches = metadata.get("switches")
    require(isinstance(goals, list) and goals, f"no goals in {source.root}")
    require(
        isinstance(switches, list) and len(switches) == len(goals),
        f"goal/switch mismatch in {source.root}",
    )
    result: list[GoalSpec] = []
    for offset, goal in enumerate(goals):
        require(isinstance(goal, Mapping), f"malformed goal {offset}")
        position = np.asarray(goal.get("pos"), dtype=np.float64)
        require(
            position.shape == (3,) and bool(np.isfinite(position).all()),
            f"malformed goal position {offset} in {source.root}",
        )
        start = int(switches[offset])
        end = int(switches[offset + 1]) if offset + 1 < len(switches) \
            else int(frame_count)
        require(0 <= start < end <= frame_count, "invalid active-goal segment")
        image = source.root / f"goal_{offset + 1}.jpg"
        require(image.is_file(), f"missing goal image: {image}")
        result.append(GoalSpec(
            goal_index=offset + 1,
            name=str(goal.get("name", f"goal_{offset + 1}")),
            image=image,
            position_xy=position[:2].copy(),
            segment_start=start,
            segment_end=end,
        ))
    return result


def _band_mask(distances: np.ndarray, lower: float,
               upper: float) -> np.ndarray:
    if lower == 0.0:
        return (distances >= lower) & (distances <= upper)
    if math.isinf(upper):
        return distances > lower
    return (distances > lower) & (distances <= upper)


def select_goal_states(source: EpisodeSource, goal: GoalSpec,
                       camera_xy: np.ndarray) -> list[StateSpec]:
    require(camera_xy.ndim == 2 and camera_xy.shape[1] == 2,
            "camera positions must be [N,2]")
    indices = np.arange(goal.segment_start, goal.segment_end, dtype=np.int64)
    distances = np.linalg.norm(
        camera_xy[indices] - goal.position_xy[None, :], axis=1)
    result: list[StateSpec] = []
    selected_frames: set[int] = set()
    for name, lower, upper in DISTANCE_BANDS:
        eligible_local = np.flatnonzero(_band_mask(distances, lower, upper))
        if not len(eligible_local):
            continue
        eligible_frames = indices[eligible_local]
        eligible_distances = distances[eligible_local]
        if name == "arrived_025":
            order = np.lexsort((-eligible_frames, eligible_distances))
        elif math.isinf(upper):
            order = np.lexsort((eligible_frames, eligible_distances))
        else:
            midpoint = 0.5 * (lower + upper)
            order = np.lexsort(
                (eligible_frames, np.abs(eligible_distances - midpoint)))
        chosen_local = int(order[0])
        frame_index = int(eligible_frames[chosen_local])
        require(frame_index not in selected_frames, "distance bands overlap")
        selected_frames.add(frame_index)
        result.append(StateSpec(
            scene=source.scene,
            episode=source.episode,
            goal_index=goal.goal_index,
            goal_name=goal.name,
            goal_image=goal.image,
            frame_index=frame_index,
            segment_start=goal.segment_start,
            distance_band=name,
            euclidean_distance_m=float(eligible_distances[chosen_local]),
        ))
    return result


def deterministic_seed(base_seed: int, state_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}:{state_id}:{int(sample_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def normalize_candidates(value: object) -> np.ndarray:
    candidates = np.asarray(value, dtype=np.float64)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    require(
        candidates.ndim == 3 and candidates.shape[-1] >= 2,
        f"unexpected all_trajectory shape {candidates.shape}",
    )
    require(bool(np.isfinite(candidates).all()), "non-finite trajectories")
    return candidates


def normalize_selected(value: object) -> np.ndarray:
    selected = np.asarray(value, dtype=np.float64)
    if selected.ndim == 3 and selected.shape[0] == 1:
        selected = selected[0]
    require(
        selected.ndim == 2 and selected.shape[-1] >= 2,
        f"unexpected selected trajectory shape {selected.shape}",
    )
    require(bool(np.isfinite(selected).all()), "non-finite selected trajectory")
    return selected


def normalize_values(value: object, count: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    require(values.shape == (count,), f"unexpected critic shape {values.shape}")
    require(bool(np.isfinite(values).all()), "non-finite critic values")
    return values


def summarize_rollout(payload: Mapping[str, Any], stop_threshold: float) -> dict:
    candidates = normalize_candidates(payload.get("all_trajectory"))
    selected = normalize_selected(payload.get("trajectory"))
    values = normalize_values(payload.get("all_values"), len(candidates))
    endpoint_lengths = np.linalg.norm(candidates[:, -1, :2], axis=1)
    zero = endpoint_lengths <= 1e-8
    order = np.argsort(-values, kind="stable")
    top4 = order[:min(4, len(order))]
    best_zero = float(values[zero].max()) if bool(zero.any()) else None
    best_nonzero = (
        float(values[~zero].max()) if bool((~zero).any()) else None)
    margin = (
        float(best_zero - best_nonzero)
        if best_zero is not None and best_nonzero is not None else None)
    selected_endpoint = float(np.linalg.norm(selected[-1, :2]))
    critic_max = float(values.max())
    return {
        "diffusion_seed": int(payload.get("diffusion_seed")),
        "candidate_count": int(len(candidates)),
        "selected_endpoint_m": selected_endpoint,
        "selected_zero": bool(selected_endpoint <= 1e-8),
        "candidate_zero_count": int(zero.sum()),
        "candidate_zero_fraction": float(zero.mean()),
        "top4_zero_fraction": float(zero[top4].mean()),
        "critic_max": critic_max,
        "critic_min": float(values.min()),
        "best_zero_critic": best_zero,
        "best_nonzero_critic": best_nonzero,
        "zero_over_nonzero_critic_margin": margin,
        "critic_fallback": bool(critic_max < float(stop_threshold)),
        "candidate_endpoint_mean_m": float(endpoint_lengths.mean()),
        "candidate_endpoint_median_m": float(np.median(endpoint_lengths)),
        "candidate_endpoint_max_m": float(endpoint_lengths.max()),
    }


def _post_json(session: requests.Session, url: str, payload: dict,
               timeout_s: float) -> dict:
    response = session.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    result = response.json()
    require("error" not in result, f"server error: {result.get('error')}")
    return result


def reset_navdp(session: requests.Session, base_url: str,
                intrinsic: np.ndarray, stop_threshold: float,
                seed: int, timeout_s: float) -> None:
    _post_json(session, base_url + "/navigator_reset", {
        "intrinsic": intrinsic.tolist(),
        "stop_threshold": float(stop_threshold),
        "batch_size": 1,
        "seed": int(seed),
    }, timeout_s)


def replay_image(session: requests.Session, base_url: str, image: Path,
                 timeout_s: float) -> None:
    with image.open("rb") as handle:
        response = session.post(
            base_url + "/memory_replay_step",
            files={"image": (image.name, handle, "image/jpeg")},
            timeout=timeout_s,
        )
    response.raise_for_status()
    payload = response.json()
    require("error" not in payload, f"replay error: {payload.get('error')}")
    require(payload.get("diffusion_sampled") is False,
            "memory replay unexpectedly sampled diffusion")


def plan_imagegoal(session: requests.Session, base_url: str, endpoint: str,
                   image: Path, depth: Path, goal: Path, seed: int,
                   timeout_s: float) -> dict:
    with image.open("rb") as image_handle, depth.open("rb") as depth_handle, \
            goal.open("rb") as goal_handle:
        response = session.post(
            base_url + endpoint,
            files={
                "image": (image.name, image_handle, "image/jpeg"),
                "depth": (depth.name, depth_handle, "image/png"),
                "goal": (goal.name, goal_handle, "image/jpeg"),
            },
            data={"diffusion_seed": str(int(seed))},
            timeout=timeout_s,
        )
    response.raise_for_status()
    payload = response.json()
    require("error" not in payload, f"plan error: {payload.get('error')}")
    require(int(payload.get("diffusion_seed", -1)) == int(seed),
            "server did not echo diffusion seed")
    return payload


def state_paths(source: EpisodeSource, frame_index: int) -> tuple[Path, Path]:
    video = source.root / "videos" / "chunk-000"
    rgb = video / "observation.images.rgb" / f"{frame_index}.jpg"
    depth = video / "observation.images.depth" / f"{frame_index}.png"
    require(rgb.is_file(), f"missing RGB: {rgb}")
    require(depth.is_file(), f"missing depth: {depth}")
    return rgb, depth


def collect_state(session: requests.Session, base_url: str,
                  source: EpisodeSource, state: StateSpec,
                  intrinsic: np.ndarray, *, samples_per_state: int,
                  context_frames: int, base_seed: int,
                  stop_threshold: float, timeout_s: float) -> list[dict]:
    reset_seed = deterministic_seed(base_seed, state.state_id, -1)
    reset_navdp(
        session, base_url, intrinsic, stop_threshold, reset_seed, timeout_s)
    context_start = max(state.segment_start, state.frame_index - context_frames)
    for frame_index in range(context_start, state.frame_index):
        rgb, _depth = state_paths(source, frame_index)
        replay_image(session, base_url, rgb, timeout_s)
    current_rgb, current_depth = state_paths(source, state.frame_index)

    rows = []
    for sample_index in range(samples_per_state):
        seed = deterministic_seed(base_seed, state.state_id, sample_index)
        endpoint = "/imagegoal_step" if sample_index == 0 \
            else "/imagegoal_resample"
        started = time.monotonic()
        payload = plan_imagegoal(
            session, base_url, endpoint, current_rgb, current_depth,
            state.goal_image, seed, timeout_s)
        diagnostics = summarize_rollout(payload, stop_threshold)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "state_id": state.state_id,
            "scene": state.scene,
            "episode": state.episode,
            "goal_index": state.goal_index,
            "goal_name": state.goal_name,
            "frame_index": state.frame_index,
            "segment_start": state.segment_start,
            "distance_band": state.distance_band,
            "euclidean_distance_m": state.euclidean_distance_m,
            "arrival_025": bool(state.euclidean_distance_m <= 0.25),
            "sample_index": sample_index,
            "request_latency_s": float(time.monotonic() - started),
            "memory_mutated": payload.get("memory_mutated"),
            **diagnostics,
        })
    return rows


def aggregate_states(samples: pd.DataFrame) -> pd.DataFrame:
    require(not samples.empty, "no samples to aggregate")
    rows = []
    for state_id, group in samples.groupby("state_id", sort=True):
        group = group.sort_values("sample_index", kind="stable")
        first = group.iloc[0]
        rows.append({
            "state_id": state_id,
            "scene": first["scene"],
            "episode": first["episode"],
            "goal_index": int(first["goal_index"]),
            "goal_name": first["goal_name"],
            "frame_index": int(first["frame_index"]),
            "distance_band": first["distance_band"],
            "euclidean_distance_m": float(first["euclidean_distance_m"]),
            "arrival_025": bool(first["arrival_025"]),
            "sample_count": int(len(group)),
            "selected_zero_rate": float(group["selected_zero"].mean()),
            "selected_zero_all": bool(group["selected_zero"].all()),
            "candidate_zero_fraction_mean": float(
                group["candidate_zero_fraction"].mean()),
            "candidate_zero_fraction_min": float(
                group["candidate_zero_fraction"].min()),
            "candidate_zero_fraction_max": float(
                group["candidate_zero_fraction"].max()),
            "top4_zero_fraction_mean": float(
                group["top4_zero_fraction"].mean()),
            "critic_max_mean": float(group["critic_max"].mean()),
            "critic_fallback_rate": float(group["critic_fallback"].mean()),
            "zero_over_nonzero_margin_mean": float(
                group["zero_over_nonzero_critic_margin"].dropna().mean())
                if group["zero_over_nonzero_critic_margin"].notna().any()
                else None,
            "request_latency_s_mean": float(group["request_latency_s"].mean()),
        })
    return pd.DataFrame(rows)


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores_array)
    labels_array = labels_array[valid]
    scores_array = scores_array[valid]
    positives = int(labels_array.sum())
    negatives = int((~labels_array).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores_array).rank(method="average").to_numpy()
    rank_sum = float(ranks[labels_array].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int((labels & predictions).sum())
    fp = int((~labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    tn = int((~labels & ~predictions).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "accept_count": int(predictions.sum()),
    }


def build_report(samples: pd.DataFrame, states: pd.DataFrame,
                 *, selection_csv: Path, split_manifest: Path,
                 samples_per_state: int, stop_threshold: float,
                 started_unix: float, source_episode_count: int | None = None,
                 episode_start_index: int = 0) -> dict:
    labels = states["arrival_025"].astype(bool).to_numpy()
    operating_points = []
    for persistence in (0.25, 0.50, 0.75, 1.00):
        for candidate_fraction in (0.00, 0.125, 0.25, 0.50, 0.75, 1.00):
            prediction = (
                (states["selected_zero_rate"].to_numpy() >= persistence)
                & (states["candidate_zero_fraction_mean"].to_numpy()
                   >= candidate_fraction)
            )
            operating_points.append({
                "selected_zero_rate_min": persistence,
                "candidate_zero_fraction_mean_min": candidate_fraction,
                **confusion(labels, prediction),
            })

    by_band = {}
    for band, group in states.groupby("distance_band", sort=False):
        by_band[str(band)] = {
            "states": int(len(group)),
            "scenes": int(group["scene"].nunique()),
            "distance_min_m": float(group["euclidean_distance_m"].min()),
            "distance_median_m": float(group["euclidean_distance_m"].median()),
            "distance_max_m": float(group["euclidean_distance_m"].max()),
            "selected_zero_rate_mean": float(
                group["selected_zero_rate"].mean()),
            "candidate_zero_fraction_mean": float(
                group["candidate_zero_fraction_mean"].mean()),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "scope": "train-only offline mechanism audit; not a deployment rule",
        "method_or_threshold_authorized": False,
        "goat_validation_read": False,
        "selection_csv": str(Path(selection_csv).resolve()),
        "selection_sha256": sha256_file(selection_csv),
        "split_manifest": str(Path(split_manifest).resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
        "scene_count": int(states["scene"].nunique()),
        "episode_count": int(states[["scene", "episode"]].drop_duplicates().shape[0]),
        "source_episode_count": (
            int(source_episode_count) if source_episode_count is not None
            else int(states[["scene", "episode"]].drop_duplicates().shape[0])
        ),
        "episode_start_index": int(episode_start_index),
        "goal_count": int(states[["scene", "episode", "goal_index"]]
                          .drop_duplicates().shape[0]),
        "state_count": int(len(states)),
        "sample_count": int(len(samples)),
        "samples_per_state": int(samples_per_state),
        "arrival_state_count": int(labels.sum()),
        "nonarrival_state_count": int((~labels).sum()),
        "navdp_stop_threshold_diagnostic": float(stop_threshold),
        "auc": {
            "selected_zero_rate": roc_auc(
                labels, states["selected_zero_rate"]),
            "candidate_zero_fraction_mean": roc_auc(
                labels, states["candidate_zero_fraction_mean"]),
            "top4_zero_fraction_mean": roc_auc(
                labels, states["top4_zero_fraction_mean"]),
        },
        "by_distance_band": by_band,
        "predeclared_operating_point_grid": operating_points,
        "runtime_s": float(time.time() - started_unix),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--allowed-role", default="train")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:8888")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples-per-state", type=int, default=4)
    parser.add_argument("--context-frames", type=int, default=7)
    parser.add_argument("--base-seed", type=int, default=2026081501)
    parser.add_argument("--stop-threshold", type=float, default=-0.5)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-start-index", type=int, default=0)
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.samples_per_state >= 1, "samples-per-state must be positive")
    require(args.context_frames >= 0, "context-frames must be non-negative")
    require(args.max_episodes >= 0, "max-episodes must be non-negative")
    require(args.episode_start_index >= 0,
            "episode-start-index must be non-negative")
    require(args.selection_csv.is_file(), "selection CSV is missing")
    require(args.split_manifest.is_file(), "split manifest is missing")

    sources = load_episode_sources(
        args.selection_csv, args.split_manifest, args.allowed_role)
    source_episode_count = len(sources)
    require(
        args.episode_start_index < source_episode_count,
        "episode-start-index lies beyond the frozen source list",
    )
    sources = sources[args.episode_start_index:]
    if args.max_episodes:
        sources = sources[:args.max_episodes]
    require(sources, "no episodes selected")

    started = time.time()
    all_samples: list[dict] = []
    inventory = []
    http = requests.Session()
    for source_index, source in enumerate(sources, start=1):
        poses, intrinsic, metadata = load_episode(source)
        goals = episode_goals(source, metadata, len(poses))
        camera_xy = poses[:, :2, 3]
        episode_states = [
            state
            for goal in goals
            for state in select_goal_states(source, goal, camera_xy)
        ]
        inventory.append({
            "scene": source.scene,
            "episode": source.episode,
            "root": str(source.root),
            "goal_count": len(goals),
            "state_count": len(episode_states),
            "states": [state.state_id for state in episode_states],
        })
        print(
            f"[{source_index}/{len(sources)}] {source.scene}/{source.episode}: "
            f"{len(goals)} goals, {len(episode_states)} states",
            flush=True,
        )
        if args.selection_only:
            continue
        for state in episode_states:
            all_samples.extend(collect_state(
                http, args.navdp_url.rstrip("/"), source, state, intrinsic,
                samples_per_state=args.samples_per_state,
                context_frames=args.context_frames,
                base_seed=args.base_seed,
                stop_threshold=args.stop_threshold,
                timeout_s=args.request_timeout_s,
            ))
        # A checkpoint is diagnostic only; final hashes are written at end.
        atomic_csv(args.out / "samples.partial.csv", pd.DataFrame(all_samples))

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "selection_only": bool(args.selection_only),
        "episodes": inventory,
    })
    if args.selection_only:
        return

    samples = pd.DataFrame(all_samples)
    states = aggregate_states(samples)
    report = build_report(
        samples, states,
        selection_csv=args.selection_csv,
        split_manifest=args.split_manifest,
        samples_per_state=args.samples_per_state,
        stop_threshold=args.stop_threshold,
        started_unix=started,
        source_episode_count=source_episode_count,
        episode_start_index=args.episode_start_index,
    )
    atomic_csv(args.out / "samples.csv", samples)
    atomic_csv(args.out / "states.csv", states)
    atomic_json(args.out / "report.json", report)
    partial = args.out / "samples.partial.csv"
    if partial.exists():
        partial.unlink()
    hashes = {
        name: sha256_file(args.out / name)
        for name in ("inventory.json", "samples.csv", "states.csv", "report.json")
    }
    atomic_json(args.out / "SHA256SUMS.json", hashes)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
