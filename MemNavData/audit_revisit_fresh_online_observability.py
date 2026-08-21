#!/usr/bin/env python3
"""Audit whether fresh Revisit goals were observable on the actual online A trace.

The fresh-160 generator labels Goal B against an expert A trajectory, while the
closed-loop evaluation stores and replays a NavDP online-A trajectory.  This
supplemental audit recomputes the generator's occlusion-aware 3-D co-visibility
against the *actual* trace.  Outcome CSVs are loaded only after observability
rows have been constructed with frozen, outcome-independent thresholds.

Two modes are supported:

* formal: ``--manifest --run-root --out`` audits every frozen episode and, when
  all three arm outputs exist, recomputes stratified paired effects;
* smoke: ``--episode-dir --trace --scene-asset --out`` audits one local trace
  without claiming anything about the formal 160-episode run.

This program intentionally does not import the episode generator's rendering or
co-visibility helpers.  The relevant camera/geometry contract is reproduced
below so that the supplemental measurement is independently inspectable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
except ModuleNotFoundError:  # direct ``python MemNavData/<script>.py`` use
    from deterministic_eval_protocol import validate_leg1_trace


W, H = 480, 270
FX, FY, CX, CY = 355.81464, 351.687, 240.0, 135.0
HFOV_DEG = float(np.degrees(2.0 * np.arctan(CX / FX)))
M_W = np.asarray([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]], dtype=float)

# Frozen before reading outcome rows.  These are the generator's native
# negative, acceptance, and retrieval-positive co-visibility boundaries.
NO_SUPPORT_THRESHOLD = 0.10
ONLINE_REVISIT_THRESHOLD = 0.20
STRONG_SUPPORT_THRESHOLD = 0.50
CAMERA_HEIGHT_DEFAULT_M = 0.50
DEPTH_TOLERANCE_M = 0.30
BACKPROJECT_STRIDE = 6
FRESH_ARMS = ("geometry_router", "known_revisit_direct", "native")
FRESH_CONTRASTS = {
    "direct_minus_geometry": ("geometry_router", "known_revisit_direct"),
    "direct_minus_native": ("native", "known_revisit_direct"),
    "geometry_minus_native": ("native", "geometry_router"),
}
CERTIFIED_ARMS = (
    "certified_relocalization",
    "known_revisit_direct",
    "geometry_router",
    "native",
)
CERTIFIED_CONTRASTS = {
    "certified_minus_native": ("native", "certified_relocalization"),
    "certified_minus_geometry": (
        "geometry_router", "certified_relocalization"),
    "certified_minus_known_revisit_direct": (
        "known_revisit_direct", "certified_relocalization"),
    "direct_minus_native": ("native", "known_revisit_direct"),
    "geometry_minus_native": ("native", "geometry_router"),
}
CERTIFIED_WILLIAMS_ORDERS = (
    ("certified_relocalization", "known_revisit_direct", "native",
     "geometry_router"),
    ("known_revisit_direct", "geometry_router", "certified_relocalization",
     "native"),
    ("geometry_router", "native", "known_revisit_direct",
     "certified_relocalization"),
    ("native", "certified_relocalization", "geometry_router",
     "known_revisit_direct"),
)

# Backward-compatible aliases used by existing focused tests/importers.
ARMS = FRESH_ARMS
CONTRASTS = FRESH_CONTRASTS


def run_contract_spec(name: str) -> dict[str, Any]:
    if name == "fresh_confirmation":
        return {
            "arms": FRESH_ARMS,
            "contrasts": FRESH_CONTRASTS,
            "orders": tuple(itertools.permutations(FRESH_ARMS)),
            "scene_contract_schema": None,
            "audit_id": "revisit_fresh_online_a_observability_v1_20260813",
            "scope": (
                "supplemental label-contract audit of the consumed fresh160 "
                "run; not a fresh-scene or blind confirmation"),
        }
    if name == "certified_relocalization":
        return {
            "arms": CERTIFIED_ARMS,
            "contrasts": CERTIFIED_CONTRASTS,
            "orders": CERTIFIED_WILLIAMS_ORDERS,
            "scene_contract_schema": "certified_relocalization_closed_loop_v1",
            "audit_id": (
                "certified_relocalization_online_a_observability_v1_20260813"),
            "scope": (
                "supplemental actual-online-A label audit of the consumed "
                "certified relocalization closed-loop run; not fresh-scene "
                "or blind confirmation"),
        }
    raise ValueError(f"unsupported run contract: {name!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def finite_float(value: Any, field: str) -> float:
    require(not isinstance(value, bool), f"{field} must be numeric")
    converted = float(value)
    require(math.isfinite(converted), f"{field} must be finite")
    return converted


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    require(normalized in {"true", "false", "1", "0", "1.0", "0.0"},
            f"invalid boolean receipt: {value!r}")
    return normalized in {"true", "1", "1.0"}


def data_to_habitat(position: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(position), dtype=float)
    require(value.shape == (3,) and np.isfinite(value).all(),
            "stored position must contain three finite values")
    return M_W.T @ value


def camera_to_world(position: np.ndarray, yaw: float) -> np.ndarray:
    """Habitat/OpenGL camera-to-world transform at a floor-relative yaw."""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[c, 0.0, s],
                           [0.0, 1.0, 0.0],
                           [-s, 0.0, c]], dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def backproject(
    depth: np.ndarray,
    *,
    stride: int = BACKPROJECT_STRIDE,
    depth_min: float = 0.15,
    depth_max: float = 10.0,
) -> np.ndarray:
    """Back-project metric depth into Habitat's OpenGL camera frame."""
    value = np.asarray(depth, dtype=float)
    require(value.shape == (H, W), f"depth shape changed: {value.shape}")
    require(isinstance(stride, int) and stride > 0, "stride must be positive")
    vs, us = np.meshgrid(
        np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    sampled = value[vs, us]
    valid = (sampled > depth_min) & (sampled < depth_max)
    u = us[valid].astype(float)
    v = vs[valid].astype(float)
    distance = sampled[valid]
    x = (u - CX) / FX * distance
    y = (v - CY) / FY * distance
    return np.stack([x, -y, -distance], axis=1)


def to_world(points_camera: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=float)
    require(points.ndim == 2 and points.shape[1] == 3,
            "camera points must have shape [N,3]")
    return points @ transform[:3, :3].T + transform[:3, 3]


def covisibility_fraction(
    points_world: np.ndarray,
    transform: np.ndarray,
    depth: np.ndarray,
    *,
    tolerance_m: float = DEPTH_TOLERANCE_M,
) -> float:
    """Generator-equivalent occlusion-aware surface co-visibility fraction."""
    points = np.asarray(points_world, dtype=float)
    if len(points) == 0:
        return 0.0
    rendered = np.asarray(depth, dtype=float)
    require(rendered.shape == (H, W), f"depth shape changed: {rendered.shape}")
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    x, y, z = camera[:, 0], -camera[:, 1], -camera[:, 2]
    valid = z > 0.05
    safe_z = np.maximum(z, 1e-6)
    u = FX * x / safe_z + CX
    v = FY * y / safe_z + CY
    valid &= (u >= 0.0) & (u < W - 1) & (v >= 0.0) & (v < H - 1)
    ui = np.clip(u.astype(int), 0, W - 1)
    vi = np.clip(v.astype(int), 0, H - 1)
    consistent = valid & (np.abs(z - rendered[vi, ui]) <= tolerance_m)
    return float(consistent.sum()) / float(len(points))


def support_band(maximum: float) -> str:
    value = finite_float(maximum, "online_max_covis")
    if value < NO_SUPPORT_THRESHOLD:
        return "no_support_lt_0p10"
    if value < ONLINE_REVISIT_THRESHOLD:
        return "ambiguous_0p10_to_0p20"
    if value < STRONG_SUPPORT_THRESHOLD:
        return "supported_0p20_to_0p50"
    return "strong_support_ge_0p50"


class HabitatRenderer:
    """Minimal RGB-D renderer matching the frozen generator camera contract."""

    def __init__(self, scene_asset: Path):
        require(scene_asset.is_file(), f"missing scene asset: {scene_asset}")
        try:
            import habitat_sim
            import magnum as mn
            import quaternion
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError(
                "Habitat rendering dependencies are unavailable; run in the "
                "project habitat environment") from exc
        self._habitat_sim = habitat_sim
        self._quaternion = quaternion
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene_asset)
        backend.enable_physics = False

        def camera(uuid: str, sensor_type: Any) -> Any:
            specification = habitat_sim.CameraSensorSpec()
            specification.uuid = uuid
            specification.sensor_type = sensor_type
            specification.resolution = [H, W]
            specification.hfov = HFOV_DEG
            specification.position = mn.Vector3(0.0, 0.0, 0.0)
            return specification

        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [
            camera("color", habitat_sim.SensorType.COLOR),
            camera("depth", habitat_sim.SensorType.DEPTH),
        ]
        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent]))

    def render(self, position: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        state = self._habitat_sim.agent.AgentState()
        state.position = np.asarray(position, dtype=float)
        state.rotation = self._quaternion.from_rotation_vector(
            [0.0, float(yaw), 0.0])
        self.sim.get_agent(0).set_state(state)
        observations = self.sim.get_sensor_observations()
        return (
            np.asarray(observations["color"])[..., :3].copy(),
            np.asarray(observations["depth"], dtype=float).copy(),
        )

    def close(self) -> None:
        self.sim.close()

    def __enter__(self) -> "HabitatRenderer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def validate_episode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    require(int(metadata.get("n_legs", -1)) == 2, "episode is not 2-leg")
    goals = metadata.get("goals")
    require(isinstance(goals, list) and len(goals) == 1,
            "2-leg episode must contain exactly Goal B")
    goal = goals[0]
    require(goal.get("name") == "B" and goal.get("kind") == "revisit",
            "Goal B is not generator-labelled Revisit")
    require(np.isclose(
        finite_float(metadata.get("covis_pos_lo"), "covis_pos_lo"),
        NO_SUPPORT_THRESHOLD, rtol=0.0, atol=1e-12),
        "generator negative co-visibility threshold changed")
    covis_band = metadata.get("covis_band")
    require(isinstance(covis_band, list) and len(covis_band) == 2,
            "missing generator co-visibility band")
    require(np.isclose(
        finite_float(covis_band[0], "covis_band[0]"),
        ONLINE_REVISIT_THRESHOLD, rtol=0.0, atol=1e-12),
        "generator Revisit acceptance threshold changed")
    require(np.isclose(
        finite_float(metadata.get("covis_pos_hi"), "covis_pos_hi"),
        STRONG_SUPPORT_THRESHOLD, rtol=0.0, atol=1e-12),
        "generator positive co-visibility threshold changed")
    return goal


def audit_episode(
    *,
    scene: str,
    episode: str,
    episode_dir: Path,
    trace_path: Path,
    renderer: HabitatRenderer,
    expected_seed: int | None = None,
    expected_trace_sha256: str | None = None,
    verify_goal_render_hash: bool = False,
) -> dict[str, Any]:
    """Measure actual online-A/Goal-B co-visibility for one episode."""
    metadata_path = episode_dir / "meta" / "gen_meta.json"
    goal_path = episode_dir / "goal_1.jpg"
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    require(goal_path.is_file(), f"missing Goal-B image: {goal_path}")
    require(trace_path.is_file(), f"missing online-A trace: {trace_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    goal = validate_episode_metadata(metadata)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    switch_index = int(metadata.get("switch_idx", -1))
    require(switch_index > 0, "invalid expert Goal-A switch index")
    goal_a_path = (episode_dir / "videos" / "chunk-000" /
                   "observation.images.rgb" / f"{switch_index - 1}.jpg")
    require(goal_a_path.is_file(), f"missing expert Goal-A image: {goal_a_path}")
    validate_leg1_trace(
        trace,
        expected_episode=episode,
        expected_seed=expected_seed,
        expected_goal_sha256=sha256_file(goal_a_path),
        expected_source_scene=scene,
    )
    require(trace.get("goal_source_episode") == episode,
            "online-A trace uses another episode's Goal A")
    require(trace.get("source_backend") == "hybrid_pose",
            "online-A trace was not produced through the frozen hybrid source")
    require(trace.get("source_hybrid_route") == "phase",
            "online-A source did not use the native phase route")
    require(trace.get("source_retrieval_candidate_min_gap") == 16,
            "online-A source candidate-gap contract changed")
    require(np.isclose(
        float(trace.get("source_graph_subgoal_spacing_m")), 0.0,
        rtol=0.0, atol=1e-12), "online-A source was not direct NavDP")
    require(np.isclose(
        float(trace.get("source_graph_subgoal_arrival_m")), 0.6,
        rtol=0.0, atol=1e-12), "online-A graph-arrival receipt changed")
    trace_sha = sha256_file(trace_path)
    if expected_trace_sha256 is not None:
        require(trace_sha == expected_trace_sha256,
                f"trace SHA mismatch: {scene}/{episode}")

    camera_height = finite_float(
        metadata.get("camera_height_m", CAMERA_HEIGHT_DEFAULT_M),
        "camera_height_m")
    require(0.0 < camera_height < 2.0, "implausible camera height")
    goal_floor = data_to_habitat(goal["pos"])
    goal_yaw = finite_float(goal["yaw_habitat"], "goal yaw")
    goal_camera = goal_floor + np.asarray([0.0, camera_height, 0.0])
    goal_rgb, goal_depth = renderer.render(goal_camera, goal_yaw)
    goal_transform = camera_to_world(goal_camera, goal_yaw)
    goal_points_world = to_world(
        backproject(goal_depth, stride=BACKPROJECT_STRIDE), goal_transform)
    rendered_goal_sha = sha256_bytes(jpeg_bytes(goal_rgb))
    stored_goal_sha = sha256_file(goal_path)
    goal_render_hash_match = rendered_goal_sha == stored_goal_sha
    if verify_goal_render_hash:
        require(goal_render_hash_match,
                f"rendered Goal-B JPEG mismatch: {scene}/{episode}")

    curve: list[float] = []
    positions: list[np.ndarray] = []
    yaws: list[float] = []
    rendered_trace_hash_matches = 0
    for pose in trace["poses"]:
        floor_position = np.asarray([
            finite_float(pose["x"], "trace x"),
            finite_float(pose["y"], "trace y"),
            finite_float(pose["z"], "trace z"),
        ], dtype=float)
        yaw = finite_float(pose["yaw"], "trace yaw")
        camera_position = floor_position + np.asarray(
            [0.0, camera_height, 0.0])
        rgb, depth = renderer.render(camera_position, yaw)
        transform = camera_to_world(camera_position, yaw)
        curve.append(covisibility_fraction(
            goal_points_world, transform, depth,
            tolerance_m=DEPTH_TOLERANCE_M))
        rendered_trace_hash_matches += int(
            sha256_bytes(jpeg_bytes(rgb)) == pose["jpg_sha256"])
        positions.append(floor_position)
        yaws.append(yaw)

    if curve:
        maximum = float(max(curve))
        argmax = int(np.argmax(curve))
        nearest_distances = [
            float(np.linalg.norm(position - goal_floor))
            for position in positions
        ]
        nearest_index = int(np.argmin(nearest_distances))
        nearest_distance = nearest_distances[nearest_index]
        yaw_error = abs(math.degrees(
            (yaws[argmax] - goal_yaw + math.pi) % (2.0 * math.pi) - math.pi))
        recall_gap = len(curve) - 1 - argmax
    else:
        maximum = 0.0
        argmax = -1
        nearest_index = -1
        nearest_distance = None
        yaw_error = None
        recall_gap = None

    return {
        "scene": scene,
        "episode": episode,
        "trace_sha256": trace_sha,
        "trace_reached_a": bool(trace["reached"]),
        "online_frame_count": len(curve),
        "expert_max_covis": finite_float(goal["covis"], "expert covis"),
        "expert_covis_argmax": int(goal["covis_argmax"]),
        "expert_recall_gap": int(goal["recall_gap"]),
        "online_max_covis": maximum,
        "online_covis_argmax": argmax,
        "online_recall_gap": recall_gap,
        "online_frames_ge_0p10": sum(
            value >= NO_SUPPORT_THRESHOLD for value in curve),
        "online_frames_ge_0p20": sum(
            value >= ONLINE_REVISIT_THRESHOLD for value in curve),
        "online_frames_ge_0p50": sum(
            value >= STRONG_SUPPORT_THRESHOLD for value in curve),
        "online_support_band": support_band(maximum),
        "online_revisit_supported": maximum >= ONLINE_REVISIT_THRESHOLD,
        "online_revisit_strong": maximum >= STRONG_SUPPORT_THRESHOLD,
        "online_path_nearest_index": nearest_index,
        "online_path_nearest_distance_m": nearest_distance,
        "online_argmax_goal_yaw_error_deg": yaw_error,
        "goal_surface_point_count": int(len(goal_points_world)),
        "goal_render_jpeg_hash_match": goal_render_hash_match,
        "rendered_trace_jpeg_hash_matches": rendered_trace_hash_matches,
        "rendered_trace_jpeg_hash_total": len(curve),
        "camera_height_m": camera_height,
        "covis_curve": [round(value, 6) for value in curve],
    }


def validate_manifest_episode(
    manifest: dict[str, Any], scene: str, record: dict[str, Any]
) -> Path:
    episode_root = Path(manifest.get(
        "_episode_root_override", manifest["paths"]["episode_root"]))
    episode_dir = episode_root / scene / record["episode"]
    files = {
        "metadata": episode_dir / "meta" / "gen_meta.json",
        "parquet": episode_dir / "data" / "chunk-000" /
                   "episode_000000.parquet",
        "goal": episode_dir / "goal_image.jpg",
        "goal_alias": episode_dir / "goal_1.jpg",
    }
    for name, path in files.items():
        require(path.is_file() and not path.is_symlink(),
                f"missing manifest episode file: {path}")
        expected = record["files"][name]
        require(path.stat().st_size == int(expected["bytes"]),
                f"manifest byte count mismatch: {path}")
        require(sha256_file(path) == expected["sha256"],
                f"manifest SHA mismatch: {path}")
    return episode_dir


def resolve_scene_asset(manifest: dict[str, Any], scene: str) -> Path:
    root = Path(manifest.get(
        "_asset_root_override", manifest["paths"]["asset_root"]))
    candidates = [root / scene / f"{scene}.glb", root / f"{scene}.glb"]
    existing = [path for path in candidates if path.is_file()]
    require(len(existing) == 1,
            f"could not uniquely resolve scene asset for {scene}: {candidates}")
    path = existing[0]
    record = manifest["assets"][scene]
    require(path.stat().st_size == int(record["bytes"]),
            f"asset byte count mismatch: {scene}")
    require(sha256_file(path) == record["sha256"],
            f"asset SHA mismatch: {scene}")
    return path


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = min(int(gains), int(losses))
    mass = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * mass / (2 ** discordant))


def load_outcomes(
    manifest: dict[str, Any], run_root: Path,
    observability: dict[tuple[str, str], dict], *, run_contract: str,
) -> dict[str, dict[tuple[str, str], dict[str, bool]]]:
    """Load raw outcomes only after all observability rows are frozen."""
    spec = run_contract_spec(run_contract)
    arms = spec["arms"]
    orders = spec["orders"]
    scenes = manifest["scenes"]
    base_seed = int(manifest["evaluation"]["base_seed"])
    outputs: dict[str, dict[tuple[str, str], dict[str, bool]]] = {
        arm: {} for arm in arms}
    for scene_index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{scene_index:02d}_{scene}"
        contract_path = scene_root / "scene_contract.json"
        require(contract_path.is_file(), f"missing scene contract: {scene}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        require(contract.get("scene") == scene, f"scene contract mismatch: {scene}")
        require(contract.get("scene_index") == scene_index,
                f"scene index mismatch: {scene}")
        require(contract.get("arm_order") == list(
            orders[scene_index % len(orders)]),
            f"arm order mismatch: {scene}")
        if spec["scene_contract_schema"] is not None:
            require(contract.get("schema_version") ==
                    spec["scene_contract_schema"],
                    f"scene contract schema mismatch: {scene}")
        require(contract.get("manifest_sha256") == sha256_file(
            Path(manifest["_manifest_path"])),
            f"scene contract manifest receipt mismatch: {scene}")
        episode_ids = [row["episode"] for row in manifest["episodes"][scene]]
        for arm in arms:
            metric_path = scene_root / arm / "metric.csv"
            require(metric_path.is_file(), f"missing metric CSV: {metric_path}")
            with metric_path.open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            require([row["episode"] for row in metrics] == episode_ids,
                    f"metric episode identity/order mismatch: {scene}/{arm}")
            for episode_index, metric in enumerate(metrics):
                key = (scene, metric["episode"])
                require(key in observability, f"outcome without audit row: {key}")
                require(int(metric["seed"]) == base_seed + episode_index,
                        f"outcome seed mismatch: {arm}/{key}")
                require(truth(metric["deterministic_plan_seeds"]),
                        f"deterministic plan seeds disabled: {arm}/{key}")
                require(metric["leg1_trace_sha256"] ==
                        observability[key]["trace_sha256"],
                        f"outcome trace SHA mismatch: {arm}/{key}")
                require(metric["leg1_goal_source"] == "own" and
                        metric["leg1_goal_source_episode"] == key[1],
                        f"Goal-A source mismatch: {arm}/{key}")
                expected_backend = "navdp" if arm == "native" else "hybrid_pose"
                expected_route = {
                    "geometry_router": "memory_geometry",
                    "certified_relocalization": "certified_relocalization",
                }.get(arm, "phase")
                expected_adapter = (
                    "verified_bearing_v1"
                    if arm == "certified_relocalization"
                    else "legacy_metric")
                require(metric["server_backend"] == expected_backend,
                        f"backend mismatch: {arm}/{key}")
                require(metric["hybrid_route"] == expected_route,
                        f"route mismatch: {arm}/{key}")
                require(metric["revisit_adapter"] == expected_adapter,
                        f"adapter contract mismatch: {arm}/{key}")
                if arm == "certified_relocalization":
                    require(np.isclose(
                        finite_float(
                            metric["revisit_adapter_fixed_radius_m"],
                            "certified fixed radius"),
                        2.5, rtol=0.0, atol=1e-12),
                        f"certified fixed-radius contract changed: {key}")
                require(metric["retrieval_override"] == "off",
                        f"oracle retrieval was enabled: {arm}/{key}")
                reached_a = truth(metric["reached_A"])
                reached_b = truth(metric["reached_B"])
                require(reached_a == observability[key]["trace_reached_a"],
                        f"trace/metric A outcome mismatch: {arm}/{key}")
                outputs[arm][key] = {
                    "reached_a": reached_a,
                    "reached_b": reached_b,
                    "joint": reached_a and reached_b,
                }
    expected = set(observability)
    require(all(set(outputs[arm]) == expected for arm in arms),
            "arm outcome keys differ from observability rows")
    for key in sorted(expected):
        require(len({outputs[arm][key]["reached_a"] for arm in arms}) == 1,
                f"shared-A outcome differs across arms: {key}")
    return outputs


def cluster_interval(
    keys: list[tuple[str, str]],
    left: dict[tuple[str, str], dict[str, bool]],
    right: dict[tuple[str, str], dict[str, bool]],
    *,
    seed: int,
    resamples: int,
) -> list[float] | None:
    if not keys:
        return None
    scenes = sorted({key[0] for key in keys})
    numerators = np.asarray([
        sum(int(right[key]["reached_b"]) - int(left[key]["reached_b"])
            for key in keys if key[0] == scene)
        for scene in scenes
    ], dtype=float)
    denominators = np.asarray([
        sum(key[0] == scene for key in keys) for scene in scenes
    ], dtype=float)
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, resamples, 10_000):
        count = min(10_000, resamples - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        denominator = denominators[indices].sum(axis=1)
        valid = denominator > 0
        chunks.append(
            numerators[indices].sum(axis=1)[valid] / denominator[valid])
    samples = np.concatenate(chunks)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def paired_effect(
    keys: list[tuple[str, str]],
    left_name: str,
    right_name: str,
    outcomes: dict[str, dict[tuple[str, str], dict[str, bool]]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    left, right = outcomes[left_name], outcomes[right_name]
    gains, losses, both, neither = [], [], [], []
    for key in keys:
        lval, rval = left[key]["reached_b"], right[key]["reached_b"]
        if lval and rval:
            both.append(key)
        elif rval:
            gains.append(key)
        elif lval:
            losses.append(key)
        else:
            neither.append(key)
    total = len(keys)
    return {
        "left": left_name,
        "right": right_name,
        "eligible": total,
        "left_successes": sum(left[key]["reached_b"] for key in keys),
        "right_successes": sum(right[key]["reached_b"] for key in keys),
        "risk_difference_right_minus_left": (
            (len(gains) - len(losses)) / total if total else None),
        "gains": [{"scene": key[0], "episode": key[1]} for key in gains],
        "losses": [{"scene": key[0], "episode": key[1]} for key in losses],
        "both": len(both),
        "neither": len(neither),
        "mcnemar_exact_two_sided_p": exact_mcnemar(
            len(gains), len(losses)),
        "scene_cluster_bootstrap_risk_difference_95": cluster_interval(
            keys, left, right, seed=seed, resamples=resamples),
    }


def summarize_population(
    name: str,
    keys: list[tuple[str, str]],
    outcomes: dict[str, dict[tuple[str, str], dict[str, bool]]],
    *,
    seed: int,
    resamples: int,
    arms: tuple[str, ...] = FRESH_ARMS,
    contrasts: dict[str, tuple[str, str]] = FRESH_CONTRASTS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "eligible": len(keys),
        "scene_count": len({key[0] for key in keys}),
        "arms": {},
        "contrasts": {},
    }
    for arm in arms:
        successes = sum(outcomes[arm][key]["reached_b"] for key in keys)
        result["arms"][arm] = {
            "successes": successes,
            "success_rate": successes / len(keys) if keys else None,
        }
    for offset, (contrast, (left, right)) in enumerate(contrasts.items()):
        result["contrasts"][contrast] = paired_effect(
            keys, left, right, outcomes,
            seed=seed + offset, resamples=resamples)
    return result


def summarize_observability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    maxima = [float(row["online_max_covis"]) for row in rows]
    reached = [row for row in rows if row["trace_reached_a"]]
    return {
        "episodes": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "shared_a_successes": len(reached),
        "online_supported_ge_0p20_all": sum(
            row["online_revisit_supported"] for row in rows),
        "online_supported_ge_0p20_given_a_success": sum(
            row["online_revisit_supported"] for row in reached),
        "online_strong_ge_0p50_all": sum(
            row["online_revisit_strong"] for row in rows),
        "online_strong_ge_0p50_given_a_success": sum(
            row["online_revisit_strong"] for row in reached),
        "support_band_counts_all": {
            band: sum(row["online_support_band"] == band for row in rows)
            for band in (
                "no_support_lt_0p10",
                "ambiguous_0p10_to_0p20",
                "supported_0p20_to_0p50",
                "strong_support_ge_0p50",
            )
        },
        "online_max_covis_mean": float(np.mean(maxima)) if maxima else None,
        "online_max_covis_median": float(np.median(maxima)) if maxima else None,
        "online_max_covis_min": min(maxima, default=None),
        "online_max_covis_max": max(maxima, default=None),
        "goal_render_hash_matches": sum(
            row["goal_render_jpeg_hash_match"] for row in rows),
        "trace_render_hash_matches": sum(
            row["rendered_trace_jpeg_hash_matches"] for row in rows),
        "trace_render_hash_total": sum(
            row["rendered_trace_jpeg_hash_total"] for row in rows),
    }


def formal_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest
    run_root = args.run_root
    run_spec = run_contract_spec(args.run_contract)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Internal-only path used by receipt validation; removed from the report.
    manifest["_manifest_path"] = str(manifest_path)
    if args.episode_root_override is not None:
        manifest["_episode_root_override"] = str(args.episode_root_override)
    if args.asset_root_override is not None:
        manifest["_asset_root_override"] = str(args.asset_root_override)
    require(manifest.get("schema_version") == 1,
            "unsupported fresh manifest schema")
    require(manifest.get("audit", {}).get("status") == "ok",
            "fresh manifest audit did not pass")
    scenes = manifest.get("scenes")
    require(isinstance(scenes, list) and len(scenes) == 20,
            "formal fresh audit requires 20 scenes")
    per_scene = int(manifest.get("episodes_per_scene", -1))
    require(per_scene == 8, "formal fresh audit requires 8 episodes per scene")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if sidecar.is_file():
        tokens = sidecar.read_text(encoding="utf-8").split()
        require(tokens and tokens[0] == sha256_file(manifest_path),
                "manifest SHA sidecar mismatch")

    rows: list[dict[str, Any]] = []
    base_seed = int(manifest["evaluation"]["base_seed"])
    for scene_index, scene in enumerate(scenes):
        scene_asset = resolve_scene_asset(manifest, scene)
        scene_root = run_root / "scenes" / f"{scene_index:02d}_{scene}"
        with HabitatRenderer(scene_asset) as renderer:
            records = manifest["episodes"][scene]
            require(len(records) == per_scene,
                    f"manifest episode count mismatch: {scene}")
            for episode_index, record in enumerate(records):
                episode = record["episode"]
                episode_dir = validate_manifest_episode(
                    manifest, scene, record)
                trace_path = (scene_root / "trace_source" /
                              f"{episode}_leg1_trace.json")
                rows.append(audit_episode(
                    scene=scene,
                    episode=episode,
                    episode_dir=episode_dir,
                    trace_path=trace_path,
                    renderer=renderer,
                    expected_seed=base_seed + episode_index,
                    verify_goal_render_hash=args.require_goal_render_hash,
                ))
    require(len(rows) == 160, "formal observability audit did not produce 160 rows")
    by_key = {(row["scene"], row["episode"]): row for row in rows}
    require(len(by_key) == len(rows), "duplicate observability row")

    # Outcome files are intentionally touched only after all rows above have
    # been classified with the frozen thresholds.
    outcomes = load_outcomes(
        manifest, run_root, by_key, run_contract=args.run_contract)
    all_keys = sorted(by_key)
    a_success = [key for key in all_keys if by_key[key]["trace_reached_a"]]
    populations = {
        "shared_a_success_all": a_success,
        "actual_online_supported_ge_0p20_given_a": [
            key for key in a_success
            if by_key[key]["online_revisit_supported"]],
        "actual_online_strong_ge_0p50_given_a": [
            key for key in a_success
            if by_key[key]["online_revisit_strong"]],
        "actual_online_unsupported_lt_0p20_given_a_diagnostic": [
            key for key in a_success
            if not by_key[key]["online_revisit_supported"]],
    }
    stratified = {
        name: summarize_population(
            name, keys, outcomes,
            seed=args.bootstrap_seed + index * 10,
            resamples=args.bootstrap_resamples,
            arms=run_spec["arms"],
            contrasts=run_spec["contrasts"])
        for index, (name, keys) in enumerate(populations.items())
    }
    return {
        "schema_version": 1,
        "audit_id": run_spec["audit_id"],
        "scope": run_spec["scope"],
        "protocol": {
            "run_contract": args.run_contract,
            "outcome_independent_thresholds": {
                "no_support_lt": NO_SUPPORT_THRESHOLD,
                "actual_online_revisit_ge": ONLINE_REVISIT_THRESHOLD,
                "strong_support_ge": STRONG_SUPPORT_THRESHOLD,
            },
            "backproject_stride": BACKPROJECT_STRIDE,
            "depth_consistency_tolerance_m": DEPTH_TOLERANCE_M,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "outcomes_loaded_after_observability_classification": True,
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "run_root": str(run_root),
            "episode_root_override": (
                str(args.episode_root_override)
                if args.episode_root_override is not None else None),
            "asset_root_override": (
                str(args.asset_root_override)
                if args.asset_root_override is not None else None),
        },
        "summary": summarize_observability(rows),
        "stratified_outcomes": stratified,
        "rows": rows,
    }


def smoke_audit(args: argparse.Namespace) -> dict[str, Any]:
    episode_dir = args.episode_dir
    episode = episode_dir.name
    scene = args.scene or episode_dir.parent.name
    with HabitatRenderer(args.scene_asset) as renderer:
        row = audit_episode(
            scene=scene,
            episode=episode,
            episode_dir=episode_dir,
            trace_path=args.trace,
            renderer=renderer,
            expected_seed=args.expected_seed,
            verify_goal_render_hash=args.require_goal_render_hash,
        )
    return {
        "schema_version": 1,
        "audit_id": "revisit_online_a_observability_smoke_v1_20260813",
        "scope": "single-episode implementation smoke; no formal effect claim",
        "protocol": {
            "no_support_lt": NO_SUPPORT_THRESHOLD,
            "actual_online_revisit_ge": ONLINE_REVISIT_THRESHOLD,
            "strong_support_ge": STRONG_SUPPORT_THRESHOLD,
            "backproject_stride": BACKPROJECT_STRIDE,
            "depth_consistency_tolerance_m": DEPTH_TOLERANCE_M,
        },
        "summary": summarize_observability([row]),
        "rows": [row],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--episode-root-override", type=Path)
    parser.add_argument("--asset-root-override", type=Path)
    parser.add_argument(
        "--run-contract",
        choices=("fresh_confirmation", "certified_relocalization"),
        default="fresh_confirmation",
    )
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--scene-asset", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--require-goal-render-hash", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    formal = args.manifest is not None or args.run_root is not None
    smoke = any(value is not None for value in (
        args.episode_dir, args.trace, args.scene_asset))
    if formal == smoke:
        parser.error("choose exactly one of formal mode or smoke mode")
    if formal and (args.manifest is None or args.run_root is None):
        parser.error("formal mode requires --manifest and --run-root")
    if not formal and (args.episode_root_override is not None or
                       args.asset_root_override is not None):
        parser.error("root overrides are only valid in formal/mirror mode")
    if smoke and any(value is None for value in (
            args.episode_dir, args.trace, args.scene_asset)):
        parser.error("smoke mode requires --episode-dir, --trace, --scene-asset")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    report = formal_audit(args) if args.manifest is not None else smoke_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "audit_id": report["audit_id"],
        "summary": report["summary"],
        "stratified_outcomes": report.get("stratified_outcomes"),
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
