"""Audited geometry primitives for deterministic Habitat rollouts.

This module deliberately has no Habitat dependency.  It owns the two pieces of
geometry that otherwise tend to be duplicated (and subtly changed) by
collectors and evaluators:

* generated parquet poses use a data-frame Z-up convention, while Habitat uses
  Y-up; and
* a policy point goal is local ``(forward, left)`` at *one* planning instant,
  whereas the navigation target itself is a fixed point in the Habitat world.

The frozen-geometry contract at the end of the file is similarly narrow.  A
collector may load a content-addressed, pre-baked navmesh through
``load_pinned_navmesh_for_collector``.  It must never call
``Simulator.recompute_navmesh``: recomputation is a data-generation operation,
not a rollout operation, and can silently change every geodesic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Habitat (x, y, z), Y-up -> generated data (x, -z, y), Z-up.
HABITAT_TO_DATA_ROTATION = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
DATA_TO_HABITAT_ROTATION = HABITAT_TO_DATA_ROTATION.T
DATA_ZUP_FRAME_CONVENTION_PREFIX = "positions+parquet in data(Zup,M_W)"

FROZEN_GEOMETRY_SCHEMA_VERSION = 1

NAVMESH_FLOAT_FIELDS = (
    "agent_height",
    "agent_max_climb",
    "agent_max_slope",
    "agent_radius",
    "cell_height",
    "cell_size",
    "detail_sample_dist",
    "detail_sample_max_error",
    "edge_max_error",
    "edge_max_len",
    "region_merge_size",
    "region_min_size",
    "verts_per_poly",
)
NAVMESH_BOOL_FIELDS = (
    "filter_ledge_spans",
    "filter_low_hanging_obstacles",
    "filter_walkable_low_height_spans",
    "include_static_objects",
)
NAVMESH_SETTING_FIELDS = tuple(sorted(
    NAVMESH_FLOAT_FIELDS + NAVMESH_BOOL_FIELDS))


class PoseConventionError(ValueError):
    """A pose cannot be interpreted under the generated-data convention."""


class FrozenGeometryError(RuntimeError):
    """The runtime scene/navmesh does not match its frozen identity."""


def _matrix4(value: Any, label: str) -> np.ndarray:
    array = np.asarray(
        value.tolist() if hasattr(value, "tolist") else value,
        dtype=np.float64,
    )
    if array.size != 16:
        raise PoseConventionError(f"{label} must contain 16 values")
    result = array.reshape(4, 4).copy()
    if not np.isfinite(result).all():
        raise PoseConventionError(f"{label} contains NaN or infinity")
    if not np.allclose(
            result[3], np.asarray([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0, atol=1e-8):
        raise PoseConventionError(f"{label} is not a homogeneous transform")
    rotation = result[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0,
                              rtol=0.0, atol=1e-6)):
        raise PoseConventionError(f"{label} rotation is not in SO(3)")
    return result


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return 0.0 if result == 0.0 else result


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector of shape ({size},)")
    return result.copy()


def wrap_yaw(yaw_rad: Any) -> Any:
    """Wrap yaw to the half-open interval ``[-pi, pi)``.

    Scalars return a Python ``float``; NumPy arrays retain their shape.  In
    particular, both +pi and -pi canonicalize to -pi, preventing a 358-degree
    loss at the -179/+179 boundary.
    """
    value = np.asarray(yaw_rad, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("yaw contains NaN or infinity")
    wrapped = (value + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def relative_yaw(goal_yaw_rad: float, current_yaw_rad: float) -> float:
    """Shortest signed yaw from the current heading to the goal heading."""
    return float(wrap_yaw(float(goal_yaw_rad) - float(current_yaw_rad)))


@dataclass(frozen=True)
class HabitatPlanarPose:
    """Robot floor pose in Habitat's Y-up world frame."""

    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for field in ("x_m", "y_m", "z_m", "yaw_rad"):
            value = _finite_float(getattr(self, field), field)
            if field == "yaw_rad":
                value = float(wrap_yaw(value))
            object.__setattr__(self, field, value)

    @property
    def position(self) -> np.ndarray:
        """A defensive copy of ``[x, y, z]``."""
        return np.asarray([self.x_m, self.y_m, self.z_m], dtype=np.float64)


def _generated_camera_mount(camera_height_m: float) -> np.ndarray:
    height = _finite_float(camera_height_m, "camera_height_m", positive=True)
    mount = np.eye(4, dtype=np.float64)
    mount[:3, :3] = HABITAT_TO_DATA_ROTATION
    mount[:3, 3] = HABITAT_TO_DATA_ROTATION @ np.asarray(
        [0.0, height, 0.0], dtype=np.float64)
    return mount


def resolve_generated_camera_extrinsic(
    camera_extrinsic: Any,
    *,
    camera_height_m: float,
    frame_convention: str,
    allow_legacy_identity: bool = True,
) -> np.ndarray:
    """Return the canonical generated camera mount.

    Correct generated episodes store ``M_W`` (including the camera-height
    translation) as ``camera_extrinsic``.  Historical episodes wrote a full
    identity matrix even though ``action`` already contained ``M_W``.  Under
    the explicit generated Z-up convention only, the full legacy identity is
    interpreted as the canonical mount.  No other mount is guessed.
    """
    if not str(frame_convention).startswith(DATA_ZUP_FRAME_CONVENTION_PREFIX):
        raise PoseConventionError(
            "generated parquet pose lacks the pinned data(Zup,M_W) convention")
    supplied = _matrix4(camera_extrinsic, "camera_extrinsic")
    canonical = _generated_camera_mount(camera_height_m)
    if np.allclose(supplied, canonical, rtol=0.0, atol=1e-6):
        return canonical
    if (allow_legacy_identity
            and np.allclose(supplied, np.eye(4), rtol=0.0, atol=1e-6)):
        return canonical
    raise PoseConventionError(
        "generated camera_extrinsic is neither canonical M_W nor legacy identity")


def parquet_data_pose_to_habitat(
    action_camera_to_world: Any,
    camera_extrinsic: Any,
    *,
    camera_height_m: float,
    frame_convention: str,
    allow_legacy_identity: bool = True,
) -> HabitatPlanarPose:
    """Decode one generated parquet row into a Habitat robot floor pose.

    The loader contract is ``base_to_world = action @ inv(camera_extrinsic)``.
    Removing the mount before taking a planar heading is essential: with the
    historical identity value, the forward motion lived on the coordinate that
    old planar loaders discarded.
    """
    action = _matrix4(action_camera_to_world, "action")
    mount = resolve_generated_camera_extrinsic(
        camera_extrinsic,
        camera_height_m=camera_height_m,
        frame_convention=frame_convention,
        allow_legacy_identity=allow_legacy_identity,
    )
    base_data = action @ np.linalg.inv(mount)
    floor_habitat = DATA_TO_HABITAT_ROTATION @ base_data[:3, 3]

    # In the generated base frame +Y is forward.  Convert that world vector
    # from data Z-up to Habitat Y-up, where yaw=0 faces camera -Z.
    forward_data = base_data[:3, :3] @ np.asarray(
        [0.0, 1.0, 0.0], dtype=np.float64)
    forward_habitat = DATA_TO_HABITAT_ROTATION @ forward_data
    horizontal_norm = math.hypot(
        float(forward_habitat[0]), float(forward_habitat[2]))
    if horizontal_norm < 1e-8:
        raise PoseConventionError("decoded base forward axis is vertical")
    yaw = math.atan2(
        -float(forward_habitat[0]), -float(forward_habitat[2]))
    return HabitatPlanarPose(
        float(floor_habitat[0]),
        float(floor_habitat[1]),
        float(floor_habitat[2]),
        yaw,
    )


def habitat_pose_to_parquet_data(
    pose: HabitatPlanarPose,
    *,
    camera_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a Habitat floor pose using the corrected generated convention.

    This inverse is primarily an audit/round-trip primitive.  It returns
    ``(action_camera_to_world, camera_extrinsic)``.
    """
    if not isinstance(pose, HabitatPlanarPose):
        raise TypeError("pose must be HabitatPlanarPose")
    cosine, sine = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    habitat_yaw = np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)
    base_data = np.eye(4, dtype=np.float64)
    base_data[:3, :3] = (
        HABITAT_TO_DATA_ROTATION
        @ habitat_yaw
        @ DATA_TO_HABITAT_ROTATION
    )
    base_data[:3, 3] = HABITAT_TO_DATA_ROTATION @ pose.position
    mount = _generated_camera_mount(camera_height_m)
    return base_data @ mount, mount


def local_forward_left_to_world(
    local_forward_left_m: Any,
    current_pose: HabitatPlanarPose,
) -> np.ndarray:
    """Fix a local ``(forward, left)`` goal as a Habitat world point.

    Habitat yaw rotates about +Y and yaw zero faces -Z.  Consequently:

    ``dx = -forward*sin(yaw) - left*cos(yaw)``
    ``dz = -forward*cos(yaw) + left*sin(yaw)``
    """
    if not isinstance(current_pose, HabitatPlanarPose):
        raise TypeError("current_pose must be HabitatPlanarPose")
    forward, left = _vector(
        local_forward_left_m, 2, "local_forward_left_m")
    sine, cosine = math.sin(current_pose.yaw_rad), math.cos(current_pose.yaw_rad)
    return np.asarray([
        current_pose.x_m - forward * sine - left * cosine,
        current_pose.y_m,
        current_pose.z_m - forward * cosine + left * sine,
    ], dtype=np.float64)


def world_to_local_forward_left(
    fixed_world_point_habitat: Any,
    current_pose: HabitatPlanarPose,
) -> np.ndarray:
    """Reproject one fixed Habitat point into the *current* policy frame."""
    if not isinstance(current_pose, HabitatPlanarPose):
        raise TypeError("current_pose must be HabitatPlanarPose")
    world = _vector(
        fixed_world_point_habitat, 3, "fixed_world_point_habitat")
    dx = float(world[0] - current_pose.x_m)
    dz = float(world[2] - current_pose.z_m)
    sine, cosine = math.sin(current_pose.yaw_rad), math.cos(current_pose.yaw_rad)
    return np.asarray([
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    ], dtype=np.float64)


def reproject_fixed_world_point_for_plan(
    fixed_world_point_habitat: Any,
    current_pose: HabitatPlanarPose,
) -> np.ndarray:
    """Planning-boundary alias that makes per-plan reprojection explicit.

    Call this immediately before *every* policy plan.  Never cache and reuse its
    local result after the robot has moved.
    """
    return world_to_local_forward_left(fixed_world_point_habitat, current_pose)


@dataclass(frozen=True)
class FixedWorldPointGoal:
    """Immutable world target obtained from a local point-goal observation."""

    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        for field in ("x_m", "y_m", "z_m"):
            object.__setattr__(
                self, field, _finite_float(getattr(self, field), field))

    @classmethod
    def from_local(
        cls,
        local_forward_left_m: Any,
        current_pose: HabitatPlanarPose,
    ) -> "FixedWorldPointGoal":
        world = local_forward_left_to_world(
            local_forward_left_m, current_pose)
        return cls(*(float(value) for value in world))

    @property
    def world_point(self) -> np.ndarray:
        return np.asarray([self.x_m, self.y_m, self.z_m], dtype=np.float64)

    def reproject_for_plan(self, current_pose: HabitatPlanarPose) -> np.ndarray:
        """Return the target in the current frame for this planning instant."""
        return reproject_fixed_world_point_for_plan(
            self.world_point, current_pose)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    path = Path(path)
    if not path.is_file():
        raise FrozenGeometryError(f"geometry file is missing: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    signature_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise FrozenGeometryError(f"geometry file changed while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def _setting_value(settings: Any, field: str) -> Any:
    if isinstance(settings, Mapping):
        return settings[field]
    if not hasattr(settings, field):
        raise FrozenGeometryError(f"NavMeshSettings is missing {field}")
    return getattr(settings, field)


def canonical_navmesh_settings(settings: Any) -> dict[str, Any]:
    """Canonicalize every Habitat 0.3.x ``NavMeshSettings`` field.

    A mapping must have exactly the known field set.  An actual Habitat object
    is read by attribute, which keeps this module importable in plain Python.
    """
    expected = set(NAVMESH_SETTING_FIELDS)
    if isinstance(settings, Mapping):
        actual = set(settings)
    else:
        actual = set()
        for field in dir(settings):
            if field.startswith("_"):
                continue
            try:
                value = getattr(settings, field)
            except Exception as exc:  # pragma: no cover - defensive pybind guard
                raise FrozenGeometryError(
                    f"cannot inspect NavMeshSettings.{field}") from exc
            if not callable(value):
                actual.add(field)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FrozenGeometryError(
            f"NavMeshSettings fields changed; missing={missing}, extra={extra}")
    result: dict[str, Any] = {}
    for field in NAVMESH_FLOAT_FIELDS:
        try:
            value = _setting_value(settings, field)
        except (KeyError, TypeError) as exc:
            raise FrozenGeometryError(
                f"NavMeshSettings is missing {field}") from exc
        try:
            result[field] = _finite_float(value, f"NavMeshSettings.{field}")
        except ValueError as exc:
            raise FrozenGeometryError(str(exc)) from exc
    for field in NAVMESH_BOOL_FIELDS:
        try:
            value = _setting_value(settings, field)
        except (KeyError, TypeError) as exc:
            raise FrozenGeometryError(
                f"NavMeshSettings is missing {field}") from exc
        if not isinstance(value, (bool, np.bool_)):
            raise FrozenGeometryError(
                f"NavMeshSettings.{field} must be boolean")
        result[field] = bool(value)
    return {field: result[field] for field in NAVMESH_SETTING_FIELDS}


def navmesh_settings_signature(settings: Any) -> str:
    """Content signature of the complete canonical NavMeshSettings record."""
    return _sha256_bytes(_canonical_json_bytes(
        canonical_navmesh_settings(settings)))


def _validate_sha256(value: Any, label: str) -> str:
    result = str(value)
    if (len(result) != 64
            or any(character not in "0123456789abcdef" for character in result)):
        raise FrozenGeometryError(f"{label} is not a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class FrozenGeometryIdentity:
    """Immutable provenance for one GLB and one pre-baked navmesh."""

    glb_sha256: str
    glb_bytes: int
    navmesh_sha256: str
    navmesh_bytes: int
    habitat_sim_version: str
    agent_radius_m: float
    agent_height_m: float
    navmesh_settings_json: str
    navmesh_settings_sha256: str
    schema_version: int = FROZEN_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (isinstance(self.schema_version, bool)
                or not isinstance(self.schema_version, (int, np.integer))
                or int(self.schema_version) != FROZEN_GEOMETRY_SCHEMA_VERSION):
            raise FrozenGeometryError("frozen geometry schema version changed")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(
            self, "glb_sha256", _validate_sha256(self.glb_sha256, "glb_sha256"))
        object.__setattr__(self, "navmesh_sha256", _validate_sha256(
            self.navmesh_sha256, "navmesh_sha256"))
        for field in ("glb_bytes", "navmesh_bytes"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise FrozenGeometryError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, int(value))
        version = str(self.habitat_sim_version)
        if not version or version.strip() != version:
            raise FrozenGeometryError("habitat_sim_version is invalid")
        object.__setattr__(self, "habitat_sim_version", version)
        radius = _finite_float(
            self.agent_radius_m, "agent_radius_m", positive=True)
        height = _finite_float(
            self.agent_height_m, "agent_height_m", positive=True)
        object.__setattr__(self, "agent_radius_m", radius)
        object.__setattr__(self, "agent_height_m", height)

        try:
            decoded = json.loads(self.navmesh_settings_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FrozenGeometryError(
                "navmesh_settings_json is not valid JSON") from exc
        canonical = canonical_navmesh_settings(decoded)
        canonical_text = _canonical_json_bytes(canonical).decode("utf-8")
        if self.navmesh_settings_json != canonical_text:
            raise FrozenGeometryError("navmesh_settings_json is not canonical")
        signature = navmesh_settings_signature(canonical)
        if signature != _validate_sha256(
                self.navmesh_settings_sha256, "navmesh_settings_sha256"):
            raise FrozenGeometryError("NavMeshSettings signature mismatch")
        if not math.isclose(
                canonical["agent_radius"], radius, rel_tol=0.0, abs_tol=1e-6):
            raise FrozenGeometryError(
                "agent_radius_m disagrees with NavMeshSettings.agent_radius")
        if not math.isclose(
                canonical["agent_height"], height, rel_tol=0.0, abs_tol=1e-6):
            raise FrozenGeometryError(
                "agent_height_m disagrees with NavMeshSettings.agent_height")

    @property
    def navmesh_settings(self) -> dict[str, Any]:
        """A defensive copy of the canonical settings."""
        return json.loads(self.navmesh_settings_json)

    @classmethod
    def capture(
        cls,
        *,
        glb_path: Path | str,
        navmesh_path: Path | str,
        habitat_sim_version: str,
        agent_radius_m: float,
        agent_height_m: float,
        navmesh_settings: Any,
    ) -> "FrozenGeometryIdentity":
        canonical = canonical_navmesh_settings(navmesh_settings)
        glb_sha, glb_size = _sha256_file(Path(glb_path))
        navmesh_sha, navmesh_size = _sha256_file(Path(navmesh_path))
        settings_json = _canonical_json_bytes(canonical).decode("utf-8")
        return cls(
            glb_sha256=glb_sha,
            glb_bytes=glb_size,
            navmesh_sha256=navmesh_sha,
            navmesh_bytes=navmesh_size,
            habitat_sim_version=str(habitat_sim_version),
            agent_radius_m=agent_radius_m,
            agent_height_m=agent_height_m,
            navmesh_settings_json=settings_json,
            navmesh_settings_sha256=navmesh_settings_signature(canonical),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "glb": {
                "content_sha256": self.glb_sha256,
                "bytes": self.glb_bytes,
            },
            "navmesh": {
                "content_sha256": self.navmesh_sha256,
                "bytes": self.navmesh_bytes,
            },
            "habitat_sim_version": self.habitat_sim_version,
            "agent_radius_m": self.agent_radius_m,
            "agent_height_m": self.agent_height_m,
            "navmesh_settings": self.navmesh_settings,
            "navmesh_settings_sha256": self.navmesh_settings_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenGeometryIdentity":
        expected = {
            "schema_version", "glb", "navmesh", "habitat_sim_version",
            "agent_radius_m", "agent_height_m", "navmesh_settings",
            "navmesh_settings_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise FrozenGeometryError("frozen geometry record fields changed")
        glb, navmesh = payload["glb"], payload["navmesh"]
        file_fields = {"content_sha256", "bytes"}
        if (not isinstance(glb, Mapping) or set(glb) != file_fields
                or not isinstance(navmesh, Mapping) or set(navmesh) != file_fields):
            raise FrozenGeometryError("frozen geometry file record fields changed")
        canonical = canonical_navmesh_settings(payload["navmesh_settings"])
        return cls(
            schema_version=payload["schema_version"],
            glb_sha256=glb["content_sha256"],
            glb_bytes=glb["bytes"],
            navmesh_sha256=navmesh["content_sha256"],
            navmesh_bytes=navmesh["bytes"],
            habitat_sim_version=payload["habitat_sim_version"],
            agent_radius_m=payload["agent_radius_m"],
            agent_height_m=payload["agent_height_m"],
            navmesh_settings_json=(
                _canonical_json_bytes(canonical).decode("utf-8")),
            navmesh_settings_sha256=payload["navmesh_settings_sha256"],
        )

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(self.canonical_json_bytes())

    def write_json(self, path: Path | str) -> str:
        """Create an immutable canonical identity artifact.

        An existing byte-identical artifact is accepted.  An existing artifact
        with different bytes is never overwritten.
        """
        destination = Path(path)
        payload = self.canonical_json_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FrozenGeometryError(
                    f"refusing to overwrite frozen geometry identity: {destination}")
        if destination.read_bytes() != payload:
            raise FrozenGeometryError(
                f"frozen geometry identity changed while writing: {destination}")
        return self.identity_sha256

    @classmethod
    def load_json(cls, path: Path | str) -> "FrozenGeometryIdentity":
        """Load an identity artifact and require canonical byte encoding."""
        source = Path(path)
        try:
            raw = source.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrozenGeometryError(
                f"cannot load frozen geometry identity: {source}") from exc
        identity = cls.from_dict(payload)
        if raw != identity.canonical_json_bytes():
            raise FrozenGeometryError(
                f"frozen geometry identity is not canonical: {source}")
        return identity

    def validate_runtime(
        self,
        *,
        glb_path: Path | str,
        navmesh_path: Path | str,
        habitat_sim_version: str,
        agent_radius_m: float,
        agent_height_m: float,
        navmesh_settings: Any,
    ) -> None:
        """Re-hash files and reject any runtime/provenance mismatch."""
        if str(habitat_sim_version) != self.habitat_sim_version:
            raise FrozenGeometryError("Habitat-Sim version changed")
        if _finite_float(agent_radius_m, "agent_radius_m", positive=True) != (
                self.agent_radius_m):
            raise FrozenGeometryError("agent radius changed")
        if _finite_float(agent_height_m, "agent_height_m", positive=True) != (
                self.agent_height_m):
            raise FrozenGeometryError("agent height changed")
        settings = canonical_navmesh_settings(navmesh_settings)
        if navmesh_settings_signature(settings) != self.navmesh_settings_sha256:
            raise FrozenGeometryError("NavMeshSettings changed")
        glb_sha, glb_size = _sha256_file(Path(glb_path))
        if glb_sha != self.glb_sha256 or glb_size != self.glb_bytes:
            raise FrozenGeometryError("GLB content changed")
        navmesh_sha, navmesh_size = _sha256_file(Path(navmesh_path))
        if (navmesh_sha != self.navmesh_sha256
                or navmesh_size != self.navmesh_bytes):
            raise FrozenGeometryError("navmesh content changed")


def load_pinned_navmesh_for_collector(
    simulator: Any,
    *,
    identity: FrozenGeometryIdentity,
    glb_path: Path | str,
    navmesh_path: Path | str,
    habitat_sim_version: str,
    agent_radius_m: float,
    agent_height_m: float,
    navmesh_settings: Any,
) -> Any:
    """Validate and load a pre-baked navmesh without recomputing geometry.

    This is the only supported collector entry point.  It intentionally calls
    ``pathfinder.load_nav_mesh`` and contains no call or fallback to
    ``Simulator.recompute_navmesh``.  A failed load is fatal.
    """
    if not isinstance(identity, FrozenGeometryIdentity):
        raise TypeError("identity must be FrozenGeometryIdentity")
    identity.validate_runtime(
        glb_path=glb_path,
        navmesh_path=navmesh_path,
        habitat_sim_version=habitat_sim_version,
        agent_radius_m=agent_radius_m,
        agent_height_m=agent_height_m,
        navmesh_settings=navmesh_settings,
    )
    pathfinder = getattr(simulator, "pathfinder", None)
    loader = getattr(pathfinder, "load_nav_mesh", None)
    if pathfinder is None or not callable(loader):
        raise FrozenGeometryError(
            "simulator.pathfinder.load_nav_mesh is unavailable")
    loaded = loader(str(Path(navmesh_path)))
    if loaded is False:
        raise FrozenGeometryError("pinned navmesh failed to load")
    # Catch replacement or mutation racing the load itself.
    identity.validate_runtime(
        glb_path=glb_path,
        navmesh_path=navmesh_path,
        habitat_sim_version=habitat_sim_version,
        agent_radius_m=agent_radius_m,
        agent_height_m=agent_height_m,
        navmesh_settings=navmesh_settings,
    )
    return pathfinder


__all__ = [
    "DATA_TO_HABITAT_ROTATION",
    "DATA_ZUP_FRAME_CONVENTION_PREFIX",
    "FROZEN_GEOMETRY_SCHEMA_VERSION",
    "FixedWorldPointGoal",
    "FrozenGeometryError",
    "FrozenGeometryIdentity",
    "HABITAT_TO_DATA_ROTATION",
    "HabitatPlanarPose",
    "NAVMESH_SETTING_FIELDS",
    "PoseConventionError",
    "canonical_navmesh_settings",
    "habitat_pose_to_parquet_data",
    "load_pinned_navmesh_for_collector",
    "local_forward_left_to_world",
    "navmesh_settings_signature",
    "parquet_data_pose_to_habitat",
    "relative_yaw",
    "reproject_fixed_world_point_for_plan",
    "resolve_generated_camera_extrinsic",
    "world_to_local_forward_left",
    "wrap_yaw",
]
