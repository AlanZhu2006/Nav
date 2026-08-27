#!/usr/bin/env python3
"""Fail-closed audit for the historical YmJ NLSR raw-data gap fill.

The historical source files no longer exist, so this audit deliberately does
not claim byte-identical recovery.  It proves that the pinned generator
reproduced the two surviving historical summaries and that every generated
episode is internally complete, readable, finite, and content-addressed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image


SCHEMA_VERSION = "nlsr_historical_ymj_gapfill_audit_v1"
SCENE = "YmJkqBEsHnH"
EXPECTED_COUNTS = {"mp3d_2leg": 15, "mp3d_3leg": 2}
EXPECTED_THREE_LEG = {
    "episode_0000": {"n_frames": 871, "switches": [218, 548]},
    "episode_0001": {"n_frames": 591, "switches": [192, 296]},
}
EXPECTED_IMAGE_SIZE = (480, 270)
PARQUET_COLUMNS = (
    "index",
    "observation.camera_intrinsic",
    "observation.camera_extrinsic",
    "action",
)


class AuditError(RuntimeError):
    """A generated artifact violated the frozen gap-fill contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def write_new_artifact(path: Path, value: object) -> str:
    """Atomically publish JSON plus SHA sidecar without replacing either."""
    payload = canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise AuditError(f"audit output already exists: {path} / {sidecar}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination, data in (
            (path, payload),
            (sidecar, f"{digest}  {path.name}\n".encode("ascii")),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # link(2), unlike replace(2), fails if a concurrent writer created
            # the destination after the preflight.
            os.link(temporary, destination)
            temporary.unlink()
            temporary_paths.remove(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return digest


def _exact_children(root: Path, expected: Iterable[str], label: str) -> None:
    actual = {path.name for path in root.iterdir()}
    wanted = set(expected)
    if actual != wanted:
        raise AuditError(
            f"{label} children differ: missing={sorted(wanted - actual)} "
            f"extra={sorted(actual - wanted)}"
        )


def _verify_image(path: Path, *, depth: bool = False) -> dict:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"image must be a physical regular file: {path}")
    try:
        with Image.open(path) as image:
            size = tuple(image.size)
            mode = image.mode
            image.verify()
    except Exception as exc:
        raise AuditError(f"unreadable image: {path}: {exc}") from exc
    if size != EXPECTED_IMAGE_SIZE:
        raise AuditError(f"image size mismatch at {path}: {size}")
    if depth:
        if mode not in {"I", "I;16", "I;16B", "I;16L"}:
            raise AuditError(f"depth image is not integer-valued at {path}: {mode}")
    elif mode != "RGB":
        raise AuditError(f"RGB image mode mismatch at {path}: {mode}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise AuditError(f"{label} must be a finite {shape} matrix")
    return matrix


def _audit_parquet(path: Path, n_frames: int) -> dict:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"parquet must be a physical regular file: {path}")
    try:
        table = parquet.read_table(path, columns=list(PARQUET_COLUMNS))
    except Exception as exc:
        raise AuditError(f"cannot read parquet {path}: {exc}") from exc
    if tuple(table.column_names) != PARQUET_COLUMNS:
        raise AuditError(f"parquet schema changed at {path}: {table.column_names}")
    if table.num_rows != n_frames:
        raise AuditError(
            f"parquet row count mismatch at {path}: {table.num_rows} != {n_frames}"
        )
    identity = np.eye(4, dtype=np.float64)
    for row_number, row in enumerate(table.to_pylist()):
        index = row.get("index")
        if isinstance(index, bool) or int(index) != row_number:
            raise AuditError(f"non-contiguous parquet index at row {row_number}")
        intrinsic = _matrix(
            row.get("observation.camera_intrinsic"), (3, 3),
            f"parquet[{row_number}].camera_intrinsic",
        )
        if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise AuditError(f"invalid intrinsic bottom row at {path}:{row_number}")
        extrinsic = _matrix(
            row.get("observation.camera_extrinsic"), (4, 4),
            f"parquet[{row_number}].camera_extrinsic",
        )
        # This is an explicit historical-distribution check, not an endorsement:
        # d6c8b56 predates the mount-extrinsic axis fix and wrote identity here.
        if not np.allclose(extrinsic, identity, atol=1e-6):
            raise AuditError(
                f"historical generator extrinsic changed at {path}:{row_number}"
            )
        action = _matrix(row.get("action"), (4, 4), f"parquet[{row_number}].action")
        if not np.allclose(action[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise AuditError(f"invalid action bottom row at {path}:{row_number}")
        rotation = action[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0,
                                    abs_tol=2e-4)):
            raise AuditError(f"action rotation is not SO(3) at {path}:{row_number}")
    return {
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _audit_episode(episode: Path, n_legs: int) -> dict:
    if episode.is_symlink() or not episode.is_dir():
        raise AuditError(f"episode must be a physical directory: {episode}")
    expected_top = {"data", "meta", "videos", "goal_image.jpg"}
    expected_top.update(f"goal_{index}.jpg" for index in range(1, n_legs))
    _exact_children(episode, expected_top, str(episode))

    metadata_path = episode / "meta/gen_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"invalid metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise AuditError(f"metadata is not an object: {metadata_path}")
    if Path(str(metadata.get("scene", ""))).stem != SCENE:
        raise AuditError(f"scene mismatch in {metadata_path}")
    if metadata.get("frame_convention") != (
            "positions+parquet in data(Zup,M_W); "
            "yaw_habitat in render frame"):
        raise AuditError(
            f"historical frame-convention marker changed in {metadata_path}")
    if metadata.get("ep_idx") != int(episode.name.rsplit("_", 1)[1]):
        raise AuditError(f"episode index mismatch in {metadata_path}")
    if metadata.get("n_legs") != n_legs:
        raise AuditError(f"n_legs mismatch in {metadata_path}")
    n_frames = metadata.get("n_frames")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise AuditError(f"invalid n_frames in {metadata_path}")
    switches = metadata.get("switches")
    if (not isinstance(switches, list) or len(switches) != n_legs - 1
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in switches)
            or sorted(switches) != switches
            or any(not 0 < value < n_frames for value in switches)):
        raise AuditError(f"invalid switches in {metadata_path}")
    goals = metadata.get("goals")
    expected_kinds = ["revisit"] if n_legs == 2 else ["novel", "revisit"]
    if (not isinstance(goals, list) or len(goals) != n_legs - 1
            or [goal.get("kind") if isinstance(goal, dict) else None
                for goal in goals] != expected_kinds):
        raise AuditError(f"goal semantics mismatch in {metadata_path}")
    for goal_index, goal in enumerate(goals):
        curve = goal.get("covis_curve")
        expected_curve_length = switches[goal_index]
        if (not isinstance(curve, list) or len(curve) != expected_curve_length
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, float))
                       or not math.isfinite(float(value))
                       for value in curve)):
            raise AuditError(
                f"goal covisibility curve mismatch in {metadata_path}: "
                f"goal={goal_index} expected={expected_curve_length}"
            )

    if n_legs == 3:
        expected = EXPECTED_THREE_LEG.get(episode.name)
        if expected is None:
            raise AuditError(f"unexpected historical three-leg episode: {episode}")
        actual = {"n_frames": n_frames, "switches": switches}
        if actual != expected:
            raise AuditError(
                f"historical summary mismatch for {episode.name}: "
                f"{actual} != {expected}"
            )

    rgb_root = episode / "videos/chunk-000/observation.images.rgb"
    depth_root = episode / "videos/chunk-000/observation.images.depth"
    data_root = episode / "data/chunk-000"
    _exact_children(episode / "videos", {"chunk-000"}, f"{episode}/videos")
    _exact_children(
        episode / "videos/chunk-000",
        {"observation.images.rgb", "observation.images.depth"},
        f"{episode}/videos/chunk-000",
    )
    _exact_children(episode / "data", {"chunk-000"}, f"{episode}/data")
    _exact_children(data_root, {"episode_000000.parquet"}, str(data_root))
    _exact_children(episode / "meta", {"gen_meta.json"}, f"{episode}/meta")
    _exact_children(rgb_root, {f"{i}.jpg" for i in range(n_frames)}, str(rgb_root))
    _exact_children(depth_root, {f"{i}.png" for i in range(n_frames)}, str(depth_root))

    rgb = []
    depth = []
    for index in range(n_frames):
        rgb.append(_verify_image(rgb_root / f"{index}.jpg"))
        depth.append(_verify_image(depth_root / f"{index}.png", depth=True))
    goals_record = {}
    for name in sorted(expected_top - {"data", "meta", "videos"}):
        goals_record[name] = _verify_image(episode / name)

    metadata_record = {
        "bytes": metadata_path.stat().st_size,
        "sha256": sha256_file(metadata_path),
    }
    parquet_record = _audit_parquet(
        data_root / "episode_000000.parquet", n_frames,
    )
    return {
        "episode": episode.name,
        "n_frames": n_frames,
        "switches": switches,
        "metadata": metadata_record,
        "parquet": parquet_record,
        "goals": goals_record,
        "rgb": rgb,
        "depth": depth,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--generator-sha256", required=True)
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_root.is_symlink() or not args.raw_root.is_dir():
        raise AuditError(f"raw root must be a physical directory: {args.raw_root}")
    _exact_children(args.raw_root, EXPECTED_COUNTS, "raw root")

    generator = args.generator_root / "MemNavData/generate_twoleg.py"
    actual_generator_sha = sha256_file(generator)
    if actual_generator_sha != args.generator_sha256:
        raise AuditError(
            f"generator SHA mismatch: {actual_generator_sha} != "
            f"{args.generator_sha256}"
        )

    legs_record = {}
    for group, count in EXPECTED_COUNTS.items():
        group_root = args.raw_root / group
        _exact_children(group_root, {SCENE}, group)
        scene_root = group_root / SCENE
        expected_episodes = {f"episode_{index:04d}" for index in range(count)}
        _exact_children(scene_root, expected_episodes, f"{group}/{SCENE}")
        n_legs = 2 if group == "mp3d_2leg" else 3
        legs_record[group] = [
            _audit_episode(scene_root / episode, n_legs)
            for episode in sorted(expected_episodes)
        ]

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "audited_historical_summary_match",
        "scene": SCENE,
        "byte_identical_historical_recovery_claim": False,
        "historical_recovery_limit": (
            "the original raw files and content hashes were pruned; only the "
            "logged frame counts and switch indices can be reproduced"
        ),
        "known_historical_axis_semantics": (
            "generator d6c8b56 writes identity camera_extrinsic; this audit "
            "pins that old-data distribution and does not claim the later mount-axis fix"
        ),
        "generator": {
            "root": str(args.generator_root.resolve()),
            "commit": args.generator_commit,
            "script": "MemNavData/generate_twoleg.py",
            "script_sha256": actual_generator_sha,
            "arguments": {
                "n2": 15,
                "n3": 2,
                "seed": 50,
                "goal_jitter_pos": 3.0,
                "goal_tries": 100,
                "window": 32,
                "num_scale": 8,
            },
        },
        "assets": {
            "scene_glb": {
                "path": str(args.scene_glb.resolve()),
                "bytes": args.scene_glb.stat().st_size,
                "sha256": sha256_file(args.scene_glb),
            },
            "navmesh": {
                "path": str(args.navmesh.resolve()),
                "bytes": args.navmesh.stat().st_size,
                "sha256": sha256_file(args.navmesh),
            },
        },
        "raw_root": str(args.raw_root.resolve()),
        "historical_three_leg_reference": EXPECTED_THREE_LEG,
        "groups": legs_record,
        "summary": {
            "two_leg_episodes": len(legs_record["mp3d_2leg"]),
            "three_leg_episodes": len(legs_record["mp3d_3leg"]),
            "files_content_addressed": sum(
                2 * episode["n_frames"] + len(episode["goals"]) + 2
                for episodes in legs_record.values()
                for episode in episodes
            ),
        },
    }
    digest = write_new_artifact(args.out, record)
    print(json.dumps({
        "status": record["status"],
        "output": str(args.out),
        "sha256": digest,
        **record["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
