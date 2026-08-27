#!/usr/bin/env python3
"""Validate three adaptively admitted flow4096 patches and build their routes.

The 97 already-audited official pairs remain under the official account and
the three newly generated pairs remain under this run.  A small canonical
artifact maps every episode to one explicit physical source root.  Downstream
consumers pin that artifact and resolve the files through ``flow_cache_routing``;
no symlink farm, hard-link permission exception, bind recipe, or payload copy is
required.  Any schema, provenance, ground-scale, frame, keyframe-budget, or
strict-split mismatch removes the partial route root and exits nonzero.  Patch
thresholds are not fixed guesses: a content-pinned admission journal proves
that each episode used the lowest approved threshold that fit the pinned
temporal-view budget.  Admission depends only on cache structure/provenance,
never an evaluation label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

import numpy as np

try:
    from MemNavData.flow_cache_routing import (
        ROUTE_SCHEMA_VERSION,
        ROUTE_STATUS,
        FlowRoutingError,
        load_route_registry,
    )
except ImportError:
    from flow_cache_routing import (  # type: ignore
        ROUTE_SCHEMA_VERSION,
        ROUTE_STATUS,
        FlowRoutingError,
        load_route_registry,
    )


SCHEMA_VERSION = ROUTE_SCHEMA_VERSION
FLOW_FILES = ("lingbot_cache.npz", "lingbot_cam_cache.npz")
FLOW_THRESHOLD_ADMISSION_SCHEMA = "nlsr_flow_threshold_admission_v1"
FLOW_THRESHOLD_TIERS = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
PATCH_EPISODES = {
    "B6ByNegPMKs/episode_0001": {
        "n_frames": 2456,
        "minimum_threshold": 60.0,
    },
    "YmJkqBEsHnH/episode_0000": {
        "n_frames": 871,
        "minimum_threshold": 25.0,
    },
    "YmJkqBEsHnH/episode_0001": {
        "n_frames": 591,
        "minimum_threshold": 20.0,
    },
}
EXPECTED_SPLIT_SHA = "97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
EXPECTED_PRECOMPUTE_SHA = (
    "ddf1e425318ea59fed51bb78d5d0e52d54860cab0c60589fcae1b264a87fcbb2"
)
EXPECTED_CACHE_SCHEMA_SHA = (
    "412fdc9ba11947406297ba639790b9d3acf7d9c3ab6125dea8e1a1246e38f227"
)
EXPECTED_FLOW_CODE_COMMIT = "b3dfae9d5647165927a5ade0d8193fe1d0d3c19d"
EXPECTED_OFFICIAL_INTERNNAV_REVISION = "3878c0650b2d70e8bf488ae8bcbc8997b44252f4"
EXPECTED_LINGBOT_COMMIT = "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
EXPECTED_WEIGHT_SHA = "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
EXPECTED_COUNTS = {
    "scenes": 50,
    "pairs": 100,
    "official_base": 97,
    "flow4096_patch": 3,
}


class FlowAuditError(RuntimeError):
    """A flow cache or merge dependency violated the strict contract."""


def _schema_keyframe_budget(cache_schema: ModuleType) -> int:
    budget = getattr(cache_schema, "DEFAULT_KEYFRAME_BUDGET", None)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise FlowAuditError("cache schema has no valid keyframe budget")
    return budget


def _patch_budget_compliant(
    anchor_count: int,
    num_scale_frames: int,
    budget: int,
) -> bool:
    return anchor_count + num_scale_frames <= budget


def _threshold_index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlowAuditError(f"{label} is not a numeric threshold tier")
    numeric = float(value)
    matches = [
        index
        for index, threshold in enumerate(FLOW_THRESHOLD_TIERS)
        if math.isclose(numeric, threshold, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise FlowAuditError(f"{label} is not an approved threshold tier: {value!r}")
    return matches[0]


def _admission_filename(episode: str) -> str:
    if (
        not isinstance(episode, str)
        or episode.count("/") != 1
        or any(part in ("", ".", "..") for part in episode.split("/"))
    ):
        raise FlowAuditError(f"invalid admission episode key: {episode!r}")
    return episode.replace("/", "__") + ".json"


def _cache_file_hashes(validation: dict) -> dict[str, str]:
    result = {}
    for record in validation.get("files", []):
        if not isinstance(record, dict):
            raise FlowAuditError("cache validation file record is invalid")
        name = record.get("name")
        digest = record.get("content_sha256")
        if (
            name not in FLOW_FILES
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FlowAuditError("cache validation lacks a patch payload SHA256")
        if name in result:
            raise FlowAuditError(f"duplicate cache validation file: {name}")
        result[name] = digest
    if set(result) != set(FLOW_FILES):
        raise FlowAuditError("cache validation patch file set changed")
    return result


def validate_threshold_admission_record(
    record: object,
    *,
    episode: str,
    minimum_threshold: float,
    keyframe_budget: int,
    final_validation: dict | None = None,
) -> dict:
    """Validate a monotone, adjacent-tier, label-free admission trajectory."""

    expected_fields = {
        "schema_version",
        "episode",
        "minimum_threshold",
        "threshold_tiers",
        "keyframe_budget",
        "num_scale_frames",
        "selection_basis",
        "status",
        "selected_threshold",
        "attempts",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise FlowAuditError("threshold admission record fields changed")
    if record["schema_version"] != FLOW_THRESHOLD_ADMISSION_SCHEMA:
        raise FlowAuditError("threshold admission schema changed")
    if record["episode"] != episode:
        raise FlowAuditError("threshold admission episode changed")
    minimum_index = _threshold_index(minimum_threshold, "minimum threshold")
    if (
        _threshold_index(record["minimum_threshold"], "record minimum threshold")
        != minimum_index
    ):
        raise FlowAuditError("threshold admission minimum changed")
    if record["threshold_tiers"] != list(FLOW_THRESHOLD_TIERS):
        raise FlowAuditError("threshold admission tiers changed")
    if (
        isinstance(keyframe_budget, bool)
        or not isinstance(keyframe_budget, int)
        or keyframe_budget < 1
        or record["keyframe_budget"] != keyframe_budget
    ):
        raise FlowAuditError("threshold admission keyframe budget changed")
    if record["num_scale_frames"] != 8:
        raise FlowAuditError("threshold admission scale-frame count changed")
    if record["selection_basis"] != (
        "lowest approved flow threshold whose anchor_count + "
        "num_scale_frames fits the pinned cache-schema keyframe budget; "
        "no evaluation label"
    ):
        raise FlowAuditError("threshold admission selection basis changed")
    if record["status"] not in {"pending", "accepted", "exhausted"}:
        raise FlowAuditError("threshold admission status is invalid")
    attempts = record["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise FlowAuditError("threshold admission attempts are empty")
    expected_attempt_fields = {
        "threshold",
        "action",
        "outcome",
        "anchor_count",
        "total_memory_frames",
        "precompute_signature",
        "cache_files_sha256",
    }
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) != expected_attempt_fields:
            raise FlowAuditError(f"threshold admission attempt {index} fields changed")
        tier_index = _threshold_index(
            attempt["threshold"], f"attempt {index} threshold"
        )
        if tier_index != minimum_index + index:
            raise FlowAuditError(
                "threshold admission did not advance one approved tier at a time"
            )
        if attempt["action"] not in {
            "compute",
            "overwrite",
            "inspect_existing",
            "resume",
        }:
            raise FlowAuditError(f"threshold admission attempt {index} action changed")
        if index > 0 and attempt["action"] != "overwrite":
            raise FlowAuditError(
                "every threshold increase must use an explicit overwrite action"
            )
        outcome = attempt["outcome"]
        if outcome == "pending":
            if index != len(attempts) - 1:
                raise FlowAuditError("only the final admission attempt may be pending")
            if any(
                attempt[field] is not None
                for field in (
                    "anchor_count",
                    "total_memory_frames",
                    "precompute_signature",
                    "cache_files_sha256",
                )
            ):
                raise FlowAuditError("pending admission attempt contains observations")
            continue
        if outcome not in {"over_budget", "accepted"}:
            raise FlowAuditError(f"threshold admission attempt {index} outcome changed")
        anchor_count = attempt["anchor_count"]
        total = attempt["total_memory_frames"]
        if (
            isinstance(anchor_count, bool)
            or not isinstance(anchor_count, int)
            or anchor_count < 0
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total != anchor_count + 8
        ):
            raise FlowAuditError("threshold admission count observation is invalid")
        signature = attempt["precompute_signature"]
        if (
            not isinstance(signature, str)
            or len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
        ):
            raise FlowAuditError("threshold admission signature is invalid")
        hashes = attempt["cache_files_sha256"]
        if (
            not isinstance(hashes, dict)
            or set(hashes) != set(FLOW_FILES)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            )
        ):
            raise FlowAuditError("threshold admission cache hashes are invalid")
        if outcome == "over_budget":
            if (
                total <= keyframe_budget
                or index == len(attempts) - 1
                and (record["status"] == "accepted")
            ):
                raise FlowAuditError(
                    "over-budget threshold admission result is invalid"
                )
        elif total > keyframe_budget or index != len(attempts) - 1:
            raise FlowAuditError("accepted threshold admission result is invalid")

    selected = record["selected_threshold"]
    last = attempts[-1]
    if record["status"] == "accepted":
        selected_index = _threshold_index(selected, "selected threshold")
        if (
            last["outcome"] != "accepted"
            or selected_index != minimum_index + len(attempts) - 1
        ):
            raise FlowAuditError("accepted threshold admission state is inconsistent")
    elif selected is not None:
        raise FlowAuditError("non-accepted threshold admission selected a threshold")
    if record["status"] == "pending" and last["outcome"] != "pending":
        raise FlowAuditError("pending threshold admission state is inconsistent")
    if record["status"] == "exhausted":
        if (
            last["outcome"] != "over_budget"
            or _threshold_index(last["threshold"], "exhausted threshold")
            != len(FLOW_THRESHOLD_TIERS) - 1
        ):
            raise FlowAuditError("exhausted threshold admission state is inconsistent")

    if final_validation is not None:
        if record["status"] != "accepted":
            raise FlowAuditError("final patch lacks accepted threshold admission")
        actual_index = _threshold_index(
            final_validation.get("flow_threshold"), "actual patch threshold"
        )
        if actual_index != _threshold_index(selected, "selected threshold"):
            raise FlowAuditError("selected and actual patch thresholds differ")
        if final_validation.get("total_memory_frames") != last["total_memory_frames"]:
            raise FlowAuditError("admission and final patch memory counts differ")
        if final_validation.get("anchor_count") != last["anchor_count"]:
            raise FlowAuditError("admission and final patch anchor counts differ")
        if final_validation.get("precompute_signature") != last["precompute_signature"]:
            raise FlowAuditError("admission and final patch signatures differ")
        if _cache_file_hashes(final_validation) != last["cache_files_sha256"]:
            raise FlowAuditError("admission and final patch payload hashes differ")
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_cache_schema(path: Path) -> ModuleType:
    if sha256_file(path) != EXPECTED_CACHE_SCHEMA_SHA:
        raise FlowAuditError(f"cache-schema SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location("_nlsr_cache_schema", path)
    if spec is None or spec.loader is None:
        raise FlowAuditError(f"cannot import cache schema: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_episode_metadata(path: Path, scene: str) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    if Path(str(record.get("scene", ""))).stem != scene:
        return None
    if record.get("n_legs") != 3:
        return None
    frames = record.get("n_frames")
    switches = record.get("switches")
    goals = record.get("goals")
    if (
        isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames < 1
        or not isinstance(switches, list)
        or len(switches) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in switches
        )
        or not 0 < switches[0] < switches[1] < frames
        or not isinstance(goals, list)
        or len(goals) != 2
        or not isinstance(goals[0], dict)
        or not isinstance(goals[1], dict)
        or goals[0].get("kind") != "novel"
        or goals[1].get("kind") != "revisit"
    ):
        return None
    return record


def _select_pairs(split_path: Path, episode_root: Path) -> dict[str, int]:
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA:
        raise FlowAuditError(f"split SHA mismatch: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    pairs: dict[str, int] = {}
    scenes = []
    for role in ("train", "development"):
        role_scenes = split.get(role)
        if not isinstance(role_scenes, list):
            raise FlowAuditError(f"split role is not a list: {role}")
        for scene in role_scenes:
            if not isinstance(scene, str) or not scene:
                raise FlowAuditError(f"invalid scene in split role {role}")
            scenes.append(scene)
            scene_root = episode_root / scene
            if not scene_root.is_dir():
                raise FlowAuditError(f"raw scene is absent: {scene_root}")
            valid = []
            for episode in sorted(scene_root.glob("episode_*")):
                if not episode.is_dir():
                    continue
                metadata = _valid_episode_metadata(
                    episode / "meta/gen_meta.json",
                    scene,
                )
                if metadata is not None:
                    valid.append((episode.name, int(metadata["n_frames"])))
            if len(valid) < 2:
                raise FlowAuditError(
                    f"scene has fewer than two valid three-leg episodes: "
                    f"{scene}: {valid}"
                )
            for episode, frames in valid[:2]:
                key = f"{scene}/{episode}"
                if key in pairs:
                    raise FlowAuditError(f"duplicate selected pair: {key}")
                pairs[key] = frames
    if len(scenes) != EXPECTED_COUNTS["scenes"] or len(set(scenes)) != len(scenes):
        raise FlowAuditError(
            f"strict split scene count/uniqueness mismatch: {len(scenes)}"
        )
    if len(pairs) != EXPECTED_COUNTS["pairs"]:
        raise FlowAuditError(f"selected pair count mismatch: {len(pairs)}")
    for key, expected in PATCH_EPISODES.items():
        if pairs.get(key) != expected["n_frames"]:
            raise FlowAuditError(
                f"patch episode selection/frame mismatch: {key}: "
                f"{pairs.get(key)} != {expected['n_frames']}"
            )
    return pairs


def _scalar(npz: np.lib.npyio.NpzFile, name: str):
    if name not in npz.files:
        raise FlowAuditError(f"cache lacks metadata field {name!r}")
    value = np.asarray(npz[name])
    if value.size != 1:
        raise FlowAuditError(f"cache metadata field {name!r} is not scalar")
    return value.reshape(-1)[0].item()


def _read_provenance(path: Path) -> tuple[str, str, dict, float | None]:
    with np.load(path, allow_pickle=False) as cache:
        signature = str(_scalar(cache, "precompute_signature"))
        config_json = str(_scalar(cache, "precompute_config_json"))
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise FlowAuditError(f"invalid precompute config in {path}") from exc
        ground = None
        if "ground_h_est" in cache.files:
            ground = float(_scalar(cache, "ground_h_est"))
    return signature, config_json, config, ground


def _validate_config(
    config: dict,
    *,
    threshold: float,
    patch: bool,
) -> None:
    exact = {
        "cache_schema_sha256": EXPECTED_CACHE_SCHEMA_SHA,
        "cache_schema_version": 2,
        "camera_num_iterations": 4,
        "dtype": "bf16",
        "enable_3d_rope": True,
        "flow_threshold": threshold,
        "ground_stride": 1,
        "image_size": 518,
        "internnav_revision": (
            EXPECTED_FLOW_CODE_COMMIT if patch else EXPECTED_OFFICIAL_INTERNNAV_REVISION
        ),
        "keyframe_interval_mode": f"flow_thr{threshold:g}px_gap30",
        "keyframe_policy": "flow_gate_v1",
        "kv_cache_sliding_window": 32,
        "lingbot_revision": EXPECTED_LINGBOT_COMMIT,
        "max_frame_num": 4096 if patch else 2048,
        "max_non_keyframe_gap": 30,
        "num_scale_frames": 8,
        "patch_size": 14,
        "precompute_script_sha256": EXPECTED_PRECOMPUTE_SHA,
        "preprocess_mode": "pad",
        "skip_ground_scale": False,
        "use_sdpa": True,
        "weights_sha256": EXPECTED_WEIGHT_SHA,
    }
    if config != exact:
        differences = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in exact.items()
            if config.get(key) != value
        }
        extras = sorted(set(config) - set(exact))
        missing = sorted(set(exact) - set(config))
        raise FlowAuditError(
            f"precompute config mismatch: differences={differences} "
            f"extras={extras} missing={missing}"
        )


def _validate_pair(
    cache_schema: ModuleType,
    aggregator: Path,
    camera: Path,
    *,
    frames: int,
    threshold: float | None,
    patch: bool,
    require_budget: bool = True,
) -> dict:
    for path in (aggregator, camera):
        if path.is_symlink() or not path.is_file():
            raise FlowAuditError(f"source cache must be a physical file: {path}")
    try:
        layout = cache_schema.validate_cache_files(
            aggregator,
            camera,
            expected_num_frames=frames,
            expected_num_scale_frames=8,
            expected_sliding_window=32,
            require_versioned=True,
        )
    except Exception as exc:
        raise FlowAuditError(
            f"invalid cache pair at {aggregator.parent}: {exc}"
        ) from exc
    if layout.keyframe_policy != "flow_gate_v1" or layout.keyframe_interval != 0:
        raise FlowAuditError(f"cache is not flow_gate_v1: {aggregator.parent}")
    if layout.max_non_keyframe_gap != 30:
        raise FlowAuditError(f"cache max gap differs: {aggregator.parent}")
    if threshold is not None and not math.isclose(
        layout.flow_threshold, threshold, abs_tol=1e-9
    ):
        raise FlowAuditError(
            f"cache threshold mismatch: {layout.flow_threshold} != {threshold}"
        )
    _threshold_index(layout.flow_threshold, "cache flow threshold")
    anchor_count = len(layout.anchor_frame_indices)
    # New caches must obey the temporal-view budget declared by the exact,
    # content-pinned cache schema used to validate them.  Binding admission to
    # that contract keeps the offline cache compatible with the live policy;
    # an unrelated local 270-anchor limit would reject the official >2048-frame
    # threshold tier even when scale + anchors remains below the real budget.
    keyframe_budget = _schema_keyframe_budget(cache_schema)
    if (
        patch
        and require_budget
        and not _patch_budget_compliant(
            anchor_count, layout.num_scale_frames, keyframe_budget
        )
    ):
        raise FlowAuditError(
            f"new patch exceeds keyframe budget at {aggregator.parent}: "
            f"anchors={anchor_count} scale={layout.num_scale_frames} "
            f"budget={keyframe_budget}"
        )

    agg_signature, agg_json, agg_config, agg_ground = _read_provenance(aggregator)
    cam_signature, cam_json, cam_config, cam_ground = _read_provenance(camera)
    if agg_ground is not None:
        raise FlowAuditError(
            f"aggregator unexpectedly stores ground scale: {aggregator}"
        )
    if (
        agg_signature != cam_signature
        or agg_json != cam_json
        or agg_config != cam_config
    ):
        raise FlowAuditError(f"cache-pair provenance differs: {aggregator.parent}")
    expected_signature = hashlib.sha256(agg_json.encode("utf-8")).hexdigest()
    if agg_signature != expected_signature:
        raise FlowAuditError(
            f"cache provenance signature is invalid: {aggregator.parent}"
        )
    _validate_config(
        agg_config,
        threshold=float(layout.flow_threshold),
        patch=patch,
    )
    if cam_ground is None or not math.isfinite(cam_ground) or cam_ground <= 0.0:
        raise FlowAuditError(f"invalid ground_h_est in {camera}: {cam_ground}")
    with np.load(camera, allow_pickle=False) as camera_cache:
        pose = np.asarray(camera_cache["cam_pose_enc"])
        if pose.shape != (frames, 9) or not np.isfinite(pose).all():
            raise FlowAuditError(f"invalid dense camera pose payload: {camera}")

    files = []
    for path in (aggregator, camera):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "content_sha256": sha256_file(path) if patch else None,
            }
        )
    return {
        "num_frames": frames,
        "num_scale_frames": layout.num_scale_frames,
        "anchor_count": anchor_count,
        "total_memory_frames": anchor_count + layout.num_scale_frames,
        "strict_patch_keyframe_budget_compliant": (
            _patch_budget_compliant(
                anchor_count, layout.num_scale_frames, keyframe_budget
            )
        ),
        "flow_threshold": float(layout.flow_threshold),
        "max_non_keyframe_gap": layout.max_non_keyframe_gap,
        "precompute_signature": layout.precompute_signature,
        "precompute_config": agg_config,
        "ground_h_est": cam_ground,
        "files": files,
    }


def _write_provenance(root: Path, record: dict) -> str:
    payload = canonical_bytes(record)
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "FLOW_ROUTE_PROVENANCE.json"
    sidecar = root / "FLOW_ROUTE_PROVENANCE.json.sha256"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _load_admission_record(
    root: Path,
    episode: str,
    minimum_threshold: float,
    keyframe_budget: int,
    final_validation: dict,
) -> dict:
    path = root / _admission_filename(episode)
    if path.is_symlink() or not path.is_file():
        raise FlowAuditError(f"threshold admission log is not physical: {path}")
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlowAuditError(f"cannot load threshold admission log: {path}") from error
    if raw != canonical_bytes(record):
        raise FlowAuditError(f"threshold admission log is not canonical: {path}")
    return validate_threshold_admission_record(
        record,
        episode=episode,
        minimum_threshold=minimum_threshold,
        keyframe_budget=keyframe_budget,
        final_validation=final_validation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--official-flow-root", type=Path, required=True)
    parser.add_argument("--patch-flow-root", type=Path, required=True)
    parser.add_argument("--flow-route-root", type=Path, required=True)
    parser.add_argument("--threshold-admission-root", type=Path, required=True)
    parser.add_argument("--cache-schema", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flow_route_root.exists() or args.flow_route_root.is_symlink():
        raise FlowAuditError(
            f"route root already exists; refusing overwrite: {args.flow_route_root}"
        )
    for label, root in (
        ("episode root", args.episode_root),
        ("official flow root", args.official_flow_root),
        ("patch flow root", args.patch_flow_root),
        ("threshold admission root", args.threshold_admission_root),
    ):
        if not root.is_dir():
            raise FlowAuditError(f"{label} is absent: {root}")
    raw_audit_sha_path = Path(f"{args.raw_audit}.sha256")
    if not args.raw_audit.is_file() or not raw_audit_sha_path.is_file():
        raise FlowAuditError("raw audit artifact pair is absent")
    raw_audit_sha = sha256_file(args.raw_audit)
    sidecar_sha = raw_audit_sha_path.read_text(encoding="ascii").split()[0]
    if raw_audit_sha != sidecar_sha:
        raise FlowAuditError("raw audit sidecar mismatch")
    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    if raw_audit.get("status") != "audited_historical_summary_match" or raw_audit.get(
        "historical_three_leg_reference"
    ) != {
        "episode_0000": {"n_frames": 871, "switches": [218, 548]},
        "episode_0001": {"n_frames": 591, "switches": [192, 296]},
    }:
        raise FlowAuditError("raw audit did not pass the exact historical gate")

    cache_schema = _load_cache_schema(args.cache_schema)
    keyframe_budget = _schema_keyframe_budget(cache_schema)
    if args.threshold_admission_root.is_symlink():
        raise FlowAuditError("threshold admission root must not be a symlink")
    selected = _select_pairs(args.split_manifest, args.episode_root)
    expected_patch_files = {
        f"{key}/videos/chunk-000/{name}"
        for key in PATCH_EPISODES
        for name in FLOW_FILES
    }
    actual_patch_files = {
        path.relative_to(args.patch_flow_root).as_posix()
        for path in args.patch_flow_root.rglob("*.npz")
        if path.is_file()
    }
    if actual_patch_files != expected_patch_files:
        raise FlowAuditError(
            f"patch file set mismatch: "
            f"missing={sorted(expected_patch_files - actual_patch_files)} "
            f"extra={sorted(actual_patch_files - expected_patch_files)}"
        )
    expected_admission_files = {_admission_filename(key) for key in PATCH_EPISODES}
    admission_entries = list(args.threshold_admission_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in admission_entries):
        raise FlowAuditError(
            "threshold admission root must contain only physical journal files"
        )
    actual_admission_files = {path.name for path in admission_entries}
    if actual_admission_files != expected_admission_files:
        raise FlowAuditError(
            "threshold admission log set mismatch: "
            f"missing={sorted(expected_admission_files - actual_admission_files)} "
            f"extra={sorted(actual_admission_files - expected_admission_files)}"
        )

    rows = []
    for key, frames in sorted(selected.items()):
        relative_chunk = Path(key) / "videos/chunk-000"
        official_chunk = args.official_flow_root / relative_chunk
        patch_chunk = args.patch_flow_root / relative_chunk
        is_patch = key in PATCH_EPISODES
        source_chunk = patch_chunk if is_patch else official_chunk
        other_chunk = official_chunk if is_patch else patch_chunk
        source_files = [source_chunk / name for name in FLOW_FILES]
        other_files = [other_chunk / name for name in FLOW_FILES]
        if not all(path.is_file() for path in source_files):
            raise FlowAuditError(f"selected cache pair is incomplete: {source_chunk}")
        if any(path.is_file() for path in other_files):
            raise FlowAuditError(
                f"base/patch ownership is ambiguous for {key}: {other_chunk}"
            )
        patch_spec = PATCH_EPISODES.get(key)
        validation = _validate_pair(
            cache_schema,
            source_files[0],
            source_files[1],
            frames=frames,
            threshold=None,
            patch=is_patch,
        )
        if patch_spec is not None:
            minimum_threshold = float(patch_spec["minimum_threshold"])
            if _threshold_index(
                validation["flow_threshold"], "actual patch threshold"
            ) < _threshold_index(minimum_threshold, "minimum patch threshold"):
                raise FlowAuditError(
                    f"patch threshold is below its minimum for {key}: "
                    f"{validation['flow_threshold']} < {minimum_threshold}"
                )
            admission = _load_admission_record(
                args.threshold_admission_root,
                key,
                minimum_threshold,
                keyframe_budget,
                validation,
            )
            validation = {
                **validation,
                "threshold_admission": admission,
            }
        rows.append(
            {
                "episode": key,
                "source_id": "flow4096_patch" if is_patch else "official_base",
                "source_relative_chunk": relative_chunk.as_posix(),
                "validation": validation,
            }
        )

    counts = {
        "scenes": len({row["episode"].split("/", 1)[0] for row in rows}),
        "pairs": len(rows),
        "official_base": sum(row["source_id"] == "official_base" for row in rows),
        "flow4096_patch": sum(row["source_id"] == "flow4096_patch" for row in rows),
    }
    if counts != EXPECTED_COUNTS:
        raise FlowAuditError(f"merge counts mismatch: {counts} != {EXPECTED_COUNTS}")
    official_snapshot = hashlib.sha256(
        canonical_bytes([row for row in rows if row["source_id"] == "official_base"])
    ).hexdigest()

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": ROUTE_STATUS,
        "split_sha256": EXPECTED_SPLIT_SHA,
        "raw_audit_sha256": raw_audit_sha,
        "route_root": str(args.flow_route_root.resolve()),
        "source_roots": {
            "official_base": str(args.official_flow_root.resolve()),
            "flow4096_patch": str(args.patch_flow_root.resolve()),
        },
        "official_snapshot_semantics": (
            "canonical episode/source/size/schema/index/config/ground metadata; "
            "official multi-GB KV payloads are not re-hashed"
        ),
        "official_snapshot_sha256": official_snapshot,
        "patch_payloads_fully_sha256": True,
        "counts": counts,
        "pairs": rows,
    }

    args.flow_route_root.parent.mkdir(parents=True, exist_ok=True)
    args.flow_route_root.mkdir(parents=False, exist_ok=False)
    try:
        digest = _write_provenance(args.flow_route_root, record)
        try:
            consumed = load_route_registry(
                args.flow_route_root / "FLOW_ROUTE_PROVENANCE.json", digest
            )
        except FlowRoutingError as error:
            raise FlowAuditError(
                f"new route artifact is not consumable: {error}"
            ) from error
        if len(consumed.files_by_episode) != EXPECTED_COUNTS["pairs"]:
            raise FlowAuditError("new route registry changed pair count")
    except BaseException:
        # This root was proven absent and created by this process above.  Removing
        # it cannot affect either source tree and prevents a partial view from
        # being mistaken for a retryable success.
        shutil.rmtree(args.flow_route_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": record["status"],
                "flow_route_root": str(args.flow_route_root),
                "provenance_sha256": digest,
                **counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
