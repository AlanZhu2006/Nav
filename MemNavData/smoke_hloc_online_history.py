#!/usr/bin/env python3
"""Build one causal HLoc SfM model from frozen online-A decision frames.

This is a representation/runtime smoke only.  It deliberately reads no pose,
depth, role label, query image, navigation outcome, or future observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import pycolmap

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)


SCHEMA_VERSION = "hloc_online_history_reconstruction_smoke_v1_20260814"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def augment_temporal_pairs(path: Path, names: list[str], radius: int = 2) -> int:
    pairs = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] != fields[1]:
            pairs.add(tuple(fields))
    for index, first in enumerate(names):
        for other in range(index + 1, min(len(names), index + radius + 1)):
            pairs.add((first, names[other]))
    canonical = sorted(
        {tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1]}
    )
    path.write_text("\n".join(f"{first} {second}" for first, second in canonical) + "\n")
    return len(canonical)


def run(online_root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    trace_path = online_root / "online_a_trace.json"
    receipt_path = online_root / "receipt.json"
    trace = json.loads(trace_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    require(trace["reached"] is True and receipt["online_a_reached"] is True, "online A failed")
    require(receipt["online_a_trace_sha256"] == sha256_file(trace_path), "trace hash changed")
    render = receipt["render_contract"]
    require(render["all_rgb_hashes_match_original_online_rollout"] is True, "RGB audit failed")
    width = int(render["width"]); height = int(render["height"])
    hfov = float(render["horizontal_fov_deg"])
    require(width == 480 and height == 270, "sensor dimensions changed")

    # Access only plan step indices and independently stored RGB hashes.  The
    # trace's pose array is intentionally never read by this baseline.
    steps = [int(plan["step"]) for plan in trace["plans"]]
    require(steps == sorted(set(steps)) and len(steps) >= 8, "decision steps invalid")
    frame_hashes = list(receipt["rgb_frame_hashes"])
    require(max(steps) < len(frame_hashes), "decision step outside RGB receipt")

    images = out / "images"; outputs = out / "hloc"; sfm_dir = outputs / "sfm"
    images.mkdir(parents=True)
    names = []
    input_rows = []
    for index, step in enumerate(steps):
        source = online_root / "rgb" / f"{step:06d}.jpg"
        require(source.is_file(), f"missing online RGB frame {step}")
        digest = sha256_file(source)
        require(digest == frame_hashes[step], f"online RGB hash changed at {step}")
        name = f"decision_{index:03d}_step_{step:06d}.jpg"
        shutil.copyfile(source, images / name)
        names.append(name)
        input_rows.append({"decision_index": index, "step": step, "sha256": digest})

    started = time.perf_counter()
    retrieval_path = extract_features.main(
        extract_features.confs["netvlad"], images, outputs,
        image_list=names, overwrite=False,
    )
    pairs_path = outputs / "pairs-netvlad-plus-temporal.txt"
    pairs_from_retrieval.main(
        retrieval_path, pairs_path, num_matched=min(5, len(names) - 1),
        query_list=names, db_list=names,
    )
    pair_count = augment_temporal_pairs(pairs_path, names, radius=2)
    feature_conf = extract_features.confs["superpoint_aachen"]
    feature_path = extract_features.main(
        feature_conf, images, outputs, image_list=names, overwrite=False,
    )
    matches_path = outputs / "matches-superpoint-lightglue.h5"
    match_features.main(
        match_features.confs["superpoint+lightglue"], pairs_path,
        feature_path, matches=matches_path, overwrite=False,
    )

    focal = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    model = reconstruction.main(
        sfm_dir, images, pairs_path, feature_path, matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_list=names,
        image_options={
            "camera_model": "PINHOLE",
            "camera_params": f"{focal},{focal},{width / 2.0},{height / 2.0}",
        },
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    registered = int(model.num_reg_images()) if model is not None else 0
    points = int(model.num_points3D()) if model is not None else 0
    registered_names = (
        {image.name for image in model.images.values()}
        if model is not None else set()
    )
    final_name = names[-1]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "consumed online-history representation smoke; no query or policy outcome",
        "online_root": str(online_root.resolve()),
        "source_trace_sha256": sha256_file(trace_path),
        "source_receipt_sha256": sha256_file(receipt_path),
        "pose_fields_read": False,
        "depth_read": False,
        "query_image_read": False,
        "analysis_role_read": False,
        "decision_frames": input_rows,
        "input_images": len(names),
        "retrieval_plus_temporal_pairs": pair_count,
        "registered_images": registered,
        "registered_fraction": registered / len(names),
        "final_online_a_decision_frame": final_name,
        "final_online_a_decision_frame_registered": (
            final_name in registered_names
        ),
        "points3D": points,
        "elapsed_seconds": elapsed,
        "camera": {
            "model": "PINHOLE", "width": width, "height": height,
            "horizontal_fov_deg": hfov, "fx": focal, "fy": focal,
            "cx": width / 2.0, "cy": height / 2.0,
        },
        "passed": model is not None and registered >= max(8, len(names) // 2),
    }
    (out / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.online_root, args.out)
    print(json.dumps({
        key: result[key] for key in (
            "input_images", "retrieval_plus_temporal_pairs",
            "registered_images", "points3D", "elapsed_seconds", "passed",
        )
    }, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
