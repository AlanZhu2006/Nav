#!/usr/bin/env python3
"""Package only the existing train40 probe's causal RGB dependencies.

This CPU utility does not run a model, alter source data, or select new pairs.
Ground-truth parquet/metadata are placed under a separate supervision directory.
The geometry producer must not read that directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


SCALE_SHA = "f6f28e20b9f21a764132af04b425f23b65e1dc779798d831a571a7df26bd78db"
ROUTES_SHA = "c840bb4395554eeb6616322db578f4b01ebc7af1d5060cd091db4638b09e8dc5"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pinned(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"source SHA mismatch: {path}")
    return json.loads(raw)


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return digest(len(header).to_bytes(8, "big") + header + value.tobytes())


def prefix_requirements(pairs: list[dict]) -> dict[str, dict]:
    histories: dict[str, dict] = {}
    for pair in pairs:
        rel = Path(pair["candidate_relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("candidate path must be relative")
        key = "/".join(rel.parts[:2])
        anchor, decision = pair["candidate_frame"], pair["decision_frame"]
        if not 8 <= anchor < decision or decision < 64:
            raise ValueError("invalid historical anchor/scale prefix")
        entry = histories.setdefault(key, {"anchors": {}, "minimum_decision": decision})
        entry["minimum_decision"] = min(entry["minimum_decision"], decision)
        entry["anchors"].setdefault(anchor, []).append(pair["pair_id"])
    for entry in histories.values():
        entry["frame_count"] = max(64, max(entry["anchors"]) + 1)
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    pairs = manifest["pairs"]
    if len(pairs) != 123 or manifest["deduplicated_pairs"] != 123:
        raise ValueError("this package is bound to the existing 123-pair probe")
    allowed = set(manifest["train_scene_ids"] + manifest["validation_scene_ids"])
    if any(p["scene"] not in allowed for p in pairs):
        raise ValueError("pair outside the fixed train40 scene universe")
    scale = load_pinned(args.scale, SCALE_SHA)
    routes = load_pinned(args.routes, ROUTES_SHA)
    scales = {f"{r['scene']}/{r['episode']}": r for r in scale["records"]}
    route_map = {r["episode"]: r for r in routes["pairs"]}
    histories = prefix_requirements(pairs)
    selected_scales = {}
    selected_routes = {}
    for key, entry in histories.items():
        record = scales[key]
        if (record["split_role"] != "train" or not record["valid"]
                or record["prefix_end_frame_exclusive"] != 64
                or record["prefix_end_frame_exclusive"] > entry["minimum_decision"]):
            raise ValueError(f"invalid causal metric receipt for {key}")
        selected_scales[key] = record
        selected_routes[key] = route_map[key]

    # Check pair identities against the raw dataset before packing any payload.
    for pair in pairs:
        for kind in ("query", "candidate"):
            path = args.raw_root / pair[f"{kind}_relative_path"]
            if digest(path.read_bytes()) != pair[f"{kind}_sha256"]:
                raise ValueError(f"raw RGB identity changed: {path}")
    rgb_files = []
    for key, entry in sorted(histories.items()):
        for frame in range(entry["frame_count"]):
            rel = f"{key}/videos/chunk-000/observation.images.rgb/{frame}.jpg"
            path = args.raw_root / rel
            rgb_files.append((path, rel, path.stat().st_size))
    plan = {
        "schema": "anchor_relation_causal_rgb_input_pack_v1",
        "probe_manifest_sha256": digest(manifest_bytes),
        "scale_artifact_sha256": SCALE_SHA, "routes_artifact_sha256": ROUTES_SHA,
        "pair_count": len(pairs), "history_count": len(histories),
        "rgb_count": len(rgb_files), "rgb_bytes": sum(n for _, _, n in rgb_files),
        "history_geometry_scope": "full RGB prefix through each anchor; never query RGB",
        "scale_scope": "existing causal first64 height receipt, not production first40",
        "source_raw_root": str(args.raw_root),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(args.out, "w:xz" if args.out.suffix == ".xz" else "w") as archive:
        def add_bytes(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
            files.append({"path": name, "bytes": len(data), "sha256": digest(data)})

        add_bytes("probe_manifest.json", manifest_bytes)
        for number, (path, rel, _) in enumerate(rgb_files, 1):
            add_bytes("rgb/" + rel, path.read_bytes())
            if number % 2000 == 0:
                print(f"[pack RGB] {number}/{len(rgb_files)}", flush=True)
        rgb_index = {r["path"]: r for r in files if r["path"].startswith("rgb/")}
        for key in histories:
            prefix = []
            for frame in range(64):
                rel = f"{key}/videos/chunk-000/observation.images.rgb/{frame}.jpg"
                record = rgb_index["rgb/" + rel]
                prefix.append({"path": rel, "bytes": record["bytes"],
                               "content_sha256": record["sha256"]})
            canonical = (json.dumps(prefix, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False) + "\n").encode()
            if digest(canonical) != scales[key]["rgb_prefix"]["content_sequence_sha256"]:
                raise ValueError(f"height receipt and RGB prefix differ for {key}")
        # Only predicted historical camera arrays are exported, not multi-GB KV.
        camera_records = {}
        for key, entry in sorted(histories.items()):
            route = route_map[key]
            camera_path = (Path(routes["source_roots"][route["source_id"]])
                           / route["source_relative_chunk"] / "lingbot_cam_cache.npz")
            raw_camera = camera_path.read_bytes()
            with np.load(io.BytesIO(raw_camera), allow_pickle=False) as camera:
                signature = str(camera["precompute_signature"].item())
                if signature != scales[key]["precompute_signature"]:
                    raise ValueError(f"camera source signature differs for {key}")
                pose = camera["cam_pose_enc"][:entry["frame_count"]]
                if pose.shape != (entry["frame_count"], 9) or not np.isfinite(pose).all():
                    raise ValueError(f"invalid predicted camera prefix for {key}")
                if array_digest(pose[:64]) != scales[key]["cam_pose_prefix_sha256"]:
                    raise ValueError(f"height receipt and pose prefix differ for {key}")
                camera_records[key] = {
                    "cam_pose_enc": pose.tolist(),
                    "source_camera_path": str(camera_path),
                    "source_camera_sha256": digest(raw_camera),
                    "precompute_signature": signature,
                }
        # Raw GT is for independent label validation only, never a model input.
        roots = set(histories)
        roots.update("/".join(Path(p["query_relative_path"]).parts[:2]) for p in pairs)
        for root in sorted(roots):
            for suffix in ("meta/gen_meta.json", "data/chunk-000/episode_000000.parquet"):
                add_bytes(f"supervision_only/{root}/{suffix}",
                          (args.raw_root / root / suffix).read_bytes())
        metadata = {"plan": plan, "histories": histories,
                    "scales": selected_scales, "camera_predictions": camera_records,
                    "source_routes": selected_routes,
                    "scale_configuration": scale["configuration"]}
        add_bytes("history_inputs.json", json.dumps(metadata, indent=2).encode())
        add_bytes("FILES.json", json.dumps(files, indent=2).encode())
    plan["archive_bytes"] = args.out.stat().st_size
    with args.out.open("rb") as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    plan["archive_sha256"] = h.hexdigest()
    receipt = args.out.with_suffix(args.out.suffix + ".receipt.json")
    if receipt.exists():
        raise FileExistsError(receipt)
    receipt.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == "__main__":
    main()
