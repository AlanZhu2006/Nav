#!/usr/bin/env python3
"""Full causal RGB replay -> cached per-anchor geometry; no goal or GT forward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from MemNavData.anchor_relation_geometry import (
    padded_image_mask, depth_to_patch_geometry, intrinsics_from_pose_fov,
)
from MemNavData.diag_m2p_s1_gct_query import _build_model


WEIGHTS_SHA = "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--history", action="append", default=[])
    p.add_argument("--lingbot-repo", type=Path,
                   default=Path("/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map"))
    p.add_argument("--weights", type=Path,
                   default=Path("/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if file_sha(args.weights) != WEIGHTS_SHA:
        raise ValueError("frozen LingBot weights changed")
    info_path = args.input_dir / "history_inputs.json"
    metadata = json.loads(info_path.read_text())
    files = {r["path"]:r for r in json.loads((args.input_dir / "FILES.json").read_text())}
    if file_sha(info_path) != files["history_inputs.json"]["sha256"]:
        raise ValueError("history metadata changed after packaging")
    histories = metadata["histories"]
    selected = sorted(args.history or histories)
    if len(set(selected)) != len(selected) or not set(selected) <= set(histories):
        raise ValueError("unknown/duplicate history request")
    # Validate only the selected original history dependencies before loading GPU.
    for key in selected:
        entry = histories[key]
        for frame in range(max(map(int, entry["anchors"])) + 1):
            rel = f"rgb/{key}/videos/chunk-000/observation.images.rgb/{frame}.jpg"
            if file_sha(args.input_dir / rel) != files[rel]["sha256"]:
                raise ValueError(f"historical RGB changed: {rel}")
    args.out_dir.mkdir(parents=True)
    config = vars(args) | {"history_inputs_sha256": file_sha(info_path),
                           "weights_sha256": WEIGHTS_SHA,
                           "selected_histories": selected,
                           "query_rgb_consumed": False, "GT_consumed": False,
                           "window":32, "num_scale":8,
                           "geometry": "full causal prefix depth at anchor; cached historical FoV",
                           "scale": "existing first64 causal height receipt",
                           "schema":"anchor_relation_geometry_v1"}
    config = {k:str(v) if isinstance(v, Path) else v for k,v in config.items()}
    (args.out_dir / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    torch.set_num_threads(8)
    args.image_size, args.patch_size = 518, 14
    args.window, args.num_scale, args.max_frame_num, args.camera_iterations = 32, 8, 4096, 4
    torch.manual_seed(11)
    model = _build_model(args)
    sys.path.insert(0,str(args.lingbot_repo))
    import lingbot_map.utils.load_fn as lf
    lf.tqdm = lambda iterable, **kwargs: iterable
    records = []
    overall = time.monotonic()
    for hi, key in enumerate(selected):
        history = histories[key]
        anchors = sorted(map(int, history["anchors"]))
        last = anchors[-1]
        predictions = metadata["camera_predictions"][key]["cam_pose_enc"]
        scale_receipt = metadata["scales"][key]
        scale = float(scale_receipt["metric_scale_m_per_raw"])
        rgb_dir = args.input_dir / "rgb" / key / "videos/chunk-000/observation.images.rgb"
        model.clean_kv_cache()
        started = time.monotonic()
        output = {}
        def consume(images, index, scale_block=False):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                images = images[None].to(args.device)
                aggregated, psi = model._aggregate_features(
                    images, num_frame_for_scale=8,
                    num_frame_per_block=8 if scale_block else 1)
                if index not in anchors:
                    return
                result = model._predict_depth(aggregated, images, psi)
                depth = result["depth"][0,-1,...,0].float().cpu()
                confidence = result["depth_conf"][0,-1].float().cpu()
            with Image.open(rgb_dir / f"{index}.jpg") as image:
                width,height = ImageOps.exif_transpose(image).size
            mask = padded_image_mask((height,width))
            k = intrinsics_from_pose_fov(predictions[index])
            geom = depth_to_patch_geometry(depth,confidence,k,mask,scale)
            if geom["roundtrip_projection_max_px"] > .001:
                raise ValueError("camera projection is inconsistent")
            for name, value in geom.items():
                output[f"{index}/{name}"] = np.asarray(value)
            output[f"{index}/intrinsic"] = k
            output[f"{index}/raw_hw"] = np.asarray([height,width])
            # Two real raster witnesses for manual inspection, not training inputs.
            if hi == 0 and index in anchors[:2]:
                np.savez_compressed(args.out_dir / f"witness_{index}.npz",
                                    depth=depth.numpy(),confidence=confidence.numpy(),
                                    content_mask=mask.numpy(),intrinsic=k,
                                    rgb_path=str(rgb_dir/f"{index}.jpg"))
            print(f"[anchor] {key}:{index} valid={int(geom['valid_mask'].sum())}/64 "
                  f"normalization={geom['normalization_m']:.4f}m", flush=True)
        first = lf.load_and_preprocess_images([str(rgb_dir/f"{j}.jpg") for j in range(8)],
                   mode="pad",image_size=518,patch_size=14)
        consume(first,7,True)
        del first
        for begin in range(8,last+1,16):
            stop = min(last+1,begin+16)
            batch = lf.load_and_preprocess_images([str(rgb_dir/f"{j}.jpg") for j in range(begin,stop)],
                         mode="pad",image_size=518,patch_size=14)
            for j in range(begin,stop):
                consume(batch[j-begin:j-begin+1],j)
            del batch
            if stop % 64 == 8 or stop == last+1:
                print(f"[history {hi+1}/{len(selected)}] {key} {stop}/{last+1} "
                      f"{time.monotonic()-started:.1f}s",flush=True)
        output["anchors"] = np.asarray(anchors,dtype=np.int64)
        output["scale_m_per_raw"] = np.asarray(scale)
        path = args.out_dir / (key.replace("/","__")+".npz")
        np.savez_compressed(path, **output)
        records.append({"history":key,"anchors":anchors,"path":path.name,
                        "sha256":file_sha(path),"causal_rgb_frames":last+1,
                        "elapsed_seconds":time.monotonic()-started})
        (args.out_dir / "progress.json").write_text(json.dumps({"completed":records,
            "total":len(selected),"elapsed_seconds":time.monotonic()-overall},indent=2)+"\n")
    (args.out_dir / "completion.json").write_text(json.dumps({"completed":True,
        "histories":records,"anchors":sum(len(r["anchors"]) for r in records),
        "elapsed_seconds":time.monotonic()-overall,
        "configuration_sha256":file_sha(args.out_dir/"configuration.json")},indent=2)+"\n")
    print(f"[complete] {len(records)} histories; {time.monotonic()-overall:.1f}s",flush=True)


if __name__ == "__main__":
    main()
