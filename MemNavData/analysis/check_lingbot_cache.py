#!/usr/bin/env python
"""Integrity check for precomputed LingBot KV caches.

For every lingbot_cache.npz / lingbot_cam_cache.npz under FEAT_ROOT:
  - np.load + FORCE-READ every array (a truncated/corrupt npz raises here, not at open)
  - verify the expected keys are present
  - verify each lingbot_cache.npz (the gate file) has its lingbot_cam_cache.npz sibling
Skips files modified within --fresh-secs (default 90) so it doesn't flag caches that an
active writer is still producing. With --delete, removes corrupt/truncated files so a
re-run of the (now atomic) precompute recomputes them.

Usage:
  python check_lingbot_cache.py [FEAT_ROOT] [--delete] [--fresh-secs N] [--expect-scale]
"""
import os, sys, time, argparse
import numpy as np

DEFAULT_ROOT = "/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data"
OUT_NAME = "lingbot_cache.npz"
CAM_NAME = "lingbot_cam_cache.npz"


def read_all(path):
    """Load an npz and force-read every member. Returns (ok, keys, err)."""
    try:
        with np.load(path, allow_pickle=True) as z:
            keys = list(z.files)
            for k in keys:
                _ = z[k].shape  # forces decompression/read of the member
        return True, keys, None
    except Exception as e:  # BadZipFile, EOFError, ValueError, ...
        return False, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("--delete", action="store_true", help="delete corrupt files so they recompute")
    ap.add_argument("--fresh-secs", type=int, default=90, help="skip files modified within N secs")
    ap.add_argument("--expect-scale", action="store_true", help="require scale_k/scale_v in out cache")
    args = ap.parse_args()

    out_keys = {"dino_cls", "anchor_k", "anchor_v", "meta"}
    if args.expect_scale:
        out_keys |= {"scale_k", "scale_v"}
    cam_keys = {"cam_k", "cam_v", "cam_pose_enc", "cam_meta"}

    now = time.time()
    n_out = n_cam = n_ok = 0
    corrupt, missing_keys, skipped, orphan_out = [], [], 0, []

    for dirpath, _, files in os.walk(args.root):
        for name, want in ((OUT_NAME, out_keys), (CAM_NAME, cam_keys)):
            if name not in files:
                continue
            p = os.path.join(dirpath, name)
            if now - os.path.getmtime(p) < args.fresh_secs:
                skipped += 1
                continue
            if name == OUT_NAME:
                n_out += 1
                if not os.path.exists(os.path.join(dirpath, CAM_NAME)):
                    orphan_out.append(p)
            else:
                n_cam += 1
            ok, keys, err = read_all(p)
            if not ok:
                corrupt.append((p, err))
                if args.delete:
                    os.remove(p)
                continue
            if not want.issubset(set(keys)):
                missing_keys.append((p, sorted(want - set(keys))))
                continue
            n_ok += 1

    print(f"root: {args.root}")
    print(f"checked: {n_out} out-caches + {n_cam} cam-caches; OK={n_ok}; "
          f"skipped(fresh<{args.fresh_secs}s)={skipped}")
    print(f"\nCORRUPT/TRUNCATED ({len(corrupt)}){' [DELETED]' if args.delete else ''}:")
    for p, err in corrupt:
        print(f"  {p}\n      {err}")
    print(f"\nMISSING KEYS ({len(missing_keys)}):")
    for p, miss in missing_keys:
        print(f"  {p}  missing={miss}")
    print(f"\nORPHAN out-cache (no cam sibling) ({len(orphan_out)}):")
    for p in orphan_out:
        print(f"  {p}")
    bad = len(corrupt) + len(missing_keys)
    print(f"\nSUMMARY: {n_ok} clean, {bad} bad, {len(orphan_out)} orphan, {skipped} skipped(fresh).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
