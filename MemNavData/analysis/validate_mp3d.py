#!/usr/bin/env python
"""Integrity check for generated MP3D revisit episodes (scenes 54-89 on disk).

Per episode, enforce the contract: gen_meta.n_frames == #rgb jpgs == #depth pngs ==
parquet num_rows, with no zero-byte frames/parquet and a readable parquet + meta.
Per scene, report episode completeness vs target (15 2leg / 25 3leg) and flag shortfalls
(distinguishing plausibly scene-limited from crash-truncated is left to the human, but we
surface the counts).
"""
import os, sys, json, glob
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data"
TARGET = {"mp3d_2leg": 15, "mp3d_3leg": 25}


def scandir_stats(d, ext):
    """Return (count, n_zero_byte) for files ending in ext under d."""
    n = z = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith(ext) and e.is_file():
                    n += 1
                    if e.stat().st_size == 0:
                        z += 1
    except FileNotFoundError:
        return -1, 0
    return n, z


def check_episode(ep):
    """Return (ok: bool, problems: list[str])."""
    probs = []
    meta_p = os.path.join(ep, "meta", "gen_meta.json")
    pq_p = os.path.join(ep, "data", "chunk-000", "episode_000000.parquet")
    rgb_d = os.path.join(ep, "videos", "chunk-000", "observation.images.rgb")
    dep_d = os.path.join(ep, "videos", "chunk-000", "observation.images.depth")

    # meta
    if not os.path.isfile(meta_p):
        return False, ["missing gen_meta.json"]
    try:
        with open(meta_p) as f:
            meta = json.load(f)
        nf = int(meta["n_frames"])
    except Exception as e:
        return False, [f"unreadable/invalid meta: {e}"]
    if nf <= 0:
        probs.append(f"meta n_frames={nf}")

    # parquet
    if not os.path.isfile(pq_p):
        probs.append("missing parquet")
        nrows = None
    elif os.path.getsize(pq_p) == 0:
        probs.append("zero-byte parquet")
        nrows = None
    else:
        try:
            nrows = pq.ParquetFile(pq_p).metadata.num_rows
        except Exception as e:
            probs.append(f"unreadable parquet: {e}")
            nrows = None

    # frames
    nrgb, zrgb = scandir_stats(rgb_d, ".jpg")
    ndep, zdep = scandir_stats(dep_d, ".png")
    if nrgb < 0:
        probs.append("missing rgb dir")
    if ndep < 0:
        probs.append("missing depth dir")
    if zrgb:
        probs.append(f"{zrgb} zero-byte rgb")
    if zdep:
        probs.append(f"{zdep} zero-byte depth")

    # the contract: all counts == n_frames
    counts = {"meta": nf, "parquet": nrows, "rgb": nrgb if nrgb >= 0 else None,
              "depth": ndep if ndep >= 0 else None}
    present = {k: v for k, v in counts.items() if v is not None}
    if len(set(present.values())) > 1:
        probs.append("count mismatch " + " ".join(f"{k}={v}" for k, v in counts.items()))

    return (len(probs) == 0), probs


def main():
    total_ep = ok_ep = 0
    bad = []            # (path, problems)
    scene_rows = []     # (scene, leg, n_ok, n_total, target)
    for leg, tgt in TARGET.items():
        scenes = sorted(glob.glob(os.path.join(OUT, leg, "*")))
        for sc in scenes:
            sid = os.path.basename(sc)
            eps = sorted(glob.glob(os.path.join(sc, "episode_*")))
            n_ok = 0
            for ep in eps:
                total_ep += 1
                ok, probs = check_episode(ep)
                if ok:
                    ok_ep += 1
                    n_ok += 1
                else:
                    bad.append((os.path.relpath(ep, OUT), probs))
            scene_rows.append((sid, leg, n_ok, len(eps), tgt))

    print(f"=== EPISODE INTEGRITY: {ok_ep}/{total_ep} pass the count contract ===\n")

    print("=== SCENES BELOW TARGET (shortfall = target - complete) ===")
    short = [r for r in scene_rows if r[2] < r[4]]
    if not short:
        print("  none — every scene meets its target episode count")
    for sid, leg, nok, ntot, tgt in sorted(short, key=lambda r: (r[2]-r[4])):
        print(f"  {leg:9s} {sid:14s} complete={nok:2d}/{tgt}  (dirs on disk={ntot})")

    print("\n=== BAD EPISODES (failed the integrity contract) ===")
    if not bad:
        print("  none")
    for p, probs in bad:
        print(f"  {p}\n      -> {'; '.join(probs)}")

    print(f"\nSUMMARY: {ok_ep}/{total_ep} episodes clean; "
          f"{len(bad)} bad; {len(short)} scene-legs below target.")


if __name__ == "__main__":
    main()
