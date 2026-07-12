#!/usr/bin/env python
"""Clean up MP3D generation artifacts found by the integrity validator.

BENIGN stale-tail episodes: delete rgb (.jpg) and depth (.png) frames whose index >= n_frames.
  HARDENED safety gate, ALL must hold or the episode is SKIPPED (left for regen):
    (1) parquet num_rows == meta n_frames
    (2) frames 0..n_frames-1 all present (rgb and depth)
    (3) mtime coherence: max(mtime of orphan frames >= nf) < min(mtime of kept frames 0..nf-1)
        i.e. every orphan is strictly older than every kept frame -> orphans are a prior run.
INCOMPLETE episode (no gen_meta.json): rm -rf whole episode dir so it regenerates clean.

Run with --apply to actually delete; default is dry-run.
"""
import os, sys, json, shutil, glob
import pyarrow.parquet as pq

OUT = "/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data"
APPLY = "--apply" in sys.argv

BENIGN = [
 "mp3d_3leg/ac26ZMwG7aT/episode_0000","mp3d_3leg/ac26ZMwG7aT/episode_0002","mp3d_3leg/ac26ZMwG7aT/episode_0003",
 "mp3d_3leg/ac26ZMwG7aT/episode_0004","mp3d_3leg/ac26ZMwG7aT/episode_0006","mp3d_3leg/ac26ZMwG7aT/episode_0009",
 "mp3d_3leg/ac26ZMwG7aT/episode_0011","mp3d_3leg/b8cTxDM8gDG/episode_0000","mp3d_3leg/b8cTxDM8gDG/episode_0002",
 "mp3d_3leg/b8cTxDM8gDG/episode_0003","mp3d_3leg/cV4RVeZvu5T/episode_0000","mp3d_3leg/cV4RVeZvu5T/episode_0001",
 "mp3d_3leg/cV4RVeZvu5T/episode_0004","mp3d_3leg/fzynW3qQPVF/episode_0000","mp3d_3leg/fzynW3qQPVF/episode_0003",
 "mp3d_3leg/fzynW3qQPVF/episode_0004","mp3d_3leg/fzynW3qQPVF/episode_0005","mp3d_3leg/fzynW3qQPVF/episode_0006",
 "mp3d_3leg/fzynW3qQPVF/episode_0008","mp3d_3leg/fzynW3qQPVF/episode_0010","mp3d_3leg/fzynW3qQPVF/episode_0011",
 "mp3d_3leg/gYvKGZ5eRqb/episode_0003","mp3d_3leg/gYvKGZ5eRqb/episode_0006","mp3d_3leg/gYvKGZ5eRqb/episode_0007",
 "mp3d_3leg/gYvKGZ5eRqb/episode_0008","mp3d_3leg/gYvKGZ5eRqb/episode_0009","mp3d_3leg/gYvKGZ5eRqb/episode_0010",
]
INCOMPLETE = ["mp3d_3leg/e9zR4mvMWw7/episode_0024"]


def mtimes(d, idxs, ext):
    return {i: os.stat(os.path.join(d, f"{i}{ext}")).st_mtime for i in idxs}


total_del = 0
skipped = []
for rel in BENIGN:
    ep = os.path.join(OUT, rel)
    nf = int(json.load(open(os.path.join(ep, "meta/gen_meta.json")))["n_frames"])
    pqp = os.path.join(ep, "data/chunk-000/episode_000000.parquet")
    nrows = pq.ParquetFile(pqp).metadata.num_rows
    rgb = os.path.join(ep, "videos/chunk-000/observation.images.rgb")
    dep = os.path.join(ep, "videos/chunk-000/observation.images.depth")
    ridx = {int(f[:-4]) for f in os.listdir(rgb) if f.endswith(".jpg")}
    didx = {int(f[:-4]) for f in os.listdir(dep) if f.endswith(".png")}

    # gate 1 + 2
    if nrows != nf or (set(range(nf)) - ridx) or (set(range(nf)) - didx):
        skipped.append((rel, f"count gate: nf={nf} rows={nrows} missing_low_rgb={len(set(range(nf))-ridx)} depth={len(set(range(nf))-didx)}"))
        continue

    keep_r = [i for i in ridx if i < nf];  orph_r = [i for i in ridx if i >= nf]
    keep_d = [i for i in didx if i < nf];  orph_d = [i for i in didx if i >= nf]

    # gate 3: mtime coherence, per modality
    def coherent(d, keep, orph, ext):
        if not orph:
            return True
        km = mtimes(d, keep, ext); om = mtimes(d, orph, ext)
        return max(om.values()) < min(km.values())

    if not (coherent(rgb, keep_r, orph_r, ".jpg") and coherent(dep, keep_d, orph_d, ".png")):
        skipped.append((rel, "mtime gate: orphan not strictly older than kept -> LEAVE for regen"))
        continue

    print(f"OK {rel}: keep 0..{nf-1}, drop {len(orph_r)} rgb + {len(orph_d)} depth orphans")
    if APPLY:
        for i in orph_r: os.remove(os.path.join(rgb, f"{i}.jpg"))
        for i in orph_d: os.remove(os.path.join(dep, f"{i}.png"))
    total_del += len(orph_r) + len(orph_d)

for rel in INCOMPLETE:
    ep = os.path.join(OUT, rel)
    n = len(glob.glob(os.path.join(ep, "**"), recursive=True))
    print(f"RM {rel}: rm -rf whole dir (incomplete, no meta) ~{n} entries")
    if APPLY:
        shutil.rmtree(ep, ignore_errors=True)

print("\n--- SKIPPED (failed hardened gate, left untouched) ---")
for r, why in skipped:
    print(f"  {r}  ::  {why}")
print(f"\n{'APPLIED' if APPLY else 'DRY-RUN'}: orphan frames {'deleted' if APPLY else 'to delete'} = {total_del}"
      f"; incomplete dirs removed = {len(INCOMPLETE)}; episodes skipped = {len(skipped)}")
