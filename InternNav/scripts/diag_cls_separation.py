"""Offline CLS-separation probe for MemNav retrieval.

Question this answers (decisive fork for the retrieval design):
    In the FROZEN DINOv2 CLS space that retrieval actually sees, is the true
    matching history frame separable from the negatives for a goal image?

Retrieval in v1 is CLS-cosine only: goal_cls = lingbot.dino(goal_image)["cls"]
vs the cached per-frame dino_cls (mem_cls). RetrievalHead adds TRAINABLE linear
projections, but this probe measures the RAW frozen space (both sides L2-norm'd,
no learned projection) — a lower bound on achievable separation. If raw-CLS
separation is at chance, the collapse we saw in training (`seen_match=0.00`) is a
representation problem, not an optimization one, and no anti-collapse trick fixes
it — retrieval needs dense features or a trainable encoder, not frozen CLS.

Groups measured under the loader's current dynamic-k rule:
  covis_revisit  — rendered B/C goals whose sampled eligible set contains a
                   co-visible positive.
  covis_novel    — rendered B/C goals with no eligible positive at the sampled k.
  goalA_novel    — first goals, which are always novel because their recent
                   approach is excluded from the revisit candidate set.

Per revisit sample the key scalars:
  AUC       — P(cos[positive] > cos[negative]) over all pos×neg pairs. 0.5=chance.
  retr@1    — is the single highest-cos labeled frame a positive?  (max_pos > max_neg)
  margin    — max_pos_cos - max_neg_cos  (how far a positive leads the field)

Run inside the SAME apptainer overlay as training (frames + goal_{j}.jpg live in
the squashfs; caches on host). See scripts/train_memnav/run_cls_probe.sbatch.
"""
import argparse
import os
import sys

import numpy as np
import torch

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from internnav.model.basemodel.memnav.memnav_policy import MemNavNet


def _auc(pos, neg):
    """P(pos > neg) + 0.5·ties over all pairs. 0.5 = chance."""
    if pos.size == 0 or neg.size == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_group", type=int, default=300)
    ap.add_argument("--dino_batch", type=int, default=16)
    ap.add_argument("--out", default=None, help="npz of per-sample scalars")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]
    feat = os.environ["MEMNAV_FEATURE_ROOT"]
    repo = os.environ["LINGBOT_REPO"]
    wts = os.environ["LINGBOT_WEIGHTS"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32))
    NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 4096))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    strict_features = os.environ.get("MEMNAV_STRICT_FEATURE_COVERAGE", "0") == "1"
    require_convention = os.environ.get("MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION", "0") == "1"
    ds = MemNav_Dataset(
        root, predict_size=24, image_size=518, lingbot_repo=repo,
        feature_root=feat, window_size=W, num_scale=NS,
        strict_feature_coverage=strict_features,
        require_generated_pose_convention=require_convention,
    )

    # ---- bucket one deterministic k draw per sample ----
    # tuple = (traj_idx, goal_path, mem_end, pos_idx, neg_idx)
    buckets = {"covis_revisit": [], "covis_novel": [], "goalA_novel": []}
    for s in ds.samples:
        k = int(rng.integers(int(s["k_lo"]), int(s["k_hi"]) + 1))
        pmask, nmask, _cand, nullp = ds._build_label(s, k)
        pos = np.where(pmask)[0]
        neg = np.where(nmask)[0]
        if nullp:
            grp = "covis_novel" if s["has_covis"] else "goalA_novel"
            if neg.size == 0:
                continue
        else:
            grp = "covis_revisit"
            if pos.size == 0 or neg.size == 0:
                continue
        buckets[grp].append((s["traj_idx"], s["goal_img_path"], k + 1, pos, neg))

    for g in buckets:
        arr = buckets[g]
        rng.shuffle(arr)
        buckets[g] = arr[: args.max_per_group]
    print("[probe] group sizes:", {g: len(v) for g, v in buckets.items()}, "| device:", device)

    # ---- frozen DINO trunk (same one that produced the cached dino_cls) ----
    net = MemNavNet(
        token_dim=384, heads=8, predict_size=24, temporal_depth=8, num_diffusion_iters=10,
        lingbot_kwargs=dict(lingbot_repo=repo, weights=wts, window=W, num_scale=NS, max_frame_num=MFN),
        novel_backbone_weights=os.environ['MEMNAV_DINO_WEIGHTS'],
        device=device,
    ).to(device).eval()

    dino_cache = {}   # traj_idx -> dino_cls [T,1024]

    def mem_of(ti):
        if ti not in dino_cache:
            dino_cache[ti] = ds._load_dino_cls(ti)
        return dino_cache[ti]

    @torch.no_grad()
    def goal_cls_of(paths):
        out = []
        for i in range(0, len(paths), args.dino_batch):
            imgs = torch.stack([ds._load_image_path(p) for p in paths[i : i + args.dino_batch]]).to(device)
            cls = net.lingbot.dino(imgs)["cls"].float().cpu().numpy()   # [b,1024]
            out.append(cls)
        return np.concatenate(out, 0)

    # ---- per-sample scalars ----
    results = {}
    for grp, items in buckets.items():
        if not items:
            results[grp] = {}
            continue
        gcls = goal_cls_of([it[1] for it in items])                    # [N,1024]
        gcls = gcls / (np.linalg.norm(gcls, axis=1, keepdims=True) + 1e-8)
        rows = []
        for (ti, _p, mem_end, pos, neg), gc in zip(items, gcls):
            mem = mem_of(ti)[:mem_end].astype(np.float32)
            mem = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)
            cos = mem @ gc                                             # [mem_end]
            ps, ns = cos[pos], cos[neg]
            if grp in ("covis_novel", "goalA_novel"):
                # no positives: how high can a DISTRACTOR score? (should be low)
                rows.append(dict(distractor_max=float(cos[neg].max()),
                                 distractor_mean=float(cos[neg].mean())))
            else:
                rows.append(dict(
                    auc=_auc(ps, ns),
                    retr1=float(ps.max() > ns.max()),
                    margin=float(ps.max() - ns.max()),
                    pos_mean=float(ps.mean()), pos_max=float(ps.max()),
                    neg_mean=float(ns.mean()), neg_max=float(ns.max()),
                    n_pos=int(pos.size), n_neg=int(neg.size),
                ))
        results[grp] = rows

    # ---- summary ----
    def col(rows, key):
        v = np.array([r[key] for r in rows if not np.isnan(r.get(key, np.nan))], dtype=np.float64)
        return v

    print("\n================ CLS-SEPARATION SUMMARY ================")
    for grp in ("covis_revisit",):
        rows = results[grp]
        if not rows:
            print(f"\n[{grp}] (empty)")
            continue
        auc = col(rows, "auc"); r1 = col(rows, "retr1"); mg = col(rows, "margin")
        pm = col(rows, "pos_mean"); nm = col(rows, "neg_mean")
        print(f"\n[{grp}]  N={len(rows)}")
        print(f"  AUC      mean={auc.mean():.3f}  median={np.median(auc):.3f}  "
              f"p10={np.percentile(auc,10):.3f}  frac>0.7={np.mean(auc>0.7):.2f}")
        print(f"  retr@1   frac correct = {r1.mean():.3f}   (positive is the single top labeled frame)")
        print(f"  margin   mean={mg.mean():+.3f}  median={np.median(mg):+.3f}  frac>0={np.mean(mg>0):.2f}")
        print(f"  cos      pos_mean={pm.mean():.3f}  neg_mean={nm.mean():.3f}  gap={pm.mean()-nm.mean():+.3f}")

    for novel_group in ("covis_novel", "goalA_novel"):
        nov = results.get(novel_group, [])
        if not nov:
            continue
        dmax = col(nov, "distractor_max")
        print(f"\n[{novel_group}] N={len(nov)}  distractor_max: mean={dmax.mean():.3f} "
              f"median={np.median(dmax):.3f} p90={np.percentile(dmax,90):.3f}")
        rev = results.get("covis_revisit", [])
        if rev:
            posmax = col(rev, "pos_max")
            print(f"   compare covis_revisit pos_max mean={posmax.mean():.3f}  "
                  f"→ {'SEPARABLE' if posmax.mean() > dmax.mean() + 0.03 else 'CONFUSABLE'}")

    print("\nRead: AUC≈0.5 / retr@1≈chance / gap≈0  →  frozen CLS carries NO match signal "
          "(collapse is representational; drop CLS-only retrieval). AUC≳0.8 / retr@1≳0.7 "
          "→ signal EXISTS; the training collapse is optimization (fix null shortcut / train longer).")

    if args.out:
        np.savez(args.out, **{f"{g}__{k}": col(r, k)
                              for g, r in results.items() for k in (r[0].keys() if r else [])})
        print(f"\nsaved per-sample scalars -> {args.out}")


if __name__ == "__main__":
    main()
