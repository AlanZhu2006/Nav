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

Groups measured:
  covis_revisit  — B/C goals (RENDERED goal image) that keep a covis positive.
                   The domain-gap case: does a render match its real frames?
  goalA_revisit  — goal A, whose goal image IS a trajectory frame (no render
                   gap). Control: separation SHOULD be easy here; if it isn't,
                   the failure is the head/null-collapse, not the domain gap.
  covis_novel    — novel goals (no positive). We report their distractor max-cos:
                   it should sit BELOW the revisit positives, else `null` is
                   confusable with a real match.

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
    ap.add_argument("--retrieval_ckpt", default=None,
                    help="memnav.ckpt to pull core.retrieval.* from; adds projected-space "
                         "(p_*) scalars next to the raw-CLS ones on the SAME samples")
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]
    feat = os.environ["MEMNAV_FEATURE_ROOT"]
    repo = os.environ["LINGBOT_REPO"]
    wts = os.environ["LINGBOT_WEIGHTS"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32))
    NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 2048))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)

    # ---- bucket samples with the CURRENT dynamic-per-k label API: one k per
    # goal-sample (uniform in [k_lo..k_hi], fixed seed), label built at that k.
    # goalA is all-negative by construction now (never a revisit) -> novel pool.
    # Tuple: (traj_idx, goal_path, mem_end=k+1, pos_idx, neg_idx, cand_idx).
    buckets = {"covis_revisit": [], "covis_novel": [], "goalA_novel": []}
    for si, s in enumerate(ds.samples):
        k = int(rng.integers(int(s["k_lo"]), int(s["k_hi"]) + 1))
        pmask, nmask, cmask, nullp = ds._build_label(s, k)
        pos, neg, cand = np.where(pmask)[0], np.where(nmask)[0], np.where(cmask)[0]
        if cand.size == 0:
            continue
        if not s["has_covis"]:
            grp = "goalA_novel"
        else:
            grp = "covis_novel" if nullp else "covis_revisit"
        # revisit needs >=1 pos AND >=1 neg to be separable
        if grp == "covis_revisit" and (pos.size == 0 or neg.size == 0):
            continue
        buckets[grp].append((s["traj_idx"], s["goal_img_path"], k + 1, pos, neg, cand))

    for g in buckets:
        arr = buckets[g]
        rng.shuffle(arr)
        buckets[g] = arr[: args.max_per_group]
    print("[probe] group sizes:", {g: len(v) for g, v in buckets.items()}, "| device:", device)

    # ---- frozen DINO trunk (same one that produced the cached dino_cls) ----
    net = MemNavNet(
        token_dim=384, heads=8, predict_size=24, temporal_depth=8, num_diffusion_iters=10,
        lingbot_kwargs=dict(lingbot_repo=repo, weights=wts, window=W, num_scale=NS, max_frame_num=MFN),
        device=device,
    ).to(device).eval()

    # ---- optional: trained RetrievalHead (proj + temp + gate) from a checkpoint ----
    # Projection is applied to the RAW (un-normalized) CLS on both sides, then L2 —
    # exactly RetrievalHead.forward. Gate here maxes over pos∪neg (the labeled subset
    # of E), a close proxy for the training-time cand_mask max.
    head = None
    if args.retrieval_ckpt:
        sd = torch.load(args.retrieval_ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        pref = "core.retrieval."
        head = {k[len(pref):]: sd[k].float().numpy() for k in sd if k.startswith(pref)}
        gate_a, gate_b = float(head["gate_a"]), float(head["gate_b"])
        print(f"[probe] RetrievalHead from {args.retrieval_ckpt}: "
              f"temp={float(np.clip(np.exp(head['log_temp']), 0.01, 1.0)):.4f} "
              f"gate_a={gate_a:.3f} gate_b={gate_b:.3f}")

    def _proj(x, which):
        y = x @ head[f"proj_{which}.weight"].T + head[f"proj_{which}.bias"]
        return y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)

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
        gcls_raw = goal_cls_of([it[1] for it in items])                # [N,1024]
        gcls = gcls_raw / (np.linalg.norm(gcls_raw, axis=1, keepdims=True) + 1e-8)
        rows = []
        for (ti, _p, mem_end, pos, neg, cand), gc, gcr in zip(items, gcls, gcls_raw):
            mem_raw = mem_of(ti)[:mem_end].astype(np.float32)
            mem = mem_raw / (np.linalg.norm(mem_raw, axis=1, keepdims=True) + 1e-8)
            cos = mem @ gc                                             # [mem_end]
            ps, ns = cos[pos], cos[neg]
            if head is not None:
                pcos = _proj(mem_raw, "mem") @ _proj(gcr[None], "goal")[0]
                pps, pns = pcos[pos], pcos[neg]
                p_gate = float(1 / (1 + np.exp(-(gate_a * pcos[cand].max() + gate_b))))
            if grp != "covis_revisit":
                # no positives: how high can a DISTRACTOR score? (should be low)
                r = dict(distractor_max=float(cos[cand].max()),
                         distractor_mean=float(cos[cand].mean()))
                if head is not None:
                    r.update(p_distractor_max=float(pcos[cand].max()),
                             p_distractor_mean=float(pcos[cand].mean()),
                             p_gate=p_gate)
                rows.append(r)
            else:
                r = dict(
                    auc=_auc(ps, ns),
                    retr1=float(ps.max() > ns.max()),
                    margin=float(ps.max() - ns.max()),
                    pos_mean=float(ps.mean()), pos_max=float(ps.max()),
                    neg_mean=float(ns.mean()), neg_max=float(ns.max()),
                    n_pos=int(pos.size), n_neg=int(neg.size),
                )
                if head is not None:
                    r.update(
                        p_auc=_auc(pps, pns),
                        p_retr1=float(pps.max() > pns.max()),
                        p_margin=float(pps.max() - pns.max()),
                        p_pos_mean=float(pps.mean()), p_pos_max=float(pps.max()),
                        p_neg_mean=float(pns.mean()), p_neg_max=float(pns.max()),
                        p_gate=p_gate,
                    )
                rows.append(r)
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
        if head is not None:
            pauc = col(rows, "p_auc"); pr1 = col(rows, "p_retr1"); pmg = col(rows, "p_margin")
            ppm = col(rows, "p_pos_mean"); pnm = col(rows, "p_neg_mean"); pg = col(rows, "p_gate")
            print(f"  [proj] AUC mean={pauc.mean():.3f}  retr@1={pr1.mean():.3f}  "
                  f"margin mean={pmg.mean():+.3f}  cos gap={ppm.mean()-pnm.mean():+.3f}")
            print(f"  [proj] gate P(revisit): mean={pg.mean():.3f}  frac>0.5={np.mean(pg > 0.5):.2f}")

    for grp in ("covis_novel", "goalA_novel"):
        nov = results.get(grp, [])
        if not nov:
            continue
        dmax = col(nov, "distractor_max")
        print(f"\n[{grp}] N={len(nov)}  distractor_max: mean={dmax.mean():.3f} "
              f"median={np.median(dmax):.3f} p90={np.percentile(dmax,90):.3f}")
        rev = results.get("covis_revisit", [])
        if rev:
            posmax = col(rev, "pos_max")
            print(f"   compare covis_revisit pos_max mean={posmax.mean():.3f}  "
                  f"→ null is {'SEPARABLE' if posmax.mean() > dmax.mean() + 0.03 else 'CONFUSABLE'} "
                  f"(revisit pos vs novel distractor)")
    if head is not None:
        pg_nov = np.concatenate([col(results[g], "p_gate")
                                 for g in ("covis_novel", "goalA_novel") if results[g]])
        pg_rev = col(results["covis_revisit"], "p_gate")
        if pg_nov.size and pg_rev.size:
            print(f"\n[proj] gate: novel P(revisit) mean={pg_nov.mean():.3f} frac<0.5={np.mean(pg_nov <= 0.5):.2f}"
                  f" | revisit frac>0.5={np.mean(pg_rev > 0.5):.2f}"
                  f" | gate AUC={_auc(pg_rev, pg_nov):.3f}"
                  f" | balanced acc@0.5={0.5 * (np.mean(pg_rev > 0.5) + np.mean(pg_nov <= 0.5)):.3f}")

    print("\nRead: AUC≈0.5 / retr@1≈chance / gap≈0  →  frozen CLS carries NO match signal "
          "(collapse is representational; drop CLS-only retrieval). AUC≳0.8 / retr@1≳0.7 "
          "→ signal EXISTS; the training collapse is optimization (fix null shortcut / train longer).")

    if args.out:
        np.savez(args.out, **{f"{g}__{k}": col(r, k)
                              for g, r in results.items() for k in (r[0].keys() if r else [])})
        print(f"\nsaved per-sample scalars -> {args.out}")


if __name__ == "__main__":
    main()
