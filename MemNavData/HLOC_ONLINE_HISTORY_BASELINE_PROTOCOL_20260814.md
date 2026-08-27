# HLoc online-history baseline protocol (frozen before paper outcomes)

Date: 2026-08-14 (Asia/Shanghai)

## Question

Can a standard multi-view localization stack replace LingBot depth plus the
pairwise atomic relocalization certificate when both receive exactly the same
causal online-A images?

This is a localization-backend comparison, not a separately trained navigation
agent.  All successful estimates are converted to the same fixed 2.5 m bearing
and executed by the same frozen NavDP controller; localization failure invokes
native ImageGoal exactly.

## Frozen inputs and causal boundary

- Official HLoc source commit:
  `c13273bd0ecc2917a35910fd843712a1c6243193`.
- Only RGB frames already present in the frozen online-A trace may enter the
  reference model.  Query rollouts, query depth, Habitat pose, geodesic path,
  `analysis_role`, co-visibility and future observations are forbidden.
- Use the online-A decision frames (the same every-eight-step frames replayed
  into NavDP/MemNav), not a result-selected subset.
- Construct one independent reference model per online history.  Match temporal
  neighbours and retrieval-proposed non-local pairs; never join scenes or
  episodes.

## Localization and controller contract

1. SuperPoint local features and LightGlue matches build a monocular SfM model
   of the decision-frame history through HLoc/pycolmap.
2. The final online-A camera is the reference origin.  If that decision frame
   is not registered in the largest SfM component, first localize it against
   that component with the identical frozen query-localization thresholds.  A
   failed endpoint localization rejects the HLoc proposal and invokes native.
   Localize the goal image against the same model and express its translation
   direction in the localized final-camera frame.
3. Metric scale is discarded.  A finite non-zero planar direction is normalized
   and projected to the frozen 2.5 m PointGoal radius.
4. Accept only if the localized query has at least 16 PnP inliers, query and
   reference inlier hull coverage are each at least 5%, and reprojection RMSE is
   at most 2 px.  These are the paper method's already-frozen atomic thresholds,
   not HLoc-outcome-tuned values.
5. Any failed reconstruction, localization, certificate or direction conversion
   returns the byte-identical native ImageGoal action path under deterministic
   seeds.

## Evaluation order

1. Dependency/import and one consumed-history reconstruction smoke.
2. Consumed Novel/Revisit localization audit: reconstruction rate, role-wise
   acceptance, bearing error (analysis only), false acceptance and latency.
3. If the implementation contract passes without changing thresholds, run it
   as a secondary arm on the already-frozen paper population.  Do not alter the
   primary five-arm design or use paper outcomes to tune HLoc.

No additional navigation dataset is required: the baseline consumes the same
online histories and goal images as the primary evaluation.

## Pre-paper implementation status

The dependency/import preflight and one causal online-history reconstruction
smoke passed on 2026-08-14.  From 30 every-eight-step decision frames, HLoc
registered 19 images and reconstructed 722 3D points in 29.30 seconds on CPU;
the mean reconstruction reprojection error was 1.014 px.  The receipt confirms
that no pose, depth, role label or query image was read.  This is only a
representation/readiness result: it is not a query-localization or closed-loop
navigation claim.  The final decision frame was not part of the largest
registered component, so the frozen endpoint-localization/abstention rule must
be exercised before this baseline can be promoted.

Receipt:
`.diagnostics/hloc_online_history_smoke_cpu_v2_20260814/receipt.json`
(SHA-256
`6f23cc67106151bf85593bc86597c31a8ad07c53c26f8f3e44871df5261a34cd`).
