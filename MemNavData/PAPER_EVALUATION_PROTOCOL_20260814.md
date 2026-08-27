# Certified Episodic Compass — paper evaluation protocol (frozen design)

Date: 2026-08-14 (Asia/Shanghai)

## 0. Construction amendment made before any query rollout

The first HPC construction attempt exposed that the scene wrapper had reused a
double-Revisit/three-leg source builder for this independent-query benchmark.
That builder unnecessarily required two historical goals, temporal separation
between them, and a B-to-C geodesic transition even though each evaluation unit
contains exactly one independent Revisit query after online A.  It also
preselected anchors before frame 39 although the deployed LingBot memory cannot
emit anchors before `S + W - 1 = 8 + 32 - 1 = 39`.

No Novel/Revisit query arm was executed before this was found.  Construction
was therefore corrected, without reading a query outcome and without changing
any method, arm, certificate threshold, controller, metric or success rule:

- eligible source frames start at frame 39, use stride 8, and retain a 16-frame
  end margin; history admission requires one such frame, not two anchors;
- one controlled V1 Revisit goal is sought at 2--9 m from the online-A
  endpoint; candidates are ordered by distance to the frozen 3 m target and
  then frame index, with at most four deterministic source candidates;
- the controlled-pose visual contract remains unchanged: translation
  0.20--0.50 m, yaw change 10--25 degrees, source co-visibility at least 0.45,
  maximum online-history co-visibility in [0.50, 0.98], argmax gap at most 20
  frames, and pixel MAE at least 5;
- Novel matching and both query protocols remain outcome-blind.

The exact incident record and validation boundary are frozen in
`PAPER_CONSTRUCTION_AMENDMENT_20260814.md`.  Consequently, the final query
evaluation remains one-shot, but the native-A success/length and construction
attrition of the first attempt are no longer blind and must be disclosed.

## 1. Claims and why two query protocols are necessary

The method claim is not merely “memory raises average SR.”  It is:

> A causal visual memory may supply a scale-free residual bearing to a frozen
> ImageGoal policy only when that historical pose hypothesis is geometrically
> self-certified; unsupported goals leave the base policy exactly unchanged.

One mixed number cannot test both halves of this claim.  We therefore freeze
two independent-query protocols after the same genuine native online-A trace:

1. **Support-controlled diagnostic.** Novel and Revisit queries are matched in
   geodesic distance and shortest-path initial bearing.  This isolates whether
   the system distinguishes historical visual support, but the paired bearing
   can make an incorrect always-on memory direction accidentally useful for a
   Novel query.  It is primary for certificate risk/coverage and exact fallback,
   not for overall navigation superiority.
2. **Natural-direction task benchmark.** Novel is sampled independently subject
   to the same distance and support constraints, with no pairwise bearing match
   (`180°` acceptance).  Revisit remains a controlled nearby view of a true
   online-A anchor.  This is primary for SR/SPL and paired policy utility.

Both queries run after fresh reset and exact replay of the same online-A trace.
They are never chained, so Novel failure cannot censor the Revisit denominator
and one query cannot contaminate the other's memory.

## 2. Frozen method and arms

All arms use the same frozen NavDP checkpoint, execution horizon 8, 600-step
budget, deterministic plan seeds, 1.0 m success radius, and no terminal U-turn,
visual refinement, graph rescue, CDEC rescue, X-NavDP, frontier or oracle.

1. `native`: frozen ImageGoal NavDP, no episodic residual.
2. `raw_metric`: always-on raw-DINO top-1 metric residual.  This preserves the
   strongest simple historical baseline but is a scale-confounded ablation.
3. `raw_fixed`: the same raw top-1 direction normalized to fixed 2.5 m.
4. `geometry_fixed`: the old DINO-floor + SIFT/essential-RANSAC router, but its
   accepted vector is normalized to the same fixed 2.5 m controller input.
5. `certified`: temporally diverse DINO top-8, SuperPoint/LightGlue fundamental
   ranking, LingBot depth + PnP, frozen atomic certificate, scale-free bearing,
   and fixed 2.5 m residual; rejection calls native exactly.

The primary method contrasts are `certified-native`, `certified-raw_fixed`, and
`certified-geometry_fixed`.  `raw_metric` isolates the effect of uncalibrated
distance.  Existing known-role-direct results are an oracle-role upper bound,
not a deployable paper arm.

Certificate thresholds remain: PnP inliers >=16, query and reference inlier
hull coverage >=5%, reprojection RMSE <=2 px.  No external-dataset outcome may
change these values.

## 3. MP3D one-shot confirmation

- Scene source: frozen `strict_graph_blind_20260806.json`, 16 scene clusters,
  two deterministic source episodes per scene.
- Before reading a query outcome, frozen native NavDP runs Goal-A and writes a
  complete causal RGB/depth/pose/hash trace.
- Every native-A success is attempted by the builder.  Constructibility uses
  only navmesh geometry, rendering and co-visibility; all failures and seeds
  are retained.  No episode is selected by downstream policy success.
- Target population: at least 20 constructible histories spanning at least 12
  scene clusters.  If the frozen pool cannot meet this target, report the full
  attrition and treat the result as underpowered; do not replace scenes.
- The 16-scene episode content remains sealed until the consumed four-scene
  readiness gate and the natural-direction integration gate both pass.
- After this explicitly recorded pre-query construction amendment, no method,
  threshold, arm, construction or exclusion change is permitted in response to
  query outcomes.

## 4. Cross-dataset confirmation

### Replica v1 (immediate external-domain benchmark)

Use the official Replica v1 Habitat assets.  Scene admission is determined
before policy execution by a simulator-only gate: complete stage/mesh/navmesh,
0.30 m agent navmesh, usable 480x270 RGB-D, and enough same-floor geodesic
extent.  The full 18-scene archive is expanded before final selection; the
compatibility gate, not navigation performance, defines eligible scenes.

For each eligible scene, deterministically attempt four source histories.  The
target is at least 20 constructible histories over at least 8 scene clusters.
The same online-A, support-controlled and natural-direction contracts and the
same five arms are used without adapting weights or thresholds.  Replica is a
cross-dataset sequential-memory benchmark, not the standard single-goal
InstanceImageNav leaderboard; it must be named accordingly.

### HM3D (standard ImageNav domain, follow-up)

HM3D is the preferred standard external domain because Habitat's official
InstanceImageNav challenge uses HM3D.  Its assets require authorized Matterport
credentials and are not currently present on the cluster.  Do not bypass that
license.  When assets are available, reuse this frozen protocol; a standard
single-goal HM3D run alone cannot show memory gain because no prior history
exists, so report both native single-goal compatibility and the sequential
memory extension explicitly.

## 5. Metrics and statistical unit

Primary natural-direction metrics:

- role-stratified SR and SPL;
- certified-vs-each-baseline paired gains/losses and two-sided exact McNemar;
- paired risk difference with 100,000-resample scene-cluster bootstrap 95% CI;
- balanced mixed SR, with Novel and Revisit always shown separately.

Primary support-controlled metrics:

- Novel false-takeover episode and plan rates;
- Revisit activation recall and conditional SR;
- certificate risk-coverage curve at the single frozen operating point;
- exact native identity for every fully rejected Novel rollout.

For all protocols report final distance, path length, SPL, steps, certificate
rejection reasons, uncached/cached localization latency, planning latency,
memory size, source/constructibility attrition, N, paired +/- and scene count.
The query is the paired outcome unit; the scene is the uncertainty cluster.

## 6. Reporting boundaries

- The four consumed MP3D scenes are implementation/power evidence only.
- The support-controlled protocol must not be presented as unbiased natural SR.
- Replica results are cross-domain evidence, not an official public leaderboard.
- HM3D is not “run” until licensed assets and standard receipts exist.
- No result is promoted because its raw percentage looks large; every positive
  claim includes N, paired gains/losses, exact test and scene-cluster interval.
