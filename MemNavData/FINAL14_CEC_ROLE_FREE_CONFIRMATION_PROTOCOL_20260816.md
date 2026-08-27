# Final14 Role-Free CEC Confirmation Protocol

**Frozen:** 2026-08-16, after the consumed Novel forced-anchor gate failed and
before any final14 scene identity, construction artifact, policy trace or
outcome is opened for this experiment.

**Population ledger:**
`.diagnostics/mp3d_scene_budget_20260816/scene_budget.json`

**Primary method:** frozen geometry-first Certified Episodic Compass (CEC)

## 1. Scientific question

This experiment does not ask whether episodic history can solve unsupported
Novel navigation.  That branch failed its consumed first-proposal promotion
gate and is closed.

The final experiment asks one factorized question:

> In a continual ImageGoal stream with no runtime role label, can a causal
> online visual history provide useful Revisit direction while an empirical
> geometric witness prevents interference on unsupported Novel goals?

The confirmation therefore targets three inseparable properties:

1. **Revisit utility:** CEC must improve frozen NavDP on historically supported
   goals.
2. **Novel interference control:** unsupported goals must retain native
   behavior when no hypothesis is certified.
3. **Minimal interface:** accepted history contributes only a scale-free
   bearing with a fixed `2.5 m` residual; no metric waypoint, oracle role,
   global map or controller replacement crosses the boundary.

## 2. Scene seal and source population

- Use all 14 scenes labeled `untouched_final14` in the immutable scene budget.
- No scene may be replaced after collection or query outcomes are observed.
- For each scene, attempt the lexicographically first eight available frozen
  source episodes.  If fewer than eight exist, attempt all and record the asset
  shortage; do not borrow another scene.
- Run native NavDP Goal-A once per source with the frozen checkpoint, seed
  derivation and online trace contract used by Attempt 7/Phase-2.
- Attempt query construction for every successful Goal-A trace.
- Retain at most the first three constructible histories per scene in source
  episode order.  The cap prevents scenes with more successful sources from
  dominating; it is applied before any query policy runs.
- Every failure at asset, Goal-A, geometry, rendering and query-construction
  stages remains in the attrition ledger.

No navigation outcome after Goal A may influence population membership.

Target power, not an adaptive admission rule:

- natural Novel + standard Revisit: at least 28 histories over at least 10
  scene clusters;
- hard-support Revisit: at least 16 queries over at least 8 scene clusters.

If a target is missed, report the entire constructible population as
underpowered.  Do not add episodes, loosen constraints or alter a method.

## 3. Independent query construction

Every query starts after a fresh reset and byte-verified replay of the same
actual-online causal Goal-A prefix.  Queries are independent, not chained, so
one failure cannot censor another denominator or contaminate memory.

The construction role and all co-visibility diagnostics remain evaluator-side
only.  Runtime receives only current RGB, ImageGoal RGB and causal history.

### 3.1 Standard Revisit

A controlled re-render near an eligible online-A pose:

- source frame at least 39 and at least 16 frames before the A endpoint;
- query geodesic from A endpoint in `[2,9] m`, target `3 m`;
- translation from source in `[0.20,0.80] m`;
- absolute yaw change in `[12,45] deg`;
- pixel MAE from source at least 5;
- maximum eligible online-A co-visibility in `[0.55,0.90]`;
- co-visibility argmax within 24 frames of the source anchor.

Candidates are ordered without method scores by:

1. absolute difference from target max-covis `0.72`;
2. absolute difference from query geodesic `3 m`;
3. larger source temporal gap from the endpoint;
4. source frame and deterministic render-attempt index.

### 3.2 Hard-support Revisit

Constructed independently from the same history when possible:

- the same causal frame, distance, pixel-MAE and navigability contracts;
- translation in `[0.30,1.00] m`;
- absolute yaw change in `[18,60] deg`;
- maximum eligible online-A co-visibility in `[0.25,0.55)`;
- co-visibility argmax within 32 frames of the source anchor.

Candidates are ordered by distance to target max-covis `0.40`, then by the
same outcome-blind tie breakers as standard Revisit.  Failure to construct the
hard query does not remove the history from the standard/Novel population; it
only enters hard-band attrition.

The hard band tests risk--coverage and is not allowed to choose a new
certificate threshold.

### 3.3 Natural unsupported Novel

- maximum eligible online-A co-visibility strictly below `0.10`;
- query geodesic from A endpoint in `[2,9] m`;
- minimum clearance `0.30 m` and same-floor tolerance `0.20 m`;
- minimum `1 m` separation from a paired Revisit target;
- goal yaw selected by an identity-bound hash from eight world-yaw bins and
  therefore independent of endpoint-to-goal bearing;
- initial shortest-path direction assigned in a frozen cyclic stratum:
  front `[-60,60] deg`, side `(60,120]` in absolute angle, or rear
  `(120,180]`; candidate sampling must satisfy the assigned stratum.

The stratum cycles by immutable `(scene-rank, source-episode-rank)` before any
query policy runs.  This removes the U-turn-heavy and goal-yaw coupling found
in Phase-2.

## 4. Frozen arms

All arms share Goal-A replay hashes, current/goal images, NavDP checkpoint,
deterministic diffusion seeds, execution horizon 8, 600-step query budget,
`1.0 m` success radius, and no terminal U-turn, visual refinement, graph
rescue, CDEC, X-NavDP, frontier or oracle.

1. `native`: frozen ImageGoal NavDP; no episodic sidecar intervention.
2. `raw_fixed`: always-on raw-DINO top-1 direction normalized to `2.5 m`;
   strongest simple high-coverage memory baseline.
3. `geometry_fixed`: previous DINO-floor plus SIFT/essential-RANSAC router,
   normalized to `2.5 m`; historical ablation.
4. `certified`: DINO temporally diverse top-8 proposal, SuperPoint/LightGlue
   geometric ordering, LingBot depth + PnP, atomic certificate, scale-free
   bearing, fixed `2.5 m` residual, and exact native fallback on rejection.

CEC thresholds stay frozen: at least 16 PnP inliers, query/reference inlier
hull coverage at least 5% each, and reprojection RMSE at most 2 px.

Arm order rotates deterministically by history and query identity.  The
runtime never receives Novel/Revisit/support-band labels.

## 5. Endpoints and hypotheses

### Primary H1: Revisit utility over native

On standard Revisit, compare CEC with native using paired `+/-`, exact
two-sided McNemar, paired risk difference and a 100,000-resample scene-cluster
bootstrap CI.  Promotion requires positive net gain and a cluster-CI lower
bound above zero.

### Primary H2: Novel interference control

On natural Novel, report:

- certificate accept and memory-takeover episode/plan rates;
- CEC versus native paired gains/losses;
- exact action/pose-trace identity for every query with no accepted
  certificate;
- false-takeover scene-cluster interval.

The strongest fail-closed result is zero takeover, zero paired loss and exact
trajectory identity.  Any exception is retained and weakens the claim; no
post-hoc threshold change is permitted.

### Primary H3: risk--coverage versus raw fixed

Do not select a favorable aggregate across roles.  Report separately:

- standard/hard Revisit SR and takeover coverage;
- Novel takeover, gains and losses;
- the two-dimensional point `(Revisit utility, Novel interference)` for raw,
  old geometry and CEC.

CEC need not exceed raw-fixed Revisit SR to validate authorization.  Its value
relative to raw is supported only if it materially reduces Novel interference
without erasing the Revisit gain over native.

### Secondary: support and runtime

- Repeat Revisit comparisons in standard and hard support bands.
- Report certificate rejection reasons and accepted-support distribution.
- Report DINO, matching, depth/PnP, uncached/cached and end-to-end latency;
  peak GPU memory; stored history bytes/frames; and intervention frequency.
- SPL, final distance, path length and steps are secondary policy metrics.

## 6. Statistical and reporting contract

- Query is the paired outcome unit; scene is the uncertainty cluster.
- Report `N`, scene count, paired `+/-`, exact McNemar and cluster CI together.
- Novel and Revisit are always shown before any role-balanced macro average.
- Standard and hard Revisit use distinct denominators and are never silently
  pooled.
- No Attempt 7, Phase-2, Fresh160 or train40 outcome is pooled into final14.
- Same-process paired effects take precedence over cross-run native rates.
- Construction, runtime failures and missing hard-band queries remain in the
  ledger.

## 7. Stop and integrity rules

- Unit tests and consumed-scene smoke tests may repair implementation, but may
  not change this construction, arm, threshold or endpoint contract.
- No final14 policy outcome may be discarded as smoke.  A failed immutable
  attempt is documented and repaired by provenance, with every valid record
  retained in the denominator.
- Do not tune from per-scene final14 results.
- Do not run a learned rescue, Novel direction adapter or extra arm after
  final14 opens.
- After independent verification, final14 closes the MP3D empirical method
  selection loop.  A negative result changes the paper claim; it does not
  trigger another internal benchmark.

## 8. Claim boundary after completion

If H1 and H2 pass, the allowed headline is:

> Causal episodic visual history can improve supported Revisit navigation for
> a frozen ImageGoal policy through a self-verified, scale-free bearing
> interface, while unsupported Novel queries empirically fall back to the
> native policy without role labels.

The experiment can never establish formal safety, deployable Novel direction,
metric localization accuracy, or superiority of each component in isolation.

