# Final14 Prospective Learned-Relocalizer Amendment

**Frozen:** 2026-08-17, after all train40 model development and the consumed
positive/negative transport smokes, and before any final14 scene identity,
asset, render, policy trace or outcome was accessed.

**Parent protocol:**
`FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md`, SHA-256
`3d1ebc6ef429fd16df4d550eda52eceb55d7b15fd181a5c00c0b8f971f7aaa32`.

This amendment adds one prospective learned arm. It does not modify the
parent population, query construction, CEC arm, existing hypotheses,
controller, arrival rule, budget or reporting denominators. In particular,
the original CEC claims remain valid or invalid on their own frozen tests.

## 1. Why this amendment is admissible

The parent protocol forbids adding an arm **after final14 opens**. Final14 has
not opened. The learned method, its thresholds and its deployment contract
were selected entirely from train40; only consumed scenes were used for
implementation and fallback transport tests. No learned result can be used to
alter final14 membership or query construction.

Adding the arm now is prospective. Adding it after seeing any final14
identity, constructibility or policy result would invalidate this amendment.

## 2. Frozen learned arm

Canonical arm name: `learned_pi3x_spatial`.

```text
actual-online causal RGB history + ImageGoal
    -> frozen DINO temporally diverse top-8
         minimum anchor 8, temporal gap 4
    -> frozen Pi3X b16 causal bridge
         live current + 16 causal bridge views
         + anchor offsets [-8,0,+8] + ImageGoal
    -> choose one proposal by Pi3X cross-view overlap
    -> four scene-crossfit spatial proof heads
    -> at least 2/4 model-bound threshold votes
         accept: unit [forward,left] bearing
                 -> fixed 2.5 m mixed ImageGoal/PointGoal residual
                 -> frozen NavDP
         reject/error: exact native ImageGoal NavDP
```

Lifecycle is also frozen:

- the DINO top-8 is chosen only on the first causal query;
- a first-query rejection is sticky for that goal;
- a first-query acceptance fixes the anchor and DINO rank;
- later replans infer only that anchor to update the current-relative bearing;
- no role label enters retrieval, proof, controller or fallback;
- no SuperPoint, LightGlue, LingBot depth, PnP, atomic certificate feature,
  simulator pose or simulator depth is consumed by this arm.

Frozen identities:

- official Pi3X weight SHA-256:
  `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a`;
- four-head deployment-manifest SHA-256:
  `1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd`;
- consumed closed-loop source receipt:
  `733aee67bd8c57e421dc06b645b5fe686cfc4eb970ab9971a8b2291547daf82a`;
- agreement threshold: `2/4` votes;
- controller residual: `2.5 m`;
- proposal top-k / temporal gap / bridge frames: `8 / 4 / 16`.

No checkpoint, threshold, consensus, bridge density, proposal ordering or
fallback behavior may change after this amendment is sealed.

## 3. Amended arm set and pairing

The parent four arms remain unchanged:

1. `native`;
2. `raw_fixed`;
3. `geometry_fixed`;
4. `certified`.

The learned arm is added as arm 5. All five arms must share the same Goal-A
trace SHA, query RGB, ImageGoal RGB, NavDP checkpoint, query budget,
deterministic diffusion seeds and evaluator success criterion. Arm order is a
deterministic five-way rotation fixed before construction. Runtime role and
support-band labels remain evaluator-only.

## 4. Prospective learned hypotheses

### L1: useful Revisit control

On standard Revisit, compare `learned_pi3x_spatial` with `native` using the
parent paired statistics. Qualification requires positive net gain and a
scene-cluster 95% CI lower bound above zero.

### L2: non-inferiority to CEC

For standard-Revisit SR define
`Delta = SR_learned_pi3x_spatial - SR_certified`. Before observing final14,
the non-inferiority margin is frozen at `-10` absolute percentage points.
Qualification requires:

- the scene-cluster 95% CI lower bound for `Delta` above `-10 pp`; and
- the paired point estimate at least `-5 pp`.

The `-10 pp` bound is a qualification margin, not an equality claim. The
stricter point-estimate rule prevents a statistically vague small pool from
promoting a visibly weaker replacement. Report paired gains/losses and exact
McNemar regardless of pass/fail.

### L3: unsupported-Novel interference and exact fallback

Report learned proof accept/takeover rates and learned versus native paired
gains/losses on natural Novel. Qualification requires:

- zero runtime-contract violations;
- every abstaining plan to match native requested/returned diffusion seeds,
  selected-trajectory SHA and executed pose/RGB trace exactly;
- no net paired Novel loss relative to native;
- zero accepted bearing errors above `90 deg`.

Any learned takeover on Novel remains in the denominator and is reported by
scene. It cannot be removed as a construction or smoke failure.

### L4: support--coverage and efficiency

Standard and hard-support Revisit remain separate. Report learned versus CEC
accept coverage, successful takeovers, rejected successes, false takeovers,
first top-8 latency, one-anchor update latency, peak GPU memory and stored
history. These are secondary and cannot rescue a failed L1--L3 gate.

## 5. Interpretation rules

- If L1, L2 and L3 all pass, the learned relocalizer may replace the
  hand-engineered certificate as the paper's primary deployment method; CEC
  becomes the training-free geometric baseline and attribution teacher.
- If L1 and L3 pass but L2 fails, CEC remains primary and the learned arm is a
  promising but weaker simplification.
- If L3 fails, the learned arm is not deployment-qualified regardless of
  Revisit SR.
- If target population power is missed, report the full constructible set as
  underpowered; do not expand, tune or reopen final14.

## 6. Pre-unseal implementation gate

This document does **not** authorize immediate final14 execution. Before any
final14 access:

1. extend the formal evaluator, summary and independent verifier to all five
   arms without changing parent query construction;
2. run one full five-arm end-to-end dry-run on consumed scenes;
3. verify accepted-anchor lifecycle, sticky rejection and exact native
   fallback from raw plans and rollout traces;
4. freeze one immutable source/SBATCH bundle and record its SHA-256 here or in
   a hash-bound execution receipt.

Only after all four checks pass may the single prospective final14 pipeline be
submitted. No final14 record may be treated as a disposable smoke.
