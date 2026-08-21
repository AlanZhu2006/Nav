# HM3D held-out val10 causal-Revisit external transfer

**Status:** frozen before any selected-scene navigation outcome is read.  The
two local GOAT-val scenes and every previous GOAT pilot/formal scene are treated
as consumed implementation evidence and excluded from this efficacy population.

## Question

Does the frozen Certified Episodic Compass (CEC) transfer beyond MP3D when the
history is an actual online native-NavDP rollout and the Revisit goal is a
different, co-visible view of a previously traversed place?

This is an external-transfer protocol on HM3D, not an official GOAT score and
not a newly claimed public benchmark.  It uses the existing two-leg causal
Revisit contract because the released GOAT repeated-instance subset did not
guarantee causal visual Revisit and yielded no executable CEC interventions.

## Population

- HM3D v0.2 `val` archive, containing 100 scene assets;
- union every scene appearing in the five prior GOAT pilot/smoke/formal
  manifests (36 unique scenes), subtract that union, sort the remaining asset
  directories by five-digit archive index, and take the first ten;
- selected directories: `00801`, `00804`, `00805`, `00806`, `00807`, `00809`,
  `00811`, `00812`, `00816`, and `00817`;
- zero overlap between these ten scenes and the audited 36-scene consumed union;
- four deterministically generated episodes per scene, 40 total;
- every episode is retained intention-to-treat, including native Goal-A
  failures;
- no MP3D scene, episode, or outcome is read by this evaluation.

The immutable machine-readable contract is
`hm3d_heldout_val10_revisit_protocol_20260816.json`.  The source hashes,
36-scene exclusion union, outcome-blind selection rule, and selected identities
are sealed in `hm3d_consumed_scene_audit_20260816.json`.

## Episode and observability contract

The frozen generator samples a 3--9 m start-to-A trajectory and a Revisit-B
goal view with max historical co-visibility in `[0.20, 1.00]`, heading offset
at most 45 degrees, and a 70% long-term sampling target.  Goal-B is not an
exact stored JPEG.

Frozen native NavDP runs Goal-A once.  Its actual RGB/depth/action/pose trace is
then replayed byte-identically to every Goal-B arm.  If Goal-A fails, Goal-B is
not executed in any arm.  Thus the conditional denominator and history are
shared, and no expert-only history can leak into the closed-loop result.

The generator executable is sealed inside this task bundle (rather than read
from the older CEC base bundle) because it is the exact version exercised by
the local HM3D smoke and records the frozen distance/covisibility provenance
fields required by the outcome-blind manifest gate.  This changes data
construction metadata only; evaluation servers, controller weights, and CEC
runtime remain the previously validated immutable base stack.

## Arms

1. `native`: frozen native NavDP ImageGoal control;
2. `raw_fixed_oracle_role`: DINO top-1 memory bearing projected to the same
   fixed 2.5 m controller radius; it is explicitly told that B is Revisit and
   is therefore an upper ablation, not deployable open-set control;
3. `geometry_router`: the older DINO + SIFT/RANSAC geometry route;
4. `certified_relocalization`: frozen DINO top-8, SuperPoint/LightGlue ranking,
   LingBot-depth PnP certificate, scale-free bearing, and exact native fallback.

No arm may change NavDP, LingBot, retrieval, geometry, PnP, certificate, fixed
radius, success distance, or action budget after seeing held-out-val outcomes.

## Reporting

Report both ITT joint success and Goal-B success conditional on the byte-shared
Goal-A success set.  Every contrast includes paired gains/losses, exact McNemar
`p`, and a scene-cluster bootstrap interval.  The two preregistered contrasts
are CEC vs native and CEC vs the oracle-role raw-fixed ablation.

The result may support a cross-domain transfer claim.  It must not be labeled
an official GOAT, MemoNav, or Habitat Challenge leaderboard score.
