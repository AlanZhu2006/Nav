# Unified Learned Relocalizer Night Goal — 2026-08-17

> **Chronology note (2026-08-17 15:52 CST):** This file is the complete
> development ledger up to the pre-unseal stage.  Its later statements that
> Final14 was “not submitted yet” describe that historical moment and are
> intentionally preserved.  Attempt 4 has since been sealed and submitted;
> current Pi3X/LingBot roles, training/generalization audit, attempt history and
> live HPC state are consolidated in
> `STATUS_20260817_PI3X_LEARNED_RELOCALIZER_FINAL14.md`.

## Frozen objective

Gradually replace the internal implementation of Certified Episodic Compass
with one learned geometric relocalizer while preserving its external contract:

```text
causal history + ImageGoal
    -> supported relative pose / bearing, or abstain
    -> fixed 2.5 m PointGoal residual
    -> frozen NavDP
```

This work does not change the Novel controller, NavDP weights, DINO history
shortlist, arrival criterion, or benchmark population.  No consumed held-out
set may be used for model selection or confidence calibration.

## Replacement ladder

1. **Shadow P0:** keep DINO top-8 and run a learned pairwise relocalizer beside
   the unchanged certificate.  It cannot alter navigation actions.
2. **P1 matcher/pose replacement:** replace SuperPoint + LightGlue + LingBot
   query-depth + PnP with a pretrained geometric model.  MicKey is first;
   MASt3R is the accuracy ceiling and Reloc3r the latency baseline.
3. **P2 learned proof:** replace hand-set certificate thresholds with
   scene-OOF calibrated support plus temporal and inverse-pair consistency.
4. **P3 deployment:** emit the same scale-free bearing and exact native
   fallback.  Remove the old certificate only after fresh paired closed-loop
   non-inferiority is established.

## Why the first candidate is MicKey

MicKey predicts 3D correspondences and metric relative pose from two RGB
images, exposes robust-solver support, includes a null hypothesis, and can be
fine-tuned using relative poses without depth supervision.  It can therefore
replace more of the present engineering stack than a learned feature matcher
alone.  Its MapFree pretraining domain is not assumed to transfer; that is the
first empirical question.

Frozen official source:

- repository: `nianticlabs/mickey`
- commit: `2391be8a35491e7b43481c069f5dab65030839b9`
- license: non-commercial research use

## Data and evaluation boundary

Initial shadow universe:

- train40 only;
- 480 sessions, 3,840 frozen DINO top-8 pairs;
- 3,203 unique RGB images in the existing immutable certificate bundle;
- no development, blind, Attempt-7, Fresh160 or external held-out read for
  tuning.

The unit of independence remains scene.  Image pairs from one trajectory are
not counted as independent evidence.

## Required outputs per pair

- relative rotation and metric translation with an explicit convention;
- solver support / inlier-like evidence;
- model support confidence;
- valid/abstain/error status and reason;
- latency and immutable model/checkpoint identity.

The common interface is implemented in
`MemNavData/learned_relocalizer_contract.py`.  It composes metric relative pose
with causal history pose only after applying the already audited external
meters-per-history-unit scale and emits `[forward, left]` without metric range.

## Promotion gates

Offline metrics are only gates to a fresh closed-loop test, never an SR claim.
The learned candidate must satisfy all of the following under scene-grouped
OOF calibration:

- accepted-pose/action precision non-inferior to certificate `93.13%`, margin
  at most `-3 pp`;
- actionable recall non-inferior to `79.74%`, and target at least `122/153`
  certified-actionable sessions;
- strict no-match/Novel false-positive rate no worse than `2.75%`;
- report bearing CDF at `15/30/45/90` degrees and zero unreported catastrophic
  (`>90°`) accepted errors;
- materially improve the present approximately `5–6 s` median uncached
  relocalization latency.

Only a candidate passing these gates may enter a fresh paired comparison:

```text
native vs current certificate vs learned relocalizer
```

The primary closed-loop hypothesis is non-inferiority to current Revisit SR,
not an unsupported claim that another paper's localization accuracy equals
our navigation SR.

## Tonight's stop/go decisions

1. Audit official MicKey dependency and checkpoint identity.
2. Pass official toy inference, then two PT1 positive and two strict-negative
   image-pair smoke tests.
3. Verify transform direction and intrinsics before any bulk inference.
4. If valid, run the immutable train40 shadow bake-off.  If zero-shot transfer
   is poor, retain its pretrained backbone and design adapter/LoRA fine-tuning;
   do not launch a blind long training job.
5. Do not change the production certificate or submit closed-loop evaluation
   tonight unless the offline non-inferiority gate is actually met.

## Results obtained tonight

### Zero-shot MicKey: fast pose proposal, failed open-set authorization

The official MicKey checkpoint was frozen and audited, then run label-blind on
all 3,840 train40 DINO-shortlisted pairs.  It returned a numerically valid pose
for every pair, including every strict no-match pair.  The result was:

- candidate ROC-AUC `0.7618`, AP `0.5627`;
- positive-session top-1 `112/155`, below LightGlue geometry `126/155` and
  raw DINO `115/155`;
- scene-grouped OOF authorization accepted only `8` sessions, of which `5`
  were correct, while retaining only `5/155 = 3.2%` positive-session recall;
- warm pair latency was approximately `36 ms`.

Therefore zero-shot MicKey is not promoted.  It demonstrates that a learned
pose backbone can be fast, but also pinpoints the missing capability: calibrated
open-set support/abstention, not merely another pose matrix.

### Pi3X deployment audit and causal visual bridge

The first Pi3X smoke aligned its predicted history camera centres to offline
trajectory poses by Sim(3) before computing bearing.  That produced an
apparently strong true-pair median error of `2.35 deg`, but the alignment is not
available to a monocular relocalizer by itself.  It is retained only as an
upper-bound diagnostic and is **not** a deployable result.

A corrected forward pass added the live current RGB and computed current-to-goal
bearing entirely inside Pi3X's predicted gauge.  With only five local frames
around an old anchor, it failed: true-pair median bearing error was `88.17 deg`.
The cause is structural.  A current view and a distant retrieved anchor may
have no visual overlap, so a joint model cannot connect their coordinate frames.

The fix is a **causal visual bridge**, not external pose injection: after DINO
finds an anchor, uniformly sample overlapping RGB frames along the already
observed trajectory from current back to that anchor, add three local support
frames around the anchor, and jointly infer current + bridge + goal in one
Pi3X pass.  No simulator pose, role label, co-visibility label or certificate
feature enters inference or the support score.

The local machine contains complete trajectories for only two PT1 scenes, so
the following `192 pairs / 24 sessions / 2 scenes` result is mechanism evidence,
not a statistical result:

- true-pair direct bearing median error: `5.54 deg`;
- `33/42` true pairs were within `30 deg`; the 90th percentile remained
  `80.84 deg`, so calibrated abstention is still necessary;
- joint point-cloud support achieved candidate AUC `0.791` on this subset;
- a diagnostic high-support level accepted `27` known-label pairs: `25/27`
  bearings were within `30 deg` and none exceeded `90 deg`;
- session selection picked the teacher-positive anchor in only `4/8` positive
  sessions, but its selected bearing was within `30 deg` in `7/8`.

The last contrast is important: co-visibility anchor classification is not the
same target as navigation-direction correctness.  It offers a concrete reason
why earlier learned rankers improved AUC without improving closed loop.  The
train40 expansion therefore reports both support labels and the actual frozen
`30 deg` bearing-utility target.  Nothing here yet justifies replacing the
current certificate because there are only two independent scene clusters.

## Frozen architecture after the mechanism audit

```text
causal RGB history
    -> frozen DINO temporal top-8 address shortlist
    -> for each shortlisted anchor:
         current RGB
         + causal RGB bridge from current to anchor
         + local anchor support views
         + ImageGoal
         -> one Pi3X-style multi-view transformer
         -> goal pose / scale-free bearing
         -> learned matchability and multi-view support
    -> listwise candidate selection + calibrated abstention
    -> accepted: fixed 2.5 m PointGoal residual -> frozen NavDP
       rejected: exact native NavDP
```

“Unified” refers initially to the complete post-retrieval relocalizer.  DINO
remains the long-history address book because the candidate-free GCT experiment
already showed that a transformer cannot reliably search hundreds of frames by
itself (`5/20` versus `18/20` with DINO addressing).  Reopening that falsified
question would violate tonight's goal.

During migration the order is:

1. high-support Pi3X prediction;
2. otherwise unchanged current certificate;
3. otherwise exact native fallback.

This cascade is temporary experimental scaffolding, not the intended paper
method.  The final method removes SuperPoint, LightGlue, LingBot depth, PnP and
the hand-set atomic certificate once a scene-grouped learned support head and a
fresh closed-loop non-inferiority test pass.  The useful *principle* of the
certificate—do not act without evidence—is internalized as learned calibrated
matchability rather than discarded.

## Train40 expansion: completed and independently reproduced

The reproducible runner is
`MemNavData/diag_pi3x_multiview_consistency.py`.  Every full run used only the
frozen train40 CSV and the manifest-bound PT1 overlay:

- official Pi3 source commit
  `d07ddaf46a222acfda6bd877f72fdd099470cae8`;
- official Pi3X weight SHA-256
  `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a`;
- frozen train40 CSV SHA-256
  `85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127`;
- 480 sessions / 3,840 shortlisted pairs / 40 scene clusters;
- nested scene-grouped OOF calibration; no development, blind, Fresh160 or
  Attempt-7 sample was read for fitting, threshold selection or architecture
  choice.

The original bridge-density setting, `b8`, completed independently on the
shared RTX 5090 and HPC.  Aggregate agreement was close despite normal GPU
numeric non-determinism:

| run | navigation AUC | learned top-1 | raw-DINO navigation top-1 | oracle candidate ceiling |
|---|---:|---:|---:|---:|
| RTX 5090 | 0.89418 | 131/155 | 130/155 | 149/155 |
| HPC | 0.89490 | 133/155 | 131/155 | 148/155 |

This is a reproducible learned geometric signal, but neither run established a
safe replacement for the certificate.

### Data-integrity incident and scope of invalidation

The local workstation's apparent PT1 tree was stale and did not match the
frozen remote overlay.  Therefore all earlier row-level results computed from
that local tree are explicitly invalidated.  The immutable remote overlay was
re-audited against the manifest and had `0/3,203` image hash mismatches.  All
results below use that remote overlay.  This incident does not invalidate the
5090/HPC full runs, which were both manifest-bound.

## The causal bridge was the first real learned gain

Increasing the causal bridge from 8 to 16 sampled history frames (`b8 -> b16`)
was tested on the same machine, rows and model weights.  It was not a generic
hyperparameter sweep: the intervention specifically tested the diagnosed
failure that distant current and anchor views had no connected visual path.

| paired quantity | b8 | b16 | paired change |
|---|---:|---:|---:|
| positive pairs with bearing error <=30 deg | 585/701 | 659/701 | +82/-8, exact p=1.38e-16 |
| raw-DINO session top-1 | 130/155 | 144/155 | +16/-2, p=0.001312 |
| candidate-set oracle ceiling | 149/155 | 153/155 | +4/-0, p=0.125 |

The gains were concentrated at history gaps greater than 96 frames.  This is
the central structural result of the learned line: long-range relocalization
requires a causally observed visual bridge, not only a current/old-frame pair.
Using raw Pi3X cross-view overlap to choose among the frozen DINO top-8 reached
`147/155` navigation-direction top-1.

A scalar learned reliability head over the b16 diagnostics achieved candidate
AUC `0.919696`, AP `0.870651` and top-1 `145/155`.  Its nested OOF activation
was `131/145` correct accepts (precision `90.34%`, positive recall `84.52%`,
strict-negative FPR `2.48%`), but contained three accepted catastrophic
bearings over 90 degrees and strong fold instability.  It therefore was not
promoted.

## Why global Pi3X tokens did not replace the certificate

For every pair, the five Pi3X register tokens from each of the 20 b16 views
were averaged and exported as a label-blind `20 x 2,048` descriptor.  The
frozen archive has shape `[3,840, 20, 2,048]` and SHA-256
`a9e3a87e11efcf979a35d000d4835d9116f33a6e3bbb75dfc69f0e644957b773`.
Three increasingly strict heads were evaluated rather than reporting the best
one post hoc.

| learned head | proposal top-1 | accepted precision | positive recall | strict-negative FPR | accepted >90 deg |
|---|---:|---:|---:|---:|---:|
| V1 candidatewise head | 145/155 | 88.52% | 34.84% | 2.48% | 0 |
| V2 bound model-threshold ensemble, frozen 3/4 | 143/155 | 90.80% | 50.97% | 2.13% | 0 |
| V3 top-8 set model with explicit REJECT | 142/155 | 83.12% | 82.58% | 5.67% | 7 |

V1 exposed a calibration bug in the experimental idea, not in the held-out
split: thresholds calibrated on pooled inner predictions did not remain bound
to the separately refit outer model.  V2 corrected this with a true cross-fit
ensemble, scale-invariant Borda ranking and one threshold bound to each model.
Its consensus curve still had no operating point satisfying safety and high
coverage simultaneously.  V3 made rejection intrinsic, but one unseen-scene
fold reached `22%` FPR.  A diagnostic post-hoc global threshold could reach
precision `90.24%`, FPR `1.42%` and zero catastrophes only by reducing recall
to `23.87%`; because it used all OOF labels, it is not a deployable threshold.

An audit also found that the earlier CDEC `set_student_oof_v1_e300` had already
tested the same abstract top-8-plus-NULL idea using patch-correlation summary
features.  Pi3X tokens improved ranking from roughly `112--118` to `142`, but
did not fix open-set safety.  The contribution is therefore not renamed old
CDEC; it is the discovery that averaged global tokens have discarded the
correspondence structure needed to prove a pose.

Under the fixed raw-Pi3X-overlap proposal (`147/155` top-1), the existing V2
votes did improve authorization:

| vote consensus | precision | positive recall | FPR | catastrophes |
|---|---:|---:|---:|---:|
| 1/4 | 88.89% | 77.42% | 4.61% | 2 |
| 2/4 | 90.91% | 51.61% | 2.48% | 0 |
| 3/4 | 95.71% | 43.23% | 1.06% | 0 |
| 4/4 | 95.00% | 12.26% | 0.35% | 0 |

This cleanly separates the two jobs: Pi3X overlap is already the strongest
proposal signal, while learned proof is the remaining bottleneck.  The `2/4`
setting was frozen before observing the spatial-head result because it was the
most permissive setting satisfying the predeclared precision, FPR and
catastrophe gates.

## Spatial-evidence learned proof: current final gate

A label-blind spatial archive was exported from the same frozen b16 forwards.
For each of 20 views it contains a `9 x 16` grid of:

- world points transformed into the live-current camera gauge;
- per-view local points and confidence;
- relative `3 x 4` camera pose;
- view role, causal age and validity mask.

Coordinates are normalized by the median positive current-view depth.  The
archive contains no reporting label, LightGlue/PnP statistic, certificate
threshold or role label.  Its shape audit covers all 3,840 rows, all values are
finite, and its SHA-256 is
`23bc0eb6357942248561a7509c3751ed4c8fe90e7f845b47ed3ad9a1bf306342`.

The export runner ended with `ABORT` only because it compared whole JSONL byte
hashes; the repeat file legitimately changed `elapsed_s`.  An independent
field-level comparison found zero differences in every scientific field and
exact row/role/age alignment.  The archive is valid and was not regenerated.

The current proof model has 311,426 trainable parameters:

```text
9-channel spatial grid per view
    -> small shared CNN
relative pose + global Pi3X token + temporal role/age
    -> view transformer
    -> actionability/support heads
    -> model-bound scene-crossfit threshold votes
```

Candidate selection is deliberately fixed to raw Pi3X overlap.  Thus this
experiment asks only whether learned spatial evidence can replace the atomic
certificate's authorization function.  It cannot improve its apparent score
by changing the candidate or bearing.  The primary `2/4` consensus was frozen
before running the model.

Immutable inputs and execution records:

- spatial export source receipt:
  `016574e131f221b8f0ce5f75633a05893a24e090c15339a951e433eeb43740b5`;
- spatial archive receipt:
  `f5fa7110f6022ce63f7e3736936c5b63b9f3bb5d47b4676eb1f149bf310fbc48`;
- learned-head source receipt:
  `749e46bbf930ab1db98fd3ea1bd21b2994f5032ad147a3b4b3c685b69eba4917`;
- runner SHA-256:
  `ffcb1a682a6d9f64ed45c8bb5c294c91ac26f2272722e1bf5b460ae6fd22e9b6`;
- shared-5090 result root:
  `/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/spatial_head_crossfit_ffcb1a682a6d9f64_20260816T215924Z`.

### Full scene-crossfit result

The five-outer-fold/four-inner-model run completed.  With the pre-frozen `2/4`
consensus and fixed Pi3X-overlap proposal:

| quantity | spatial learned proof |
|---|---:|
| proposal direction top-1 | 147/155 |
| correct positive accepts | 119/155 |
| positive recall | **76.77%** |
| accepted known sessions | 125 |
| precision | **95.20%** |
| strict-negative false accepts | 4/282 |
| strict-negative FPR | **1.42%** |
| accepted bearing median | 3.41 deg |
| accepted within 15 / 30 / 45 / 90 deg | 110 / 123 / 124 / 125 |
| accepted over 90 deg | **0** |

All five outer folds had zero accepted catastrophic bearings.  Fold precision
was `90.0--100%`; one fold had FPR `2.94%`, but the frozen aggregate FPR was
`1.42%`.  The reporting-only consensus ablation confirms that the frozen
choice mattered:

| consensus | precision | recall | FPR | catastrophes |
|---|---:|---:|---:|---:|
| 1/4 | 86.62% | 87.74% | 6.38% | 1 |
| **2/4 (primary)** | **95.20%** | **76.77%** | **1.42%** | **0** |
| 3/4 | 96.55% | 54.19% | 0.71% | 0 |
| 4/4 | 96.55% | 36.13% | 0.71% | 0 |

Compared with the earlier global-token proof under the same fixed Pi3X
proposal, spatial evidence raises correct-positive recall from `51.61%` to
`76.77%`, raises precision from `90.91%` to `95.20%`, lowers FPR from `2.48%`
to `1.42%`, and retains zero catastrophes.  This is the specific evidence that
correspondence geometry, rather than a larger generic MLP, was the missing
representation.

Remote result checksums were copied and verified locally:

- full summary SHA-256:
  `13395bfdfa23c6a14b96d503877e94bf2abfd6d99b507fa4e441e852c3c8fa0c`;
- full predictions SHA-256:
  `6a99b018d39c25f9b4ca003a8cd3d6cbabd72b97e6e625447bfb82206c890a3e`;
- local result directory:
  `.diagnostics/learned_relocalizer_20260817/pi3x_spatial_head_crossfit_5090_ffcb1a682a6d9f64`.

### Endpoint-aligned comparison with the incumbent certificate

The original certificate recall reference, `122/153 = 79.74%`, uses a
historical **metric position error <=0.75 m** label.  The learned method is
trained and deployed as a scale-free compass, whose correct endpoint is
**accepted bearing error <=30 deg**.  Silently comparing these recalls would be
invalid.  The independent reporting-only verifier
`MemNavData/compare_pi3x_spatial_proof_to_certificate.py` therefore re-counts
both frozen methods on the same 480 sessions using the same directional target:

| direction-aligned endpoint | old certificate | spatial learned proof |
|---|---:|---:|
| correct positive accepts / 155 | 107 | **119** |
| precision | **97.27%** | 95.20% |
| strict-negative false accepts / 282 | **2** | 4 |
| FPR | **0.71%** | 1.42% |
| accepted >90 deg | 1 | **0** |

Positive correct coverage changes by `+28/-16` in favor of learned proof,
exact McNemar `p=0.0961`; the gains and losses span 20 and 13 scene clusters.
Strict-negative false accepts change by `+3/-1`, `p=0.625`.  Thus the learned
method is directionally stronger and remains within the frozen safety gate,
but it is not statistically established as superior to the certificate.
Verifier output SHA-256:
`191951f963473937c78b00833b504784f0fe9e8d0917b25aba9b9adef9a38cc4`.

The strict original gate must still be recorded as **not literally passed**:
`119` does not reach the old `122` count, and the two counts use different
labels.  The endpoint-aligned audit does not retroactively rewrite that gate.
It supplies the correct reason to run, rather than claim, the decisive fresh
closed-loop non-inferiority experiment.

## Frozen deployment candidate

A four-member deployment ensemble has now been trained and frozen without
reading any new scene.  Each member was fitted on 30 train40 scenes and
calibrated on the other 10 scenes that that exact member never saw.  Every
threshold is stored inside and hash-bound to its checkpoint; takeover still
requires the pre-frozen `2/4` vote.

```text
causal RGB history + ImageGoal
    -> frozen DINO top-8 long-history address
    -> frozen Pi3X b16 causal-bridge inference
         -> overlap proposal + scale-free bearing + spatial point evidence
    -> 4 x learned spatial proof, model-bound thresholds
    -> >=2 votes: fixed 2.5 m bearing residual -> frozen NavDP
       <2 votes or any runtime error: exact native NavDP
```

This learned arm invokes none of SuperPoint, LightGlue, LingBot depth, PnP or
the hand-written atomic certificate.  Pi3X and DINO remain pretrained/frozen;
the learned task-specific component is four 311,426-parameter proof heads.

Artifacts:

- source receipt:
  `689c592debb19d7e0cced9583dcff0cfcd89d1f2135fa74f833e45bf2b19e24c`;
- runner SHA-256:
  `9a2b99aea2673b052b09036539b94042138ac402707daee726de8073ded9f624`;
- remote immutable result:
  `/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/spatial_deployment_9a2b99aea2673b05_20260816T221100Z`;
- local verified copy:
  `.diagnostics/learned_relocalizer_20260817/pi3x_spatial_deployment_5090_9a2b99aea2673b05`;
- deployment manifest SHA-256:
  `1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd`.

The production-facing loader is
`MemNavData/pi3x_spatial_proof_runtime.py`.  It verifies every checkpoint hash,
selects only by Pi3X overlap, applies member-bound thresholds, normalizes the
accepted `[forward,left]` bearing, and fails closed with no bearing on invalid
evidence.  A real eight-candidate artifact smoke loaded all four checkpoints
and correctly abstained on the first strict-negative session.

## Production integration and real closed-loop qualification

The deployment candidate is no longer only an offline artifact. It is wired
through the real MemNav policy/server and the two-leg Habitat evaluator:

- `MemNavData/pi3x_online_relocalizer.py`: real Pi3X-b16 causal-bridge
  inference and learned spatial proof;
- `NavDP/baselines/memnav/policy_agent.py`: DINO top-8 proposal lifecycle,
  sticky first rejection, fixed accepted anchor and one-anchor bearing update;
- `NavDP/baselines/memnav/memnav_server.py`: hash-checked runtime loading and
  `/learned_relocalize` endpoint;
- `MemNavData/revisit_bearing_adapter.py`: scale-free unit bearing to the
  frozen `2.5 m` mixed-controller residual;
- `MemNavData/eval_2leg_habitat.py`: label-blind learned route and exact native
  fallback;
- `MemNavData/verify_pi3x_learned_closed_loop_smoke.py`: independent receipt,
  lifecycle, selected-trajectory and rollout verification.

The online route invokes no SuperPoint, LightGlue, LingBot depth, PnP,
certificate feature or simulator pose/depth. A real-weight numerical replay
matched the frozen offline spatial archive with maximum overlap error `0.0`
and bearing discrepancy `1.21e-6 deg`. Standalone Pi3X peaked near `6.55 GB`;
the co-resident MemNav + LingBot + Pi3X process used approximately `9.65 GB`.

### Positive consumed takeover smoke

One consumed train episode, `1pXnuDYAj8r/episode_0000`, was used strictly as a
transport test. Goal A was collected once and byte-identically replayed. It is
not a fresh efficacy sample and is not pooled into any SR table.

| property | observed result |
|---|---:|
| shared Goal-A trace SHA-256 | `32d8be9bc1b073eeae2c7fb391462f6977ef725c5bd0c2c856d27d33b3e4e25f` |
| native Goal B | fail, 249 steps |
| learned Goal B | success, 40 steps |
| learned requests / accepts / runtime errors | 5 / 5 / 0 |
| memory takeover plans | 5 |
| first request | DINO top-8, anchor 58 at rank 4, 3/4 proof votes, 1182 ms |
| later requests | same anchor/rank, one Pi3X candidate, median about 149 ms |
| handwritten certificate requests | 0 |

Every accepted request emitted a unit-norm bearing and the fixed `2.5 m`
adapter. The selected anchor never changed. Independent verification passed;
the immutable run receipt SHA-256 is
`dda85f50ef6fd971df9e3b3ad11628ff4e9067c11ce527951eb24eafc25d28cb`.
Local result:
`.diagnostics/learned_relocalizer_20260817/pi3x_learned_closed_loop_5090_e92d570b_1pX_ep0`.

### Negative consumed fallback-equivalence smoke

The same simulator episode and Goal-A process were copied into an immutable
counterfactual, replacing only Goal-B RGB with the known cross-scene negative
`17DRP5sb8fy/episode_0001/goal_2.jpg`, SHA-256
`7af221f302aba0a327c13730a83acb4aa4c935905995ef040a29f378938996f2`.
Metadata and the evaluation target were intentionally unchanged. This makes
the run a safety/transport test, never a meaningful navigation task or SR
sample.

The first real Pi3X top-8 proof produced `0/4` votes and rejected. The next 30
requests reused the sticky abstention without rerunning Pi3X:

| property | observed result |
|---|---:|
| requests / accepts / runtime errors | 31 / 0 / 0 |
| adapter abstentions / takeovers | 31 / 0 |
| first learned proof time | 1095.7 ms |
| later cached-abstention median | about 0.086 ms |
| native / learned steps | 247 / 247 |
| selected NavDP trajectory hashes | 31/31 identical |
| rollout pose/yaw/RGB records | 247/247 identical |

The equality audit additionally matched every requested/returned diffusion
seed, server-selected candidate index and candidate count. Thus fallback was
action-path equivalent, not merely equal in final success. Independent
verification says `exact_native_fallback_verified=true`.

Immutable identities:

- source-bundle receipt:
  `733aee67bd8c57e421dc06b645b5fe686cfc4eb970ab9971a8b2291547daf82a`;
- counterfactual input manifest:
  `f43dafdebf5c19d253c2733caeb28ce48ff11ff3e05b5180e0363ef4ede5b167`;
- run receipt:
  `3107994648961ea0eb6219ee8d91209925e7a51c987c8a8a63681cb17e850c51`;
- independent verification:
  `71bde55e0a03bc1ce651f37ff9d71d884aebc6c81d2d31fffaa58e3a457f4be2`;
- local result:
  `.diagnostics/learned_relocalizer_20260817/pi3x_learned_fallback_5090_733aee67_1pX_ep0_17DRP_goal2`.

Four earlier source-bundle attempts failed before a scientific outcome: three
were missing deployment imports and one reached a single action before an old
server omitted the deterministic-seed receipt. They remain preserved as
interface/package failures and are not hidden or counted as method trials.
The fifth positive attempt and the later negative run used complete immutable
dependency closure. The RTX 5090 was returned to `0%` utilization after both
servers exited.

## Final decision and next experiment

Tonight establishes a **GO to prospective fresh scene-disjoint closed-loop
qualification**, not a GO to replace the published method or claim SR. The two
consumed runs establish that the same real deployment can both act and abstain
exactly as designed; they do not establish its success rate.

The untouched final14 pool remains sealed. Before opening it, the learned arm
is frozen in
`FINAL14_LEARNED_RELOCALIZER_PROSPECTIVE_AMENDMENT_20260817.md`. The parent CEC
hypotheses and population stay unchanged; the added comparison is:

```text
native frozen NavDP
vs current Certified Episodic Compass
vs learned Pi3X Spatial Relocalizer
```

The learned qualification tests paired Revisit utility over native,
non-inferiority to CEC, and Novel interference/fallback safety. The model,
thresholds, `b16`, top-8 proposal, `2.5 m` residual and `2/4` consensus are now
frozen and cannot change after any final14 access.

Final14 is not submitted yet. The remaining pre-unseal gate is mechanical:
extend the formal final14 evaluator and independent verifier to the fifth arm,
run the complete five-arm workflow once on consumed scenes, then seal the
source/SBATCH bundle and arm rotation. Only a passing consumed dry-run may
authorize the single prospective final14 execution.

Only after that test passes should proposal ranking and proof be trained jointly
or Pi3X receive adapters.  DINO remains the long-history address book until a
new experiment overturns the candidate-free `5/20 vs 18/20` result.  The
intended final method is still a unified learned relocalizer; gradual
replacement is the attribution protocol that got there without letting an
offline ranking gain hide an unsafe rejection failure.

## Five-arm formal mechanics completed

The pre-unseal learned-controller gate is now complete on consumed data.  The
immutable five-arm run is:

`/home/cv/memnav_eval/results/final14_learned_five_arm_consumed_gxdo_attempt6_20260817`

with local mirror:

`.diagnostics/learned_relocalizer_20260817/final14_learned_five_arm_consumed_gxdo_attempt6_5090`.

This is an execution/lifecycle test, not an SR result.  It contains one
consumed scene, two queries and all five arms.  All five arms happened to fail
both queries, so no efficacy statement is permitted.  The learned arm did,
however, satisfy the formal transport contract:

- Novel: zero accepts, zero takeovers and exact native fallback;
- Revisit: 15 accepted plans and 15 takeovers;
- one initial top-8 inference per query, zero runtime failures;
- 15/15 accepted plans have finite evaluator-only GT bearing diagnostics;
- maximum accepted angular error `16.3199896 deg`, zero above `90 deg`;
- no burst exhaustion.

The independent audit SHA-256 is
`0f0a60a28cf42db787741bbcebf8be6c0fde83b3b3ddf15e7b151578c7ca154e`;
the aggregate output receipt is
`55ac91a902a2fb989ad0ae1eba76a2e186cd503bc533ef1ebdd0d7f61dda64c0`.

The real-output formal summarizer/verifier fixture also passed.  It correctly
reported a failed L1 efficacy gate rather than treating method failure as an
infrastructure exception, while L2/L3 passed.  The independent verifier
returned `verified=true`.  This fixture remains a mechanics test and is not
pooled into any result table.

The learned-controller-only source bundle is read-only on both RTX 5090 and
HPC:

`final14_learned_execution_c7971aad9104f45c`, with receipt SHA-256
`c7971aad9104f45ccdb2b1ad8f7eeecb9f28944c59e81d2d72d974996566554b`.

## Parent-population mismatch found before unsealing

A post-gate audit found that the Attempt-7 builder bundled above did not fully
implement the already-frozen parent final14 population.  It produced legacy
`support_controlled` and `natural_direction` matched pairs, whereas final14
requires:

1. natural unsupported Novel + standard Revisit (`covis [0.55,0.90]`);
2. an independently constructed hard-support Revisit subset
   (`covis [0.25,0.55)`);
3. the first at most three constructible histories per scene in the frozen
   eight-source order;
4. hard-support attrition that never removes a valid standard/Novel history.

Therefore the `c797...` bundle proves learned five-arm execution but is not a
complete final14 submission bundle.  No final14 identity, asset, construction
artifact or policy outcome was opened, and no job was submitted.

The population layer has now been implemented separately rather than mutating
the old Attempt-7/Replica builders:

- `final14_role_pair_contract.py`: CPU-only frozen schema, lexicographic
  front/side/rear cycle, identity-hash yaw and disjoint support bands;
- `build_final14_role_pair_scene.py`: all successful online-A histories are
  attempted, standard/Natural membership is capped before policy execution,
  and hard support is an independent subset;
- `finalize_final14_role_pairs.py`: distinct denominators and power targets for
  natural/standard and hard support;
- `audit_final14_role_pairs.py`: renderer-free independent support, stratum,
  identity, cap, subset, asset and role-visibility audit;
- the collection, evaluation, summary and independent-verification SBATCH
  path now supports `FINAL14_POPULATION_MODE=1` and the fifth learned arm.

The first consumed construction attempt was retained as a failed attempt: its
endpoint used the final rendered floor pose instead of the exact replay-trace
endpoint, and the independent audit rejected it.  The implementation now uses
the exact trace endpoint throughout.

With the final outcome-blind global stratum cycle
`scene_rank * 8 + source_episode_rank`, the four consumed histories give:

| construction property | result |
|---|---:|
| standard Revisit constructible | 4/4 |
| hard-support Revisit constructible | 4/4 |
| natural stratum query constructible | 2/4 |
| retained standard+natural histories | 2/4 |

The two natural failures are attrition, not repaired by choosing a more
favorable direction.  This is precisely why final14 target power is a target,
not an adaptive admission rule.  The exact-final-code one-scene artifact
passes both the generic role-pair audit and the final14-specific audit; its
construction receipt SHA-256 is
`01061d5908b80463857e5ee49740ca2efa5ff1df8a847e0cbcda756070bac01f`
and audit SHA-256 is
`a86f6cd423446c2a21b34d029c3b3c0db0339bbafed75eb0f1eb51a44010fbe7`.

The CPU-only finalizer was also run on the exact-final-code fragment, and the
population receipt/audit hashes are respectively
`457031957700fac5967eb20e3ea96bf596b78cea846dc60b3c833fee7515ae45`
and
`da7bcf71e0fdac40e5b6285596a2c2d84067997a33cccd724bb769ad431b10cc`.
Legacy real-output summary/verifier compatibility remains intact; the new
summary SHA is
`9f56b359b6e13b2363cbcf480ec2dbc8d4cf5b134c7e17c29b3767382398ec9a`
and independent verifier SHA is
`25d0e94fd84abafd1c435c869dd035d45a98f307bb47678358614e8f3de934da`.

The only remaining pre-unseal action is to seal these population corrections
with the already-qualified learned controller into one new immutable source
bundle and record a hash-bound execution receipt.  The formal final14 run
remains unsubmitted until that receipt exists and the user explicitly
authorizes opening the sealed pool.
