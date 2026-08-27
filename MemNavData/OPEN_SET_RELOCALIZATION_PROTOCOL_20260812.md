# Open-set goal relocalization protocol v2 (2026-08-12)

## Question

Can the existing NavDP + memory architecture decide when an ImageGoal is
supported by history and, in the same operation, produce a reliable metric
goal for the existing PointGoal+ImageGoal controller?

This is deliberately **not** framed as forced binary Novel/Revisit
classification.  A low-overlap goal can still be accurately localized, while
a failed match does not prove that the place is novel.  The deployment states
are therefore:

1. **localized/actionable**: history supports a certified metric goal;
2. **unsupported/unknown**: no safe pose certificate; use native ImageGoal
   NavDP;
3. **ambiguous** remains part of the audit vocabulary, never a forced positive.

## Frozen architecture

1. Existing DINO retrieval generates eight temporally separated history
   candidates.  It is a high-recall proposal mechanism only.
2. SuperPoint + LightGlue and Fundamental-MAGSAC rank those eight candidates
   without using co-visibility labels.  The frozen lexicographic order is
   fundamental inliers, query grid coverage, query hull coverage, median match
   confidence, DINO cosine, then earlier frame.
3. For the selected history frame, LingBot-Map supplies causal history depth,
   camera pose and metric scale.
4. LightGlue reference pixels are lifted through LingBot depth into map 3-D;
   PnP estimates the ImageGoal camera pose and hence a metric relative
   PointGoal.
5. A fail-closed v2 certificate accepts only one atomic center-view hypothesis
   with at least 16 PnP inliers, at least 5% convex-hull coverage in **both the
   history reference and ImageGoal**, and at most 2 px reprojection RMSE.
   Rejection means fallback to native NavDP, not a semantic claim that the
   target is novel.

GT-only actionability is aligned with the emitted controller variable: the
metric PointGoal position must be within 0.75 m of the true goal position.
The benchmark success event is distance-only and the PointGoal controller does
not consume the PnP camera yaw.  Direction and yaw errors remain diagnostics;
they are not valid vetoes for a short target vector whose position is already
accurate.

This preserves the current main architecture.  There is no trained binary
selector and no new navigation policy.  The localization expert either emits
the same PointGoal interface already consumed by the revisit controller or
abstains.

## Local evidence used to authorize HPC

The local source set contains 24 train-only sessions from two scenes.  After
enforcing the exact shared universe (teacher evidence present, anchor at least
8, camera-cache `anchor+1` available), 23 sessions and 167 DINO top-8 pairs
remain.  The absent session is operationally Unknown rather than silently
replaced by an unexecutable RGB frame.

- On this exact universe, Fundamental-inlier AUC was 0.9910 and query-grid-
  coverage AUC was 0.9986 against descriptive co-visibility labels; DINO AUC
  was 0.9491.
- The geometry-ranked metric PnP audit found 13/23 poses within the GT-only
  navigation tolerance.  The frozen certificate accepted 11/13 and rejected
  all 10 non-actionable estimates (11 TP, 0 FP, 2 FN).
- On the 11 accepted/actionable estimates, median position error changed from
  LingBot's direct goal-pose estimate 1.329 m to PnP 0.052 m; median bearing
  error changed from 85.51 to 1.41 degrees (10/11 paired improvements,
  two-sided sign-test p=0.0117).
- The earlier executable top-DINO single-candidate audit also certified 11
  correct poses with zero false positives.  Geometry ranking therefore has no
  local certified-coverage gain; its value remains a candidate-ranking
  hypothesis that the 20-scene HPC run must confirm.
- A low co-visibility example localized to 0.05 m / 0.19 degrees after
  geometric ranking.  Its 2.8% query coverage remained below the frozen 5%
  certificate, so it correctly abstained.  This
  falsified the assumption that the old co-visibility threshold is an
  operational Novel/Revisit label.
- A repeated-structure counterexample produced a stable but wrong pose about
  9 m away from all three neighboring anchors.  Multi-view consistency alone
  is therefore not a safe gate; its low 2.8% query coverage correctly forces
  abstention.
- True geometric/colored ICP was rejected: a strict negative achieved colored
  fitness 0.867, while a positive refinement moved position error from 1.11 m
  to 3.00 m and bearing error from 12.98 to 131.5 degrees.  Dense monocular
  clouds can align repeated surfaces without localizing the camera.
- SIFT PnP was also rejected as the primary expert: it missed the positive
  test pair and produced a small bogus solution on a negative pair.

These are train-only mechanism results, not a closed-loop SR claim.

## First HPC audit and v2 amendment

The first frozen v1 job (`15633271`) completed normally on 24 sessions from 20
scenes.  Its original gate failed: 8 TP, 2 FP, 1 FN and 13 TN.  The two apparent
false positives had different causes and cannot be repaired by threshold
tuning:

- `VLzqgDo317F` was a real 16 m repeated-wood-panel alias.  Query support was
  11.46%, but all reference inliers occupied only 1.93% of the history image.
  This exposed a missing symmetric-support invariant.
- `JF19kD82Mey` had only 0.354 m position error and 2.64 degrees rotation error
  in a visually verified same-room pair.  Its 50.3-degree direction error came
  from a true anchor-to-goal distance of only 0.429 m.  The old audit was
  testing an ill-conditioned bearing rather than the metric PointGoal that the
  adapter emits.

The v2 amendment therefore adds the same frozen 5% support floor on the
reference image and makes PointGoal position error the actionability target.
On already observed data this changes the local 23-session audit by nothing
(11 TP / 0 FP) and changes the v1 HPC rows to 9 TP / 0 FP / 1 FN / 14 TN.
Those numbers are post-hoc design evidence only, not confirmation.

## Disjoint v2 HPC confirmation

The frozen job has two stages and touches neither development nor
final-reserved scenes.

### Stage A: retrieval/verification universe

- 480 train sessions, eight causal DINO candidates each (3,840 pairs).
- Minimum executable anchor is frame 8 and temporal gap is 4.  A read-only
  universe audit confirmed this preserves exactly eight candidates for all
  480 sessions; gap 16 would leave 85 sessions under-covered.
- Produce raw LightGlue/Fundamental evidence and a hash-bound report.  Fit no
  threshold and use no label in candidate ranking.

### Stage B: metric actionability sample

- Hash-frozen 24 sessions spanning all 19 train scenes unseen by both the local
  design audit and v1 HPC audit.  The 21 exposed scenes are excluded before
  selection; one session is selected per remaining scene and five extras are
  selected by a separate hash.  Labels are never read by selection.
- Select exactly one candidate per session with the frozen Stage-A ranking.
- Run causal LingBot replay for three neighboring history views and raw-RGB
  LightGlue-to-LingBot-depth PnP.
- The center hypothesis alone is used for the atomic v1 certificate; neighbors
  are diagnostic evidence and cannot cherry-pick maxima from different views.

### Decision gate

The run authorizes a later same-process closed-loop comparison only if:

- the source, model, cache, split, teacher, scale and Stage-A artifacts pass
  all SHA/provenance contracts;
- Stage B has exactly 24 rows from the frozen 20-scene sample;
- the v2 actionability certificate has zero GT-audited false positives;
- at least five certified actionable sessions occur across at least five
  scenes; and
- accepted estimates improve median position error over LingBot's direct
  goal-pose output.

Failure is informative: retain the proven geometry-memory baseline and do not
spend a long closed-loop evaluation on this router.  Passing still does not
establish navigation improvement; the next experiment must be a same-machine,
same-process three-arm closed-loop comparison: native NavDP, current geometry
router, and certified relocalization router.

### Frozen v2 result

Job `15634113` completed on one A100 in 27m28s.  All bundle, causal-input,
scale, cache, split, model and output contracts passed.  On the 24 sessions
from 19 design-unseen scenes, the v2 result was 8 TP, 0 FP, 1 FN and 15 TN.
Precision was 100% (Wilson 95% CI 67.6%--100%), recall was 88.9% (56.5%--
98.0%), and certified true positives covered six scenes.  The effectiveness
gate therefore passed.

The sole false negative had an accurate 0.401 m PointGoal but only 4.59%
reference support, just below the frozen 5% floor; it remains abstained.  A
teacher-negative row with co-visibility 0.184 was certified and localized to
0.212 m, independently confirming that co-visibility class is not deployment
actionability.  Accepted PnP position error had median 0.131 m versus 0.154 m
for LingBot direct pose (5 improvements / 3 regressions, sign-test p=0.727), so
the result supports safe certification and usable metric goals, not a claim of
significant pose-error improvement.

An independent raw-CSV recomputation using only each row's unique center
hypothesis reproduced all counts and medians.  The immutable row SHA256 is
`eed09072d87973bc7232571dcf93fe517596e752e0ac7fa2af43486be80f73c5`.

## Claim discipline

- Co-visibility labels are descriptive audit metadata, not deployment inputs.
- GT pose is used only to measure actionability, never to select or certify.
- No development threshold tuning and no final-reserved access are permitted.
- “No certificate” means unknown/unsupported, not proven Novel.
- Offline localization is a gate for closed loop, never a substitute for SR.
