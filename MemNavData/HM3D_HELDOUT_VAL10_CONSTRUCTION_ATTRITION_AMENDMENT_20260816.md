# HM3D held-out val10 construction-attrition amendment

Date: 2026-08-16 (Asia/Shanghai)

## Causal boundary

The frozen HM3D held-out-val10 chain stopped during episode construction.  The
prepare job and nine of ten generation tasks completed.  Generation task 8,
for `q3hn1WQ12rz`, exhausted all 240 frozen outer attempts and generated zero
of four requested episodes.  The manifest, every policy-evaluation task, the
primary summary, and the independent verifier were cancelled by dependency;
the run root contains no scene-policy output.  Consequently no navigation
outcome, arm comparison, SR, SPL, final distance, certificate decision, or
policy trajectory was available when this amendment was frozen.

The preserved parent run is:

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_heldout_val10_revisit_20260816/hm3d_heldout_val10_20260816T0448Z`

Its jobs were prepare `15814346`, generation array `15814347`, manifest
`15814348`, evaluation array `15814349`, summary `15814350`, and verification
`15814351`.  The failed generation summary has SHA256
`672055791985b4199a6c60e6ce639bfa0e45d4abdbdd840f66e5207e40fc39b7`.

## Frozen repair

The selected population remains the same ten outcome-disjoint HM3D scenes in
the original order.  No scene is replaced and no episode, geometry threshold,
seed, generator constraint, controller, model, arm, or metric is changed.

The repair treats pre-navigation constructibility as explicit attrition:

- all nine scenes that completed the original frozen generator contribute all
  four generated episodes, for 36 evaluated episodes;
- `q3hn1WQ12rz` remains in the selected population with zero evaluated
  episodes and its complete failure receipt is carried into the manifest and
  final report;
- original scene indices are retained, so `X4qjx5vquwH` remains index 9 and
  every Williams arm order remains exactly as originally frozen;
- cluster uncertainty uses the nine constructible scene clusters;
- the result is explicitly underpowered relative to the original 10-scene,
  40-episode target.

Only the already generated episode files may be consumed.  Re-running the
failed scene with a larger budget, relaxing clearance/covisibility constraints,
or selecting a replacement scene would change the construction population and
is not authorized by this amendment.

## Resume contract

The repair appends only the missing identity manifest, the sparse evaluation
array for original indices `0-7,9`, the paired summary, the independent raw-file
recount, and a repair submission receipt.  Existing generated data are reused
read-only.  Every stage remains fail-closed and creates outputs exclusively.

This is an outcome-blind population-accounting amendment, not a method change
and not evidence for or against CEC, raw DINO, geometry routing, or NavDP.
