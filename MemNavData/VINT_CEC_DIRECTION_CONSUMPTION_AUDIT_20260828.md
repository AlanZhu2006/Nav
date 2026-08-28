# ViNT + CEC direction-consumption audit

Date: 2026-08-28
Scope: **post-hoc mechanism audit; not a new success-rate experiment**

## Question

The completed HM3D ViNT controller-native comparison passed the CEC proof to
ViNT by replacing its ImageGoal with the certified historical anchor.  This
audit asks a narrower question:

> Does the first physical ViNT horizon align with the scale-free bearing that
> is already sealed inside the same CEC handoff packet?

It does not change an action, select an episode, rescore success, or use an
oracle to choose a trajectory.

## Frozen source

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_vint_controller_native_cec_20260828/
  hm3d_vint_cec_table1_20260828
```

The input population is all 28 formal grant-arm Revisit plan files.  The audit
fails closed unless exactly 28 distinct formal cells are present.

## Measurement

For each cell:

1. read the first accepted `public_proof.direction_vector = [forward, left]`;
2. calculate its relative bearing with `atan2(left, forward)`;
3. convert the displacement over the first eight executed simulator steps to
   the same robot-local `[forward, left]` convention;
4. report the wrapped absolute angle between proof bearing and executed
   heading;
5. compare the evaluator's ground-truth goal distance at plan step 0 and plan
   step 8 only as a diagnostic of physical progress.

The output must retain the SHA-256 of every source plan JSON.  Because this
audit was designed after observing the negative formal result, it may explain
that result but cannot serve as an independently preregistered confirmation.

## Interpretation boundary

- Large mismatch supports an interface-consumption failure: the anchor-goal
  projection does not preserve the certified direction in physical control.
- Small mismatch with continued failure would instead point downstream toward
  ViNT waypoint semantics, collision execution, or longer-horizon planning.
- This audit cannot establish that a bearing-first ViNT intervention succeeds;
  that requires a separately frozen closed-loop mechanism test.
