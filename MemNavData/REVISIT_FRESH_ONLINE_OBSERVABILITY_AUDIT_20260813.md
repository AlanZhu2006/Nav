# Fresh160 actual-online-A observability audit

Date frozen: 2026-08-13 (Asia/Shanghai)

## Why this supplemental audit is required

The fresh160 causal comparison itself is valid: all three arms replay the same
online NavDP Goal-A trace, use paired seeds, and differ only in the Goal-B
controller/router.  Its frozen result is therefore a real architecture result:

- shared A: `118/160`;
- geometry B given A: `93/118`;
- raw-DINO direct B given A: `109/118`;
- native B given A: `31/118`;
- direct minus geometry: `+20/-4`, exact McNemar
  `p=0.0015438795`, conditional risk difference `+13.56 pp`.

However, the generator chose Goal B using co-visibility with the **expert A
trajectory**, while evaluation memory contains the **actual online NavDP A
trajectory**.  The original manifest did not recompute the Revisit label against
that online trace.  Consequently, `109/118` is currently a B-given-A result on a
generator-defined Revisit benchmark; it must not yet be described as actual
online-Revisit SR without this audit.

This is a label-contract audit, not a new controller experiment.  It does not
rerun NavDP and cannot change any action or success outcome.

## Frozen measurement

For every episode, render Goal B at its frozen metadata pose and render every
pose from the frozen online-A trace.  Use the generator's exact camera model and
occlusion-aware 3-D surface re-projection:

- resolution: `480 x 270`;
- intrinsics: `fx=355.81464`, `fy=351.687`, `cx=240`, `cy=135`;
- goal-depth back-projection stride: `6`;
- depth-consistency tolerance: `0.30 m`;
- camera height: episode metadata, defaulting to the frozen `0.50 m` contract.

For each episode record the entire online co-visibility curve, maximum,
argmax, supported-frame counts, recall gap, and a spatial-nearest diagnostic.
Spatial distance is not used as the Revisit label because nearby poses can face
away and more distant poses can retain strong common visible surface.

## Outcome-independent support bands

The thresholds are copied from the episode generator and frozen before the raw
Goal-B outcome CSVs are loaded:

| actual online-A max co-visibility | label | source |
|---:|---|---|
| `<0.10` | no support | generator negative threshold |
| `[0.10,0.20)` | ambiguous | gap between negative and acceptance bands |
| `[0.20,0.50)` | supported Revisit | generator Revisit acceptance threshold |
| `>=0.50` | strongly supported Revisit | retrieval positive threshold |

The program must first finish all 160 observability rows.  Only then may it read
the three arm CSVs and stratify the already-frozen outcomes.

## Required reports

Report, without retuning:

1. coverage of `max covis >=0.20` and `>=0.50` over all 160 episodes and over
   the 118 shared-A successes;
2. geometry/direct/native B success in all shared-A successes;
3. the same three-arm counts in the `>=0.20` and `>=0.50` subsets;
4. direct-minus-geometry paired gain/loss, exact McNemar, and scene-cluster
   bootstrap interval in each subset;
5. `<0.20` outcomes only as a diagnostic, never as evidence of Revisit ability.

The already-consumed episodes cannot be used to choose a new threshold.  This
audit cannot authorize blind evaluation or a paper-level generalization.

## Interpretation contract

- The original direct-minus-geometry contrast remains a valid causal comparison
  on the frozen generator-defined benchmark regardless of observability coverage.
- Actual-online-Revisit SR must use only episodes with `max covis >=0.20`.
- The `>=0.50` subset is the stronger retrieval-support sensitivity analysis.
- If a material number of shared-A successes fall below `0.20`, the unstratified
  `109/118` must not be presented as Revisit SR; report the support-conditioned
  denominator instead.
- No result from this audit fixes online Novel/Revisit selection.  It only checks
  whether the evaluated Goal-B view was in the memory that the controller
  actually received.

## Implementation and preflight evidence

- auditor: `MemNavData/audit_revisit_fresh_online_observability.py`;
- tests: `MemNavData/test_audit_revisit_fresh_online_observability.py`;
- local tests: `13 passed`;
- real-trace smoke: one `17DRP5sb8fy/episode_0003` online-A trace reproduced
  `max covis=0.7246906636670416`, `argmax=153`, `105` frames at `>=0.50`, and
  `109` frames at `>=0.20`;
- the smoke also reproduced the stored Goal-B JPEG and all `175/175` trace JPEG
  hashes exactly.

The formal input is the immutable run:

`/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000`

with manifest SHA256
`8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`.

