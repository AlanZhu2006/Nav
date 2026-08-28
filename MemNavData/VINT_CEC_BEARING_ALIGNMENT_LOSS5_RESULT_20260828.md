# ViNT--CEC bearing-consumption mechanism result (2026-08-28)

## Result

The frozen five-query, outcome-aware mechanism test completed and passed both
pre-registered gates.  Every cell ran all three arms in the same loaded
MemNav/ViNT processes and on the same GPU.  The final summary and an independent
raw-file verifier both report `verified=true`.

| Arm | Heading within 30 deg | First horizon closer | Mean distance change | Endpoint success |
|---|---:|---:|---:|---:|
| certified anchor, unaligned | 0/5 | 0/5 | +0.242 m | 1/5 |
| native ImageGoal + aligned bearing | 5/5 | 5/5 | -0.288 m | 4/5 |
| certified anchor + aligned bearing | 5/5 | 5/5 | -0.281 m | 5/5 |

The accepted CEC bearing was therefore accurate and executable on every tested
failure.  The earlier certified-anchor adapter failed because it replaced the
ViNT goal image but did not consume the certified direction: without alignment,
the first ViNT horizon pointed 118--177 degrees away from the bearing in these
five cases and increased goal distance in all five.

The native-goal aligned arm establishes that direction is the first missing
interface.  The one case in which it moved correctly but did not reach the goal,
while the aligned-anchor arm succeeded, also suggests that the verified anchor
can remain useful as the visual target after orientation is corrected.

## Frozen gates

- exactly one accepted alignment in each aligned rollout: `5/5` for both arms;
- heading error no greater than 30 degrees: `5/5` for both arms;
- first-horizon distance reduction: `5/5` for both arms;
- endpoint recovery at least `3/5`: native-goal aligned `4/5`, anchor aligned
  `5/5`.

Both the primary direction-consumption gate and the exploratory endpoint gate
passed.

## Evidence boundary

This is not a paper-level controller-portability SR result:

- the five cases were selected after reading the earlier ViNT outcomes;
- the alignment is an idealized zero-translation simulator yaw, not bounded
  physical turn execution;
- the earlier 28-history HM3D result remains a negative result for the
  anchor-only adapter (`Revisit 5/28 -> 0/28`);
- a fresh, outcome-blind population and bounded turn/re-observation contract are
  required before filling the ViNT+CEC paper row.

The correct next adapter is therefore not another matcher or retriever.  It is
`verified bearing -> bounded turn with fresh observations -> verified anchor
ImageGoal -> unchanged ViNT local policy`.

## Source of truth

Authoritative run root:

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  vint_cec_bearing_alignment_loss5_20260828/
  mechanism_retry1_20260828T025800Z
```

- summary: `mechanism_summary.json`, SHA-256
  `81030e707912c48ae51eb78229540271a9f78d76677bfbdc26de6abb4858e61a`;
- independent verification: `mechanism_independent_verification.json`,
  `verified=true`;
- selection manifest SHA-256:
  `ac7fc01f5ac039fe736de00c7365393a92c2310c9d407917271ce1432de116e2`;
- jobs: gate `16497965`, remaining array `16497973`, analysis/verifier
  `16497977`, all completed with exit code `0:0`.

The superseded one-cell submission and its preservation rationale are recorded
in `VINT_CEC_BEARING_ALIGNMENT_SUBMISSION_INCIDENT_20260828.md`.
