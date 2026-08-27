# Final14 Support-Band Constructibility Preflight

**Date:** 2026-08-16  
**Status:** passed on consumed scenes; no policy rollout and no final14 access

## Purpose

Before unsealing the final MP3D population, test whether the support bands in
`FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md` can be realized by
controlled re-rendering around actual-online causal history.

This is not a navigation result.  It uses four previously consumed online-A
histories and scores only geometry, rendering difference and co-visibility.

## Frozen smoke scope

- 4 consumed histories / 4 scenes;
- at most the two source frames closest to the frozen `3 m` endpoint-distance
  target per history;
- deterministic pose grid with translation `0.20--1.00 m` and yaw offsets
  `12--60 deg`;
- standard support: max online covis `[0.55,0.90]`, argmax gap `<=24`;
- hard support: max online covis `[0.25,0.55)`, argmax gap `<=32`;
- no NavDP, MemNav or CEC policy request;
- no final14 identity, asset, render or outcome read.

## Result

| band | constructible histories |
|---|---:|
| standard Revisit | `4/4` |
| hard-support Revisit | `3/4` |

Selected standard candidates had max-covis `0.7169--0.7573`; selected hard
candidates had max-covis `0.4626--0.5328`.  They therefore occupy genuinely
different support bands.

Other observed ranges:

- standard translation `0.22--0.65 m`, yaw delta `12--36 deg`;
- hard translation `0.30--0.85 m`, yaw delta `24--54 deg`;
- standard pixel MAE `21.1--60.5`;
- hard pixel MAE `23.4--61.3`;
- query geodesic `2.21--3.43 m`.

The one hard-band miss remained a miss after both frozen source frames and is
retained as constructibility attrition.  It was not repaired by loosening a
band.

## Reproducibility

Implementation:

- `audit_final14_support_band_constructibility.py`;
- `test_audit_final14_support_band_constructibility.py`, `2/2` passed.

Machine-readable report:

`.diagnostics/final14_support_band_constructibility_smoke_20260816/report.json`

SHA-256:

`4e2dba2fb88fee5c206577bd25e4820e213d0cb4e62357974d0fa6d2775d4797`

Frozen final protocol SHA-256:

`3d1ebc6ef429fd16df4d550eda52eceb55d7b15fd181a5c00c0b8f971f7aaa32`

## Decision

`support_stratified_final14_design_is_constructible_continue_consumed_dry_run`

This preflight authorizes implementation and a full consumed-scene dry-run.
It does not authorize changing the support bands and does not yet unseal
final14.

