# HM3D held-out val10 formal Revisit result

Date: 2026-08-17

## Headline

The non-MP3D external evaluation is complete and independently verified.
Across 36 intention-to-treat episodes from nine constructible, outcome-disjoint
HM3D val scenes, the role-free Certified Episodic Compass raised joint success
from `7/36` to `19/36` relative to frozen native NavDP.  Conditional on the
byte-identically shared Goal-A trace succeeding, Revisit-B success rose from
`7/21` to `19/21`.

The paired certified-versus-native result is `+12/-0`, exact McNemar
`p=0.000488`.  The independent raw-file recount reports `verified=true` and
reproduces every arm count and paired contrast.

This is strong evidence for cross-dataset Revisit utility.  It is not evidence
that certificate is significantly better than the two strong memory controls:
certified was `+2/-0` versus geometry (`p=0.5`) and `+1/-0` versus the
non-deployable oracle-role raw-fixed arm (`p=1.0`).

## Frozen population and estimands

- Ten HM3D val scenes were selected before navigation outcomes.
- One scene failed the pre-navigation constructibility contract and remains
  explicit attrition; nine scenes and 36 episodes were evaluated.
- All 36 episodes are retained under intention-to-treat.
- Every arm replays the same actual-online native Goal-A trace.
- Goal A succeeded in `21/36` episodes; therefore conditional Revisit-B has
  denominator 21 for every arm.
- No MP3D evaluation data, public GOAT score, role label, threshold tuning, or
  post-outcome filtering is used.

## Primary results

| Arm | Goal A | Revisit B given A | Joint | Mean B steps given A |
|---|---:|---:|---:|---:|
| native NavDP | 21/36 | 7/21 = 33.33% | 7/36 = 19.44% | 347.52 |
| old geometry router | 21/36 | 17/21 = 80.95% | 17/36 = 47.22% | 182.62 |
| raw fixed, oracle role | 21/36 | 18/21 = 85.71% | 18/36 = 50.00% | 132.86 |
| role-free certified | 21/36 | 19/21 = 90.48% | 19/36 = 52.78% | 133.48 |

### Paired contrasts

| Contrast | B given A gain/loss | Conditional risk difference (95% scene-cluster CI) | Exact p | Joint risk difference (95% scene-cluster CI) |
|---|---:|---:|---:|---:|
| certified − native | +12/−0 | +57.14 pp `[+36.36,+78.95]` | 0.000488 | +33.33 pp `[+22.22,+44.44]` |
| geometry − native | +10/−0 | +47.62 pp `[+26.09,+72.22]` | 0.001953 | +27.78 pp `[+16.67,+38.89]` |
| raw oracle-role − native | +12/−1 | +52.38 pp `[+22.73,+78.95]` | 0.003418 | +30.56 pp `[+13.89,+44.44]` |
| certified − geometry | +2/−0 | +9.52 pp `[0,+22.22]` | 0.5 | +5.56 pp `[0,+13.89]` |
| certified − raw oracle-role | +1/−0 | +4.76 pp `[0,+14.29]` | 1.0 | +2.78 pp `[0,+8.33]` |

The 12 certified gains over native span eight of the nine scene clusters, so
the effect is not driven by one room.  There are no certified losses relative
to native in this population.

## Certificate audit

- Takeover episodes: `20/21` Goal-A-eligible episodes (95.24% coverage).
- Fallback episodes: 1; exact-native fallback episodes: 1.
- Fallback behavior mismatches: 0.
- Runtime failures: 0.
- Requests / accepted plans / abstained plans: `358 / 325 / 33`.
- Selected DINO rank: median 2, maximum 8 over 21 episodes.
- PnP inliers: median 234 over 21 episodes.
- Uncached certificate latency: median 5.83 s, p95 25.83 s.

The rank audit is useful mechanistically: the geometric stage often selected
an anchor other than DINO top-1, and needed the full top-8 at least once.  The
latency result is also a real deployment limitation and must not be hidden;
the current implementation demonstrates utility, not a real-time optimized
front end.

## What this establishes

1. **External transfer is real.**  A method developed on the MP3D lineage
   provides a large, lossless paired Revisit gain on disjoint HM3D scenes.
2. **The evidence is actual-online.**  The memory is the frozen controller's
   causal RGB history, not an expert-only trajectory or future observation.
3. **Role-free authorization retains utility.**  Certified is not told the
   phase role, yet reaches `19/21`, numerically matching or exceeding the
   oracle-role raw-fixed ceiling in this sample.
4. **The method-specific SR margin is not yet proven.**  The result strongly
   beats no-memory native, but does not establish a statistically significant
   advantage over geometry or raw-fixed.
5. **This protocol cannot establish Novel safety.**  It contains Revisit-B
   queries only.  Open-set abstention must be supported by the separate mixed
   Novel/Revisit evidence, not inferred from this table.

## Correct paper interpretation

The defensible claim is not “a complex certificate raises SR beyond every
retrieval baseline.”  It is:

> A causal online visual history can be converted into a self-authorized,
> scale-free episodic compass for a frozen ImageGoal controller.  On held-out
> HM3D, this recovers 12 native failures without a loss, while exact fallback
> preserves native behavior when certification abstains.

The raw oracle-role result shows that much of the attainable utility comes
from supplying a useful historical bearing once Revisit is known.  The
certificate's distinct role is to make that intervention role-free and
verifiable.  The old geometry arm remaining strong means the paper should
emphasize the clean output/fallback contract, cross-dataset evidence, and
open-set authorization rather than claiming that every geometric component is
individually novel.

## Analysis-interface incident and repair

All nine GPU scene tasks completed `0:0`.  Original summary job `15839655`
then failed before reading trace or arm metrics because the runtime-repair
runner wrote the frozen schema name
`hm3d_heldout_runtime_repair_scene_v1_20260816`, while the summarizer accepted
only the legacy schema name.  It created no report; verifier `15839656` was
cancelled by dependency.

All nine contracts were audited field by field.  The only interface extension
was the new schema name plus `runtime_repair_method_change=false`.  The repair
explicitly accepts the two frozen schema lineages, requires that guard to be
false for the runtime-repair lineage, and continues to reject every unknown
schema.  Twenty tests pass.  No evaluator, rollout, metric, population, method,
threshold, or estimand changed, and no GPU episode was rerun.

The sealed analysis-only bundle is:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_summary_schema_analysis_repair_045f1cb3b7a21ac9
```

Its receipt-file SHA-256 is
`045f1cb3b7a21ac9bacdc13ef6d3795810d9d8b1ae2030c499d0b6da98fc675d`.
Repair summary `15847580` and verifier `15847581` completed `0:0`.

## Artifacts

Remote run root:

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_heldout_val10_runtime_repair_20260816/
  hm3d_heldout_val10_rt_20260816T1345Z
```

Local sealed copies:

```text
.diagnostics/hm3d_heldout_val10_runtime_repair_20260817/formal_result/
```

- summary SHA-256:
  `4e9946a3580c3eff0ffa35567c30527eb96dbbbfc822de11f982630183e8bd34`;
- independent verification SHA-256:
  `7bb99848bf4c573af62c3f020015996cd1fba9dfa0f5045af79a6fba5a1def97`;
- independent verifier: `verified=true`.

The machine-readable analysis-repair receipt is
`MemNavData/HM3D_SUMMARY_SCHEMA_REPAIR_RECEIPT_20260817.json`.
