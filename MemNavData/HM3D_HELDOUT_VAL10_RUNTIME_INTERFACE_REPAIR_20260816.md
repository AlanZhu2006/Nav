# HM3D held-out val10 runtime-interface repair

Date: 2026-08-16

## Scope

This amendment repairs an execution-interface mismatch only. It does not
change the frozen scene population, episode population, arm definitions,
controller settings, seeds, success criterion, or statistical analysis.

The preserved outcome-blind construction-attrition manifest has SHA-256
`62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89`
and contains 36 episodes from nine constructible scenes at original indices
`0,1,2,3,4,5,6,7,9`.

## Incident

The construction manifest job `15826322` completed. Evaluation array
`15826323` and all nine array executions failed before Goal-A evaluation.
Summary `15826324` and independent verification `15826325` were cancelled by
dependency.

All nine `eval_trace_source.log` files have the identical SHA-256
`5ea7d8036dba93d2d08be8264e2114158589edaa404561770ce308bfeb67fa82`
and end with:

```text
eval_2leg_habitat.py: error: unrecognized arguments:
  --certified_cdec_rescue off --certified_stagnation_graph off
```

There are zero `metric.csv`, evaluation `summary.json`, plan JSON, or Goal-A
trace JSON outputs under the failed scene roots. Therefore no navigation
outcome was generated or read before defining this repair.

The original base evaluator also lacks the frozen raw control arm
`raw_fixed_bearing_v1`; deleting the two rejected flags would not repair the
four-arm protocol.

## Frozen runtime repair

Use the complete, previously successful actual-online NNR runtime bundle:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  shared_online_nnr_11458cb2b75ee334
```

Its complete `SOURCE_BUNDLE.sha256` validation returns zero and is bound by:

- receipt-file SHA-256:
  `31b3e087b855e0220f6821ad96e6f5e74114bc12dc6c3afa6f7f79150dfb4575`;
- evaluator SHA-256:
  `4552a93910d91c4957f170ef311ddd7a9151d6754eea246fc91141b41f349d75`;
- native runtime adapter SHA-256:
  `bdc467df7592bd3078de291ae09837f500a76aff40aa0d5d88e02291eeb7c098`.

That evaluator already supports the two rejected CLI options and the
certificate runtime used by the successful actual-online NNR lineage. The
only overlay is the audited controller-boundary module
`revisit_bearing_adapter.py`, whose SHA-256 before packaging is
`c1f10b3c831f00a5b4742e0b34ac0675f10e161c4795ed1497c74b9551fdaf78`.
It adds the already frozen `raw_fixed_bearing_v1` ablation while preserving
the certified fixed-bearing interface. The evaluator is invoked as a Python
module so this single overlay is resolved before the runtime's adapter; the
evaluator and both servers remain byte-identical to the verified runtime.

The older sealed base bundle remains dependency-only: InternNav, LightGlue,
Python dependencies, Torch cache, and immutable checkpoints. It is not used
as the evaluator or server source.

## Mandatory consumed-scene gate

Before any held-out array task can run, a four-arm smoke must complete on the
already consumed HM3D scene `5cdEh9F2hJL`, episode `episode_0001`. This exact
episode previously completed Goal A and all three then-existing Goal-B arms;
it is outside the held-out val10 scene set.

The smoke must verify, without using success as an acceptance criterion:

1. the runtime and overlay receipts;
2. evaluator CLI import and all four adapter names;
3. one byte-identical shared Goal-A trace across native, raw-fixed, geometry,
   and certified arms;
4. non-empty Goal-B execution for all arms after successful Goal A;
5. raw-fixed adapter takeover fields;
6. certified request/response fields;
7. complete metric/summary/plan output schemas.

The formal sparse array has an `afterok` dependency on this smoke. A smoke
failure therefore cannot consume held-out navigation outcomes.

## Formal-output isolation

The failed run root is preserved unchanged. Formal repair evaluation writes
to a new run root containing a byte-identical copy of the sealed data manifest
and receipt. The submission receipt binds the failed incident, runtime
bundle, one-file overlay, smoke data, smoke job, formal array, summary, and
independent-verification job IDs.

This is an infrastructure repair, not a new experimental attempt and not a
method-selection opportunity.

## Consumed-smoke attempt 1

The first dependency-gated smoke was job `15838383`; downstream jobs were
`15838384`--`15838386`.  It generated no trace, metric, plan, or navigation
outcome.  The runtime NavDP server failed during Python import with
`ModuleNotFoundError: No module named 'depth_anything'`, before binding its
port.  The job and its untouched dependency chain were cancelled.

Comparison against the sealed successful lineage's own
`slurm_shared_online_nnr_eval.sbatch` identified the exact omitted runtime
paths: `${BASE_SOURCE_ROOT}/NavDP/baselines/navdp` and
`${BASE_SOURCE_ROOT}/NavDP/baselines/memnav`.  The former contains the required
top-level `depth_anything` package.  The repair adds those two dependency-only
paths in the same order used by the successful lineage; it does not change an
evaluator, server, model, checkpoint, arm, or navigation parameter.  A new
read-only source bundle and new smoke/formal run roots are required.  The
failed smoke and first formal-repair root remain preserved.

## Consumed-smoke attempt 2

Runtime-path repair job `15838486` started both servers successfully and
confirmed the frozen evaluator and overlay hashes.  Before any policy rollout,
the evaluator rejected the smoke data bundle because Goal A is reconstructed
from expert RGB frame `191.jpg`, while the minimal bundle contained only
metadata, parquet, and Goal-B image files.  It generated zero trace, metric,
plan, or navigation outputs; dependency jobs `15838487`, `15838490`, and
`15838491` therefore could not run.

The next immutable smoke-data bundle includes the complete pre-existing
consumed episode directory, including its expert RGB/depth streams.  This is a
smoke-input completeness repair only.  The runtime source, one-file adapter
overlay, formal manifest, methods, and navigation settings remain unchanged.

## Consumed-smoke attempt 3: passed

Job `15839649` completed the four-arm consumed-scene gate on L40S and wrote
`receipt.json` with `passed=true`.  The shared Goal-A trace SHA-256 was
`a68b42259cf242f0cf2f50c6888eea5a042ca50fe910ac1e488c7ee22ee09a1d`.
All four arms executed Goal B from that byte-identical trace.  Raw-fixed used
the expected adapter for 9/9 plans; geometry executed 11 plans with 10
takeovers; certified issued 9 requests with 9 verified-bearing takeovers and
zero runtime failures.  The receipt revalidated evaluator SHA-256
`4552a93910d91c4957f170ef311ddd7a9151d6754eea246fc91141b41f349d75`
and overlay SHA-256
`c1f10b3c831f00a5b4742e0b34ac0675f10e161c4795ed1497c74b9551fdaf78`.

Goal-B success values observed in this consumed smoke are not an efficacy
result and are not used as a gate.  Formal sparse array `15839654` was released
only by the interface/schema checks above; summary `15839655` and verifier
`15839656` remain downstream.

## Formal execution checkpoint: 2026-08-16 23:00 CST

The new isolated formal root is:

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_heldout_val10_runtime_repair_20260816/
  hm3d_heldout_val10_rt_20260816T1345Z
```

Four of the nine sparse-array scene tasks have completed.  Each wrote four
shared Goal-A trace rows and four rows for every frozen Goal-B arm, emitted a
scene-level `status=complete` contract for four episodes and all four arms,
and exited `COMPLETED 0:0`:

| Frozen index | Scene | Slurm elapsed |
|---:|---|---:|
| 0 | `HaxA7YrQdEC` | 00:29:32 |
| 1 | `BHXhpBwSMLh` | 00:29:10 |
| 2 | `SUHsP6z2gcJ` | 00:28:22 |
| 3 | `tQ5s4ShP627` | 00:22:53 |

Indices `4,5,6,7,9` remain in the original array and are pending only on
`QOSGrpGRES`.  No duplicate or replacement jobs have been submitted.  The
account has no other running GPU job.  The summary and independent verifier
remain dependency-held.  This checkpoint deliberately records only execution
completeness and schema/contract state; success fields and incremental SR have
not been used or reported.

## Formal completion and independently verified result

All nine sparse-array scene tasks completed `0:0`; every task emitted the
four-episode/four-arm `status=complete` scene contract.  The frozen formal
population is therefore 36 intention-to-treat episodes from nine constructible
scenes, with Goal A successful in 21 episodes.

Original summary `15839655` correctly failed closed before reading metric rows
because this runtime-repair runner's explicit scene schema name was not yet in
the legacy summarizer's whitelist.  It created no report, and dependent
verifier `15839656` was cancelled.  A field-level audit found all nine
contracts identical in schema lineage and confirmed
`runtime_repair_method_change=false`.  The analysis-only repair accepts exactly
the legacy and frozen runtime-repair schema names, requires the method-change
guard to remain false, and rejects unknown schemas.  No rollout was rerun.

Repair summary `15847580` and independent verifier `15847581` completed `0:0`.
Both sealed output receipts validate and the independent report says
`verified=true`.  Primary counts are:

| Arm | Revisit B given shared A | Joint |
|---|---:|---:|
| native | 7/21 | 7/36 |
| geometry | 17/21 | 17/36 |
| raw fixed, oracle role | 18/21 | 18/36 |
| role-free certified | 19/21 | 19/36 |

Certified versus native is `+12/-0`, exact McNemar `p=0.000488`; joint risk
difference is +33.33 pp with scene-cluster 95% CI `[+22.22,+44.44]`.
Certified versus geometry is only `+2/-0`, `p=0.5`, and versus raw oracle-role
is `+1/-0`, `p=1.0`.  Thus external Revisit utility is confirmed, while an
incremental SR advantage of certificate over the strong memory controls is
not.

The full result and incident audit are in
`MemNavData/HM3D_HELDOUT_VAL10_FORMAL_RESULT_20260817.md` and
`MemNavData/HM3D_SUMMARY_SCHEMA_REPAIR_RECEIPT_20260817.json`.
