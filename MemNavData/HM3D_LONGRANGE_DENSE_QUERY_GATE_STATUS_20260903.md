# HM3D long-range dense-query motion gate

Date: 2026-09-03 (Asia/Shanghai)  
Status: **consumed gate complete and independently verified; failed 1/3
geometry-stop criterion**

## Question

The independently verified 23-history same-floor experiment showed that a
route tangent improves over a single endpoint bearing (`4/23` versus `0/23`),
but the monocular route stream stopped on `14/23` episodes. Thirteen live
failures followed coarse controller intervals with little translation and a
mean absolute yaw of `33.23 deg`. This gate asks only whether estimating every
causal query-frame transition removes that observed turn-heavy series failure.
It is not a navigation or generalization experiment.

Frozen protocol:

```text
MemNavData/hm3d_longrange_dense_query_gate_protocol_v2_20260903.json
SHA-256 8231222ad2a91658987dd640d20697d1bf78614d0f9939090f215f0dfe95f391
```

The three consumed histories are the first three formal **live-query**
geometry failures in the sealed parent-manifest order (`0, 1, 6`). History 2
is excluded because its route failed during historical initialization and
cannot identify a live-query intervention. The pass rule is `0/3` typed
geometry stops, unchanged initial CEC authority and historical route
estimator, and a receipt for every available live frame transition.

## Exact candidate

The candidate changes one inference boundary only:

```text
historical causal tape: fixed stride 8, Fundamental-MAGSAC -> depth PnP
live query interval:    every action frame, direct depth PnP
route state:            same path-budgeted monotone projection
policy payload:         same unit bearing at fixed 2.5 m radius
controller:             unchanged frozen NavDP
```

The initial DINO/LightGlue/PnP CEC proof, historical route sampling, LingBot
depth and frame-40 height scale, tangent baseline, policy seed/FIFO, execution
budget, and success definition are unchanged. Runtime receives no role,
distance, evaluator pose, metric-depth sensor, wheel odometry, stuck trigger,
endpoint fallback, or native fallback after CEC accepts.

Direct PnP is restricted to live consecutive frames because pure or nearly
pure camera rotation is degenerate for a Fundamental-matrix prefilter while a
depth-backed 2D--3D pose remains defined. It is not used as a blanket
replacement for the historical estimator.

## Superseded v1 attempt

The first source bundle was:

```text
hm3d_longrange_dense_query_gate_77d6445d3653bf65
evaluation 16839451_[0-2]
analysis   16839452
verifier   16839453
```

The v1 protocol selected the first three geometry failures (`0, 1, 2`) while
its motivating question concerned live motion. It also did not explicitly
freeze the historical edge estimator. The implementation selected direct PnP
for both historical and live edges; history `2` then failed before any active
route plan and the runner aborted with
`dense-query arm produced no active accepted plan`. This is an implementation
and experimental-design failure, not a failed live-query gate and not a
navigation outcome. Its output is retained under its own source-addressed run
root and must not be merged with the v2 attempt.

The incident exposed a second audit defect: a typed geometry stop before route
initialization was treated as missing output. The repaired runner serializes
that state as a legitimate failed mechanism record, so downstream analysis can
distinguish candidate failure from infrastructure failure.

The v2 protocol was frozen before its new candidate outcomes were run. It
explicitly selects only live-query failures and fixes the historical estimator
to the formal parent's Fundamental-MAGSAC -> PnP path.

## V2 sealed implementation

The code now resolves the estimator per edge kind:

```text
direct_pnp_dense_query + historical_adjacent_sample -> fundamental_then_pnp
direct_pnp_dense_query + history_to_query_bridge     -> fundamental_then_pnp
direct_pnp_dense_query + live_query_adjacent_sample  -> direct_pnp
```

Every plan and edge receipt records both the requested composite model and the
resolved stage-specific estimator. Unit tests exercise the resolver and the
actual PnP-call boundary. Bundle-local and staged read-only self-tests both
passed:

```text
56 passed
source bundle: hm3d_longrange_dense_query_gate_a29950244cc53b6f
receipt SHA-256:
a29950244cc53b6fa3951f741626937f7b6fe7e42a59cb9e31f752be0bc7b14c
```

## Decision boundary

- Pass: run a consumed full-population interface attribution combining this
  dense visual route clock with the already implemented deterministic NavDP
  support projection. Any SR claim still requires a newly frozen population.
- Fail: do not relax PnP thresholds and do not add a fallback. Inspect the
  first failed atomic RGB transition, then test a continuous prediction-plus-
  visual-evidence route-state estimator.

The canonical short/mid-range CEC paper method and its existing results remain
unchanged regardless of this development gate.

## Formal execution status

The v2 bundle was independently staged and verified on the authoritative
`yz11502` account. The formal Slurm DAG is:

```text
evaluation  16842471_[0,1,6]
analysis    16842472
verifier    16842473
run root    /scratch/yz11502/Research/Nav-axis-uturn-results/
            hm3d_longrange_dense_live_query_gate_20260903/
            consumed_a29950244cc53b6f
```

All three selected histories completed, followed by the sealed aggregate and
independent verifier:

```text
summary job                         16842472  completed
independent verifier               16842473  completed
independent verification           true
geometry-stream failures           1/3
descriptive navigation successes   0/3
formal decision                    gate failed
```

Histories `0` and `1` individually pass the live-motion mechanism check:

```text
history                         0       1
completed live intervals      137     112
per-action dense intervals    137     112
geometry-stream stops           0       0
accepted route plans          138     113
initial CEC proof unchanged   true    true
```

The navigation outcomes remain descriptive because these histories were
selected after reading the parent failures. Both still terminate `stuck`:

```text
history                         0       1
executed actions             1098     899
executed path (m)           26.09   18.80
final 3-D goal distance (m) 11.42    4.44
```

History `0` additionally exposes accumulated route-state error. Its estimated
route coordinate reaches 100% at plan 89 while the evaluator-only goal
distance is still 6.63 m; cross-track error then grows from 3.15 m to 7.04 m
and the held terminal tangent drives away. History `1` remains internally
close to the route but exhausts its rollout at 89.7% estimated progress and
4.44 m true distance. Dense live PnP therefore removes the brittle stop on
these two examples, but does not make accumulated visual dead reckoning a
reliable long-range route coordinate.

History `6` stops before route initialization, so it cannot exercise the
intended live-only intervention:

```text
initial CEC proof unchanged       true
historical estimator              Fundamental-MAGSAC -> PnP
failure                           adjacent visual motion rejected: status_ok
active route plans                0
geometry-stream stops             1
```

The formal parent for history `6` ran on A100 `ga030`, whereas this gate
element ran on H100 `gh008`. The initial CEC proof stayed identical, but a
borderline historical edge did not reproduce across GPU classes. This is a
known numerical reproducibility nuisance, not permission to replace the
failed formal result. A same-A100 rerun may be used only as a consumed
sensitivity diagnostic and must remain separate from this verified gate.

That sensitivity was submitted without replacing any formal output:

```text
job          16848570_[6]
partition    a100_tandon only
run root     /scratch/yz11502/Research/Nav-axis-uturn-results/
             hm3d_longrange_dense_live_query_gate_20260903/
             sensitivity_index6_a100_a29950244cc53b6f
status       pending at 2026-09-03 11:42 EDT (QOSGrpGRES)
```

Its only question is whether the live-query intervention can be exercised
when the nuisance historical stage runs on the same GPU class as its parent.
It cannot change the verified `1/3` formal gate result or create an SR claim.

Because the frozen requirement was zero failures, no full-population or fresh
SR experiment is authorized from this gate. The next method-level object is a
continuous route-state estimator that combines a motion prior with repeated
visual evidence, rather than integrating an unbounded chain of relative
poses.

## Same-edge shadow diagnostic

The separately frozen, no-authority shadow diagnostic is complete and its
independent verifier reports `verified=true`:

```text
primary geometry stops reproduced       14/14
direct-PnP shadow valid                   1/14
accurate direct-PnP shadow                1/14
decision   simple_epipolar_explanation_not_established
```

In particular, simply deleting Fundamental-MAGSAC is not a valid historical
route-motion fix. On the exact H100 retry of history `17`, the unfiltered PnP
estimate had 155.6 degrees of yaw error. This shadow result and the dense
live-query gate answer different questions: the former rejects a blanket
historical-estimator replacement, while the latter tests direct PnP only on
short consecutive live-query transitions.
