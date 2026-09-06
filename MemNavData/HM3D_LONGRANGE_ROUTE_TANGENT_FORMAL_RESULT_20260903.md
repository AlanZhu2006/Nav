# HM3D same-floor long-range route-tangent result

Date: 2026-09-03 (Asia/Shanghai)  
Status: **independently verified positive result, but the preregistered confirmation rule was not met**  
Scope: fresh-history, reused-scene, controlled causal-RGB survey; same-floor Revisit queries at 20--30 m

## 1. Frozen question

Canonical CEC converts an accepted historical target into one fixed-radius endpoint bearing. This experiment asks whether the same accepted target can instead expose the first feasible tangent of its causally observed route, without a distance switch, stuck trigger, endpoint fallback, or native fallback after acceptance.

The three paired arms were:

1. `mono_native`: frozen monocular NavDP without memory control;
2. `mono_cec_endpoint`: the canonical fixed-2.5 m endpoint bearing;
3. `mono_cec_route_tangent`: the first feasible local tangent of one sparse causal monocular route, normalized to the same 2.5 m PointGoal radius.

All arms shared the query, causal history, initial CEC proof, NavDP checkpoint, monocular-depth stream, deterministic diffusion seed algebra, FIFO, execution budget, and three-dimensional success rule. Runtime did not receive the Revisit role. Initial CEC rejection retained exact native behavior; after a CEC acceptance, route-geometry failure stopped atomically and counted as failure.

Frozen protocol:

```text
MemNavData/hm3d_longrange_route_tangent_freeze_protocol_v2_20260903.json
SHA-256 435fe26646a8c42229230213ff70d9fc07c3d7b746b44641251d4eb4ce84227e
```

## 2. Population and evaluation contract

- all 23 result-blind eligible histories from the unused same-floor 20--30 m stratum;
- 8 HM3D scene clusters;
- scene multiplicities: `1 / 3 / 2 / 2 / 2 / 2 / 6 / 5`;
- balanced sensitivity population: at most two histories per scene, `N=15`;
- initial geodesic range: approximately `20.37--24.66 m`;
- success: full 3-D Euclidean distance strictly below `1.0 m`;
- history source: controlled causal RGB geodesic survey, not an actual NavDP Goal-A rollout;
- no metric-depth sensor, evaluator pose, Habitat path, runtime distance bin, role label, wheel odometry, or post-accept fallback reached the route reader.

The population reused scenes but excluded every history identity previously consumed by route-tangent development. It therefore tests fresh trajectories in familiar scene identities, not unseen-scene generalization.

## 3. Formal closed-loop result

| Arm | Success | SR | Mean final 3-D distance | Mean realized path |
|---|---:|---:|---:|---:|
| Mono native | 0/23 | 0.0% | 16.46 m | 18.50 m |
| CEC endpoint bearing | 0/23 | 0.0% | 13.24 m | 10.81 m |
| CEC route tangent | **4/23** | **17.39%** | 13.05 m | 9.23 m |

Primary paired comparison, route tangent versus endpoint:

- gains/losses: `+4/-0`;
- paired risk difference: `+17.39 pp`;
- exact two-sided McNemar: `p=0.125`;
- scene-cluster bootstrap 95% interval: `[+5.56,+31.82] pp`;
- balanced max-two-per-scene sensitivity: `+3/-0`, risk difference `+20.0 pp`, `p=0.25`, cluster interval `[+6.25,+37.5] pp`.

Route tangent versus native was also `+4/-0`, `p=0.125`, with a cluster interval of approximately `[+5.0,+31.82] pp`.

The preregistered confirmation rule required all of: positive risk difference, `p<=0.05`, positive cluster lower bound, and positive balanced sensitivity. Three conditions passed; the exact paired significance condition did not. The frozen decision is therefore:

```text
positive_but_underpowered_no_tuning
```

This result supports a route-geometry mechanism and justifies further work. It does **not** confirm the current implementation as a paper-ready long-range method.

## 4. Integrity and independent verification

- technical gate `16830968_0`: complete, exit `0:0`;
- remaining array `16830969_[1-22]`: 22/22 complete, every task exit `0:0`;
- aggregate `16830970`: complete, exit `0:0`;
- independent result verifier `16830971`: complete, exit `0:0`;
- frozen failure audit `16834088`: complete, exit `0:0`;
- all-discordant video renderer `16834097`: complete, exit `0:0`;
- 23 completion files and 23 completion sidecars;
- 31 run-level sidecars independently checked, zero failures;
- independent verifier: `verified=true`, with the same `0/23`, `0/23`, and `4/23` recount;
- 81 local route/population/runner/verifier/diagnostic tests passed after unsealing;
- reported path length and terminal 3-D distance match independent pose-trace recomputation to floating-point tolerance.

Remote source of truth:

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b
```

Local verified copies:

```text
.diagnostics/hm3d_longrange_route_tangent_formal_20260903/
```

## 5. What the failures identify

All 23 endpoint and tangent arms shared an accepted initial target proof. Thus this population does not diagnose target retrieval or target certification; the failures occur after the target is authorized.

### 5.1 Endpoint bearing loses route topology

Endpoint CEC failed all 23 queries: 22 terminated as stuck and one exhausted its budget. Its blocked-step fraction was `16.23%`, and its critic was below `-0.5` on `63.52%` of planning decisions. A long endpoint ray does not encode the first door, corridor, turn, or homotopy.

The four tangent gains occurred in four different scenes. Each successful run advanced to approximately `94--97%` of its reconstructed route and reached `0.977--1.000 m` in 3-D. Pre-frozen videos show the tangent arm progressively traversing a long polyline while endpoint CEC stalls or enters a different branch. The gains are not one-scene duplication or a planar cross-floor success artifact.

### 5.2 The current visual route tracker is an all-edge series system

The tangent arm stopped on a geometry-stream failure in `14/23` episodes:

- `13` failures: adjacent PnP status was not `ok`;
- `1` failure: PnP inliers fell below 16;
- one failure occurred while initially materializing the historical route;
- thirteen occurred during live query tracking after one or more valid updates.

The 13 live failures followed intervals with mean realized translation `0.120 m` and mean absolute yaw `33.23 deg`; 12/13 had at least 25 degrees of turning, and 11/13 translated at most 0.20 m. Valid intervals translated `0.213 m` on average and turned `26.10 deg` on average. This is consistent with a local-motion frontend that becomes brittle on low-parallax, turn-heavy observations. Because the current implementation requires every visual edge in a long sequence to pass, even a small per-edge failure probability compounds into poor episode coverage.

This is not evidence that the initial CEC certificate should be weakened. Initial target identity and continuous route-state estimation are different inference problems.

### 5.3 Frozen NavDP has a signed-PointGoal observability boundary

Among the nine tangent runs that did not suffer a geometry stop:

- four successes had median maximum route progress `95.35%`; only `3.93%` of their requested headings exceeded 90 degrees and `1.31%` exceeded 165 degrees;
- five stuck failures had median maximum progress approximately zero; `36.79%` of their headings exceeded 90 degrees, `25.39%` exceeded 165 degrees, and `15.54%` of post-processed PointGoals collapsed below `0.25 m`.

The NavDP PointGoal adapter clips negative forward coordinates. Therefore a valid backward local tangent is not faithfully represented to the frozen decoder. Critic score does not explain the success split: the successful group actually had a larger fraction of critic values below `-0.5` than the stuck group. The controller interface, not critic thresholding, is the more direct cause.

## 6. Visual audit

Every endpoint/tangent outcome discordance was rendered before case selection:

```text
.diagnostics/hm3d_longrange_route_tangent_formal_20260903/
  posthoc_discordant_videos/
    005_eF36g7L6Z9M_endpoint_vs_tangent.mp4
    011_dVW2D7TDctW_endpoint_vs_tangent.mp4
    013_mv2HUxq3B53_endpoint_vs_tangent.mp4
    015_BHXhpBwSMLh_endpoint_vs_tangent.mp4
```

All four MP4 hashes match the signed video manifest. The videos are explanatory sealed-trace visualizations, not additional statistical observations.

## 7. Scientific decision and next admissible step

The experiment establishes a useful distinction:

> A local historical route tangent contains control information that a long endpoint bearing discards, but the present all-edge monocular PnP tracker and signed PointGoal adapter do not deliver that information reliably enough for a confirmed long-range method.

No threshold, radius, range gate, or fallback will be tuned on these 23 consumed histories. The next development step must address the two observed failure modes directly:

1. replace the all-or-nothing pairwise route clock with one continuous, redundant route-state estimator whose prediction and visual evidence are fused rather than switched;
2. canonicalize any route tangent into NavDP's representable forward hemisphere with a deterministic bounded yaw action before diffusion planning, rather than clipping negative forward motion.

A result-dependent development diagnostic may use this consumed population to test those mechanisms, but any SR claim requires another population frozen before its outcomes are read. The canonical short/mid-range CEC paper result remains unchanged.

