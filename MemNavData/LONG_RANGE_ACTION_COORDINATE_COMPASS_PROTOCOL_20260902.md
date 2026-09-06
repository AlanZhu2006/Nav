# Long-range Revisit action-coordinate compass protocol

Date: 2026-09-02 (Asia/Shanghai)  
Status: **frozen before reading the prospective manifest-array index 40**

## 1. Question

The endpoint-bearing stress test showed that one global bearing is inadequate
over 10--40 m routes.  The subsequent deployment audits isolated two separate
failures:

1. opposite-facing DINO observations do not reliably identify a historical
   frame;
2. a metric coordinate inferred from LingBot translation accumulates route
   pace error (5.16 m at 32.5 m), and first-64 scale recovery does not repair
   the first-40 estimate.

This protocol tests a simpler hypothesis:

> Once CEC has certified one historical target, can the already executed
> action sequence parameterize the causal monocular route, while LingBot is
> used only for its scale-free route shape?

This is an estimator and bearing-interface gate.  It is not a closed-loop SR
claim.

## 2. Frozen method

During the original traversal, each causal RGB frame is paired with the
executor's own motion receipt:

- translated distance issued/executed since the preceding frame;
- signed yaw issued/executed since the preceding frame.

These are controller efference/proprioceptive receipts, not simulator pose or
an additional exteroceptive sensor.  A real runner must record them online.
For the controlled HM3D causal survey, the same scalars are reconstructed from
the commands that generated the frozen frames.  Absolute construction poses
are separated from the method input and retained only by the scorer.

After one existing CEC proof provides the authorized historical anchor:

```text
causal LingBot route positions (arbitrary scale)
          + outbound executor action arc
          + current executor translation/yaw receipts
          -> continuous action-coordinate route state
          -> route point 2.5 m ahead in action arc
          -> normalize the LingBot displacement
          -> fixed 2.5 m PointGoal bearing + unchanged ImageGoal
          -> frozen NavDP
```

The design has one representation for every route length.  It contains:

- no short/long classifier;
- no metric scaling of LingBot translation;
- no per-frame DINO or geometry acceptance threshold;
- no stuck detector or graph search;
- no endpoint controller and no native fallback after authorization.

At the target boundary, the same vector field retains its terminal tangent
while the concurrently encoded ImageGoal supplies terminal appearance.  CEC's
initial open-set authorization remains unchanged; this protocol removes gates
and fallbacks from the long-range route controller, not from the role-free
memory safety contract.

Implementation frozen for the mechanism test:

- `MemNavData/episodic_route_filter.py::ActionCoordinateRouteCompass`;
- `MemNavData/analyze_action_coordinate_route_compass.py`.

## 3. Information boundary

The compass may consume:

- the causal LingBot pose sequence of the already observed history;
- the CEC-selected anchor from the initial proof;
- cumulative scalar translation and signed yaw receipts from the executor.

It may not consume:

- Habitat/global pose, shortest path, current source-frame label, or role;
- simulator depth or a metric LingBot scale;
- DINO similarity during route tracking;
- navigation success, distance stratum, or an accept/reject signal after
  initialization.

The audit may read construction coordinates only after the compass has emitted
its route coordinate and bearing.

## 4. Development evidence and chronology

Manifest SHA-256:

```text
cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
```

Manifest-array indices 0, 16, and 32 were already consumed while diagnosing
the failed visual/metric route estimators.  They are development mechanism
checks, never confirmation data.  Using their actual CEC-selected anchors,
the frozen action-coordinate computation obtained:

| Array index | Reverse length | Address within 1 m | Bearing within 30 deg | Median bearing error |
|---:|---:|---:|---:|---:|
| 0 | 10.2 m | 31/31 | 31/31 | 2.01 deg |
| 16 | 23.0 m | 72/72 | 60/72 | 15.16 deg |
| 32 | 32.5 m | 102/102 | 102/102 | 4.44 deg |

These values motivated the prospective test but cannot confirm it.

## 5. Prospective holdout

Before reading any content or prior method output for the next route, freeze:

- manifest array index: `40`;
- source history index: `451`;
- scene: `Qpor2mEya8F`;
- episode: `episode_table3_survey_451`;
- length stratum: `30_to_50_m`;
- history frames: `1693`;
- history trace SHA-256:
  `165fa65df002c15be263427d26cbee2229cf2a5d36500a006fa07f90b4a0cf32`;
- history receipt SHA-256:
  `991425566332a16ee6216d1e416e2cfd0f5987a872ab0f4b5716ffed47244753`.

Index 40 is the fixed upper-middle member of the remaining 16-history long
stratum (manifest-array indices 32--47).  Selection uses only manifest order
and length stratum, not prior SR, certificate quality, geometry, or route
appearance.

The existing, already frozen CEC proof supplies the authorized anchor.  Its
navigation result and the route's previous long-range outcome remain unread
until the mechanism output is sealed.

## 6. Frozen pass gate

The prospective route passes only if all conditions hold:

### Route coordinate

- at least 90% of translated readouts are within 1.0 m;
- median position error is at most 0.50 m;
- final position error is at most 1.0 m;
- emitted action progress is finite and monotone.

### Bearing interface

- at least 80% of scored bearings are within 30 degrees;
- median bearing error is at most 20 degrees;
- every emitted PointGoal has norm 2.5 m;
- the information-boundary flags verify.

If any gate fails, do not submit a long closed-loop array and do not tune on
index 40.  The next architecture must be justified from the failure mode and
tested on another untouched history.

If all gates pass, freeze the runtime bridge and run one controller smoke with
SR hidden.  Only then prepare a paired long-range comparison:

1. canonical endpoint-bearing CEC;
2. action-coordinate route CEC;

with identical histories, initial CEC proofs, ImageGoals, NavDP weights,
diffusion seeds, execution budget, and success contract.  No endpoint fallback
arm is permitted.

## 7. Paper claim boundary

Even a passed mechanism gate supports only:

> Executor action coordinates can remove monocular route-scale drift while
> preserving a scale-free bearing interface.

It does not by itself establish navigation success, robustness to real-robot
slip, or superiority to canonical CEC.  Those require paired closed loop and,
for deployment, an explicit audit of the executor receipt under collision and
slippage.
