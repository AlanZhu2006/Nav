# Long-range CEC causal route filter

Date: 2026-09-02 (Asia/Shanghai)  
Status: **mechanism protocol frozen while the index-32 holdout was running**

## Question

The failed episodic path-field gate established that neither a hard
opposite-view relocalization at every replan nor unconstrained LingBot
odometry is a valid long-range state estimator.  This protocol tests the
remaining minimal hypothesis:

> After CEC certifies a historical target once, can the robot maintain one
> continuous coordinate on the authorized temporal RGB route by combining a
> causal motion prior with soft DINO observations?

This is a route-state mechanism test, not a navigation SR result.

## Frozen estimator

The authorized route contains every causal history address from the live
history tail to the certificate-selected target anchor, in reverse temporal
order.  State zero is known exactly at the goal switch.

For one translational controller update, the expected temporal advance is
derived separately from each route:

1. use only LingBot poses already produced from the causal RGB history;
2. for lags 1--64, compute the mean route chord length;
3. select the lag whose mean chord is closest to 0.30 m, the frozen simulator
   execution contract (`8 * 0.0376 m`, rounded only for this diagnostic);
4. set transition sigma to half the IQR of local first-passage lags;
5. cap one-step support at mean plus four sigma.

Each DINO cosine vector is standardized over the authorized route and added
as a soft log potential.  There is no similarity threshold.  The max-product
state update is one-way and bounded by the physical transition support.  A
known in-place turn has zero route advance; action class is controller
efference, not an evaluator pose.

The implementation is:

- `MemNavData/episodic_route_filter.py`;
- `MemNavData/run_local_causal_route_filter_audit.py`.

## Information boundary

Runtime may consume:

- causal history RGB and its frozen LingBot pose receipt;
- current RGB;
- the controller's own previous action class (turn or translation);
- the target anchor and route already authorized by the initial CEC proof.

Runtime may not consume:

- Habitat pose, shortest path, source-frame identity, or Novel/Revisit role;
- a short/long distance class;
- a per-observation accept/reject decision;
- endpoint-bearing or native-policy fallback.

Construction pose and source-frame identity are read only after inference to
measure route-address error.  No controller, success detector, or SR is run.

## Population and chronology

The three immutable histories were selected before this estimator:

| Index | Length bin | Role |
|---:|---|---|
| 0 | 0--20 m | first transfer check |
| 16 | 20--30 m | consumed design route |
| 32 | 30--50 m | final held-out mechanism route |

Index 16 was already consumed while diagnosing the failed hard-reanchor
design.  Index 0 completed before this written protocol and is therefore a
transfer check, not a prospective confirmation.  The following gate was
written before reading index 32.

## Frozen gate

Index 32 passes only if all conditions hold:

- at least 80% of reverse-route readouts are within 1.0 m of the hidden
  evaluator route coordinate;
- median error is at most 1.0 m;
- final error is at most 1.0 m;
- emitted state never regresses or exceeds one frozen transition support;
- all information-boundary flags verify.

If the gate fails, no closed-loop array is submitted.  If it passes, the next
test is not another bearing-radius sweep.  It is a controller bridge gate:

```text
one initial CEC proof
        -> continuous authorized route coordinate
        -> one receding historical ImageGoal on that same route
        -> unchanged frozen NavDP local controller
```

The historical ImageGoal is a local route reference, not a second target
retrieval or a discrete fallback.  The route coordinate updates continuously;
the reference saturates naturally at the certified target endpoint.  A
controller bridge must be validated on the same three histories before any
48-episode paired SR job is authorized.

The lower-change controller bridge is tested first: maintain the camera frame
from the known history tail using only accumulated executor yaw, read a 2.5 m
lookahead tangent from the filtered route, and pass that unit tangent through
CEC's unchanged mixed ImageGoal/PointGoal interface.  On index 32 it must have
at least 80% of scored bearings within 30 degrees and median error no greater
than 20 degrees.  It holds its last finite tangent at the route boundary;
there is no endpoint mode or fallback.  If this bearing bridge fails, only
then test receding historical ImageGoal substitution as the controller bridge.

## Results available at freeze time

- index 16 (consumed design route): 64/72 within 1 m with the final bounded
  one-way implementation, median 0.37 m, final 0.26 m;
- index 0 (transfer check): 31/31 within 1 m, median 0.15 m, final 0.30 m;
- index 32: running and unread at protocol freeze.

These numbers establish neither long-range navigation success nor superiority
to canonical 2.5 m CEC.  They only determine whether a deployable continuous
route coordinate exists strongly enough to justify testing its control
interface.
