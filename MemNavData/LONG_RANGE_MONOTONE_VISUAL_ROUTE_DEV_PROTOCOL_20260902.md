# Long-range Revisit: monotone visual-route development protocol

Frozen: 2026-09-02 (Asia/Shanghai)  
Status: frozen after the RGB mechanism gate and before any closed-loop outcome  
Scope: consumed HM3D length population; development evidence only

## 1. Question

Canonical CEC proves that a goal image corresponds to causal history and sends
the endpoint's unit bearing at a fixed `2.5 m` residual to frozen NavDP. This is
strong on the main 2--9 m regime but a single endpoint chord loses route
topology on non-convex 20--40 m returns. Increasing the residual or restoring
estimated endpoint magnitude did not fix that stress population.

The frozen question is therefore:

> After one target certificate authorizes memory, can the robot follow the
> temporally ordered visual route by continuously re-localizing its route
> coordinate, while retaining one frozen policy and one fixed local bearing
> interface?

## 2. Method fixed before closed-loop evaluation

### 2.1 Proof once

The existing strict CEC target certificate is evaluated exactly once. An
accept freezes the causal tail-to-anchor route. A target rejection retains the
canonical mixed-role behavior and executes the unchanged native ImageGoal
request. Runtime receives no Novel/Revisit label.

### 2.2 Track one monotone route coordinate

During the original causal traversal, the existing LingBot stream writes:

- causal RGB and DINO features for every frame;
- camera poses from the same stream;
- CPU-resident depth/confidence on every eighth frame.

The sparse depth writer reuses the short-range stream. It does not instantiate
a second LingBot KV cache and cannot change the initial target certificate.

For each later RGB observation, proposal is restricted to the still-unvisited
part of the one authorized temporal route. DINO supplies a temporally diverse
top-8 shortlist; LightGlue evidence selects an address; historical sparse
depth and PnP produce a local current-pose witness. This witness observes
state only. It does not re-authorize memory and does not apply the operational
target-certificate thresholds a second time.

Let `gamma(s)` be the frozen route, `a_t` the visually localized historical
address, `s(a_t)` its immutable arc coordinate, and `rho = 2.5 m`:

```text
s_t^- = max(s_(t-1), s(a_t))
s_t   = local_projection(x_t^PnP; [s_t^-, s_t^- + rho])
q_t   = gamma(min(s_t + rho, S))
b_t   = unit(R_t^T [q_t - x_t^PnP])
p_t   = rho b_t
```

Appearance addressing may advance farther than one controller horizon;
control never does. This separation is essential: accumulated monocular route
scale must not become a localization window. The policy receives only the
unit bearing at the same fixed residual used by canonical CEC.

### 2.3 Explicit exclusions

After a target accept there is:

- no short/long classifier;
- no distance-dependent branch;
- no endpoint-bearing fallback;
- no native fallback;
- no stuck trigger or graph rescue;
- no alternate policy or planner;
- no evaluator pose, geodesic path, or runtime role input.

A missing local pose is an explicit geometry-stream failure. If route progress
completes before the benchmark position-success event, execution also fails
explicitly rather than switching methods.

## 3. Frozen local mechanism evidence

The corrected same-height query construction uses one already-consumed medium
history (index 16) only as a mechanism development case. Twenty-four RGB views
cover the route in return order; construction poses remain analysis-only.

The first deployable cache attempt that copied a second dense LingBot KV stream
was stopped before target proof because GPU memory reached `39.4 GB` by frame
388 and continued growing. It produced no method result.

The next attempt coupled visual localization to the next `2.5 m` internal arc
window. It updated 8/24 observations, then failed with `34.43 m` remaining.
Selected anchors increasingly lagged the rendered positions. This diagnosed a
scale-coupling error: a physical movement below one controller horizon can
exceed that horizon in accumulated monocular route arc.

The frozen method above decouples the horizons. On the same 24 queries it
passed all preregistered checks:

- route-coordinate updates: `24/24`;
- anchor-error median: `0.0553 m`;
- anchor-error maximum: `0.5405 m`;
- final remaining route: `0.8828 m` from an initial `48.9812 m`;
- monotone progress: `24/24` transitions;
- finite active bearing: every active update;
- endpoint fallback / native fallback / distance gate / local authority:
  all absent.

Result SHA-256:

```text
2cccd935155ffc5ac0bf1167db82815cec528bb26c2ff0c77936a1f81ff16c94
```

This is a mechanism result, not navigation SR.

## 4. Outcome-blind HPC deployment gate

Before reading any closed-loop success outcome, run exactly one frozen history
from each consumed distance stratum: indices `0`, `16`, and `32`. Each array
element executes the complete hidden Novel/Revisit query pair for at most 80
query steps. The gate reads only runtime receipts and process status.

It passes only if:

1. the hidden Novel query has zero target-certificate accepts;
2. the Revisit target certificate accepts and freezes one route;
3. at least one later plan contains a visual route-coordinate update;
4. every attempted route update succeeds and progress never regresses;
5. every active output is finite;
6. no receipt exposes endpoint fallback, native fallback after authorization,
   a distance gate, evaluator pose, Habitat path, or role label;
7. both servers and the evaluator exit cleanly.

Navigation success is not read or summarized by this gate. A failure stops the
formal development comparison and creates a new immutable repair attempt; it
does not authorize threshold changes.

## 5. Consumed-population paired development comparison

Only after all three deployment gates pass, evaluate all 48 sealed histories
with three same-process arms:

1. frozen mono NavDP native;
2. canonical strict CEC with fixed endpoint bearing;
3. strict CEC with the monotone visual route readout above.

The three arms share the exact online history, target image, initial target
certificate contract, NavDP checkpoint, depth stream, seed algebra, success
distance, step budget, and action executor. Arm order is balanced. No episode
is filtered by runtime acceptance or outcome.

Primary comparison: route CEC versus endpoint CEC, paired Revisit SR over all
48 histories and separately in the three frozen distance strata. Report
gain/loss counts, exact McNemar p-value, scene-cluster bootstrap confidence
interval, SPL, path length, collisions, runtime failures, route update count,
and per-query latency. Native is context, not the primary comparator.

This consumed population may authorize a fresh confirmation only if all of the
following are met without changing the frozen method:

- route CEC has at least six more gains than losses versus endpoint CEC;
- gains are at least twice losses;
- both 20--30 m and 30--50 m strata have positive paired net gain;
- the 0--20 m stratum loses no more than two episodes;
- no accepted route silently switches to endpoint or native guidance;
- geometry/runtime failure rate is at most 5%.

These are development gates, not paper significance claims.

## 6. Claim boundary

Even a favorable 48-history development result is not confirmation because
the length population and index-16 mechanism case were already consumed. A
paper claim requires a new disjoint long-range population frozen after this
protocol, followed by paired evaluation and independent recount. Until then,
the confirmed paper method remains canonical fixed-bearing CEC and this branch
is reported as a prospective long-range extension.

