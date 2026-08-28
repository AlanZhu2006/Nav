# HM3D Table-1 fresh-query reserve protocol

Status: **frozen construction-only gate**. This document does not authorize a
navigation result.

## Why this population is needed

The prior HM3D controller-portability population contains 28 actual-online
histories. Its ViNT anchor-only outcomes and five selected losses have already
been inspected, so a physical bearing executor cannot be promoted to a formal
method result by rerunning those same queries.

The parent run contains 130 materialized Goal-A histories. The old builder
attempted all 130 and retained 28. The remaining 102 are not an untouched copy
of the old mixed-role benchmark: 52 lacked its single preassigned Natural-Novel
direction and 50 lacked both a standard Revisit and that Novel query. They may
therefore be reused only through a newly frozen, construction-only protocol.

## Frozen construction

All 28 previously consumed `(scene, episode)` identities are excluded before
query construction. For every remaining history, the builder searches for:

- one standard-support Revisit query with online-A co-visibility in
  `[0.55, 0.90]`;
- one unsupported Novel query with online-A co-visibility `< 0.10`;
- both queries at geodesic distance `2--9 m` from the same online-A endpoint.

The earlier benchmark forced each history into one preassigned
front/side/rear direction bin. That requirement served a direction-stratified
analysis, not CEC safety or controller portability, and caused structural
attrition. The reserve protocol tries that same preferred bin first, then the
other two bins in a deterministic identity-bound order. It retains the first
structurally valid Novel query and records the selected bin. This change was
frozen after inspecting constructibility receipts but without opening any
navigation outcome for the new query identities.

## Power and stopping gate

Scene prefixes are inspected only at `30, 36, 42, 48, 54`. The smallest prefix
meeting all of the following is sealed:

- at least 24 histories;
- at least 15 scene clusters;
- at least four Novel queries in each of front, side and rear strata.

If the full 54-scene construction misses any target, the result is sealed as
underpowered and **no policy rollout is submitted**. Construction counts are
never interpreted as SR.

## Evaluation authorized only after a passing verifier

If and only if the construction verifier passes the power gate, the same
sealed queries may support four arms:

```text
NavDP native       vs NavDP + CEC
ViNT native        vs ViNT + CEC bounded bearing
```

Within each controller, native and CEC use the same checkpoint, query, causal
history, seed, step budget, loaded process and order-balanced paired rollout.
ViNT rejection is exact native ViNT. Acceptance consumes the certified bearing
through physical zero-translation turns of at most 30 degrees, with one fresh
observation and one controller/history update per turn, before unchanged ViNT
replans on the verified historical anchor.

## Claim boundary

The population can support a **fresh-query, scene-overlap controller
portability** claim. It cannot be called fresh-scene generalization. The older
28 queries and five inspected losses remain mechanism evidence only.
