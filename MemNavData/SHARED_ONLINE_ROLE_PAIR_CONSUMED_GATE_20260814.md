# Four-arm consumed role-pair gate — frozen 2026-08-14

## Scope and timing

This gate governs whether the current role-free Certified Episodic Compass
implementation is ready for a one-shot scene-disjoint paper evaluation.  The
input is the four already-consumed matched Novel/Revisit pairs in
`.diagnostics/shared_online_role_pair_heading30_v3_smoke_20260814`, evaluated
for at most 600 steps with the four frozen arms `native`, `raw_direct`,
`raw_fixed_bearing`, and `certified`.

This document was frozen after the first scene's `native` and `raw_direct`
arms had completed, but before that scene's `raw_fixed_bearing` and
`certified` arms and before all arms for the other three scenes.  Therefore it
is an **internal readiness gate**, not a fully preregistered scientific result.
No threshold, arm, certificate, controller, construction rule, or exclusion
rule may be changed from this point in response to these outcomes.

## Required validity checks (all must pass)

1. Exactly four scene/episode identities, one matched Novel/Revisit pair per
   identity, four arms, and 32 arm-query records are present.
2. Every arm has exact online-A replay hashes, zero replay diffusion samples,
   paired query/seed/geodesic identity, and no role/covisibility field visible
   to runtime.
3. There are zero runtime-failure plans in every arm.
4. Every `raw_fixed_bearing` takeover reports adapter mode
   `raw_fixed_bearing_v1`, reason `raw_uncertified_fixed_bearing`, and both
   controller radius fields equal to 2.5 m within `1e-6`.

## Safety/liveness gate (all must pass)

1. **Novel fail-closed:** certified has zero certificate accepts and zero
   adapter takeovers on all four Novel queries.  Its executed rollout trace,
   success, steps, path length, and final distance are exactly equal to native.
2. **Revisit liveness:** certified has at least one certificate accept and at
   least one adapter takeover on each of the four Revisit queries.
3. **Internal utility sanity:** certified has no Novel loss relative to native;
   on Revisit it has at least one paired gain and no more than one paired loss.
   This is not a significance test and cannot be reported as paper evidence.
4. Both always-on raw arms take over at least once on every query.  This proves
   that the raw-versus-certified contrast actually exercises coverage versus
   abstention rather than comparing two inert implementations.

## Decision

- If every check passes, freeze the exact source bundle and analysis code and
  authorize one-shot paper evaluation.  Final outcomes cannot feed back into
  architecture or thresholds.
- If any check fails, do not open final-reserved scenes.  Diagnose only on the
  consumed pool or create a new development protocol; the failing result must
  remain in the ledger.

The paper evaluation must report role-stratified SR/SPL, paired gain/loss and
exact McNemar, scene-cluster bootstrap intervals, certificate risk/coverage,
false-takeover plan and episode rates, compute cost, and exact-fallback audit.
Balanced mixed SR is secondary; it must not hide Novel harm or Revisit utility.

