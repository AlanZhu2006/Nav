# Natural-direction role-pair readiness gate (frozen before rollout)

Date: 2026-08-14 (Asia/Shanghai)

Purpose: decide whether the role-free four-arm implementation is safe to open
the final scene-disjoint paper population.  This is an internal implementation
and plausibility gate on four already-consumed MP3D scenes.  It is not an
efficacy claim and its thresholds must not be reported as paper confirmation.

## Frozen input

- Benchmark root:
  `.diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814`
- Manifest SHA-256:
  `191473c90ab7eefff54c7ae752e2c03bc723ea3415d9b97b42c105b5e62a8848`
- Four online-A histories, one independent Novel/Revisit pair per history.
- Novel/Revisit geodesic distances are matched within 0.5 m.  Unlike the
  support-controlled diagnostic, shortest-path initial bearings are not
  matched (`180 deg` construction tolerance).
- 600 query steps, 1.0 m success radius, execution horizon 8, deterministic
  plan seeds, frozen NavDP, no oracle, U-turn, graph rescue, CDEC rescue,
  frontier, X-NavDP, or result-dependent retry.

## Frozen arms

1. `native`
2. `raw_direct` (always-on raw-DINO metric residual)
3. `raw_fixed_bearing` (same proposal projected to exactly 2.5 m)
4. `certified` (frozen certificate and fixed 2.5 m bearing; reject to native)

Arm order is balanced by scene and all arms replay the byte-identical online-A
history.  The runtime receives no `analysis_role`, co-visibility, construction
diagnostics, or goal position.

## Mechanical pass conditions

All conditions are conjunctive:

1. Four scenes produce 32 query records (4 arms x 2 roles x 4 scenes), with no
   duplicate identity, missing output, source-hash mismatch, role leak, or
   runtime failure.
2. Every `raw_fixed_bearing` takeover reports a controller radius of exactly
   2.5 m (floating tolerance `1e-6`) and the frozen adapter mode/reason.
3. On all four Novel queries, `certified` has zero certificate accepts and zero
   adapter takeovers, and its complete scored rollout equals native in success,
   steps, termination, path length and final distance (floating tolerance
   `1e-9`).
4. On all four Revisit queries, `certified` accepts and takes over at least once;
   no certificate endpoint failure occurs.
5. Both raw arms take over at least once on every query, proving that the
   ablations were actually exercised.

## Small-sample utility safeguard

Relative to the paired native arm:

- certified Novel: zero losses;
- certified Revisit: at least one gain and at most one loss.

This safeguard only blocks a clearly harmful implementation.  It is not a
significance threshold.  Passing it cannot promote the four consumed scenes to
paper evidence; failing it keeps the held-out pool sealed and triggers diagnosis
without changing the frozen certificate thresholds.

## Decision

- Pass every item: permit one-shot opening and execution of the frozen paper
  protocol in `PAPER_EVALUATION_PROTOCOL_20260814.md`.
- Fail any item: do not inspect final query outcomes and do not submit the
  paper policy array.  Diagnose only on consumed scenes; any revised method
  requires a new named protocol and a new held-out population.
