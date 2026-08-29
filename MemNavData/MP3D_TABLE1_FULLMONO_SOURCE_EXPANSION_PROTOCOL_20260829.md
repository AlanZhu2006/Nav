# MP3D Table-1 full-monocular source expansion

## Frozen trigger

The first outcome-blind MP3D construction retained 14 histories from 10 scene
clusters, with front/side/rear counts 5/0/9.  Its independent verifier passed,
but the prospective 20-history, 12-scene, four-per-direction power gate did
not.  No controller rollout was submitted and no new query navigation outcome
exists.  This constructibility failure is the sole trigger for this expansion.

The query thresholds, success definition, controller, certificate, direction
assignment and power gate are unchanged.  In particular, this protocol does
not lower co-visibility thresholds or hand-select side-view cases.

## Why the old phase-2 traces are not reused

The 2026-08-14 phase-2 population fixed 16 disjoint MP3D scenes and
`episode_0002` through `episode_0005` in every scene before its old query
outcomes existed.  Those identities are therefore a legitimate prospective
source reserve.  Its recorded Goal-A trajectories used metric controller
depth, however, whereas the conference Table 1 requires the same full-
monocular policy boundary used by the HM3D rows and the first MP3D source
population.

This expansion re-runs only Goal A for those already-frozen source identities
with frozen NavDP and causal LingBot monocular depth.  It does not reuse the
old metric trajectory and does not run a query arm during source collection.

## Outcome-blind exclusion of consumed queries

Old Attempt-7 and phase-2 navigation outcomes are never opened by the source
collector or query constructor.  Because their query images have nevertheless
been consumed, every old natural-direction and support-controlled manifest is
hashed into a forbidden-identity ledger.  A newly rendered Novel or Revisit
query is rejected if either its JPEG SHA-256 or its pose-and-yaw identity
matches:

- the source episode's original Goal B; or
- any old Attempt-7/phase-2 query in the same scene.

This is stricter than excluding only the original Goal B and prevents a
deterministic constructor from silently recreating a previously evaluated
query.

## Full-monocular Goal-A contract

- frozen NavDP checkpoint and frozen LingBot state;
- one causal RGB stream, no simulator-depth read;
- first-40 scale receipt and frame-bound monocular depth transaction;
- 500-step budget, horizon 8, 1 m success radius;
- deterministic base seed 20260803 plus frozen episode rank;
- all 64 sources retained before Goal-A success or constructibility attrition;
- source collection produces no Novel/Revisit query outcome.

## Construction and release gate

The 20-scene base source ledger and the 16-scene expansion ledger are combined
without deduplication by performance.  All 36 scenes are passed through the
same new-query constructor.  Formal NavDP/ViNT rollout remains forbidden until
an independent construction verifier reproduces all file hashes, confirms
zero consumed-query overlap, and observes at least:

- 20 retained histories;
- 12 scene clusters;
- four Novel queries in each of front, side and rear strata.

Failure remains a valid constructibility null.  It does not authorize another
threshold change or adaptive source selection.
