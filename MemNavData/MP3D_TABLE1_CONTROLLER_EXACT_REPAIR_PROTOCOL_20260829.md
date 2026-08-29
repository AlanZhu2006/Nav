# MP3D Table-1 controller exact infrastructure repair

Date frozen: 2026-08-29 22:20 Asia/Shanghai
Status: prospective exact repair; no estimator, threshold, method, or population change

## 1. Frozen scientific contract

The independently verified MP3D Table-1 population remains fixed at 42 causal
histories from 25 scene clusters, with one Natural Novel and one Revisit query
per history (84 queries total). Its benchmark manifest SHA-256 is
`a33f210fdd0cfa84e82c4d403ac79056dcc7959cd1ce84bf62bec8c5632deb69`.
The construction verifier SHA-256 is
`618c409f7c7c62ad739687935cdd6f2e564e96aed6ccf6059d887d795c3e953e`.

This repair does not change the query images, actual-monocular histories,
checkpoints, DINO/LightGlue/PnP implementation, certificate thresholds, fixed
2.5 m residual, controller inputs, arm order, seeds, 600-step budget, 1 m
success radius, hidden runtime role, or controller-specific paired estimands.

## 2. Result-independent missing set

The original NavDP formal array `16548405` has 40/42 canonical completion
receipts. Array rank 27 failed `1:0`; the two missing histories are exactly:

- `29`: `029_kEZ7cmS4wCh_episode_0004` (partial directory exists);
- `30`: `030_kEZ7cmS4wCh_episode_0005` (never started).

The replacement ViNT formal array `16548592` has 41/42 canonical pair-audit
receipts. Index 24 failed `2:0`; the only missing history is:

- `24`: `024_pRbA3pwrgk9_episode_0004` (startup-only partial directory exists).

The repair set is therefore uniquely defined by missing canonical receipts and
the failed Slurm indices. No success/failure value is used to select a repair
cell. The 40 completed NavDP histories and 41 completed ViNT histories remain
immutable and are not replayed.

## 3. Failure classification and minimal corrections

### 3.1 NavDP: identical JPEG under a new causal transaction

The failed rank stopped when a stationary agent produced byte-identical JPEGs
at different causal stream positions:

```text
RuntimeError: cached monocular depth belongs to a different transaction
```

The active 2026-08-21 server overlay keyed its one-frame cache by JPEG digest
and rejected the newer valid transaction token. The repository's already
verified 2026-08-22 fix treats this case as a cache miss, refetches by the exact
new token, and still fails closed for unknown or mismatched tokens. The repair
reuses the HM3D-verified immutable overlay:

```text
hm3d_table1_navdp_cache_repair_2ae34ad0c1503958
```

Receipt SHA-256:
`2ae34ad0c150395849d4461913fc086f3b6ea7acf7249c763fe3e8808356ed6d`.
Only scene rank 27 is relaunched, with an explicit scene-confined history
override `29,30` and a new runtime-attempt namespace.

### 3.2 ViNT: shared-node TCP port collision

The failed cell stopped during server startup with:

```text
Address already in use
ABORT: memnav exited during startup
```

No pair audit was produced. The old PID-modulo allocator had a time-of-check to
time-of-use race on shared GPU nodes. The replacement wrapper claims one
consecutive six-port block with a node-local `flock`, checks all six live
listeners, and holds the lock for the cell lifetime. Controller code and
scientific factors are unchanged. Only history index 24 is relaunched.

## 4. Preservation and disclosure

Before either exact retry, the two partial canonical directories are moved—not
deleted—into a new additive repair archive. Every archived file is hashed and
the archive is made read-only. The old NavDP rank-27 runtime logs remain intact;
the repaired runtime writes under a distinct attempt name.

During post-failure infrastructure diagnosis, one already-completed history's
outcome was incidentally printed from a runtime log. This was not a planned
analysis, was not used to classify either failure, and did not influence the
missing set, code correction, method, or downstream decision. This disclosure
replaces any blanket claim that no partial outcome was ever visible. No partial
SR, aggregate, threshold comparison, or failed-cell navigation estimator was
computed.

## 5. Replacement DAG and opening rule

```text
NavDP exact retry: scene rank 27, histories 29 and 30 only
  -> replacement NavDP aggregate -> independent raw verifier

ViNT exact retry: history 24 only
  -> replacement ViNT aggregate -> independent raw verifier

both independent verifiers -> MP3D joint controller seal
```

All original aggregate, verifier, and seal jobs were dependency-cancelled and
produced no canonical outputs. New downstream jobs write the original missing
summary/verifier/seal paths with exclusive-create guards. Results may be opened
only after both independent verifiers and the joint seal complete successfully.
