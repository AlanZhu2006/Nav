# Semantic-Proposal Gate-B Task-18 Infrastructure Incident

**Recorded:** 2026-08-15 (before any Task-18 repair execution)  
**Scope:** consumed method-development Gate B; never confirmation evidence  
**Affected population index:** `18` only  
**Original array job:** `15763485`; Slurm task job: `15764627`

## Observed failure

Task 18 ran on `gh133` from `2026-08-14T20:43:02` to
`2026-08-14T21:02:14` (HPC local time) and ended `FAILED`, exit `6:0`.
The Habitat evaluation process aborted in its native runtime:

```text
Aborted (core dumped) ... eval_shared_online_role_pairs.py
```

This was not an OOM or wall-time failure: the batch step used approximately
`15,538,064 KiB` of the requested `96 GiB` and ran for `00:19:12` against an
`08:00:00` limit. Both policy servers had launched and served requests before
the simulator process aborted.

## Why it is not an experimental outcome

The task produced only the first arm's startup/evaluation log. It produced no
`completion.json`, no second-arm directory, and no paired Gate-B record. No
Task-18 method outcome was read. It therefore cannot be counted as either a
success or a failure under the frozen Gate-B rule: that rule applies runtime
failure accounting to a *completed paired arm record*, whereas this was an
unpaired native-process crash before a record existed.

At incident time, 23 other population indices had valid completion records;
none will be changed or rerun.

## Frozen repair authorization

Exactly one repair is allowed:

1. preserve the complete partial Task-18 directory, its buffer, server logs,
   Slurm logs, and hashes under the formal run's `failed_attempts/` tree;
2. rerun population index `18` only, against the same immutable source bundle,
   population manifest, causal prefix, arm-order rule, seeds, controller,
   thresholds, `600`-step budget, and output root;
3. exclude `gh133`, the node on which the native simulator aborted;
4. do not rerun or overwrite any completed population index;
5. aggregate only after all 28 unique completion records exist, then run the
   independent verifier over raw completions.

Excluding the failed node is an infrastructure repair, not a method change.
The paired arms of the repaired task must still execute on the same allocated
node and process environment. The frozen promotion rule remains unchanged:
semantic-first is nominated for fresh confirmation only when its paired gains
strictly exceed its losses; a tie or net loss retains geometry-first CEC.

## Repair submission receipt

- preserved incident root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/semantic_proposal_gate_b_20260815/formal_consumed_final_20260814T232857Z/failed_attempts/task018_job15764627_sigabrt_gh133`;
- archived files: `203`; archived bytes: `15,759,678`;
- `INCIDENT_ARTIFACTS.sha256` SHA-256:
  `b85f5e110dae618203321ed259e8305cf1cf2c339aa829230260ffd71471115d`;
- exact-index repair: `15764888_18`, constrained to
  `h100_tandon,a100_tandon` because this cluster rejects explicit node
  exclusion;
- repair-aware summary: `15764892`, dependent on both completion of original
  array `15763485` and successful repair `15764888`;
- independent verifier: `15764893`, dependent on `15764892`.

The original summary/verifier jobs `15763486` and `15763487` were cancelled
before execution because they did not depend on the exact-index repair and
therefore could not safely aggregate the then-incomplete 27-record root. This
replacement changes only failure recovery and aggregation dependencies; it
does not change the frozen experimental contract.

## Repair completion

Repair `15764888_18` ran on H100 node `gh014` and completed in `00:02:59`, exit
`0:0`. The formal root then contained exactly 28 unique completion records with
population indices `0..27`. Summary `15764892` and verifier `15764893` both
completed, and the verifier returned `verified=true`. Task 18 therefore entered
the final analysis only through its repaired paired completion; the archived
unpaired crash artifacts remain outside `records/` and outside every outcome
denominator.
