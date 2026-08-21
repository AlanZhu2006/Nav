# Paper evaluation infrastructure amendment: server-port bind race

## Trigger

Attempt6 (`paper_role_pair_20260814T045433Z_attempt6`) was submitted from the
frozen source bundle with receipt
`bf0f6c5b142ef5a69b3893ed187a25b54dc8dc3997185b78bc9d1114a2197410`.
Collection task 5 failed on `gl061` before constructing its source history:

```text
Address already in use
Port 39788 is in use by another program.
ABORT: NavDP server exited
```

The exact stdout, stderr and NavDP-log hashes are recorded in
`PAPER_ATTEMPT6_PORT_RACE_INCIDENT_20260814.json`.

## Outcome-blind scope

The failure occurred during Goal-A source collection. Construction summary,
all query arms, policy summary and independent verification had not started;
the run had no `evaluation/` directory. The complete dependency chain was
cancelled and its partial collection outputs will not be reused. Attempt7 must
use a new run root and rerun all 16 collection tasks.

This amendment does not change the episode/scene population, online-history
construction, retrieval, geometry certificate, controller, five arms,
thresholds, 600-step budget, success definition or statistical analysis.

## Root cause

The old launchers performed `ss` before loading model weights. Flask bound its
port only after model initialization, leaving a check-then-bind interval in
which another process on the shared node could claim the candidate port. A
simple preflight check therefore could neither prevent the collision nor prove
that a later listener belonged to the intended server.

## Infrastructure-only correction

`retrying_server_launcher.py` is now shared by collection and evaluation:

1. skip a candidate that is already bound;
2. launch the server and wait for a listening socket owned by that exact
   process tree;
3. if the child reports an address-in-use bind failure, move through a frozen,
   deterministic port sequence and retry;
4. atomically publish the selected port and a machine-readable launcher
   receipt only after ownership is proven;
5. forward termination to the whole child process group.

MemNav and NavDP use disjoint even/odd retry sequences during paired evaluation.
The caller still fails closed for non-bind startup errors and readiness timeout.

## Verification and rerun rule

`test_retrying_server_launcher.py` deterministically simulates the TOCTOU case:
the first child exits with an address-in-use error and the second candidate
must become an owned, connectable listener. The test, Python compilation,
`bash -n`, bundle hashing, remote identity guard and all five Slurm
`--test-only` checks must pass before attempt7 submission.

No method result from attempt6 may be used to alter attempt7. The only admissible
difference between their task contracts is this server-lifecycle correction and
the resulting immutable source receipt.
