# ViNT + CEC bounded bearing-consumption protocol

Status: implementation/preflight only.  No formal SR is authorized by this
document.

## Question

Can a frozen ViNT controller consume an accepted CEC proof without replacing
ViNT by NavDP and without the ideal instantaneous yaw used in the five-loss
mechanism audit?

## Adapter

```text
accepted, hash-bound CEC bearing
  -> discard the pre-turn ViNT horizon
  -> turn in place by at most 30 degrees
  -> render one fresh RGB observation
  -> write that observation exactly once to causal memory/controller state
  -> repeat until the certified signed turn is exhausted
  -> render a final fresh observation
  -> run unchanged ViNT on the verified historical anchor ImageGoal
```

The adapter never consumes Habitat goal pose, geodesic bearing, runtime
Novel/Revisit role, NavDP trajectory, or CEC translation magnitude.  Every
turn receipt records the proof packet hash, input-frame hash, monotonic memory
frame index, yaw transition, zero translation, and remaining angle.

Normal CEC rejection executes byte-identical native ViNT.  An invalid proof,
nonfinite turn, nonmonotonic observation receipt, or exhausted step budget
fails closed.  The initial CEC-bearing decision is evidence/authorization only;
its pre-turn ViNT trajectory is never executed.

The held-out authority-pair runner now closes this interface explicitly with
`PORTABILITY_CEC_ACCEPT_ALIGNMENT=first_certified_bounded`. Only the `grant`
arm receives that evaluator mode; `forced_reject_native` is pinned to `off`.
The runner rejects this mode unless the controller is ViNT, the query protocol
is complete-population `paper_heldout`, the reject branch is
`controller_native_exact`, and no outcome-selected query manifest is supplied.
Its v3 authority receipt records both arm modes. The per-cell auditor then
reconstructs the signed turn from the proof packet and verifies every
zero-translation action, fresh-observation hash, monotonic memory index, and
the summed turn angle. This closes the implementation path but does not by
itself authorize or constitute a formal SR result.

## Why the previous 28-history result cannot be reused

The anchor-only formal run and its five selected losses have already been
inspected.  The ideal-yaw result therefore establishes root cause and mechanism
only.  It cannot be upgraded by rerunning the same outcome-aware subset with a
more physical executor.

Before any formal bounded-executor result, freeze a new outcome-blind
population with:

- causal actual-online RGB history;
- balanced Natural Novel/Revisit queries;
- no selection using ViNT, CEC, or adapter outcomes;
- identical ViNT checkpoint, goal, seed, budget and observation condition in
  native and CEC arms;
- same-process arm pairing with order balance;
- episode-level path, SPL, accept/reject, proof and turn receipts;
- independent raw-distance and action-contract verification.

Until such a population is sealed, the controller-portability row remains
missing rather than filled by the ideal five-loss diagnostic.
