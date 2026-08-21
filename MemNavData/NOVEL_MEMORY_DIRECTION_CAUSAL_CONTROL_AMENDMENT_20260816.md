# Novel Memory-Direction Causal Control: Scene-Budget Amendment

**Frozen:** 2026-08-16, before any deranged-history or randomized-bearing
closed-loop outcome exists.  This amendment changes population status only;
it does not change an arm, controller, radius, seed rule, threshold, or outcome
definition in `NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL_20260815.md`.

## 1. Trigger

The original protocol requested at least 40 episodes from at least 20 fresh
MP3D scene clusters.  A complete read-only inventory of the now-available
official 90-scene MP3D asset set proves that population is not constructible:

| partition | scene clusters |
|---|---:|
| train40 | 40 |
| consumed development pool | 20 |
| consumed Attempt-7/Phase-2 blind pool | 16 |
| untouched | 14 |

The four sets are pairwise disjoint and exhaust all 90 scenes.  The immutable
receipt is
`.diagnostics/mp3d_scene_budget_20260816/scene_budget.json`, SHA-256
`779e2d7d63faa0f9b9e735680b1d620f04428c11a57ac83158933306b62407ef`.

Changing the fresh-scene target from 20 to 14 and consuming all remaining
scenes for a post-hoc mechanism question would leave no prospective population
for the final method confirmation.  That would improve neither validity nor
paper readiness.

## 2. Frozen allocation

### Stage D: consumed-development causal mechanism

The four-arm Novel control will first run only on the already consumed
Phase-2 role-pair population:

1. native;
2. raw factual history;
3. raw deranged sidecar history with factual NavDP FIFO;
4. raw factual history with deterministic randomized bearing.

This stage may explain the previously observed Phase-2 raw-DINO Novel gains.
It is explicitly post-hoc/consumed development evidence:

- `confirmation_claim_allowed=false`;
- it cannot establish generalization;
- it cannot promote a new method or retune CEC;
- all four arms must be rerun in the same process/machine pairing because old
  Phase-2 outcomes cannot be mixed with new CUDA trajectories.

### Stage F: untouched final confirmation

All 14 untouched scenes are reserved, unread, for one prospective role-free
mixed Novel/Revisit confirmation.  Their identities are sealed in the scene
budget receipt.  No Novel-control smoke, donor, query, or outcome may use
them.  The final protocol must be frozen separately before generating or
reading any policy outcome in those scenes.

## 3. Unchanged causal contracts

- The deranged arm changes only the RGB stream replayed to the long-term
  MemNav sidecar.  NavDP receives the factual decision-frame FIFO.
- Donors form a no-fixed-point permutation.  Same-scene donors are preferred;
  remaining donors are matched without goal, score, pose, or outcome access.
- The randomized arm runs the complete raw proposal path.  If and only if a
  finite non-zero raw proposal exists, its angle is replaced by a SHA-256
  deterministic angle in `[-pi, pi)`; the 2.5 m radius and intervention
  availability are unchanged.
- Novel/Revisit role, co-visibility, goal pose, donor identity, and all
  construction diagnostics remain evaluator-side only.
- Every contrast reports paired gains/losses and intervention coverage.  Stage
  D results are never pooled with Attempt 7 or Phase-2 as confirmation.

## 4. Decision after Stage D

- factual approximately equal to randomized and deranged: treat previous raw
  Novel gains as exploration perturbation, supporting CEC abstention;
- factual greater than both controls: report weak history-specific context and
  acknowledge that strict certification trades coverage for authorization;
- all interventions worse than native: direct evidence for exact fallback;
- mixed outcomes: report the risk--coverage surface without selecting a
  favorable aggregate post hoc.

No Stage-D outcome authorizes opening the 14 final scenes until the final
mixed-role manifest, arms, support strata, statistics, and stop conditions are
independently frozen.

## 5. Runtime gate without a leaked smoke scene

The original protocol permits a runtime smoke only on a scene excluded from
the formal population.  The 14 untouched scenes are reserved for Stage F, so
none may be spent on a Stage-D smoke.  Stage D therefore has no discardable
closed-loop smoke.

Instead, all local unit tests, immutable-file checks and Slurm `--test-only`
checks run before submission.  Formal episode index 0 then runs first as a
staged **formal** record.  Only an `afterok` dependency releases indices 1..N;
index 0 remains in the denominator, and neither source nor protocol may change
after its outcome exists.  A failure terminates that immutable attempt and
requires a separately documented repair attempt; it cannot be silently
discarded as smoke.
