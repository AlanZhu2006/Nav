# Novel Memory-Direction Causal Control Protocol

**Frozen:** 2026-08-15, before constructing or running the fresh population  
**Status:** protocol only; no result exists yet  
**Purpose:** explain the Phase-2 Novel gains of the always-on raw-DINO
fixed-bearing ablation without promoting an incidental intervention into a
localization claim.

## 1. Question

On Phase-2 natural-direction queries, raw fixed bearing succeeded on `9/19`
Novel queries while native and Certified Episodic Compass (CEC) each succeeded
on `4/19`. The goal images in this stratum have no supported Revisit in the
causal online-A history. The extra successes could therefore arise from at
least three different mechanisms:

1. the factual history contains weak but goal-relevant contextual direction;
2. the factual history merely sends the agent back through useful explored
   space; or
3. any direction perturbation can change frozen NavDP's stochastic exploration
   and occasionally help.

This experiment distinguishes those explanations. It does **not** tune CEC and
does not test a deployable Novel module.

## 2. Fresh population

- Use MP3D scenes disjoint from train40, Attempt 7, Phase-2, and every scene
  used to design this protocol.
- Freeze the source list and every construction seed before reading any arm
  outcome.
- Target at least 40 episodes from at least 20 scene clusters. If construction
  yields less, report the achieved denominator and label the result
  underpowered; do not silently relax the population contract.
- Every query starts from the endpoint of a successful, actual-online frozen
  NavDP A rollout.
- A Novel query must satisfy the existing role-pair contract:
  `max_online_a_covis < 0.10`, measured across the full causal online-A RGB
  history, and the frozen geodesic/budget/navmesh constraints.
- Runtime projection removes `analysis_role`, co-visibility, goal position,
  source identity, and construction diagnostics. They remain sidecar-only for
  stratified scoring.

This population must not reuse Attempt 7 or Phase-2 as confirmation because
their outcomes have already been read.

## 3. Four paired arms

All arms share the same scene, start pose, goal image, action budget, execution
horizon, NavDP checkpoint, causal NavDP observation FIFO, and diffusion seed
schedule. Arm order is balanced by a frozen Latin square.

### A. `native`

Unchanged frozen ImageGoal NavDP. No episodic direction is sent.

### B. `raw_factual_history`

Replay the factual online-A RGB stream into the memory sidecar and the factual
online-A decision RGBs into NavDP's FIFO. Use the existing raw-DINO/pose route
and `raw_fixed_bearing_v1`: whenever it produces a finite non-zero proposal,
normalize it and send a `2.5 m` mixed ImageGoal+PointGoal residual. There is no
certificate and no role label.

### C. `raw_deranged_history`

Keep NavDP's FIFO factual, but replay a different frozen online-A stream into
the memory sidecar. The donor is chosen by a precomputed, no-fixed-point
derangement within the same scene whenever possible; otherwise use a donor
from the same MP3D split matched by decision-frame count. Donor selection may
use scene ID and history length, but never the goal image, outcomes, goal pose,
or retrieval scores.

The complete raw retrieval, pose, validity, fixed-radius adapter, and fallback
path remains unchanged. This isolates whether the *particular factual
trajectory* carries useful information while preventing the donor history from
altering NavDP's own short observation FIFO.

### D. `raw_randomized_bearing`

Replay the factual online-A history and run the complete raw proposal path.
Conditional on that path producing a finite non-zero proposal, discard only
its angle and replace it with a deterministic angle sampled uniformly from
`[-pi, pi)` using a frozen hash of
`(global_seed, scene, episode, plan_index, "random_bearing")`. Send the same
`2.5 m` residual through the same mixed controller. If raw proposal generation
is unavailable, use the identical native fallback.

This preserves proposal availability and adapter semantics on that arm's own
trajectory while destroying directional content. It is preferable to an
always-on random intervention, which would confound direction quality with
intervention frequency.

## 4. Non-negotiable pairing and audit fields

Each record must contain:

- scene, episode, arm and balanced arm-order index;
- source and donor history receipts and hashes;
- factual NavDP FIFO hashes, separately from sidecar memory hashes;
- episode seed and every requested/echoed diffusion plan seed;
- raw proposal availability, raw unit bearing, randomized unit bearing, and
  fixed radius;
- takeover count, fallback count, path length, steps, final geodesic distance,
  SR, and SPL;
- proof that `analysis_role` and construction diagnostics were not forwarded;
- immutable source bundle, manifest, checkpoint, and dependency receipts.

The random-bearing implementation must have unit tests for deterministic
replay, angle range, zero/invalid proposal fallback, and preservation of the
`2.5 m` radius. The derangement must be audited for no fixed points and no
goal/outcome-dependent assignment.

## 5. Frozen analysis

Primary paired contrasts, reported in this order:

1. `raw_factual_history` versus `raw_randomized_bearing`;
2. `raw_factual_history` versus `raw_deranged_history`;
3. each intervention arm versus `native`.

For every contrast report both success totals and paired `gain/loss`, exact
two-sided McNemar p-value, paired risk difference, and scene-cluster bootstrap
95% confidence interval. Also report takeover coverage and intervention count;
aggregate SR alone is insufficient.

No arm, seed, threshold, donor mapping, or population rule may change after the
first closed-loop outcome is read. A runtime smoke may verify contracts but
must use scenes excluded from the formal population.

## 6. Interpretation fixed in advance

- Factual `> random` and factual `> deranged`: evidence that the causal history
  supplies goal-relevant or trajectory-relevant direction beyond a generic
  perturbation. This still does not make the goal a geometrically supported
  Revisit.
- Factual `~= random` and factual `~= deranged`: Phase-2 Novel gains are best
  interpreted as exploration perturbation. They cannot support a Novel
  localization claim; CEC's exact fallback is the principled method behavior.
- Deranged `> random` but factual `~= deranged`: environment/history priors may
  help, but instance-specific episodic addressing is not established.
- All intervention arms lose to native: unverified memory is harmful on Novel
  queries, directly supporting abstention.
- Mixed gains and losses: report the risk--coverage trade-off without selecting
  the favorable aggregate post hoc.

Regardless of outcome, this control cannot establish that CEC improves Novel
navigation: CEC intentionally abstains when no historical hypothesis is
certified. A deployable Novel direction source remains separate future work.
