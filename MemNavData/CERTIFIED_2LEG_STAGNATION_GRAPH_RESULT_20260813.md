# Certified graph rescue on 2-leg Revisit: consumed internal gate

Date: 2026-08-13

## Question and scope

Question: can the stagnation-triggered history-graph rescue that recovered all
three failures in the actual-online 3-leg `fresh20` benchmark also recover the
remaining failures of the stronger 2-leg certified Revisit controller?

This is a **post-outcome mechanism gate** on the already consumed `fresh160`
set. It is neither a new population success-rate estimate nor paper
confirmation. The frozen cohort contains:

- all five Goal-B failures where Goal A succeeded, the certificate accepted,
  and certified direct bearing terminated `stuck`;
- one same-scene accepted certified success control for each failure;
- three arms per episode: frozen direct, equal extra-budget control, and graph
  rescue;
- balanced arm order, the same online-A trace and deterministic plan seeds;
- exact causal-plan, physical-rollout and memory-prefix checks up to treatment.

The full raw run is under
`.diagnostics/certified_2leg_graph_gate_20260813`. The machine-readable report
is `report.json`, SHA256
`8198552af84f5044f6225b4bf8bbac38c38edddd95513333cbb7c70c3ffbfced`.

## Frozen result

| cohort / arm | direct | equal budget | graph rescue |
|---|---:|---:|---:|
| five known accepted-stuck failures | 0/5 | 0/5 | **1/5** |
| five matched successful controls | 5/5 | 5/5 | 5/5 |

Primary graph-minus-budget contrast on the five failures:

- paired `+1/-0`;
- exact two-sided McNemar `p=1.0`;
- controls had zero interventions, zero losses, and byte-identical causal
  plans and physical frames across all three arms.

If one merely inserts this observed single rescue into the consumed
`fresh160` certified denominator, the descriptive counterfactual is
`112/120 -> 113/120`, only `+0.83 pp`. This was not a new 120-episode rerun and
must not be presented as a formal SR result.

Per-failure audit:

| scene / episode | direct final m | budget final m | rescue | rescue final m | actual history-subgoal plans | first route nodes |
|---|---:|---:|---|---:|---:|---:|
| `e9zR4mvMWw7/episode_0003` | 2.030 | 4.138 | fail | 3.976 | 28 | 3 |
| `mJXqzFtmKg4/episode_0002` | 2.500 | 2.545 | fail | 2.545 | **0** | 0 |
| `oLBMNvg9in8/episode_0007` | 2.334 | 2.330 | **success** | **0.993** | 13 | 3 |
| `rPc6DW4iMge/episode_0000` | 2.566 | 2.566 | fail | 4.599 | 36 | 11 |
| `qoiz87JEwZ2/episode_0005` | 4.189 | 4.184 | fail | 4.184 | 18 | 3 |

## What this establishes

The mechanism is structurally portable and safely gated, but its present
implementation is **not an effective general 2-leg residual**. The trigger did
not damage any successful control; the weakness appears only after activation.
Running the remaining 110 successful/unsupported episodes or launching a large
HPC expansion would therefore not repair the missing effect.

This does not invalidate the 3-leg result. The two cohorts isolate different
failure classes:

- in `fresh20` 3-leg, the three accepted-bearing failures had small bearing
  error and historical arcs of 12, 4 and 8 nodes; equal budget rescued `0/3`
  while the graph rescued `3/3`. They were consistent with a direct-chord /
  homotopy failure;
- in 2-leg, only one of five accepted-stuck failures was rescued. The remaining
  failures are heterogeneous and cannot all be called topology failures.

## Execution-grounded diagnosis

The post-outcome Habitat audit is diagnostic only and was never policy input.
It reveals two concrete architectural mismatches.

### 1. Route-start localization can be wrong after prolonged stagnation

In `mJXqzFtmKg4/episode_0002`, the server's LingBot pose selected the target
anchor itself as the nearest route-start node, producing an empty route and
falling back to direct bearing for all 29 post-trigger plans. At that first
request, the physical robot was still **2.341 m** from the historical target
anchor. Thus a late rescue cannot safely infer its graph start from monocular
pose proximity alone.

The old diagnostic field also marked this empty-route direct fallback as
`certified_graph_rescue_active=true`. The field has been corrected so
`active` now means that a historical subgoal was actually returned. This is a
logging correction only; it does not alter any recorded action or outcome.

### 2. A history graph needs a waypoint executor, not only a bearing adapter

For the four episodes that really emitted historical subgoals, the first
subgoal's angular errors were only `0.29`, `1.96`, `4.56`, and `7.82 deg`.
Initial graph direction selection was therefore usually accurate. Yet only one
episode succeeded.

The reason is visible in the controller contract. The graph constructs a
metric sequence of nearby history nodes, but `verified_bearing_v1` discards
each node's magnitude and normalizes every command to the endpoint adapter's
fixed **2.5 m** radius. The median pre-normalization norms in the four episodes
were `0.951`, `1.641`, `1.219`, and `1.903`; the resulting fixed-radius rays can
overshoot corners or repeatedly push into the same obstacle. This mismatch is
an evidence-supported hypothesis, not yet a causal counterfactual.

One case also shows pose drift during recovery: in
`e9zR4mvMWw7/episode_0003`, the first historical-subgoal error was `0.29 deg`
but the request-level median rose to `78.0 deg` (maximum `170.8 deg`). By
contrast, `rPc6DW4iMge` and `qoiz87JEwZ2` retained median direction errors of
only `6.29` and `7.69 deg` and still failed, confirming that bearing accuracy
alone does not explain the residual failures.

## Decision

Do **not** expand the current 2-leg stagnation rescue to HPC and do not add it
to the 2-leg headline method. Retain the frozen 3-leg graph result as an
internal, topology-specific mechanism pending scene-disjoint confirmation.

If 2-leg graph navigation is revisited, the principled replacement is a
certified topological follower:

1. visually certify or snap the robot's current history node; never choose the
   late route start solely by LingBot nearest-pose distance;
2. execute the history arc with a graph-specific, distance-aware PointGoal
   contract (or a fixed radius tied to graph spacing), rather than the endpoint
   bearing adapter's unconditional 2.5 m projection;
3. fail closed to certified direct/native when current-node certification or
   waypoint progress is unsupported;
4. freeze this contract before a scene-disjoint full-set comparison, including
   all successful controls.

An even cleaner future design would choose the certified history arc at the
goal switch, while the route start is still causally known, instead of waiting
150 stagnant steps and then trying to re-localize onto the route. That is a new
method hypothesis and has not been validated by this gate.

## Verification

- 30/30 arm runs completed (10 episodes x 3 arms);
- every direct outcome and termination class reproduced the source;
- causal prefixes passed for all treated failures;
- all five controls were exact no-ops;
- an independent raw JSON/plan recount reproduced
  `failures=5, rescue=1, budget=0, gain=1, loss=0, actual-graph=4,
  empty-route=1, controls=5, control-interventions=0, control-losses=0`;
- graph and summarizer tests: `23 passed`;
- Python compilation audit passed.
