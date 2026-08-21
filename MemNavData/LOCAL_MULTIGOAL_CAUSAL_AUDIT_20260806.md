# Local Multi-Goal Causal Audit (2026-08-06)

## Scope

This audit tests the second Novel goal in a true three-leg sequence:

```text
start -> A (Novel) -> B (Novel) -> C (Revisit)
```

It is a local development diagnostic, not the formal ten-scene result. The
machine contains two complete older three-leg scenes, with three episodes per
scene:

- `1LXtFkjw3qL`;
- `17DRP5sb8fy`.

The local NavDP checkpoint is byte-identical to the frozen HPC checkpoint:

```text
SHA256 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
```

All arms use `seed=20260803`, per-request deterministic diffusion seeds,
`exec_horizon=8`, `max_steps=600`, position success radius `1.0 m`, the same
episode files, and the same Habitat scene. Goal-A metrics are required to be
exactly identical before Goal-B results are compared.

## Interventions

Four arms were evaluated:

1. `server`: preserve NavDP's eight-frame FIFO and use the server-selected
   diffusion trajectory;
2. `reset-B`: clear only NavDP's FIFO before Goal B, preserving all other
   state and seeds;
3. `oracle-B/H8`: preserve the FIFO, but only on Goal B select the candidate
   with the smallest Habitat geodesic distance after eight simulated
   pure-pursuit steps;
4. `oracle-B/H24`: identical to the previous arm, but score each candidate
   after 24 simulated steps while still executing only eight real steps before
   replanning.

A fifth upper-bound arm was then run on the three `17DRP5sb8fy` episodes:

5. `oracle-B/H24/seed4`: the first normal request advances NavDP's FIFO once;
   three read-only resample requests reuse that exact FIFO with deterministic
   seeds, pooling `4 x 16 = 64` candidates before oracle selection.

The oracle arms are privileged causal diagnostics. They are not deployable
methods and do not change the diffusion candidate set.

## Aggregate result

Five of the six episodes reach Goal A and are eligible to evaluate Goal B.

| Arm | Goal A | Goal B given A | Joint A/B/C |
|---|---:|---:|---:|
| server | 5/6 | 4/5 | 3/6 |
| reset-B | 5/6 | 4/5 | 2/6 |
| oracle-B/H8 | 5/6 | 3/5 | 1/6 |
| oracle-B/H24 | 5/6 | 3/5 | 1/6 |

Joint success is included only for provenance. It is not a causal measure of
the Goal-B intervention because changing the Goal-B path also changes the
physical start state of Goal C.

## Per-episode Goal-B result

| Scene / episode | Server | Reset-B | Oracle H8 | Oracle H24 |
|---|---:|---:|---:|---:|
| `1L/ep0` | success, 16.722 m | success, 16.686 m | success, 9.086 m | success, 9.087 m |
| `1L/ep1` | success, 8.268 m | success, 8.444 m | success, 7.895 m | success, 7.817 m |
| `1L/ep2` | A failed | A failed | A failed | A failed |
| `17D/ep0` | success, 5.033 m | success, 5.035 m | success, 4.956 m | success, 4.957 m |
| `17D/ep1` | success, 8.494 m | failure, 11.248 m | failure, 8.971 m | failure, 9.586 m |
| `17D/ep2` | failure, 14.365 m | success, 6.629 m | failure, 8.158 m | failure, 7.692 m |

Hard reset changes behavior, but its paired gain and loss cancel. It is not a
stable correction. The short-horizon oracle makes three already-successful
episodes more efficient, but loses one server success and does not recover the
remaining failure. Extending its scoring horizon to all 24 predicted
waypoints does not change Goal-B success.

## Multi-seed candidate-pool result

Increasing the candidate pool from 16 to 64 does not recover either difficult
`17D` episode:

| Episode | H24, 16 candidates | H24, 64 candidates |
|---|---:|---:|
| `17D/ep0` | success, 4.957 m path | success, 4.917 m path |
| `17D/ep1` | failure, 5.254 m final | failure, 5.254 m final |
| `17D/ep2` | failure, 5.207 m final | failure, 5.207 m final |

For `ep2`, the minimum geodesic reached changes only from `5.932 m` to
`5.887 m`; for `ep1` it remains at the initial `5.952 m`. Thus the failure is
not explained by unlucky sampling from only 16 trajectories. Repeated seeds
from the same ImageGoal-conditioned diffusion distribution do not expose a
missing globally useful mode.

The 64-candidate rerun also records compact direction/path diversity. Circular
heading resultant `R=1` means every candidate has the same endpoint direction:

| Episode | Mean heading R | Mean maximum heading separation | Mean progress fraction |
|---|---:|---:|---:|
| `17D/ep0` success | 0.9961 | 22.17 deg | 100.0% |
| `17D/ep1` failure | 0.9834 | 34.23 deg | 15.0% |
| `17D/ep2` failure | 0.9930 | 30.59 deg | 22.7% |

Despite occasional angular outliers, nearly all probability mass remains
concentrated around one high-level direction. In the successful episode that
mode points along useful progress; in the failures the same low-diversity mode
is wrong. Four seeds therefore provide many local perturbations, not four
independent global route hypotheses.

## Candidate diagnostics

For the three oracle-success episodes, the fraction of candidates that reduce
geodesic distance is high:

- H8: approximately `81.7%` to `100%`;
- H24: approximately `85.6%` to `100%`.

For the two difficult `17D` episodes it is much lower:

- H8: `12.9%` and `12.1%`;
- H24: `21.2%` and `23.6%`.

The mean difference between the server candidate and the privileged best
candidate after the scoring horizon is also small:

- H8: `0.0010 m` to `0.0224 m` per plan;
- H24: `0.0117 m` to `0.0446 m` per plan.

On the two H24 failures, the best observed geodesic distance never improves
materially beyond the initial approximately six meters, and the final
geodesic grows to approximately `6.7 m`. Thus a longer greedy value cannot
manufacture a globally useful direction from the current candidate set.

## Metric-target upper bound (2026-08-07)

The candidate audit shows that NavDP does not sample a useful high-level
direction on the difficult Novel-B states. A second intervention separates
that failure from local collision-aware control. It preserves the image goal,
frozen NavDP checkpoint, server trajectory selector, eight-frame FIFO,
pure-pursuit executor, per-request DDPM seeds, and all Goal-A behavior, while
replacing only the metric point token on Novel-B:

1. `server-pair`: native ImageGoal NavDP;
2. `geodesic-1.25`: at every replan, Habitat's privileged shortest path is
   truncated 1.25 m ahead and transformed to NavDP `[forward, left]`;
3. `final-point`: the same code uses a 100 m truncation distance, which clamps
   to the final GT endpoint. This supplies the exact relative goal point but
   no intermediate shortest-path turn.

All three arms ran sequentially against the same live NavDP process. For every
episode, `reached_A`, `steps_A`, `len_A`, and `final_dist_A` are exactly equal
across arms. Five episodes reach A:

| Arm | Goal B given A | Mean final B distance | Mean path over B successes | Mean SPL over B successes |
|---|---:|---:|---:|---:|
| native server pair | 3/5 | 2.113 m | 10.151 m | 0.764 |
| geodesic 1.25 m | 5/5 | 0.981 m | 6.669 m | 0.977 |
| exact final point | 5/5 | 0.979 m | 7.834 m | 0.886 |

The per-episode paths make the recovery explicit:

| Scene / episode | Native server | Geodesic 1.25 m | Exact final point |
|---|---:|---:|---:|
| `17D/ep0` | success, 5.034 m | success, 5.063 m | success, 5.384 m |
| `17D/ep1` | failure, 21.929 m | success, 6.857 m | success, 7.679 m |
| `17D/ep2` | failure, 21.938 m | success, 5.496 m | success, 6.661 m |
| `1L/ep0` | success, 15.889 m | success, 8.850 m | success, 8.944 m |
| `1L/ep1` | success, 9.531 m | success, 7.080 m | success, 10.502 m |
| `1L/ep2` | A failed | A failed | A failed |

Therefore A* is not required for Goal-B success on this small set: the exact
final metric point also reaches all five goals. Short geodesic subgoals do,
however, reduce the mean successful path by 14.9% relative to the exact final
point and improve SPL. The decisive missing signal is a stable long-horizon
metric target/direction; shortest-path structure is a secondary efficiency
gain. Once that signal is provided, frozen NavDP is a competent local
collision-aware controller.

This is still a privileged upper bound. It uses the unknown GT location of a
Novel goal and perfect Habitat localization at every replan. LingBot can
provide a metric point directly only after a goal is localized in memory. For
a truly Novel goal, a deployable system must instead use persistent geometry
to propose frontiers or topological subgoals until visual evidence turns it
into a localized revisit.

Goal-C and joint SR are deliberately not interpreted here: changing the B
route changes the physical B arrival position/yaw and therefore changes C's
initial state. A causal C comparison requires replaying a frozen B trace or
canonicalizing its terminal state.

The native baseline in this new live process is `3/5`, whereas the earlier
reset/selector process produced `4/5`. Request-level seeds reproduce all arms
exactly within one live server, but the current CUDA stack is not guaranteed
bitwise identical across fresh processes. Formal paired arms must therefore
remain in one pinned server process, as the benchmark runner already does.

## Conclusion

The local evidence rejects three simple fixes:

1. uniformly clearing temporal state at every goal switch;
2. replacing NavDP's selector with a myopic or 24-step geodesic critic;
3. increasing the same conditional diffusion pool from 16 to 64 candidates.

The metric-target upper bound then positively identifies the missing level of
control. In difficult states, most diffusion candidates share an unhelpful
local direction, but the same frozen local controller succeeds when it is
given a correct long-horizon metric target. The server critic cannot choose a
globally correct route that the candidate set does not contain.

The most justified next architecture is therefore:

```text
LingBot coverage / pose graph
        + explicit Novel posterior
        -> image-goal-conditioned frontier or graph-subgoal proposal
        -> diverse short-horizon waypoint candidates
        -> frozen NavDP collision-aware local control
        -> switch to memory localization and reverse graph once a match appears
```

This uses LingBot not only after a revisit has already been recognized, but
also as persistent exploration state for Novel goals. A smaller secondary
direction is a goal-conditioned temporal adapter that learns selective FIFO
retention; the hard-reset audit shows that such retention must be adaptive,
not a constant rule.

Before training either component, the formal ten-scene development set should
repeat the server and B-only oracle arms. The primary metrics are Goal-B SR,
candidate progress fraction, heading diversity, minimum geodesic reached, and
paired changes under identical Goal-A traces and diffusion seeds.

## Goal-blind observed-frontier residual (2026-08-07)

The metric-target upper bound motivates a deployable question: can persistent
observed geometry supply a useful Novel-goal direction without receiving the
unknown goal coordinate? A local diagnostic now builds a sparse two-dimensional
map from only the depth images and poses available along the executed rollout.
It ray-carves free space, marks occupied endpoints, extracts free/unknown
boundaries, and ranks connected frontiers by boundary size, travel distance,
and distance from already visited cells. No goal coordinate is accepted by the
frontier API.

This implementation is nevertheless privileged. It projects depth with the
Habitat evaluator pose and uses the Habitat navmesh to reject unreachable
frontiers and produce a 1.25 m subgoal along the route to the selected
frontier. It is a feasibility intervention, not a deployable LingBot result.

An always-on arm established both sides of the intervention. It rescued one
`17D` failure, but it also destroyed both previously successful `1L` Goal-B
runs. Therefore unconditional coverage is rejected: a Novel ImageGoal is not
equivalent to generic exploration.

A shadow run then measured the distance from every native ImageGoal trajectory
endpoint to the visited trace. In the five Goal-B episodes whose Goal A was
reached, the three native successes never proposed an endpoint closer than
approximately 0.89 m to the visited trace. The two failures spent 35% and 39%
of their plans below 0.60 m, with consecutive runs of 16 and 10 plans. This is
a useful stagnation signal, but it is not yet evidence that an arbitrary
frontier is the correct recovery direction.

The residual arm therefore preserves the native ImageGoal proposal by default.
Only after three consecutive native endpoints are less than 0.60 m from the
visited trace does it latch a frontier target. The 0.60 m threshold is twice
the fixed 0.30 m agent radius and is also the map's pre-existing minimum
frontier novelty; it was not selected from Goal-B coordinates. At an active
plan, the evaluator first runs the normal ImageGoal request, advancing NavDP's
eight-frame FIFO exactly once, and then uses a read-only mixed image/point-goal
resampling endpoint with the identical diffusion seed. Unit tests verify that
this second proposal does not change FIFO length or contents.

The final comparison was rerun sequentially against one live NavDP process.
This constraint is important because fresh CUDA processes produced different
closed-loop outcomes even with request-level diffusion seeds. Goal-A path,
steps, success and final distance are exactly equal for every paired episode.

| Scene / episode | Native ImageGoal | Coverage residual | Interpretation |
|---|---:|---:|---|
| `17D/ep0` | success, 5.033 m | success, 5.033 m | no trigger; bit-exact |
| `17D/ep1` | success, 8.494 m | failure, 20.037 m; final 8.830 m | harmful frontier |
| `17D/ep2` | failure, 14.365 m; final 9.729 m | success, 12.211 m | rescued by frontier |
| `1L/ep0` | success, 16.722 m | success, 16.722 m | no trigger; bit-exact |
| `1L/ep1` | success, 8.268 m | success, 8.268 m | no trigger; bit-exact |
| `1L/ep2` | Goal A failed | Goal A failed | excluded from B-given-A |

Thus both arms obtain Goal-B `4/5`. The residual swaps which difficult episode
succeeds and provides no net SR improvement. It is a **No-Go for promotion or
large-scale training** in its current form. The negative result is still
diagnostic: repeated-space endpoint novelty can detect a class of native
ImageGoal failures, while a goal-blind coverage score cannot determine which
branch is relevant to the requested image.

The next justified component is not another fixed trigger threshold. It is an
ImageGoal-conditioned frontier or graph-subgoal ranker. Its inputs should be
the goal image, persistent LingBot keyframe/patch features, candidate frontier
geometry, pose uncertainty, and the native waypoint; its supervised target can
be geodesic progress during training, but evaluation must remove the Habitat
goal coordinate and pathfinder. Native ImageGoal remains the default, and a
ranked subgoal is accepted only when both native repetition and a calibrated
goal-conditioned advantage are present. In parallel, replacing evaluator pose
with aligned LingBot pose/depth will measure how much of the privileged-map
upper bound survives real long-term memory noise.
