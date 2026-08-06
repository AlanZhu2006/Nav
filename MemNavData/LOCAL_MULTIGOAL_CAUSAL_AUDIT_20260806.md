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

## Conclusion

The local evidence rejects three simple fixes:

1. uniformly clearing temporal state at every goal switch;
2. replacing NavDP's selector with a myopic or 24-step geodesic critic;
3. increasing the same conditional diffusion pool from 16 to 64 candidates.

The dominant failure is closer to candidate generation and long-horizon
exploration under a Novel/no-match goal. In difficult states, most diffusion
candidates share an unhelpful local direction. The server critic cannot choose
a globally correct route that the candidate set does not contain.

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
