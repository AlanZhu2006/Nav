# Long-range Revisit attribution status — 2026-09-03

> Status correction, 2026-09-06: the 16-history formal array did not complete.
> Jobs `16817050_32..47` failed/were cancelled; no complete four-arm history
> or verified population aggregate was found in the frozen formal run.
> At least two server logs report disk-quota exhaustion. The historical
> pending descriptions below are preserved, not current queue status.
> See [latest full audit, Section 7](OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md#7-长程什么知道什么仍不知道).

## Why the action-coordinate gain is currently small

The sealed partial result is not a clean navigation improvement: among the
first 38 Revisit queries, endpoint-bearing CEC is 6/38 and action-coordinate
CEC is 7/38 (`+3/-2`, exact McNemar `p=1.0`).  The action-coordinate arm does
reduce mean final distance from 7.84 m to 5.85 m, but this average improvement
rarely crosses the strict 1.0 m success boundary.

The mechanism explains why.  Its state is accumulated executed translation
plus yaw, not a periodically corrected 2-D localization on the historical
route.  It therefore advances a route clock even when collision avoidance or
controller error moves the robot away from the route.  At 30--50 m, that
cross-track error can compound.  In addition, the reverse observed route can
be a poor current homotopy, and the frozen mixed decoder may not follow the
PointGoal when its ImageGoal condition disagrees.  Existing results do not
separate these causes.

## Decisive four-arm fork

The consumed 16-history `30_to_50_m` cohort is assigned four paired arms:

1. current action-coordinate + mixed NavDP;
2. evaluator-pose projection onto the reverse historical route + mixed NavDP;
3. current oracle geodesic direction + mixed NavDP;
4. the same oracle geodesic direction + pure PointGoal NavDP.

All three oracle arms still require the ordinary CEC proposal and certificate
to accept.  Their controller output is produced by a read-only NavDP resample
bound to the same RGB FIFO, monocular-depth transaction, and diffusion seed.
Every receipt explicitly marks evaluator pose and Habitat geometry as
privileged.  This is post-hoc attribution, not a paper result or deployable
method.

## Implementation and validation

- Pure geometry: 5/5 local tests passed.
- Read-only pure PointGoal FIFO contract: 5/5 NavDP replay tests passed.
- Initial runtime contract suite: 46/46 passed locally and on the HPC login
  stack; the lifecycle repair extends this to 48/48.
- Four complete evaluator CLI dry-runs passed inside the production
  Singularity/Python environment.
- Aggregate and independent verifier passed a synthetic 16-history end-to-end
  receipt test.
- Initial immutable source bundle:
  `hm3d_longrange_oracle_attribution_1b298b20c8d97bb9`
- Bundle receipt SHA-256:
  `1b298b20c8d97bb9b454b8f7aa90091b3c96552d444f50032bfb5d5f67bcc356`

## HPC state

- First full four-arm gate job: `16807141`, array index `32` only.
- Requested resources: `h100_tandon,a100_tandon`, one GPU, 10 CPU, 72 GiB,
  `04:00:00`.
- The job ran on A100 node `ga003` for `00:29:30` and failed closed before
  writing any completion record.  It is an infrastructure/software-lifecycle
  failure, not a navigation outcome.
- Root cause: `runtime_query(query)` binds the Revisit goal before
  `replay_prefix(frozen)`.  The first implementation reset `goal_floor` while
  binding that causal prefix; the first accepted CEC plan therefore raised
  `route oracle lacks a frozen trace/query`.  The population is Revisit-only;
  this was not a Novel-query intervention.
- Failed-attempt run root, retained unchanged:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_oracle_attribution_20260903/gate_1b298b20c8d97bb9`
- The lifecycle is now explicit: a new query invalidates all stale history and
  route state, while binding its causal prefix preserves the already-bound
  query goal.  Two regression tests cover both transitions.  The complete
  runtime suite passes `48/48` locally, the staged bundle passes `43/43`, and
  the NavDP read-only FIFO/resample suite passes `5/5` under the exact remote
  production dependency stack.
- Replacement immutable source bundle:
  `hm3d_longrange_oracle_attribution_ca97c1aa4ed3f26a`
- Replacement bundle receipt SHA-256:
  `ca97c1aa4ed3f26abe05e076345088d4727577c435d69cb60420e326ec5fed72`
- Replacement gate job: `16809384`, index `32` only.  It completed `0:0` on
  A100 node `ga004` in `01:24:46`; its sole completion and SHA sidecar verify.
- Replacement run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_oracle_attribution_20260903/gate_ca97c1aa4ed3f26a`
- All four arms start at the same 33.79 m geodesic distance, share the same
  prefix and target proof, and execute 296 accepted plans.  The one-history
  gate result is 0/1 for every arm.  Final distances are 10.80 m
  (action-coordinate mixed), 10.69 m (oracle historical-route mixed), 12.33 m
  (oracle current-geodesic mixed), and 12.29 m (oracle current-geodesic pure
  PointGoal).  This N=1 result shows the wiring is real but cannot decide the
  population-level fork.
- The measured four-arm wall time is 4,973 s, so the retained four-hour
  per-element request supplies approximately 2.9x headroom.
- The frozen 16-history diagnostic array is now job `16817050` (`32-47%3`),
  with summary `16817051` and independent verifier `16817052`.  It is pending
  on `QOSGrpGRES` at submission time.  Run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_oracle_attribution_20260903/formal_ca97c1aa4ed3f26a`.
- Because the gate outcome for overlapping index 32 was inspected in response
  to a status request before the full diagnostic array was submitted, the
  original submission receipt's generic outcome-visibility field is narrowed
  by the immutable
  `submission_outcome_visibility_amendment.json`: index 32 was known, indices
  33--47 were unread, and no source, arm, population, or denominator changed.
  The experiment remains consumed-development attribution and is not paper
  confirmation.

The older endpoint-versus-action-coordinate array currently has 46/48
hash-valid completions.  Its sealed partial count is 7/46 versus 7/46
(`+3/-3`; exact McNemar `p=1.0`), while mean final distance is 7.69 m versus
5.91 m.  Pending indices 43--47 and the old
summary/verifier were cancelled by Slurm UID 0; they produced no outcomes.
After the replacement four-arm gate had been admitted, those five indices were
resubmitted with the unchanged scientific and repair bundles as exact retry
`16809853`.  Retained index 42 and repaired indices 43--45 have completed
`0:0`; only 46 and 47 remain in flight.  Replacement summary `16810085`
depends on retained task `16807902` and the new retry; independent verifier
`16810086` depends on that summary.  The immutable repair receipt is under
`repairs/action_coordinate_admin_cancel_exact_retry_20260903/submission.json`.

## Interpretation after completion

- route oracle ≈ geodesic oracle and both high: repair route-state estimation;
- geodesic mixed > route mixed: the historical reverse route is the problem;
- geodesic point > geodesic mixed: mixed goal conditioning is the problem;
- all oracle arms weak: stop extending frozen NavDP to long-range Revisit.

No threshold or preferred outcome was introduced after observing the partial
action-coordinate result.
