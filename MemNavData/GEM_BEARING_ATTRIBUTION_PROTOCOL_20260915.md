# Initial alignment versus continued GEM bearing

2026-09-15. Local, fixed four-history mechanism experiment authorized by the
user. No new history collection, HPC, robot, manuscript or production changes.

## Question and fixed population

Does continuing to supply GEM's goal bearing improve navigation beyond the
initial bearing-directed turn followed by the original ImageGoal controller?

Use all four existing local MP3D histories, in their original order:
gxdoqLR6rwA, pLe4wQe7qrG, yqstnuAEVhm, mJXqzFtmKg4. These are already-consumed
development histories collected by metric-depth NavDP. Both query arms use
the same causal RGB prefix, goal, initial state, diffusion seed, monocular
depth and model weights. This is not a fresh end-to-end monocular population.

Two arms, one Revisit per history, eight rollouts total:

- `full_gem`: current native interval7/W64 writer, detector-support archive,
  reader-precision KV, fixed sparse reader and continued goal bearing.
- `initial_turn_only`: identical first sparse query and bounded rearward
  alignment. After the first turn finishes, disable all later sparse control
  queries and use the original ImageGoal through NavDP's native endpoint.
  Keep the GEM writer and monocular depth active. Do not reset NavDP's FIFO,
  change the goal, grant another turn, or add a fallback.

If the first cue is rejected or is not rearward, the initial-only arm uses
native ImageGoal from its first trajectory onward; no sample is removed.
The initial rearward plan is calculated in both arms but is not executed
during alignment, as in the existing adapter. Fresh turn observations enter
memory, and both arms replan immediately afterward. Subsequent turning and
trajectory changes in the full arm are part of continued bearing's effect.

Both arms use original SP/LightGlue/PnP and the manuscript's inlier/RMSE
acceptance rule (`certificate_without_coverage`), the fixed 2.5 m cue, 600
total actions, 8-step replanning and at most 4.5 degrees yaw per action.
Arm order alternates by history. No role, goal pose, simulator depth or
executor odometry enters the memory or policy; ideal low-level state is used
only by Habitat execution and evaluation. Success uses the existing planar
distance below 1 m. SPL counts translation, so also report turn/total actions.

## Required checks and reporting

Seal input, program and weight hashes before navigation. Independently
recompute actual paths, endpoints, success and action budgets. Verify all
depth transactions and causal frame indices. Pairwise, check initial
certificate outputs, geometry, selected candidate trajectories and executed
turn prefix; at the first post-turn decision, compare the RGB, original goal,
depth and sampling seed before the intervention changes the policy endpoint.
After cutoff, the initial-only arm must make no mixed-goal request, no sparse
query and no further explicit alignment. The dense writer must continue.

Keep every failure. Report four paired outcomes, terminal distances, total
actions and post-turn paths. Any equality is descriptive, not proof of
equivalence; any gain is local mechanism evidence, not a general SR claim.
This does not test memory-free scanning or establish the need for precise PnP
over a coarse historical direction. No outcome-dependent expansion or tuning.
