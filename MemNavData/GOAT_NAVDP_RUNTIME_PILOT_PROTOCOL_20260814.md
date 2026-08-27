# GOAT native-NavDP ten-scene runtime pilot

Date: 2026-08-14 (Asia/Shanghai)

## Question and boundary

The official two-scene simulator/task contract has passed on HPC.  The next
question is narrower than navigation quality: can the frozen native NavDP
server consume the official GOAT Stretch RGB-D stream and raw ImageGoal image,
produce deterministic metric trajectories, and execute them through the
frozen 0.25 m / 30 degree discrete adapter without an interface or runtime
failure?

This pilot executes only the first ImageGoal subtask of one official episode
from each of ten `val_unseen` scenes.  It does not provide ObjectGoal or
LanguageGoal controllers and does not finish full sequential episodes.
Consequently:

- `is_goat_navigation_score=false`;
- no aggregate from this pilot is a paper SR/SPL result;
- outcome or success diagnostics cannot select a threshold, controller, or
  method variant;
- the pilot may determine only runtime feasibility, formal sharding and
  whether a contract bug blocks the next stage.

## Frozen population

Eligible episodes begin with an ImageGoal.  Scene and within-scene episode are
selected without navigation outcomes using SHA-256 rank and salt
`goat-runtime-pilot-v1`.  The exact ten pairs, selection rule, seed and action
guard are stored in
`MemNavData/goat_navdp_runtime_pilot_manifest.json` (SHA-256
`652cbe0f731c3b817e9c1e0f5e516ae4f386d74380a7ed06c4910651357b5db5`).

## Frozen execution contract

- Official GOAT commit:
  `74c41d19d4a4c3608d1575b512087b5a529aee0e`.
- Official HM3D `val v0.2` scenes and GOAT episode order.
- Official Stretch RGB sensor and an added co-located metric-depth sensor used
  only because frozen NavDP requires RGB-D.
- Goal pixels are freshly rendered from the episode's official
  `InstanceImageParameters`; cached CLIP embeddings are not passed to NavDP.
- Native NavDP checkpoint SHA-256:
  `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`.
- Request-specific diffusion seeds are derived from the frozen base seed,
  scene, episode and plan index, so episode length cannot shift later noise.
- NavDP cumulative local metric waypoints are converted by the already frozen
  adapter: 0.25 m forward, 30 degree turns, lookahead 4, execution horizon 8.
- A near-zero endpoint within 0.20 m emits official `SUBTASK_STOP`.  No
  ground-truth distance auto-stop is allowed.
- At 600 navigation actions, a guard emits `SUBTASK_STOP` only to finalize and
  audit official metric state.  It is logged as `forced_guard_stop`, never as
  an autonomous policy stop.

The official camera is portrait and has a different field of view from the
MP3D camera used by the project.  The runner passes the pixels unrotated and
computes the true intrinsic from the official RGB dimensions/HFOV.  This
domain shift is part of the external runtime test, not a parameter to correct
after viewing ten outcomes.

## Runtime gate

The gate passes when all ten frozen pairs finish, all ten scene/pathfinder and
goal-render contracts remain valid, every NavDP response has a finite
`[K,3]` trajectory and every environment action is one of the frozen official
actions.  Wall time, request latency, Slurm MaxRSS and allocated-GPU memory are
used only to choose formal scene sharding.

If zero of ten episodes emit an autonomous stop, the full run remains blocked
pending a code-level stop-contract audit; this would indicate a likely
systematic adapter mismatch.  Any smaller number of action-guard failures is
retained as native-policy behavior and cannot trigger threshold tuning.

After this gate, the next honest stages are:

1. reproduce the official released GOAT reference policy/metrics;
2. freeze a shared controller for non-image subtasks;
3. run complete sequential episodes with paired
   `hybrid_navdp_native`/`hybrid_navdp_certified` ImageGoal arms.

