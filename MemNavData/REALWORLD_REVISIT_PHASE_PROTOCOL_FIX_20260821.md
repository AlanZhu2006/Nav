# Real-world revisit phase protocol fix (hub protocol v3)

Date: 2026-08-21.  Trigger: failed tethered B->A revisit trial
(`deployment/go2/goals/results/20260821_135437_revisit.json`); robot e-stopped
safely, assist distance grew 2.81 m -> ~6.07 m, CEC returned
`no_causal_candidate` on every plan.

## Root cause

The protocol-v2 hub exposed only goal-conditioned stepping
(`/imagegoal_step`), so the Goal-A image was queried from frame 0.  MemNav
freezes the goal session at the first query (`goal_start_frame=0`,
`candidate_ceiling=-1`), which excludes the entire A->B history from Revisit
candidacy for the rest of the episode.  The system silently degraded to plain
ImageGoal exploration.  A post-hoc diagnostic that reopened a fresh Goal-A
session over the intact 613-frame memory produced 8 candidates, an accepted
certificate (anchor frame 39, ~3.99 s), and a valid bearing -- proving memory
and pipeline were fine and only the session ordering was wrong.

This is the same bug class fixed in the simulation harness on 2026-08-18
(leg A must run without opening a goal session; cf. the
`native_sidecar`-vs-`phase` hybrid-route fix in the CEC+mono composition
evaluator).  The deployment hub must mirror the simulator's two-phase
contract.

## Fix (MemNavData/realworld_cec_hub.py, protocol v2 -> v3)

Explicit two-phase episode contract, enforced server-side:

1. `POST /navigator_reset` -> phase `memory_recording`, `frames_recorded=0`.
2. `POST /memory_step` (file: `image`) -- record-only append to the shared
   LingBot stream via MemNav `/memory_step`; no goal, no retrieval, no plan.
   Rejected outside the recording phase.  Transport failure fails closed
   (`memory_degraded` latch + `reset_required`), same as the query-phase
   stream contract.
3. `POST /begin_revisit` -- requires >=1 recorded frame; switches to phase
   `revisit_query` and records `revisit_started_after_frame`.
4. `POST /imagegoal_step` -- now REJECTED (HTTP 400) during recording with an
   explanatory error; legal only after `/begin_revisit`, so the first goal
   query freezes `goal_start_frame` at the revisit start point with the full
   recorded history eligible.

`/healthz` and the reset response now report `phase`, `frames_recorded`, and
`revisit_started_after_frame` for operator audit.

## Verification

- `python -m pytest MemNavData/test_realworld_cec_hub.py`: 11 passed,
  including new tests for: goal query rejected during recording (router and
  HTTP, with zero upstream traffic), memory_step counting and post-switch
  rejection, begin_revisit frame precondition, memory_step fail-closed latch,
  and the full HTTP happy path reset -> blocked query -> memory_step ->
  begin_revisit -> imagegoal_step.
- `py_compile` clean.

## Addendum: goal-candidate capture and weak-covis scoring

Two further gaps versus the simulator protocol were identified and closed at
the tooling level the same day:

- `POST /goal_candidate` (recording phase only): registers a goal-candidate
  photo that is explicitly NOT appended to the memory stream, with capture
  receipt (`captured_after_frame`, sha256, optional file via
  `--goal-candidate-dir`).  Mirrors the simulator rule that the revisit goal
  is constructed from the walk but excluded from memory.  Zero upstream
  traffic; rejected after `begin_revisit`.
- `MemNavData/score_realworld_revisit_goal.py`: scores candidates against
  the recorded `rgb_dir/{idx}.jpg` frames using only frozen server
  components -- strided stateless DINO cosine sweep
  (`/imagegoal_similarity`) plus LightGlue verification of the argmax frame
  (`/retrieval_verify`, non-mutating).  Reports max_cos / argmax /
  gap-from-end / inliers and a PROVISIONAL band
  (`min_inliers>=16`, `max_cos<=0.90` upper bound against near-duplicate
  goals).  The simulator's GT-covis bands (standard `[0.55,0.90]`, hard
  `[0.25,0.55)`) are recorded in the report as reference semantics only; the
  proxy thresholds must be calibrated on the disabled-adapter walk before
  any band label is treated as final.

The NavDP observation-FIFO divergence is now CLOSED rather than accepted:
`begin_revisit` replays a stride-8 tail of the recorded frames (up to 8
frames, chronological, mirroring the simulator's one-plan-per-8-steps
shared-trace replay) through NavDP's `/memory_replay_step`, and hard-verifies
the resulting `queue_lengths` exactly as `replay_shared_leg1` does.  Any
warm-up transport failure or queue-length mismatch latches
`native_state_uncertain` (partial warm-up leaves NavDP state unknown) and
requires a reset.  The hub retains a rolling 64-frame tail buffer for this;
warm-up receipts (`navdp_warmup_frames`, frame indices, queue lengths) are
returned by `/begin_revisit`.  Note the stride matches the simulator in
frame counts; wall-clock equivalence depends on the client's memory_step
rate.

## Not covered by this change (still open before retest)

1. Jetson/Go2 adapter must be updated to drive the new flow
   (A reset -> `/memory_step` during A->B -> `/begin_revisit` at B -> goal
   queries) and synced to the RTX hub checkout plus the release repo; the
   old adapter will now fail fast at the first goal query instead of
   silently exploring, which is the intended safe behavior.
2. Existing physical gates are unchanged: measured camera optical-center
   height, camera-only disabled-adapter smoke, frame-40 receipt audit,
   left/right bearing sign check, tunnel/MemNav fault injection.
3. First-certificate latency at the revisit switch (~4 s observed in the
   diagnostic) must be absorbed by the stale-plan/watchdog stop behavior --
   the robot should be stationary when `/begin_revisit` is issued.
