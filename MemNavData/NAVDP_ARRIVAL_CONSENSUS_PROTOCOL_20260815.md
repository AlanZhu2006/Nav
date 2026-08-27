# NavDP candidate-consensus arrival audit

Date: 2026-08-15 (Asia/Shanghai)

## Question

The GOAT runtime pilot showed that mapping a selected near-zero NavDP path
directly to `SUBTASK_STOP` is invalid: 9/10 autonomous stops occurred 2.56--
14.87 m from the goal.  NavDP clamps every sampled endpoint shorter than 0.5 m
to zero before critic ranking, so one selected zero may be a short diffusion
candidate rather than semantic arrival.

This train-only audit asks whether repeated candidate sets supply a deployable
proposal signal: does true arrival produce cross-candidate and cross-seed
contraction, while isolated false-zero selections do not?

## Frozen data and sampling

- Input universe: the existing 40 train scenes and 80 expert 3-leg episodes
  named by the frozen train40 candidate table.  Development, final-reserved,
  GOAT validation outcomes, and blind data are not read.
- Each B/C goal is evaluated only inside its active trajectory segment.
- At most one deterministic state is selected from each fixed Euclidean band:
  `<=0.25`, `(0.25,0.5]`, `(0.5,1]`, `(1,2]`, `(2,4]`, and `>4` m.
- Each state receives its causal within-leg seven-frame context, one normal
  NavDP query, and three read-only resamples with independently hashed seeds.
- The frozen NavDP checkpoint and `stop_threshold=-0.5` match the GOAT runtime
  pilot.  No weights are changed.

## Logged signals

For every seed the audit records selected endpoint length, critic range, the
fraction of candidate endpoints clamped to zero, the top-four zero fraction,
and the best-zero versus best-nonzero critic margin.  Per-state aggregation
records cross-seed persistence.

Ground-truth distance is joined only after inference.  The report evaluates a
predeclared grid of consensus operating points and AUCs; it does not select or
authorize a threshold.

## Decision boundary

- If no high-precision region exists, policy consensus is rejected and the
  next candidate is an independent current-to-goal LightGlue/LingBot-PnP
  arrival certificate.
- If a high-precision region exists across many train scenes, its rule must be
  frozen before a fresh, disjoint GOAT confirmation.  The original ten-scene
  pilot remains mechanism/debug data and cannot choose that rule.
- Until such confirmation, zero trajectory means `abstain/replan`, never
  `SUBTASK_STOP`.

## Frozen HPC execution

Submitted at run tag `20260814T171428Z` (UTC):

- immutable audit source:
  `/scratch/yz11502/Research/source_bundles/navdp_arrival_consensus_921ad26e291c1794`
- source receipt SHA-256:
  `921ad26e291c17940b46b57e3e5146d2be4d1ca6c8a92a75b248fb03be59b141`
- one-episode structural smoke: Slurm `15753017`
- full 40-scene / 80-episode formal audit: Slurm `15753043`, with
  `afterok:15753017`
- smoke result root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/navdp_arrival_consensus_20260815/smoke_20260814T171428Z`
- formal result root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/navdp_arrival_consensus_20260815/formal_20260814T171428Z`

The immutable submission records explicitly set
`goat_validation_read=false` and `method_or_threshold_authorized=false`.
Their SHA-256 values are respectively
`a8e60dad910dab401dade19d7cf860acc04064aa0d93907734ea879a39cff78f`
and
`bddf92f19c776cb45f1b05a4a0887df3b7ac61390cd7bdc2b750700b2024ea83`.

### Post-submission smoke amendment

The frozen prose incorrectly called each NavDP response a set of 64
candidates.  Smoke `15753017` verified the checkpoint's actual contract:
`predict_imagegoal_action(..., sample_num=16)` returns 16 candidates per
query.  The audit issues four independent queries per state (one normal query
and three read-only resamples), so it observes 64 sampled trajectories per
state in four groups of 16.  The collector itself is shape-generic and logs
`candidate_count=16`; no inference record or operating point depends on the
incorrect prose count.  The formal run therefore remains valid as a
four-seed, 16-candidate-per-seed audit.  This amendment does not modify the
immutable submitted bundle.

The corrected independent smoke verification passed with 1 scene, 1 episode,
2 goals, 12 states, 48 unique-seed queries, 16 candidates per query, and 36
read-only resamples.  The sealed `report.json` SHA-256 is
`b4509a11f1302d00641678ad51215914357f90e162c2442686efd18b3e12ae96`.

### Formal decision gate frozen before reading formal outputs

At `2026-08-14T17:26:49Z`, while formal job `15753043` was still collecting,
the direct-STOP gate was fixed as follows.  At least one point in the already
predeclared persistence/candidate-fraction grid must have:

1. zero false accepts over all non-arrival states, including the
   `(0.25, 0.50]` near-miss band;
2. at least 20 true accepts; and
3. true accepts spanning at least 10 train scenes.

If multiple grid points pass, the diagnostic candidate is chosen by maximum
true accepts, then higher selected-zero persistence, then higher candidate-zero
fraction.  Passing only permits freezing a candidate for a disjoint GOAT
test; it does not authorize deployment.  If no point passes, policy consensus
cannot directly emit STOP.  It may only serve as a cheap proposal for an
independent current-to-goal geometric arrival certificate.

## Formal suffix repair

Formal job `15753043` stopped after 74/80 episodes because the historical
train scene `YmJkqBEsHnH` is the one known 3-leg gap absent from the pt1
overlay.  This was an input-mount omission, not an inference failure.  Its
3,480-row / 870-state prefix spans exactly the first 74 frozen episodes and is
pinned by SHA-256
`24315d919863f11497715a0dc5d64e66461b99027eb011337325618c4d571949`.

The missing scene already exists in the audited historical gap-fill tree.  Its
`raw_audit.json` has status `audited_historical_summary_match`, content-addresses
11,342 files, and has SHA-256
`d5a9e7548aa897f04be4e75cf27ad0634b4573f8c8b2ad79a7dbbf79997f771d`.
It is mounted read-only at the original episode path.

A deterministic suffix repair was submitted as Slurm `15753536`:

- immutable source:
  `/scratch/yz11502/Research/source_bundles/navdp_arrival_repair_0d1e54b4fd347af3`
- source receipt SHA-256:
  `0d1e54b4fd347af3b8dfa5eb83bd2bae91ab708a458957767e05d4fdf7127cb6`
- result root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/navdp_arrival_consensus_20260815/repair_20260814T174400Z`
- submission receipt SHA-256:
  `7996346cfceb9c66ebdf78870db978d5dd7e4586ff33e988952c9eb6c67207bf`

The repair evaluates only source indices 74--79 with the same state-derived
seeds and checkpoint.  It then fail-closed merges those six episodes with the
pinned prefix, requiring exact episode order, disjoint state IDs, four unique
seeds per state, 16 candidates per query, and final 40-scene/80-episode
coverage.  No partial signal values were read before defining this repair.

## Final train40 result

Repair job `15753536` completed successfully in 1m54s.  The independently
verified merged population contains:

- 40 scenes, 80 episodes, and 160 active ImageGoals;
- 939 distance-stratified states: 160 arrival (`<=0.25 m`) and 779
  non-arrival states;
- 3,756 unique-seed NavDP queries;
- 16 candidates/query and 64 sampled trajectories/state.

The sealed merged `report.json` SHA-256 is
`4b72015567a200d6c158858edbc693a7060d469836fca71743b3385f3f651bb7`.
Independent recomputation reproduced all 24 predeclared operating points and
the following AUCs:

- selected-zero persistence: `0.7087`;
- mean all-candidate zero fraction: `0.7244`;
- mean top-four zero fraction: `0.7164`.

The frozen direct-STOP gate failed.  Even the lowest-false-positive operating
region (all 16 candidates zero in every query) accepted 56/160 arrival states
but also 43/779 non-arrival states.  False accepts span 27 scenes, and 39/43
are in the `(0.25, 0.50] m` near-miss band.  No predeclared point has zero
false accepts, so no threshold is selected or authorized.

This failure has a direct contract explanation: frozen NavDP clamps every
candidate whose endpoint is shorter than `0.5 m` to zero, while the GOAT
success radius is `0.25 m`.  Candidate consensus can rank proximity weakly,
but the policy output has already discarded the resolution needed to certify
the benchmark's arrival event.  A selected zero must remain `abstain/replan`,
never `SUBTASK_STOP`.

The next frozen comparison should evaluate an independent current-to-goal
LightGlue/LingBot-PnP distance certificate on these exact 939 states.  This is
a fair paired test of whether geometric relative pose retains the sub-25 cm
information that NavDP's post-clamp trajectory set cannot represent.
