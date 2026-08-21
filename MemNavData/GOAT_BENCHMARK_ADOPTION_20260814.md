# GOAT-Bench adoption protocol

Date: 2026-08-14 (Asia/Shanghai)

**2026-08-15 status update:** the separate 20-scene first-ImageGoal
semantic-arrival confirmation completed with `0/20` certified successes and
zero STOP coverage; its preregistered gate failed and the result was
independently verified. It did not test episodic Revisit bearing. The dataset
recurrence inventory below remains valid (`338/822` repeated-instance
ImageGoals), but stages 4--5 are blocked until a shared, independently
validated GOAT STOP/non-image controller exists. The failed arrival adapter
will not be retuned on its held-out episodes. See
`GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md`.

## Decision

GOAT-Bench is promoted ahead of another custom cross-domain benchmark because
it directly tests persistent navigation memory over 5--10 sequential goals.
The project will not call a run an "official GOAT score" unless it uses the
official episode order, simulator actions, `SUBTASK_STOP`, success/SPL
measurements, and causal observation stream.

The primary project result on GOAT will be the paired effect of Certified
Episodic Compass on **ImageGoal subtasks**, not a claim that the current method
solves ObjectGoal and LanguageGoal.  Full-episode partial success and SPL are
secondary system metrics.

## Pinned public inputs

- Official code: `Ram81/goat-bench`, commit
  `74c41d19d4a4c3608d1575b512087b5a529aee0e`.
- Official InternNav NavDP-to-discrete reference: commit
  `7a5c62400ac45b313d9b709c740b64191556a242`, function
  `internnav/model/utils/vln_utils.py::traj_to_actions`.  Its local file hash
  at protocol freeze is
  `cf51aafbd49b833d0264f09ad4cd4e4bb1b1fc8b9159f2b8fd771237fb6a04ac`.
- Official episode archive SHA-256:
  `d731a81dccb1c4ba2b825a03d8ff6ed0cd8fc22935d7aacd98e78b53dfdbf904`.
- The released `val_unseen` public assets are cached locally: the monolithic
  checkpoint, all 36 ImageGoal scene embedding files, the LanguageGoal cache,
  and the ObjectGoal cache (39 files, 156 MiB).  The checkpoint SHA-256 is
  `55e89c3d083198d4add4e9e70164b54ff892900963a2925471362e2d4761b3eb`.
- The HM3D v0.2 validation meshes were obtained through an authorized
  Matterport API token.  The complete validation archive is retained with a
  SHA-256 receipt, and exactly the 36 `val_unseen` scene GLB/navmesh pairs were
  selected, hashed, and sealed.  Public episodes themselves do not include
  these licensed scene meshes.

## Dataset-only audit before any navigation outcome

The official `val_unseen` split contains 36 scenes, 360 episodes, and 2,669
subtasks: 991 ObjectGoal, 856 LanguageGoal, and 822 ImageGoal.  Among the 822
ImageGoal subtasks, 338 (41.1%) name an instance already named by an earlier
image or description subtask; these queries occur across 211 episodes.  There
are 216 image queries preceded by an ImageGoal for the same instance and 222
preceded by a LanguageGoal for it (the sets overlap).

These counts establish that the benchmark contains a substantial recurrence
stratum.  They are **not** runtime Revisit labels: an earlier subtask can fail,
and a target can also enter the camera incidentally.  Formal support is defined
only by causal online RGB history; task type, instance ID, goal location and
future frames are forbidden method inputs.  Instance IDs may be read only by
the evaluator for post-hoc strata.

The reproducible audit command is:

```bash
python MemNavData/audit_goat_bench_dataset.py \
  .diagnostics/datasets/goat-bench/extracted/data/datasets/goat_bench/hm3d/v1
```

## Paired system arms

All arms use the same official non-image controller and the same episode
order.  Non-image components are held fixed; only the ImageGoal skill changes.

1. `official_reference`: reproduce the released GOAT policy/checkpoint before
   modifying the evaluator.  This is an installation and metric anchor.
2. `hybrid_navdp_native`: frozen NavDP for ImageGoal, with no episodic
   residual.
3. `hybrid_navdp_certified`: the same frozen NavDP, preceded by the frozen
   top-8 retrieval, SuperPoint/LightGlue, LingBot-depth PnP certificate and
   fixed-radius bearing residual.  Rejection must be byte-identical native.

If the released skill-chain component checkpoints cannot be reproduced, the
released monolithic policy may be used as the shared non-image controller, but
the result must be named a hybrid GOAT evaluation and cannot be compared as a
drop-in replacement for the paper's skill-chain row.

## Official-contract requirements

- Use the official HM3D validation scene meshes and unmodified episode order.
- Render each ImageGoal from its stored `InstanceImageParameters`, including
  pose, resolution and HFOV; a cached CLIP embedding is not a NavDP goal image.
- Use the official Stretch embodiment, no-sliding navmesh, 0.25 m forward and
  30-degree turn actions.
- NavDP trajectory output must be converted to those discrete actions by one
  frozen deterministic adapter shared by native and certified arms.  Start
  from InternNav's released inner
  `trajectory_to_discrete_actions_close_to_goal` logic rather than inventing a
  new controller, changing only its turn quantum from 15 to GOAT's official 30
  degrees.  The current NavDP HTTP server already returns cumulative metric
  waypoints, so the outer InternNav delta-to-cumulative conversion and `/4`
  scale must not be applied a second time.
- An ImageGoal subtask ends only when the agent emits `SUBTASK_STOP` or reaches
  its budget.  The current custom evaluator's ground-truth distance auto-stop
  is prohibited.  The released trajectory-to-discrete adapter's autonomous
  no-motion/STOP decision is used; NavDP critic values are logged but no GOAT
  validation outcome may tune a stop threshold.
- NavDP receives current RGB-D and the current goal image, matching this
  project's existing controller contract.  LingBot-estimated depth, not
  simulator depth, is used by the episodic PnP certificate.  GPS, target pose,
  instance ID and ground-truth depth are not certificate inputs.  Ground truth
  remains evaluator-only.
- Store all causal RGB observations across every subtask, including failed
  ones.  Never seed memory with rendered goal images or expert trajectories.

## Metrics and reporting

Primary paired metrics:

- ImageGoal subtask SR and SPL for certified versus native;
- paired gain/loss and exact McNemar, clustered by scene;
- certificate activation, exact-fallback rate and false takeover;
- results split by causal visual support and task-list recurrence, with the
  latter explicitly marked as evaluator-only analysis;
- SR/SPL versus subtask index.

Secondary metrics are official partial success, composite SPL and whole-episode
success.  Published full GOAT scores are contextual references, not directly
subtracted from an image-only intervention unless every sensor, controller and
goal modality matches.

## Execution ladder

1. Asset gate: authorized HM3D val v0.2 meshes load and all 36 GOAT scene IDs
   resolve. **Passed** for 36/36 selected scenes with sealed hashes.
2. Contract smoke: one episode from two scenes, verifying raw goal rendering,
   discrete action conversion, autonomous stop, task transitions and official
   metric recomputation.  No threshold changes are allowed from this smoke.
   **Passed locally on the exact Habitat-Lab/Sim 0.2.3 stack** for frozen
   episodes `4ok3usBNeis:3` and `5cdEh9F2hJL:4`, including the metric-depth
   observation required by frozen NavDP.  First HPC environment job `15738230`
   failed before writing its receipt because an unnecessary `transformers`
   install attempted to build a modern `safetensors` dependency on Python 3.7;
   dependent smoke `15738231` therefore ran zero seconds.  The repaired,
   minimal immutable environment job `15740152` completed and wrote a valid
   receipt.  Its first dependent smoke `15740159` reached no episode because
   Python 3.7 `py_compile` attempted to write into the immutable bundle; the
   read-only AST-check smokes `15740384` and `15740638` then reached simulator
   initialization on L40S and H100 but both failed before an episode.  Dynamic
   linking audit showed the wrapper had replaced Singularity `--nv`'s NVIDIA
   `/.singularity.d/libs/libEGL.so.1` with the image's generic EGL library.
   That override was removed in immutable smoke job `15746123`, which completed
   successfully on HPC in 38 seconds for both frozen scenes.  All scene,
   pathfinder, raw-goal rendering, metric-depth, action-adapter and subtask
   transition checks passed.  This was a wrapper-level cause, not evidence
   against either GPU family.  The smoke receipt explicitly states
   `is_navigation_score=false`.
3. Native ImageGoal runtime pilot: before wiring an unverified non-image
   controller into full sequential episodes, run the first ImageGoal subtask
   of one outcome-blind frozen episode from ten scenes.  This intermediate
   gate measures NavDP/GOAT observation, action, stop, wall-time and memory
   compatibility only and is explicitly not a GOAT score.  Immutable job
   `15750812` was submitted with manifest SHA-256
   `652cbe0f731c3b817e9c1e0f5e516ae4f386d74380a7ed06c4910651357b5db5`.
   See `GOAT_NAVDP_RUNTIME_PILOT_PROTOCOL_20260814.md`.
4. Full runtime pilot: one complete episode from ten scenes after the official
   reference and a shared non-image controller are reproduced; measure wall
   time and memory, not choose a method or threshold.
5. Formal confirmation: all 360 `val_unseen` episodes, scene-sharded on HPC,
   with native and certified arms paired by episode and deterministic seed.

Until stages 1--2 pass, no percentage is reported as a GOAT result.
