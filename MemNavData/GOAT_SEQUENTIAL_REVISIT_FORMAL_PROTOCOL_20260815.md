# GOAT Sequential-Revisit Targeted External Evaluation

**Date:** 2026-08-15  
**Status:** code/manifest frozen locally; formal HPC submission pending a live
`yz11502` shared SSH master  
**Scope:** targeted scene-balanced GOAT `val_unseen` sequential-Revisit
evaluation. This is **not** a full GOAT benchmark score.

## 1. Why this replaces the first-ImageGoal arrival branch

The frozen first-ImageGoal semantic-arrival experiment completed as a clean
negative result:

- 20 episodes / 20 disjoint scenes;
- `0/20` certified successes and zero certified STOP coverage;
- all 20 runs ended at the guard;
- the preregistered coverage gate failed and the branch is closed without
  threshold retuning.

That experiment did not test the paper's established positive mechanism. CEC
is an episodic Revisit method: it retrieves a place from an actual causal RGB
history, geometrically self-certifies the relation, emits only a scale-free
bearing, and otherwise leaves the frozen controller unchanged. The new
evaluation therefore tests sequential Revisit directly instead of trying to
turn a first-goal NavDP zero endpoint into semantic STOP.

Formal negative result:
`MemNavData/GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md`.

## 2. Outcome-blind source population

The released GOAT HM3D `val_unseen` task lists were audited without reading a
rollout, RGB observation, retrieval score, geometry result, or navigation
outcome:

| quantity | count |
| --- | ---: |
| scenes | 36 |
| episodes | 360 |
| ImageGoal subtasks | 822 |
| exact-instance repeated ImageGoals | 338 |
| episodes containing such a target | 211 |
| scenes containing such a target | 36 |

An exact recurrent target is an ImageGoal whose instance id occurred in an
earlier instance-specific ImageGoal or LanguageGoal task. ObjectGoal does not
carry an exact instance id and cannot establish this evaluator stratum.

This metadata is used only to freeze a target and stop/stratify evaluation.
The controller never receives the target index, prior modality, instance id,
or a Novel/Revisit label. Runtime support must still be inferred solely from
the causal RGB stream.

## 3. Frozen selection and non-contamination rule

Two scenes used for local engineering (`4ok3usBNeis`, `5cdEh9F2hJL`) are
excluded from formal evaluation and retained only in a two-scene smoke
manifest.

For each of the remaining 34 scenes:

1. find every episode containing an exact recurrent ImageGoal;
2. choose the episode whose first recurrent target occurs earliest in its task
   list;
3. break equal-index ties using
   `SHA256(salt | scene_id | episode_id)`;
4. choose the first exact recurrent ImageGoal in that episode.

The rule intentionally improves target constructibility under finite compute;
it does not claim to represent all recurrence horizons. It is scene-balanced
and cannot use method outcomes. The resulting target indices are:

| target subtask index | scenes |
| ---: | ---: |
| 1 | 14 |
| 2 | 9 |
| 3 | 7 |
| 4 | 4 |

Frozen manifests:

- formal 34 scenes:
  `MemNavData/goat_sequential_revisit_formal_manifest_20260815.json`,
  SHA-256
  `aaedc6fb0c6d3787b5c8c61eed2c2d943320f595f9b1783f881febc544121397`;
- consumed-scene smoke:
  `MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json`,
  SHA-256
  `7e23655af2578c39c8435981584dbe65b2de8eb2025478f3cfd50921d07628ab`.

## 4. Paired arms and controller contract

Each frozen episode is run from time zero in two arms with the released GOAT
monolithic checkpoint:

### Official GOAT

- released stochastic evaluation semantics (`deterministic=False`);
- CUDA policy execution;
- official seed 100 reset identically for the paired arms;
- recurrent state persists across subtasks as in the released configuration;
- maximum 5000 actions, matching the official GOAT task configuration.
- arm order is frozen in the manifest and balanced 17/17 between
  native-first and CEC-first, preventing a fixed CUDA/environment warm-state
  order confound.

### Role-free CEC

The official GOAT policy still runs at every step and controls all unsupported
states. On ImageGoal steps only:

```text
causal online RGB history + current ImageGoal RGB
  -> frozen DINO temporally diverse top-8
  -> SuperPoint + LightGlue + Fundamental geometry ranking
  -> LingBot history depth + dual-intrinsic PnP
  -> atomic certificate
       inliers >= 16
       query/reference hull coverage >= 5%
       reprojection RMSE <= 2 px
  -> accept: scale-free bearing -> fixed 2.5 m residual -> frozen NavDP motion
  -> reject/error: exact released-GOAT action
```

The certificate is unchanged. No GOAT scene is used to retune its thresholds.
The goal and online RGB cameras have distinct intrinsics and are mapped
separately into LingBot's padded coordinate system.

The CEC shortlist no longer waits for the unrelated learned decoder's
`S+W=40`-frame warm-up. Once the eight-frame LingBot scale block and dense DINO
features exist, the same causal top-8 and unchanged certificate can run. This
is a runtime-dependency correction, not a threshold relaxation. On train-only
data, the pre-existing certificate's `gap<=32` stratum had precision `12/14 =
85.71%`; GOAT must measure its actual open-set behavior without further
tuning.

### Exact STOP authority

GOAT success is position-only (`<0.25 m`) and does not require terminal image
orientation. Consequently:

- every official `SUBTASK_STOP` is executed exactly and immediately;
- CEC may override only a non-stop motion action after certificate acceptance;
- no terminal U-turn or image-alignment action is in the formal method.

This removes an unnecessary confound and makes “official STOP authority” a
runtime assertion rather than a narrative claim.

## 5. Frozen analysis

### Primary estimand

All 34 episodes are retained intention-to-treat, including cases that never
enter the target, have an unsuccessful prior task, or have no certifiable
history.

- outcome: success of the frozen repeated ImageGoal target;
- effect: CEC minus official GOAT paired risk difference;
- test: two-sided exact McNemar;
- interval: scene-cluster percentile bootstrap 95% CI;
- because there is one selected episode per scene, the primary sample is also
  scene balanced.

### Constructibility diagnostics, not filters

The report separately counts:

- target entered by native / CEC / both;
- success of the earlier exact-instance task;
- target DINO-candidate and certificate-accept coverage;
- pre-target, task-list-nonrecurrent certificate takeover;
- rejection reasons and guard exhaustion.

Mechanistic interpretation requires at least 20 pairs across at least 12
scenes entering the target in both arms. Failure of this gate means the
targeted GOAT population was under-constructible; it does not permit selecting
a favorable subset. The 34-episode ITT result is reported regardless.

## 6. Audit and failure conditions

The strict summarizer refuses to produce a result if any of the following
occurs:

- fewer or more than 34 raw records;
- CPU official policy;
- changed manifest, checkpoint, GOAT commit, seed, or action budget;
- evaluator role metadata reaching the controller;
- official STOP replacement;
- mismatch before CEC's first actual override;
- any zero-accept episode that is not action/pose-exact fallback;
- incomplete or duplicate manifest index.

An independent verifier re-reads raw episode JSON without importing the
summarizer and recomputes the denominators, success counts, paired gain/loss,
McNemar p-value, constructibility counts, and certificate event counts.

## 7. Local engineering evidence (not paper SR)

The complete manifest-index path was run on the already-consumed `5cd...`
scene using the official CUDA policy:

- target entered and transitioned after 28 actions in both arms;
- native and CEC target result were both zero;
- certificate accepted zero times;
- all 28 actions and poses matched exactly;
- official STOP preservation and manifest hash checks passed.

The balanced-order path was then rerun with the same consumed scene's frozen
`CEC -> native` order. The recorded order matched the manifest and the same 28
actions/poses remained exactly paired. Thus pairing is not contingent on
always evaluating native first.

Earlier local probes showed that decoder-warm-up decoupling changed an
otherwise invisible short history from zero candidates to 4--8 causal DINO
candidates, while unchanged geometry rejected them for insufficient spatial
support. This validates availability plus fail-closed behavior; the two scenes
are consumed and cannot estimate formal SR.

## 8. Implementation and launch state

Core files:

- manifest builder: `MemNavData/build_goat_sequential_revisit_manifest.py`;
- paired evaluator: `MemNavData/goat_sequential_revisit_pilot.py`;
- strict summary: `MemNavData/summarize_goat_sequential_revisit.py`;
- independent verifier: `MemNavData/verify_goat_sequential_revisit.py`;
- GPU array: `MemNavData/slurm_goat_sequential_revisit_eval.sbatch`;
- submit chain: `MemNavData/submit_goat_sequential_revisit_hpc.sh`.

Current local audit: 48 relevant tests pass; all launch scripts pass Bash
syntax checks; all imported method/evaluator modules parse in the pinned
Python 3.7 GOAT environment.

The formal `MEMNAV_MAX_FRAME_NUM=6000` capacity path was also exercised rather
than inferred from startup alone. After a 5001-step reset, the causal stream
processed 2052 frames continuously, crossing both 1024 and 2048 camera/RoPE
boundaries with no exception. Runtime was 384.2 s; observed GPU memory was
about 24.0 GiB at the end and did not exhibit per-frame linear growth. The
105 MiB temporary JPEG buffer was deleted after the audit and the GPU returned
to its idle 255 MiB state.

The immutable bundle closure was audited against dynamic runtime imports.
In addition to the runner, it explicitly carries the colored-registration
quaternion helpers, dual-intrinsic PnP module, certificate contract, GOAT
arrival status contract required at reset, and terminal-rotation helper still
used internally to produce diagnostics. Phase-B/CDEC modules remain absent
because their checkpoints/options are disabled in the frozen launch.

The submission chain is deliberately blocked until a live shared SSH master
authenticates as `yz11502`. The only detected shared socket authenticated as a
different account, so no remote upload, `sbatch`, or result-directory mutation
was attempted.
