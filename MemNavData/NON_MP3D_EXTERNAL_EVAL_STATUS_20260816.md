# Non-MP3D external evaluation status — 2026-08-16

> **Current continuation:** HM3D 的最终封账、2026-08-15--17 三日时间线、
> Pi3X learned relocalizer、论文定位和 Final14 当前状态，已统一汇总到
> `STATUS_20260817_PI3X_LEARNED_RELOCALIZER_FINAL14.md`。本文保留为
> 2026-08-16 非 MP3D 评测与 repair 的完整历史账。

## Frozen decision

Tonight's efficacy budget is restricted to non-MP3D environments.  No MP3D
scene, episode, arm, or outcome may be evaluated.  The primary external test is
an outcome-disjoint ten-scene subset of the 100-scene HM3D v0.2 val asset
archive.  Gibson/MemoNav is audited as a possible second route, but is not
allowed to masquerade as an official score when its released interface is
under-specified.

| Route | Current status | Honest claim |
|---|---|---|
| HM3D held-out val10 causal Revisit | Runtime-parity smoke passed; formal array waiting for GPU | Cross-dataset transfer of the frozen CEC/NavDP stack |
| Official GOAT val-unseen | Consumed clean null; do not rerun or tune | CEC made no executable intervention under the official first-ImageGoal protocol |
| MemoNav Gibson multi-goal | Not directly runnable | Dataset/interface compatibility audit only |
| Replica long-horizon Revisit | Constructibility failure under the frozen contract | No method-efficacy conclusion |

## Current HPC submission

The frozen HM3D chain was submitted on 2026-08-16 through the verified shared
PTY/SCP path.  The immutable identifiers are:

- task bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_heldout_val10_revisit_fd8b9f5343c9534e`;
- task receipt SHA256:
  `999a28e27f0d68dee673cc1c6c4edd9fe31a05ecf54c4b30bd14f7f27e6cca4d`;
- protocol SHA256:
  `a019a49248950a537b14c651b7a812ba7ccb421504901f8a8de030d63ae3a230`;
- run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_heldout_val10_revisit_20260816/hm3d_heldout_val10_20260816T0448Z`.

Jobs are `15814346` (prepare), `15814347` (generation array), `15814348`
(manifest), `15814349` (evaluation array), `15814350` (summary), and
`15814351` (independent verification).  Prepare completed in 22 seconds.
At initial submission, generation was pending under `QOSGrpGRES`; every
downstream dependency was intact.  This was queue state, not an evaluation
result or failure.

The generation array subsequently finished for nine scenes and failed for one.
Original task 8 (`q3hn1WQ12rz`) exhausted all 240 frozen outer attempts and
generated `0/4` episodes; every other task generated `4/4`, leaving 36 complete
episodes over nine scene clusters.  This was a pre-navigation constructibility
failure, not a controller or CEC outcome.  The original manifest, evaluation,
summary, and verifier were cancelled by dependency, and the run root contained
no policy scene output.

An outcome-blind construction-attrition amendment was frozen before any arm
was run.  It keeps the original ten selected scene identities, records
`q3hn1WQ12rz` as explicit zero-episode attrition, evaluates all 36 episodes
from the nine fully generated scenes, and preserves original scene indices and
Williams arm order.  It does not replace a scene, retry generation, relax a
constraint, or alter any method component.  The result will be explicitly
underpowered relative to the original 10-scene/40-episode target.

The immutable repair bundle is
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_heldout_val10_attrition_repair_e89eda740d3b4fb9`,
with receipt SHA256
`1bbd86b4bb790821e87956ad757dc38918f17928d3adf1e40ff23099c06bc996`.
Remote read-only preflight rebuilt the manifest as exactly `10 selected / 9
constructible / 36 episodes / original indices 0-7,9`.  The resumed jobs are
`15826322` (manifest), `15826323` (sparse evaluation array), `15826324`
(summary), and `15826325` (independent verification).  At submission they were
pending with intact dependencies; there was still no new SR.

### Runtime-interface incident and current repair chain

Manifest `15826322` completed, but all nine executions of array `15826323`
failed before Goal A because the old base evaluator rejected
`--certified_cdec_rescue off --certified_stagnation_graph off`. The same base
adapter also lacked `raw_fixed_bearing_v1`. All nine trace logs have identical
SHA256
`5ea7d8036dba93d2d08be8264e2114158589edaa404561770ce308bfeb67fa82`;
there were zero metric, summary, plan, or Goal-A-trace outputs. This was an
interface failure with no navigation outcome and no scientific contamination.

The repair uses the fully validated successful actual-online NNR runtime
`shared_online_nnr_11458cb2b75ee334`, receipt SHA256
`31b3e087b855e0220f6821ad96e6f5e74114bc12dc6c3afa6f7f79150dfb4575`,
plus exactly one controller-boundary overlay: the already frozen
`revisit_bearing_adapter.py`, SHA256
`c1f10b3c831f00a5b4742e0b34ac0675f10e161c4795ed1497c74b9551fdaf78`.
The evaluator and servers remain byte-identical to that runtime.

Consumed-smoke attempt 1 (`15838383`, downstream `15838384`--`15838386`)
correctly failed before server readiness and generated zero scientific
outputs. Direct comparison with the successful lineage's own launcher found
two missing dependency-only Python paths for `depth_anything`; they are now
mirrored exactly. This was not a method or parameter change.

Consumed-smoke attempt 2 (`15838486`, downstream `15838487`, `15838490`,
`15838491`) then started both servers and verified the evaluator/overlay
hashes, but failed before rollout because its minimal input package omitted
the expert RGB stream from which the evaluator reconstructs Goal A. It also
generated zero trace, metric, plan, or navigation outputs. The replacement
smoke bundle contains the complete pre-existing consumed episode, including
all RGB/depth frames.

The current immutable repair bundle is
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_runtime_interface_repair_b49e7ef92e24e35c`,
receipt SHA256
`dbbc362d68c0dc97bd25722915f7dbc361b72807f599b621c9fb1a0d62aae48d`.
The consumed-scene smoke data bundle is
`hm3d_consumed_runtime_smoke_7f16b7768c2dd054`, receipt SHA256
`7f16b7768c2dd054977ea00bc5d275c5e0bd552f6d49340fa9e8f2286bb58466`.
It uses prior-consumed scene `5cdEh9F2hJL/episode_0001`, disjoint from the
held-out val10 population.

Current dependency chain:

- consumed four-arm runtime/schema smoke: `15839649`;
- formal sparse nine-scene array: `15839654`, strictly
  `afterok:15839649`;
- summary: `15839655`;
- independent verifier: `15839656`.

The fresh formal output root is
`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_heldout_val10_runtime_repair_20260816/hm3d_heldout_val10_rt_20260816T1345Z`.
It contains a byte-identical copy of manifest SHA256
`62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89`.
Smoke `15839649` completed with `passed=true`. Its receipt verifies a
single shared Goal-A trace across all four arms, non-empty Goal-B execution,
the raw-fixed adapter/takeover fields, nine certified requests with zero
runtime failure, and the frozen evaluator/overlay hashes. Formal array
`15839654` subsequently completed all nine sparse scene tasks `0:0`, with a
complete four-episode/four-arm contract for every scene. Original summary
`15839655` failed before metric-row access on an explicit old/new scene-schema
name mismatch and created no report; verifier `15839656` was cancelled by
dependency. After a nine-contract field audit, sealed analysis-only repair
summary `15847580` and independent verifier `15847581` completed `0:0` without
rerunning an episode. Both output receipts validate and the independent report
says `verified=true`.

The formal counts are native `7/21`, geometry `17/21`, raw oracle-role
`18/21`, and role-free certified `19/21` for Revisit B given the shared Goal-A
success; joint counts are respectively `7/36`, `17/36`, `18/36`, and `19/36`.
Certified versus native is `+12/-0`, exact McNemar `p=0.000488`. Certified is
not significantly better than geometry (`+2/-0`, `p=0.5`) or raw oracle-role
(`+1/-0`, `p=1.0`). The consumed smoke remains excluded from all efficacy
statistics.

An outcome-blind formal-input audit then checked all 36 sealed episodes. All
36 contain both the exact Goal-A source frame at `switch_idx-1` and
`goal_1.jpg` (72/72 required files, zero missing). This closes the input
completeness issue exposed by consumed-smoke attempt 2 without reading a
navigation outcome.

The prepare receipt independently records ten selected scenes, 376,348,208
extracted asset bytes, and source-archive SHA256
`04c97761cb16ed8bd6f6600d4211ab10b9d3649d981401b527f0c0264a60371b`.
Local copies of the immutable submission and extraction receipts are under
`.diagnostics/hm3d_heldout_val10_revisit_20260816/`; their SHA256 values are
`cd0d778233ab38a762993a118a480d8067f718540bcfe3a7a86296cbc4d1fdff`
and
`2e8839eb9c6fb1e01fcdb332eb45649a20967e6fc7ec28d25b3a621202e2c9f7`,
respectively.  A local verifier checked that their protocol, selection receipt,
and ten asset identities agree exactly.

Queue alternatives were tested without creating jobs.  Slurm rejected
`cpu_short` with GPU resources and rejected explicit `gpu168`/`gpuplus` for
this account/job contract.  The validated GPU route remains the multi-partition
request already encoded by the frozen evaluator; queue state does not authorize
changing scientific scope.

The SSH diagnosis and mandatory future procedure are in
`MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md`.  In particular, a stalled
no-PTY mux command is not evidence that the user's normal shared SSH or MFA is
broken.

## 1. HM3D held-out val10: the evaluation that should run

The immutable protocol is
`MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json`, SHA256
`a019a49248950a537b14c651b7a812ba7ccb421504901f8a8de030d63ae3a230`.
It freezes:

- the five authoritative prior GOAT pilot/smoke/formal manifests and their
  hashes, whose union contains 36 unique consumed HM3D scenes;
- an outcome-blind deterministic selection: subtract those 36 identities from
  all 100 val assets, sort by five-digit archive index, take the first ten;
- selected asset directories `00801`, `00804`, `00805`, `00806`, `00807`,
  `00809`, `00811`, `00812`, `00816`, and `00817`, with zero consumed overlap;
- four generated causal-Revisit episodes per scene, 40 total;
- one actual-online native NavDP Goal-A trace, byte-identically replayed by all
  Goal-B arms;
- intention-to-treat retention of every Goal-A failure;
- four paired arms: native, raw-fixed oracle-role, old geometry router, and
  certified relocalization;
- joint and conditional `B|A` success, paired gain/loss, exact McNemar, and
  scene-cluster bootstrap reporting;
- no threshold tuning, no post-outcome filtering, and no public GOAT/MemoNav
  leaderboard claim.

The first draft incorrectly proposed archive indices `00800`--`00809`; a
cross-manifest audit found four of those scenes had already appeared in prior
GOAT work.  Submission was stopped before any job was launched.  The corrected
selection above leaves 64 unconsumed scenes and is independently recomputed
from the historical manifests plus the official archive member list.

The already downloaded official HM3D v0.2 val archive contains all ten selected
directories and each required `.basis.glb`/`.basis.navmesh` pair.  The prepare
job reuses that validated val archive read-only and downloads the official val
archive only if the former is absent.  It extracts only the 20 selected files.

### Local closed-loop implementation gate

The first smoke episode correctly failed closed because native Goal-A failed;
no arm ran Goal-B.  A second preselected engineering episode then exercised the
entire path:

| Arm | Goal A | Revisit B | B steps | Certificate requests / accepts |
|---|---:|---:|---:|---:|
| native | 1 | 0 | 185 | 0 / 0 |
| raw fixed | 1 | 1 | 74 | 0 / 0 |
| certified | 1 | 1 | 73 | 10 / 10 |

This is an implementation smoke on a consumed HM3D scene, not an efficacy
number.  It proves that HM3D rendering, actual-online history replay, DINO
retrieval, LightGlue/PnP certification, the fixed-radius bearing adapter, and
frozen NavDP control execute together.

The formal chain is implemented as:

1. asset selection and sealing;
2. ten-scene episode-generation array;
3. outcome-blind identity/dependency manifest;
4. ten-scene four-arm closed-loop array, one server pair per scene;
5. fail-closed paired summary;
6. independent raw-CSV/plan recount that does not import the primary summary.

The local test suite includes selection-union reconstruction, manifest guard
checks, a synthetic ten-scene end-to-end summary, and an independent recount.
Final pass counts are recorded at bundle sealing rather than asserted here in
advance.

## 2. Why released MemoNav/Gibson is not an official score we can run now

The reproducible auditor is
`MemNavData/audit_memonav_gibson_readiness.py`.  Its current sealed receipt is
`.diagnostics/memonav_gibson_readiness_20260816.json`, SHA256
`e91566c5be0f735f93ce81b8cc2f9752ef77633f27df77973a888940795571a7`.
It inspected the official MemoNav repository at commit
`25fa5077e2f408d67f2382955b65f45680fa97c0` and the available local dataset
roots.

### What is actually released

- Fourteen Gibson scene episode files are present for each difficulty.
- The released multi-goal sets contain 700 episodes at each of 2, 3, and 4
  goals; `1goal`/`val_4200` contain 4,901 records in this checkout.
- The repository contains no Python source outside `.git`; its README still
  lists training and evaluation code as TODO.
- All episode records contain goal positions only.  Across every released
  split there are zero goal rotations and no goal RGB images.
- None of the fourteen required Gibson GLB assets is available under the
  checked local dataset roots.

The missing rotation is not cosmetic.  Habitat 0.2.x's default
`ImageGoalSensor` renders a random yaw from `hash(episode_id)`, caches one goal,
and reads `episode.goals[0]`.  It does not define how MemoNav's sequential
second/third/fourth goal changes the goal image.  The absent MemoNav evaluator
is precisely the code that would have to settle this ambiguity.

The sensor and metric contracts also differ.  MemoNav evaluates panoramic
RGB-D agents and reports multi-goal progress/PPL, whereas this project uses a
forward monocular NavDP stream and paired joint/conditional SR.  Therefore a
number produced by inventing a goal yaw and switching rule cannot be compared
directly with MemoNav's published PR/PPL.

The official sources support this reading:

- [MemoNav repository](https://github.com/ZJULiHongxin/MemoNav) lists training
  and evaluation release as TODO and provides the episode datasets;
- [MemoNav paper](https://arxiv.org/html/2402.19161) specifies panoramic RGB-D,
  700 multi-goal Gibson episodes, a final goal near a previous one, 500 steps,
  and PR/PPL;
- [Habitat-Sim dataset documentation](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
  requires accepting the Gibson terms before downloading scene assets.

### The dataset is related, but it is not an explicit Revisit benchmark

No adjacent goal positions are exact repeats.  For the final goal's Euclidean
distance to the closest previous goal:

| Split | Median | Fraction within 1 m | Fraction within 2 m |
|---|---:|---:|---:|
| 2-goal | 5.52 m | 0.0% | 0.29% |
| 3-goal | 3.56 m | 19.43% | 30.0% |
| 4-goal | 2.21 m | 26.86% | 48.29% |

Thus it contains increasingly revisit-like final goals, especially at four
goals, but causal visual support cannot be known from coordinates alone.  It
must be measured against each method's actual online history after rendering.

### Honest route if Gibson assets are later acquired

A valid project-specific compatibility protocol could freeze a deterministic
goal-render and sequential-switch contract, then run native/raw/certified on
the same actual-online prefixes.  It should report full ITT PR/PPL plus an
ex-ante causal-support stratum.  That would be useful cross-dataset evidence,
but must be named **Gibson compatibility transfer**, not an official MemoNav
score and not a reproduction of MemoNav's published numbers.

Until the licensed meshes and the goal-image contract exist, submitting GPU
jobs for these JSON files would be theatre rather than an evaluation.  The
outcome-disjoint HM3D val10 transfer is the only non-MP3D external route tonight
that is both runnable and scientifically identifiable.

## 3. Execution state

1. Shared SSH, archive, all three source receipts, checkpoints, LightGlue
   dependencies, and Slurm partitions were verified.
2. The 36-episode constructible manifest is sealed; runtime-parity smoke
   `15839649` gates formal array `15839654` and both analysis jobs.
3. All nine formal scene tasks, the repaired summary, and independent verifier
   completed successfully; the verifier reports `verified=true`.
4. External HM3D Revisit utility is confirmed, but certified superiority over
   geometry/raw controls is not established by this sample.
5. Do not start Gibson, Replica, GOAT, or MP3D jobs unless a separate frozen
   protocol resolves their present construct/interface mismatch.
