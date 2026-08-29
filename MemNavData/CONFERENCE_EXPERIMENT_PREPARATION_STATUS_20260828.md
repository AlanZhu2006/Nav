# Conference experiment preparation status — 2026-08-28

This document records infrastructure and protocol state only.  A staged arm,
completed construction receipt, or scheduler state is not a navigation result.
Paper numbers remain governed by raw episode receipts, a frozen summary, and an
independent verifier.

## 0. Superseding execution snapshot

The earlier preparation-only statements below are retained as an incident
history.  The current source of truth is
`PAPER_EXPERIMENT_NIGHT_PLAN_20260828.md`.

- Final14 zero depth is complete and independently verified after an exact
  one-index infrastructure repair: Novel `3/21`, Revisit `1/21`, overall
  `4/42`; replacement jobs `16502265 -> 16502270`, `verified=true`.
- The first portsafe authority smoke `16502418` exposed a missing source-bundle
  dependency before producing an outcome: the new server came from the
  authority overlay, while the omitted `monocular_depth_runtime.py` silently
  resolved from the older base namespace package.  Formal `16502420`, verifier
  `16502421`, and lifelong launcher `16502570` were therefore cancelled by
  dependency as intended.
- The provenance-locked smoke `16503212` then completed both arms but failed
  its post-run audit because the generic plan serializer omitted the already
  computed authority policy/evidence fields.  This was a receipt-schema defect,
  not a controller change; formal `16503217`, verifier `16503241`, and launcher
  `16503597` were dependency-cancelled.
- A dependency-free receipt helper now copies those two diagnostics into every
  persisted plan and is tested in both memnav and Habitat interpreters.  The
  replacement authority DAG `16504303 -> 16504304 -> 16504307` is complete:
  all 21 cells and 42 proposal pairs passed independent verification.  Strict
  CEC versus the proposal-matched finite-PnP witness is `28/42` versus `25/42`
  overall (`+4/-1`, +7.14 percentage points, exact McNemar `p=0.375`); on
  Novel it is `8/21` versus `5/21` (`+4/-1`), while Revisit is identical at
  `20/21` in both arms.  These are matched authority-ablation results, not a
  fresh-generalization table.
- Natural-V4 factual B and population sealing are complete and independently
  verified: 22 supported histories in 15 scenes, population SHA
  `ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5`.
  Because the original 40-history gate was not met, a separate result-blind
  underpowered amendment was frozen.  CPU launcher `16504366` failed on a
  duplicate-partition submission bug after creating factual-C array
  `16505696`.  That array left 16 complete and six failed indices.  The six
  failed partials were archived without deletion.  The first exact repair then
  exposed a source-closure defect before evaluator startup: its overlay runner
  passed `--reject-policy shared_native_exact` to the older frozen hub, whose
  `ComparisonPlan` already hard-coded the same policy/NavDP fallback but did
  not declare that CLI.  Jobs `16509621,16509634,16509637` failed before any
  navigation step; the other lane and downstream jobs were cancelled before
  start.  A fail-closed AST compatibility audit now preserves the frozen hub
  and omits only that proved-redundant legacy option.  Attempt-2 bundle receipt
  is `899141ad23b4ff0ca3012bb68b9bc6aa0e5a8e1ee45bbfe4abcf8fa98ab89f26`;
  its submitted chain is collect smoke `16514058`, smoke verifier `16514066`,
  exact repairs `16514071,16514101,16514136,16514150,16514153,16514157`,
  integrity barrier `16514159`, seal `16514162`, and B2 resume launcher
  `16514165`.  No successful factual-C navigation outcome and no B2 outcome
  was read to select either repair.
- Attempt-2 smoke `16514058` completed, but gate `16514066` exposed one more
  receipt-only mismatch: the immutable legacy `/healthz` schema did not emit
  `reject_policy/reject_controller`, although its hash-bound compute identity
  and the pre-start AST audit already proved `shared_native_exact` with NavDP
  fallback.  All six scientific jobs and B2 descendants were dependency-
  cancelled before start.  The verifier now accepts the legacy health fields
  only together with those two independent authority receipts; navigation
  code and the frozen scientific contract are unchanged.  Local 25-test and
  remote 10-pass/1-skip preflights plus five Slurm test-only checks passed.
  The active exact-recovery chain is gate `16521565`, repairs
  `16521578,16521597,16521614,16521638,16521647,16521653`, integrity
  `16521666`, seal `16521671`, and B2 resume `16521679`.  At the latest audit,
  the gate and indices 0/1 were complete, index 9 was running on its frozen
  source node, and index 7 was waiting only for group GPU capacity; no outcome
  was opened.
- The ViNT ideal-bearing alignment loss-five result passed its mechanism gates
  but remains outcome-aware and nonphysical; no formal ViNT SR was submitted.
- The physical ViNT bearing executor is now wired into the same-process
  authority-pair runner rather than existing only inside the low-level
  evaluator. It is fail-closed to complete held-out populations, leaves the
  forced-reject arm at exact native ViNT, and emits an auditable bounded-turn /
  fresh-observation receipt. This is implementation readiness, not a result.
- A read-only audit of the 2026-08-20 HM3D parent population closed the apparent
  Table-1 "reserve" loophole.  All 130 materialized histories had already been
  attempted by the original constructor: 28 were retained, 52 lacked its
  preassigned Natural-Novel query, and 50 lacked both a standard Revisit and
  that Novel query.  The 102 cannot be relabelled as a ready fresh population.
  A separate fresh-query, scene-overlap construction protocol now excludes all
  28 consumed identities, tries the frozen preferred direction first and the
  other two directions only for structural constructibility, and requires
  `>=24 histories / >=15 scenes / >=4 per direction stratum` before any policy
  evaluation.  Its construction-only DAG is smoke `16525112`, array
  `16525114`, finalizer `16525132`, and verifier `16525152`; no NavDP or ViNT
  rollout is part of this submission.  The smoke completed with `0/2`
  constructible reserve histories in scene 0, and the formal construction
  array started normally.  This early count is not used to alter the protocol.
- The fresh-query construction array, finalizer, and independent verifier have
  now completed. The frozen selected prefix contains `28` histories from `21`
  scene clusters, with direction strata `front/side/rear = 4/5/19`; it passes
  every pre-registered power gate. The benchmark manifest SHA is
  `f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01`,
  and the independent verification SHA is
  `2a7b8f86f61a6f55762640dcbaef4b975539ec3d93cfb06649bddd6fa4c96dc8`.
  No policy outcome existed or was read during prefix selection.
- The post-gate controller evaluation is now prepared as a single fail-closed
  submission. NavDP runs `mono_native` versus `mono_cec`; ViNT runs exact
  native versus CEC with the audited `first_certified_bounded` physical-turn
  executor. Each controller has an independent raw verifier, followed by one
  joint provenance seal. A live pre-submission rehearsal passed `114` local
  tests and then stopped before `sbatch` because the construction verifier did
  not yet exist. This is the intended behavior and created no remote policy
  job. The frozen estimands and claim boundary are recorded in
  `HM3D_TABLE1_CONTROLLER_PORTABILITY_PROTOCOL_20260829.md` and its JSON
  companion.
- The gate-authorized controller evaluation was subsequently submitted. ViNT
  smoke `16526696` passed and formal array `16526731` is active; its aggregate
  and verifier are `16526745` and `16526759`. NavDP smoke `16526559` failed
  before a completed arm because the evaluator required the 2026-08-21
  SHA/frame/depth transaction while the wrapper launched the older Final14
  server that omitted those receipt fields. Only infrastructure logs were
  inspected. A minimal additive repair binds the already-verified transaction
  server overlay without changing population, checkpoints, thresholds, seeds,
  budgets, or controllers. Replacement NavDP jobs are smoke `16527714`, formal
  `16527718`, aggregate `16527722`, verifier `16527860`; replacement joint seal
  is `16527863`. Full diagnosis and guards are in
  `HM3D_TABLE1_NAVDP_TRANSACTION_REPAIR_20260829.md`.
- Repair smoke `16527714` then exposed one transitive-import omission before
  server startup: the narrow transaction overlay did not contain the unchanged
  vendored Depth-Anything directory, and the base baseline-local path was not
  on `PYTHONPATH`. Its formal descendants were dependency-cancelled and no arm
  ran. The v2 preflight now performs a real `policy_backbone`/Depth-Anything
  import from the two receipt-bound bundles. It and all Slurm test-only gates
  passed; replacement v2 jobs are smoke `16528367`, formal `16528369`,
  aggregate `16528383`, verifier `16528385`, and joint seal `16528391`.

## 1. Current priority order

1. Preserve the completed and independently verified proposal-matched
   authority ablation (`16504303 -> 16504304 -> 16504307`) as a sealed result.
2. Let the health-receipt-gated HM3D factual-C exact repair finish, then require the
   retained-output/runtime barrier, population seal, node-affine B2 smoke,
   formal paired B2, aggregation, and independent verifier in that order.
3. Do not interpret either formal array while it is partial; only frozen raw
   receipts plus the independent verifier can promote a number to the paper.
4. Keep ViNT formal SR blocked until the bounded physical-turn executor has a
   newly frozen, outcome-blind population.  The old 28-history adapter result
   and its five inspected losses must not be reused for method selection.
   The active Table-1 reserve DAG is construction-only; a failing power gate
   stops the line, while a passing independently verified gate merely
   authorizes a later four-arm submission.
5. Run the real-robot paired protocol only with an operator present; do not
   substitute an unattended HPC job for that evidence.

No new learned model, threshold search, development-set selection, or Novel
direction module is introduced by this queue.

The low-priority trajectory-length item has now been closed at the
constructibility level without reading policy outcomes.  On the sealed fresh
HM3D population, `0--20/20--30/30--50 m` contains `56/0/0` queries (Novel
`28/0/0`, Revisit `28/0/0`); the construction contract itself is `[2,9]` m.
Therefore Table 3 cannot be produced by post-hoc binning and requires a new
result-blind long-range benchmark if schedule permits.

## 2. HM3D continual repair (completed historical stage)

The factual-B source contains 99 frozen candidate histories.  Four completion
receipts were missing (`51, 52, 62, 63`); only shards `31` and `37` required
rerun.  The repair bundle keeps the original parserfix collector and rejects
overwriting any completed receipt.

- repair array: `16485965_[31,37%1]`;
- last scheduler snapshot: both tasks `COMPLETED`, exit `0:0`;
- completion paths visible after repair: `99/99`;
- deferred prefix launcher: `16486000`;
- dependency: factual-B repair plus completion of ViNT array `16482393`;
- planned prefix array: `0-98%4`, 20 minutes per element.

The repaired factual-B population has since been materialized and independently
verified: 22 supported histories in 15 scenes, with 18 strong-support
histories.  It missed the original 40-history power gate.  A separate,
result-blind underpowered amendment was therefore frozen before any C/B2
outcome was run.  Launcher `16501659` is queued after `16501320`; it will run
shared factual C first, seal successful C prefixes, and only then submit the
paired B2 arms.  This result can never be relabelled as the original powered
confirmation.

Repair sources:

- `repair_hm3d_fullmono_lifelong_natural_v4_factual_b.py`
- `slurm_hm3d_fullmono_lifelong_natural_v4_missing_b_repair.sbatch`
- `HM3D_FULLMONO_LIFELONG_NATURAL_V4_MISSING_B_REPAIR_SUBMISSION_20260828.json`

## 3. Final14 zero-depth arm

The previous `23/40` zero-depth result is a Novel-A Gate-D experiment and
cannot fill the mixed-role Final14 table.  A new arm was therefore prepared on
the exact 21-history/42-query Final14 population.

- arm: unchanged frozen NavDP with explicit zero depth;
- shared Goal-A replay, query images, seeds, budget, and success threshold are
  identical to the verified Final14 factorial;
- no metric sensor read and no monocular scale receipt are permitted;
- result aggregation joins the original four factorial rows only after checking
  their summary and verifier hashes.

Remote immutable bundle, smoke, and submission receipts are complete:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  final14_zero_depth_4c061bd6b86da365
```

Smoke `16499686` completed successfully.  Formal array `16499701` is running
with `0-20%2`, and summary/independent verifier `16499709` is dependency-held.
No partial outcome is a result.

## 4. Proposal-matched CEC authority ablation

Raw-DINO versus CEC is not an authorization-only comparison because both the
proposal and the intervention rule differ.  The new paired arm isolates the
certificate boundary:

```text
same causal RGB replay
same monocular query depth
same DINO top-8
same SuperPoint + LightGlue correspondences
same Fundamental-MAGSAC ranking
same LingBot historical depth + PnP
same fixed 2.5 m bearing adapter + frozen NavDP
                    |
          only authority differs
       strict certificate vs finite PnP pose
```

The diagnostic arm is named `mono_unthresholded_witness`.  It is not
retrieval-only and not geometry-free.  Fewer than eight matches still cannot
form a PnP witness; once a finite pose exists, however, the diagnostic arm does
not enforce the 16-inlier, 5%/5% hull, or 2-pixel RMSE certificate thresholds.

Prepared components:

- pure authority contract with strict default;
- server endpoint and evaluator route;
- same-process, order-rotated Final14 runner;
- first-decision proposal-equality audit;
- fail-closed summary and independent raw-receipt verifier;
- one-hour H100/A100 Slurm template;
- immutable-bundle preparation/submission script;
- local tests: 17 new/authority tests, 38 policy-agent tests, and 33 existing
  integration/adapter tests all pass.

Primary output is strict CEC minus unthresholded witness SR by Novel, Revisit,
and all queries.  Secondary outputs are Novel authorization, Revisit rejection,
and first-decision authority discordance.

The original, first portsafe, and provenance-only DAGs are retained only as
incident records.
The current formal DAG was submitted through the verified `yz11502` shared
master: smoke `16504303`, formal array `16504304`, and analysis/verifier
`16504307`.  Its receipt-complete immutable bundle receipt begins
`3f5783aca521b0a5`.

One earlier preparation attempt stopped before upload because a manually
selected responsive socket belonged to another account; the identity gate
correctly prevented any remote write or Slurm job.  This is retained here as
an incident record, not as the current submission state.

## 5. SSH incident, resolution, and safety boundary

Two local control sockets existed.  The newer responsive socket actually
reported `id -un = yz11445`; attaching it to the `alantorch` alias did not
change the master identity.  The intended persistent `yz11502` master remained
the correct shared entry, while one scripted channel was stalled during shell
startup.  The wrong account could enumerate some paths but produced hundreds
of `Permission denied` errors on owner-only bundle files.  No bundle was
uploaded and no Slurm job was created.

Both Final14 preparation scripts now validate a real remote command **and**
exact username before any read/write or `sbatch`.  Details are recorded in
`HPC_SHARED_SSH_OPERATIONS_20260816.md`, Sec. 1.5.

The default assumption is now explicit: the user's shared `alantorch` master is
already authenticated and available.  A stalled no-PTY/scripted channel is
diagnosed through the documented PTY, stdin/job-control, SFTP/SCP, and
ControlPath procedures; it is not evidence that SSH, Torch HPC, or Slurm is
down, and it is not a reason to request another user login.  Remote tests, two
route-level contract dry-runs, both `sbatch --test-only` checks, and immutable
bundle verification all passed before submission.

## 6. Interpretation discipline

- A finite-PnP arm measures the cost/benefit of operational thresholds; it is
  not a proposed deployable method.
- Final14 remains consumed and underpowered relative to the original target;
  it is used for a paired mechanism ablation, not new model selection.
- Continual `99/99` factual-B is construction completeness, not SR.
- The completed anchor-only ViNT formal result is a controlled negative because
  that adapter discarded the certified bearing.  The five-loss ideal-yaw repair
  is mechanism evidence only.  Neither may fill the intended bounded-executor
  controller-portability row.
- Zero-depth Novel-A Gate D and zero-depth Final14 mixed-role results must remain
  separate estimands.
