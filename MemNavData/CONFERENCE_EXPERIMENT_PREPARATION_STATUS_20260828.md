# Conference experiment preparation status — 2026-08-28

This document records infrastructure and protocol state only.  A staged arm,
completed construction receipt, or scheduler state is not a navigation result.
Paper numbers remain governed by raw episode receipts, a frozen summary, and an
independent verifier.

## 1. Current priority order

1. Finish the already-running HM3D ViNT/ViNT+CEC formal comparison.
2. Resume the frozen HM3D continual Natural-V4 chain after factual-B repair.
3. Run Final14 zero-depth native on the exact mono-factorial query population.
4. Run the Final14 proposal-matched CEC authority ablation.
5. Add path/SPL aggregation only after the four result tables above close.

No new learned model, threshold search, development-set selection, or Novel
direction module is introduced by this queue.

## 2. HM3D continual repair

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

The last path-count/scheduler check was read-only.  Final scientific promotion
still requires the deferred launcher to recompute all completion SHA files,
seal the population, and pass the independent population verifier.

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

Remote immutable bundle already staged and `sbatch --test-only` passed:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  final14_zero_depth_e2033c8a0771cd84
```

Formal submission remains intentionally off.  Start it with:

```bash
SUBMIT=1 bash MemNavData/prepare_final14_zero_depth_hpc.sh
```

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

The formal GPU job has **not** been submitted.  One preparation attempt stopped
before upload because a manually selected responsive socket belonged to another
account; the identity gate correctly prevented any remote write or Slurm job.
The persistent shared `alantorch` connection has since been verified as
`yz11502`, so the remaining action is to rerun the documented preparation chain
through that shared master.

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
down, and it is not a reason to request another user login.  Rerun through the
verified shared master:

```bash
SUBMIT=0 bash MemNavData/prepare_final14_authority_ablation_hpc.sh
```

Only after remote tests, two route-level contract dry-runs, and both
`sbatch --test-only` checks pass should `SUBMIT=1` be considered.

## 6. Interpretation discipline

- A finite-PnP arm measures the cost/benefit of operational thresholds; it is
  not a proposed deployable method.
- Final14 remains consumed and underpowered relative to the original target;
  it is used for a paired mechanism ablation, not new model selection.
- Continual `99/99` factual-B is construction completeness, not SR.
- ViNT smoke results do not become controller SR; only the formal frozen array
  and its verifier may fill the main table.
- Zero-depth Novel-A Gate D and zero-depth Final14 mixed-role results must remain
  separate estimands.
