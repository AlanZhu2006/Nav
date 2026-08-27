# Nav / CEC workspace main-release ledger — 2026-08-28

This ledger records the repository boundary, release contents, validation state, and
scientific status at the point where `feat/memnav-graph-blind-20260806` is integrated
into `main` and published to `AlanZhu2006/Nav`. It is a release/navigation document, not
a substitute for a frozen protocol, sealed summary, or independent raw-file verifier.

Evidence priority remains:

```text
independent raw-file verifier
  > sealed summary and frozen protocol
  > paper/EVIDENCE_LEDGER.md
  > this release ledger
  > partial scheduler state or verbal status
```

## 1. Version-control boundary

The `AlanZhu2006/Nav` repository contains the simulation/research implementation:

- `NavDP/`: frozen-policy inference and runtime adapters;
- `InternNav/`: training/model definitions retained for reproducibility;
- `MemNavData/`: CEC protocols, evaluators, audit chain, HPC orchestration, and receipts;
- `deployment/`: repository-side deployment contracts and bridges.

The following workspace-owned data are intentionally not part of this repository:

- `paper/`: the local ICRA authoring workspace, excluded by the root `.gitignore`;
- `/home/asus/Research/Memnav_Realworld`: the independent real-robot repository;
- `.diagnostics/`, model weights, datasets, scene assets, generated evaluation outputs,
  and root `eval2leg_results*` directories.

No checkpoint, scene asset, generated rollout, credential, private key, or SSH socket is
included in this release. HPC submission receipts are small provenance records and are
versioned deliberately.

## 2. Frozen method boundary

The primary system remains **Certified Episodic Compass (CEC)**:

```text
causal monocular RGB
  +-- dense short-range LingBot readout -> causal scaled depth -> frozen NavDP
  +-- sparse long-range history
        -> DINO proposal
        -> SuperPoint/LightGlue + Fundamental-MAGSAC evidence
        -> LingBot historical depth + PnP witness
        -> operational certificate
             accept: unit bearing + fixed 2.5 m residual
             reject: exact native controller request
```

CEC does not receive a Novel/Revisit label, does not output actions, and does not become
a second planner. NavDP remains the only trajectory policy. “Training-free” means no new
task-specific optimization; all constituent learned models are pretrained and frozen.

The current paper phrase is:

> one causal stream, two time scales, one frozen policy

The method principle is:

> proof before control

## 3. Contents added since the previous release

### 3.1 Authority and evidence isolation

- A pure CEC handoff contract separates proposal, finite geometric witness, certificate,
  and control authority.
- Final14 authority ablation compares strict CEC with an unthresholded finite-PnP witness
  while holding proposal, geometry, bearing adapter, controller, and paired query fixed.
- Final14 zero-depth evaluation targets the exact mixed-role factorial population; the
  older `23/40` Novel-A zero-depth result is explicitly rejected as a substitute.
- The certificate evidence ladder is independently recounted from raw closed-loop
  records and distinguishes high-recall pose availability from high-precision authority.

### 3.2 Controller portability

- The controller interface now supports exact native rejection and proof-bound accepted
  handoff without treating CEC as a controller.
- ViNT receives an authenticated historical anchor ImageGoal only on accept; on reject it
  executes native ViNT exactly.
- Grant/forced-reject arms run in the same process with order rotation and per-history
  audits. A formal 28-history/56-query HM3D ViNT comparison was submitted under a frozen
  protocol; no partial outcome is promoted by this release.
- Earlier all-CEC portability smokes remain interface tests, not comparative SR.

### 3.3 Full-mono lifelong memory

- The shared-prefix treatment is now explicit:
  `A Novel -> B Novel -> C Revisit -> B2 Revisit -> C2 Revisit`.
- `all_prior`, `initial_leg_only`, and `forced_reject_native` replay hash-identical factual
  prefixes; the primary estimand is B2 after a common C prefix.
- Result-blind constructibility audit showed why the first expansion could not reach its
  power gate. The replacement Natural-B construction produced 99 candidates over 35
  scene clusters and passed independent materialization verification.
- Factual-B collection is preserved as 99 immutable completion receipts. Prefix
  construction, population sealing, and independent verification are dependency-gated;
  C/B2/C2 navigation outcomes are a later formal stage and are not claimed here.

### 3.4 Operational hardening

- Immutable source bundles, local and remote self-tests, route contract dry-runs,
  safe-partition/QOS lint, sealed array bounds, exact-index repair, paired execution,
  summary, and independent verification remain mandatory.
- The persistent shared `alantorch` SSH master is the default authenticated entry. A
  stalled scripted channel is diagnosed as a PTY/stdin/ControlPath issue first and must
  not be generalized to “HPC is down.” Any socket used for a state-changing operation is
  transparently verified as `yz11502`; another account's responsive socket is never a
  fallback.

## 4. Scientific evidence included in the repository

This release does not replace the evidence ledger. The current established results remain
those recorded in `paper/EVIDENCE_LEDGER.md` and the corresponding MemNavData result files,
including:

- Final14 mixed-role proof-before-control;
- MP3D supported-Revisit full-mono composition;
- fresh HM3D full-mono mixed Novel/Revisit confirmation;
- actual-online N--N--R and the verified 18-episode lifelong dose response;
- certificate support/evidence ladders and controlled negative results.

The following are not yet paper results at release time:

- the running formal ViNT/ViNT+CEC paired array until aggregation and independent
  verification both pass;
- Natural-v4 `99/99` factual-B completeness, which is population construction rather
  than C/B2/C2 SR;
- the prepared Final14 zero-depth and authority-ablation arms until formally submitted,
  summarized, and independently verified.

## 5. Release validation

The complete changed-file release gate was run from the repository root:

- secret-pattern scan: no credential/private-key match;
- large-file scan: no changed or untracked file above 1 MB;
- Python syntax: 65 changed/new files passed `py_compile`;
- JSON: 20 new protocol/receipt files parsed successfully;
- Shell/Slurm: 40 changed/new scripts passed `bash -n`;
- MemNav interpreter: 195 tests passed;
- Habitat interpreter: 8 renderer/construction tests passed;
- `git diff --check`: passed.

The split test environments are intentional: renderer-facing construction modules require
the Habitat interpreter and its `quaternion` dependency; runtime/audit modules use the
MemNav interpreter.

## 6. Merge and publication policy

The feature branch and local `main` have independent descendants of a shared base. The
release therefore uses an explicit non-destructive merge rather than resetting either
history. Existing untracked local evaluation directories in the `main` worktree are
preserved and ignored; no generated result is deleted.

Publication sequence:

1. commit and push the feature branch as a recovery point;
2. merge it into the checked-out `main` worktree, resolving overlapping legacy training
   and current runtime files by preserving both valid contracts;
3. rerun merge-sensitive tests;
4. push `main` to `fork` (`AlanZhu2006/Nav`) and set local `main` to track `fork/main`;
5. verify the remote commit and a clean tracked worktree.

## 7. Immediate scientific continuation

1. Finish and independently verify the formal HM3D ViNT native/CEC comparison.
2. Seal the Natural-v4 prefix population, then submit the frozen three-arm C/B2/C2
   evaluation; only the verified result can populate the paper's continual table.
3. Run the already prepared Final14 zero-depth and authority-boundary ablations without
   reopening threshold selection.
4. Add SPL/path aggregation only after each SR table closes.
5. Complete real-robot closed-loop trials in the separate deployment repository.
