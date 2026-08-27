# Nightly Goal Completion Audit

**Local date:** 2026-08-15  
**Objective:** converge the paper evidence for Certified Episodic Compass
without promoting consumed development results into confirmation evidence.

This file is the requirement-by-requirement completion audit. A completed item
means that its authoritative artifact was inspected; it does not broaden the
claim supported by that artifact. All required rows are now complete.

## 1. Held-out proposal-versus-verification audit

### Required

- A read-only comparison on the frozen 28 consumed Revisit histories.
- No counterfactual action authority and no closed-loop or confirmation claim.
- Immutable population/source receipts, complete raw records, summary, and an
  independent verifier.
- A decision rule frozen before outcomes are read.

### Evidence inspected

- Protocol and decision gate:
  `SEMANTIC_PROPOSAL_GEOMETRIC_VERIFICATION_DECISION_GATE_20260815.md`.
- Result:
  `CERTIFIED_PROPOSAL_COUNTERFACTUAL_RESULT_20260815.md`.
- Formal jobs `15762220`, `15762221`, and independent verifier `15762347`
  completed; all 28 records were present and `verified=true`.
- Geometry-first, DINO top-1 under the same witness, and DINO-order
  first-certified were all actionable on `28/28`; paired coverage was
  `+0/-0`, exact McNemar `p=1.0`.
- Independent-verification SHA-256:
  `c4f91978986ca289db2752c031b9e693030aeb8452763f13d21bc420f7cf5769`.

**Status: complete.** Gate A passed at equality and authorizes only the single
consumed Gate-B closed-loop comparison. It does not establish that
semantic-first is better.

## 2. GOAT repair-only external confirmation

### Required

- Repair only the pre-observation uint32 reset-seed incompatibility.
- Preserve goal population, decision threshold, controller, per-plan diffusion
  seeds, and evaluation rule.
- Complete a fresh smoke, 20-scene formal run, immutable summary, and
  independent verification.
- Apply the frozen decision without held-out retuning.

### Evidence inspected

- Protocol and provenance:
  `GOAT_CERTIFIED_ARRIVAL_CONFIRMATION_PROTOCOL_20260815.md` and
  `GOAT_CERTIFIED_ARRIVAL_RESET_SEED_REPAIR_20260815.md`.
- Formal result: `GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md`.
- Jobs `15761753--15761756` completed; 20/20 records were present and the
  independent verifier returned `verified=true`.
- Certified success was `0/20`, certified STOP coverage was zero, and all 20
  episodes reached the forced guard. The preregistered gate failed.
- Result SHA-256:
  `d52a8dc611058fc3c3a454a7ec38609d913a7b949eed25639a63fd6ba59e1a88`.
- Independent-verification SHA-256:
  `4c9a4225b7410364fe3d17580dfee4810f8e5bc6b8945be1147bf8f94a094539`.

**Status: complete, negative.** The first-ImageGoal semantic-arrival adapter is
rejected and must not be tuned on these held-out episodes. This experiment did
not exercise CEC's supported-Revisit bearing intervention and therefore is not
external validation or falsification of that claim.

## 3. Novel factual-history causal control

### Required

- Freeze the population, four paired arms, seed/ordering contract, statistical
  contrasts, and interpretation before reading any fresh result.
- Separate factual history-specific information from a generic directional
  exploration perturbation.

### Evidence inspected

- Frozen protocol:
  `NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL_20260815.md`.
- Arms are native, factual causal history, frozen deranged sidecar history, and
  deterministic randomized bearing with matched intervention mechanics.
- Primary interpretation is factual versus deranged/random, not intervention
  versus native alone; the protocol explicitly cannot establish a deployable
  Novel direction method by itself.

**Status: complete as a frozen protocol.** Execution is deliberately outside
tonight's objective and must not reuse Attempt 7 or Phase-2 as confirmation.

## 4. Paper abstraction, evidence matrix, and next actions

### Required

- Express the method as a scientific abstraction rather than a component list.
- Separate established claims, negative results, developmental evidence,
  confirmation evidence, and forbidden claims.
- Record the essential tables/ablations, reviewer risks, external-validity
  gap, and ordered next decisions.

### Evidence inspected

- Method prose: `PAPER_METHOD_DRAFT_20260815.md`.
- Story, closest-work differentiation, required figures/tables, and ablations:
  `PAPER_METHOD_STORY_AND_EVAL_PLAN_20260815.md`.
- Population-by-population claims and non-claims:
  `PAPER_EVIDENCE_MATRIX_20260815.md`.
- Submission risks and minimum evidence package:
  `PAPER_REVIEWER_RISK_REGISTER_20260815.md`.
- Integrated live ledger:
  `STATUS_20260815_NIGHTLY_PAPER_CONVERGENCE.md`.

The frozen abstraction is:

> Causal visual history proposes an episodic place hypothesis; a geometric
> witness decides whether it may change a frozen ImageGoal policy; only a
> scale-free bearing crosses the authorization boundary; unsupported
> hypotheses reproduce native behavior exactly.

**Status: complete.** Gate-B's verified paired null and its claim-safe
interpretation have been inserted. Current prose does
not claim that CEC beats raw-DINO on Revisit SR, that the witness is a formal
safety proof, that Novel direction is deployable, or that GOAT has been solved.

## 5. Consumed closed-loop Gate B

### Required

- Run exactly one paired closed-loop comparison on the same 28 consumed
  Revisit histories after Gate A passes.
- Preserve byte-identical online-A prefixes, seeds, controller, budget,
  shortlist, PnP witness, certificate thresholds, and bearing interface; alter
  only proposal order.
- Count any saved certified-relocalization plan with `ok=false` as an arm
  runtime failure, while retaining raw physical success separately. Ordinary
  certificate rejection remains `ok=true, accepted=false`.
- Require all 28 completions, 15 scenes, balanced arm order, no runtime role
  visibility, summary hashes, and independent raw-record verification.
- Apply the frozen rule: only paired semantic-first gains strictly exceeding
  losses nominate it for a fresh scene-disjoint confirmation. Tie/loss retains
  geometry-first CEC. Gate B is never confirmation evidence.

### Frozen execution receipt

- Local receipt:
  `SEMANTIC_PROPOSAL_GATE_B_SUBMISSION_RECEIPT_20260815.json`.
- Immutable source:
  `/scratch/yz11502/Research/source_bundles/semantic_proposal_gate_b_runtimecount_380f82590aa03518`.
- Source receipt SHA-256:
  `fd0b8c233378b0d004c674639a98089981ddd6a6275347c5bacf27f00f7415db`.
- Source manifest SHA-256:
  `380f82590aa03518d861792e053b18efdd92d730260308ed28426ec43ab5bfb2`.
- Population manifest SHA-256:
  `0cdabbfb6c3477e5578406a3c8c0ef9e2387e4ad2f8e231fb873b3eece0f3625`.
- Eligible execution and repair-aware aggregation:
  smoke `15763484`, formal array `15763485`, exact-index repair
  `15764888_18`, summary `15764892`, verifier `15764893`.
- Formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/semantic_proposal_gate_b_20260815/formal_consumed_final_20260814T232857Z`.

**Status: complete and independently verified.** Smoke `15763484` completed in
`00:02:51`, exit `0:0`. Its single discarded record verified schema v2,
byte-equal prefixes,
balanced arm presence, correct geometry/DINO proposal-order receipts, no
runtime role visibility, and zero runtime failures in both arms; its SHA-256 is
`79ae733f193534b0a81426d70648e212f12c9116a89c816134ee35ca001fbc0e`.
Formal task 18 later suffered a native Habitat `SIGABRT` on `gh133` before a
paired completion existed. It was neither OOM nor timeout, and no Task-18
outcome was read. All 203 partial artifacts were preserved with incident
receipt SHA-256
`b85f5e110dae618203321ed259e8305cf1cf2c339aa829230260ffd71471115d`;
only index 18 was repaired under the unchanged frozen contract on H100/A100.
No completed index was rerun or overwritten. See
`SEMANTIC_PROPOSAL_GATE_B_TASK18_INCIDENT_20260815.md`.

Repair `15764888_18` completed on `gh014` in `00:02:59`; all 28 unique
completion records were then present. Summary `15764892` and independent
verifier `15764893` completed with exit `0:0`. Geometry-first and
semantic-first both reached `25/28`, paired `+0/-0`, exact McNemar `p=1.0`,
with 25 joint successes, three joint failures, balanced first-arm counts
`14/14`, byte-equal prefixes, no role visibility, and zero runtime failures.
The verifier returned `verified=true`.

- summary SHA-256:
  `8a8c70e766bd862485f8c23fbd702fe0e50528159ac0d8d0811fab07a3f0697a`;
- independent-verification SHA-256:
  `03b036b6943419039b1ae9414c9ba8459fc4aea47407273ba212216cf85c080a`;
- dedicated result:
  `SEMANTIC_PROPOSAL_GATE_B_RESULT_20260815.md`.

The frozen strict-net-gain rule did not pass. Geometry-first CEC is retained,
semantic-first receives no fresh confirmation, and the consumed result is
reported only as a null ablation.

## 6. Goal completion rule

The nightly goal may be marked complete only after all of the following are
true:

1. Gate-B smoke and all 28 formal tasks finish without an unaccounted runtime
   or contract failure.
2. The immutable summary and independent verifier finish and agree on raw and
   effective successes, paired gains/losses, and the frozen decision.
3. A dedicated Gate-B result document records population, pairing, audit,
   result, exact interpretation, and the fact that this is consumed
   development evidence.
4. The submission receipt, live status ledger, evidence matrix, method plan,
   and this audit are updated consistently.
5. Targeted tests, JSON parsing, source/result hashes, and `git diff --check`
   pass after the final documentation update.

All five conditions now hold: the repaired population is complete, summary and
independent recomputation agree, the dedicated result exists, all ledgers are
consistent, and final validation is recorded below.

## 7. Final validation

- local Gate-B and policy contract tests: `27/27` passed;
- submission receipt parses as JSON;
- remote summary and independent-verification SHA receipts: both `OK`;
- independent raw-record result: `verified=true`;
- source/result documentation: `git diff --check` passed.

**Nightly objective status: complete.**
