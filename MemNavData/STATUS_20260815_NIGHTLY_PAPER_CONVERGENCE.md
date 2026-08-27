# Project Status: Nightly Paper Convergence

**Local date:** 2026-08-15  
**Previous full ledger:** `STATUS_20260814_PAPER_EVAL.md`  
**Status of this file:** completed ledger for the frozen 2026-08-15 nightly
objective; all reported Gate-A, Gate-B, and GOAT results passed their stated
independent verifiers.

**Current continuation:** 2026-08-17 的 Pi3X learned relocalizer、
Pi3X/LingBot-Map 角色审计、训练规模与泛化边界，以及 Final14 正式链已汇总到
`STATUS_20260817_PI3X_LEARNED_RELOCALIZER_FINAL14.md`。本文仍是 2026-08-15
冻结夜间目标的历史账，不应被当作当前运行状态。

## 1. Tonight's objective

The project is no longer searching broadly across memory, Novel direction,
learned decoders, controller replacement, and graph rescue. Tonight's work is
restricted to four questions that directly determine paper readiness:

1. Is Phase-2's geometry-versus-DINO discrepancy caused by proposal ordering
   or by the PnP/certificate itself?
2. Does the frozen method survive an external GOAT-Bench runtime and semantic
   arrival contract?
3. Are raw-DINO's Phase-2 Novel gains factual memory information or generic
   exploration perturbations?
4. Can the implementation stack be expressed as one scientific abstraction
   with defensible claims and explicit non-claims?

The paper-level abstraction is:

> Causal visual history proposes an episodic place hypothesis; a geometric
> witness authorizes whether it may change a frozen ImageGoal policy; the only
> transferred control variable is a scale-free bearing; unsupported hypotheses
> retain exact native behavior.

The working method name remains **Certified Episodic Compass (CEC)**.

## 2. Evidence entering tonight

### Revisit utility already established

- Original geometry memory: `4/40 -> 19/40`, paired `+15/-0`, exact McNemar
  `p=6.1e-5`.
- Fresh160 supported Revisit, conditional on shared A success: native `27/120`,
  old geometry `91/120`, raw DINO `106/120`, certified `112/120`.
- Actual-online 3-leg NNR: native `5/19`, role-free certified `16/19`, paired
  `+11/-0`, `p=0.0009766`.
- Fresh20 double Revisit: native joint `0/20`, role-free certified `17/20`.

### Held-out mixed-role evidence

Attempt 7, 9 histories / 9 scenes / 18 queries:

| arm | Novel | Revisit | total |
|---|---:|---:|---:|
| native | 2/9 | 2/9 | 4/18 |
| raw fixed | 1/9 | 8/9 | 9/18 |
| old geometry | 2/9 | 7/9 | 9/18 |
| certified | 2/9 | 8/9 | 10/18 |

Certified versus native was `+6/-0`, `p=0.03125`; Novel had `0/9`
certificate accepts and exact fallback. The population was below its target
size and is underpowered.

Phase-2, 19 histories / 19 scenes / 38 natural-direction queries:

| arm | Novel | Revisit | total |
|---|---:|---:|---:|
| native | 4/19 | 1/19 | 5/38 |
| raw fixed | 9/19 | 18/19 | 27/38 |
| old geometry | 4/19 | 19/19 | 23/38 |
| certified | 4/19 | 17/19 | 21/38 |

Certified versus native was `+16/-0`, `p=3.05e-5`; Novel had `0/19`
takeovers and `19/19` exact fallback. Certified did **not** beat raw fixed:
`+1/-7`, `p=0.0703`. Revisit alone was `17/19` versus `18/19`, `+0/-1`.

### Precise Phase-2 failure motivating tonight's audit

For `r1Q1Z4BcV1o/episode_0005`, raw DINO selected frame 132 and succeeded;
geometry-first CEC selected frame 141 and failed. Frame 141 nevertheless had
522 PnP inliers, `0.934 px` reprojection RMSE, and roughly `41--42%` hull
coverage. Thus local geometric consistency is not identical to semantic goal
relevance or downstream control utility.

The current method remains frozen. The alternative factorization is only a
developmental hypothesis:

```text
DINO semantic order
  -> unchanged PnP/certificate on h1, h2, ...
  -> first accepted hypothesis emits scale-free bearing
  -> none accepted means exact native fallback
```

## 3. Proposal-versus-verification audit

### Scope

- 28 already consumed Revisit histories: Attempt 7 (9) plus Phase-2 (19).
- Same factual causal online-A JPEGs and endpoint views.
- Same DINO shortlist, LightGlue, LingBot depth, PnP, and atomic certificate.
- No query rollout and no SR claim.
- The counterfactual branch is read-only and logs
  `action_authority=false`; deployed geometry-first output remains unchanged.

### Local implementation

- `NavDP/baselines/memnav/memnav_server.py`
- `NavDP/baselines/memnav/policy_agent.py`
- `MemNavData/run_certified_proposal_counterfactual_episode.py`
- `MemNavData/summarize_certified_proposal_counterfactual.py`
- `MemNavData/certified_proposal_counterfactual_manifest.json`
- `MemNavData/slurm_certified_proposal_counterfactual_audit.sbatch`
- `MemNavData/slurm_summarize_certified_proposal_counterfactual.sbatch`

Local policy tests passed `21/21`; the targeted remote action-authority and
summary tests passed `2/2`.

### Infrastructure incidents and repairs

1. The Habitat environment lacked `requests`. The diagnostic runner was
   changed to standard-library JSON/multipart HTTP; the base source root was
   added to `PYTHONPATH`. This changed no method decision.
2. Repair-1 array `15761657` revealed that the frozen base bundle did not
   contain `retrying_server_launcher.py`. Tasks 0--9 failed in 11--18 seconds
   before server startup; the rest and summary `15761658` were cancelled.
   Episode record count was exactly zero.
3. Repair 2 placed the tested launcher in the immutable diagnostic bundle and
   pointed Slurm to that copy. It passed receipt, target unit, launcher CLI,
   and Habitat runner CLI preflight. Smoke `15761873` then reached the first
   reset but failed because the shared policy reset receipt unconditionally
   imports the pure GOAT arrival contract, which repair 2 had not packaged.
   It produced zero records; dependency-blocked formal `15761874` and summary
   `15761875` were cancelled without work.
4. Repair 3 added only `goat_certified_arrival_contract.py`. Its stronger
   preflight directly constructed the immutable agent and successfully called
   both reset-status methods, in addition to the existing target tests and
   dual-environment CLI checks. A Habitat-encoder/Flask-decoder multipart
   round trip passed, and all 28 frozen trace reset seeds were independently
   checked inside the uint32 domain (`20260803--20260821`).

Full provenance is in
`CERTIFIED_PROPOSAL_COUNTERFACTUAL_RUNTIME_REPAIR_20260815.md`.

### Completed repair-3 chain and result

- immutable bundle:
  `/scratch/yz11502/Research/source_bundles/cec_prop_audit_7768fb855e9335ec`
- source receipt SHA:
  `7768fb855e9335ec715d9e073c4928711566684cf5c53f5aa6e0347148638193`
- one-record infrastructure smoke: job `15762219`
- formal 28-record audit, after smoke: job `15762220`
- immutable summary, after formal: job `15762221`
- independent raw-record verifier, after summary: job `15762347`
- immutable verifier bundle:
  `/scratch/yz11502/Research/source_bundles/cec_prop_verifier_c23938db995c4002`
- verifier receipt SHA:
  `c23938db995c40020387255ccd1850368b15eeb93d408e85ac2968318a30d0fc`
- smoke root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/certified_proposal_counterfactual_20260815/smoke_repair3_20260814T220115Z`
- formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/certified_proposal_counterfactual_20260815/formal_repair3_20260814T220115Z`

Repair-3 smoke `15762219` completed in `00:01:30`, exit code zero, and produced
exactly one discarded record after exercising the complete pipeline. Formal
`15762220` then completed all 28 records; summary `15762221` and independent
verifier `15762347` both completed, with `verified=true`. The verifier imports
neither formal runner nor summarizer and re-read every raw record. Summary SHA
is `8c287b12b7261d3d52dc47bf02fe4cce7cb3438fafc67da18e143e4372882c4b`;
verification SHA is
`c4f91978986ca289db2752c031b9e693030aeb8452763f13d21bc420f7cf5769`.

All three proposal factorizations had saturated certificate coverage:

- deployed geometry-first: `28/28`;
- DINO top-1 under the same certificate: `28/28`;
- DINO-order first-certified: `28/28`;
- both paired contrasts: `+0/-0`, exact McNemar `p=1.0`.

Geometry nevertheless changed the DINO top-1 anchor in `21/28` histories. DINO
rank 1 passed immediately in every history, so ordered attempts had mean/max
`1.0/1`. This is not evidence that DINO is better; it shows that the witness
cannot rank two locally self-consistent hypotheses and that downstream
closed-loop utility is the deciding measurement. Full result:
`CERTIFIED_PROPOSAL_COUNTERFACTUAL_RESULT_20260815.md`.

### Frozen decision rule

`SEMANTIC_PROPOSAL_GEOMETRIC_VERIFICATION_DECISION_GATE_20260815.md` was
written before any successful audit smoke produced a record:

- semantic-first certificate coverage must be at least geometry-first coverage;
- paired coverage gains must be no fewer than losses;
- only then may one consumed closed-loop development comparison run;
- only a strict paired closed-loop net gain may promote semantic-first to a
  candidate for a new scene-disjoint confirmation;
- no result from these 28 consumed histories is confirmation evidence.

Observed `S=G=28` and `g=l=0`, with every action-authority invariant verified.
Gate A therefore passes at equality and authorizes exactly one consumed
closed-loop Gate B comparison. It does not promote semantic-first by itself.

### Gate B implementation and submission

Gate B is now implemented as a minimal proposal-order change on top of the
exact Attempt-7 runtime:

```text
geometry-first: LightGlue/F-matrix ranking -> top hypothesis -> PnP certificate
semantic-first: DINO order -> first hypothesis passing the same PnP certificate
both: accepted bearing -> fixed 2.5 m residual -> frozen NavDP
      no accepted hypothesis -> exact native NavDP
```

The experiment evaluates only the Revisit member of each consumed role pair.
`analysis_role` remains evaluator-only and is not forwarded to MemNav or NavDP.
Both arms replay the same byte-verified online-A prefix, use the same seeds,
600-step budget, success radius, controller, candidate set, PnP thresholds,
certificate, and bearing adapter. Arm order alternates by population index.

To prevent an unrelated-code confound, the HPC bundle was rebuilt from the
Attempt-7 immutable parent instead of copying the live worktree. Its machine
audit permits exactly five changed runtime/test files and nine new Gate-B
protocol/execution files; the GOAT arrival implementation is explicitly absent.
Remote compile, shell, legacy geometry-first, cache isolation, fail-closed, and
semantic-order tests passed `25/25`.

- final receipt-repair source bundle:
  `/scratch/yz11502/Research/source_bundles/semantic_proposal_gate_b_runtimecount_380f82590aa03518`
- source receipt SHA:
  `fd0b8c233378b0d004c674639a98089981ddd6a6275347c5bacf27f00f7415db`
- source manifest SHA:
  `380f82590aa03518d861792e053b18efdd92d730260308ed28426ec43ab5bfb2`
- frozen population manifest SHA:
  `0cdabbfb6c3477e5578406a3c8c0ef9e2387e4ad2f8e231fb873b3eece0f3625`
- infrastructure smoke: `15763484`
- 28-history formal array, concurrency four: `15763485`
- exact-index infrastructure repair: `15764888_18`
- repair-aware immutable summary: `15764892`
- independent raw-record verification: `15764893`
- formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/semantic_proposal_gate_b_20260815/formal_consumed_final_20260814T232857Z`

All four jobs were submitted as one dependency chain before reading any smoke
outcome. Smoke `15763484` subsequently completed in `00:02:51`, exit `0:0`;
its discarded record passed schema, prefix-equality, proposal-order, role
isolation, and zero-runtime-failure checks (SHA-256 `79ae733f193534b0a81426d70648e212f12c9116a89c816134ee35ca001fbc0e`).
The formal array, exact-index repair, summary, and independent verifier are now
complete. Geometry-first and semantic-first both achieved `25/28` (`89.29%`),
with paired `+0/-0`, exact McNemar `p=1.0`, 25 joint successes, and three joint
failures. Raw and runtime-penalized outcomes were identical because neither arm
had a runtime-failure plan. The independent verifier returned `verified=true`.
The frozen Gate-B rule therefore fails to promote semantic-first and retains
geometry-first CEC. This is consumed development evidence, never confirmation.
Full result: `SEMANTIC_PROPOSAL_GATE_B_RESULT_20260815.md`; submission receipt:
`SEMANTIC_PROPOSAL_GATE_B_SUBMISSION_RECEIPT_20260815.json`.

The first anchor changed in `21/28` histories, yet a post-hoc all-record
diagnostic found that the first authorized bearings differed by only `0.770°`
on average (`0.413°` median, `4.478°` maximum; all `28/28 <5°`). Both arms
authorized at query step zero in all 28 histories. Thus proposal order changed
frame identity but not the direction delivered to the controller in this
high-support Revisit population.

Formal task 18 (`15764627`, population index `18`) suffered a Habitat native
`SIGABRT` on H200 node `gh133` after `00:19:12`. It used only about `15.5 GiB`
of the requested `96 GiB`, was far below the eight-hour limit, and produced no
second arm or `completion.json`; no Task-18 outcome was read. Its 203 partial
artifacts were preserved under `failed_attempts/` with receipt SHA-256
`b85f5e110dae618203321ed259e8305cf1cf2c339aa829230260ffd71471115d`.
Only index 18 was rerun, on the unchanged frozen inputs but restricted to
H100/A100 partitions; repair `15764888_18` completed on `gh014` in `00:02:59`.
The stale original summary/verifier `15763486/15763487`, which did not depend
on the exact-index repair, were cancelled before execution and replaced by
`15764892/15764893`, both of which completed successfully. Full incident receipt:
`SEMANTIC_PROPOSAL_GATE_B_TASK18_INCIDENT_20260815.md`.

Two outcome-blind protocol repairs preceded the final chain. First, the raw
physical success is now retained separately while any certificate-endpoint
runtime failure conservatively makes that arm's Gate-B outcome zero, as the
frozen rule required. Second, smoke `15763288` ran both arms but produced no
completion because `proposal_order` was present in the planner response yet
missing from the serialized per-plan audit record. Only that diagnostic copy
was added. Jobs `15763289--15763291` were cancelled before execution; no formal
record existed and no method outcome was read. The replacement smoke/formal
chain above is therefore the only eligible Gate-B run.

One final pre-execution audit broadened `runtime_failure_plans` from the narrow
HTTP-exception counter to every saved plan with
`certified_relocalization_ok=false`, while retaining the endpoint count as a
separate diagnostic. Pending jobs `15763428--15763431` were cancelled at
`0:00`; the method, controller, thresholds, population, and raw physical
outcomes were unchanged. Smoke `15763484`, formal array `15763485`, exact-index
repair `15764888_18`, and repair-aware postprocessing `15764892/15764893` are
the resulting auditable execution chain.

## 4. GOAT external semantic-arrival confirmation

### Exact scope

This is a frozen 20-scene, first-ImageGoal semantic-arrival confirmation using
GOAT-Bench assets and runtime. It is **not** a full sequential GOAT score and
does not cover ObjectGoal or LanguageGoal subtasks.

Native NavDP proposes motion. The certified sidecar may authorize arrival only
under the frozen `0.075 m` scale/geometry contract. Habitat ground truth is
used for scoring, not for decisions.

### Failed parent and repair

Parent smoke `15759008` failed before the first observation because its frozen
63-bit episode hash was sent directly to MemNav's legacy NumPy reset path.
Formal and post-processing ran zero work.

Repair-only change:

```text
service_reset_seed = frozen_episode_hash % 2**32
```

The same uint32 value is sent to NavDP and MemNav. Per-plan 63-bit diffusion
seeds stay unchanged. Machine comparison proved exactly one existing source
file changed and exactly two provenance/test files were added. Remote tests
passed `23/23`.

### Completed formal result

- immutable child bundle:
  `/scratch/yz11502/Research/source_bundles/goat_certified_arrival_bc3c3c887d1063ee`
- source receipt SHA:
  `bc3c3c887d1063ee600254fea0f1533118fcdc6423044463775b2feb33c355d1`
- smoke: `15761753`, completed successfully in `00:02:27`;
- 20-scene formal: `15761754`, all 20 records completed, exit `0:0`;
- sealed summary: `15761755`, completed, exit `0:0`;
- independent verifier: `15761756`, completed, `verified=true`;
- formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_certified_arrival_20260815/formal_seedrepair_intent_20260814T213723Z`.

Aggregate result:

- certified success `0/20`;
- certified STOPs `0`, true STOPs `0`, false STOPs `0`;
- all 20 episodes ended through the forced guard;
- legacy first-zero counterfactual success `1/20`;
- paired certified-minus-legacy `+0/-1`, McNemar `p=1.0`;
- preregistered gate failed because it required at least five true certified
  stops in at least five scenes while retaining zero false stops.

The zero false-stop count is vacuous at zero certified coverage. Across 28
native-zero events, 26 were outside the official `<0.25 m` arrival region and
the median official distance was `6.219 m`. Six events lacked the frozen
64-frame scale prefix; of the remaining 22, 19 failed the epipolar precheck,
three reached PnP, and only two passed the geometric certificate. Both passed
matches were still far from the goal and correctly exceeded the frozen
distance threshold. The two genuinely arrived events both failed the local
geometry precheck.

Therefore this confirmation rejects the proposed deployable first-ImageGoal
semantic-STOP adapter. It does not evaluate or negate CEC's causal Revisit
retrieval and bearing takeover. No threshold or certificate parameter may be
retuned on these held-out episodes. Full provenance and the failure audit are
in `GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md`; report SHA is
`d52a8dc611058fc3c3a454a7ec38609d913a7b949eed25639a63fd6ba59e1a88`, and
independent-verification SHA is
`4c9a4225b7410364fe3d17580dfee4810f8e5bc6b8945be1147bf8f94a094539`.

A pre-completion health check had opened formal `episode_00.json`; that partial
read remains sealed in `partial_monitoring_receipt.json` (SHA
`ac22cae2cceb54c5b735cf323ff5de04babd5fa69538d81ed57588335350b722`). No
method, threshold, manifest, population, or code was changed from it.

## 5. Novel causal control frozen tonight

Phase-2 raw fixed's `9/19` Novel successes cannot be called Novel localization
without controls. `NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL_20260815.md`
freezes a fresh scene-disjoint, at-least-40-episode experiment with four arms:

1. native frozen NavDP;
2. raw factual history;
3. raw deranged sidecar history while keeping NavDP's FIFO factual;
4. raw proposal availability with only its angle replaced by a deterministic
   random bearing.

The primary contrasts are factual versus randomized bearing and factual versus
deranged history. Their interpretation was frozen before any new outcome:
factual superiority supports history-specific information; equality supports
the exploration-perturbation explanation; intervention losses support exact
fallback. This experiment cannot establish that CEC solves Novel navigation.

## 6. Paper method and claims

The detailed paper plan is
`PAPER_METHOD_STORY_AND_EVAL_PLAN_20260815.md`.
Paper-ready method prose and the pre-submission risk audit are in
`PAPER_METHOD_DRAFT_20260815.md` and
`PAPER_REVIEWER_RISK_REGISTER_20260815.md`. The population-by-population result,
null-result, claim, and pending-experiment ledger is
`PAPER_EVIDENCE_MATRIX_20260815.md`; it explicitly forbids pooling consumed
populations to manufacture significance.

Defensible claims now:

1. causal online episodic history gives large, repeated closed-loop Revisit
   gains to frozen NavDP;
2. a scale-free bearing is sufficient for the tested supported Revisit tasks;
3. role-free geometric authorization can abstain on unsupported Novel queries
   and preserve exact native behavior in the held-out tests;
4. learned replacement attempts failed for identifiable reasons: long-range
   content addressing and actionable geometric coverage, not simply too few
   training epochs.

Claims still forbidden:

- CEC significantly exceeds raw fixed on Revisit SR;
- the certificate guarantees zero false positives;
- Novel direction is deployably solved;
- semantic-first improves CEC; consumed Gate B tied exactly and did not promote
  it;
- the GOAT chain is a full GOAT sequential benchmark score;
- X-NavDP or a learned decoder significantly improves the frozen controller.

The contribution is therefore not “DINO + LightGlue + PnP + LingBot.” It is the
evidence-carrying authorization interface, direction-only control abstraction,
exact fallback contract, and causal mixed-role evaluation that makes utility
and interference separately measurable.

## 7. Next decisions, in order

1. Freeze the completed GOAT arrival result as a failed external transfer; do
   not retune it. Any future GOAT protocol must test the actual sequential
   Revisit-bearing claim rather than first-goal STOP authorization.
2. Freeze Gate B as a paired null: both proposal orders reached `25/28`, paired
   `+0/-0`; retain geometry-first CEC and do not launch a semantic-first
   confirmation.
3. Use the all-record bearing-alignment diagnostic only to explain the null:
   different co-visible anchors collapsed to nearly identical scale-free
   directions. Do not turn the post-hoc diagnostic into a superiority claim.
4. Never report these 28 consumed histories as confirmation evidence and do
   not tune proposal order from per-episode outcomes.
5. Implement and smoke the Novel causal control only after its current written
   contract is independently audited; do not retrofit it to Phase-2 as new
   confirmation.
6. Build the next prospective mixed-role confirmation around Revisit utility,
   Novel interference, coverage, and exact fallback, not another top-1 ranker.
7. Build the paper around Revisit utility + authorization + minimal interface,
   while treating the failed GOAT arrival experiment and the Novel controls as
   boundary evidence and causal explanation rather than extra modules.

## 8. 2026-08-16 addendum: Attempt 7 versus Phase-2 raw Novel

A read-only all-record audit has now resolved the apparent reversal without
running another episode. Attempt 7 and Phase-2 used the same evaluation stack,
checkpoints, 600-step budget, execution horizon, success radius, deterministic
seed contract, and hidden-role interface. The Phase-2 Slurm change only
generalized array size. Implementation drift is therefore not the explanation.

Correct population accounting is Attempt 7 `9 histories / 9 scenes` and
Phase-2 `19 histories / 12 scene clusters`, not 19 scenes. Raw fixed versus
native was:

- Attempt 7: `1/9` versus `2/9`, paired `+1/-2`, `p=1.0`;
- Phase-2: `9/19` versus `4/19`, paired `+6/-1`, `p=0.125`.

The Phase-2 contrast is not significant. However, exact reconstruction from
evaluator-logged goal distances shows that all six Phase-2 raw gains received
a first direction within `2.82--24.08 deg` of the direct goal bearing, whereas
the one loss had `110.74 deg` error. This is not arbitrary CUDA noise.

The underlying cohort effect is directional. Raw first bearings collapse near
the rear (`R=.932`, mean `166.1 deg` in Attempt 7; `R=.840`, mean `176.5 deg`
in Phase-2), while the correct shortest-path direction was behind the agent in
`7/9` versus `16/19` queries. Phase-2 therefore aligned unusually well with the
raw head's U-turn-heavy inductive bias.

A 100,000-derangement static diagnostic shows that Phase-2 factual bearings
are better than a constant U-turn or bearings swapped across query identities.
This implies query-specific information, but does not isolate whether it comes
from the causal history, goal image, current image, or their interaction. The
natural-direction builder also leaves route bearing unmatched (`180 deg`
tolerance) and couples rendered goal yaw to endpoint-to-goal direction.

Decision: do not call raw Novel a method result, and do not immediately launch
the expensive 600-step four-arm causal control. First run a consumed,
proposal-only factorial that holds current/goal inputs fixed and deranges the
history's image-to-pose association. Only a clear factual-history advantage
promotes a fresh, bearing-stratified closed-loop confirmation. Main paper
priority remains Revisit CEC and its final prospective evaluation.

Full audit and reproducible artifacts are in
`NOVEL_RAW_COHORT_SHIFT_AUDIT_20260816.md` and
`.diagnostics/raw_novel_cohort_audit_20260816/report.json`.

## 9. 2026-08-16 addendum: forced-anchor attribution resolves the branch

The proposal-only attribution has now completed on all 19 consumed Phase-2
Novel queries spanning 12 scene clusters.  It replayed the exact factual
online-A RGB stream, current image and goal image, then compared the completed
raw-DINO anchor with 12 identity-seeded uniformly sampled legal anchors per
query.  The frozen MemNav/LingBot proposal endpoint was used; Habitat and the
untouched final14 were not accessed.

Replay validity is strong: the local RTX 4090 factual bearing differed from the
HPC record by `1.079 deg` on average, median `0.804 deg`, maximum `3.656 deg`.

Primary shortest-path result:

- factual DINO mean error `38.630 deg`;
- random-anchor expected mean error `42.778 deg`;
- factual advantage `+4.148 deg`;
- 100,000-resample scene-cluster CI `[-1.357,+8.898] deg`;
- useful `<=30 deg` coverage `10/19` versus `9.75/19` expected.

Secondary exact direct-goal result:

- factual advantage `+2.169 deg`;
- scene-cluster CI `[-3.710,+7.069] deg`;
- useful coverage `10/19` versus exactly `10.0/19` expected.

The promotion rule required a cluster-CI lower bound above zero.  It failed.
Combined with the all-eligible physical-anchor audit (`+0.048 deg`, CI
`[-2.522,+3.531] deg`), the result says that DINO does not select a
historically better route location.  LingBot goal insertion occasionally
adds a small useful correction, but it is heterogeneous and does not improve
the `30 deg` deployability threshold.

Frozen decision:

`stop_novel_dino_branch_and_preserve_final14_for_cec_confirmation`

Therefore the goal-shuffle diagnostic and paused 600-step Novel four-arm
control will not run.  Raw Novel remains explanatory boundary evidence, not a
paper method.  The next prospective population is reserved for CEC's actual
claim: role-free Revisit utility, Novel interference control, and exact native
fallback.  Full details and hashes are in
`NOVEL_RAW_FORCED_ANCHOR_ATTRIBUTION_RESULT_20260816.md`.

## 10. Final14 protocol frozen; support-band preflight passed

After the Novel-DINO stop decision, the final prospective question was frozen
without opening the final14 population.  Protocol:

`FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md`

SHA-256:

`3d1ebc6ef429fd16df4d550eda52eceb55d7b15fd181a5c00c0b8f971f7aaa32`

The final design tests CEC's actual claim rather than another Novel adapter:

- role-free standard Revisit utility versus native;
- unsupported Novel interference and exact fallback;
- risk--coverage versus always-on raw fixed and old geometry;
- a distinct hard-support Revisit band;
- Novel front/side/rear stratification and goal-yaw decoupling, eliminating
  the Phase-2 construction confounds.

Before final14 access, a renderer-only constructibility smoke ran on four
already consumed actual-online histories, considering at most two source
frames per history.  Standard Revisit was constructible in `4/4`; hard-support
Revisit in `3/4`.  Selected max-covis ranges were `0.7169--0.7573` and
`0.4626--0.5328`, respectively, confirming distinct support bands rather than
renaming the previous saturated benchmark.  No policy was called.

Report:

`.diagnostics/final14_support_band_constructibility_smoke_20260816/report.json`

SHA-256:

`4e2dba2fb88fee5c206577bd25e4820e213d0cb4e62357974d0fa6d2775d4797`

Next integrity gate: implement the immutable builder/evaluator and complete an
end-to-end dry-run only on consumed scenes.  Final14 remains sealed until that
dry-run and its independent contract audit pass.
