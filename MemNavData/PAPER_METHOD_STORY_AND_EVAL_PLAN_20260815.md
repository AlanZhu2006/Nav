# From Engineering Stack to Paper Method: Episodic Compass

**Date:** 2026-08-15  
**Status:** paper-planning document; every statement below is marked as
established, developmental, or unverified.  This document does not promote a
post-hoc diagnostic into a result.

## 1. The honest scientific question

The project should no longer be described as “LingBot + DINO + LightGlue + PnP
+ NavDP.”  That description is implementation-level and invites the valid
reviewer objection that the method is an engineering pipeline.

The paper-level question is:

> **When should causal visual experience be allowed to change the action of a
> frozen ImageGoal navigation policy?**

The proposed answer is a control abstraction rather than a new feature stack:

> Treat episodic recall as open-set hypothesis testing.  Visual history may
> propose a previously observed place, but it receives action authority only
> when the current goal supplies a geometric witness.  The authorized output is
> a scale-free bearing, not a map, metric waypoint, or replacement policy;
> unsupported hypotheses fall back exactly to the frozen policy.

This framing explains all four parts of the system:

1. **content addressing** asks which historical observation may depict the
   goal;
2. **verification** asks whether that hypothesis is supported by the two
   images;
3. **minimal control interface** transfers only the directional information
   already shown to be useful;
4. **exact fallback** prevents an unsupported memory hypothesis from replacing
   native exploration.

The contribution is not that image retrieval or geometric verification is new.
Both are standard in visual localization.  The candidate contribution is the
use of a geometrically witnessed, scale-free episodic compass as an
**authorization boundary for a frozen generative navigation policy**, together
with a role-free continual evaluation that separates Revisit utility from
Novel interference.

## 2. What is already established

### 2.1 Causal online Revisit memory has large closed-loop utility

- Original paired geometry-memory result: `4/40 -> 19/40`, paired `+15/-0`,
  exact McNemar `p=6.1e-5`.
- Fresh160 supported-Revisit, conditional on shared Goal-A success:
  native `27/120`, old geometry `91/120`, raw-DINO direct `106/120`, certified
  `112/120`.
- Actual-online 3-leg NNR: native `5/19`, certified `16/19`, paired `+11/-0`,
  `p=0.0009766`, scene-cluster CI `[+27.8,+85.7] pp`.
- Fresh20 double-Revisit: native joint `0/20`, role-free certified `17/20`.

These results support the statement that the agent's own causal visual history
can repair a frozen controller.  They do not require expert-history leakage:
Fresh160 has `120/120` conditional queries with online-A co-visibility at least
`0.20`, `115/120` at least `0.50`, and median maximum co-visibility `0.898`.

### 2.2 Role-free abstention is operationally real

On phase-2 natural-direction held-out data:

- Revisit: certified `17/19` versus native `1/19`;
- Novel: `0/19` certificate accept, `0/19` takeover, and `19/19` exact native
  fallback;
- all queries: certified `21/38` versus native `5/38`, paired `+16/-0`,
  `p=3.05e-5`.

Attempt 7 independently showed the same operational pattern on a smaller
population: Revisit `8/9` versus native `2/9`, and `0/9` Novel takeovers.
Attempt 7 and phase-2 must remain separate tables because their scene
populations overlap and they are not independent pooled confirmations.

The train40 certificate challenge set also prevents an exaggerated safety
claim: TP/FP/FN/TN is `122/9/31/318`, precision `93.13%`, recall `79.74%`, and
FPR `2.75%`.  Therefore “certificate” means a frozen operational geometric
test, not a mathematical guarantee of zero error.

### 2.3 Scale-free bearing is a sufficient controller interface in the tested
Revisit setting

The current method sends only a relative direction with a fixed `2.5 m`
residual to frozen NavDP.  It does not expose metric translation recovered by
PnP.  The strong 2-leg and actual-online 3-leg results show that a full global
map or replacement controller is unnecessary for these supported Revisit
queries.

The Novel oracle experiment is a separate mechanism result: native `28/40`
became `40/40` under oracle periodic yaw and oracle-bearing token control,
paired `+12/-0`, `p=0.000488`.  It establishes recoverability given privileged
direction, not a deployable Novel method.

## 3. What phase-2 disproved or left unresolved

Phase-2 did **not** establish that the current certificate improves navigation
success over a simple raw-DINO bearing baseline:

- certified versus raw fixed, all natural queries: `21/38` versus `27/38`,
  paired `+1/-7`, `p=0.0703`;
- Revisit alone: `17/19` versus `18/19`, `+0/-1`, `p=1.0`;
- certified versus old geometry: `21/38` versus `23/38`, `+0/-2`, `p=0.5`.

The six-query aggregate advantage of raw fixed comes mainly from Novel queries,
where an always-on unverified historical direction sometimes behaves like an
exploration perturbation.  It is not evidence that raw DINO localized a Novel
goal.  The completed forced-anchor attribution now tests that interpretation:
on 19 consumed Phase-2 Novel queries, factual DINO context reduced route error
by only `4.15 deg` versus 12 uniform legal anchors, with scene-cluster CI
`[-1.36,+8.90] deg`; useful `<=30 deg` coverage was `10/19` versus `9.75/19`.
The history-specific promotion gate failed.

The sole Revisit certified/raw discordance in phase-2 exposes a concrete design
issue.  For `r1Q1Z4BcV1o/episode_0005`, raw DINO selected frame 132 and
succeeded, while the current geometry lexicographic rank selected frame 141
and failed.  Frame 141 had 522 accepted PnP inliers, `0.934 px` reprojection
RMSE, and approximately `41--42%` hull coverage.  The failure therefore shows:

> local two-view geometric self-consistency is not identical to semantic goal
> relevance or downstream navigational usefulness.

This is a known localization hazard rather than an isolated coding anomaly.
Geometric inlier counts can be distorted by repeated local structures, and
retrieval metrics need not correlate with localization outcomes.  Relevant
primary references include [Large-Scale Location Recognition and the Geometric
Burstiness Problem (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Sattler_Large-Scale_Location_Recognition_CVPR_2016_paper.html)
and [Investigating the Role of Image Retrieval for Visual Localization
(IJCV 2022)](https://arxiv.org/abs/2205.15761).

## 4. Current method versus developmental refinement

### 4.1 Frozen current method: Certified Episodic Compass (CEC)

```text
current ImageGoal + causal online RGB history
    -> temporally diverse raw-DINO top-8 retrieval
    -> SuperPoint/LightGlue evidence for all eight
    -> geometry-lexicographic top-1 proposal
    -> LingBot-depth PnP
    -> frozen atomic certificate
       inliers >= 16
       query/reference hull coverage >= 5%
       reprojection RMSE <= 2 px
    -> accept: scale-free bearing -> fixed 2.5 m residual -> frozen NavDP
    -> reject: exact native ImageGoal NavDP
```

This is the method supported by existing closed-loop results.

### 4.2 Developmental hypothesis: semantic proposal, geometric verification

The more principled factorization is:

```text
DINO orders semantic place hypotheses h1 ... hK
    -> for h1, h2, ... in semantic order:
           apply the unchanged geometric/PnP certificate
           stop at the first accepted hypothesis
    -> accepted: emit only its scale-free bearing
    -> none accepted: exact native fallback
```

Here geometry has veto authority but not semantic re-ranking authority.  This
is analogous to a hypothesis test: retrieval supplies the prior order and the
geometric witness supplies feasibility/evidence.  It is simpler to explain,
avoids letting a few extra local inliers override semantic relevance, and may
stop early on easy Revisit queries.

This refinement is **not the frozen method**. It was evaluated through the
following sequence:

1. Read-only PnP counterfactual on the 28 already consumed held-out Revisit
   histories from Attempt 7 and phase-2.  This is development evidence only.
2. If semantic-first maintains or increases certificate-actionable coverage,
   run one explicitly post-hoc consumed closed-loop comparison against current
   CEC.  It cannot be reported as confirmation.
3. Promote it only after a strict paired closed-loop net gain, then freeze the
   method and acceptance criteria before any new scene-disjoint population.
4. Confirm any promoted variant with paired
   native/raw/current-CEC/semantic-first arms; never tune on that population.

The read-only counterfactual branch is structurally forbidden from changing an
action and is logged as `action_authority=false`.

The go/no-go rules were frozen before the repair-2 audit produced a record in
`SEMANTIC_PROPOSAL_GEOMETRIC_VERIFICATION_DECISION_GATE_20260815.md`. In
particular, semantic-first must not lose certificate-actionable coverage before
it is allowed to consume a closed-loop development run; any promoted candidate
still requires new scene-disjoint confirmation.

Final execution status: the read-only Gate A completed at equality
(`28/28` actionable for both geometry-first and DINO-order first-certified,
paired `+0/-0`) and an independent verifier passed. After two outcome-blind
receipt/accounting repairs with zero formal records, the authorized consumed
Gate B was submitted with smoke `15763484` and formal array `15763485`.
Population index 18 suffered an unpaired Habitat native abort; its evidence was
preserved and only that index was repaired as `15764888_18`. Repair-aware
summary/verifier jobs `15764892 -> 15764893` completed and independently
verified the exact tie: geometry-first `25/28`, semantic-first `25/28`, paired
`+0/-0`, `p=1.0`. The frozen strict-net-gain rule therefore retains
geometry-first CEC and authorizes no semantic-first confirmation.

The tie is mechanistically informative. The first selected anchor changed in
`21/28`, but the first authorized bearings differed by only `0.770°` on average
and at most `4.478°`; all 28 were authorized at query step zero. In these
high-support Revisit histories, proposal order changes which nearby co-visible
frame is named but not the scale-free direction delivered to NavDP. This
post-hoc diagnostic supports de-emphasizing top-1 ranking as a contribution;
it does not show that geometric authorization is unnecessary. Full result:
`SEMANTIC_PROPOSAL_GATE_B_RESULT_20260815.md`.

## 5. Resolved Novel attribution and stop decision

Phase-2 raw-DINO gains were first audited without a new rollout.  Raw bearings
collapsed near the rear, while the correct route was behind in `16/19`
Phase-2 queries.  The definitive cheap gate then replayed the exact consumed
history/current/goal inputs and compared each factual raw-DINO anchor against
12 identity-seeded uniform legal anchors through the complete LingBot
goal-pose/bearing path.

- shortest-path error: factual `38.63 deg`, random expectation `42.78 deg`;
- mean factual advantage: `+4.15 deg`;
- scene-cluster 95% CI: `[-1.36,+8.90] deg`;
- `<=30 deg` coverage: `10/19` versus `9.75/19` expected;
- direct-goal advantage: `+2.17 deg`, CI `[-3.71,+7.07] deg`;
- independent verifier: `verified=true`.

The gate required the cluster-CI lower bound to exceed zero.  It failed, and
useful angular coverage was unchanged.  Therefore:

- do not call raw DINO a Novel localization or global-direction method;
- do not run goal shuffle or the written 600-step four-arm control;
- do not consume final14 to rescue this branch;
- keep the oracle-bearing result as privileged capability evidence only;
- use final14 for the paper's actual prospective question: CEC Revisit utility,
  Novel interference and exact native fallback.

Full result:
`NOVEL_RAW_FORCED_ANCHOR_ATTRIBUTION_RESULT_20260816.md`.

The complete frozen construction, arm, audit, statistical, and interpretation
contract is in `NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL_20260815.md`.

## 6. Paper claims that are defensible now

### Claim A — utility

**Causal online episodic visual memory can provide large Revisit gains to a
frozen ImageGoal diffusion policy through a direction-only interface.**

Evidence: original N=40, Fresh160, actual-online NNR, double-Revisit.

### Claim B — authorization

**A frozen geometric witness enables role-free abstention: supported Revisit
hypotheses can take over while unsupported Novel queries exactly retain the
native controller.**

Evidence: phase-2 `0/19` Novel takeover and Attempt 7 `0/9`, plus train40
open-set error accounting.  Wording must remain probabilistic/empirical rather
than “guaranteed safe.”

### Claim C — minimal adapter

**Long-term memory need not replace or retrain the navigation policy; a
scale-free episodic compass is sufficient on the evaluated Revisit tasks.**

Evidence: fixed-bearing results and frozen-controller audit.

### Claims that are not currently allowed

- CEC has higher Revisit SR than raw-DINO fixed bearing;
- the certificate has zero false positives in general;
- a deployable direction source improves Novel navigation;
- GOAT full sequential SR/SPL has been improved;
- learned CDEC/GCT can replace explicit retrieval and geometry;
- the current system is controller-agnostic or sensor-agnostic.

## 7. Why training-free is not automatically “just engineering”

Training-free methods are publishable when the contribution is a new problem
decomposition, interface, or verified capability, and when simple alternatives
are controlled.  It is not enough merely to combine pretrained modules.

Recent examples make the distinction clear.  [TANGO (CVPR
2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Ziliotto_TANGO_Training-free_Embodied_AI_Agents_for_Open-world_Tasks_CVPR_2025_paper.html)
is explicitly training-free but contributes a general embodied composition
framework and evaluates multiple tasks.  Conversely, [AstraNav-Memory (CVPR
2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_AstraNav-Memory_Contexts_Compression_for_Long_Memory_CVPR_2026_paper.html)
learns compressed visual memory coupled to a VLM policy.  Training is not the
novelty criterion; the scientific abstraction and evidence are.

This project has a defensible reason to remain training-free:

- CDEC improved candidate top-1 only weakly and reduced real
  certificate-actionable coverage (`115` versus geometry `122`);
- geometry-first CDEC rescue added only `1/349` actionable sessions;
- candidate-free long-context GCT collapsed from `18/20` with DINO addressing
  to `5/20` without it, paired `+0/-13`, `p=0.000244`;
- the small OOF residual improved DINO only `74/80 -> 76/80`, `p=0.5`, driven by
  two scenes.

Those negative results support a concrete representation claim: current
learned sequence models can use short working memory but fail to content-address
hundreds of causal frames reliably.  Explicit retrieval and verification are
not arbitrary scaffolding; they solve the experimentally isolated long-range
addressing and authorization problems.

## 8. Differentiation from the closest work

### MemoNav

[MemoNav (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MemoNav_Working_Memory_Model_for_Visual_Navigation_CVPR_2024_paper.html)
learns short-, long-, and working-memory representations inside a navigation
policy/topological graph.  Episodic Compass instead retrofits a frozen policy,
does instance-level causal history localization, transfers only bearing, and
studies open-set authorization and exact fallback.  We must not claim to be the
first multi-goal memory navigation method.

### GOAT and GOAT-Bench

[GOAT](https://arxiv.org/abs/2311.06430) uses a continually augmented
instance-aware semantic memory and modular planning on real robots.
[GOAT-Bench (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Khanna_GOAT-Bench_A_Benchmark_for_Multi-Modal_Lifelong_Navigation_CVPR_2024_paper.html)
evaluates sequential multimodal targets and explicit/implicit memory.  Our
distinctive analysis is causal ImageGoal history support, Revisit/Novel
interference without exposing role labels, and a minimal bearing adapter for a
frozen diffusion controller.  GOAT remains the necessary external sequential
benchmark, not evidence that our custom role-pair task alone is sufficient.

### Visual localization and IGL-Nav

Hierarchical localization already uses global retrieval followed by local
matching/PnP; this component sequence cannot be claimed as novel.  [IGL-Nav
(ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.html)
builds an incrementally optimized renderable 3D Gaussian representation for
goal localization.  Our method uses only the raw causal image stream and
monocular depth, avoids a persistent global reconstruction, and exposes a
scale-free control witness with exact policy fallback.

### Exploration--verification methods

[IEVE (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html)
actively switches among exploration, verification, and exploitation to reject
same-category instance distractors.  Our verification is not an active approach
phase for a Novel target; it authorizes whether a prior episodic observation may
control the current policy.

### New long-memory systems

[TrajRAG (CVPR
2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_TrajRAG_Retrieving_Geometric-Semantic_Experience_for_Zero-Shot_Object_Navigation_CVPR_2026_paper.html)
retrieves geometric-semantic trajectory experience for zero-shot ObjectNav.
The distinction must be empirical: instance ImageGoal revisit, an explicit
geometric witness, direction-only policy adaptation, and role-free fallback.
“Retrieval-augmented navigation” by itself is no longer a novelty claim.

## 9. Required paper tables and figures

### Main Table 1 — supported two-leg Revisit utility

Rows: native, old geometry, raw-DINO direct, raw-DINO fixed, current CEC, and
semantic-first CEC only if prospectively confirmed.  Columns: `B|A SR`, paired
gain/loss, McNemar p, cluster CI, SPL/path length, takeover rate, and runtime.

### Main Table 2 — role-free open-set evaluation

Report Novel and Revisit separately before aggregate SR:

- Revisit success / coverage;
- Novel false takeover rate;
- Novel losses versus native;
- exact-fallback equality;
- utility at fixed false-intervention rate.

Attempt 7 and phase-2 remain separate blocks.  A new prospective set is needed
for any redesigned proposal rule.

### Main Table 3 — continual 3-leg evaluation

Actual-online NNR and double-Revisit, with equal budgets and identical prefix
hashes.  Do not use the old expert-A strict-v4 result as the main comparison.

### Main Table 4 — external GOAT evaluation

The frozen 20-scene first-ImageGoal semantic-STOP adapter failed its
preregistered gate (`0/20` certified successes and zero authorized STOPs). It
is a negative transfer result, not external validation of CEC. A paper-facing
public benchmark therefore still needs sequential ImageGoal subtasks with
official GOAT metrics, reporting native versus CEC per goal index and
separating causally supported from unsupported targets. ObjectGoal/LanguageGoal
scores are secondary because the current method does not define those
controllers.

### Figure 1 — method abstraction

Show history hypotheses entering an evidence boundary, with only a bearing
crossing into the frozen controller.  Components should appear as replaceable
instances below the abstraction, not as the headline.

### Figure 2 — safety--utility frontier

X axis: Novel false intervention/loss rate.  Y axis: Revisit successful
takeovers or conditional SR.  Plot native, raw-DINO, old geometry, current CEC,
and any prospective semantic-first variant.  This is more informative than a
single aggregate SR where incidental Novel perturbations can dominate.

### Figure 3 — causal continual protocol

Visualize online A history, Novel query, and Revisit query while emphasizing
that the runtime never receives the construction role label.

## 10. Essential ablations, not parameter sweeps

1. **Addressing:** no memory / raw DINO / shuffled history.
2. **Verification authority:** always-on / current geometry-rank certificate /
   semantic-first first-certificate.
3. **Output interface:** raw metric PnP / normalized bearing / no takeover.
4. **Memory provenance:** expert history / actual-online causal history.
5. **Temporal memory:** recent-only / full causal history, with identical
   candidate budget.
6. **Certificate evidence:** correspondence precheck only / full PnP certificate.
7. **Controller boundary:** fixed NavDP; X-NavDP may be a secondary adapter
   check but existing `21/26` versus `20/26` does not support a controller claim.
8. **Cost:** retrieval, matching, PnP, and total planning latency; memory growth
   versus episode length.

Avoid broad threshold/K sweeps on held-out outcomes.  Threshold sensitivity may
be reported from train-only/offline data after the operating point has been
frozen.

## 11. GOAT track: exact current status and next action

The ten-scene native runtime pilot produced `0/10` first-subtask SR because a
selected zero NavDP trajectory was incorrectly mapped to GOAT `SUBTASK_STOP`.
NavDP clips predicted endpoints below `0.5 m` to zero, whereas GOAT success is
strictly below `0.25 m`; zero therefore means “arrival proposal/abstain,” not
arrival.

The train-only arrival audit selected a frozen `0.075 m` predicted-distance
threshold with TP/FP/FN/TN `76/0/84/779` when combined with native zero and the
PnP certificate. The parent scene-disjoint confirmation failed before its first
observation because a deterministic 63-bit reset seed was passed directly to
NumPy's uint32-only legacy RNG. Its formal array and postprocessing ran zero
episodes.

The authorized repair is only:

```text
service_reset_seed = frozen_episode_hash % 2**32
```

The same uint32 reset seed was sent to NavDP and MemNav; per-plan 63-bit
diffusion seeds remained unchanged. The repair cloned the failed immutable
bundle and proved that only the reset helper/test/provenance changed. Remote
tests passed `23/23`, repair smoke `15761753` completed, and the full
`15761754 -> 15761755 -> 15761756` chain finished with independent
`verified=true`.

The confirmation failed: certified success was `0/20`, certified STOP coverage
was zero, and all episodes reached the forced guard. Across 28 zero-endpoint
events, 26 were not official arrivals; both true arrival events failed the
local-geometry precheck. The frozen decision is to stop this adapter and never
tune its `0.075 m` threshold on the held-out episodes. This does not test the
paper's Revisit bearing takeover; a valid external test must evaluate that
actual interface on sequential, causally supported ImageGoal subtasks.

## 12. Statistical and reproducibility contract

- Every closed-loop comparison is same-machine, same-process or rigorously
  paired under byte-identical prefixes and deterministic plan seeds.
- Report `N`, scene count, paired gains/losses, exact McNemar p, and
  scene-cluster bootstrap CI together.
- Never combine the old `77.5%` native A rate with the later same-run `70%`
  native rate.
- `N<20` is explicitly underpowered/anecdotal unless an exact paired test is
  still decisive.
- Development, confirmation, and blind populations are recorded separately.
- Negative results remain in the paper where they justify the factorization:
  top-K null, failed learned gate calibration, GCT addressing collapse, CDEC
  actionable-coverage loss, active-glance degradation, and X-NavDP controller
  tie.

## 13. Tonight's frozen exit criteria and completion

Tonight is successful if the following artifacts exist; a higher SR is not
required:

1. The semantic-first read-only PnP counterfactual and authorized consumed
   closed-loop Gate B complete on Attempt 7 and Phase-2 Revisit histories with
   immutable summaries and independent verification.
2. The GOAT reset-seed repair is provenance-audited and its fresh smoke/formal
   chain completes under the frozen decision rule.
3. The Novel shuffled/random-bearing protocol is written with fixed hypotheses,
   arms, seeds, and interpretation before any new outcome is read; its cheaper
   first-step promotion gate subsequently failed, so the closed-loop protocol
   remains unexecuted by design.
4. The paper claims, main tables, and non-claims in this document are reflected
   in the project status ledger.
5. No long closed-loop run is submitted unless the candidate-level audit gives
   a concrete decision that cannot be answered offline.

All five criteria were met. Gate B was an exact paired null (`25/28` versus
`25/28`, `+0/-0`) and retained geometry-first CEC. GOAT semantic arrival was a
verified negative (`0/20`) and was rejected without retuning. Success tonight
therefore means evidence and method-scope convergence, not a post-hoc SR gain.

The desired paper story is therefore concise:

> **Episodic Compass turns causal visual history into evidence-carrying
> directional control for frozen ImageGoal policies.  It separates semantic
> recall from geometric authorization, transfers only scale-free bearing, and
> falls back exactly when history cannot support the current goal.**
