# Paper Evidence Matrix: Certified Episodic Compass

**Frozen snapshot:** 2026-08-15; updated through 2026-08-16  
**Purpose:** keep populations, causal contracts, statistical strength, and
allowed claims separate while the paper is assembled. Pending rows are not
results. Results from different rows must not be pooled unless a protocol
explicitly defines that analysis.

## 1. Positive closed-loop evidence

| Evidence block | Population and observability | Paired result | Statistical status | What it supports |
|---|---|---:|---|---|
| Original geometry memory | 20 MP3D scenes, 40 two-leg episodes; shared paired execution | native `4/40`, memory `19/40`; `+15/-0` | exact McNemar `p=6.1e-5` | Episodic Revisit information can materially improve a frozen ImageGoal policy. |
| Fresh160 supported Revisit | 20 scenes, 160 sources; conditional on the same 120 successful A prefixes; actual-online support later audited | native `27/120`, old geometry `91/120`, raw direct `106/120`, CEC `112/120` | CEC vs raw `+9/-3`, `p=0.146`; CEC vs geometry and native significant | Strong utility on high-support Revisit; not proof that certification raises the saturated Revisit ceiling over raw DINO. |
| Actual-online NNR | 19 shared online A/B prefixes; Revisit C constructed only from causal online-A observations | native `5/19`, known-role direct `14/19`, role-free CEC `16/19` | CEC vs native `+11/-0`, `p=0.0009766`, cluster CI `[+27.8,+85.7]` pp | The effect is not an expert-history artifact; role-free authorization works in a continual 3-leg setting. |
| Fresh20 double Revisit | 20 three-leg episodes; two memories must remain available | native joint `0/20`, full memory `12/20`, role-free CEC `17/20` | primary preservation contrast only `12/14` vs `8/14`, `+6/-2`, `p=0.289` | Feasibility of retaining and reusing multiple episodic memories; the specific “older memory preservation” mechanism is not confirmed. |
| Attempt 7 mixed role | 9 histories / 9 held-out scenes; 9 Novel + 9 Revisit; no runtime role label | native `N2/R2`, raw fixed `N1/R8`, geometry `N2/R7`, CEC `N2/R8` | CEC vs native total `+6/-0`, `p=0.03125`; Novel takeovers `0/9` | Small held-out evidence for Revisit utility plus exact Novel fallback; underpowered relative to the preregistered target. |
| Phase-2 mixed role | 19 histories / 12 held-out scene clusters; 19 Novel + 19 Revisit; no runtime role label | native `N4/R1`, raw fixed `N9/R18`, geometry `N4/R19`, CEC `N4/R17` | CEC vs native `+16/-0`, `p=3.05e-5`; CEC vs raw `+1/-7`, `p=0.0703`; raw Novel vs native `+6/-1`, `p=0.125`; Novel takeovers `0/19` | Replicates role-free fail-closed behavior and Revisit utility. Raw Novel's nonsignificant gain is now attributed to a backward-biased head aligning unusually often with this cohort's U-turn-heavy route distribution; history-specific information remains unproven. |
| Oracle Novel bearing | 20 scenes, 40 Novel-A episodes; same-machine paired mechanism probe | native `28/40`, oracle periodic yaw `40/40`, oracle bearing+token `40/40`; `+12/-0` | exact McNemar `p=0.000488`; cluster CI `[+15,+47.5]` pp | Direction is a strong recoverable Novel bottleneck and NavDP can execute it. Oracle bearing is privileged information, so this is not a deployable method result. |

## 2. Authorization and offline evidence

| Evidence block | Population | Result | Interpretation boundary |
|---|---|---|---|
| Train40 certificate challenge | 480 sessions, 40 train scenes | TP/FP/FN/TN `122/9/31/318`; precision `93.13%`, recall `79.74%`, FPR `2.75%` | Certificate is useful but not a formal safety guarantee and not zero-FP. |
| Fresh160 online-support audit | 120 conditional-B histories | `120/120` have online max-covis `>=0.20`; `115/120 >=0.50`; median `0.898` | Rules out expert-only observability leakage while identifying this benchmark as a comparatively easy, high-support band. |
| Train40 role stratification | 14,172 candidates | RANSAC pass precision: Novel start `34.2%`, Novel midpoint `50.0%`, true Revisit `90.9%` | Local geometric consistency is discriminative for supported Revisit but is not a semantic Novel/Revisit oracle. |
| Scene-grouped OOF ranking | 40 train scenes | candidate AUC: DINO `.8395`, geometry `.8802`, combined `.9041`; positive-session top-1: `116/155`, `107/155`, `113/155` | Better candidate AUC does not imply better anchor selection or closed-loop control. |

## 3. Negative and null results retained as evidence

| Experiment | Result | Decision |
|---|---|---|
| Wider candidate chain/top-K | `18/40` vs `18/40`, paired `p=1.0` | Candidate count/diversity is not the main bottleneck; do not widen K as a default fix. |
| GLP Stage 1 | single-evidence DINO tied max-DINO | No closed-loop gain from the framework alone. |
| GLP Stage 2 learned gate | development `72.7%` vs DINO `87.3%`; train-optimal threshold `.397` vs dev `.807` | Feature ranking signal exists, but activation calibration does not transfer; do not use the learned score as authorization. |
| CDEC learned proposal | OOF top-1 `128/155` vs geometry `126/155`, but actionable certificates `115` vs `122`; CDEC-only `+1/-8`, `p=0.039` | Candidate top-1 accuracy is the wrong endpoint; CDEC significantly loses actionable coverage. |
| Candidate-free long-memory GCT | DINO-addressed `18/20`, full-prefix anchor-free `5/20`; `+0/-13`, `p=0.000244` | Current learned memory cannot perform content addressing over hundreds of causal frames by itself. |
| Learned residual | DINO `74/80`, DINO+OOF residual `76/80`; `+2/-0`, `p=0.5`, two-scene concentration | No justification for a long training run. |
| Active glance | native `31/40`; best gated scan `25/40`; paired `1` gain / `7` losses | Initial panoramic scanning is intervention-heavy and harmful; stop this branch. |
| X-NavDP controller | base PointGoal `20/26`, official X+MPC `21/26`; paired `+2/-1`, `p=1.0` | Controller replacement is not the current bottleneck and is not a paper claim. |
| Graph rescue | actual-online NNR `16/19` with or without rescue; 2-leg rescued only `1/5` stuck failures | Remove from the main method; it is not a stable general contribution. |
| Proposal versus verification, Gate B | 28 consumed Revisit histories: geometry-first `25/28`, DINO-order first-certified `25/28`; paired `+0/-0`, `p=1.0`; independent verifier passed | Retain geometry-first CEC. Proposal order changed the first anchor in `21/28` but not one outcome; post-hoc first-bearing difference was at most `4.478°`. Ranking among supported co-visible anchors is not the current SR bottleneck. Never confirmation evidence. |
| Raw Novel cohort-shift audit | Attempt 7: raw `1/9` vs native `2/9`, `+1/-2`, `p=1.0`; Phase-2: raw `9/19` vs native `4/19`, `+6/-1`, `p=0.125`. Raw first-bearing circular `R=.932/.840`, centered at `166.1/176.5°`; correct route was behind in `7/9` vs `16/19` | No code/checkpoint drift found. Phase-2 gains coincide with accurate first directions, but the cohort is U-turn-heavy and construction does not balance bearing. This is post-hoc mechanism evidence, not deployable Novel localization or a confirmation result. |
| Raw Novel forced-anchor attribution | 19 consumed Phase-2 Novel queries / 12 scene clusters; exact current/goal/history replay; factual DINO anchor versus 12 frozen uniform legal anchors; no Habitat | shortest-path error `38.63°` vs random expectation `42.78°`, advantage `+4.15°`, scene-cluster CI `[-1.36,+8.90]°`; `<=30°` coverage `10/19` vs `9.75/19`; direct-goal advantage `+2.17°`, CI `[-3.71,+7.07]°` | Promotion gate failed. DINO does not provide a stable history-specific Novel compass; stop goal-shuffle and fresh Novel-DINO closed-loop work. Preserve final14 for CEC. |
| Replica | pilot native `7/8`, CEC `7/8`; formal construction yielded `0` valid long histories | Current Replica incompatibility is benchmark constructibility, not method failure or support for cross-domain gain. |
| GOAT first-ImageGoal semantic arrival | 20 scenes: certified success `0/20`, STOP coverage `0`, all 20 forced-guard terminations; preregistered gate failed and independent verifier passed | Reject this arrival adapter and do not retune on held-out data. It did not evaluate CEC's Revisit retrieval/bearing intervention, so external validity remains open. |

## 4. Frozen decision gates and pending protocols

| Experiment | Frozen scope | Job chain / status | Decision supplied |
|---|---|---|---|
| Proposal versus verification, Gate A | 28 consumed Revisit histories; read-only same-PnP audit; no action authority | geometry-first `28/28`, DINO-order first-certified `28/28`, paired `+0/-0`; independent `verified=true` | Gate A passed at equality and authorizes one consumed closed-loop comparison. It does not establish semantic-first superiority. |
| Novel first-step attribution | 19 consumed Phase-2 Novel queries / 12 scene clusters; factual raw-DINO anchor versus 12 frozen uniform eligible anchors; proposal-only | completed; factual advantage `+4.15°`, cluster CI `[-1.36,+8.90]°`; useful coverage tied in expectation | Gate failed; the apparent query-specific signal is too weak and heterogeneous to promote. Development evidence only. |
| Prospective Novel causal control | native, factual history, deranged history and randomized bearing; previously written protocol | unexecuted and stopped after first-step attribution failed | Do not consume fresh scenes or HPC time on this branch. A future independent Novel project would require a newly frozen, direction-balanced construction. |
| Final14 role-free CEC confirmation | all 14 sealed untouched scenes; direction-balanced/yaw-decoupled Novel plus standard and hard-support Revisit; native/raw fixed/old geometry/CEC | protocol frozen, SHA `3d1ebc6e...`; consumed constructibility smoke standard `4/4`, hard `3/4`; no final14 access yet | Next and only prospective MP3D method confirmation. Complete consumed end-to-end dry-run before unsealing. |

## 5. Claim mapping

| Claim | Evidence required | Current state |
|---|---|---|
| Revisit utility for frozen NavDP | paired closed-loop gains on causal history | **Established internally**, including actual-online 3-leg evidence. |
| Role-free empirical authorization | Revisit takeovers plus Novel false-takeover/loss accounting | **Supported on 28 held-out Novel queries**, but final prospective mixed-role confirmation remains necessary. |
| Scale-free bearing as sufficient interface | bearing-only closed loop, no metric waypoint crossing the boundary | **Supported for tested Revisit tasks**; oracle Novel evidence is mechanism-only. |
| Certificate improves over raw DINO SR | prospective paired comparison at matched population | **Not established**; current honest value is interference control and exact fallback. |
| Public-benchmark external validity | executable public protocol testing the actual Revisit-bearing claim | **Not established**; the first-ImageGoal semantic-arrival confirmation failed and a proper sequential supported-Revisit protocol remains future work. |
| Learned replacement of retrieval/geometry | actionable closed-loop superiority | **Rejected by current evidence**; do not add training merely for appearance. |

## 6. Reporting rules

1. Keep Attempt 7, Phase-2, Fresh160, and actual-online NNR as distinct
   populations; never manufacture power by pooling their p-values.
2. Every closed-loop result reports episode count, scene count, paired
   gain/loss, exact McNemar, cluster CI where available, and whether the role
   label was available to runtime.
3. “Certificate reject” means insufficient self-verifiable historical support,
   not semantic proof that a goal is Novel.
4. “Training-free” describes adaptation of the navigation system: DINO,
   SuperPoint/LightGlue, LingBot, and NavDP retain pretrained parameters.
5. Same-machine/same-process paired effects take precedence over cross-run
   native percentages because CUDA trajectory nondeterminism is material.
6. Consumed development audits choose hypotheses; only fresh scene-disjoint
   populations may support final method claims.
