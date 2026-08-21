# Semantic-Proposal versus Geometric-Ranking Gate-B Result

**Date:** 2026-08-15  
**Scope:** consumed closed-loop method development; never confirmation evidence  
**Frozen population:** 28 Revisit histories, 15 scenes  
**Decision:** Gate B did not pass; retain geometry-first Certified Episodic
Compass (CEC)

## 1. Question and frozen comparison

Gate A showed that geometry-first and DINO-order first-certified both produced
an actionable certificate on all 28 already consumed Revisit histories. Gate B
therefore tested the only question that matters for navigation:

> If proposal order changes but the PnP witness, certificate, bearing adapter,
> controller, seeds, causal history, and budget remain fixed, does
> semantic-first improve paired closed-loop success?

The two arms were:

```text
geometry-first:
  DINO shortlist -> LightGlue/F-matrix ranking -> top hypothesis
  -> unchanged PnP certificate -> 2.5 m scale-free bearing residual

semantic-first:
  DINO shortlist order -> first hypothesis passing the same PnP certificate
  -> the same 2.5 m scale-free bearing residual
```

Both arms used byte-identical actual-online A prefixes, deterministic plan
seeds, the same frozen NavDP controller, `600`-step budget, success radius,
candidate set, PnP thresholds, and certificate. The runtime received no
Novel/Revisit role label. Arm order was balanced by population-index parity.

The preregistered rule was deliberately conservative: semantic-first would be
nominated for a new scene-disjoint confirmation only if paired gains strictly
exceeded losses. A tie or net loss retained geometry-first CEC.

## 2. Primary closed-loop result

| Arm | Effective success | Raw physical success |
|---|---:|---:|
| Geometry-first CEC | 25/28 (89.29%) | 25/28 (89.29%) |
| DINO-order first-certified | 25/28 (89.29%) | 25/28 (89.29%) |

The tie holds within both pre-existing cohorts: Attempt 7 was `8/9` versus
`8/9`, and Phase 2 was `17/19` versus `17/19`.

Paired semantic-first minus geometry-first:

- gains: `0`;
- losses: `0`;
- both succeed: `25`;
- both fail: `3`;
- exact McNemar: `p=1.0`.

There were zero saved runtime-failure plans in either arm, so the conservative
runtime-failure penalty did not alter the physical outcomes.

This is a clean paired null on this consumed population. It does not establish
general statistical equivalence, but it gives no empirical basis for replacing
the frozen geometry-first method. The preregistered decision is therefore:

```text
gate_b_passed = false
next_action = retain_geometry_first_cec
```

No semantic-first confirmation experiment is authorized from this result.

## 3. Execution and provenance audit

- population: `28` histories from `15` scenes;
- cohorts: Attempt 7 `9`, Phase 2 `19`;
- first arm: geometry-first `14`, semantic-first `14`;
- all replayed causal prefixes equal: `true`;
- runtime role visibility: `none`;
- runtime-failure plans: geometry-first `0`, semantic-first `0`;
- first selected anchor changed between arms: `21/28`;
- independent raw-record verifier: `verified=true`.

Formal root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/semantic_proposal_gate_b_20260815/formal_consumed_final_20260814T232857Z`

Artifacts:

- summary SHA-256:
  `8a8c70e766bd862485f8c23fbd702fe0e50528159ac0d8d0811fab07a3f0697a`;
- independent-verification SHA-256:
  `03b036b6943419039b1ae9414c9ba8459fc4aea47407273ba212216cf85c080a`;
- immutable source receipt SHA-256:
  `fd0b8c233378b0d004c674639a98089981ddd6a6275347c5bacf27f00f7415db`;
- frozen population-manifest SHA-256:
  `0cdabbfb6c3477e5578406a3c8c0ef9e2387e4ad2f8e231fb873b3eece0f3625`.

Jobs:

- smoke: `15763484`;
- original formal array: `15763485`;
- exact-index repair: `15764888_18`;
- repair-aware summary: `15764892`;
- independent verifier: `15764893`.

### Task-18 infrastructure repair

Original task 18 (`15764627`) suffered a Habitat native `SIGABRT` on H200 node
`gh133` before a second arm or paired completion existed. It was not OOM
(`~15.5 GiB / 96 GiB`) or timeout (`00:19:12 / 08:00:00`). No Task-18 outcome
was read. All 203 partial artifacts were preserved under `failed_attempts/`
with receipt SHA-256
`b85f5e110dae618203321ed259e8305cf1cf2c339aa829230260ffd71471115d`.
Only index 18 was rerun, under identical frozen experimental inputs on H100
node `gh014`; it completed in `00:02:59`. All other 27 completed records were
left untouched. See
`SEMANTIC_PROPOSAL_GATE_B_TASK18_INCIDENT_20260815.md`.

## 4. Post-hoc mechanism diagnostic

The primary result is surprising only if “different anchor” is assumed to mean
“different useful direction.” It did not here.

Across all 28 histories, both arms obtained their first accepted certificate at
query step `0`. Semantic-first selected DINO rank 1 in `28/28`; geometry-first
selected DINO rank 1 in only `7/28` and selected ranks 2–8 in the remaining
histories. Consequently, the first anchor differed in `21/28`, with absolute
frame-index difference median `6.5`, mean `10.7`, and maximum `51`.

Despite those anchor changes, the first authorized scale-free bearings were
nearly identical:

- absolute angular difference mean: `0.770°`;
- median: `0.413°`;
- maximum: `4.478°`;
- at most `1°`: `23/28`;
- below `5°`: `28/28`;
- among the 21 changed-anchor histories, median `0.570°`, mean `1.027°`.

This aggregate diagnostic was not part of the frozen Gate-B decision and is
reported as post-hoc mechanism evidence only. The most direct explanation is
that high-support Revisit retrieval returns several temporally nearby,
co-visible observations of the same place. PnP transforms different local
anchors into almost the same global direction, and the direction-only adapter
then discards residual metric and frame-identity differences.

Other descriptive quantities are consistent with practical invariance:

- mean steps: geometry-first `132.36`, semantic-first `133.89`;
- median steps: `119.5` versus `121.0`;
- mean final distance: `1.1573 m` versus `1.1562 m`;
- certificate/takeover plans: `474` versus `481` in total.

These secondary summaries were not preregistered endpoints and are not used to
claim equivalence or efficiency superiority.

## 5. What the result means for the method and paper

The result does **not** show that geometric verification is unnecessary.
Semantic-first still required exactly the same LightGlue/depth/PnP certificate
before action, and Gate B contained only supported Revisit queries. It says
that, once several candidates are all views of the same supported place,
re-ranking those candidates is not the closed-loop bottleneck.

The paper should therefore:

1. retain geometry-first CEC as the frozen confirmed implementation;
2. treat DINO-versus-geometry proposal order as a null ablation, not a new
   method contribution;
3. emphasize the scientific abstraction that survived the test:
   **hypothesis proposal -> evidence-based authorization -> minimal bearing ->
   exact native fallback**;
4. make open-set Novel interference and Revisit coverage the central
   risk--utility evaluation, because Gate B cannot test either Novel safety or
   generalization;
5. avoid claiming that a learned or geometric top-1 ranker improves SR in the
high-support regime.

It also resolves the specific Phase-2 attribution question: proposal order
does not explain the raw-fixed-versus-CEC difference. On the 19 Phase-2 Revisit
histories, both certified orderings reproduced `17/19`; raw fixed had previously
reached `18/19`. The remaining difference lies downstream of proposal ordering
(certificate/authorization and its control consequences), while raw fixed's
Novel gains remain confounded by uncontrolled directional perturbation.

The null is useful: it removes proposal ranking from the list of plausible
high-leverage improvements and localizes future work to harder support,
authorization coverage, and genuinely fresh mixed-role confirmation.
