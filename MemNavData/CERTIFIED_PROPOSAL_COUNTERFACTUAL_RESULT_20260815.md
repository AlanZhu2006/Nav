# Semantic-Proposal / Geometric-Verification Audit Result

**Date:** 2026-08-15  
**Status:** complete consumed-data mechanism audit; independently verified  
**Scope:** 28 previously consumed Revisit histories from Attempt 7 and Phase-2,
15 scenes. This is post-hoc method development, not confirmation evidence.

## 1. Provenance and safety boundary

- repair-3 immutable source:
  `/scratch/yz11502/Research/source_bundles/cec_prop_audit_7768fb855e9335ec`;
- source receipt SHA:
  `7768fb855e9335ec715d9e073c4928711566684cf5c53f5aa6e0347148638193`;
- infrastructure smoke `15762219`: completed, exit `0:0`, one discarded
  smoke record;
- formal array `15762220`: 28/28 completed, all exit `0:0`;
- summary `15762221`: completed;
- independent verifier `15762347`: completed, exit `0:0`, `verified=true`;
- formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/certified_proposal_counterfactual_20260815/formal_repair3_20260814T220115Z`;
- summary SHA:
  `8c287b12b7261d3d52dc47bf02fe4cce7cb3438fafc67da18e143e4372882c4b`;
- independent-verification SHA:
  `c4f91978986ca289db2752c031b9e693030aeb8452763f13d21bc420f7cf5769`.

Every record verifies:

- `is_closed_loop_evaluation=false`;
- `method_action_unchanged=true`;
- DINO top-1 and DINO-order counterfactuals both have
  `action_authority=false`;
- only causal factual history and the already frozen PnP/certificate were used.

## 2. Result

| Proposal factorization | Certified coverage |
|---|---:|
| deployed geometry-first | `28/28` |
| DINO top-1 + same certificate | `28/28` |
| DINO order, first certificate accepted | `28/28` |

Paired against geometry-first:

- DINO top-1: `+0/-0`, exact McNemar `p=1.0`;
- DINO-order first-certified: `+0/-0`, exact McNemar `p=1.0`.

Selection diagnostics:

- geometry selected a different anchor from DINO top-1 in `21/28` histories;
- only `7/28` selected the same anchor;
- DINO-first always accepted rank 1: histogram `{1: 28}`;
- ordered attempt mean/max were both `1.0 / 1`.

## 3. Frozen Gate A decision

The preregistered Gate A required:

1. all 28 records to prove no action authority and no method change;
2. semantic-first coverage `S >= G`;
3. paired coverage gains `g >= l`.

Observed values were `S=28`, `G=28`, `g=0`, `l=0`; all conditions pass.
Therefore exactly one consumed closed-loop development comparison between
current geometry-first CEC and DINO-top1-plus-certificate is authorized.

Gate A passing does not promote the candidate and is not a paper result. Gate B
must use shared causal prefixes, deterministic seeds, equal budgets and
balanced arm order. Only a strict paired success gain can freeze semantic-first
for a new scene-disjoint confirmation.

## 4. Scientific interpretation

This audit does not show that DINO is better. It shows a more precise fact:
within this high-support Revisit population, proposal coverage is saturated
before closed-loop execution. Local geometric re-ranking changes the selected
place hypothesis in 75% of histories, yet both choices pass the same atomic
certificate every time.

Consequently:

- widening the candidate chain cannot improve certificate coverage here;
- PnP acceptance cannot decide which of two locally self-consistent anchors
  has better semantic/downstream utility;
- the meaningful factorization is now testable: semantic retrieval proposes
  the anchor and geometry authorizes it, but closed-loop navigation—not this
  offline audit—must decide whether that factorization is actually better.

No threshold, candidate K, certificate boundary, or consumed population may be
changed before Gate B.
