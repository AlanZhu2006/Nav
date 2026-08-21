# Semantic-Proposal / Geometric-Verification Decision Gate

**Frozen:** 2026-08-15, before the first repair-2 audit record exists  
**Audit population:** the 28 already consumed Revisit histories from Attempt 7
and Phase-2  
**Status:** method-development gate, never confirmation evidence

## Candidate factorization

Current CEC lets local geometry reorder the DINO shortlist, then runs PnP on
the geometry-selected top hypothesis. The developmental candidate instead
keeps DINO as the semantic hypothesis order and applies the unchanged
LightGlue/LingBot-depth/PnP certificate in that order, stopping at the first
accepted hypothesis. Geometry has veto authority but not semantic re-ranking
authority. Exact native fallback is unchanged.

## Gate A: read-only same-PnP audit

The counterfactual branch has no action authority and runs no query rollout.
All 28 records and the immutable summary must complete. Let:

- `G` be the number accepted by deployed geometry-first CEC;
- `S` be the number accepted by DINO-order first-certified;
- `g` be sessions rejected by geometry-first but accepted by semantic-first;
- `l` be sessions accepted by geometry-first but rejected by semantic-first.

Proceed to one consumed closed-loop development comparison if and only if:

1. every record proves `action_authority=false` and
   `method_action_unchanged=true`;
2. `S >= G`; and
3. paired coverage has no net loss, `g >= l`.

The DINO-top1-only result is diagnostic. It cannot independently pass Gate A;
the deployable candidate is DINO-order first-certified because it retains
geometric veto and fallback.

If Gate A fails, retain current CEC and do not run a semantic-first closed-loop
experiment. No threshold or candidate order may be adjusted on these records.

## Gate B: consumed closed-loop development comparison

If Gate A passes, compare current CEC and semantic-first on exactly the same 28
consumed Revisit histories with shared causal prefixes, diffusion seeds,
budgets, and arm-order balancing. This run is explicitly post-hoc development.

- If semantic-first has strictly more paired successes than losses, freeze it
  as the candidate for a new scene-disjoint confirmation.
- If gains equal losses or losses dominate, retain current CEC. Elegance alone
  is not sufficient to replace the empirically supported method.
- Runtime errors count as failures; they may not be removed from the
  denominator.

No claim may be made from Gate B. The frozen winner must still face a fresh
mixed Novel/Revisit population with native, raw fixed, current CEC, and the
candidate arm. Novel false takeover and exact fallback are co-primary safety
measurements.

## Post-freeze execution receipt (rule unchanged)

Gate A completed with geometry-first `28/28`, DINO-order first-certified
`28/28`, paired `+0/-0`, and independently verified zero action authority.
It therefore passed at equality.

The authorized Gate B was implemented from the immutable Attempt-7 parent.
The first smoke `15763288` exposed a missing serialized proposal-order receipt
after both arms; it emitted zero completion records. Jobs
`15763289--15763291` were cancelled before execution. After outcome-blind
receipt and conservative runtime-error-accounting repairs, a pending interim
chain `15763428--15763431` was also cancelled at zero runtime so every
`certified_relocalization_ok=false` plan—not only HTTP exceptions—counts as an
arm failure. The eligible smoke/formal execution began as jobs
`15763484 -> 15763485`. Formal task 18 then suffered an unpaired Habitat
native abort before a completion record existed. Its partial evidence was
preserved, only index 18 was repaired as `15764888_18`, and repair-aware
summary/verifier jobs `15764892 -> 15764893` independently completed.
The 28-history formal root is
`/scratch/yz11502/Research/Nav-axis-uturn-results/semantic_proposal_gate_b_20260815/formal_consumed_final_20260814T232857Z`.

Both geometry-first and semantic-first reached `25/28`, paired `+0/-0`, exact
McNemar `p=1.0`; the verifier returned `verified=true`. The frozen rule
therefore retains geometry-first CEC and authorizes no semantic-first
confirmation. The result remains consumed development evidence, not a paper
confirmation result. Full provenance and interpretation are in
`SEMANTIC_PROPOSAL_GATE_B_RESULT_20260815.md`.

## Why this gate exists

Offline geometry, DINO similarity, co-visibility, and PnP acceptance are not
closed-loop navigation outcomes. The gate spends long evaluation only when the
cheap audit shows that the proposed factorization does not sacrifice the very
certificate coverage it is meant to preserve, while reserving the actual
method claim for independent data.
