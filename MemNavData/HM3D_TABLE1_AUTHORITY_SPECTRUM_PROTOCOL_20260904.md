# HM3D Table-I memory-authority spectrum

Frozen on 2026-09-04 before running the two previously absent arms.

## Question

On one sealed, role-hidden HM3D Novel/Revisit population, which part of the
memory path changes closed-loop behavior: retrieval alone, availability of a
finite geometric pose witness, or the strict operational certificate?

## Population and disclosure

- Benchmark manifest SHA-256:
  `f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01`.
- 28 causal actual-mono Goal-A histories from 21 HM3D scenes.
- Each history has one natural Novel query and one supported Revisit query.
- The population was originally selected without reading policy outcomes.
- Native and strict-CEC outcomes on this population were known before this
  ablation was designed. Raw-memory and finite-witness outcomes were not.
- Consequently this is a **retrospective fixed-population causal ablation**,
  not a fresh confirmation or a new generalization population.
- No threshold, distance, retrieval, or controller parameter may be tuned from
  the new outcomes. Every completed history is reported.

## Four paired arms

All arms replay the identical causal RGB history, use the same first-40
height-calibrated monocular controller depth, the same frozen NavDP checkpoint,
the same 600-step budget, and receive no Novel/Revisit role label.

1. `mono_native`: no memory intervention.
2. `mono_raw_fixed`: DINO top-1 directly supplies a unit bearing with the
   frozen 2.5 m residual.
3. `mono_unthresholded_witness`: the CEC proposal and geometric pipeline is
   retained, but any finite PnP pose can authorize control.
4. `mono_cec`: the identical proposal/witness pipeline is authorized only when
   the frozen certificate passes.

The strict and finite-witness arms must agree exactly on the first DINO order,
geometry order, selected anchor, DINO rank of that anchor, and—when strict CEC
accepts—the PnP witness and emitted bearing. Their sole intervention is the
authorization rule.

## Estimands

Report Novel, Revisit, and pooled query results separately. The pre-registered
paired contrasts are:

- strict CEC minus native;
- raw memory minus native;
- finite witness minus native;
- strict CEC minus raw memory;
- strict CEC minus finite witness.

For each contrast report successes, paired gains/losses, risk difference,
exact two-sided McNemar p-value, and a scene-cluster bootstrap 95% interval.
Authorization rates are mechanism outcomes, not semantic Novel/Revisit
classification accuracy.

## Runtime and failure contract

- One persistent MemNav/LingBot and NavDP server pair is shared by all four
  arms of one history.
- Arm order is cyclically balanced by history index.
- Role labels are analysis-only and must not be forwarded at runtime.
- CEC rejection must preserve exact native requests.
- Runtime failures are invalid cells, not navigation failures.
- Smoke must pass before the formal array starts.
- Formal array cells request one GPU for one hour and run one history each.
- Summary and independent verification run only after every formal cell passes.

## Claim boundary

This experiment can support a causal description of the proposal–witness–
authority ladder on the fixed HM3D Table-I population. It cannot be described
as an unseen-query confirmation because two of its four arm outcomes were
already known when the ablation was selected.
