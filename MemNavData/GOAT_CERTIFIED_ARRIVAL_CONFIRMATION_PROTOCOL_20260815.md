# GOAT Certified Arrival Confirmation Protocol (2026-08-15)

**Final status:** completed and independently verified; the preregistered gate
failed (`0/20` certified successes, zero authorized STOPs). The frozen protocol
below is retained unchanged. See
`GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md` for the result and failure
audit; no threshold was retuned.

## Question

Can the frozen LingBot-depth + LightGlue/PnP certificate turn a native NavDP
no-motion output into a safe GOAT `SUBTASK_STOP` decision on scenes that were
not used by the train-only arrival audit?

This is a first-ImageGoal semantic-arrival confirmation, not a complete GOAT
benchmark score. Native NavDP remains the navigation controller. The only
tested change is who is allowed to emit the official subtask transition.

## Frozen train-only evidence

The operating point was selected before any GOAT confirmation result was read.

- train population: 40 MP3D scenes, 80 episodes, 939 exact states;
- strict `<0.25 m` arrival states: 160; non-arrival states: 779;
- selected conjunction: native no-motion proposal and certified PnP distance
  `<=0.075 m`;
- train confusion matrix: TP 76, FP 0, FN 84, TN 779;
- train precision 1.0, recall 0.475, true positives in 38 scenes;
- report SHA-256:
  `13f265b200f02c877557bdc18a846688274961ddc451ead463dbcb319d528373`;
- independent verification SHA-256:
  `ffb2576ef25f1a0ff571d66640ec7cddd611417858a35d4e15de1c6ef2ea7dfd`.

No GOAT validation result may change the 7.5 cm threshold, certificate, scale
configuration, fallback rule, selected scenes, or confirmation gate.

## Frozen runtime contract

For every closed-loop native NavDP decision:

1. A nonzero selected trajectory is converted to GOAT motion actions with the
   already frozen pure-pursuit adapter.
2. A near-zero selected trajectory is only an `ARRIVAL_PROPOSAL`; the adapter
   is structurally unable to emit `SUBTASK_STOP`.
3. The proposal is authorized only if all of the following hold:
   - at least 64 causal online RGB frames have been streamed;
   - current RGB to ImageGoal SuperPoint/LightGlue geometry passes the frozen
     atomic certificate;
   - current-frame LingBot depth and PnP produce a relative pose;
   - a strict first-64-frame causal ground scale is available, with no pooled
     constant fallback;
   - the resulting planar distance is `<=0.075 m`.
4. If authorization is rejected, execute the highest-critic executable motion
   candidate from the same already sampled 16-candidate batch.
5. If that batch contains no motion, draw at most three deterministic,
   read-only candidate batches from the identical observation. These calls
   must not mutate NavDP's FIFO. If all fail, stop the evaluation as a safe
   stall; never fabricate motion or semantic success.

The metric-scale configuration is frozen to confidence quantile 0.5, pixel
stride 4, 60 histogram bins, peak threshold 0.3, bias correction 1.15, and
scale range `[0.8, 6.0]`. Simulator depth is used by native NavDP as in the
baseline but is forbidden to the arrival sidecar. Official distance-to-target
is read only after the frozen decision and is stored solely for audit.

## Populations

### Engineering smoke

One already consumed runtime-pilot episode (`bCPU9suPUw9`, episode `1`) is used
only to catch environment, API, memory, and Python-compatibility failures. Its
outcome cannot select or alter any method component.

### Formal confirmation

`goat_certified_arrival_manifest.json` freezes one ImageGoal-first episode from
each of 20 HM3D `val_unseen` scenes. All 20 scenes are disjoint from the frozen
ten-scene GOAT runtime pilot. Scene and episode selection are deterministic
SHA-256 rankings under the salt `goat-certified-arrival-confirmation-v1`, made
without navigation outcomes.

The manifest itself is the population authority. Each array task runs exactly
one manifest identity and writes an immutable per-episode record.

## Primary gate

The deployment claim passes only if all conditions hold across the 20 formal
episodes:

- zero false certified stops under the official strict `<0.25 m` success
  boundary;
- at least five true certified stops;
- true certified stops span at least five scenes.

The report also gives the paired difference against the legacy policy that
would stop on the first native no-motion proposal. That comparison is
diagnostic and does not replace the primary safety/coverage gate.

## Interpretation boundary

A pass supports a deployable, visually certified semantic-stop layer for
ImageGoal subtasks. It does not establish a full multi-subtask GOAT score,
ObjectGoal stopping, or a general navigation improvement. A failure is kept as
a result; the disjoint set must not be reused for threshold or architecture
selection.
