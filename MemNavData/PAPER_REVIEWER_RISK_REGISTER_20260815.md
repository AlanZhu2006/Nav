# Reviewer Risk Register for Certified Episodic Compass

**Date:** 2026-08-15  
**Purpose:** decide what evidence is still required before a WACV-style
submission. This is a risk document, not promotional prose.

## 1. Executive judgment

CEC has a paper-worthy scientific core if the submission is framed as
**open-set authorization of causal episodic direction for a frozen ImageGoal
policy**, not as a novel retrieval/PnP stack. The current evidence is already
strong for Revisit utility and actual-online causality. The submission is not
yet complete because external validity, hard-support coverage, and the value
of certification relative to the strongest simple baseline remain vulnerable.

The central reviewer question will be:

> Why is this more than image retrieval plus visual localization feeding a
> PointGoal controller?

The answer must be empirical: role-free continual operation, no role oracle,
causal online history, scale-free output, exact fallback, Novel interference
accounting, and paired closed-loop evidence. Architectural adjectives alone
will not answer it.

## 2. Risk matrix

| Risk | Current evidence | What would resolve it | Priority |
|---|---|---|---|
| Engineering-pipeline perception | DINO/LightGlue/PnP/LingBot are standard components | Lead with hypothesis/witness/authorization/interface; show replaceable backends and decisive interface ablations | P0 writing + experiments |
| Raw fixed has higher Phase-2 aggregate SR | raw `27/38`, CEC `21/38`; the gap is Novel-only. Forced-anchor attribution found only `+4.15°` factual advantage, cluster CI `[-1.36,+8.90]°`, and `10/19` versus `9.75/19` useful coverage | Report the consumed negative attribution; use final14 only for prospective Revisit utility versus Novel interference, not to rescue raw Novel | P0 wording + final confirmation |
| Certificate not proven better on Revisit | Phase-2 `17/19` vs raw `18/19`; Attempt 7 tie `8/9` | Do not claim higher SR; claim authorization unless a new prospective set establishes utility at fixed risk | P0 wording |
| Held-out populations are still small | Attempt 7 has 9 scenes; Phase-2 has 19 histories but only 12 scene clusters | New scene-disjoint confirmation with frozen winner and power target | P0 |
| Custom benchmark may look tailored | Strong MP3D role-pair and 3-leg results; GOAT first-ImageGoal arrival transfer failed `0/20` and did not test Revisit bearing | Freeze an executable public protocol that tests CEC on causally supported sequential ImageGoal subtasks, or use another compatible benchmark | P0/P1 |
| Fresh160 is near saturation/high support | online max-covis median `0.898`; CEC `112/120` | Stratify by support, temporal gap, viewpoint/yaw, and path distance; add low-support but constructible Revisit set | P1 |
| “Safe certificate” overclaim | held-out Novel takeover `0/28`, but train40 FPR `2.75%` | Use “empirical fail-closed/abstaining” language; report calibration and false accepts, never formal guarantee | P0 wording |
| Novel navigation remains unsolved | oracle direction `28/40 -> 40/40`; no deployable source | Keep Novel as fallback/safety scope; do not make direction-source claim | Scope boundary |
| Training-free may look unambitious | learned CDEC/GCT/residual did not transfer | Present negative results as factorization evidence; compare against simple training-free and learned alternatives under the same actionability/closed-loop contract | P1 |
| Proposal ranking may look arbitrary or overengineered | Gate A tied actionable coverage `28/28`; consumed Gate B tied SR `25/28`, paired `+0/-0`, even though the first anchor changed in `21/28`; authorized bearings remained within `4.478°` | Report the null honestly, claim no ranking novelty, and test authorization coverage on a prospective lower-support band rather than adding another ranker | P1 generalization |
| Runtime may be too expensive | top-8 local matching, depth replay, PnP latency not yet in main tables | Report per-stage latency, first-query/cache cost, memory growth, and takeover frequency | P1 |
| Monocular scale and backend dependence | scale-free bearing avoids metric leakage, but LingBot supplies depth/poses | Explicit scale ablation and backend limitation; avoid sensor/controller-agnostic claim | P1 |
| Multiple consumed analyses invite selection bias | many mechanisms were tested on overlapping scenes | Separate development/confirmation tables, immutable manifests, frozen gates, no pooled Attempt7+Phase2 p-value | P0 |

## 3. Minimum evidence package before submission

### Mandatory

1. A prospective, scene-disjoint role-free mixed Novel/Revisit confirmation
   using the final frozen proposal rule.
2. Report the completed consumed Novel forced-anchor attribution as a negative
   mechanism result; do not spend the final population on further raw-DINO
   rescue after its promotion gate failed.
3. At least one external/public benchmark result testing the actual CEC
   Revisit-bearing interface with exact scope labeling. The completed GOAT
   first-ImageGoal semantic-arrival adapter failed its frozen gate and is not a
   substitute.
4. Revisit support-stratified results, including a non-saturated band.
5. Runtime/memory overhead and exact fallback audit.
6. Main comparisons reported with N, scene count, paired gain/loss, exact
   McNemar p, and scene-cluster confidence interval.

### Strongly desirable

1. HLoc or another hierarchical-localization backend under the same bearing
   and fallback interface, to show that the abstraction is not tied to one
   matcher implementation.
2. Full sequential GOAT ImageGoal evaluation by target index, separating
   previously observed and unobserved targets.
3. A controlled certificate evidence ablation: retrieval only, epipolar
   precheck only, full PnP certificate.

### Not required for this paper

- A new large learned decoder merely to add trainable parameters.
- A deployable Novel frontier ranker.
- X-NavDP as the default controller without a significant paired advantage.
- Graph rescue, active scanning, or broader candidate chains already rejected
  by current evidence.

## 4. Recommended claim hierarchy

### Primary claim

Causal episodic visual history can improve Revisit navigation of a frozen
ImageGoal diffusion policy through a scale-free directional interface.

### Secondary claim

Geometric witnessing provides role-free authorization and exact fallback,
making memory utility and unsupported-goal interference independently
measurable.

### Analysis claim

Long-history content addressing and action authorization, rather than local
controller capacity, explain why the explicit training-free factorization
outperformed the tested learned replacements.

### Forbidden headline claims

- first use of retrieval or PnP for navigation;
- guaranteed safety or zero false positives;
- solved Novel ImageGoal navigation;
- full GOAT state of the art;
- learned methods are generally inferior;
- controller- or sensor-agnostic performance.

## 5. Submission decision rule

Proceed toward submission only if the final evidence package supports all
three axes:

1. **utility:** prospective Revisit improvement over native;
2. **authorization:** lower Novel intervention/harm than the always-on raw
   baseline at comparable Revisit coverage;
3. **external validity:** a public-protocol result beyond the custom role-pair
   construction.

If only utility is established, the work remains a strong internal system
result but is vulnerable as a full conference paper. If utility and
authorization are established but the external benchmark remains incomplete,
the method story is credible but the evaluation package is still the dominant
review risk.
