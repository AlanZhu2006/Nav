# Paper Abstract and Contributions Draft

**Date:** 2026-08-15  
**Scope:** claim-safe writing draft for the current geometry-first Certified
Episodic Compass. Consumed semantic-first Gate B tied geometry-first exactly
(`25/28` each, paired `+0/-0`) and therefore did not change the frozen method
or enter the abstract.

## Working title

**Certified Episodic Compass: Geometrically Authorized Visual Memory for
Frozen ImageGoal Navigation**

Short alternative:

**When Should Visual Memory Control a Frozen Navigation Policy?**

## One-sentence paper story

Causal visual experience should not replace a frozen ImageGoal navigator; it
should propose a previously observed place, prove that proposal geometrically,
and communicate only the direction needed by the existing controller.

## Abstract draft

Frozen ImageGoal policies are capable local controllers, yet they repeatedly
re-explore places that they have already observed. Naively attaching long-term
visual memory is risky: retrieval can return a plausible but unsupported past
view, and an always-on memory direction can alter exploration even when the
goal is novel. We formulate episodic reuse as an open-set action-authorization
problem and introduce **Certified Episodic Compass (CEC)**. Causal visual
history proposes a place hypothesis; an image-pair geometric witness decides
whether that hypothesis may influence control; and only a scale-free bearing,
rather than a metric waypoint or map, crosses into an unchanged diffusion
navigation policy. Unsupported hypotheses reproduce native behavior exactly.
Across paired Matterport3D evaluations, episodic direction produces large
Revisit gains, including `4/40 -> 19/40` in the original two-leg test and
`5/19 -> 16/19` in an actual-online three-leg test. In two separate held-out
mixed-role populations, CEC accepted no evaluated Novel query (`0/9` and
`0/19` takeovers) while retaining strong Revisit utility. We additionally show
why task-specific learned replacements failed under the same actionable
contract: candidate-free long-history addressing fell from `18/20` to `5/20`,
while learned proposal ranking reduced certificate-actionable coverage. These
results support a minimal, training-free interface for converting causal
visual experience into evidence-carrying directional control, while exposing
the remaining gap between empirical geometric abstention and general open-set
safety.

## Contribution statements

1. **Open-set memory authorization.** We cast continual ImageGoal memory reuse
   as a hypothesis/witness/authority problem in which the runtime is never told
   whether a target is Novel or Revisit.
2. **A minimal scale-free control interface.** A certified historical
   hypothesis transfers only a normalized bearing through a fixed residual to
   a frozen NavDP controller; monocular metric scale, maps, policy gradients,
   and role labels do not cross the boundary.
3. **Causal continual evaluation.** We replay byte-verified actual-online
   histories and report Revisit utility separately from Novel interference,
   with paired gain/loss counts, exact McNemar tests, scene-cluster intervals,
   and exact-fallback audits.
4. **Evidence-driven factorization.** Closed-loop and actionable-coverage
   ablations identify long-history content addressing and authorization—not
   local controller capacity or simply too little training—as the failure
   points of the tested learned alternatives.

## Claims deliberately excluded from the abstract

- CEC significantly exceeds raw-DINO fixed bearing on Revisit SR.
- The geometric witness is a formal or zero-error safety certificate.
- A deployable memory direction improves Novel ImageGoal navigation.
- The failed GOAT first-goal semantic-arrival adapter validates CEC externally.
- Semantic-first is superior before a fresh scene-disjoint confirmation.
- Retrieval, local matching, monocular depth, or PnP is individually novel.

## Result insertion rule after Gate B

The consumed Gate-B result appears only in the method-development/ablation
section. Its exact tie leaves both the title and abstract unchanged. The
post-hoc finding that different anchors yielded bearings within `4.478°` is a
mechanism diagnostic, not confirmation evidence or an abstract-level claim.
