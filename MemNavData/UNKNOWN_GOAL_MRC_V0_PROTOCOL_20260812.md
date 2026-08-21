# Unknown-Goal Memory Renderability Certificate (MRC-v0)

Date frozen: 2026-08-12

## 0. Post-smoke hold (2026-08-12)

The 24-session Stage-S contract/timing smoke passed, but a subsequent
label-authorized audit of older artifacts found a target mismatch, strong
scene-scale nuisance, highly correlated nominal views, and a top-1 proposal
ceiling.  **Stage F is therefore on hold and must not be submitted** until the
local T1--T4 attribution tests in
`MRC_SIGNAL_ATTRIBUTION_AND_LITERATURE_20260812.md` show genuinely new
decision information.  This hold supersedes the earlier automatic
"Stage S pass -> Stage F authorized" transition; it does not alter or consume
the frozen Stage-F evaluation population.

## 1. The one unresolved question

The benchmark tells us whether a goal is Novel or Revisit, but deployment does
not. Repeated experiments have already shown that another classifier over the
same DINO/RANSAC summary is not a justified next step. MRC-v0 asks a narrower,
independent question:

> Can a frozen geometric foundation model consistently insert the query goal
> into the local 3-D map around the raw-DINO top-1 memory anchor?

This is a **memory certificate**, not a new navigation controller and not a
bearing estimator. If the certificate passes, use the already validated
Revisit residual. If it abstains, leave Novel navigation to native NavDP.

## 2. Frozen evidence path

1. Retrieve exactly one memory anchor using raw DINO cosine. No teacher label,
   RANSAC score, learned localizer, or goal-kind metadata may alter retrieval.
2. Select exactly three historical views in radius four around that anchor.
   The preferred clip is `[-4, 0, +4]`. At a scale/decision boundary, use the
   deterministic `maximin_spacing_v1` rule. The center view is mandatory.
3. Replay only the causal prefix and append the goal independently at each of
   the three anchors using frozen LingBot-map.
4. Export pose dispersion, 3-D cloud overlap, refinement magnitude, depth
   confidence, and the predicted relative goal vectors.
5. Later, fit only a small scene-grouped OOF certificate on **train scenes**.
   The development and final-reserved roles are forbidden for design,
   threshold selection, or calibration.

The adaptive clip prevents `n_hypotheses` from revealing whether an anchor is
near a trajectory boundary. A preflight audit over all 480 train sessions found
that every raw-DINO top-1 anchor supports exactly three legal views: 449 use
`[-4,0,+4]`, and 31 use a deterministic boundary-safe pattern.

## 3. Stage S: 24-session contract/timing smoke

The frozen session list is
`unknown_goal_mrc_v0_smoke_sessions_20260812.json`. It is the 24 lowest salted
SHA-256 session IDs from all 480 train sessions; labels were not used. It covers
20 scenes.

Stage S passes only if:

- all source, teacher, manifest, scale, model, container, and overlay pins pass;
- there is exactly one deployment-top-1 row for each of the 24 sessions;
- every row has exactly three unique offsets, includes offset zero, stays in
  radius four, and precedes its causal decision frame;
- every required pose/overlap/refinement output is finite;
- the run completes with elapsed-time and peak-CUDA-memory receipts.

No label-conditioned accuracy, AUC, feature choice, threshold, or method claim
may be derived from Stage S. Its receipt status is deliberately
`contract_smoke_passed_not_effectiveness_evidence`.

## 4. Original Stage F: full train-only evidence gate (held)

The original protocol authorized Stage F only after Stage S.  The post-smoke
hold in Section 0 now blocks that transition.  If the hold is ever lifted by
the stated local tests, Stage F would collect the same frozen evidence
for all 480 train sessions and evaluates the following three fixed models with
nested scene-grouped out-of-fold predictions:

- `D`: raw DINO top-1 confidence only;
- `H`: MRC geometric evidence only;
- `D+H`: DINO plus the same frozen MRC evidence.

No architecture search is allowed after viewing OOF results. The certificate
advances to fresh closed-loop confirmation only if, in every frozen seed:

- strict Novel false activations are no more than 9;
- wrong-anchor activations are no more than 14;
- correct-anchor activations exceed 93;
- `D+H` improves over `D` in scene-clustered paired analysis rather than only
  candidate AUC.

Failure means MRC is retired as the unknown-goal selector. It does not weaken
the known-Revisit result.

## 5. What this experiment can and cannot establish

Passing Stage F would establish that goal-to-memory **renderability** supplies
independent deployment evidence for choosing the Revisit expert. It would not
yet establish an SR gain; that requires one fresh, same-process closed-loop
comparison after the certificate and threshold are frozen.

The current evidence-backed fallback remains unchanged:

- known Revisit: raw-DINO top-1 direct residual;
- known Novel: native NavDP;
- unknown goal kind: native NavDP until MRC passes both Stage F and fresh
  closed-loop confirmation.
