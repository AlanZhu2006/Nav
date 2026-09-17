# Reviewer experiment expansion: assessment and proposed protocol

Prepared 2026-09-16, China time. Status: CPU assessment and candidate identities prepared; no new GPU jobs submitted. This document does not change the frozen 13-query protocol, implementation, or manuscript.

## Evidence motivating the next stage

The completed main-table experiment contains 13 paired histories from 12 scenes, with five arms and 65 rollouts. Successes are native 3, fixed positive half-turn 8, no explicit turn 7, initial memory-guided turn only 6, and full GEM 11. Full versus fixed turn has five paired wins and two losses; the exact two-sided McNemar p-value is 0.453125. This is useful preliminary evidence, not an established advantage over simple turning.

The full Table I population has 70 Revisit histories in 46 scenes. Absolute initial goal bearing is 0–60 degrees in two queries, 60–120 degrees in three, and 120–180 degrees in 65. Completing this population answers attribution for Table I; it does not establish performance across balanced orientations or long-distance navigation.

A CPU projection audit sampled all 41 headings along the fixed positive 180-degree turn. Only five of 70 queries, including three of the completed 13, have a goal-surface overlap upper bound below 0.1 at every sampled heading. Occlusion is omitted: a high upper bound does not establish visibility, and this is not a full 360-degree scan. Even a low bound is not proof that navigation requires memory. A direct retrieval-domain intervention is needed.

## A. Complete control attribution on the existing main population

Retain the completed 13 histories. The remaining 57 identities are listed without outcome-based selection in `.diagnostics/gem_reviewer_p0_20260915/expansion_001/main70_remaining57_candidate.json`. They cover 39 scenes. Keep the same five arms, original goals and histories, 600-action budget, original 1 m arrival criterion, legacy/window32 geometry, canonical reference depth, strict certificate, and frozen NavDP.

Under the existing within-history GPU-pairing protocol, the expansion requires 57 × 5 = 285 additional rollouts, giving 350 across all 70 histories. The measured pilot runtime extrapolates to approximately 14.6 A100 GPU-hours for the additional 57; this excludes queueing and is not a wall-clock completion promise. Original main-table native/full results remain reference checks. The already completed 65 supplemental rollouts are not rerun.

Use the existing three primary contrasts: full versus fixed turn, initial-only, and no-turn. Report all paired wins/losses, effect sizes, scene-cluster intervals, and Holm-adjusted exact tests. Fix coverage at all 70 before examining new control outcomes; do not extend the population until a p-value crosses a threshold. Distinguish the original 13-query subgroup from its 57-query complement. Neither the combined population nor the complement is a newly collected independent benchmark.

Separate cases with and without an actual initial guided turn. Two original full-GEM main-table trajectories contain three explicit turns, so an expanded full-versus-initial comparison can include later turning as well as continued bearing. Report those actions rather than interpreting the entire difference as bearing alone. Summaries of trajectory bearing should use queries as units and identify distance-to-goal effects, rather than treating every correlated decision as a separate sample.

Before submission, generalize the launcher and verifier, which currently require exactly 13 cells/65 rollouts, in a new source bundle. Preserve the old bundle and receipts. Verify candidate/input identities and output-file capacity. Keep the existing geometry and control algorithms unchanged.

## B. Directly measure the contribution of access to older archive entries

On the same 13 metadata-selected queries, compare full eligible historical retrieval with retrieval restricted to the exact last seven historical decision observations. Use `decision_steps[-7:]` from the archived history metadata, and verify the raw-observation-to-archive-ID mapping before execution. The candidate domain manifest is `expansion_001/recent_seven_candidate.json` beneath the diagnostic root.

Change only which archived entries may be retrieved. Preserve the full causal geometry writer/state, current monocular depth, model weights, retrieval ranking rule, SP/LightGlue/PnP, verification thresholds, one-shot query timing, and control interface. Apply the whitelist before candidate selection. Record every eligible and selected index; disallow hidden access to older entries through another retrieval path. This estimates the contribution of older external archive access, not the elimination of all past information from the geometric model.

The original subgroup condition is old support >= 0.5, maximum support over the last seven historical decision views < 0.1, and initial current-view overlap upper bound < 0.1. Seven decision observations are not 64 raw frames. In particular, pRbA3pwrgk9/episode_0004 and sT4fr6TAbpF/episode_0002 have recent-64 support above 0.1. A recent-64 experiment can be a separate archive-window baseline, but must not be described as the same subgroup-matched short-context intervention.

First use these existing 13 as a bounded mechanistic diagnostic. Report accepted recalls, support age, direction error, paired navigation success, and actions for both domains. A reference full-history arm must match the archived full-GEM execution before pooling. Reuse a compatible full result where the execution contract permits; otherwise label new matched reference runs explicitly. No new sample-size claim follows from adding arms on the same histories.

For a subsequent independent cohort, prioritize additional scenes, weak recent support, and trajectories whose goal evidence remains limited after active observation. Include front/side/rear starts and occlusion/detours where feasible. Determine eligibility from geometry and history before policy outcomes, freeze the sampling rule and size after the feasibility inventory, and keep this cohort separate from Table I. Do not assume that simply sampling more backward-facing queries resolves the task-distribution limitation.

## C. Test whether goal-position readout adds value beyond returning to a support camera

Use all seven existing accepted queries with at least 1 m goal-to-selected-support position offset as an initial diagnostic. Keep the same verified support and gate. Compare control directed toward its estimated historical camera position with control directed toward the PnP-estimated goal position. Use predicted geometry in both arms; ground truth is only for evaluation and stratification. This is a matched-support position-readout comparison, not a relabeling of the paper's raw-retrieval baseline.

Run both arms under a common 600-action budget, continuing until a shared 0.3 m automatic arrival threshold or budget exhaustion. Record first passage at 1.0, 0.5, and 0.3 m, actual endpoint error, geodesic distance, action/turn counts, and bearing error along the trajectory. Save complete trajectories so each threshold is computed independently. The older rollouts stopped at 1 m cannot establish tighter-threshold success and must not be reused for that claim. This still does not evaluate autonomous STOP.

These seven queries are too few for a broad success-rate claim. Their purpose is to reveal whether support-camera substitution fails in the predicted way and to establish the protocol for a separately frozen, larger offset cohort. Retain unfavorable outcomes. If position readout provides little benefit even with meaningful offsets, narrow the contribution claim rather than changing thresholds on these cases.

## Priority and interpretation

Prepare A and B next, with B providing the more direct test of the archive-memory contribution; C addresses the separate goal-localization question. Keep the current architecture fixed while collecting this evidence. The observed 47.77-degree maximum navigation bearing error warrants diagnosis, but does not by itself identify a component to redesign or justify tuning gates on the test cases.

Power calculations in `expansion_001/feasibility.json` are sensitivity analyses only. Assuming independent queries and the pilot win/loss rates, an unadjusted two-sided exact comparison has approximately 72% power at N=70 and 87% at N=100. These are not promises: the effect is uncertain, scenes repeat, the remaining population differs, and multiple-comparison correction lowers power. The N=70 target is chosen to complete the existing population, not to guarantee significance. The implementation uses the binomial formulation documented in [statsmodels' McNemar reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

Evidence: `.diagnostics/gem_reviewer_p0_20260915/hpc_001/reduced.json`, `hpc_001/navigation_bearing_audit.json`, `expansion_001/feasibility.json`, and `expansion_001/planning_receipt.json`.
