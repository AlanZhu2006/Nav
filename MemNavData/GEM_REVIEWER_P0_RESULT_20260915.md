# Reviewer P0 experiments: current evidence and execution status

Active task: quickly address attribution to turning, dependence on older-than-policy-context history, and the value/reliability of geometric goal localization. The active manuscript at `/home/asus/Research/Memnav_Paper` has not been edited.

## Complete local pilot

20 new rollouts and 8 previously verified rollouts on four consumed MP3D development histories. Revisit successes: native 1/4, unconditional +180 degree initial turn 3/4, GEM without explicit in-place turning 2/4, GEM initial guided turn only 4/4, full GEM 4/4. Novel native and unconditional-turn controls both 2/4. Every control action is budgeted, all goals and seeds are fixed, and native/scan arms cannot call episodic retrieval or PnP. Independent motion and RGB/depth input checks passed for all 28 relevant rollouts. Local experimental GPU processes exited; unrelated robot processes were preserved.

These pilot queries use the current W64 support-storage writer with reader-precision KV and the no-coverage certificate; their A histories are consumed development data. They are not additional main-table trials.

## Complete original-table audit

Main-table geometry: 69 accepted Revisit recalls out of 70, median initial bearing error 1.72 degrees, P90 6.63 degrees, maximum 16.84 degrees. Chosen-view age is 12–407 raw frames, median 90. 67/70 true goals initially lie in the rear half-plane.

All 13 queries in 12 scenes satisfying old support >=0.5, recent seven historical decision views <0.1, and current-view co-visibility upper bound <0.1 were selected independently of success. Every selected current-view upper bound is zero. Original main-table successes on this subset: NavDP 3/13 to 11/13, ViNT 0/13 to 8/13, NoMaD 4/13 to 7/13. This is a metadata-defined post-hoc subgroup, not a new independent population or an actual recent-window-only retrieval intervention.

At support-view/goal offsets of 1–2 m (seven accepted queries), the median bearing error from pointing to the same historical camera position using oracle poses is 15.84 degrees, versus GEM 1.14 degrees. This offline comparator is not Table IV raw retrieval. It establishes a directional-accuracy difference, not a closed-loop success-rate difference.

## Complete main-table five-arm controls

Frozen plan: `.diagnostics/gem_reviewer_p0_20260915/hpc_001/plan.json`, SHA-256 `accddc88afc439afba96efddfe9378f3f1e04ae09144347e699693088e58c199`.

13 histories, five NavDP arms each, 65 total rollouts. The main-table geometry (`legacy`, window32, canonical reference depth, strict certificate) is fixed to isolate control changes. These results remain separate from the W64 local pilot. No model weights, SP/LightGlue/PnP algorithms, thresholds or navigation physics were changed.

Gate: Slurm `17844106_0`, A100 node ga011, original longest history FnSn2KSrALj / episode_0002 / 564 frames. All five results independently verified: native 0/599 steps, fixed half-turn 0/384, no explicit turn 1/138, initial-only 0/599, full GEM 1/138. This case has no initial guided turn and therefore cannot by itself estimate the benefit of continued bearing after an initial alignment.

The verified launcher submitted array `17844507` after the gate completed, then exited. Each history's five arms ran on one actual GPU UUID, with a 40-minute bound calculated from the longest-history gate. Pending index1 was tried once on H100 job `17844616_1`; abnormal roughly19-second frame intervals prompted cancellation before any rollout completed. The complete interrupted directory is preserved. The identical cell subsequently completed on the original A100 stack as `17844962_1`, and original array indices2–12 also completed on A100. No H100 outcome is counted.

Final collection: 2026-09-15 22:13:53 China time. All13 histories (12 scenes),65 rollouts independently verified. Successes: native3/13, fixed turn8/13, no explicit turn7/13, initial-only6/13, fullGEM11/13. All26 native/full references reproduce original main-table success, action count, path length, SPL and explicit-turn count exactly. The frozen source and input receipts are unchanged. These results remain separate from the four-history local W64 pilot.

Full GEM versus fixed turn:5 paired wins,2 losses, exact p=0.453125. Versus initial-only:6 wins,1 loss, exact p=0.125, Holm p=0.375. Versus no-turn:4 wins,0 losses, exact p=0.125, Holm p=0.375. These small-sample comparisons do not establish statistical superiority. In the10 cases with actual initial guided alignment, fullGEM9/10 versus initial-only6/10; their guided prefixes match exactly and fullGEM has no later explicit turns. The remaining three include two cases where an initial turn is not triggered and one rejected recall.

CPU postprocessor17845161 and additional read-only bearing audit17846647 both completed0:0. Final artifacts were downloaded, checked against their hashes, and reduced locally to byte-identical outputs. The CSV was transferred as raw bytes to preserve its CRLF record delimiters. Results are in `hpc_001/reduced.json`, `controls.csv`, `RESULT.md`, `postprocess_completion.json`, and `figures/paired_controls.pdf`.

The navigation bearing audit covers150 recorded decision bearings in12 accepted histories plus one rejected history. All initial errors agree with the archived main-table audit. Three accepted queries have a maximum error above30 degrees during navigation: one failure reaches47.77 degrees (initial14.81 degrees); two successes reach34.54 and31.08 degrees. These observations do not by themselves identify drift as the cause of failure. They show why initial readout accuracy cannot stand in for whole-trajectory reliability. Every point was independently recomputed with a body-frame dot/cross-product calculation. No new model or simulator execution was used.

Primary paired contrasts are full GEM versus fixed turn, initial-only and no-turn. Reduction includes all scheduled coverage, per-query wins/losses, action counts, Holm-adjusted exact McNemar tests, and descriptive scene-cluster intervals. Cases with and without an actual initial guided turn are distinguished. Small-N nonsignificance is not equivalence.

## Evidence paths

- `.diagnostics/gem_reviewer_p0_20260915/RESULT.md`
- `.diagnostics/gem_reviewer_p0_20260915/attribution_results.csv`
- `.diagnostics/gem_reviewer_p0_20260915/attribution_002/independent_verification.json`
- `.diagnostics/gem_reviewer_p0_20260915/geometry_audit_final/geometry_audit.json`
- `.diagnostics/gem_reviewer_p0_20260915/current_overlap_bounds.json`
- `.diagnostics/gem_reviewer_p0_20260915/reviewer_figures/memory_support_and_localization.pdf`
- `.diagnostics/gem_reviewer_p0_20260915/hpc_001/`
- Remote run: `/scratch/yz11502/Research/Nav-axis-uturn-results/gem_reviewer_20260915_001`
- Remote immutable-input bundle: `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/gem_reviewer_20260915_001`

Still outside this bounded first batch: actual recent-window retrieval restriction, historical-position versus PnP closed-loop controls, tighter arrival/autonomous STOP, ViNT/NoMaD image-substitution ablations, and new real-world routes. Initial bearing accuracy does not establish long-trajectory drift robustness.
