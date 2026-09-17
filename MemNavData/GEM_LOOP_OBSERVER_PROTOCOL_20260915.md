# Frozen loop-observer reference experiment

This is a memory-side reference implementation using MASt3R-SLAM retrieval
and pair verification while retaining the native LingBot archive and original
goal SP+LG/PnP. It is neither a new claimed matcher nor complete MASt3R-SLAM,
and it does not publish unvalidated coordinates into the live memory.

Use exactly the four already consumed native_interval7 A/B/A traces and 36
queries frozen in gem_spatial_revisit_20260915/attempt_003. Candidate pool is
the same committed frames 8+7k at least64 frames old. Retain this fixed pool to
separate candidate retrieval from keyframe insertion; do not claim reproduction
of upstream adaptive keyframing. All images are actual earlier observations.

Two retrievers, each with at most3 candidates: original frozen DINO ordering,
and official incremental ASMK with the pretrained MASt3R codebook/whitening
and default minimum score0.005. Add only temporally eligible references to
the database before each query. Current images never self-enter the index;
each episode gets a fresh database. No GT, future observations or depth
confidence scores are used for retrieval. Different score thresholds can
produce fewer than3 ASMK candidates; report actual counts, not equal-cost claims.

Two independent verification arms on the union of candidate pairs: original
SP+LG, and official MASt3R-SLAM symmetric pair decoding/matching with Q>1.5
and both valid fractions>=0.1 for nonconsecutive edges. Reuse archived SP+LG
results wherever available; only newly selected ASMK pairs require new SP+LG.
Do not invoke one arm as fallback for the other. Also evaluate the two already
known task7 failures(327/204,328/218) as labeled extra diagnostic pairs, never
as additional choices in the primary top3 observer policy.

For native-map pose transfer, convert MASt3R pixel addresses through its
official resize/crop inverse into LingBot's original padded raster. Uniformly
subsample valid current-pixel addresses to at most2048, without score sorting.
This cap is an explicit adapter choice, not upstream SLAM. Use original stored
LingBot depth/confidence/reference pose and current frame's predicted K,
the existing F precheck, PnP and certificate without changing thresholds.
The SP+LG current-K control already exists and is reused on common pairs.
MASt3R arm acceptance requires both official pair verification and native-map
PnP acceptance. Score the two stages separately to reveal transfer failures.

The primary deterministic policy chooses the first passing reference in each
retriever's original rank order. Report 36-query coverage, direction-scored
count, median/P95, errors>15deg and>90deg, and paired native LingBot errors on
the same chosen reference. Separately report top1 and all-pair results. Bearing
points from current camera to historical reference, not to the task goal;
exclude true horizontal distances<0.5m only from bearing scoring, never from
selection. GT is evaluator-only after inference completes. These four scenes
are development data, not independent generalization evidence.

Validate causal identity, raster conversion, unchanged source/result hashes,
and direct agreement of the extracted pair decision with the official
FactorGraph.add_factors using the already decoded tensors (no duplicate model
calls). Record all candidates, rejected evidence, transformed correspondences,
native-PnP outputs, feature caching costs, component timing and memory. No
threshold/layer sweeps, model replay, extra simulated views, closed-loop
navigation, manuscript/default changes, commits or pushes.

Success of this experiment means completion of a bounded and auditable
retrieval/verification reference, not automatic publication readiness or a
proven long-range coordinate updater. It determines which remaining problem
is candidate retrieval, correspondence verification or native-geometry transfer.
