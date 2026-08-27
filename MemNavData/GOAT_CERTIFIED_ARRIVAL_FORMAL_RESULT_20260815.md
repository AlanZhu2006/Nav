# GOAT First-ImageGoal Certified-Arrival Formal Result

**Date:** 2026-08-15  
**Status:** complete negative confirmation; independently verified  
**Scope:** 20 scene-disjoint GOAT episodes, first ImageGoal only. This is not a
full sequential GOAT score and is not a Revisit-memory benchmark.

## 1. Frozen provenance

- immutable source bundle:
  `/scratch/yz11502/Research/source_bundles/goat_certified_arrival_bc3c3c887d1063ee`
- source receipt SHA:
  `bc3c3c887d1063ee600254fea0f1533118fcdc6423044463775b2feb33c355d1`
- repair smoke: `15761753`, completed `00:02:27`, exit `0:0`;
- formal array: `15761754`, all 20 records completed;
- sealed summary: `15761755`, completed `00:00:16`, exit `0:0`;
- independent verifier: `15761756`, completed `00:00:10`, exit `0:0`;
- formal root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_certified_arrival_20260815/formal_seedrepair_intent_20260814T213723Z`;
- manifest SHA:
  `3120625b8b6e86d9d517f08dd4d3b366c0d417cfdb738b50dc01834186458b79`;
- sealed report SHA:
  `d52a8dc611058fc3c3a454a7ec38609d913a7b949eed25639a63fd6ba59e1a88`;
- independent verification SHA:
  `4c9a4225b7410364fe3d17580dfee4810f8e5bc6b8945be1147bf8f94a094539`;
- independent result: `verified=true`.

The only repair relative to the failed parent was the preregistered service
seed mapping `episode_hash % 2**32`; the same uint32 seed was sent to NavDP and
MemNav. Ground truth was recorded after each frozen decision and was never an
input to the decision.

## 2. Primary result

| Quantity | Result |
|---|---:|
| episodes / scenes | `20 / 20` |
| certified successes | `0/20` |
| certified STOPs | `0` |
| true certified STOPs | `0` |
| false certified STOPs | `0` |
| true-STOP scenes | `0` |
| safe stalls | `0` |
| forced guard stops | `20` |
| legacy first-zero counterfactual successes | `1/20` |
| certified minus legacy paired gain/loss | `+0/-1`, McNemar `p=1.0` |
| same-batch motion fallbacks | `30` |
| extra resamples | `0` |

The preregistered gate required all of:

1. at most zero false certified STOPs;
2. at least five true certified STOPs;
3. true STOPs in at least five scenes.

Observed values were `0`, `0`, and `0`; therefore
`primary_gate_passed=false` and the frozen next action is
`do_not_claim_deployable_goat_semantic_stop`.

Zero false STOPs is vacuous here because certified coverage was also zero. It
must not be described as successful safety transfer.

## 3. Frozen-result failure localization

There were 28 native-zero proposal events across the 20 episodes:

| Frozen decision reason | Count |
|---|---:|
| `causal_scale_prefix_incomplete` | 6 |
| `geometry_certificate_rejected` | 20 |
| `predicted_distance_above_frozen_threshold` | 2 |
| `certified_arrival` | 0 |

The simulator distance, read only after each decision, showed:

- 26/28 zero proposals were outside the official `<0.25 m` arrival region;
- only 2/28 were inside it;
- distance range was `0.142--13.550 m`, with median `6.219 m`.

Thus NavDP's zero endpoint is not itself a calibrated semantic STOP event in
this runtime. The certificate correctly prevented many far-away stops, but it
did not recover any usable arrival coverage.

Among the 22 events with a mature causal prefix:

- 18 failed the fundamental-inlier precheck;
- 1 failed query-hull coverage;
- 3 reached PnP;
- one of those failed the PnP minimum-inlier certificate;
- two passed the geometric certificate but produced frozen metric-distance
  estimates `5.527 m` and `4.550 m`, while their official distances were
  `0.471 m` and `1.118 m`; both were correctly rejected by the `0.075 m` gate.

The two events that were actually inside `<0.25 m` both failed before PnP: one
for insufficient fundamental inliers and one for insufficient query-hull
coverage. The directly observed failure locus is therefore zero arrival recall
under the frozen current-view-to-goal geometric witness on this population.

## 4. Interpretation

This result rejects the proposed deployable first-ImageGoal semantic-arrival
adapter. It does **not** reject the established CEC Revisit result:

- this experiment evaluated STOP authorization from the current view to a
  first ImageGoal;
- it did not evaluate retrieval from a previously observed episodic history;
- it did not run the certified scale-free bearing takeover used by CEC;
- it did not compare native versus episodic Revisit navigation.

The failure nevertheless matters for the paper. It removes the active GOAT
row as positive external-validity evidence and demonstrates that a certificate
calibrated on the internal arrival audit did not transfer to GOAT's visual and
trajectory distribution. The first-ImageGoal adapter must not be retuned on
these 20 held-out episodes.

## 5. Frozen decision and next public-benchmark direction

1. Stop this semantic-arrival branch; do not sweep the `0.075 m` threshold,
   matching thresholds, or scale contract on the formal episodes.
2. Retain the 20 episodes as a negative external-transfer result and report
   the complete coverage/error accounting if discussed.
3. A future GOAT experiment must test the actual CEC claim: causal episodic
   support proposes a prior place, an unchanged geometric witness authorizes
   it, and only bearing enters frozen NavDP. It should be run on sequential
   ImageGoal subtasks with separately reported supported versus unsupported
   histories, not by turning first-goal zero endpoints into STOP commands. The
   frozen dataset-only audit already establishes a viable recurrence stratum:
   `338/822` ImageGoal subtasks repeat a previously named instance across 211
   episodes; runtime support must still be derived only from causal RGB.
4. Until that protocol is frozen and completed, public-benchmark external
   validity remains open and cannot appear as an established paper claim.
