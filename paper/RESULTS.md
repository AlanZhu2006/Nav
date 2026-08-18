# Evidence ledger

This file separates confirmatory evidence, strong internal evidence, mechanisms, and negative results. Counts are never pooled across machines or overlapping populations.

## Primary paper evidence

### Final14: fresh MP3D mixed-role evaluation

Population: 21 histories, 10 scene clusters, 42 Natural queries; each history contributes one Novel and one Revisit query. Runtime role visibility is `none`. The frozen target was 28 histories, so the result is valid and independently verified but underpowered relative to its preregistered history target.

| Arm | Novel SR | Revisit SR | Aggregate SR | Aggregate SPL |
|---|---:|---:|---:|---:|
| native | 7/21 | 4/21 | 11/42 | 0.1217 |
| raw fixed bearing | 2/21 | 19/21 | 21/42 | 0.3509 |
| geometry fixed | 9/21 | 18/21 | 27/42 | 0.4145 |
| learned Pi3X spatial | 8/21 | 19/21 | 27/42 | 0.4249 |
| **CEC** | **8/21** | **20/21** | **28/42** | **0.4400** |

Key paired contrasts:

| Contrast | Role | Gains/losses | Risk difference | Exact McNemar |
|---|---|---:|---:|---:|
| CEC - native | Revisit | +16/-0 | +76.19 pp | 3.05e-5 |
| CEC - native | Novel | +1/-0 | +4.76 pp | 1.0 |
| CEC - native | all | +17/-0 | +40.48 pp | 1.53e-5 |
| CEC - raw | Revisit | +1/-0 | +4.76 pp | 1.0 |
| CEC - raw | Novel | +7/-1 | +28.57 pp | 0.0703 |
| CEC - raw | all | +8/-1 | +16.67 pp | 0.0391 |

Novel authorization audit: 19/21 Novel queries were fully rejected and exactly reproduced native behavior; two were accepted, with no paired net loss. This is an empirical open-set result, not a formal safety guarantee.

Sources:

- [`paper/results/final14/paper_role_pair_summary.json`](results/final14/paper_role_pair_summary.json)
- [`paper/results/final14/paper_role_pair_independent_verification.json`](results/final14/paper_role_pair_independent_verification.json)
- [`MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md`](../MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md)

### HM3D external Revisit transfer

Population: 36 intention-to-treat episodes over nine constructible held-out HM3D scenes. Goal A succeeded for 21 episodes and is byte-identically shared by all B arms.

| Arm | B given A | Joint |
|---|---:|---:|
| native | 7/21 | 7/36 |
| old geometry | 17/21 | 17/36 |
| raw fixed, oracle role | 18/21 | 18/36 |
| **role-free CEC** | **19/21** | **19/36** |

CEC versus native: +12/-0, conditional risk difference +57.14 pp with scene-cluster CI [+36.36,+78.95], exact p=0.000488. The gains span eight of nine scenes. CEC does not significantly exceed geometry or the non-deployable oracle-role raw arm in this population.

Sources:

- [`paper/results/hm3d/hm3d_heldout_val10_revisit_summary.json`](results/hm3d/hm3d_heldout_val10_revisit_summary.json)
- [`paper/results/hm3d/hm3d_heldout_val10_revisit_independent_verification.json`](results/hm3d/hm3d_heldout_val10_revisit_independent_verification.json)
- [`MemNavData/HM3D_HELDOUT_VAL10_FORMAL_RESULT_20260817.md`](../MemNavData/HM3D_HELDOUT_VAL10_FORMAL_RESULT_20260817.md)

## Continual and high-support evidence

| Evaluation | Native | Memory/CEC | Paired interpretation |
|---|---:|---:|---|
| Original 40-episode two-leg joint | 4/40 | geometry 19/40 | +15/-0, p=6.10e-5; earliest clean memory result |
| Fresh160 high-support B given A | 27/120 | CEC 112/120 | strong utility; CEC vs raw 106/120 is +9/-3, p=0.146 |
| Actual-online three-leg NNR | 5/19 | CEC 16/19 | +11/-0, p=0.000977; 8 scene clusters |
| Fresh20 double Revisit joint | 0/20 | role-free CEC 17/20 | feasibility; preservation-specific contrast underpowered |

The three-leg sealed report and independent recount are under [`paper/results/three_leg`](results/three_leg).

## Mechanisms, not deployable methods

| Experiment | Result | Valid conclusion |
|---|---|---|
| Novel-A oracle bearing, N=40 | 28/40 -> 40/40, +12/-0, p=0.000488 | direction is a recoverable bottleneck; oracle is privileged |
| Pi3X b8 -> b16 causal bridge | <=30 degree positives 585/701 -> 659/701 | causally observed bridges reconnect distant visual frames |
| Bearing tolerance, N=6 | about +/-30 degrees sufficient; 45 critical; 60 degrades | small-sample mechanism only |

## Negative results that constrain the method

| Route | Result | Decision |
|---|---|---|
| temporal top-K | 18/40 vs 18/40, p=1.0 | candidate count is not the bottleneck |
| active glance | best 25/40 vs native 31/40 | stop scanning/intervention branch |
| semantic-first proposal | 25/28 vs geometry-first 25/28 | retain frozen CEC order |
| graph rescue | 16/19 vs equal-budget 16/19 | remove from primary method |
| X-NavDP controller | 21/26 vs 20/26, +2/-1, p=1.0 | controller replacement is not a claim |
| candidate-free GCT | 5/20 vs addressed 18/20 | retain explicit long-history addressing |
| CDEC learned proposal | actionable +1/-8 vs geometry, p=0.039 | do not replace certificate |
| learned Pi3X proof | 19/21 Revisit vs CEC 20/21; L2/L3 fail | secondary learned result only |
| Replica | 0 constructible formal histories | benchmark-contract failure, no efficacy claim |
| GOAT first ImageGoal | no executable Revisit intervention | incompatible first-goal protocol; no score claim |

## Latency

- CEC first query median: 3.40 s over 42 Natural queries.
- CEC Revisit first query median: 10.64 s.
- CEC first query p95: approximately 26.35 s.
- Cached bearing update median: 0.152 ms.
- Stored history: median 166 frames, maximum 364.

The current implementation establishes utility, not a final real-time front end. See [`paper/results/final14/final14_cec_latency_audit_20260818.json`](results/final14/final14_cec_latency_audit_20260818.json).
