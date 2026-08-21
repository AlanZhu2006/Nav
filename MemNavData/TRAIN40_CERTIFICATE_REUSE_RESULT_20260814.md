# Train40 certificate exhaustive reuse result

Date: 2026-08-14

Status: complete, independently recounted, train/consumed only. No development,
scene-disjoint final, or blind data were read. This is an offline actionability
audit, not a closed-loop SR result.

## Why the queued GPU run was stopped

The completed CDEC dual-proposal collector already contains exactly one frozen
geometry-proposal measurement for every session in the later train40 challenge
manifest:

- manifest and geometry rows contain the same `480/480` session IDs across 40
  scenes;
- source selection origin is `lightglue_fundamental_rank_v1`;
- the collector used the deployment-aligned center hypothesis (`offset=0`),
  full causal replay, LightGlue PnP, top-8, and candidate gap 4;
- its raw CSV and collector report are bound to an earlier independent verifier
  with `verified=true`.

On the 24 sessions shared with the later three-view confirmation, the chosen
anchor is equal `24/24` and the center certificate decision is equal `24/24`.
The queued job `15703087` would therefore have repeated the same center
certificate endpoint while adding neighbor hypotheses not used by the frozen
runtime decision. It was cancelled while pending under `QOSGrpGRES`, with
elapsed GPU time `00:00:00`.

The extraction is mechanical and label-blind: it selects the already recorded
geometry origin from both proposal rows and requires exact equality with the
complete frozen manifest. No session, threshold, or operating point is selected
from the outcomes.

## Exhaustive result

The existing v2 certificate requires one center PnP hypothesis to have at least
16 inliers, at least 5% query and reference hull coverage, and at most 2 px
reprojection RMSE. The historical audit-only actionability label is global goal
position error at most 0.75 m.

| quantity | result |
|---|---:|
| sessions / scenes | 480 / 40 |
| accepted | 131 |
| GT-actionable | 153 |
| TP / FP / FN / TN | **122 / 9 / 31 / 318** |
| precision | **93.13%** |
| precision Wilson 95% CI | 87.46%--96.34% |
| recall | 79.74% |
| recall Wilson 95% CI | 72.68%--85.34% |
| false-accept rate among non-actionable | 2.75% |

The previously frozen zero-false-positive effectiveness gate does **not** pass
on the exhaustive population. This forbids a claim that the operational
certificate guarantees a correct metric PointGoal on unknown inputs.

## Stratified counts

| audit-only stratum | N | accepted | TP / FP | FN / TN | precision |
|---|---:|---:|---:|---:|---:|
| session max support `<=0.10` | 274 | 2 | 0 / 2 | 7 / 265 | 0% |
| session max support `(0.10,0.50]` | 51 | 21 | 18 / 3 | 3 / 27 | 85.71% |
| session max support `>0.50` | 155 | 108 | 104 / 4 | 21 / 26 | 96.30% |
| history gap `<=32` | 101 | 14 | 12 / 2 | 14 / 73 | 85.71% |
| history gap `33--96` | 116 | 19 | 18 / 1 | 1 / 96 | 94.74% |
| history gap `>96` | 263 | 98 | 92 / 6 | 16 / 149 | 93.88% |
| `goal_b_t0` | 160 | 12 | 10 / 2 | 5 / 143 | 83.33% |
| `goal_b_midpoint_t1` | 160 | 19 | 17 / 2 | 14 / 127 | 89.47% |
| factual/counterfactual `goal_c_t0` | 160 | 100 | 95 / 5 | 12 / 48 | 95.00% |

Co-visibility and causal state are audit coordinates only. They are not visible
to the runtime and cannot be converted post hoc into a Novel/Revisit gate.
In particular, adding `max_support > 0.10` as a deployment condition would use
ground truth that the robot does not possess.

The long-delay result is positive: most certified-actionable cases occur beyond
96 history frames, so the method is not merely exploiting NavDP's short FIFO.
On the 122 accepted-actionable rows, PnP improves median global position error
from 0.207 m to 0.117 m and anchor-relative direction error from 5.54 degrees to
2.11 degrees.

## Important deployment-label mismatch

The 0.75 m actionability definition was created for the earlier metric
PointGoal interface. The deployed adapter no longer exposes monocular metric
scale: it sends only a scale-free bearing through a fixed 2.5 m residual.
Consequently, the nine metric false positives are not automatically nine
harmful takeovers. Seven of nine have anchor-relative direction error below 30
degrees; two are clearly directionally wrong at 48.6 and 121.5 degrees. The
anchor-relative diagnostic is still not identical to the online
current-to-goal bearing consumed by NavDP.

Therefore this offline table neither proves safety nor proves nine closed-loop
losses. It identifies the exact remaining question: whether occasional
geometry-consistent but globally wrong poses cause net harm when the same
role-free adapter is run on unsupported Novel queries.

## Decision

1. Do not tune certificate thresholds against these train40 outcomes.
2. Do not rerun the same 480 center hypotheses on a GPU.
3. Keep the minimal single-proposal, scale-free certified residual; do not add
   CDEC or graph rescue.
4. The next method-changing experiment is one scene-disjoint, role-free
   mixed-goal closed-loop test with matched Novel and Revisit queries. It must
   compare native, raw-DINO direct, and the certified residual without exposing
   the role label to any arm.

Passing that experiment means preserving the Revisit gain while bounding Novel
harm. Failing it means the method remains a known-Revisit adapter; this offline
audit does not authorize another learned selector or a post-hoc support gate.

## Receipts

- source dual rows SHA256:
  `b02fb5d4940a11c83dc3ee7b49788320fbb51d027772e9a2916c6a941b659c72`;
- extracted geometry rows SHA256:
  `8e1b22901a7520e5bec5c6cb753eac9fab1342a19652d980f39264ecaa4bb24f`;
- materialization receipt SHA256:
  `fd94b5df89d66733667d9ef0a8b0508f66f2220c71438d7c296f148b36446b9f`;
- actionability audit SHA256:
  `a2cce3c981dc60db1f7e84473c8077a9602d3109cc1f8dc0b10047ec2a066e11`;
- independent recount SHA256:
  `e993abe97a3d7e648867097ec06d88a21c1d822f1a61bdc2c98e7db8fa2f4385`.

The independent implementation reconstructed every overall and stratified
TP/FP/FN/TN count directly from the extracted CSV.
