# HM3D Table-II Powered Continual Result

Date: 2026-08-30

## Status

The complete formal chain passed:

- paired policy array `16602105_[0-53%4]`: 54/54 `COMPLETED`, zero failed;
- aggregate `16602106`: `COMPLETED 0:0`;
- independent raw verifier `16602107`: `COMPLETED 0:0`;
- conference meeting verifier `16602108`: `COMPLETED 0:0`.

No partial policy result was read before all three post-array nodes passed.

## Frozen estimand

The experiment uses actual-mono Novel-A rollouts, result-blind factual Novel-B
candidates, and a role-hidden balanced Novel/Revisit-C query for each of 20
successful, supported A+B histories from 13 HM3D scene clusters.  Native and
CEC share the same factual A+B prefix.  The identifiable treatment estimand is

`C | successful, supported factual A+B`.

It is not an unconditional three-leg joint cohort: the factual-B stage contains
183 distinct B candidates but covers 67 unique successful-A histories, with
some A prefixes repeated across B candidates.  The stage rates must not be
multiplied.

## Independently verified result

| Stage / role | n | Native | CEC | Paired gain/loss | Exact McNemar p | Native SPL | CEC SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Leg 1 Novel-A, shared factual rollout | 196 | 131/196 | 131/196 | -- | -- | 0.6011 | 0.6011 |
| Leg 2 Novel-B, shared factual rollout | 183 | 54/183 | 54/183 | -- | -- | 0.1921 | 0.1921 |
| Leg 3 Novel-C given A,B | 20 | 4/20 | 4/20 | +0/-0 | 1.0 | 0.1723 | 0.1723 |
| Leg 3 Revisit-C given A,B | 20 | 8/20 | 17/20 | +10/-1 | 0.01171875 | 0.1422 | 0.6750 |
| Leg 3 balanced all given A,B | 40 | 12/40 | 21/40 | +10/-1 | 0.01171875 | 0.1572 | 0.4236 |

Additional verified diagnostics:

- Leg-1 SR: `131/196 = 0.66837`;
- Leg-1 SPL: `0.60106` from all 196 sealed raw metric rows;
- Leg-2 SR: `54/183 = 0.29508`;
- Leg-2 SPL: `0.19211` from all 183 sealed raw metric rows;
- supported A+B prefixes entering construction: 41;
- final constructible population: 20 histories / 13 scene clusters;
- Leg-3 balanced risk difference: `+22.5 pp`;
- Leg-3 Revisit risk difference: `+45.0 pp`;
- Novel takeover queries: 1/20; Revisit takeover queries: 18/20;
- fully rejected, byte-exact native fallback queries: 21;
- raw metric rows: 80; audited causal-monocular plan receipts: 3,482;
- Revisit source segment balance: 10 from A and 10 from B.

## Claim boundary

This result establishes that accumulated causal history yields a significant
conditional Revisit benefit while preserving Novel SR on the powered hidden-role
population.  It does not establish an unconditional three-leg joint SR, and it
does not establish zero intervention on Novel: one Novel query was authorized,
although its success outcome and aggregate Novel SPL were unchanged.

The meeting verifier records:

- `verified=true`;
- `runtime_role_visibility=none`;
- `fallback_completion_used=false`;
- `threshold_relaxation_used=false`;
- `partial_policy_outcomes_used=false`;
- `unconditional_three_leg_joint_sr_reported=false`.

## Evidence

Formal root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7/table2_leg3_power/policy_authority_closure_repair_v3`

Primary independent verification:

`meeting_result/hm3d_table2_meeting_result_independent_verification.json`

SHA-256:

`a3b4adf9f5c29cab775da30fc19fd60704201070b87c35aa755ffe1e34457f50`

Raw policy independent verification:

`formal/navdp/navdp_table2_leg3_independent_verification.json`

SHA-256:

`46be8cf42ca7e42250783bab3d5b6332297076a599d6a501ee57b23f4327400a`

Post-seal factual Leg-1/Leg-2 SPL verification:

`meeting_result_stage_spl_v1/hm3d_table2_stage_spl_independent_verification.json`

SHA-256:

`321f40534e1d467f4bf47b58414ac09f714b54c4b032460ee67235debd9f97ee`

This additive verifier is bound to the primary meeting-verification SHA.  It
recounts all `196 + 183` factual rollout rows and records
`selection_or_policy_execution_performed=false`; it neither reruns a policy nor
changes the frozen population.
