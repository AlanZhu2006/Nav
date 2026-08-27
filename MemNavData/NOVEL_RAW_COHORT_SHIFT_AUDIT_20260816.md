# Attempt 7 vs Phase-2 Raw-Novel Cohort-Shift Audit

**Audit date:** 2026-08-16

**Status:** read-only post-hoc mechanism audit; no new rollout, no threshold
selection, and no confirmation claim

**Reproducible implementation:** `audit_raw_novel_cohort_shift.py`

**Machine-readable result:**
`.diagnostics/raw_novel_cohort_audit_20260816/report.json`

## 1. Executive conclusion

Attempt 7 and Phase-2 did not use different methods. The evaluation code,
controller, checkpoints, success rule, step budget, and deterministic seed
contract are materially identical. The reversal in raw-DINO Novel behavior is
therefore not explained by an implementation or checkpoint change.

The most accurate diagnosis is:

> Raw-DINO has a strong backward/U-turn directional bias. Phase-2 happened to
> contain substantially more Novel queries whose correct initial route was
> also behind the agent, and the query-specific raw bearing was unusually well
> aligned on those queries. This is a structured cohort-composition effect,
> not evidence that raw history has solved Novel localization.

There are two simultaneous statistical facts:

1. Phase-2's closed-loop raw-fixed contrast remains underpowered:
   `9/19` versus native `4/19`, paired `+6/-1`, exact McNemar `p=0.125`.
   Sampling variation therefore remains a valid explanation for the observed
   SR difference.
2. The six Phase-2 raw gains are not arbitrary CUDA accidents. Their first
   raw bearings were all close to the actual goal direction: exact direct
   bearing errors were `2.82--24.08 deg`, with mean `11.56 deg`. The single
   raw loss had `110.74 deg` error.

Thus it is wrong both to say "Phase-2 proves deployable Novel direction" and
to dismiss the result as "pure random noise." It is a real alignment event in
a small, directionally imbalanced cohort; what supplied that alignment is not
yet causally identified.

## 2. Protocol-parity audit

The two runs used the same frozen runtime stack.

| Item | Audit result |
|---|---|
| Episode runner | Same `run_paper_role_pair_episode.sh` hash |
| Habitat evaluator | Same `eval_shared_online_role_pairs.py` and `eval_2leg_habitat.py` hashes |
| Bearing adapter | Same `revisit_bearing_adapter.py` hash |
| MemNav runtime | Same `memnav_server.py` and `policy_agent.py` hashes |
| NavDP runtime | Same `navdp_server.py` hash |
| Policy contract | Same `multigoal_policy_contract.py` hash |
| Dependency receipt | `4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e` |
| MemNav checkpoint | `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7` |
| NavDP checkpoint | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` |
| LingBot weights | `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409` |
| Closed-loop contract | 600 steps, execution horizon 8, success radius 1 m, hidden role label |
| Verification | Both official summaries independently recomputed with `verified=true` |

The only Slurm-script change generalized a fixed `0--63` array to a frozen
`MAX_POPULATION_PER_PROTOCOL`; it did not change episode behavior.

The observed flow thresholds also do not indicate drift. They are selected by
the same history-length rule:

- Attempt 7: threshold 20 on `1/9`, threshold 25 on `8/9`;
- Phase-2: threshold 20 on `2/19`, threshold 25 on `17/19`.

## 3. Correct population accounting

| Cohort | Histories | Unique scene clusters | Native Novel | Raw-fixed Novel | Paired raw vs native |
|---|---:|---:|---:|---:|---:|
| Attempt 7 | 9 | 9 | `2/9` | `1/9` | `+1/-2`, `p=1.0` |
| Phase-2 | 19 | 12 | `4/19` | `9/19` | `+6/-1`, `p=0.125` |

Earlier prose calling Phase-2 "19 scenes" was incorrect: it contains 19
histories but only 12 unique scene clusters. The paired episode counts remain
unchanged, but scene-level generalization must use 12 as the cluster count.

Attempt 7 and Phase-2 must remain separate populations. Their outcomes may
not be pooled to manufacture significance.

## 4. Directional mechanism audit

The evaluator logs ground-truth Euclidean goal distance from many known
rollout poses. The audit reconstructs each hidden goal's `(x,z)` by
multilateration, then checks it against every logged distance. Maximum
reconstruction residual was `1.51e-14 m` in Attempt 7 and `1.20e-14 m` in
Phase-2, so the direct-bearing calculation is numerically exact for the logged
rollout.

| Diagnostic | Attempt 7 | Phase-2 |
|---|---:|---:|
| Raw first-bearing circular mean | `166.1 deg` | `176.5 deg` |
| Raw first-bearing resultant `R` | `0.932` | `0.840` |
| Correct shortest-path direction behind agent | `7/9` | `16/19` |
| Correct-direction resultant `R` | `0.306` | `0.534` |
| Raw bearing within `30 deg` of route | `2/9` | `10/19` |
| Raw success when aligned | `0/2` | `8/10` |
| Raw success when unaligned | `1/7` | `1/9` |

Angles near `+/-180 deg` mean directly behind the agent. The high raw
resultants therefore reveal a backward mode, not a broadly calibrated
360-degree compass. Phase-2 simultaneously contains a stronger backward
target distribution. This is the central cohort-shift explanation.

For Phase-2, the post-hoc aligned-versus-unaligned success table gives Fisher
exact `p=0.00548`. This supports bearing alignment as the mechanism of the raw
successes, but it is not a preregistered method comparison and must not be
reported as confirmation.

The six raw-only gains are especially diagnostic:

- exact direct-goal error: `2.82--24.08 deg`, mean `11.56 deg`;
- locally recomputed shortest-path first-segment error: `2.14--19.54 deg`,
  median `5.12 deg`;
- the sole raw-only loss: direct error `110.74 deg`, locally recomputed route
  error `122.10 deg`.

The local MP3D navmesh reconstruction is secondary evidence because its
geodesic distance differs from the stored evaluator value by up to `0.127 m`
in Attempt 7 and `0.191 m` in Phase-2. The exact reconstructed direct-bearing
result does not have that version boundary, and both analyses give the same
mechanistic conclusion.

## 5. What the static derangement says—and does not say

A fixed-seed, 100,000-resample diagnostic swapped first-step raw bearings
across completed query identities.

| Cohort | Factual mean route error | Shuffled median | Empirical probability shuffled is at least as good |
|---|---:|---:|---:|
| Attempt 7 | `61.15 deg` | `70.88 deg` | `0.0973` |
| Phase-2 | `38.47 deg` | `58.67 deg` | `0.00079` |

For Phase-2, factual bearings also put `10/19` within `30 deg`, versus a
shuffled median of `6/19`; the corresponding empirical tail probability was
`0.01635`. A constant `176.5 deg` backward bearing reached only `7/19` within
`30 deg`, below the factual `10/19`.

This rules out a *constant U-turn alone* as a complete explanation of the
Phase-2 bearings. It does **not** prove history-specific memory information:
swapping a completed bearing across query identities changes the combined
goal/current-view/history association. The signal may come from the goal
image, current observation, history retrieval, or their interaction.

## 6. Additional construction coupling

The natural-direction builder uses an initial-bearing tolerance of `180 deg`;
unlike the support-controlled protocol's `30 deg`, it does not balance or
match initial route direction. It matches distance and covisibility, not
front/lateral/back bearing strata.

It also renders a Novel goal with
`yaw_facing((position - endpoint)[[0, 2]])`, coupling goal-view orientation to
the endpoint-to-goal direction. The runtime never receives yaw metadata, so
this is not direct label leakage. Nevertheless, the rendered goal RGB may
contain direction-correlated visual structure. This is a construction
confound that a future Novel-direction experiment must remove rather than
explain away after observing SR.

## 7. Raw direct versus raw fixed sensitivity

Across all 28 Novel queries, raw direct and raw fixed produced an exactly
identical first proposal:

- same decision step;
- same anchor;
- same raw score;
- same bearing, maximum discrepancy `0 deg`.

Nevertheless, their full rollouts disagreed on success:

- Attempt 7: `+1/-1`, `p=1.0`;
- Phase-2: `+3/-3`, `p=1.0`.

The only intended difference is how the direction is executed/scaled. This
shows that a plausible first bearing is not itself a stable closed-loop method
result; NavDP trajectories are sensitive to the controller interface and
subsequent replanning. It is another reason not to infer a deployable Novel
module from `9/19`.

Neither raw DINO score nor retrieval margin cleanly separates gains from
failures. Phase-2 failures include raw scores as high as `0.946`, overlapping
the gains. A score threshold is therefore not justified by this audit.

## 8. Decision and completed next experiment

The previously written 600-step four-arm Novel causal-control protocol remains
an immutable record, but it is paused. Running it now would spend substantial
HPC time before identifying whether the first proposal contains
history-specific information.

The next experiment was a cheap, proposal-only attribution test on the
already consumed queries:

1. Hold current RGB, goal RGB, start pose, candidate count, and model weights
   fixed.
2. Compare the factual raw-DINO anchor with 12 identity-seeded uniformly
   sampled legal anchors while preserving the full factual trajectory,
   current image, goal image and LingBot cache.
3. Evaluate only first-proposal angular error to the frozen shortest-path
   bearing, circular concentration, and `<=30 deg` coverage. Do not run a
   600-step controller rollout and do not report SR.
4. Treat the 19 consumed Phase-2 queries as mechanism development only.

Promotion rule:

- Continue only if the scene-cluster 95% CI lower bound for factual-anchor
  advantage is greater than zero.
- Otherwise archive raw Novel behavior as a goal/controller prior and return
  all main evaluation budget to Revisit/CEC.

The experiment completed on 2026-08-16.  The local replay reproduced the HPC
factual bearing within `1.079 deg` on average.  Against the shortest-path first
segment, factual DINO anchors had `38.630 deg` mean error versus `42.778 deg`
for the random-anchor expectation: a `+4.148 deg` advantage with
scene-cluster 95% CI `[-1.357,+8.898] deg`.  Useful `<=30 deg` coverage was
`10/19` versus `9.75/19` expected.  Against the exact direct goal, advantage
was only `+2.169 deg`, CI `[-3.710,+7.069] deg`, and both arms had expected
coverage `10/19`.

The promotion gate therefore failed.  Goal-image shuffle and the expensive
closed-loop control are stopped, not merely postponed.  The untouched final14
remains reserved for CEC's role-free Revisit utility and Novel-interference
confirmation.  Full protocol, result, implementation and hashes are in
`NOVEL_RAW_FORCED_ANCHOR_ATTRIBUTION_RESULT_20260816.md`.

Any future independent Novel study must be frozen before outcomes, stratify
front/lateral/back initial route direction, sample goal yaw independently of
the endpoint-to-goal bearing, use scene-clustered uncertainty, and retain
native/factual/deranged/random-bearing arms. Until then, the paper claim stays:

> CEC safely abstains on unsupported Novel queries; raw-DINO's Phase-2 Novel
> gains are an informative but unconfirmed directional cohort effect.
