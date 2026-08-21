# Certified stagnation graph rescue: frozen fresh20 expansion

## Authorization and question

The six-episode post-hoc mechanism pilot passed its frozen gate: all three
certified-direct Goal-B failures were reproduced, budget-control recovered
`0/3`, graph rescue recovered `3/3`, and the three successful controls had zero
interventions and zero losses. Every paired causal prefix passed. The pilot is
mechanism evidence only because its failures were selected after observing the
baseline outcomes.

This expansion asks the remaining safety/total-effect question on the same
internal fresh20 benchmark:

> When the already-frozen stagnation-triggered graph rescue is applied to every
> episode, does it preserve all unselected direct successes, and what is the
> paired full-20 effect after accounting for an equal-budget control?

The benchmark is heavily used development data. Even a positive result is an
internal total-effect estimate, not scene-disjoint paper confirmation.

## Frozen population and arms

The authorizing pilot already contains indices `0,1,2,3,7,14`. The expansion
runs exactly the previously unselected indices:

`4,5,6,8,9,10,11,12,13,15,16,17,18,19`.

Each new episode runs the unchanged three arms on one persistent server/GPU
pair with balanced order:

- `direct`: original certified scale-free bearing and original stuck stop;
- `budget_control`: one identical stuck-history reset, direct bearing retained;
- `rescue`: the same reset plus the causal recorded-history graph direction.

All controller, certificate, route spacing (`1.25 m`), arrival threshold
(`0.60 m`), 600-step budget, online-A replay, deterministic diffusion seeds,
and actual-online memory contracts remain frozen. No development/blind data,
Habitat pose, geodesic, or outcome is available to the policy.

## Audit and interpretation

The final report re-audits all 20 raw episode directories uniformly. It must:

1. reproduce the authorizing pilot report byte-for-byte at the record level;
2. reproduce direct failures at exactly indices `2,7,14`;
3. verify exact causal plan, physical-rollout, and memory prefixes through each
   treatment branch;
4. report rescue vs budget-control and rescue vs direct paired gains/losses,
   exact McNemar p, and scene-cluster bootstrap intervals;
5. list every treatment intervention and require every gain to contain an
   actually active graph plan.

The internal result freezes only if there are no rescue losses, no gain without
an active graph, the direct classification reproduces, and all 14 previously
unselected direct-success episodes have zero treatment interventions. It is not
expected to be statistically significant with only three possible baseline failures:
even a perfect `+3/-0` contrast has exact McNemar `p=0.25`. Its purpose is to
establish safety and full-set mechanics before any scene-disjoint replication,
not to manufacture a significance claim.

## Next decision

- Clean full20 behavior authorizes one scene-disjoint confirmation with the
  architecture and all thresholds frozen.
- Any new loss, unexplained intervention, baseline mismatch, or causal-prefix
  failure stops expansion and triggers episode-level audit; it does not
  authorize tuning on this benchmark.

## Submission receipt

- Authorizing pilot report SHA256:
  `aebb64d9c5c6819a55b112a412603cd24a4fc1739c3f103a0600ed595aca8434`.
- Immutable expansion source bundle:
  `certified_stagnation_graph_89e0cab3fcf2d697`.
- Source receipt SHA256:
  `f751a6137e43c90be0e4e2b1cb3e1d66a001b63da6c00fa09f5e340a275f89fe`.
- Evaluation array: `15675574`; CPU summary: `15675586`.
- Submission receipt SHA256:
  `0590c270dc3007bc9913258c9608f8d73be78907403b24d4c4045957b8a21f5e`.

Before any expansion outcome was read, a code review tightened the safety gate
to require zero interventions—not merely zero losses—on all 14 unselected
direct-success episodes. The original pending summary `15675586` was cancelled
without starting. Replacement summary `15675715` is bound to immutable bundle
`certified_stagnation_graph_230103bc6268bc3f`, source receipt SHA256
`8d0ecfa958d97ff7ceae317df74bc0243c8d53d9662e38ff4d87180a40056195`.
The repair receipt SHA256 is
`8ac7861d062bcfda165fa2177f2607a5033610d80ecc0504d5dccee1cac2125e`.

## Frozen full20 result

The import-bootstrap replacement summary `15676260` completed. The official
report is `cgraph_pilot_v2_budget_20260813/full20_report.json`, SHA256
`981975dbc8a4eee93ffc1e30bef6d576e230c754441bbc8b03541edb86e57b0c`.
An independent script reread all 60 raw arm CSVs and reproduced every count,
paired contrast, exact McNemar value, and 100,000-resample scene-cluster
interval; verification SHA256 is
`9fc82224e0e0c85e863fc535291970a86437d9f57e3dbb9c1928a457f63f1fa8`.

| arm | B | C | joint | graph-active B plans |
|---|---:|---:|---:|---:|
| direct | 17/20 | 17/20 | 17/20 | 0 |
| budget-control | 17/20 | 17/20 | 17/20 | 0 |
| graph rescue | **20/20** | **20/20** | **20/20** | 90 |

Primary graph-rescue minus equal-budget-control joint contrast:

- `+3/-0`, risk difference `+15.0 pp`;
- exact McNemar `p=0.25`;
- 13-scene cluster bootstrap 95% CI `[0,+30] pp`;
- gain indices are exactly `2,7,14`; loss indices are empty;
- all 14 previously unselected direct-success episodes had zero treatment
  interventions, so their three arms remained exact no-ops through both B and
  C;
- every gain contained active graph execution; no gain was credited to reset
  or extra budget alone.

The strict decision is `freeze_internal_fresh20_result`. Scientifically, this
supports a narrower diagnosis than “bearing is sufficient”: the certificate's
direct endpoint bearing is sufficient on 17/20, while the remaining 3/20 need
the homotopy/topology encoded by the already traversed history arc. The result
is deterministic and internally complete, but it is not statistically
significant because there are only three discordant pairs and this benchmark
was used to select the failure mechanism. Architecture and thresholds must now
stay frozen for a scene-disjoint confirmation; no tuning on full20 is allowed.

Execution note: strict summary `15675715` failed at module import before
reading outcomes because direct script execution omitted the immutable bundle
root from `sys.path`. The summarizer now bootstraps only its own physical
bundle parent. The completed 14-task array was independently verified from
`sacct` before dependency-free replacement submission; its repair receipt is
`expansion_summary_import_repair_submission.json`, SHA256
`f9dcd1305bb55c2a7cb502b7e3290ef8f52711873f595237d9381014773d085b`.

## Scene-disjoint inventory and stop line

A metadata-only inventory was completed before any confirmation submission.
It inspected directory names/counts and frozen scene-role lists only; it did
not read episode targets, policy outcomes, or any blind evaluation result.
The receipt is `THREE_LEG_SCENE_ROLE_RECEIPT_20260813.json`.

The nominal 526-episode 3-leg pool contains only 36 non-empty scene clusters:

- 19 intersect the already consumed 20-scene pool;
- 1 intersects train40;
- none intersect development10 or final-reserved4;
- all remaining 16 are precisely the frozen blind16;
- zero remaining clusters lie outside blind.

Therefore there is no scene-disjoint, non-held-out confirmation set in this
pool. The correct decision is `stop_before_blind_confirmation`: keep the
full20 graph-rescue architecture and thresholds frozen, do not submit another
development run, and require explicit one-shot authorization before opening
blind. The pool's scene/count digest is
`960366e83c7fb1946e2cb7b0c025edae32c5902e07de140ccc427472cf98227d`;
the byte-reproduced receipt SHA256 is
`0a255ca09d601cc6b8a49a5b1ca9c77ef760c8a44fbfd2b9894ee21c5844dc5d`.
