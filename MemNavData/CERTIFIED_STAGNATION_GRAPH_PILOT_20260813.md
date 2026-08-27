# Certified stagnation graph rescue: actual-online 3-leg pilot

## Question

The frozen certified baseline succeeds on 17/20 strict fresh episodes. Its
three Goal-B failures all accept a certificate and end 1.86--2.98 m from the
goal. The audit-only first-query bearing errors are small, so this pilot asks a
narrow controller question: can the already traversed history arc rescue a
certified direct bearing only after the unchanged odometric stuck event?

This is not a selector, not a new localization threshold, and not a population
SR estimate. The selected failures are post-hoc and are used only as a cheap
mechanism gate.

## Frozen intervention

- Direct certified bearing remains the default.
- The evaluator's existing 150-step / 0.10 m odometric stuck event may activate
  one rescue burst instead of immediately terminating.
- Goal B follows the causal end-of-A to certified-anchor history segment.
- Goal C follows the explicitly recorded previous-B-anchor to current-C-anchor
  segment; time may increase or decrease.
- The server uses LingBot poses only. Habitat pose, geodesic and outcomes never
  enter the policy.
- Historical nodes are spaced at 1.25 m with a 0.60 m arrival radius. After the
  route is complete, control returns to the certified final bearing.
- Any invalid route, scale or changed contract falls back to the direct bearing.

## Paired pilot

Frozen manifest indices:

- post-hoc known certified-B failures: 2, 7, 14;
- manifest-order successful controls: 0, 1, 3.

Each episode runs three arms on one GPU and persistent server pair:

- `direct`: terminate at the original stuck event;
- `budget_control`: reset the same stuck history once, but keep the certified
  direct bearing;
- `rescue`: reset the same stuck history once and request the historical graph
  direction.

The budget arm isolates graph direction from the otherwise confounded extra
execution budget. Arm order is balanced and every arm uses deterministic
per-plan diffusion seeds. The audit requires exact online-A replay, exact
physical/memory prefixes, and exact causal plan fields through the
intervention boundary; wall-clock `*_ms` diagnostics are excluded from the
plan equality check.

An audit-only feasibility check on the frozen Habitat traces gives full
A-end-to-B-anchor history-arc lengths of `9.709`, `5.346`, and `8.267 m` for
failure indices 2, 7, and 14 (approximately 8, 5, and 7 nodes at 1.25 m).
Their original stuck terminations leave 393, 394, and 409 of the 600 control
steps, respectively. The runtime starts from the nearest point on each arc, so
these full lengths are upper bounds on the rescue residual. This makes the
probe feasible but does not predict success; Habitat geometry is never sent to
the controller.

## Frozen gate

Expansion to all unselected fresh20 episodes is allowed only if:

1. the paired direct rerun reproduces all three failure and three control
   classifications;
2. rescue recovers at least 2/3 known Goal-B failures;
3. budget-control alone recovers at most 1/3;
4. rescue has at least one gain and zero losses versus budget-control, and an
   actual graph plan is active for every rescued B;
5. there are zero joint losses and zero interventions on the three controls;
6. every causal prefix audit passes.

Anything else is `stop_or_repair_before_expansion`. Even a passing pilot is
mechanism evidence only; the unselected paired set is required for an SR claim.

## Superseded two-arm submission

The first submission (`15674155`, summary `15674247`) contained only direct and
rescue. Before any known-failure task ran, audit identified the extra-budget
confound above. The array and summary were cancelled. Only manifest control
index 0 completed; index 1 was interrupted, and indices 2/3/7/14 never ran.
Those outputs are retained as engineering receipts and are excluded from the
three-arm report.

## Three-arm execution incident and retry contract

The first three-arm array (`15674406`, immutable source receipt
`7031f6f7a9d0be7a4f508d342e652eda49c63c5081eec744b518e8f785d86979`)
completed controls 0, 1, and 3. It did not produce usable results for the three
known failures:

- indices 2 and 14 failed when the first rescue request lazily invoked
  LingBot's metric-scale estimator;
- index 7 completed its budget-control arm and was proactively cancelled
  before its rescue arm could traverse the same unsafe code path.

The incident is an implementation failure, not a navigation outcome.
`get_metric_scale` calls `clean_kv_cache`; when invoked late in an online
rollout this erased both the live aggregation cache and camera-head cache, so
the next `add_frame` received `camera_head.kv_cache=None`. The graph branch now
uses the same reference-snapshot/restore transaction already required around
other destructive planning operations. A regression test deliberately clears
both caches and checks restoration of tensor identity, frame counters, and the
camera-head stream.

All partial 2/7/14 outputs and Slurm logs are preserved under
`failed_attempts/pre_stream_restore_15674406`; archive receipt SHA256 is
`222683c6845fe8a32c62e6698c045ac3a001042c6834cbc0205865b34ad09986`.
They are excluded from the final report. The causal retry is restricted to
indices 2, 7, and 14 from a new immutable bundle. Completed controls are
reused, and the final summarizer runs only after all three clean retries
finish. No partial failure-arm outcome may be inspected or promoted.

## Frozen pilot result

The clean retry array was `15675046`; the idempotent replacement summary was
`15675466`. The official report is
`cgraph_pilot_v2_budget_20260813/report.json`, SHA256
`aebb64d9c5c6819a55b112a412603cd24a4fc1739c3f103a0600ed595aca8434`.
It is byte-identical to the independently invoked preview report.

- Direct reproduced all three known Goal-B failures and all three successful
  controls.
- Budget-control recovered `0/3` failures.
- Graph rescue recovered `3/3`, with `+3/-0` versus budget-control.
- The rescued B legs contained `44`, `16`, and `30` active graph plans.
- Budget-control ended at exactly the same B distances as direct (`1.920`,
  `2.979`, `1.861 m`), whereas rescue ended inside the success radius
  (`0.977`, `0.967`, `0.965 m`). Thus extra budget/reset alone did not even
  alter these terminal distances.
- All three routes traversed the recorded A history in reverse, with frozen
  resampled route lengths of `12`, `4`, and `8` nodes. The first graph request
  occurred exactly at the audited treatment branch (`steps 208`, `208`, and
  `200` after stuck events at `207`, `206`, and `197`).
- Every rescued B continued to a successful C, so these three pilot joints
  changed from `0/3` to `3/3`.
- Controls had zero interventions and zero joint losses across both treatment
  arms.
- All exact causal plan, physical-rollout, memory-prefix, online-A replay, arm,
  manifest, and controller-contract audits passed.

The frozen decision is `expand_to_unselected_fresh20`. This is a clean
mechanism result, but `3/3` is post-hoc failure-selected and must not be quoted
as a population success rate. The authorized full-set expansion is defined in
`CERTIFIED_STAGNATION_GRAPH_EXPANSION_PROTOCOL_20260813.md`.
