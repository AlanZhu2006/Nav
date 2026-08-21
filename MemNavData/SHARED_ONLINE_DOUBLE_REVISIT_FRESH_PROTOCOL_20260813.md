# Shared-online double-Revisit fresh160 internal gate — 2026-08-13

> 当前修订：预注册 fresh40 构造门未满足；正式降级为 **fresh20、13 scenes 的
> feasibility-limited internal gate**。它可以检验效应方向与 3-leg 因果链，不能冒充
> 原定 fresh40 的统计确认。

## Question

After a genuinely online, frozen native NavDP Goal-A rollout, can the system
retain and independently use **two** old visual memories?  The causal target is
Goal C: does access to long-term A memory improve C when Goal B and its complete
physical rollout are byte-identical?

## Population and causal seal

- Dataset: immutable fresh160 manifest SHA
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`.
- Online-A trace source: the later audited certified-relocalization run
  `certrel_bearing_v1_20260812T1050`, which has 120 native Goal-A successes.
  The older fresh-confirmation run has 118 and is not mixed into this task.
- All 120 native-controlled Goal-A successes are attempted; there is no
  score-based preselection.
- The frozen V2 constructor requires two separated online-A anchors, at least
  2 m geodesic legs, controlled V1 pose perturbations, and a route-negative
  proxy for C.
- Before any navigation arm runs, a deterministic scene-round-robin takes
  exactly 40 constructible episodes spanning at least 15 scenes.  The manifest
  is hashed and made read-only.  If fewer qualify, evaluation does not start.
- fresh160 has already informed project decisions.  Therefore this is a
  statistically powered **internal fresh-target gate**, not paper-final
  untouched-scene confirmation.

## Four paired arms

1. `full_memory`: known Revisit control on B and C; C may retrieve only up to
   the frozen online-A boundary.
2. `memory_b_native_c`: identical memory-controlled B, but native NavDP on C.
3. `certified`: deployable LightGlue/PnP-certified bearing adapter on B and C.
4. `native`: frozen native NavDP on B and C.

All four replay the same online-A RGB bytes without diffusion sampling.  NavDP
short FIFO is reset immediately before C.  The factual B trajectory is checked
against Goal C; contaminated C inputs are censored rather than counted as
success or failure.

## Primary endpoint

`full_memory C − memory_b_native_c C`, restricted to episodes where the shared
B succeeds and the factual B rollout passes the C hard-negative check.  The
audit requires exact equality of B plans, rollout traces and memory traces.
Report N, gains/losses, exact McNemar p, risk difference, and scene-clustered
bootstrap 95% CI.  Whole-pipeline joint contrasts against native are secondary.

## Submission chain

`prepare (GPU) -> evaluation[0:39] (GPU, max 4 concurrent) -> independent audit (CPU)`.
Every stage verifies immutable source receipts; downstream jobs use `afterok`
dependencies and fail closed.

## Formal submission receipt

- Run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z`
- Immutable task bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/shared_online_double_revisit_fresh_e222ff0822ce45e7`
- Task source receipt SHA256:
  `d14e186bb1d5be2a5742a02070086f669fe4054fd63da4fe6b0c755c04e137a6`
- Preparation job: `15663165` (H100, running at submission audit).
- Evaluation array: `15663213_[0-39%4]` (`afterok:15663165`).
- Independent summary: `15663266` (`afterok:15663213`).

Operational note: a diagnostic re-entry briefly created prep job `15663263`
and eval array `15663304`; both were cancelled while pending with zero elapsed
GPU time.  They are not part of the formal run, whose unique machine-readable
receipt is `submission.json` under the run root above.

## Fresh40 feasibility failure and frozen fresh20 amendment

Preparation job `15663165` exhausted all 120 native-A-success candidates. Only
20 episodes across 13 scenes met every preregistered constructibility and causal
condition, so it correctly refused to seal 40. No navigation arm from the
fresh40 chain ran. The deficit is a property of this strict finite source pool;
loosening geometry after seeing the count would change the benchmark.

The deterministic full eligible population was therefore sealed as a lower-power
fresh20 amendment:

- finalizer job `15668929`: `COMPLETED`;
- episodes/scenes: `20 / 13`;
- benchmark manifest SHA256:
  `7b33777289daa5ec10b29e6e275d970ea959b2fe2ef35b8ced947ba4f6904078`;
- `formal_power_target_met: false` is embedded in the receipt;
- the four arms and primary causal contrast remain unchanged.

## Evaluation runtime audit

Two startup-only attempts produced no SR observation:

1. `15668938`: the task overlay did not expose the nested `depth_anything`
   package; zero scene outcome files;
2. `15669808`: after fixing that path, Habitat ran outside the pinned container
   and resolved host `libstdc++`, causing `GLIBCXX_3.4.30 not found` while
   importing `llvmlite`; zero `metric.csv`, zero `summary.json`, zero plan files.

Attempt 2's 10 partially created scene directories, buffers and logs are preserved
with per-file SHA256 under:

`failed_attempts/retry1_eval_15669808/failure_receipt.json`.

The current retry executes the Habitat evaluator inside the same pinned CUDA
container used by the successful 2-leg evaluation and preflights
`quaternion`, `numba`, and `habitat_sim` before starting any arm:

- evaluation array: `15670094_[0-19%4]`;
- independent summary: `15670116`;
- immutable task bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/shared_online_double_revisit_fresh20_2cbfac9bd2ab10ef`;
- task source receipt SHA256:
  `4860ab4b42a10868d9d01433688fccff3465bd457a7bd657b1b4dbbddeafed59`.

The first live tasks passed both runtime imports and entered real navigation.
Any partial arm outcome observed before the 20-episode audit is only a progress
signal; the result is frozen only when `15670116` emits the paired,
scene-clustered report.

Operational progress note (not a result): indices 0--16 now have all four arms
and their per-episode causal equality checks complete.  Indices 15--19 had been
cancelled before start in the original array and were resubmitted unchanged at
concurrency one as `15672054_[15-19%1]`; 15--16 have since completed and 17--19
remain.  Replacement independent summary job `15672056` audits the full 20 only
after all five finish.  The exact scheduling amendment is recorded in
`submission_fresh20_resume_after_cdec.json`; no episode, arm, source bundle,
manifest, or method changed.

At the 17/20 checkpoint, the descriptive counts are: `full_memory` B `12/17`,
joint `10/17`; `memory_b_native_c` B `12/17`, joint `6/17`; `certified` B `14/17`,
joint `14/17`; and `native` B `4/17`, joint `0/17`.  For the primary paired
contrast, the currently eligible denominator is 12 and the discordance is
`+6/-2`.  These numbers are explicitly **not** a result: the frozen endpoint is
the independent 20/20 report, and neither inference nor stopping is allowed at
this checkpoint.

Operational update, 2026-08-12 20:17 EDT: index 17 completed normally, so
`18/20` episodes now have all four arm summaries (`72` immutable
`summary.json` files). Indices 18--19 remain pending for scheduler priority;
there is no runtime or method failure. No 18/20 outcome statistic was used or
reported, and the frozen endpoint remains job `15672056` over exactly 20/20.

The final audit does not treat raw joint SR as the primary causal answer. It
first verifies byte-identical B plans, B rollout traces, and B memory traces
between `full_memory` and `memory_b_native_c`; it also requires identical
pre-C fields. Only episodes where that shared B succeeds and the factual B
state passes the frozen C co-visibility/input contract enter the primary C
contrast. All other C outcomes are censored rather than silently counted as
failures. This is essential because fresh20 tests whether long-term A memory
helps a *second valid Revisit*, not whether one arm happened to arrive at an
incomparable B terminal state.

An independent local verifier is now available at
`MemNavData/independent_verify_shared_online_double_revisit.py`. It deliberately
does not import the production HPC auditor. After `report.json` is frozen, it
will re-read the sealed manifest, all four-arm `metric.csv` files, and the raw B
plan/rollout/memory receipts; independently rebuild the C-eligible denominator,
McNemar table and scene-cluster bootstrap; and require field-level agreement
with the HPC report. Its standalone statistical tests pass in the pinned
Habitat environment. No result is accepted until this second implementation
also passes.

## Frozen 20/20 result

All 80 arm runs completed normally. Summary job `15672056` froze the official
report, and the independent local implementation reproduced it from the raw
CSV/plan/rollout/memory receipts.

| arm | B success | C eligible | C success | joint |
|---|---:|---:|---:|---:|
| native | 5/20 | 3 | 0/3 | 0/20 |
| full memory | 15/20 | 14 | 12/14 | 12/20 |
| memory-B / native-C | 15/20 | 14 | 8/14 | 8/20 |
| certified | 17/20 | 17 | 17/17 | 17/20 |

The preregistered primary causal comparison is `full memory C` versus
`memory-B/native-C` only after byte-identical B prefixes and a valid C input:
`N=14`, `12/14` versus `8/14`, paired `+6/-2`, risk difference `+28.57 pp`,
exact McNemar `p=0.2890625`, and 100,000-resample scene-clustered 95% CI
`[-15.38,+64.29] pp` over 11 scene clusters. The point estimate is positive,
but the formal power target was not met and the interval includes zero. It is
therefore **evidence in the predicted direction, not confirmation** that a
second Revisit benefits from retained A memory.

Secondary whole-chain comparisons are:

- certified versus native: `17/20` versus `0/20`, paired `+17/-0`,
  `p=1.5259e-5`;
- full memory versus native: `12/20` versus `0/20`, paired `+12/-0`,
  `p=0.0004883`;
- certified versus full memory: `17/20` versus `12/20`, paired `+5/-0`,
  `p=0.0625`.

The last contrast is promising but is secondary and just above 0.05; it must
not be reported as a confirmed superiority claim. Immutable local receipts:

- official report SHA256:
  `ba239438fb96d656ed609ffed7c1b5f0666a48db3e03277e5858a051c79811f0`;
- independent verification SHA256:
  `9d9e4379faee57dab9d9d441c0096e7f4282043f4d9cd8327d5bca4f7569a936`.

## Certified lifecycle correction

`certificate.requests` in the ordinary report counts planning calls, not
independent visual localizations. The standalone lifecycle audit
`MemNavData/audit_certified_3leg_lifecycle.py` establishes the exact execution
semantics:

| leg | planning calls | uncached LightGlue/PnP certificates | cached reuse | accepted independent certificates | navigation success |
|---|---:|---:|---:|---:|---:|
| B | 346 | 20 | 326 | 20/20 | 17/20 |
| C | 201 | 17 | 184 | 17/17 | 17/17 |

Thus there are 37, not 547, independent localization events. Every episode-leg
has exactly one uncached call followed only by cache hits; the shortlist and
selected anchor remain fixed, while the current-relative bearing changes after
motion in all 37 legs. The three certified failures are all B controller
`stuck` terminations *after an accepted certificate*, at final distances
`1.920`, `2.979`, and `1.861 m`; they are not certificate rejections.

The geometry stage selected raw-DINO rank 1 in only `11/20` B localizations and
`7/17` C localizations. This is useful evidence that geometric reranking is
doing real work even in this high-support set. Median uncached latency is
`11.55 s` for B and `11.25 s` for C, so cache-once/update-bearing is essential
to the deployed runtime contract.

The lifecycle audit receipt SHA256 is
`ad16d03e41a394caa50b0563f89699c4eba00e5aebd8831bf432dcc956b5b4e0`.

The post-outcome Habitat diagnostic also compares the scale-free output to the
true current-to-goal chord; this ground truth is audit-only and is never sent
to the policy. Across B, median first-query angular error is `2.62 deg`, median
per-episode error is `3.10 deg`, and request-level p90 is `10.07 deg`. More
importantly, the three B failures still have episode-median errors only
`2.31`, `2.76`, and `0.67 deg` (maxima `8.15`, `4.04`, `3.27 deg`). All three
terminate `stuck` at `1.92`, `2.98`, and `1.86 m`. This rules out a large
bearing error as the proximate explanation for those failures; it does not by
itself prove which alternative controller will rescue them.

## Why the old reverse graph is not the right 3-leg graph

The old graph starts at the newest memory frame and can move only backward in
time. That is appropriate for the first Revisit after a Novel rollout, but not
for a sequence of Revisit goals. After B, the robot is localized near B's
historical anchor; C may lie before **or after** that anchor on the original A
trajectory. In the 17 certified B-to-C goal switches, 8 require increasing and
9 decreasing historical time.

A trace-only audit quantifies the structural penalty. Routing from the newest
post-B frame first reverses the whole B rollout and then reverses A toward C;
its recorded-path length is a median `2.689x` (mean `2.978x`) the direct
B-anchor-to-C-anchor segment. This offers a concrete explanation for why the
old always-on reverse graph could hurt strict conditional-C (`8/10 -> 6/10`):
it was solving the wrong start-node problem, not merely using an imperfect
spacing value. It is still a hypothesis rather than a new SR result.

`NavDP/baselines/memnav/reverse_memory_graph.py` now contains a tested
bidirectional `metric_nodes_between` primitive while the old
`reverse_metric_nodes` wrapper retains byte-compatible reverse-only semantics.
The production/default arm still does not enable this primitive. A separate,
explicit pilot arm implements the intended multi-goal contract: localize B and
C with proof-carrying certificates; bind the next graph start to the previous
completed goal's certified anchor; traverse either time direction; use a short
historical node only after observable direct-control stagnation; then return to
the certified final bearing. This avoids both the old route detour and the
known harm from always-on graph intervention.

## What this benchmark can and cannot establish

Only 20 of 120 native-A successes (`16.67%`) support the frozen double-Revisit
construction: 33 lack enough source anchors, 60 lack a route-negative B/C
pair, and 7 fail the joint V0/V1 support condition. V1 explicitly requires
co-visibility with online A (`source >=0.45`, best historical frame `>=0.50`).
Navigation outcomes were never observed during selection, so paired causal
validity is intact; nevertheless the estimand is the **constructible,
high-support positive-Revisit subset**.

Consequently this run is a memory-execution/compositionality test. It does not
measure Novel-versus-Revisit classification, certificate specificity, or
fail-closed behavior: all 37 independent positive queries passed the geometry
certificate. It also means that adding the learned-on-geometry-reject CDEC
adapter to this exact frozen set would be a structural no-op. No expensive
rerun of these 20 episodes is justified merely to exercise the new code.

## Frozen next 3-leg decisions

The current fresh20 endpoint remains closed for **certificate-coverage** claims.
The next learned-certificate 3-leg experiment is allowed only after the
same-process CDEC proposal audit passes its preregistered gate. If it passes,
build a **train-scene, actual-online hard-Revisit stress set**
without reading development/blind labels: retain the same two-Revisit causal
chain, but freeze a broader controlled-pose support band before navigation so
that some geometry-first proposals genuinely reject. Compare the existing
geometry certificate against `geometry first -> learned proposal on reject ->
same certificate`; preserve the identical-prefix C ablation and report both
independent localization rescue and closed-loop SR. This is a consumed-pool
integration gate, not paper confirmation. If the CDEC proposal audit fails,
do not manufacture a 3-leg benchmark to make it fire; keep fresh20 solely as
the positive compositionality result above.

There is an orthogonal **controller-execution** question that does not require
CDEC to fire: the three certified B failures already accepted geometrically
accurate bearings and then terminated on the frozen odometric stuck event. The
post-hoc six-episode mechanism pilot in
`CERTIFIED_STAGNATION_GRAPH_PILOT_20260813.md` therefore compares three arms on
the three known failures plus three manifest-order success controls: original
direct termination; one extra stuck budget with the direct bearing unchanged;
and the same extra budget with bidirectional graph rescue. It is not an SR
estimate. Its frozen gate requires direct classification reproduction, at
least `2/3` graph rescues, at most `1/3` budget-only rescues, graph-over-budget
gain with zero paired loss, zero control losses/interventions, and exact causal
prefixes. Only a passing gate permits a new run on the unselected fresh20
episodes.

Submitted pilot:

- navigation array: `15674406` (`0,1,2,3,7,14`, concurrency 3);
- dependent summary: `15674429`;
- run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/certified_stagnation_graph_pilot_20260813/cgraph_pilot_v2_budget_20260813`;
- immutable bundle receipt:
  `7031f6f7a9d0be7a4f508d342e652eda49c63c5081eec744b518e8f785d86979`.

The superseded two-arm array `15674155` was cancelled as soon as the
extra-budget confound was identified: only success control 0 completed,
control 1 was interrupted, and no known-failure task ran. Its summaries
`15674158`, `15674247`, and an accidental duplicate `15674254` were cancelled
before execution. The v1 files are retained as engineering receipts and are
excluded from the v2 report. The v2 audit excludes only wall-clock `*_ms`
diagnostics from exact plan equality while preserving every causal decision
field.
