# Certified relocalization closed-loop run record

Date: 2026-08-12 (Asia/Shanghai)

Current state: complete. All 20 scenes / 160 episodes finished; the frozen
report and an independent raw-CSV recount agree.

## Frozen method

- Runtime schema: v3.
- Geometry certificate: unchanged v2 (`16` PnP inliers, `5%` query and
  reference hull coverage, `2 px` maximum reprojection RMSE).
- Output: scale-free `[forward,left]` bearing in LingBot raw units.
- Controller adapter: `verified_bearing_v1`, fixed `2.5 m` radius.
- Rejection/exception: native ImageGoal NavDP on the same planning step.
- Online metric scale: prohibited and reported as uncertified.

The reason for the metric-to-bearing correction and all frozen statistical
rules are in `CERTIFIED_RELOCALIZATION_CLOSED_LOOP_PROTOCOL_20260812.md`.

## Local evidence and engineering audit

1. v2 accepted-set re-audit: `8/8` accepted samples have bearing error below
   `4.45°`, median `2.35°`.
2. Exact real-episode localization smoke:
   - selected anchor `26` (DINO rank 2);
   - PnP inliers `217`;
   - query/reference hull coverage `0.2854 / 0.2610`;
   - reprojection RMSE `1.0698 px`;
   - emitted raw bearing vector `[-1.4467, 0.1448]`;
   - fixed controller token `[-2.4876, 0.2490] m`;
   - predicted/GT bearings `174.28° / 174.61°`;
   - first call `2.09 s`, cached call `0.15 ms`.
3. Moving one frame preserved the frozen causal shortlist and updated the
   current-relative vector from the cached absolute goal pose.
4. Reset removed goal/cache state; a query without a new causal probe returned
   `goal_not_probed_causally`.
5. Accepted end-to-end path invoked `navdp_image_point_mix`, returned 16 NavDP
   candidates, and executed one eight-step horizon.
6. Novel-A with no causal history abstained, and the full evaluator invoked
   `navdp_image_router`, returned 16 native candidates, and executed normally.
   Final lifecycle hardening records this case as one cached
   `no_causal_candidate` decision, preventing repeated work or later
   goal-session self-matches.
7. Local policy servers were shut down after smoke; RTX 4090 returned to
   `254 MiB / 0%` idle state.

## Formal evaluation

- Source data: immutable fresh160 manifest, 20 scenes × 8 episodes.
- Manifest SHA256:
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`.
- Four arms: native, old geometry router, known-Revisit direct upper reference,
  certified relocalization.
- One Goal-A trace per episode, byte-identically replayed into all four arms.
- One MemNav and one NavDP process per scene task.
- Four-row Williams order repeated five times across scenes.
- Primary: certified minus native; paired exact McNemar plus 100,000-resample
  scene-cluster bootstrap.

## Reproducible submission preflight

Final local dry run:

- 63 submission tests passed, including the PnP, LingBot pose-loop,
  cached-empty lifecycle, and complete synthetic 20-scene summary tests;
- Python compile and all shell syntax checks passed;
- bundle files: `2,758`;
- bundle bytes: `310,142,085`;
- source receipt SHA256:
  `74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98`;
- bundle manifest SHA256:
  `d3bd281fc374cc809fa67368a45e0cc80d53adc3b054db6b23f386001e7ff12e`;
- intended immutable remote source:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/`
  `certified_relocalization_closed_loop_d3bd281fc374cc80`.

The submitter will independently hash the three frozen checkpoints once on the
login node, make the source and run receipts read-only, run a `--test-only`
Slurm validation, submit array `0-19%4`, and attach the CPU summary with
`afterok`. No outcome exists yet.

## Formal submission

- run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/`
  `certified_relocalization_closed_loop_20260812/`
  `certrel_bearing_v1_20260812T1050`;
- evaluation array: `15641052` (`0-19%4`);
- frozen summary: `15641067`, dependency `afterok:15641052`;
- dependency receipt SHA256:
  `4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e`;
- submitted source receipt matches the final local receipt:
  `74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98`.

At submission, `scontrol ping` reported the primary controller `UP`; both GPU
and CPU sbatch files passed `--test-only`. The first scheduler observation had
all evaluation tasks pending under the shared `gpu48` QoS GRES quota, with no
task yet started and therefore no outcome read. This is queue state, not an
evaluation failure.

## Final result

All 20 scenes completed. The four arms shared exactly the same Novel-A trace,
with `120/160 = 75.0%` A success. Revisit-B is therefore evaluated on the same
120 paired A-success episodes:

| arm | Revisit B given A | joint A and B |
|---|---:|---:|
| certified relocalization bearing residual | **112/120 = 93.33%** | **112/160 = 70.00%** |
| known-Revisit raw-DINO direct | 106/120 = 88.33% | 106/160 = 66.25% |
| old geometry router | 91/120 = 75.83% | 91/160 = 56.88% |
| native ImageGoal | 27/120 = 22.50% | 27/160 = 16.88% |

Paired contrasts:

- certified minus native: `+86/-1`, conditional `+70.83 pp`, exact McNemar
  `p=1.14e-24`, scene-cluster 95% CI `[+59.32,+81.68] pp`;
- certified minus geometry: `+23/-2`, conditional `+17.50 pp`,
  `p=1.94e-5`, cluster CI `[+8.87,+27.64] pp`;
- certified minus raw direct: `+9/-3`, conditional `+5.00 pp`, `p=0.146`,
  cluster CI `[-1.74,+12.60] pp`.

Thus the certificate residual has clear closed-loop value and significantly
beats native and the old geometry router. It is numerically better than the
simpler known-role direct reference, but that difference is not statistically
resolved; the honest claim is **matching the strongest known-Revisit baseline
while providing a fail-closed, scale-free deployment interface**, not a proven
SR win over direct.

Runtime receipts: 115/120 A-success episodes had at least one certified
takeover, five fell back to native, 1,544 plans were accepted, and runtime
failures were zero. The certificate executed once per goal; median uncached
latency was `5.01 s` (p95 `26.83 s`), so latency remains an engineering issue.

Report:

- SHA256:
  `0e41a6d9b339d143229ba405b04802654d2053b5d641a03ed2d09aefc1a589f4`;
- audit: 20 scenes, 160 episodes, balanced Williams order, shared Goal-A
  trace, training overlap empty, no development/blind read, no online metric
  scale.

## Analysis-only repair record

Original array tasks 4--7 were cancelled by Slurm uid 0 before payload and
were successfully replaced by repair array `15642562`; all four replacements
completed with exit 0. Replacement summary `15642571` then failed before
reading results because the Habitat Python environment did not provide
`pytest`. No report existed at that point.

The analysis-only replacement `15645446` changed only the interpreter used for
the frozen pre-summary unit tests (MemNav Python for pytest); rollout code,
manifest, frozen summarizer, statistical rules, and output path were unchanged.
It passed 8/8 tests and completed. Repair script SHA256 is
`9a35a1f9cbdbfea78e4b9975771565c88c51621a175da841a508b3643e6e9af9`.
An inadvertently duplicated submission `15645456` encountered the existing
read-only report and failed closed before analysis; it did not overwrite data.

A separate raw-CSV reader that imports none of the project summarizer code
recounted `112/106/91/27`, the paired `+9/-3`, `+23/-2`, and `+86/-1`, and the
same exact McNemar values.

## 2026-08-13 actual-online-A observability supplement

Goal B was generator-labelled against expert A, so the certificate run's own
shared online-A traces were independently re-rendered and audited.  All
`120/120` A-success episodes have actual online maximum co-visibility
`>=0.20`; `115/120` have strong support `>=0.50`.  All 11 episodes below
`0.20` are A failures and never enter the conditional-B denominator.  Thus the
reported `112/120` certificate SR is not inflated by an expert/online memory
label mismatch.

In the strong-support subset, certificate remains `108/115`, direct `101/115`,
geometry `87/115`, and native `24/115`; certificate versus direct is `+9/-2`,
`p=0.06543`, so the claim remains “matches direct with a safer deployment
interface,” not a proven direct-baseline win.  Independent RTX 4090 and RTX
5090 runs reproduced all 160 rows and 34,437 trace-frame measurements exactly.
The original-path HPC audit's first job `15657882` failed before rendering on a
manifest-sidecar working-directory check; path-only replacement `15677956`
completed in 6m42s and exactly reproduced all substantive fields.  It is
tracked in
`CERTIFIED_RELOCALIZATION_ONLINE_OBSERVABILITY_RESULT_20260813.md`.
