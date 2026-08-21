# MRC Existing Evidence and Local Falsification Pilot

Date: 2026-08-12

## Bottom line

MRC is no longer supported only by intuition: archived true three-view artifacts
contain cross-scene anchor-verification signal, and a new local deployment-top-1
pilot shows especially strong signal in pose dispersion. It is **not yet a
reliable Novel/Revisit classifier**. The same local pilot also exposes a high
scoring ambiguous case, failure of cloud overlap on a true positive, and poor
transfer when DINO is naively added to geometry.

No threshold or method claim is authorized by this note.

Post-audit update: the full 480-session HPC collection is now **on hold**.
Further variance, dependence, proposal-ceiling, and literature analysis showed
that the current absolute top-1 certificate conflates candidate verification
with unknown-goal existence.  See
`MRC_SIGNAL_ATTRIBUTION_AND_LITERATURE_20260812.md` for the superseding
decision and local falsification plan.

## 1. Archived artifact that the earlier summary undercounted

The archived HPC run
`multiscene100_20260806_job_15400645` contains:

- 93 rows, 25 sessions, 22 scenes;
- 90 rows with exactly three hypotheses at offsets `[-4,0,+4]`;
- 46 positive and 44 negative candidates after requiring all three views;
- full causal replay;
- only `revisit_b` sessions, sampled to contain both positive and negative
  anchors.

Therefore this artifact measures **correct versus wrong anchor inside a known
Revisit session**. Its sampler excludes strict no-match sessions and cannot
answer Novel versus Revisit existence.

On the 90 exact-three-view rows, primitive candidate AUCs are:

| signal | AUC | scenes with expected mean direction |
|---|---:|---:|
| DINO cosine | 0.596 | 18/22 |
| cloud-overlap F1 | 0.737 | 18/22 |
| translation pose consistency | 0.657 | 15/22 |
| rotation pose consistency | 0.571 | 17/22 |
| translation refinement | 0.684 | 18/22 |
| rotation refinement | 0.585 | 11/22 |

A fixed `C=1` logistic probe evaluated by leave-one-scene-out gives:

| input | OOF AUC | OOF AP | balanced-set top-1 |
|---|---:|---:|---:|
| DINO only | 0.568 | 0.596 | 18/25 |
| geometry only | 0.694 | 0.760 | 21/25 |
| DINO + geometry | 0.735 | 0.770 | 21/25 |

Scene-cluster bootstrap 95% intervals for OOF AUC improvement over DINO are
`[+0.020,+0.228]` for geometry and `[+0.032,+0.272]` for DINO+geometry. This is
evidence that the multi-view signal is not wholly driven by one scene. It is
still candidate verification, not existence detection.

## 2. New local deployment-top-1 pilot

The local RTX 4090 has the exact LingBot checkpoint and complete raw/cache data
for only two old train scenes. A label-blind availability preflight found 19
legal fixed-three-view top-1 sessions. To obtain the most informative quick
falsification set, the broad run was stopped after three session-atomic rows and
an eight-session targeted run prioritized strict no-match and wrong-anchor
cases. The two outputs combine to 11 unique rows:

- 3 correct anchors;
- 6 strict no-match top-1 anchors;
- 2 ignore-band/wrong-anchor rows;
- two scenes only;
- legacy runtime scale rather than the formal external causal scale.

On the nine signed rows, without fitting or selecting a threshold locally:

| signal | candidate AUC | positive median | negative median |
|---|---:|---:|---:|
| DINO cosine | 0.722 | 0.93785 | 0.92046 |
| cloud-overlap F1 | 0.806 | 0.24002 | 0.00000 |
| translation pose dispersion | **0.944** | 0.04024 | 0.19487 |
| rotation pose dispersion | 0.833 | 0.693° | 24.297° |
| translation refinement | 0.389 | 0.01464 | 0.01222 |
| rotation refinement | 0.528 | 0.05595° | 0.11851° |

This small test directly rejects two overly intuitive rules:

1. A true positive in the second scene has cloud overlap `0.0`; overlap alone
   would reject it.
2. Refinement magnitude does not separate the classes in this deployment-style
   subset.

Translation pose consensus is the promising signal here, but nine signed rows
cannot establish its false-positive tail.

## 3. Fixed cross-artifact transfer probe

To avoid training on the local labels, the fixed logistic probe was trained on
the archived exact-three-view rows after excluding every occurrence of the
local scenes. This leaves 86 rows from 21 scenes. Applied unchanged to the local
deployment-top-1 rows:

| input | local candidate AUC |
|---|---:|
| geometry only | 0.778 |
| DINO + geometry | 0.556 |

The geometry-only scores for the three correct anchors are
`[0.661, 0.760, 0.233]`; strict no-match scores are
`[0.001, 0.290, 0.259, 0.086, 0.472, 0.414]`. One ignore-band case scores
`0.851`. Thus there is no clean universal threshold in this pilot, and adding
DINO can amplify domain shift rather than stabilize it.

## 4. Runtime receipt

The targeted eight-session run took `601.63 s` on the local RTX 4090 with full
offline replay:

- `75.2 s/session` average;
- peak CUDA allocated: `15.43 GiB`;
- peak CUDA reserved: `30.98 GiB`.

This proves the local machine can execute the extractor. The offline replay
cost is not an acceptable deployment implementation; an online version would
need to reuse the already accumulated KV/map state.

## 5. Superseded decision

This note originally concluded that the evidence justified completing the
frozen MRC train-only experiment.  That transition is superseded by the
post-smoke attribution audit: full HPC is held until short local controls first
separate scene-relative verification, geometric-baseline quality, pose error,
and action utility.  No thresholds are tuned on this pilot.
