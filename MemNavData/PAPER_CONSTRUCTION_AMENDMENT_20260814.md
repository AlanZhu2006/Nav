# Paper role-pair construction amendment

Date: 2026-08-14 (Asia/Shanghai)

## Status and causal boundary

The fifth sealed MP3D pipeline attempt was stopped during native Goal-A
collection/construction.  No Novel or Revisit query arm had run and no query
SR, SPL, final distance, certificate decision or policy trajectory was read.
The exposed information was limited to native-A completion/history length and
whether the benchmark builder could construct a query pair.

This is therefore a pre-query benchmark-construction correction, not a method
change prompted by navigation efficacy.  The original attempt is retained at:

`/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_role_pair_20260814T072000Z_attempt5`

Its job IDs were collection `15713414`, construction summary `15713424`, query
evaluation `15713434`, policy summary `15713443`, and verification `15713453`.
All were cancelled; the preserved directory is a failed construction audit and
must never be reported as an efficacy run.

## Defect

The paper task has one independent Revisit query after one genuine online-A
history.  The wrapper nevertheless invoked the double-Revisit B/C builder.  It
therefore required two historical goals separated by at least 32 frames and a
valid B-to-C transition of at least 2 m.  Those constraints belong to a
three-leg/double-Revisit benchmark and are causally unrelated to this task.

There was also an eligibility mismatch.  Materialization could preselect frame
16, whereas the deployed LingBot memory has anchor margin
`S + W - 1 = 8 + 32 - 1 = 39`.  A downstream V1 check then correctly rejected
the early frame.  In the audited example, role B remained constructible but
role C was rejected solely because its selected source frame was 16.

## Frozen correction

The replacement builder is `build_single_revisit_source.py`.  It:

1. considers runtime-eligible online frames 39 through `end - 16`, stride 8;
2. retains only candidates 2--9 m from the online-A endpoint;
3. orders them by absolute error to a frozen 3 m target, then frame index;
4. applies the unchanged controlled V1 perturbation and co-visibility checks;
5. exposes at most four deterministic candidate historical goals so that the
   outcome-blind role-pair sampler can find a matched Novel goal;
6. never reads a navigation query outcome.

Materialization likewise requires only one runtime-eligible frame; it no longer
uses a pair of frames separated by 32 as an admission condition.  The
single-query correction removes only the irrelevant second-Revisit and B-to-C
constraints.  It does not loosen certificate thresholds, Revisit visual
support, Novel support exclusion, role hiding, controller radius, rollout
budget, arms or statistics.

## Pre-query validation

- Consumed MP3D histories: 3/4 histories constructible under both
  support-controlled and natural-direction protocols; the fourth is retained
  as Novel-pair construction attrition.
- Replica `room_0`: 4/4 histories constructible under both protocols.
- Replica `office_4`: 1/4 histories constructible under both protocols; all
  three failures remain explicit attrition.
- Unit tests verify that separate start/end anchor margins admit valid long
  histories while preserving the runtime lower bound of frame 39.
- A corrected consumed-scene four-arm integration smoke is required to show
  zero certified Novel activation, exact native fallback, positive Revisit
  activation and no runtime failure before a replacement HPC query job may be
  released.

These checks measure construction and implementation only.  They are not SR
evidence and cannot be used to tune the frozen method.
