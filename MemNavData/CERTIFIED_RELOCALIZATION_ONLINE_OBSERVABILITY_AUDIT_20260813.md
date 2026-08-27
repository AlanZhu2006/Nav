# Certified relocalization actual-online-A observability audit

Date: 2026-08-13 (Asia/Shanghai)

## Purpose

Audit the label contract of the completed certified-relocalization closed-loop
run.  Goal B was sampled as a Revisit against the expert-A trajectory, whereas
the four closed-loop arms received a separate shared NavDP online-A trace.

This audit asks whether Goal B was actually visible in the memory supplied to
the certificate, direct, geometry, and native arms.  It does not rerun NavDP,
train a model, change a success outcome, or test autonomous Novel/Revisit
selection.

## Immutable inputs

Run root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812/certrel_bearing_v1_20260812T1050`

- manifest SHA256:
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`;
- report SHA256:
  `0e41a6d9b339d143229ba405b04802654d2053b5d641a03ed2d09aefc1a589f4`;
- 20 scenes, 160 shared traces, four paired arms;
- frozen shared-A successes: `120/160`;
- frozen conditional-B outcomes: certificate `112/120`, direct `106/120`,
  geometry `91/120`, native `27/120`.

Those outcomes were already known before this retrospective label audit.  No
claim of outcome pre-registration is made.  However, the co-visibility
definition and thresholds are outcome-independent: they are copied exactly
from the generator and were already implemented and validated in the prior
fresh-confirmation audit before this certificate run was inspected.

## Frozen measurement

For each episode, render the frozen Goal-B pose and every pose in the
certificate run's own shared online-A trace.  Recompute the generator-equivalent
occlusion-aware 3-D surface co-visibility using:

- `480 x 270` RGB-D camera;
- `fx=355.81464`, `fy=351.687`, `cx=240`, `cy=135`;
- Goal-B back-projection stride `6`;
- metric-depth consistency tolerance `0.30 m`;
- episode camera height, frozen default `0.50 m`.

Support bands remain:

| actual online-A maximum co-visibility | interpretation |
|---:|---|
| `<0.10` | no support |
| `[0.10,0.20)` | ambiguous |
| `[0.20,0.50)` | operationally supported Revisit |
| `>=0.50` | strongly supported Revisit |

The implementation constructs and freezes all 160 observability rows before
loading any of the four outcome CSVs.  It then reports the original paired
effects in the full A-success, `>=0.20`, `>=0.50`, and unsupported diagnostic
populations.  No threshold may be changed from these results.

## Run-specific fail-closed receipts

The audit must verify:

- four-row Williams arm order and
  `certified_relocalization_closed_loop_v1` scene contracts;
- manifest, episode, asset, Goal-A, Goal-B, trace, scene, seed, and shared-A
  identities;
- certificate route `certified_relocalization`;
- adapter `verified_bearing_v1` with fixed controller radius `2.5 m`;
- direct route `phase`, geometry route `memory_geometry`, native backend
  `navdp`, and `retrieval_override=off` for all arms;
- byte-identical reproduction of rendered Goal-B and online-A JPEGs.

The same 160-row computation is run independently on a local RTX 4090 mirror
and the original HPC data.  Formal interpretation requires exact equality of
the per-episode rows and stratified outcome summaries across those runs.

## Interpretation

- If all 120 A-success episodes have actual online support `>=0.20`, the frozen
  `112/120` certificate result is valid on actual online-observable Revisit
  episodes.
- If not, the headline denominator must be replaced by the supported subset;
  unsupported outcomes are only diagnostics.
- The `>=0.50` subset tests whether the certificate result depends on marginal
  visual overlap.
- This audit cannot establish that the system autonomously knows whether a
  goal is Novel or Revisit.  Certificate rejection remains
  `Unknown/unsupported -> native`, never proof of Novelty.
