# Fixed INT8 working memory

`memory_kv_storage="int8_storage"` is an explicit experimental representation
inside the existing central GEM. It keeps native interval7/W64, the original
model parameters, SDPA, archive, SP/LightGlue/PnP and controller. It is not a
new learned writer or a new navigation policy.

## What is retained

Each of the 24 global layers owns an `Int8LayerState`:

- Initial eight views: every patch K/V remains BF16.
- Most recent 64 committed views: every patch remains present; its K/V is
  encoded as signed INT8 with saved BF16 scales.
- Every committed view: all six special tokens remain BF16, even after that
  view's patches leave W64.
- Logical commit count and ring-slot identity are explicit. Observations,
  original temporal positions and native 320-commit limit are unchanged.

For each view and head, key scales use the maximum magnitude over patches,
separately per channel. Value scales use the maximum magnitude over channels,
separately per patch. Each scale is maximum/127 rounded to its stored BF16
format. Codes use that saved scale, round to nearest and clamp to [-127,127].
Genuine all-zero groups use scale 1 and zero codes. Nonfinite data or a
nonzero group underflowing BF16 scale storage raises an error requiring reset.
No goal, future query, attention gate, threshold search or fallback is used.

## Read and commit

The current observation always contributes its complete BF16 K/V to its own
attention. A commit compresses those patches only after that attention call.
Noncommit observations are temporary and do not change the stored state.

One fused Triton kernel writes the current layer into a shared BF16 buffer,
in the exact native SDPA token order: evicted-view specials, initial eight
views, retained window views, current view. A full-window commit excludes the
oldest window patches before its attention; a noncommit read retains all 64
previous views plus the temporary current view.

Only one layer is decoded at a time. The model shares one decode buffer across
its 24 layers and serialized scale-calibration calls; there are no 24 resident
decoded caches. Its actual 417,230,848 bytes at the fixed model shape are
included in measured GPU costs. Scale calibration temporarily owns distinct
semantic state and restores the original online containers and counters.

`reset` clears all semantic state; the model-owned decode scratch is reusable
and carries no cross-frame meaning. Backend selection requires an empty
verified stream. An error stops writes and requires reset. The representation
is selected explicitly and never changes automatically during an episode.

## Evidence and status

As of 2026-09-14, all 31 related CPU contracts pass. All 36 CUDA decode
comparisons are bit-exact to an independent chronological Torch decoder,
including real model heads/channels, window rollover and the 319 boundary.
This tests representation decoding, not equality to unquantized memory.

The two candidate-only 234/1948 full-model runs are complete, with original
SP/LG/PnP queries and source-bound archive reduction. The long-history model
peak fell from 13.117 to 9.344 GiB, including the shared decode workspace and
isolated scale state. Observed mean writes were 235.409 versus 216.117 ms;
these are shared-GPU single measurements, not isolated speed guarantees.

Four fixed development-scene A/B/A chains completed all 12 goals, with 2,421
continuous observations and independent motion/depth/state audits. Goal
changes do not reset or replay memory. Actions and geometry differ from the
reader-precision reference. Long-history rotation differences have P95
0.331 degrees and maximum 5.268 degrees; these are not ground-truth errors.

This backend remains an explicit experiment. Before adopting it for paper
results, it requires complete frozen Novel/Revisit population evaluation;
the previous lossless support-archive results do not establish INT8 behavior.
Evidence: [implementation result](../../../../.diagnostics/gem_int8_memory_20260914/RESULT.md).
