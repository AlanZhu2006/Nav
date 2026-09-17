# GEM paged BF16 working memory

`memory_kv_storage="paged_bf16"` selects an explicit candidate backend under
`native_interval7`, W64, initial8, interval7 and online-history readout. It shares
the original weights, frame attention, camera/depth heads and control interfaces.
The support archive, original SP/LightGlue/PnP and certificates remain in GEM.
Full-model geometry and cost validation is complete for the two fixed histories.
Observed writes are faster, GPU peaks are not lower, and geometry is not
bit-equivalent; this remains an experimental backend. See the stage RESULT.md.

## Per-observation lifecycle

1. Eight initial observations use LingBot's original joint SDPA initialization.
   Their consumed BF16 keys/values enter the paged working state.
2. A committed observation appends its full patch page and six special tokens.
   Before attention, the original rule evicts the oldest excess window patch.
3. A noncommit observation temporarily appends the same full representation,
   reads the existing window plus itself, then rolls back. It does not evict an
   old patch page. A kernel error still rolls back that temporary append and
   puts the working state into a failed state requiring reset.
4. The complete model publishes current geometry. GEM writes the corresponding
   first-observation evidence into the archive, independently of KV commitment.

FA2 outputs drive subsequent layers and predictions. This is not shadow
attention. BF16 information and visible-token membership match the intended
reader-precision representation; token ordering and floating-point reduction
change, so identical future geometry cannot be inferred from that membership.
Residual tokens may be FP32. Storage follows the CUDA BF16 attention-read
contract, not the dtype of the residual stream before QKV projection.

## Physical allocation and logical time

Each global layer has an independent tensor pool and matching page table.
Patch and special pages share one recycled free list. Pool buffers grow in
16-page increments, one layer at a time, with one explicit copy on growth.
There is no whole-history cat/clone or gather in the FA2 attention path.

At the fixed 518×518 raster, each patch page has exactly 1369 tokens. FA2
supports this size; padded intermediate pages would introduce unintended keys.
The visible order is initial patch pages, recent patch pages, then chronological
special-token pages. Only the final special page can be partial.

The native 320-commit limit requires at most two special pages. There are at
most 8+64+1 live patch pages, including the temporary observation, so the pool
cap is 75 pages per layer. For 24 layers, 16 heads and head dimension64, this is
about9.40GiB of pool capacity at full growth, before workspace and other memory.
It is an allocation bound, not a measured process peak. Short histories allocate
fewer pages; allocation slack is reported rather than hidden. This does not
extend the native temporal protocol or change RoPE positions.

One128MiB FlashInfer workspace is shared across the 24 layers. Planning is once
per single-frame step using the common page structure. Cache statistics expose
effective KV bytes, allocated pool bytes, workspace bytes and growth copies.
Torch allocated/reserved peaks must additionally include models, current frame,
camera state, initial-eight computation and scale calibration.

## Isolation, reset and selection

The first40 scale routine leases the same model weights with a separate empty
working state. `isolated_stream` detaches both the live paged manager and camera
containers before cleanup, releases temporary state, and restores the exact
original containers/counters even when inference or cleanup raises. It neither
clones the long-history pool nor reuses its mutable pages for calibration.

Episode reset drops the pool and workspace manager rather than just marking
pages empty. CUDA reserved memory may remain in PyTorch's allocator. Selection
requires an empty evaluation stream and the inspected LingBot source contract.
Missing FlashInfer, changed upstream implementations, invalid shapes/precision
or inference errors fail explicitly. There is no automatic backend switch.

## Evidence required for adoption

The CPU tests in `MemNavData/test_gem_paged.py` check real storage operations,
token membership through eviction/special-page crossing, temporary-read failure,
pool release, parameter identity and state isolation. A mock wrapper is used
only in CPU tests; those tests do not validate FA2 numerics or GPU costs.

The fixed complete-model comparison uses234and1948observations, all original
archive support values, per-frame poses/current-depth samples, scale and original
goal readouts. It preserves independent controls and records selected actual
Q/KV samples for later spatial-compression research. Those diagnostic captures
add measured overhead. See
[the execution directory](../../../../.diagnostics/gem_paged_memory_20260914/).

This stage implements paging, not spatial compression, quantization, a learned
gate, a second geometry model or a solution to long-range drift. Extra compression
requires a separate same-budget comparison after the paged baseline is assessed.
