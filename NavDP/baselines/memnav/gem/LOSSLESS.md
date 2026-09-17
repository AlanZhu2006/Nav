# Exact BF16 working-memory storage

`memory_kv_storage=lossless_bf16` selects an explicit storage backend for
`native_interval7`. It restores the original reader-precision BF16 K/V and
uses the original SDPA. The default remains `native`.

The standard `MemNavData/run_gem_memory_navigation.py run` entry point can
select the verified configuration with these explicit options:

```text
--mode native_interval7 --dense-window 64
--geometry-storage detector_support --kv-storage lossless_bf16
```

Supply the existing frozen plan and a new output directory as usual. The
launcher records these selections in `manifest.json` and forwards them to
the memory service, with separate window/archive/KV configuration receipts.
An incompatible writer or window is rejected before starting a service.
Omitting the options preserves the previous launcher choices. This entry
point is available for subsequent runs; the runtime follow-up did not launch
navigation or change defaults.

**Representation.** The initial eight views' patch K/V and all committed
special tokens remain BF16. Each of the latest 64 committed views has one
independently decodable patch record per global layer, in
`[K/V, head, patch, channel]` order. There is no quantizer, temporal predictor,
goal-dependent selection, or alternate inference backend.

Each group of 256 BF16 values stores:

| Field | Representation |
|---|---|
| Sign and seven fraction bits | One raw byte per value |
| Minimum exponent and exponent-range bit width | One 32-bit metadata word per group |
| Start/end of the group's bit planes | 32-bit prefix offsets, in words |
| Exponent difference | All required 0–8 bit planes, eight 32-bit words per plane |

For an original word `u`, the exponent is `(u >> 7) & 255`, and the raw byte
is `((u >> 8) & 128) | (u & 127)`. Only integer bit operations reconstruct
`u`. The codec preserves signed zero, infinities, and NaN payloads; no
floating-point addition is used even in the fused gather's protected paths.
An incompressible block may be larger than raw BF16 because of metadata.
The format does not promise a compression ratio for arbitrary input.

**Allocation and read order.** A GPU analysis kernel writes sign/fraction
bytes, exponent ranges, and word counts. A prefix sum and one size readback
determine the actual compact allocation; encoding then fills the bit planes.
This size synchronization and the pointer-table update are included in
online write timing. The 64-entry pointer table refers to GPU allocations
owned by the layer's Python record objects, which are released as their
views leave the window.

A fused gather restores one complete chronological layer into a shared
BF16 workspace: evicted views' special tokens, initial views, retained
window views, and the current observation. Commit reads contain at most 63
previous window views plus the current view; noncommit reads contain up to
64 previous views plus the current view. This preserves native eviction
timing. After the current attention, a committed view is encoded and stored.
Noncommit reads do not mutate the working state.

The current restore kernel assigns one workgroup to each stored 256-value
codec block and writes its original bits directly to chronological output
positions. Other workgroups in the same launch copy initial patches,
historical special tokens, and the current view into disjoint output slots.
This avoids locating the owning history record separately for each output
element. The storage format, visible views, output shape/strides and shared
workspace size are unchanged; the restored tensors feed the original SDPA.

Twenty-four global layers reuse one workspace. The isolated first-40-view
scale computation shares that scratch and model parameters while detaching
and restoring the live state containers. Reset releases semantic records;
scratch may remain allocated for reuse. Access requires the existing single,
serialized CUDA execution stream. Cross-stream asynchronous access is not
an evaluated contract.

**Numerical scope.** Exactness is relative to the verified BF16 reader
contract, not reconstruction of pre-cast FP32 post-RoPE keys. Model weights,
current Q/K/V, SDPA, W64, interval7, temporal positions, camera head,
historical geometry archive, and original SP/LG/PnP retain their roles.
The module does not address long-range pose drift or change the native
320-committed-view boundary.

`lossless_statistics()` reports values only: allocated tensor bytes,
effective stored bytes, raw/encoded window bytes, and shared decode bytes.
Each layer maintains compressed-window bytes when a view is committed or
evicted. Queries compute the remaining totals from integer dimensions and
the live special-token capacity, without traversing blocks or inspecting
their tensor handles. The aggregate visits the 24 layer states; query cost
no longer scales with the number of retained block tensors. Failure status
and allocated capacity remain live, including a write that fails after an
allocation grows. Returned dictionaries do not retain evicted payloads.
Neither effective payload nor the sum of tensor sizes is a replacement for
the full process's measured Torch peak. The allocator, model, initialization,
scale isolation, and temporary allocations must be included in cost claims.

**Validation records.** The bounded GPU proof covers all BF16 patterns,
real saved K/V, chronological boundary reads through 320 commits, initial
and special-token survival, failure state, and six real SDPA comparisons.
The complete new long-history run and independent archive/readout audit are
recorded in the [stage directory](../../../../.diagnostics/gem_lossless_memory_20260914/).
Use its final result rather than a transient running status for numerical
and cost claims.

The subsequent [runtime follow-up](../../../../.diagnostics/gem_lossless_runtime_20260914/RESULT.md)
covers command-line propagation and incremental accounting. It uses CPU
checks, tiny CUDA states, and a CPU metadata microbenchmark; it does not add
model, query-latency, or closed-loop measurements. At that frozen stage,
codec and tensor computations were unchanged from the sealed full-history
implementation.

The latest [decoder follow-up](../../../../.diagnostics/gem_lossless_component_cost_20260914/RESULT.md)
changes only the GPU restore indexing and its launch. It checks all BF16
patterns, chronological boundaries through 320 commits, six paired
shape-matched decode/SDPA cases, and the integrated decoder entry point.
Recorded samples contain only two heads and sampled queries; the component
diagnostic explicitly repeats them to production tensor shapes. Its timing
results are not measurements of a new complete model or navigation run.

The separate [descriptor-read follow-up](../../../../.diagnostics/gem_cached_retrieval_20260915/RESULT.md)
adds a small, lazily populated FP32 descriptor buffer to EpisodicGEM. It
uploads each new history descriptor once while retaining the original full
cosine input shape. This is independent of KV storage and does not change
the codec, decoder or attention. Its device allocation is reported separately
in `memory_status.descriptor_read_cache`; frozen full-model peak measurements
above predate this addition. Its timings cover cached descriptor retrieval
only, with existing goal embeddings, and are not full-model or navigation costs.
