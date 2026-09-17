# GEM storage: current implementation and evidence

The integrated candidate combines the native LingBot interval-7 W64 writer,
`memory_geometry_storage="detector_support"`, and the explicit
`memory_kv_storage="reader_precision"` option. Constructor/server defaults
remain `legacy`, `dense`, and `native`, respectively. Existing experiments use
their own frozen configurations; adding this option does not change them.

## What is saved, and when is a model replay needed?

Every RGB observation delivered to GEM is processed and archived. The first
eight observations initialize the geometry stream together; afterward each
observation produces its own depth, confidence, pose, and image descriptor.
The interval-7 setting governs neural KV commits, not observation archival.

|State|Where it lives|What a later read does|
|---|---|---|
|Current complete predicted depth|One current CPU tensor|Supplies current dense readout and monocular-depth delivery|
|Historical geometric support|One compressed NPZ file per observed frame|Loads exact FP32 depth/confidence values needed by the original point matcher|
|Historical RGB, descriptors, poses|RGB files and the existing CPU memory records|Provides visual retrieval and reference identities|
|LingBot neural K/V|GPU tensors with the native context/eviction rules|Supports the next streaming geometry prediction|

Historical support consists of the four bilinear sampling neighbours of every
frozen SuperPoint keypoint, plus the actual global finite-confidence minimum
used by the existing zero-quantile filter. Pixel indices, original depth and
confidence values, raster shape, scale, and RGB identity are retained. This is
lossless for the original reader's sampling support, not a recoverable complete
historical depth image. Unstored pixels remain unknown (NaN) when constructing
the CPU raster expected by the unchanged PnP interface. A query loads and
decodes stored data; it does not rerun LingBot to predict that reference depth.

Historical archives accumulate throughout an episode and are not discarded
when a view leaves W64. They are not all kept in GPU memory. The archive keeps
one decoded support record cached; a requested raster is materialized on CPU.
No cross-episode retention or constant-size disk archive is claimed.

There are two separate uses of the word *replay*:

1. The current metric-scale contract performs one additional computation on
   the first 40 RGB frames, when observation 40 arrives. This freezes a scale
   receipt without changing the causal stream. It is not a per-goal historical
   depth reconstruction.
2. Evaluation can load a previously recorded observed prefix once before the
   first query. Continuous A/B/A evaluation does not reset/replay at later goal
   boundaries. This prefix loading is separate from the old `legacy` canonical
   reference-depth path, which reconstructs an uncached selected reference by
   replay and caches its result afterward.

`EpisodicGEM.read_online_depth` reports `replayed_frames=0`.
`EpisodicGEM.read_replayed_depth` rejects canonical replay. Archives contain
first-observation geometry; later reads do not revise historical depth or
repair long-range drift.

## The optional neural storage change

The verified native attention implementation produces post-RoPE keys in FP32,
stores them in FP32, and converts the complete key input to BF16 for its actual
CUDA SDPA read. Values already use BF16. `read_precision.py` converts each new
key to the reader's BF16 precision before the original reshape/concatenation.
The original attention computation, committed-view sequence, eviction,
special-token handling, tensor shapes, camera and depth heads remain intact.

The implementation is a normal static `ReadPrecisionAttention` class. The
public GEM option installs it on the 24 global attention instances, retaining
the same weight objects and bytes. It requires the verified native SDPA source,
CUDA BF16 execution, W64, and a fresh/reset stream. Unsupported configurations
raise an error. They do not silently select another memory algorithm.

At an identical cache state, FP32 K plus BF16 V uses six bytes per matching
key/value element pair; BF16 K plus BF16 V uses four. Thus these K/V tensors use
two thirds of the original bytes. This ratio does not apply to model weights,
all GPU allocations, disk archives, or the whole process. Camera/special state
still follows native growth rules; W64 is not a bound on all memory.

## Completed evidence and remaining scope

The CPU algebra check follows the native eviction sequence over 1,948 raw
observations. The current implementation passes 47 relevant CPU tests,
including weight identity, allowed API routing, strict reset failures, and the
single-statement attention change.

Both the initial probe and the public GEM API have completed 234-frame and
1,948-frame GPU comparisons against sealed native outputs. Independent CPU
audits recomputed all published pose/descriptor and archive comparisons, every
available current-depth digest, final dense-depth/scale receipts, original
SP/LightGlue/PnP query results, cache shapes/bytes, and actual worker GPU UUIDs.
All compared outputs are exact on these two histories.

The public API's instrumented service Torch peaks are 10.766593 and 13.117271
GiB, compared with the original probes' 14.102546 and 18.288589 GiB. These are
validation runs on the same physical RTX 4090, not independent timing repeats
or a general accuracy claim. Existing real-world services remain co-resident.

Evidence root: `.diagnostics/gem_connected_memory_20260913/`.

- Native/prototype audit: `kv_read_precision_001/full_independent_001.json`,
  SHA `a1323d692f177a733fe88ce4476298add984b4aa57c8cdb851164d2e69148cf4`.
- Public API audit: `reader_precision_integration_002/full_independent_001.json`,
  SHA `4e27e3a26b97f84937625e8422a875ec6f446db72e269e01588d389735c69411`.
- Prior source versions: `source_snapshot_before_reader_precision_001/manifest.json`.
- First verifier import-order failure: `reader_precision_integration_001/`;
  preserved in full, before GPU inference.

The four fixed continuous A/B/A trials completed under
`reader_precision_continuous_001/`: all four chains and all twelve legs
succeeded. Actions and the 2,419 cumulative RGB/pose records match the sealed
native-KV support-archive references exactly. The independent reduction has
SHA `eff67f0cd5e67470f8f881dcd129f49a6c4dcfe3ee14f57d980db51e56864607`.
These are existing development histories, not new scenes or timing repeats.

The separate twelve-trial cost plan in `reader_precision_resources_001/`
is complete, including its original independent reduction. In 234-frame runs,
mean amortized write time is 164.191 ms/frame with native KV and 165.250 with
reader-precision KV; Torch peaks are 14.103 and 10.767 GiB. In 1,948-frame runs,
the corresponding write times are 302.015 and 236.143 ms/frame, and peaks are
18.289 and 13.117 GiB. All three long-history candidate runs were faster than
their paired native runs; the short-history mean does not show acceleration.
All repeats and co-resident GPU contexts are retained. Tables, raw values and
figures are in `final_memory_costs_001/`; the complete reduction has SHA
`07eb7a23da867fe11dea832de0c0ac3d483693fd6abde15ab44bde3809be1bf0`.

The existing HPC population controls and separate A100 archive-cost cohort
remain unfinished. Real-world integration is outside the current step.
Completed numerical, public-API and continuous-navigation evidence should be
reused while the implementation is unchanged. Further correctness runs require
a substantive change, a specific failure or an explicit unresolved requirement.
This storage implementation does not establish a new geometric estimator,
learned memory policy, drift repair or complete real-time control performance.
