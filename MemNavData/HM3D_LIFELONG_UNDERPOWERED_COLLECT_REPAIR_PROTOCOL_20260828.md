# HM3D underpowered factual-C exact-repair protocol

Frozen on 2026-08-28 after inspecting only Slurm state, output completeness,
compute-node provenance, and failure logs.  Successful factual-C navigation
outcomes and all B2 outcomes remained unread.  The machine-readable twin is
`hm3d_lifelong_underpowered_collect_repair_20260828.json`.

The authority-gated launcher `16504366` failed because the submission wrapper
placed its default GPU partition before the explicit `cpu_short` partition.
Before that failure, it created factual-C array `16505696`.  Sixteen of its 22
elements completed and six failed: `0,1,7,9,11,13`.  No shared-C population was
sealed and no B2 evaluation was created.

There are two failure mechanisms.  Indices 0 and 13 reached a fail-closed audit
that incorrectly assumed Goal-B replay begins with an empty NavDP FIFO.  In the
actual continual sequence Goal-A has already populated the bounded FIFO, so a
short Goal-B trace can correctly finish at the memory limit.  The repair checks
one-frame monotone/saturating growth relative to the first post-append length;
it does not alter a frame, memory entry, proposal, action, seed, or result.

Indices 1, 7, 9, and 11 failed exact rendered-RGB hashes after being scheduled
on a different node from the node that created their frozen factual-B trace.
Every retry is therefore pinned to its own factual-B source node.  Failed
partials are moved, not deleted, into the frozen archive path in the JSON
protocol.  The 16 completed directories are retained byte-for-byte, and only
the exact six failed indices may be resubmitted.  Two deterministic dependency
lanes cap concurrency at two GPUs.

The same node-affinity rule continues into B2: each sealed-C item is replayed
on the `compute_identity.host` that both reproduced B and generated C.  This is
an execution-integrity constraint derived from exact RGB receipts, not a
navigation-outcome filter.  Population sealing, one-item smoke, formal paired
B2, aggregation, and independent raw-file verification remain dependency
ordered.  The original powered-confirmation claim remains permanently
withheld; any completed result is still labelled an underpowered external
continual-memory mechanism test.
