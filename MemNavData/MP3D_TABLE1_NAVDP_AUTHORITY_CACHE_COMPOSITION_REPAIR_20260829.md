# MP3D Table-1 NavDP authority/cache composition repair

Date frozen: 2026-08-29 22:45 Asia/Shanghai
Status: additive infrastructure repair after exact-retry attempt 1 failed closed

## Incident

Exact-retry attempt 1 (`16558664_27`) correctly fixed the identical-JPEG/new-
transaction cache collision, but failed before producing a canonical completion:

```text
ABORT: mono_cec/novel: certificate runtime failure is not a valid policy outcome
RuntimeError: certificate endpoint used wrong authority policy
```

The cause is a source-composition error. The reused cache-repair overlay contains
the fixed NavDP server, but its MemNav server predates the strict authority
endpoint. File hashes make the mismatch exact:

- cache overlay `navdp_server.py`: `222f1be1...3529` (correct cache semantics);
- cache overlay `memnav_server.py`: `f2f1f697...a630` (old authority endpoint);
- original MP3D task `memnav_server.py`: `edd67074...ee61` (strict authority);
- original MP3D task `navdp_server.py`: `222f1be1...3529` (correct cache semantics).

Thus attempt 1 composed one repaired transport component with one obsolete
authorization component. The runtime-failure guard stopped the cell as designed;
the canceled downstream jobs produced no summary, verifier, or seal.

## Minimal correction

The original immutable MP3D task bundle
`mp3d_table1_controller_portability_eb7cdf82477f6aa1` already contains the exact
required composition: its MemNav authority files are byte-identical to the
verified authority overlay, while its NavDP server is byte-identical to the
verified cache repair. It was frozen before the current outcomes and its complete
receipt remains valid. Attempt 2 therefore uses that same task bundle as both:

- evaluator/runner source; and
- executable server source.

No file is newly synthesized and no policy, threshold, model, population, seed,
arm, query, budget, or success criterion changes. The preflight now checks both
component file hashes and both runtime symbols from the same resolved source root;
checking only the cache symbol is no longer sufficient.

## Exact continuation

The canonical completion set still lacks only histories 29 and 30. Attempt 1's
partial history-29 directory is moved intact into a second read-only archive; its
distinct runtime-attempt directory is hashed and made read-only. History 30 still
has no canonical directory. Attempt 2 reruns only scene rank 27 with history
override `29:30` and a new runtime-attempt name.

The ViNT exact retry is independent and is retained. A replacement NavDP
aggregate/verifier chain is submitted; a new joint seal depends on that verifier
and the already-submitted ViNT verifier `16558669`.

## Outcome-visibility disclosure

Classification inspected the failed cell's exception text and runtime-failure
fields. No success flag, final distance, path metric, paired contrast, partial SR,
or aggregate estimator was read or computed. Together with the previously
disclosed accidental view of one completed history, these diagnostics are recorded
explicitly; neither affected the frozen missing set or scientific contract.
