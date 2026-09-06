# HM3D Table-I authority-spectrum retry 2

Date: 2026-09-04  
Scope: infrastructure repair; population and four policy arms unchanged

## Second pre-outcome failure

Retry 1 used jobs `16924414`, `16924419`, and `16924421`.  Its smoke stopped
before any query arm completed because Python resolved an older namespace copy
of `run_hm3d_fullmono_query_history.py`.  That copy did not define the
`SCHEMAS` contract required by the authority-spectrum runner:

```text
AttributeError: module MemNavData.run_hm3d_fullmono_query_history
has no attribute SCHEMAS
```

No raw-memory or finite-witness navigation outcome was produced.  The formal
array and analysis were dependency-cancelled, so the previously absent arm
results remained unread.

## Repair

Retry 2 packages the exact query-history module inside the immutable task
bundle and performs a pre-submission provenance assertion on both:

- the imported module path must resolve inside that bundle;
- the imported module must expose the expected `SCHEMAS` symbol.

This changes import closure only.  It does not change the 28 histories, 21
scene clusters, 56 queries per arm, four arms, thresholds, controller,
first-40 monocular depth receipt, seeds, 600-step budget, or estimands.

## Current retry

Immutable task bundle:

`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_authority_spectrum_9bb6fc2fcc16303b`

Receipt SHA-256:

`9bb6fc2fcc16303b7025c708ea4f786e6b0fcf7536ed66b2f2b8860ba62bcb8e`

Jobs:

- smoke `16929024`: completed in 7:26 with exit code 0;
- formal array `16929030` (`0-27%4`): running;
- independent analysis `16929036`: waits on the complete formal array.

Result root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_authority_spectrum_20260904/formal_9bb6fc2fcc16303b`

The disclosure remains: this is a fixed-population causal mechanism ablation,
not a fresh generalization claim.  Native and strict-CEC outcomes were known
before design; raw-memory and finite-witness outcomes were not.

