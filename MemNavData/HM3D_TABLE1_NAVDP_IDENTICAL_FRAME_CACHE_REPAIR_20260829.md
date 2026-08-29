# HM3D Table-1 NavDP identical-frame cache repair

Date frozen: 2026-08-29  
Status: result-blind transport repair; no failed-cell navigation outcome read

## Incident

Formal NavDP array `16528369_[0-53]` completed 45 scene ranks, then three
non-empty ranks (`36`, `38`, and `41`) failed with the same server exception:

```text
RuntimeError: cached monocular depth belongs to a different transaction
```

The agent was stationary, so two different causal stream positions produced
byte-identical JPEGs. The 2026-08-21 server overlay cached depth by JPEG digest;
it therefore found the prior frame's valid cache entry but compared it against
the newer append's transaction token and failed closed. All three cells stopped
before a canonical scene completion. No success, SPL, distance, paired-arm
outcome, or partial estimator was read.

## Existing verified fix

The repository already contained a 2026-08-22 regression fix and three tests:

- a byte-identical JPEG with a new valid token is treated as a cache miss and
  refetched by that exact token;
- repeated access under the same token still hits the one-frame cache;
- an unknown or mismatched token remains a hard failure.

A direct diff against the active 2026-08-21 overlay shows that this cache block
is the only difference in `navdp_server.py`. The repair overlay is cloned from
the active immutable server bundle and replaces only this file; the MemNav
server, policy implementation, checkpoints, and all other runtime files remain
byte-identical.

## Exact continuation

Ranks `48-53` are the original array's predeclared empty ranks. They completed
normally before the repair transaction and are retained without replay.
Partial runtime/evaluation directories for the three failed non-empty ranks
`36`, `38`, and `41` are archived intact and made read-only. The replacement
array therefore runs exactly:

```text
36,38,41
```

with the original task bundle, benchmark, seeds, arms, order, 600-step budget,
and success criterion. Completed ranks are retained without replay. Replacement
aggregate and independent-verifier jobs still require complete coverage of the
frozen 28-history/21-scene population. The final joint seal depends on that new
NavDP verifier and the independently repaired ViNT verifier.

## Frozen repair receipt

- minimal server overlay:
  `hm3d_table1_navdp_cache_repair_2ae34ad0c1503958`;
- overlay receipt SHA-256:
  `2ae34ad0c150395849d4461913fc086f3b6ea7acf7249c763fe3e8808356ed6d`;
- failed-partial manifest SHA-256:
  `52276bdc08e6158f7642b3ce70735d10c2504633048cb4887da9c8bed25fb5b1`;
- exact repair array: `16541366_[36,38,41%1]`;
- replacement aggregate / verifier: `16541367` / `16541368`;
- replacement joint seal: `16541369`, dependent on NavDP verifier `16541368`
  and ViNT verifier `16540208`.
