# HM3D Table-3 trajectory-length cache exact retry

Date frozen: 2026-09-02 (Asia/Shanghai)  
Status: infrastructure-only exact retry; no partial navigation outcome read

## Incident

The frozen 48-history trajectory-length population retained 45 canonical
completions. Array identities `35`, `41`, and `44` stopped before writing a
canonical completion. Their persistent NavDP server logs contain the same
exception:

```text
RuntimeError: cached monocular depth belongs to a different transaction
```

The failing server bundle predates the verified identical-JPEG/new-transaction
cache correction. A stationary agent can emit byte-identical JPEG observations
under two different valid causal transaction tokens. The old cache treated the
second observation as a hit and then rejected its newer token. This is a known
transport/cache failure, not a policy outcome or a trajectory-length effect.

Only exception text, job state, completion presence, and source hashes were
read to classify the failure. No success flag, final distance, path metric,
paired contrast, partial SR, or aggregate estimator was read or computed.

## Frozen repair

The repair retains the 45 hash-verified completions and reruns exactly indices
`35,41,44`. Their partial evaluation and runtime directories are moved intact
to a read-only failed-attempt archive and covered by a SHA-256 manifest.

The query wrapper, task source, benchmark population, protocol, model weights,
seeds, two arms, arm order, per-history step budget, certificate thresholds,
success radius, and downstream analysis remain unchanged. The only runtime
source substitution is the already verified combined server bundle
`mp3d_table1_controller_portability_eb7cdf82477f6aa1`:

- its MemNav server, policy agent, and router are byte-identical to the original
  strict-authority Table-3 server bundle;
- its NavDP server is the previously verified identical-frame cache repair;
- its remaining NavDP sibling files are byte-identical to the frozen base
  implementation.

The replacement summary depends on the exact retry, and the independent
verifier depends on the replacement summary. Neither stage may run unless all
48 canonical completion receipts exist and pass their SHA sidecars.

## Claim boundary

This repair changes no scientific variable and creates no new estimand. Formal
trajectory-length results may be opened only after the independent verifier
returns `verified=true`. A further failed cell remains an infrastructure
failure and does not authorize fallback completion or a reduced denominator.
