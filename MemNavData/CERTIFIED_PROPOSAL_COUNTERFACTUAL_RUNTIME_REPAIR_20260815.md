# Certified Proposal Counterfactual Runtime Repair (2026-08-15)

## Incident boundary

The first immutable audit bundle was never submitted as a Slurm evaluation.
Remote preflight found two packaging/runtime issues before any episode result
was produced:

1. the Habitat Python environment does not install `requests`, while the new
   read-only diagnostic runner imported it; and
2. unchanged transitive `MemNavData` modules live in the frozen base source
   bundle, but the audit Slurm script did not put that bundle root on
   `PYTHONPATH`.

The method-level counterfactual unit test passed remotely before this repair.
The full generic policy test suite was not a valid standalone-bundle test
because it exercises unrelated optional CDEC modules that were intentionally
outside this diagnostic bundle.

## Authorized repair

- Replace `requests` in the diagnostic runner with Python-standard-library
  HTTP and multipart encoding.
- Add the already frozen base source root and its `MemNavData` directory to
  the Slurm runtime module path.

No population, endpoint image, candidate order, PnP implementation,
certificate threshold, model checkpoint, action authority, or statistical
analysis changes. The audit remains post-hoc, consumed, read-only, and cannot
produce a closed-loop success-rate claim. A new immutable source bundle and
new output root are required; the failed preflight bundle is not reused.

## Repair 2: launcher packaging omission

The repair-1 bundle passed code and dual-environment preflight, and Slurm
accepted its resource request. Array job `15761657` then showed that the
declared frozen base bundle does not contain `retrying_server_launcher.py`.
Tasks 0--9 failed in 11--18 seconds before server startup; remaining tasks and
summary `15761658` were cancelled. The output root contains zero episode
records.

Repair 2 adds the already tested, method-agnostic launcher to the diagnostic
source bundle and points the Slurm script to that immutable copy. No policy,
model, data, proposal, PnP, certificate, or analysis logic changes. Repair 2
requires another fresh source receipt, output root, and submission receipt.

## Repair 3: reset-receipt dependency closure

Repair-2 smoke `15761873` started the server successfully and reached
`/navigator_reset`, then failed before history replay. The current shared
`policy_agent.py` always advertises both certified-relocalization and GOAT
arrival contracts in the reset receipt. The audit bundle contained the former
pure contract but omitted `goat_certified_arrival_contract.py`, so reset raised
`ModuleNotFoundError`. Formal `15761874` was dependency-blocked and ran zero
tasks; summary `15761875` ran zero work. The smoke produced zero episode
records.

Repair 3 adds that single pure, frozen contract module. It does not enable the
GOAT arrival endpoint, change the counterfactual computation, or affect any
policy/controller decision. Before another Slurm smoke, preflight must call
both status methods directly from the immutable bundle, in addition to the
existing dual-environment and launcher tests. A new source receipt and fresh
roots are required.

Repair-3 smoke `15762219` subsequently completed in `00:01:30` with exit code
zero and produced exactly one record. It exercised reset, full causal-history
replay, endpoint rendering, retrieval, geometry-first PnP/certificate, and the
read-only DINO counterfactual. Formal array `15762220` and summary `15762221`
were therefore released by `afterok`; no smoke record is reused by formal.
