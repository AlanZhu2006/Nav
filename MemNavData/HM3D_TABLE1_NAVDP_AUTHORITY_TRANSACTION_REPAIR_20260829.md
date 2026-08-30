# HM3D Table-1 NavDP authority/transaction composition repair

## Trigger

The first sealed HM3D Table-1 NavDP branch was internally valid as an exact
fallback audit, but it did not execute the intended CEC treatment.  Only after
the independent NavDP verifier and the joint controller seal completed were
the frozen summaries opened.  They showed zero CEC takeovers and 2,657 runtime
failure plans.  Raw receipts reproduce one deterministic cause on every such
plan:

```text
RuntimeError: certificate endpoint used wrong authority policy
```

The NavDP runtime combined a transaction-aware NavDP server with an older
MemNav server endpoint.  The latter could compute a strict certificate but did
not accept or echo the `authority_policy` request field required by the frozen
evaluator.  The evaluator therefore failed closed to exact native navigation.
The same population's ViNT CEC path used the newer authority hub and accepted
27/28 Revisit queries, independently ruling out “no geometric witness exists”
as an explanation for NavDP's zero takeovers.

## Repair boundary

The repair composes already receipt-bound components; it changes no navigation
method:

- MemNav server and policy agent: the independently verified Final14 authority
  bundle `final14_cec_authority_3f5783aca521b0a5`;
- NavDP server: the independently verified full-monocular transaction repair
  bundle `hm3d_fullmono_transaction_repair_67e1132783ce2cb1`;
- controller checkpoints, CEC thresholds, DINO/LightGlue/PnP implementation,
  fixed 2.5 m residual, query images, actual-mono histories, seeds, 600-step
  budget, 1 m success radius, and balanced arm order: unchanged.

The complete NavDP pair (`mono_native`, `mono_cec`) is rerun for every history
in the same process.  Reusing only the old native arm would violate the
project's same-machine/same-process paired contract.

## New fail-closed guard

`certificate_endpoint_failure` is an infrastructure failure, not a legitimate
certificate rejection.  The history runner, aggregate, and independent raw
verifier now all require `runtime_failure_plans == 0`.  A normal evidence
rejection remains valid and must execute the exact native request; an endpoint
or proof-runtime failure aborts the cell and blocks all downstream statistics.

The formal DAG is:

```text
one complete two-arm smoke
  -> all 54 parent scene ranks (empty ranks exit before model loading)
  -> NavDP aggregate
  -> independent raw verifier
  -> cross-controller seal using the retained read-only ViNT verifier
```

The retained ViNT branch is not rerun.  Its verifier SHA-256 is pinned in the
new seal.  The old NavDP 0/0 comparison remains an infrastructure incident and
must not be reported as a CEC performance result.

## First smoke-gate incident and dependency-closure repair

The first full-pair repair submission stopped at its one-cell smoke
(`16543736`, exit `2:0`) before any formal rollout.  The composed authority
overlay contained the authority parent's `memnav_server.py` and
`policy_agent.py`, but omitted that same parent's `router_candidates.py`.
Script-local import resolution therefore fell through to the older base copy,
which does not export `causal_goal_support_indices`.  No controller outcome
was produced, and all downstream jobs were cancelled by dependency.

The replacement overlay adds the byte-identical `router_candidates.py` from
the same receipt-bound Final14 authority parent.  It changes no model,
threshold, query, or execution rule.  In addition to bundle checks, the
preflight now imports `policy_agent` under the exact overlay-first runtime
`PYTHONPATH` and asserts that `router_candidates` resolves inside the overlay
and exports `causal_goal_support_indices`.  The new overlay receipt is:

```text
718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
```

The next smoke (`16544226`) also stopped before formal evaluation.  Importing
the MemNav authority closure now succeeded, but the common server
`PYTHONPATH` placed MemNav's script-local `policy_agent.py` before NavDP's
unchanged sibling module.  Consequently `navdp_server.py` imported
`MemNavAgent`'s module and could not find `NavDP_Agent`.

The second closure repair changes only process-local module precedence:

- the MemNav process resolves `memnav/` before `navdp/`;
- the NavDP process resolves `navdp/` before `memnav/`;
- both orders remain restricted to the same receipt-bound overlay and base
  roots;
- the evaluator keeps the common frozen source set.

The preflight now imports both `policy_agent` modules under their exact
process-specific order and asserts the expected public class and source path.
This is an infrastructure namespace repair, not a policy or method change.
