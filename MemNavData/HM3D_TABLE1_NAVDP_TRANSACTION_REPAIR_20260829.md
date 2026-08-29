# HM3D Table-1 NavDP transaction repair (2026-08-29)

## Result-blind failure classification

The fresh-query construction gate completed before any policy evaluation was
submitted.  Its independent verifier authorized a frozen population of 28
histories from 21 HM3D scenes.  The benchmark manifest SHA-256 is
`f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01`.

The first NavDP smoke job (`16526559`) failed before producing a completed arm.
Only infrastructure logs were inspected.  The evaluator failed closed at
`bind_navdp_monocular_transaction` with:

```text
RuntimeError: MemNav planning append received different JPEG bytes
```

This message did not indicate a second JPEG encoding.  The formal evaluator
required the SHA/frame/depth-bound transaction introduced on 2026-08-21, while
the Slurm wrapper launched the server from the older immutable Final14 bundle.
That server's `/memory_step` response contains neither `image_sha256` nor a
transaction token.  The client therefore compared the expected digest against
a missing receipt field and aborted.  No partial SR or final-distance outcome
was read to diagnose or design the repair.

## Minimal repair

The model dependency remains the frozen Final14 source bundle and the same
NavDP, MemNav, and LingBot checkpoints.  Only the executable server overlay is
rebound to the previously verified full-mono transaction-repair bundle:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_fullmono_transaction_repair_67e1132783ce2cb1
```

Its receipt SHA-256 is
`05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97`.
That bundle was already used for the full-mono runtime repair and provides the
matching `append_request_frame` and
`require_monocular_depth_transaction` interfaces.

The repair does not change:

- the 28-history / 21-scene population;
- Novel/Revisit role visibility;
- checkpoints, thresholds, seeds, action budget, or success criterion;
- the two NavDP arms (`mono_native`, `mono_cec`);
- the already-running ViNT branch.

The failed smoke directory is retained as an immutable diagnostic artifact.
The replacement smoke writes to an additive path, while the never-started
formal NavDP directory remains the canonical formal output.  A replacement
joint seal depends on the repaired NavDP independent verifier and the original
ViNT independent verifier.

The additive submission completed with the following jobs:

- replacement NavDP smoke: `16527714`;
- replacement NavDP formal array: `16527718`;
- NavDP aggregate / independent verify: `16527722` / `16527860`;
- existing ViNT independent verify: `16526759`;
- replacement joint seal: `16527863`.

The repair bundle is
`hm3d_table1_navdp_transport_repair_c04fd5aa08b65126`, with receipt SHA-256
`c04fd5aa08b65126f2c2ff67ee2234a0dcd7428d3a642c9ccee08bb37be695aa`.
At submission, no partial policy outcome had been read.  The account permits
two simultaneous GPUs; after submission, the already-running ViNT array was
throttled from two concurrent cells to one so that the repair smoke and ViNT
formal evaluation could progress concurrently.  This is scheduler-only and
does not alter either evaluation contract.

## First repair smoke: dependency-closure failure

Replacement smoke `16527714` failed during NavDP server import, before Habitat
or either policy arm started:

```text
ModuleNotFoundError: No module named 'depth_anything'
```

The transaction overlay was intentionally narrow and did not carry the
unchanged vendored Depth-Anything package.  The runner had receipt-bound both
the overlay and the Final14 base bundle, but exposed only their repository
roots on `PYTHONPATH`; NavDP imports `depth_anything` from its baseline-local
directory.  Syntax compilation could not detect this transitive import
closure.  Formal job `16527718` and its descendants were dependency-cancelled,
again before producing an outcome.

The second additive repair exposes the `NavDP/baselines/navdp` and
`NavDP/baselines/memnav` directories from both receipt-bound bundles.  It also
imports `policy_backbone` and `DepthAnythingV2` in the remote container before
any `sbatch`.  This changes module resolution only; it neither replaces the
frozen package nor changes any scientific factor.  The first repair receipt is
retained at SHA-256
`a5b19eaab1a76eae0e1c2ac4f71305b12a34cf626cdee5816b8c081dcaaf7f86`.

The second repair submission passed the real import gate and all five Slurm
test-only gates. Its immutable evaluator bundle is
`hm3d_table1_navdp_transport_repair_6edacc0c6c13b389`, receipt SHA-256
`6edacc0c6c13b389d9eee5b1371d0353b14ad94c4dd6ea4eb520bd4bde63ffaa`.
Submitted jobs are replacement smoke `16528367`, formal array `16528369`,
aggregate `16528383`, independent verifier `16528385`, and replacement joint
seal `16528391` (also dependent on the unchanged ViNT verifier `16526759`).
Smoke `16528367` subsequently completed `0:0` in 5:43 and released the formal
array. This confirms the executable dependency and transaction interface; its
navigation outcomes were not read or used for any further change.
The outcome-blind scheduler/error-scan receipt is
`HM3D_TABLE1_NAVDP_TRANSACTION_REPAIR2_SMOKE_GATE_20260829.json`.

## Prevention

The primary submitter and NavDP Slurm wrapper now treat model/base source and
executable server source as separate, receipt-bound dependencies.  Static
tests reject a wrapper that silently substitutes the older base server.  The
remote preflight compiles both server entry points, checks the transaction
symbols, verifies both immutable bundle receipts, imports the baseline-local
NavDP dependency closure, and runs every Slurm script through
`sbatch --test-only` before creating jobs.
