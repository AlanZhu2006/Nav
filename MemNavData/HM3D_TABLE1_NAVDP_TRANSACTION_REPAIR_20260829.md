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

## Prevention

The primary submitter and NavDP Slurm wrapper now treat model/base source and
executable server source as separate, receipt-bound dependencies.  Static
tests reject a wrapper that silently substitutes the older base server.  The
remote preflight compiles both server entry points, checks the transaction
symbols, verifies both immutable bundle receipts, and runs every Slurm script
through `sbatch --test-only` before creating jobs.
