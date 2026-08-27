# GOAT Certified Arrival Reset-Seed Repair (2026-08-15)

## Incident

Engineering smoke job `15759008` failed before the first observation was
streamed and before any navigation action, arrival query, or GOAT outcome was
produced. The runner derived a deterministic signed-int64-range episode seed.
NavDP's request helper supports that range and reduces it for NumPy, but the
historical MemNav reset path called `np.random.seed(seed)` directly. NumPy's
legacy RNG rejected the value outside `[0, 2**32-1]`, so `/navigator_reset`
returned HTTP 500.

The formal job `15759010` remained `DependencyNeverSatisfied`; it ran zero
episodes. Summary `15759014` and verifier `15759020` also ran zero work.

## Authorized repair

Map the deterministic episode hash to `seed % 2**32` once and send that exact
uint32 reset seed to both NavDP and MemNav. Per-plan and read-only-resample
diffusion seeds remain the original frozen 63-bit hashes. Add a boundary test
and record the uint32 reset seed in every episode receipt.

No manifest identity, model, checkpoint, controller, certificate, scale,
7.5 cm threshold, fallback, resample limit, budget, or statistical gate may
change. A new immutable bundle and fresh smoke/formal run roots are required;
failed outputs are never reused.
