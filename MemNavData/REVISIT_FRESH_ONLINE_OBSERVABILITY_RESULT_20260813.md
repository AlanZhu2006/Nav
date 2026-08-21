# Fresh160 actual-online-A observability audit — result

Date: 2026-08-13 (Asia/Shanghai)

## Bottom line

The fresh160 result is **not invalidated** by the expert-A versus online-A
trajectory distinction.

All `118/118` episodes in the frozen conditional-B denominator have actual
online-A maximum co-visibility `>=0.20`; `113/118` additionally have strong
support `>=0.50`.  Thus the original raw-DINO direct result
`109/118 = 92.37%` is a valid success rate on actual online-observable Revisit
episodes under the frozen known-Revisit protocol.

The direct-versus-geometry effect is unchanged in the actual-online-supported
population because that population is exactly the full shared-A-success
population:

- geometry: `93/118 = 78.81%`;
- raw-DINO direct: `109/118 = 92.37%`;
- native: `31/118 = 26.27%`;
- direct minus geometry: `+20/-4`, `+13.56 pp`, exact McNemar
  `p=0.0015438795`, scene-cluster bootstrap 95% CI
  `[+6.45,+21.43] pp` under the supplemental frozen seed.

The audit therefore strengthens, rather than weakens, the architectural
conclusion: once the phase is known to be Revisit, the RANSAC/SIFT hard gate and
candidate re-selection should not control the policy; raw-DINO top-1 direct is
the stronger frozen branch.

It still does **not** solve online Novel/Revisit selection.  The experiment
continues to assume a known-Revisit phase at Goal B.

## Actual observability ledger

| population | episodes | `max covis >=0.20` | `max covis >=0.50` |
|---|---:|---:|---:|
| all frozen episodes | 160 | 149 | 136 |
| shared A success | 118 | **118** | **113** |
| shared A failure | 42 | 31 | 23 |

All 11 episodes below `0.20` are A failures.  They never enter the conditional-B
denominator:

- `<0.10`: 9 episodes;
- `[0.10,0.20)`: 2 episodes;
- A successes among these 11: `0`.

This pattern has a direct causal explanation: when NavDP fails to complete A,
its partial online trajectory may never traverse the expert path region from
which Goal B was sampled.  When NavDP does complete A, every evaluated Goal B
is present in its actual memory under the generator's operational Revisit
threshold.

The five A-success episodes in `[0.20,0.50)` are retained by the operational
Revisit definition but excluded from the strong-support sensitivity analysis.

## Strong-support sensitivity analysis

Restricting to the `113` A-success episodes with actual online maximum
co-visibility `>=0.50` gives:

| arm | B successes | conditional SR |
|---|---:|---:|
| geometry router | 89/113 | 78.76% |
| raw-DINO direct | 104/113 | 92.04% |
| native | 28/113 | 24.78% |

Direct minus geometry remains significant and nearly identical in magnitude:

- paired `+19/-4`;
- risk difference `+13.27 pp`;
- exact McNemar `p=0.00259948`;
- scene-cluster bootstrap 95% CI `[+6.14,+21.01] pp`.

Therefore the primary result is not driven by marginally observable Goal-B
views.

## Integrity audit

Formal job:

- Slurm job: `15655698`;
- state: `COMPLETED(0:0)`;
- runtime: `00:05:13`;
- node/GPU: `gl008`, NVIDIA L40S;
- source bundle:
  `/scratch/yz11502/Research/source_bundles/revisit_fresh_online_obs_2b3e1fedddedb82c`;
- source receipt SHA256:
  `2b3e1fedddedb82c4444fbe99da93924b0c620dfe294da6d32e62869777c1dda`;
- frozen auditor SHA256:
  `7f437cdfd51d60d565e4abf495099948c609cab76b7ea11caa5e97d95ebcba75`.

Formal output:

`/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_online_observability_20260813/online_obs_2b3e1fedddedb82c/online_observability_audit.json`

SHA256:

`4467033a64578f74313db3b0c17563a4ed261b8417fcae9265df24cdc10bd1bd`

The audit independently checked:

- original manifest SHA256
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`;
- all episode metadata/parquet/Goal-B hashes and all 20 scene-asset hashes;
- 160 trace schemas, seeds, scene identities, own Goal-A JPEG hashes, native
  phase route, direct-controller contract, and raw trace SHA receipts;
- all three arms' scene order, seed, backend, route, adapter, oracle-off, shared
  A outcome, and trace SHA receipts;
- `160/160` rendered Goal-B JPEG hashes;
- `34,798/34,798` rendered online-trace JPEG hashes.

A separate local RTX 4090 run used an exact 70 MB minimal raw-artifact mirror
and independently stored MP3D assets.  Its `summary`, all 160 per-episode rows,
all co-visibility curves, and all stratified statistics are exactly equal to
the HPC L40S output.  Only the recorded filesystem input paths differ.

Local output:

`/home/asus/Research/Nav-graph-blind/.diagnostics/revisit_fresh_online_observability_local_preview_20260813.json`

Local SHA256:

`2df23a7184ec29c66b786b866795c866eb55f7091a54f83c6e46135d8e2b4eb5`

The implementation has `13/13` focused tests passing.

## Fail-closed record

The first submission, job `15655662`, stopped after 16 seconds before rendering
any episode because its preflight incorrectly rejected the HPC Habitat Python
symlink.  It produced no audit output and did not modify the original run.  The
replacement resolved the interpreter with `readlink -f`, revalidated the
physical target, built a new immutable source bundle, and then completed.

## Correct paper wording

Safe:

> On 118 shared-A-success episodes, every Revisit goal was geometrically
> observable in the actual online-A memory (`max covis >=0.20`).  Replacing the
> geometry hard gate/re-selection with direct raw-DINO top-1 increased
> conditional Revisit SR from 78.8% to 92.4% (`+20/-4`, exact McNemar
> `p=0.00154`).

Not yet safe:

> The system autonomously detects whether a new goal is Novel or Revisit.

That latter claim requires a separate online open-set phase selector; it is not
provided by fresh160 or by this observability audit.
