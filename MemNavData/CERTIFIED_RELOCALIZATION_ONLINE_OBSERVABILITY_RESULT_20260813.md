# Certified relocalization actual-online-A observability — result

Date: 2026-08-13 (Asia/Shanghai)

Status: complete. RTX 4090, RTX 5090, and formal HPC original-path runs are
exactly concordant.  HPC job `15657882` failed during a pre-render
manifest-sidecar path check and produced no method output; path-only
replacement `15677956` completed successfully.

## Result

The completed certificate experiment remains valid after correcting the
expert-A versus online-A label-contract gap.

All `120/120` episodes in its conditional-B denominator have actual online-A
maximum co-visibility `>=0.20`; `115/120` additionally have strong support
`>=0.50`.  Therefore none of the certificate/direct/geometry/native
conditional-B outcomes comes from a Goal B absent from the memory actually
supplied to the four arms.

The original result remains the supported-population result exactly:

| arm | B given shared A success |
|---|---:|
| certified scale-free bearing | **112/120 = 93.33%** |
| known-Revisit raw-DINO direct | 106/120 = 88.33% |
| geometry router | 91/120 = 75.83% |
| native ImageGoal | 27/120 = 22.50% |

Paired certificate effects on the actual-online-supported population:

- versus native: `+86/-1`, `+70.83 pp`, exact McNemar
  `p=1.137e-24`, scene-cluster 95% CI `[+59.32,+81.74] pp`;
- versus geometry: `+23/-2`, `+17.50 pp`, `p=1.943e-5`, CI
  `[+8.77,+27.59] pp`;
- versus direct: `+9/-3`, `+5.00 pp`, `p=0.1460`, CI
  `[-1.75,+12.60] pp`.

The honest claim is unchanged: certificate significantly beats native and the
old geometry router, and matches the strongest known-Revisit direct baseline
while providing a fail-closed, scale-free interface.  Its numerical advantage
over direct is not statistically established.

## Strong-support sensitivity

On the `115` A-success episodes with actual online maximum co-visibility
`>=0.50`:

| arm | B success |
|---|---:|
| certificate | **108/115 = 93.91%** |
| direct | 101/115 = 87.83% |
| geometry | 87/115 = 75.65% |
| native | 24/115 = 20.87% |

Certificate contrasts:

- versus native: `+85/-1`, `+73.04 pp`, `p=2.249e-24`, CI
  `[+61.32,+83.93] pp`;
- versus geometry: `+22/-1`, `+18.26 pp`, `p=5.722e-6`, CI
  `[+9.62,+28.33] pp`;
- versus direct: `+9/-2`, `+6.09 pp`, `p=0.06543`, CI
  `[-0.87,+13.51] pp`.

The gain is therefore not driven by marginally observable goals.  The strong
subset still does not cross the frozen significance boundary against direct.

## Complete observability ledger

| population | episodes | `max covis >=0.20` | `max covis >=0.50` |
|---|---:|---:|---:|
| all frozen episodes | 160 | 149 | 136 |
| shared A success | 120 | **120** | **115** |
| shared A failure | 40 | 29 | 21 |

All 11 episodes below `0.20` are A failures and never enter conditional B:

- `<0.10`: 9;
- `[0.10,0.20)`: 2;
- A successes among them: `0`.

The five supported-but-not-strong A-success episodes are not a hidden failure
mode: certificate took over in four and all four succeeded; the remaining one
fell back and failed.

The equality between `115` strong-support episodes and `115` certificate
takeover episodes is coincidental, not a one-to-one gate:

| | certificate takeover | native fallback |
|---|---:|---:|
| strong support | 111 | 4 |
| `[0.20,0.50)` support | 4 | 1 |

Thus the certificate is checking a localizable pose hypothesis, not merely
thresholding ground-truth visual overlap.  Four strongly supported views still
failed the atomic certificate and safely fell back; four moderately supported
views passed it and succeeded.

## Integrity evidence

The outcome-independent co-visibility protocol is documented in
`CERTIFIED_RELOCALIZATION_ONLINE_OBSERVABILITY_AUDIT_20260813.md`.  The shared
auditor validates the certificate run's own four-row Williams order,
`verified_bearing_v1` adapter, fixed `2.5 m` radius, backend/routes,
oracle-off receipts, manifest/asset/episode hashes, Goal-A identity, trace SHA,
and shared-A outcome before reading B outcomes.

Both completed machines reproduced:

- `160/160` Goal-B JPEG hashes;
- `34,437/34,437` online-A trace JPEG hashes;
- all 160 co-visibility curves and per-episode rows exactly;
- all summary and stratified paired statistics exactly.

RTX 4090 output:

`/home/asus/Research/Nav-graph-blind/.diagnostics/certified_relocalization_online_observability_local_preview_20260813.json`

SHA256:
`543f8c275b5d3535cec3b9e5642a8a405d73e8de434a91bd4bc889a692031135`.

RTX 5090 output:

`/home/cv/memnav_eval/certrel_online_observability_20260813/online_observability_audit.json`

SHA256:
`fed38e6a29309158704ce1c59f087c62ad9e8980055fa80e468c56884ed5ffb2`.

The JSON hashes differ only because each report records its absolute input
paths; `protocol`, `summary`, `stratified_outcomes`, and `rows` compare exactly.

Formal HPC submission history:

- first job: `15657882`, `FAILED` after 14 seconds before rendering because
  `sha256sum -c` resolved the sidecar's relative `data_manifest.json` entry
  from the Slurm spool directory;
- source bundle:
  `/scratch/yz11502/Research/source_bundles/certified_relocalization_online_obs_e980840a2e60d021`;
- source receipt SHA256:
  `e980840a2e60d0216b208b2a37a8d69870e68fbd3d628e9ade794cf847dd9bbf`;
- intended output:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_online_observability_20260813/online_obs_e980840a2e60d021/online_observability_audit.json`.

The launcher now verifies the sidecar from the manifest directory.  Fifteen
focused tests and shell syntax checks pass; replacement job `15677956` used
immutable bundle
`/scratch/yz11502/Research/source_bundles/certified_relocalization_online_obs_bd856c2c2e34f357`
with receipt SHA256
`bd856c2c2e34f357f98515965cf394c7f3ebade0460acfffce0c322d480785cd`.

Replacement result:

- job state: `COMPLETED`, elapsed `00:06:42`, exit `0:0`, node `ga014`;
- output SHA256:
  `d904aed865b451e5463ea3009f19b96459fc063ec5c313cce1b7296b5ee00ade`;
- formal output `protocol`、`summary`、`stratified_outcomes` 和全部 160
  `rows` 与 RTX 4090 本地报告逐字段完全一致；
- 再次复现 `160/160` goal render hashes 和
  `34,437/34,437` trace render hashes。

因此 formal HPC confirmation 也已完成，不再有 pending observability 项。

## Scope boundary

This audit confirms that the certificate experiment used real online-observable
Revisit goals.  It does not show that the system can classify arbitrary unknown
goals as Novel versus Revisit.  The deployable semantics remain:

```text
certified history pose -> scale-free bearing residual
unsupported/rejected   -> native ImageGoal fallback
```

Certificate rejection is not evidence that a goal is Novel.
