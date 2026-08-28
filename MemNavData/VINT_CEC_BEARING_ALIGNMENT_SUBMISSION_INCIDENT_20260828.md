# ViNT–CEC bearing-alignment submission incident (2026-08-28)

## Scope

This note records an infrastructure-only duplicate gate submission for the
outcome-aware five-query ViNT–CEC bearing-consumption mechanism test. It does
not change the frozen scientific population, arms, or decision gates, and it
does not authorize a paper SR claim.

## What happened

The first remote preflight detected that the Habitat interpreter could not
import `requests` from its normal `site-packages`. The required module is
available under Habitat's vendored pip path, matching the established formal
runtime. The remote compound shell command did not begin with fail-fast shell
options, so later commands continued after the self-test failure and submitted
only the first gate array element.

The duplicate gate was:

- job: `16497747_0`;
- run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/vint_cec_bearing_alignment_loss5_20260828/mechanism_20260828T025302Z`;
- source bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/vint_cec_direction_loss5_e06544d49f5fff48`;
- state: `COMPLETED`, exit code `0:0`;
- downstream jobs: none were submitted;
- submission receipt: absent.

The completed output is retained as a superseded diagnostic and is not deleted
or counted in the frozen aggregate. Its single cell independently showed that
both aligned arms consumed the CEC bearing and succeeded, while the unaligned
anchor arm moved in the opposite direction and failed.

## Authoritative retry

After adding the Habitat vendored-`requests` path to the remote self-test, the
full remote preflight passed. The authoritative immutable DAG is:

- run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/vint_cec_bearing_alignment_loss5_20260828/mechanism_retry1_20260828T025800Z`;
- source bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/vint_cec_direction_loss5_526f043b173ce55f`;
- source receipt SHA-256:
  `7b43f0c195bd8d752247f564b1a93e35839ab5c02423a5baae3a10945536bee7`;
- gate: `16497965`;
- remaining array: `16497973`;
- analysis and independent verification: `16497977`.

The two scientific bundles contain identical experiment code. Their only file
difference is `MemNavData/bundle_selftest.sh`; the retry version adds explicit
support for the extra interpreter search path used during environment
validation. Therefore the old completed cell is useful diagnostic evidence but
the retry DAG remains the sole authoritative aggregate.

## Permanent prevention

`submit_vint_cec_bearing_alignment_loss5_hpc.sh` now starts the remote compound
preflight with `set -euo pipefail`. Any future remote self-test, checksum, or
immutability failure stops before bundle publication or `sbatch`.
