# MP3D phase-2 infrastructure repair protocol

Date: 2026-08-14 (Asia/Shanghai)

## Incident

The frozen phase-2 source collection completed for all 16 scenes.  Job
`15729702` then failed after 17 seconds, before benchmark sealing or any query
policy execution, because `summarize_paper_online_a.py` still asserted exactly
two traces per scene.  The frozen phase-2 manifest contains four traces per
scene (`episode_0002` through `episode_0005`).

The original downstream jobs `15729707`, `15729708`, and `15729714` were
cancelled by the failed dependency with zero elapsed runtime.  At repair
freeze time the official run root contained no `online_a_inventory.json`, no
sealed `benchmarks`, no `evaluation` output, no policy summary, and no
independent verification.

## Allowed repair

The repair bundle is copied byte-for-byte from the immutable original phase-2
bundle.  Exactly two executable files may differ:

1. `MemNavData/summarize_paper_online_a.py`: read scene and
   `episodes_per_scene` counts from the SHA-pinned manifest, and validate exact
   scene/episode identities rather than asserting two traces;
2. `MemNavData/slurm_paper_online_a_summary.sbatch`: pass the same frozen
   manifest explicitly to the summarizer.

The repair protocol document itself is the only permitted added file.  The
bundle builder independently hashes every parent and child file and aborts if
any policy, controller, checkpoint binding, benchmark builder, evaluation arm,
metric, threshold, seed, scene, or episode file changes.

## Read-only validation before resubmission

The repaired summarizer was streamed to the remote Python interpreter and run
against the existing collection with its output written only to `/tmp`:

- source scenes: 16;
- source episodes: 64 (`4` per scene);
- native Goal-A successes/failures: `36/28`;
- materialized online histories: `35` over `15` scenes;
- query outcomes read: `false`.

The already-created construction fragments contain 19 unique role-pair
histories over 12 scenes.  This count is construction-only and contains no
Novel/Revisit policy outcome.

## Resume contract

- Do not rerun the completed Goal-A collection.
- Reuse the original run root and append only the missing inventory,
  immutable benchmark, evaluation, summary, and verification artifacts.
- Submit a new dependency chain:
  `construction summary -> paired evaluation array -> policy summary ->
  independent verification`.
- Keep the original five arms, two protocols, exact online-A replay, hidden
  role, fixed thresholds, fixed seeds, fixed 2.5 m residual, 600-step budget,
  and scene-cluster statistics.
- If any expected output already exists when a stage begins, fail closed rather
  than overwrite it.

This repair is infrastructure-only.  It does not authorize method adaptation,
threshold tuning, episode replacement, or reading partial evaluation outcomes.
