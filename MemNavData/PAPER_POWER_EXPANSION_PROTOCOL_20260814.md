# MP3D phase-2 power expansion protocol

Date: 2026-08-14 (Asia/Shanghai)

## Trigger and timing

Attempt 7 sealed its construction population before query evaluation:

- 32 source episodes over the fixed 16 MP3D scenes;
- 17 native Goal-A successes over 12 scenes;
- 9 constructible role-pair histories over 9 scenes;
- pre-registered target: at least 20 histories over at least 12 scenes.

The population receipt SHA-256 is
`2ecb102f137f0ec25abd615ec544f342cb4d259a9d945fa069041a8a5bb611bc`
and records `policy_outcomes_read=false`.  Phase 2 was frozen while attempt 7's
evaluation array was still running and before its policy summary or independent
verification existed.  Its sole trigger is insufficient sealed sample size;
no query outcome, effect direction, SR, SPL, acceptance rate, or error case was
used.

## Frozen expansion population

- Keep exactly the same 16 scenes and training-scene exclusion as attempt 7.
- Do not replace or prioritize scenes based on Goal-A or query behavior.
- For every scene use exactly `episode_0002` through `episode_0005`, ordered
  lexically.  These 64 source episodes did not enter attempt 7.
- Freeze metadata, parquet and Goal-B carrier hashes before collection.
- Use base native seed `20260818`; per-episode deterministic behavior follows
  the unchanged evaluator contract.

Manifest:
`.diagnostics/paper_power_expansion_freeze_20260814_pre_result/paper_power_expansion_manifest.json`,
SHA-256 `c148c9695d0a03f877cd860b1c1810ace36e4750da9a7ed5ec385bb29336a598`.

## Method and evaluation

The policy and method implementation are byte-identical to attempt 7:

- frozen NavDP and MemNav checkpoints;
- native / raw metric / raw fixed / geometry fixed / certified arms;
- both support-controlled and natural-direction protocols;
- certificate thresholds 16 inliers, 5%/5% hull coverage and 2 px RMSE;
- fixed 2.5 m bearing residual, 600 steps, horizon 8, 1 m success radius;
- exact online-A replay, role hidden at runtime, no rescue or oracle.

Only two orchestration files change: source validation now reads the frozen
`episodes_per_scene=4`, and the two-protocol array capacity increases from 32
to 64 histories per protocol.  These do not alter a policy rollout.

The phase-2 job chain is submitted before attempt-7 results and starts only
after attempt 7's independent verifier exits successfully.  Phase 2 is reported
separately first.  Any pooled attempt-7 + phase-2 estimate must retain query
pairing and cluster bootstrap by the unchanged scene identity; it cannot count
episodes from one scene as independent scene evidence.

If phase 2 still misses 20 constructible histories or 12 scene clusters, report
the full attrition and retain the underpowered label.  No third adaptive source
expansion is authorized by this protocol.
