# Native NavDP / ViNT Novel-A local smoke

Date: 2026-08-28 (Asia/Shanghai)
Status: completed exploratory integration test; **not a statistical paper result**.

## Question

Can the frozen ViNT checkpoint navigate one ordinary, previously unseen
ImageGoal in the same Habitat evaluator used by NavDP, rather than only execute
a CEC-authorized historical anchor?

## Frozen unit

- scene: `gxdoqLR6rwA`;
- episode: `episode_0000`;
- evaluator geodesic start-to-A distance: `4.0575 m`;
- identical start pose, target JPEG, Habitat scene, `1.0 m` success radius,
  `400`-step budget, and pure-pursuit trajectory executor;
- both arms stop after Novel-A; neither receives CEC or episodic memory;
- NavDP consumes its native metric-depth request;
- ViNT consumes current RGB and target RGB only through the audited
  `native_imagegoal` proxy.

The sensor inputs are therefore deliberately disclosed as different. This
smoke tests native controller capability and integration, not a sensor-matched
ranking.

## Result

| controller | sensor | reached A | steps | path | final distance | SPL |
|---|---|---:|---:|---:|---:|---:|
| frozen NavDP | RGB-D metric request | yes | 85 | 3.193 m | 0.986 m | 1.0 |
| frozen ViNT | RGB only | yes | 126 | 3.772 m | 0.996 m | 1.0 |

The complete run was repeated once after correcting the ViNT receipt's
non-applicable depth-source field. Both controllers reproduced exactly the same
success, steps, path length, and final distance.

Primary receipt:

```text
.diagnostics/navdp_vint_novel_a_pair_20260828/20260827T190850Z/summary.json
sha256 310a037a9ca0db0e54b7889901f7885fa0b7cc24320e754d1597070a241d9053
```

Runner:

```text
MemNavData/run_local_navdp_vint_novel_a_pair.sh
```

## Interpretation

This establishes that the current ViNT integration can complete a native
Novel-A ImageGoal under the shared simulator executor. It is no longer true
that ViNT has only passed a Revisit-anchor interface smoke. On this one easy
episode, NavDP is faster and shorter, but `N=1` and the sensor contracts differ,
so neither controller can be called better.

The result also does not establish CEC portability. Native ViNT receives the
original target ImageGoal, whereas CEC-to-ViNT receives a proof-bound historical
anchor image. The formal portability question remains whether the latter
projection improves over the same controller's paired forced-reject baseline
on a sufficiently large accepted Revisit population.

## Decision

- Keep this as implementation/context evidence, not a main result table.
- If a broader native-controller comparison is needed, freeze multiple scenes
  before reading outcomes and either match sensors or report separate RGB-only
  and RGB-D tiers.
- Do not select only known NavDP failures and then describe the resulting set as
  an unbiased Novel benchmark.
