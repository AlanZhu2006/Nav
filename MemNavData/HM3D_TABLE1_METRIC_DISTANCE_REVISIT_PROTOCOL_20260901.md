# HM3D Table-1 full-metric Revisit promotion gate（2026-09-01）

## Question

When CEC's proposal, geometric witness, bearing, frozen NavDP controller, and
full-monocular depth are held fixed, does exposing the first-40-calibrated
translation norm improve closed-loop Revisit navigation over the canonical
fixed 2.5 m residual?

This experiment tests controller value, not whether the offline distance
estimate correlates with ground truth.  It is a consumed-population promotion
gate and cannot be presented as fresh confirmation.

## Frozen population

- HM3D Table-1 manifest SHA-256:
  `f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01`;
- 28 actual-mono Goal-A histories from 21 scene clusters;
- only each history's standard-support Revisit query is evaluated;
- the analysis role selects the ablation population but is never forwarded to
  runtime;
- query, history, scene asset, checkpoint, seed, 600-step budget, 1 m success
  radius, execution horizon 8, and certificate are unchanged.

## Arms

For the same verified raw LingBot vector `v_t` and frozen first-40 scale
`s_hat_40`:

```text
fixed_2p5m:  p_t = 2.5 * v_t / ||v_t||
full_metric: d_hat_t = s_hat_40 * ||v_t||
             p_t = min(d_hat_t, 10 m) * v_t / ||v_t||
```

The 10 m bound is NavDP's frozen native PointGoal input envelope, not a tuned
radius.  The formal treatment is interpreted as full metric only if its audit
records zero bound activations.

Missing/invalid scale is fail-closed to native ImageGoal, but the run is not a
clean radius attribution if scale availability differs at the first
authorization.  Such a cell must abort rather than count as a navigation
failure.

## Pairing and execution

- exactly two arms per history in one persistent MemNav/NavDP server pair;
- arm order alternates by frozen history-index parity;
- exact Goal-A RGB, pose, and memory replay must match;
- first accepted anchor, raw vector, bearing, and certificate must match;
- after the first different controller radius, physical trajectories may
  diverge and are not forced equal;
- no Novel, native, ViNT, MP3D, Goal-A collection, threshold sweep, or radius
  sweep is included.

Total closed-loop work: 28 histories x 2 arms = 56 Revisit rollouts.

## Outcomes

Primary: paired position success within 1 m and 600 steps.

Report:

- fixed/full SR;
- paired gain/loss, exact McNemar, and scene-cluster bootstrap interval;
- SPL;
- path length and steps among both-success pairs;
- histories and planning steps whose metric radius differs from 2.5 m;
- first-40 scale availability and 10 m support-bound activations.

## Frozen decision rule

- paired net gain `>=2`, losses `<=1`, and zero 10 m cap hits: authorize one
  independent Revisit-only replication;
- any 10 m cap hit: stop and retain fixed 2.5 m, because the treatment is no
  longer an untruncated full-metric test;
- paired net `<=-2`: reject full metric distance;
- otherwise keep fixed 2.5 m;
- an SR tie may retain metric distance only as an efficiency ablation when
  both-success path length or steps improve by at least 10% with zero losses;
  efficiency alone does not replace the paper method.

Because canonical CEC is already near ceiling on this consumed population,
this gate is not powered to prove superiority.  Its purpose is to stop an
uninformative branch or justify a separate confirmation.
