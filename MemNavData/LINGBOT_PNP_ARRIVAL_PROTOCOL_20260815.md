# LingBot-PnP arrival certificate: frozen train-only audit

Date: 2026-08-15 (Asia/Shanghai)

## Question

Can the current RGB and ImageGoal certify the official GOAT arrival radius
(`<0.25 m`) when native NavDP proposes a zero trajectory?

The preceding exact-state audit established that NavDP clips every candidate
endpoint below `0.5 m` to zero.  Even unanimous zero candidates therefore
cannot distinguish true arrival from the `(0.25, 0.50] m` near-miss band.  A
zero trajectory remains `abstain/replan`, never `STOP`, unless the independent
image-geometry certificate below passes.

## Frozen population and inputs

- Exact same train40 population as the sealed consensus audit:
  40 scenes, 80 episodes, 939 states, 160 strict-arrival states and 779
  non-arrival states.
- Ground-truth distance is used only by the summarizer after inference.
- Runtime inputs are current RGB, ImageGoal RGB, causal RGB history, routed
  versioned LingBot cache, and the already frozen first-64-frame monocular
  ground-scale record.
- All 80 episodes have an audited route and valid causal scale.  The earliest
  audited state is frame 119, so the 64-frame scale prefix is causal.
- Simulator depth is not consumed by this audit.

## Frozen inference contract

1. The unmodified native sample-0 zero trajectory is the only primary trigger.
2. SuperPoint + LightGlue matches current RGB to goal RGB.
3. Fundamental-MAGSAC uses the existing 1.5 px threshold and the existing
   monotone precheck.
4. LingBot depth is reconstructed with the frozen flow-keyframe lifecycle.
5. PnP uses the existing configuration and v2 atomic certificate:
   inliers >=16, reference/query hull coverage >=5%, reprojection RMSE <=2 px.
6. The frozen causal scale converts the certified relative translation to m.
7. `STOP` is proposed only when the certificate passes and predicted metric
   distance is below one member of the predeclared grid.

Distance grid (m):

`0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50`.

## Frozen decision gate

The primary `native sample-0 zero + certificate + distance` rule passes train
design only if at least one grid point has:

- zero false positives over all 779 non-arrival states;
- at least 20 true positives;
- true positives spanning at least 10 train scenes.

Tie-break: maximum true positives, then maximum positive-scene coverage, then
the smaller distance threshold.  No train result directly authorizes GOAT.
A passing point may only be frozen and tested once on a disjoint GOAT pool,
without retuning.  Failure leaves zero trajectory as abstention only.

An engineering smoke may use the first lexicographic episode only to verify
runtime, memory, output schema, and rough wall time.  It cannot change the
grid, certificate, gate, or tie-break above.
