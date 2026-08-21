# External benchmark readiness audit

Date: 2026-08-14 (Asia/Shanghai)

Status: data/interface/constructibility audit complete; no long policy rollout
submitted; no benchmark outcome read.

## 1. Decision

There is no public benchmark found so far that simultaneously preserves all of
the paper's deployed contract:

1. sequential ImageGoals in one persistent episode;
2. causal online RGB history only;
3. mixed Novel and Revisit queries without revealing the role;
4. a monocular forward-facing sensor;
5. closed-loop navigation with the same frozen controller.

Therefore external evaluation must be reported in three non-interchangeable
tiers:

| Tier | Benchmark | What it can establish | What it cannot establish |
| --- | --- | --- | --- |
| Primary external closed loop | GOAT-Bench ImageGoal subtasks | persistent-history operation in an official sequential benchmark | exact equivalence to the current MP3D role-pair population |
| Standard ImageGoal context | NRNS MP3D/Gibson splits | frozen NavDP and published full-system performance on standard single-goal strata | a Revisit-memory gain at task start |
| Localization component | MapFree/7Scenes and causal HLoc | retrieval/localization/certificate accuracy, coverage and abstention | end-to-end CEC SR under its exact LingBot-history contract |

The current paper P0 remains the already-frozen MP3D phase-2 power expansion.
No external long run should consume GPUs before its infrastructure-only
summarizer repair is complete.

## 2. NRNS public ImageGoal splits

### 2.1 Files and schema

The official NRNS test episodes were downloaded to temporary audit storage and
opened without running any policy:

- MP3D archive SHA-256:
  `638328eeca6767b0918576eaaf43d3bac319c38819b0ba2fcb6964fbd10433a5`;
- Gibson archive SHA-256:
  `8505c1356b7204d9ab7227d9ebc4b5f3a5efd83314ddf63c51447e9bcafb2b89`.

Both datasets expose deterministic Habitat episodes with
`scene_id`, `start_position`, `start_rotation`, one goal `position`, one goal
`rotation`, and the reference shortest-path length.  This is materially better
specified than the public MemoNav episode files, whose goal rotation/RGB is
absent.

| Dataset | Scenes | straight easy/medium/hard | curved easy/medium/hard | Total |
| --- | ---: | ---: | ---: | ---: |
| MP3D | 18 | 1000 / 1000 / 1000 | 1000 / 1000 / 1000 | 6000 |
| Gibson | 14 | 1000 / 1000 / 806 | 1000 / 1000 / 1000 | 5806 |

Official source: <https://meerahahn.github.io/nrns/data>

### 2.2 Local constructibility result

All 18 NRNS-MP3D scene pairs are already present locally under
`/home/asus/Research/datasets/mp3d_official_20260814/extracted`:

- `18/18` GLBs found;
- `18/18` navmeshes found.

A no-policy Habitat 0.3.3 smoke loaded the first episode from each of the six
MP3D strata and rendered the official start and goal camera poses.  Results:

- `6/6` starts and goals navigable;
- `6/6` shortest paths found;
- maximum absolute difference from the stored shortest-path length: `0.0 m`;
- `6/6` start/goal RGB hashes differed, confirming that the supplied goal
  rotations are actually consumed by rendering.

This establishes dataset and simulator compatibility.  It is not a navigation
result.

### 2.3 Critical task-contract limitation

NRNS is a standard **single-goal** ImageGoal benchmark.  At the initial decision
time, CEC has no preceding episodic history.  Under the current exact runtime
contract it must therefore reject memory takeover and reduce to native NavDP.

Consequences:

- running `native` versus unchanged `CEC` from `t=0` cannot demonstrate the
  Revisit contribution;
- periodically querying the growing within-leg history would be a new method,
  not the current CEC evaluation;
- concatenating NRNS episodes into multi-goal streams would be a useful custom
  extension, but it must not be described as the official NRNS benchmark.

The fair use of NRNS is consequently:

1. reproduce or cite the official NRNS full-system numbers on the unchanged six
   strata;
2. evaluate frozen NavDP on the same strata as controller context;
3. if a sequential extension is later built, label it explicitly as
   `NRNS-sequential extension` and report it separately.

The official implementation targets Python 3.6 and a legacy Habitat stack, so
NRNS reproduction should use an isolated environment.  Porting its controller
into the current Habitat 0.3.3 environment would no longer be an exact
reproduction without action/sensor parity tests.

### 2.4 Scene-use audit

The 18 MP3D scenes are not a fresh scene pool relative to this project:

| Existing project role | NRNS scenes in that role |
| --- | ---: |
| train40 | 5 |
| development10 | 2 |
| final-reserved4 | 2 |
| consumed20 | 4 |
| strict-blind16 | 5 |

Thus a full NRNS-MP3D run is a standard-benchmark evaluation, not a new
scene-disjoint generalization claim.  It must be frozen before outcomes and
must not be used to tune the certificate.  The readiness smoke rendered one
public episode from each stratum but read no policy output.

### 2.5 Gibson status

The public Gibson split is schema-compatible and supplies all goal rotations,
but the 14 licensed Gibson scene assets are absent locally and on the checked
HPC paths.  Gibson is the stronger cross-dataset option once those assets are
obtained under the Gibson license.  Downloading the small episode JSON alone is
not sufficient.

## 3. MapFree Relocalization / 7Scenes

### 3.1 What is reusable now

The current CEC implementation already has a clean matcher/certificate seam:

- `LightGluePointMatcher.match_paths` accepts arbitrary RGB paths;
- `fundamental_support` provides generic 2-D geometric evidence;
- `correspondence_pnp_localize` consumes matched pixels, reference depth,
  reference confidence and a reference pose;
- `certificate_decision` can evaluate an external PnP payload once the same
  audit fields are populated.

### 3.2 Why this is not exact full CEC

The official MapFree task provides one reference image and one query image.
The deployed CEC depth is not a standalone single-image estimator: LingBot
predicts the reference depth after replaying a causal warm history around the
anchor.  Supplying only the MapFree reference changes that information
contract; supplying additional frames changes the official benchmark.

MapFree/7Scenes can therefore be reported as a **localization-component
benchmark**, not as exact end-to-end CEC.

Two additional adapter requirements were found:

1. MapFree reference and query images can have different camera intrinsics;
   the current helper reconstructs one intrinsic matrix from LingBot `pose9`
   and reuses it for both lifting and query PnP.  A proper adapter must accept
   explicit `K_ref` and `K_query`.
2. Depth variants must be labelled separately: ground-truth RGB-D is an oracle
   component result; PlaneRCNN or another frozen monocular estimate is the
   deployable component result.  They cannot be pooled.

The official metrics to retain are translation error, rotation error, virtual
correspondence reprojection error, precision/AUC at the official thresholds,
and estimate coverage.  Failed estimates must remain failed/abstained rather
than being silently removed.

Official source: <https://github.com/nianticlabs/map-free-reloc>

## 4. Other comparisons

### HLoc

HLoc remains the strongest architecture-matched baseline because it can build
a causal multi-view SfM map from the same online history.  The existing smoke
registered `19/30` decision frames and reconstructed `722` points, but query
localization and closed-loop execution are unfinished.  Completing this
baseline answers a sharper question than adding another unrelated navigation
controller: whether LingBot pairwise depth/PnP offers value beyond classical
online SfM localization.

### RNR-Map

RNR-Map is a useful representation-level comparison but requires its isolated
Habitat/PyTorch environment and an RGB-D/history contract.  It should be a
secondary localization comparison, not a blocker for the paper's paired CEC
result.

### VGM / TSGM / standard Habitat ImageNav

These are useful related full systems, but their panoramic/RGB-D sensors,
single-goal protocol or controller differ.  Published SR can be shown as
context only unless an exact official reproduction is run.  They do not
replace the controller-matched `native/raw/geometry/certified` comparison.

## 5. Frozen execution order

1. Repair the phase-2 summarizer so it reads `episodes_per_scene=4` from the
   immutable manifest; resume the already-collected Goal-A pipeline without
   changing the method.
2. Finish phase-2 paired evaluation.  This decides whether certificate safety
   produces a better utility-risk tradeoff than raw DINO at adequate power.
3. Continue GOAT as the only currently identified official sequential external
   closed-loop benchmark near the method's actual use case.
4. Run a small NRNS-MP3D **NavDP constructibility/controller pilot** on
   train/development-role scenes only.  Do not launch all 6000 episodes merely
   to produce a number that is guaranteed to compare native with itself.
5. Implement and unit-test an explicit-`K_ref`/`K_query` MapFree adapter, then
   run a component-level calibration-free comparison of raw retrieval,
   geometry and certificate.
6. Acquire Gibson scene assets only if a scene-disjoint cross-dataset result is
   still needed after GOAT and phase-2.  Freeze the exact official versus custom
   sequential protocol before any policy outcome is read.

## 6. What is explicitly not authorized by this audit

- no full 6000/5806-episode NRNS rollout;
- no tuning on NRNS, MapFree, final-reserved or strict-blind outcomes;
- no claim that MapFree single-reference depth is the deployed LingBot-history
  method;
- no claim that a custom concatenated NRNS stream is an official NRNS score;
- no cross-paper subtraction of SR when sensors, budgets, stopping rules or
  episode populations differ.

## 7. Primary sources

- NRNS code and reported scores: <https://github.com/meera1hahn/NRNS>
- NRNS public test episodes: <https://meerahahn.github.io/nrns/data>
- MapFree Relocalization: <https://github.com/nianticlabs/map-free-reloc>
- HLoc: <https://github.com/cvg/Hierarchical-Localization>
- RNR-Map: <https://github.com/rllab-snu/RNR-Map>
- VGM: <https://github.com/rllab-snu/Visual-Graph-Memory>
- TSGM: <https://github.com/rllab-snu/TopologicalSemanticGraphMemory>
