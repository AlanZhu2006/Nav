# External comparison matrix (frozen before paper outcomes)

Date: 2026-08-14 (Asia/Shanghai)

## Comparison principle

No published SR is directly subtracted from ours unless goal images, sensors,
action space, episode population, step budget, success radius and history
contract are identical.  Comparisons are separated into controller-matched
baselines, standard localization systems, and prior navigation systems.

## Tier A — mandatory controller-matched comparison

Every arm replays the byte-identical online Goal-A history and uses the same
frozen NavDP controller:

| Arm | Question answered |
| --- | --- |
| native | What happens without episodic memory? |
| raw metric DINO | Is simple always-on retrieval already sufficient? |
| raw fixed-bearing DINO | Is any gain merely removal of scale error? |
| geometry fixed | Does the older DINO + SIFT/RANSAC gate suffice? |
| certified | Does learned matching plus an atomic PnP certificate improve utility while preserving exact fallback? |
| known-role direct (upper bound only) | What is lost because the runtime is not told Novel versus Revisit? |

These are the primary statistical comparisons because controller, episodes and
execution are paired exactly.

## Tier B — standard localization-system baselines

### HLoc-style localization

HLoc is not a wholly different navigation method.  It shares retrieval,
SuperPoint/LightGlue and PnP with our method.  Its distinguishing assumption is
a multi-view SfM map with globally consistent 3D points, while our method uses
online causal history plus LingBot monocular depth and a pairwise certificate.

The fair baseline is therefore:

`NetVLAD retrieval -> SuperPoint/LightGlue -> online-history SfM/PnP -> fixed
2.5 m bearing -> identical frozen NavDP -> exact-native fallback on failure`.

It is a secondary but useful baseline: it tests whether LingBot and the atomic
certificate offer anything beyond a standard localization stack.  It is not a
replacement for a full prior navigation method.

Official HLoc was pinned locally before paper outcomes at commit
`c13273bd0ecc2917a35910fd843712a1c6243193`.  Its current package uses
`pycolmap`, so a separate system COLMAP executable is not a prerequisite, but
the baseline still has to build an online-history SfM model and must not use
future frames.  It consumes the same stored history/query images and therefore
requires no additional navigation dataset download.
The frozen causal-history reconstruction smoke has passed: 19/30 online
decision frames registered and 722 3D points were reconstructed without
reading pose, depth, role or query fields.  Query localization and closed-loop
SR are still untested, and the final online-A frame was not registered in the
largest component.  The endpoint must therefore be localized under the same
certificate or the arm abstains.  This is readiness evidence rather than a
baseline performance result.
The exact causal construction, acceptance and controller contract is frozen in
`HLOC_ONLINE_HISTORY_BASELINE_PROTOCOL_20260814.md`.

### RNR-Map (CVPR 2023 Highlight)

RNR-Map is a stronger representation-level comparison.  It builds a renderable
map from posed RGB-D history and localizes seen or nearby novel-view queries in
that map.  Official code and pretrained components exist, but its Habitat 0.2.1
and PyTorch 1.12 environment should first pass an isolated localization smoke.
If promoted, its localized bearing is passed to the same fixed NavDP executor.

## Tier C — full prior navigation systems

| Method | Relevance | Reproduction status | Fair use |
| --- | --- | --- | --- |
| NRNS (NeurIPS 2021) | MP3D/Gibson ImageGoal; topological map plus learned target direction | Official code, checkpoints and episodes; old Python/Habitat stack | Reproduce official scores first; any sequential-history extension reported separately |
| VGM (ICCV 2021) | Online visual graph memory | Official code and checkpoint | Official single-goal benchmark; panoramic RGB contract differs from monocular NavDP |
| TSGM (CoRL 2022 oral) | Online topological semantic graph memory | Official code/checkpoints; RGB-D panorama and detector | Useful full-system reference, not a direct sensor-matched result |
| MemoNav (CVPR 2024) | The closest published multi-goal memory problem | Public 1/2/3/4-goal datasets, but upstream repo still lacks training/evaluation code | External episode benchmark after reconstructing and freezing its goal-render contract; published scores are not yet a direct comparator |
| IGL-Nav (ICCV 2025) | Incremental 3DGS localization for ImageGoal | Official repository says code is coming soon | Related work only until executable code is released |

## Dataset readiness

- Current MP3D role-pair benchmark: ready; final outcomes remain sealed until
  both consumed-scene gates pass.
- Replica v1: all archive parts are local; full 18-scene extraction and the
  simulator/sensor compatibility gate precede policy execution.
- MemoNav MP3D: 18 public test scenes and 1008 episodes per 1/2/3-goal level.
  Eleven scene assets were present locally at first audit.  After explicit user
  confirmation of the MP3D Terms of Use, the official Habitat archive was
  downloaded, integrity-tested and extracted: 90 scene directories, 90 GLBs
  and 90 navmeshes.  This resolves asset availability but does not by itself
  make MemoNav's episode contract equivalent to ours.
- HM3D/UniGoal/Habitat 2023 InstanceImageNav: different instance-image task and
  licensed HM3D assets; follow-up, not a dependency of the current paper run.

## Execution order

1. Finish the two frozen four-arm readiness gates.
2. Run the five controller-matched arms once on the sealed MP3D population.
3. Complete Replica compatibility and start the cross-dataset sequential run.
4. Reproduce NRNS in its official environment and run an RNR-Map localization
   smoke on our already-built histories.
5. Promote HLoc/RNR-Map to full closed-loop only if their localization output
   can be converted without oracle pose or result-dependent tuning.

Primary sources:

- NRNS: https://github.com/meera1hahn/NRNS
- HLoc: https://github.com/cvg/Hierarchical-Localization
- VGM: https://github.com/rllab-snu/Visual-Graph-Memory
- TSGM: https://github.com/rllab-snu/TopologicalSemanticGraphMemory
- MemoNav: https://github.com/ZJULiHongxin/MemoNav
- RNR-Map: https://github.com/rllab-snu/RNR-Map
- IGL-Nav: https://github.com/GWxuan/IGL-Nav
