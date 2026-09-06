# Certified Episodic Compass for Monocular Continual ImageGoal Navigation

This repository extends a frozen NavDP ImageGoal controller with a persistent,
causal visual memory. The current method is **Certified Episodic Compass
(CEC)**: history may influence control only after a runtime geometric witness
passes an atomic certificate; otherwise execution falls back to the same
native NavDP policy.

The system is monocular at the navigation-policy boundary:

```text
causal RGB observations
  +-- RGB archive + DINO descriptors --> historical retrieval <-- ImageGoal
  +-- frozen LingBot streaming geometry
        +-- current monocular depth --> frozen NavDP observation
        +-- historical depth/poses --> retrieved-candidate geometric witness
                                         |
                          CEC accepts: unit bearing --> 2.5 m PointGoal
                          CEC rejects: native request unchanged

frozen NavDP: current RGB + estimated depth + original ImageGoal
             + optional directional PointGoal --> trajectory
```

The memory is neither a classical explicit map nor only a LingBot KV cache.
It combines immutable causal RGB frames, per-frame DINO retrieval keys,
LingBot streaming pose/depth state, and on-demand geometric certification.
CEC exposes only a scale-free direction; it does not provide a global map or
path to NavDP.

## Start here

- [最新项目总账 / current state](MemNavData/STATUS_20260907_MAIN_SYNC.md)
- [代码、数据与仓库边界 / repository guide](docs/REPOSITORY_GUIDE.md)
- [论文实验与探索分支索引 / experiment index](docs/EXPERIMENT_INDEX.md)
- [Current architecture and interfaces](MemNavData/CEC_CANONICAL_ARCHITECTURE_20260901.md)
- [Table III exact-SPL replay protocol](MemNavData/FINAL14_TABLE3_EXACT_SPL_REPLAY_PROTOCOL_20260907.md)
- [Shared SSH and Slurm operations](MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md)

The primary method is training-free: its components are pretrained and frozen.
Learned relation readers, route-tangent extensions, and oracle diagnostics are
separate research branches, not part of the validated default method.

## Evidence at a glance

| Experiment | Scope | Main observation |
|---|---|---|
| Controller transfer | HM3D/MP3D × NavDP/ViNT | Paired Revisit gains in all four groups |
| Full-mono composition | Actual mono Goal-A and mixed-role queries | Native 17/56; CEC 32/56 |
| Continual navigation | Third goal after a shared successful A+B prefix | Revisit 8/20 → 17/20; Novel 4/20 → 4/20 |
| Query-depth ablation | Final14; shared metric-A history | Mono CEC 28/42; metric CEC 26/42; exact SPL replay submitted |
| HM3D authority ablation | Retrospective four-arm comparison | CEC 32/56; raw memory 35/56; no significant CEC-over-raw SR gain |

These rows use different populations and denominators. See the
[experiment index](docs/EXPERIMENT_INDEX.md) for paired statistics, provenance,
and limitations. Long-range spatial navigation remains unresolved.

## Main implementation

- `NavDP/baselines/memnav/policy_agent.py`: causal memory, retrieval,
  relocalization, certificate cache, bearing and monocular depth state.
- `NavDP/baselines/memnav/memnav_server.py`: runtime HTTP boundary.
- `MemNavData/certified_relocalization_runtime.py`: certificate and scale-free
  geometry contract.
- `MemNavData/monocular_depth_runtime.py`: causal first-40 monocular scale
  transaction.
- `MemNavData/realworld_cec_hub.py`: research-side recording/query control
  plane; the authoritative Jetson/RTX deployment is maintained in the
  separate `AlanZhu2006/Memnav_Realworld` repository.
- `MemNavData/realworld_visual_convergence_contract.py`: fail-closed,
  shadow-only scale-free arrival evidence contract.

No model weights, scene assets, generated rollouts, local diagnostics, or
robot credentials are stored in this repository.

The active paper is maintained separately in `AlanZhu2006/Memnav_Paper`
(`/home/asus/Research/Memnav_Paper` on the development machine). The ignored
`paper/` directory here is a legacy local copy, not the Overleaf source of truth.
Real-robot deployment is maintained in `AlanZhu2006/Memnav_Realworld` and is
outside this release's execution scope.
