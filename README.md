# Certified Episodic Compass for Monocular Continual ImageGoal Navigation

This branch extends a frozen NavDP ImageGoal controller with a persistent,
causal visual memory. The current method is **Certified Episodic Compass
(CEC)**: history may influence control only after a runtime geometric witness
passes an atomic certificate; otherwise execution falls back to the same
native NavDP policy.

The system is monocular at the navigation-policy boundary:

```text
one causal RGB stream
  -> frozen LingBot streaming geometry
       -> short-range monocular depth -> frozen NavDP
       -> long-range episodic retrieval + CEC -> bearing or abstain
  -> one frozen NavDP trajectory policy
```

The memory is neither a classical explicit map nor only a LingBot KV cache.
It combines immutable causal RGB frames, per-frame DINO retrieval keys,
LingBot streaming pose/depth state, and on-demand geometric certification.
CEC exposes only a scale-free direction; it does not provide a global map or
path to NavDP.

## Start here

- [Current workspace and release source of truth](MemNavData/STATUS_20260828_WORKSPACE_MAIN_RELEASE.md)
- [Previous full project ledger](MemNavData/STATUS_20260825_GIT_RELEASE.md)
- [Architecture and claim boundaries](MemNavData/ARCHITECTURE_20260819_PAPER_SOURCE_OF_TRUTH.md)
- [Final14 formal result](MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md)
- [Fresh HM3D protocol](MemNavData/HM3D_FRESH_FULLMONO_MIXED_ROLE_PROTOCOL_20260820.md)
- [Real-world deployment guide](MemNavData/REALWORLD_GO2_FULLMONO_DEPLOYMENT_20260820.md)
- [Scale-free real-world arrival calibration](MemNavData/REALWORLD_SCALE_FREE_ARRIVAL_CALIBRATION_PROTOCOL_20260825.md)
- [HPC operational hardening](MemNavData/HPC_HARDENING_20260821.md)
- [Shared SSH and Slurm operations](MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md)

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
