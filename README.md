# GEM: Geometric Episodic Memory for Image-Goal Navigation

GEM reuses a frozen streaming geometry model to remember observed places across
navigation goals. The model's causal state and an archive of images and geometry
connect a goal image to the current camera. Geometric verification determines
whether the resulting direction is passed to the frozen controller.

For NavDP, the same RGB stream supplies current monocular depth and historical
goal guidance; the original goal image and controller weights are retained.
ViNT and NoMaD use the accepted historical image through their RGB interfaces.
The earlier **CEC** name remains in the bearing interface and deployment config.

```text
Observed RGB --> streaming geometry --> current depth --> frozen NavDP
                         |
                         +--> image / descriptor / pose / geometry archive
                                                     |
Goal image --> retrieval --> geometric verification --> current goal bearing
```

## Start here

- [Current project status](MemNavData/STATUS_20260917_GIT_SYNC.md)
- [Repository and implementation guide](docs/REPOSITORY_GUIDE.md)
- [Experiment and evidence index](docs/EXPERIMENT_INDEX.md)
- [Memory implementation](NavDP/baselines/memnav/gem/README.md)
- [One local workspace](WORKSPACE.md) and [data recovery](docs/LOCAL_STORAGE.md)
- [Consolidation checks](docs/VALIDATION_20260917.md)

The navigation repositories at [AlanZhu2006/Nav](https://github.com/AlanZhu2006/Nav)
and [glbreeze/Nav](https://github.com/glbreeze/Nav) share this release on `main`.

## Implementation

| Path | Role |
| --- | --- |
| `NavDP/baselines/memnav/gem/` | Causal writes, archived geometry, goal sessions, dense depth and sparse goal readout |
| `NavDP/baselines/memnav/policy_agent.py` | Agent integration and compatibility interfaces |
| `NavDP/baselines/memnav/memnav_server.py` | HTTP requests and frame-bound geometry receipts |
| `MemNavData/certified_relocalization_runtime.py` | Geometric acceptance rules and bearing contract |
| `MemNavData/image_controller_goal_adapter.py` | ViNT/NoMaD goal and heading interface |
| `MemNavData/` | Evaluation runners, reducers, protocols and dated research records |

Compatible server defaults remain `legacy / dense / native`. Storage studies
explicitly select `native_interval7`, compact historical geometry and optional
lossless KV encoding. Use each experiment's recorded configuration and source
revision. The [memory guide](NavDP/baselines/memnav/gem/README.md) explains the
configurations and their evidence.

## Evaluation

The manuscript reports paired Revisit and Novel queries on HM3D and MP3D with
three frozen controllers, continuous goal sequences, geometric and control
ablations, and real-robot revisits. The heading study retains the verified
53-query reporting cutoff. The original 13-query low-recent-overlap subgroup
has a separate definition and remains separately identified.

See the [experiment index](docs/EXPERIMENT_INDEX.md) for populations, protocols
and verification tools. Dated exploratory studies remain available, including
unsuccessful alternatives; committing their source does not change their status.

## Related repositories and local data

- [Paper](https://github.com/AlanZhu2006/Memnav_Paper): current `main.tex` and
  manuscript, optionally checked out at `projects/paper`.
- [Real-world deployment](https://github.com/AlanZhu2006/MemNav-RealWorld):
  Jetson/RTX services, Survey preparation and Go2 execution, optionally checked
  out at `projects/realworld`.

These are independent Git repositories. The navigation repository excludes their
working directories, model weights, scenes, runtime recordings and large arrays.
The old local `paper/` directory is not the active manuscript. See
[WORKSPACE.md](WORKSPACE.md) for the editor layout and
[LOCAL_STORAGE.md](docs/LOCAL_STORAGE.md) for archived data and scene recovery.
