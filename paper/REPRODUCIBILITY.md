# Reproducibility guide

## What is included

- the deterministic CEC proof and bearing contracts;
- the deployed MemNav integration used by the formal experiments;
- MP3D Final14 population/evaluation/summary code;
- HM3D external summary and independent verification code;
- sealed summary JSON and independent-verifier receipts;
- the secondary Pi3X learned relocalizer implementation and tests.

## What is not included

- MP3D and HM3D scene assets, which require their respective licenses;
- NavDP, MemNav, LingBot-Map, Pi3X and SuperPoint/LightGlue model weights;
- raw RGB/depth traces and large intermediate feature caches;
- HPC container images and user-specific scratch paths.

No checkpoint or licensed scene file is required to verify the published result receipts.

## Lightweight verification

```bash
python paper/verify_release.py
```

The verifier:

1. recomputes SHA-256 for each sealed summary;
2. checks the independent-verification receipt;
3. checks headline counts and paired statistics;
4. parses source constants and compares them with `configs/cec_v1.json`;
5. confirms that learned Pi3X was not promoted to the primary method.

## Contract tests

Install the lightweight test dependencies:

```bash
python -m pip install numpy opencv-python pytest
```

Run the deterministic contracts:

```bash
python -m pytest -q \
  MemNavData/test_certified_relocalization_runtime.py \
  MemNavData/test_lingbot_pnp_localization.py \
  MemNavData/test_revisit_bearing_adapter.py \
  MemNavData/test_shared_online_role_pair_contract.py
```

The Final14 construction test imports Habitat rendering primitives and must run in the Habitat environment:

```bash
python -m unittest MemNavData/test_final14_role_pair_construction.py
```

Tests involving Pi3X, LingBot-Map, NavDP inference, Habitat rendering, or formal HPC wrappers require the corresponding external repositories, weights and environments.

## Full runtime dependencies

The formal stack used two processes:

1. a Habitat process for rendering, navigation metrics and action execution;
2. a MemNav/NavDP process for DINO retrieval, LightGlue geometry, LingBot depth, PnP and diffusion-policy inference.

Required external components:

- Habitat-Sim/Habitat-Lab compatible with the scene assets;
- NavDP and InternNav/MemNav checkpoints;
- LingBot-Map checkout and weights;
- pinned official LightGlue/SuperPoint checkout with 2048 keypoints;
- CUDA/PyTorch versions compatible with the frozen checkpoints.

The server is enabled with `--certified_relocalization`; the evaluator must use `certified_relocalization` plus `verified_bearing_v1`. Do not enable CDEC, graph rescue, Pi3X or X-NavDP in a CEC-v1 reproduction.

## Experimental discipline

- Do not pool absolute SR across different machines or CUDA runs; compare paired arms in one process lineage.
- Do not add Final14 histories after observing outcomes; its 21-history Natural population remains underpowered relative to the frozen target.
- Do not use development scenes or role labels for runtime selection.
- Keep Novel and Revisit results separate before reporting role-balanced aggregates.
- Report scene-cluster uncertainty in addition to query-level McNemar statistics.
- Preserve failed construction as attrition; do not replace scenes after policy outcomes.

## Release artifacts

`paper/results` contains compact result summaries, not raw trajectories. Each independent verifier states which raw files it originally re-read. The release verifier ensures those summaries are byte-identical to the verified artifacts.
