# Certified Episodic Compass

This branch is the paper-oriented release of a research project on **causal episodic relocalization for sequential ImageGoal navigation**.

The central question is not whether visual memory can always steer a robot. It is:

> When should a hypothesis retrieved from the robot's own visual history be allowed to change the actions of a frozen ImageGoal policy?

Certified Episodic Compass (CEC) treats recall as open-set hypothesis testing. A causal history proposes a previously observed place; explicit geometric evidence either authorizes a scale-free bearing or abstains. Rejected and malformed hypotheses preserve the exact native NavDP action path.

```text
ImageGoal + causal online RGB history
        |
        v
temporally diverse DINO top-8
        |
        v
SuperPoint + LightGlue + Fundamental geometry
        |
        v
LingBot historical depth + PnP
        |
        v
atomic certificate
  | accepted                         | rejected / error
  v                                  v
unit bearing -> fixed 2.5 m          exact native ImageGoal NavDP
residual -> frozen NavDP
```

## Headline results

| Evaluation | Native | CEC | Paired result | Evidence status |
|---|---:|---:|---:|---|
| Fresh MP3D mixed-role, Revisit | 4/21 | 20/21 | +16/-0, p=3.05e-5 | fresh, independently verified; history target underpowered |
| Fresh MP3D mixed-role, Novel | 7/21 | 8/21 | +1/-0, p=1.0 | open-set safety diagnostic |
| Fresh MP3D mixed-role, all queries | raw 21/42 | CEC 28/42 | +8/-1, p=0.0391 | first significant CEC-vs-always-on-memory result |
| Held-out HM3D Revisit, B given A | 7/21 | 19/21 | +12/-0, p=0.000488 | external dataset transfer |
| Actual-online MP3D Novel-Novel-Revisit | 5/19 | 16/19 | +11/-0, p=0.000977 | strong internal continual result |

The primary contribution is the **utility-interference trade-off** created by proof-before-control. The data do not support claims that CEC formally guarantees safety, that it significantly raises the saturated Revisit ceiling over every retrieval baseline, or that the learned Pi3X proof has replaced the explicit certificate.

## Repository map

- [`paper/PROJECT_LEDGER_ZH.md`](paper/PROJECT_LEDGER_ZH.md): complete Chinese project ledger, including positive and negative results.
- [`paper/RESULTS.md`](paper/RESULTS.md): paper-facing quantitative tables and evidence grades.
- [`paper/CODE_MAP.md`](paper/CODE_MAP.md): exact primary runtime, evaluation, and secondary-code boundaries.
- [`paper/mainline_manifest.json`](paper/mainline_manifest.json): machine-readable source boundary.
- [`paper/REPRODUCIBILITY.md`](paper/REPRODUCIBILITY.md): dependencies, assets, tests, and result verification.
- [`paper/configs/cec_v1.json`](paper/configs/cec_v1.json): frozen paper method contract.
- [`paper/results`](paper/results): sealed summary JSON and independent verification receipts.
- [`MemNavData/certified_relocalization_runtime.py`](MemNavData/certified_relocalization_runtime.py): deterministic geometric proposal/certificate boundary.
- [`MemNavData/revisit_bearing_adapter.py`](MemNavData/revisit_bearing_adapter.py): scale-free controller interface and exact abstention contract.
- [`NavDP/baselines/memnav/policy_agent.py`](NavDP/baselines/memnav/policy_agent.py): deployed streaming integration.

## Verify the release

The lightweight verifier needs only Python 3.10+ and checks result hashes, independent-verifier receipts, headline counts, and the frozen constants in source:

```bash
python paper/verify_release.py
```

Pure contract tests can be run without Habitat, NavDP weights, or MP3D/HM3D assets:

```bash
python -m pytest -q \
  MemNavData/test_certified_relocalization_runtime.py \
  MemNavData/test_revisit_bearing_adapter.py
```

Full closed-loop reproduction additionally requires licensed MP3D/HM3D assets, the frozen NavDP and MemNav checkpoints, LingBot-Map, and the pinned LightGlue checkout. Weights and licensed scene assets are intentionally not committed.

## Scope

The validated task is **monocular sequential ImageGoal navigation with episodic revisits**. Lifelong navigation is motivation, not a completed claim. Novel goals remain under the frozen native policy when memory cannot self-authorize.

Project status frozen on 2026-08-18.
