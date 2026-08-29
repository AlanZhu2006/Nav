# Final14 CEC proposal--witness--authority mechanism audit

审计日期：2026-08-30（Asia/Shanghai）。这是 consumed Final14 上的 post-hoc 机制归因，
不是 fresh confirmation。所有 navigation thresholds 均保持冻结，运行时 role 隐藏。

## 1. 为什么做这次统一复算

已有结果分别回答了 retrieval、certificate precision 和 closed-loop SR，但过去的表格会
混用 metric/mono query depth，或把 raw-DINO 与 proposal-matched authority ablation 当成
同一种对照。本审计把同一 21-history / 10-scene / 42-query population 上的三层证据
严格分开：

1. proposal 是否包含有历史支持的地址；
2. 同一个 proposal 是否能产生有限 PnP witness；
3. witness 是否通过 frozen operational certificate，因而有权改变控制。

`covis >= 0.50` 只作历史地址覆盖诊断，不是运行时 threshold，也不被称为 localization
accuracy。Novel 构造要求 max history covis `<0.10`；Revisit 构造要求至少 `0.55`。

## 2. Proposal diagnostics

| role | DINO top-1 supported | DINO top-8 contains support | geometry-selected supported |
|---|---:|---:|---:|
| Novel | 0/21 | 0/21 | 0/21 |
| Revisit | 19/21 | **21/21** | 19/21 |

在 supported Revisit 上，top-8 完成 21/21 address coverage，而 top-1 只有 19/21。
几何选择的 anchor 分布于 DINO rank 1--8：`5/4/0/3/3/1/2/3`。因此 top-8 的作用是
提供短名单召回；不能把 DINO top-1 与 CEC 的 temporally-diverse top-8 视为完全相同的
proposal contract。

## 3. Operational ladder and closed loop

全部 query controller depth 都是 causal monocular sidecar。raw-DINO 使用 direct-top1
contract；finite-PnP 与 strict CEC 则逐 query proposal matched，只改变 authority rule。

| arm | Novel auth. | Revisit auth. | Novel SR | Revisit SR | All SR |
|---|---:|---:|---:|---:|---:|
| raw DINO, always on | 21/21 | 21/21 | 3/21 | 20/21 | 23/42 |
| matched finite-PnP witness | 18/21 | 21/21 | 5/21 | 20/21 | 25/42 |
| strict CEC | **2/21** | **21/21** | **8/21** | **20/21** | **28/42** |

关键归因是：有限 PnP pose 并不足以授权控制。即使 21 条 Novel 的真实历史支持都低于
0.10，finite-PnP 仍授权 18 条；完整 certificate 把它压到 2 条，同时没有减少任何
Revisit authorization。

闭环边界必须诚实保留：

- strict CEC vs proposal-matched finite-PnP：`28/42 vs 25/42`，paired `+4/-1`，
  risk difference `+7.14 pp`，exact McNemar `p=0.375`；
- strict CEC vs mono raw-DINO：`28/42 vs 23/42`，paired `+5/-0`，
  risk difference `+11.90 pp`，`p=0.0625`；
- 两个差异均不能写成已确认的 SR superiority；可确认的是 authority 行为发生了预期且
  大幅的变化，并保留了 Revisit utility。

旧 metric-depth raw 对照为 21/42；它不能与本次 mono raw 的 23/42 混写。本次报告优先
使用 23/42，因为它与 authority ablation 的 query depth 条件一致。

## 4. 对论文方法的影响

这组结果把 CEC 的贡献从“DINO + matcher + PnP 的工程串联”收敛为一个可检验的接口：

> proposal 可以高召回地寻找地址，geometric witness 可以产生一个位姿假设，但只有
> operational certificate 才授予历史证据改变 frozen policy 的权力。

因此论文可以主张 proof-before-control / evidence-gated authority，而不能主张形式化
safety guarantee，也不能声称 certificate 已显著超过所有简单 retrieval baseline。

## 5. 可复算资产

- compact raw ledger（远端 POSTHOC）：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_authority_20260828/formal_3f5783aca521b0a5/POSTHOC/final14_cec_mechanism_ledger_20260830.json`；
- ledger SHA-256：
  `4e2abfd2f6781e636260206ae0675813f1b7227e09ab4709742bd8bcc2715277`；
- audit：`FINAL14_CEC_MECHANISM_AUDIT_20260830.json`；
- independent verification：
  `FINAL14_CEC_MECHANISM_AUDIT_VERIFICATION_20260830.json`；
- raw extractor：`extract_final14_cec_mechanism_ledger.py`；
- analyzer/verifier：`analyze_final14_cec_mechanism.py`、
  `independent_verify_final14_cec_mechanism.py`。

独立 verifier 从 42 条 compact records 与两个 sealed summaries 重新计算全部 proposal、
authorization 和 closed-loop counts，结果为 `verified=true`。
