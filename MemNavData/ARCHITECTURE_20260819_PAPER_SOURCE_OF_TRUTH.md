# 当前论文架构与证据边界（2026-08-19）

状态：**2026-08-20 23:00 mono-primary 更新后的论文/代码 source of truth**。本文只描述已经实现并审计过的架构；
正式论文数字仍以 `paper/EVIDENCE_LEDGER.md` 及各冻结结果文档为准。

## 1. 一句话主线

本项目不是“再训练一个更强的导航策略”，也不是“DINO + LightGlue + PnP 的工程堆叠”。
它研究的是一个更窄但可检验的问题：

> 单个因果 RGB 流如何同时支撑短程控制与长程记忆，并且只让可自认证的历史影响冻结策略？

CEC 把答案实现成 `proposal -> witness -> authority -> narrow control interface`：历史先提出
候选，几何证据对候选做一次运行时自认证，认证成功只输出 scale-free bearing，失败或异常则
逐动作回到同一状态、同一 seed 的原生 NavDP。

## 2. 长程分支：CEC（已确认）

```text
actual-online causal RGB history + current ImageGoal
        |
        v
DINOv2 temporally-diverse top-8 proposal
        |
        v
SuperPoint + LightGlue correspondences
        |
        v
Fundamental-MAGSAC support/ranking
        |
        v
LingBot historical reference depth + PnP-RANSAC
        |
        v
atomic certificate
  inliers >= 16
  query/ref hull coverage >= 5%
  reprojection RMSE <= 2 px
        |
        +-- reject/error --> exact native ImageGoal NavDP
        |
        `-- accept -------> unit [forward,left] bearing
                            x fixed 2.5 m residual
                            + original goal image
                            --> frozen NavDP
```

边界必须保持清楚：

- DINO 只负责地址提议，不直接授权控制；
- certificate reject 只表示“当前历史证据不足”，不等于语义 Novel 分类；
- 不把 LingBot 的单目 translation norm 当 metric distance；
- NavDP 仍是唯一生成轨迹和动作的 policy；
- Final14、actual-online NNR 和较大的 HM3D Revisit 是“固定 metric controller depth、只改变
  memory authority”的受控实验，不能改写成 end-to-end mono；actual-online Full-Mono HM3D
  integration 才是 history 与 controller 全程只使用 causal RGB-derived depth 的组合证据。

## 3. 当前主架构：同一因果流、两个时间尺度、一个冻结策略

当前实现把同一个 frozen LingBot streaming state 暴露为两个 readout：

```text
causal monocular RGB stream
        |
        v
one frozen LingBot streaming geometry state
        |
        +-- dense / short-range -------------------------------+
        |   frames 0..39: zero depth                           |
        |   replay exactly first 40 frames once                |
        |   predicted ground height + known camera height      |
        |   -> immutable scale receipt                         |
        |   frame >= 40: relative depth x frozen scale         |
        |   -> unchanged NavDP RGB-D observation encoder       |
        |                                                      |
        +-- sparse / long-range ---------------------------+    |
            CEC retrieval + geometric proof              |    |
            -> certified bearing or abstain               |    |
                                                         v    v
                                  one frozen NavDP goal encoder / decoder / critic
                                                         |
                                                         v
                                                   one trajectory
```

这是论文与实机当前统一的主方法，而不再是图中的虚线 deployment extension。它是“共享几何
状态的双时间尺度读出”，不是两个 expert 各自投票，也不是第二个 planner。
它比 Geometry Token Adapter 更简单：Gate C 已经否决 learned token bridge，直接 raw depth
反而更容易被冻结 NavDP 消费。

严格单目 depth 合约：

1. first 40 decisions 固定输出 zero depth；
2. 只重放实际执行的 causal frames `0..39` 一次；
3. scale 只来自 LingBot predicted ground height 与已知相机安装高度；
4. scale receipt 此后冻结，invalid 时 fail closed 到 zero；
5. JPEG、depth payload、receipt 以 SHA-256 绑定；
6. replay 前后 snapshot/restore streaming state；
7. 禁止 future frame、pooled/oracle scale、Habitat pose/depth 和第二个 LingBot stream。

## 4. 为什么主张不是“Mixture of Experts”

“MoE”会让人以为 LingBot 与 NavDP 都在生成候选动作并由 gate 选择。当前系统并不是这样：

- LingBot/CEC 输出的是证据与一个二维方向；
- NavDP 才输出轨迹；
- authorization 决定额外方向是否有权进入 NavDP，而不是在两个 controller 中选一个。

因此论文最准确的概念是 **proof-before-control / open-set action authorization**。单目扩展
的架构短语是 **one causal stream, two time scales, one frozen policy**。

## 5. 当前证据如何支撑架构

### 5.1 CEC 主结果

- Final14 mixed-role：CEC `28/42`，raw fixed `21/42`，paired `+8/-1`，`p=.0391`；
- Final14 Revisit：native `4/21` -> CEC `20/21`，`+16/-0`；
- Natural Novel：CEC `19/21` reject 且 exact fallback，另有 2 次 takeover，不能写成零误激活；
- actual-online NNR：Goal-C `5/19 -> 16/19`，`+11/-0`；
- HM3D Revisit：`7/21 -> 19/21`，`+12/-0`。

CEC 相对 geometry fixed 只是 `+2/-1, p=1`；因此贡献不是“更强 matcher”，而是统一、可
审计的 authority/fallback contract。CEC 相对 raw 的显著优势来自控制 Novel interference，
不是继续提高已接近饱和的 Revisit ceiling。

### 5.2 单目控制器 Gate C / D

- Gate C：raw depth 的 token error `.1812`、epsilon MSE `.00591`、critic top-1 `.7266`；
  zero 对应 `.3024/.01152/.6016`；6.02M adapter 在五项冻结指标上均劣于 zero；
- Gate D：metric `30/40`，raw mono `27/40`，zero `23/40`；
- raw vs metric `+2/-5`，RD `-7.5 pp`，CI `[-20,+5]`；
- raw vs zero `+6/-2`，RD `+10 pp`，CI `[-2.5,+22.5]`；
- 40/40 raw episodes 无 simulator metric-depth 消费，scale receipt 有效且被实际使用。

结论：Gate D 只通过 engineering continuation gate，没有通过 10 pp non-inferiority。单目
可以作为当前主架构，但不能宣称它与 metric controller 等价，也不能把下方受控的
metric-depth CEC 数字改写成 end-to-end mono 数字。

### 5.3 已完成的 actual-online Full-Mono 组合

在 reused-scene HM3D mixed-role 上：mono native `2/16`，mono raw fixed `11/16`，mono CEC
`9/16`。CEC 对 native 为 `+7/-0, p=.015625`；Revisit 为 `0/8 -> 7/8`。CEC accept
`8/8` Revisit、`0/8` Novel，全部 Novel rejection exact fallback。1,231 个 Goal-A 与
1,809 个 query plan receipt 均确认无 simulator metric-depth read。

这建立完整 RGB-only integration，但场景已被旧 HM3D 实验消费，且 CEC 没有超过 raw
fixed。Fresh 54-scene reserve confirmation 已按固定 prefix rule 提交；完成前不能写成
fresh-scene 结果。

## 6. 精准代码入口

| 责任 | 当前入口 |
|---|---|
| MemNav server 与配置 | `NavDP/baselines/memnav/memnav_server.py` |
| CEC proposal/witness/cache/lifecycle | `NavDP/baselines/memnav/policy_agent.py` |
| certificate 与 bearing 数学合约 | `MemNavData/certified_relocalization_runtime.py` |
| first-40 scale 与 raw mono depth 合约 | `MemNavData/monocular_depth_runtime.py` |
| Gate D evaluator | `MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py` |
| Gate D summary/verifier | `MemNavData/summarize_mdtec_raw_depth_gate_d.py`, `MemNavData/independent_verify_mdtec_raw_depth_gate_d.py` |
| CEC+mono evaluator | `MemNavData/eval_mdtec_monocular_cec_composition_habitat.py` |
| CEC+mono summary/verifier | `MemNavData/summarize_mdtec_monocular_cec_composition.py`, `MemNavData/independent_verify_mdtec_monocular_cec_composition.py` |
| 论文主张总账 | `paper/EVIDENCE_LEDGER.md` |
| 当前论文源码 | `paper/main.tex`, `paper/sec/`, `paper/tables/` |

## 7. 当前禁止的叙述

- 不写 CEC formal safety / zero Novel takeover；
- 不写 CEC 在 Revisit 上显著超过 raw 或 geometry；
- 不写 learned Pi3X 已替代 explicit proof；
- 不写 GOAT 验证了本方法；
- 不写所有 headline CEC 数字都是 end-to-end monocular；Final14/NNR/大 HM3D 是 metric-depth
  固定的 memory-isolation 证据；
- 不写 raw mono 已与 metric RGB-D non-inferior；
- 不把正在运行的 partial CEC+mono 计数写进论文。
- 不写真机“完全没有 depth sensor”；导航 policy 单目，但 Jetson 本地 depth safety 仍保留。

## 8. 论文当前最强 storytelling

1. **问题**：单目持续导航同时需要逐步局部几何和跨目标长期记忆；历史既有 utility，也有
   interference。
2. **方法**：一个 frozen causal geometry state 提供 dense short-range depth 与 sparse
   proof-before-control bearing，最后只有一个 frozen NavDP 生成轨迹。
3. **证据分解**：Gate D 隔离 depth、Final14 隔离 authority、Full-Mono HM3D 验证两者组合；
   不跨 population 拼接效果。
4. **主要发现**：raw memory 已接近 Revisit ceiling，CEC 的显著价值是保留 Revisit utility
   同时控制 Novel harm；mono depth 有用但尚未达到 metric non-inferiority。
5. **机制结果**：learned proposal 或 learned proof 的高 AUC/SR 不能代替可靠几何授权。

这个叙事比“LingBot + NavDP 的工程拼接”更准确，也比声称一个尚未成立的统一 learned
relocalizer 更稳：创新点位于 **证据如何获得控制权**，而不是每个现成 backbone 的名字。
