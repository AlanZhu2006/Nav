# 目标位置后验决策层（GLP：Goal-Location Posterior）设计稿

日期：2026-08-07

工作树：`/home/asus/Research/Nav-graph-blind`

状态：设计稿。Stage 0/1 允许立即实现（纯本地 CPU）；Stage 2/3 依赖 Phase-B
artifact 完成审计。本文不推翻 `NOVEL_MEMORY_RESIDUAL_V2_20260807.md` 的安全
契约，只重构其决策层。

---

## 0. 一句话主张

把 NLSR-V2 的六个独立 head 加十余个阈值的选择状态机，重构为**一个在持久
memory graph 与 observed frontier 上维护的目标位置后验分布**，配一条期望
代价决策律。Novel/Revisit、match/no-match、frontier 选择、stagnation 触发、
coverage 缺口和 abstain 全部变成同一个分布的读数。

它直接回答目前证据链确认的核心缺失（`LOCAL_MULTIGOAL_CAUSAL_AUDIT_20260806.md`）：
diffusion policy 在困难 Novel 状态下方向 mode collapse（heading resultant
0.98–0.99），而给定正确 metric 方向后 frozen NavDP 是合格 local controller
（3/5 → 5/5）。GLP 提供的正是那个缺失的 long-horizon direction prior。

---

## 1. 为什么要重构决策层（而不是继续加阈值）

NLSR-V2 §5.2/§8.2 的选择规则当前需要同时满足约十个独立门槛，且用
2-plan confirmation、3-plan stagnation、3-plan burst 等手写状态机做时间
平滑。这带来三个具体问题：

1. **机制冗余**：match/rank/coverage 三个 head 实际上是同一个"目标在哪"
   问题的三个投影；stagnation 触发和 coverage 判断又是同一个"已看过的
   空间里没有目标"证据的两种手写代理。
2. **信息丢失**：argmax + 阈值丢掉了多峰结构。当两个房间都像目标时，
   正确行为是先走向分叉点；threshold router 只能选边或 abstain。
3. **论文叙事**：十个阈值的 router 是工程；一个分布 + 一条决策律是方法。

重要边界：`novel_memory_residual_v2.py` 的外层安全壳（native 逐位回退、
harm/pose/clearance 门、burst 上限、幂等重试、fail-closed）**全部保留**。
GLP 只替换其中"目标在哪、该往哪走"的判断，不替换"该不该动、动了是否
安全"的判断。

---

## 2. 假设空间

在第 t 个 plan，目标图 g 的位置假设集合为：

```text
Omega_t = { n_1 .. n_M }        # memory graph keyframe nodes（目标在已观测空间，
                                #   从 node i 的可视区域可见 —— Revisit 假设）
        ∪ { f_1 .. f_F }        # observed frontier components（目标在未观测空间，
                                #   必须经过 portal f_j 进入 —— Novel 假设）
        ∪ { omega_0 }           # unmodeled：frontier 抽取漏掉了通往目标的 portal，
                                #   或目标不可达 / OOD
```

关键语义：

- 任何未观测但可达的位置，都必须经过某个 frontier portal 进入。因此
  frontier 集合（multi-scale union，本地已验证 proposal proxy `6/6`，见
  `NOVEL_MEMORY_RESIDUAL_V2_20260807.md` §1.5）构成对未观测空间的划分；
- `omega_0` 就是原 coverage head 的语义：candidate universe 漏项。它的
  先验质量由 proposal-coverage 审计的实测漏检率校准，不是拍脑袋常数；
- **Novel/Revisit 不再是分类输出**：`P(goal in nodes)` 高即 Revisit，
  质量集中在某个 frontier 即"有方向的 Novel"，质量摊平或 `P(omega_0)`
  高即 abstain。600-session teacher 的分布（31.5% shortlist-positive、
  57.5% strict no-match、11% ambiguous）说明这三种形态在真实数据里都
  大量存在，单一 gate 标量必然混淆它们。

---

## 3. 证据模型：三类似然，一次性讲清楚哪些是静态的

后验为：

```text
P_t(omega) ∝ prior(omega) * L_static(g, omega) * C_t(omega)
```

### 3.1 静态节点证据 L_static（不随 plan 重复相乘 —— 防止校准漂移）

对每个 node：`L_static(g, n_i) = exp(psi_node(g, n_i))`，其中 `psi_node`
是 learned matcher 输出的**校准 log-likelihood-ratio**。输入特征就是
Phase-B 正在采集的集合：directional patch/temporal、LingBot cloud
overlap、pose consistency/refinement、depth confidence、DINO。监督标签就是
causal teacher 的 covis 标签。**这一项对固定的 (g, n_i) 只计算一次**：
goal 不变、node 内容不变，证据就不变。后验的"递归"只来自假设集合的
增长（新 node/新 frontier 加入时增量计算），不存在每 plan 重复相乘同一
证据导致的过置信饱和。这是本设计与朴素贝叶斯滤波的关键区别，也是它能
复用现有 KV-cache/incremental 计算路径的原因。

对每个 frontier：`L_static(g, f_j) = A_j * exp(psi_frontier(g, f_j))`。
`A_j` 是可达面积/portal 尺寸先验（来自 `observed_frontier.py` 的
boundary_m/component_cells），`psi_frontier` 是 goal-conditioned frontier
matcher：输入为 §4.4（V2 文档）定义的 frontier 视觉 context（朝向该边界
的历史 keyframe patches、approach sequence、局部 occupancy crop、拓扑
特征）。单 CLS 的 `-7.293 m` 灾难反例已经证明这一项必须是 set-level
learned model，不能是单一 cosine。

### 3.2 走近时的增量证据（唯一随时间累积的项）

当机器人向某个高后验 node 移动时，新 keyframe 与该 node 的几何一致性
（LingBot pose consistency、patch 对应）构成新的、近似独立的证据项，
乘入该假设。这就是现有 "2-plan confirmation + cached reverify" 的概率化
版本：确认不再是计数器，而是走近过程中证据自然增强或崩塌。为防相关
证据重复计入，只在**新增 keyframe** 上更新，且单条证据的
|log-ratio| 设上限（校准层输出截断）。

### 3.3 可见性剥蚀 C_t（carving）——stagnation 的原理化替代

每当一个区域被以足够的深度置信度观测过而目标没有在该处产生匹配证据，
该区域承载的假设质量按检测功效打折：

```text
C_t(omega) *= (1 - d(x))    # d(x) = depth_conf * 视角质量 * matcher 灵敏度，
                            # 有下限 eps，永不清零
```

推论：native policy 在已剥蚀区域内反复兜圈时，node/近处假设的质量被
draining，frontier 假设的相对质量自动上升——**原来的 "3 连续 endpoint
< 0.60 m novelty" stagnation 触发器成为涌现行为**，不再是手写规则。
goal-blind frontier 实验的教训（4/5 换 4/5，救一个毁一个）在这里的解释
是：它有 carving 没有 psi_frontier —— 知道"这里没有"却不知道"哪里更
可能有"。GLP 两者都有。

### 3.4 frontier 的转化（conversion）与 lineage

frontier 被探索后，其质量不消失，按面积比例流向：(a) 新暴露的更深
frontier（继续 Novel），(b) 新建的 node（若出现目标证据 → 转 Revisit），
(c) 被 carving 销毁（看过了，没有）。frontier 跨 graph revision 的身份
用 portal cell 的空间 hash 追踪，合并/分裂时质量守恒。这是实现中最繁琐
的部分，见 §8 风险。

### 3.5 goal 切换语义（3-leg 的统一处理）

goal image 变化时：所有 `psi_node`/`psi_frontier` 对新 goal 重算（node
特征已缓存，一次 batched 前向），carving 中"目标相关"的部分重置，
**observed 区域图、graph、frontier 集合全部保留**。这正是 V2 不变量
"goal change 只 reset goal posterior，不清 persistent graph" 的自然实现，
2-leg 与 3-leg 不再需要不同代码路径。

---

## 4. 决策律

### 4.1 候选 subgoal

- candidate 0：native `a0` 的 endpoint（永远在场，V2 契约不变）；
- 通往 MAP node 的下一个 graph node（≈1.25 m，Revisit 路径）；
- 后验 top-K frontier 的 approach point（Novel 路径）；
- **hedge 候选**：最小化期望图距离 `E_P[d_graph(x, omega)]` 的节点——
  后验多峰时它自动落在分叉点。这是 threshold router 结构上给不出的行为，
  也是论文里最值得单独展示的定性新能力。

### 4.2 打分与执行

```text
score(s) = Σ_omega P_t(omega) * [ d_hat(x_t, omega) - d_hat(s, omega) ]
           # d_hat 只用 observed graph/free-space 距离，不用 Habitat pathfinder
执行 s*  当且仅当:
  LCB(score(s*) - score(native)) >= delta        # 期望进展优势（ensemble LCB）
  AND H(P_t) <= H_max 或存在质量 >= p_min 的单一区域   # 后验足够确定
  AND P(omega_0) <= p_miss                       # 假设空间本身可信
  AND harm/pose/clearance/burst 门（V2 原样保留）
否则逐位执行 native a0。
```

### 4.3 阈值压缩对照表

| NLSR-V2 §5.2/§8.2 | GLP 中的对应物 |
|---|---|
| match LCB ≥0.90 / UCB ≤0.10 + defer gap | 后验单区域质量 `p_min`、熵 `H_max`（2 个参数） |
| coverage LCB ≥0.90 | `P(omega_0) <= p_miss`（校准自实测漏检率） |
| candidate listwise rank | 后验在 node/frontier 上的归一化本身 |
| 2-plan confirmation latch | §3.2 走近增量证据 |
| 3-plan native stagnation 触发 | §3.3 carving 的涌现结果 |
| advantage LCB ≥0.25 m | `delta`（语义不变，保留） |
| harm UCB / pose p90 / clearance / burst | 原样保留（安全层与定位层正交） |

自由参数从约 10 个独立阈值压到 5 个左右（`p_min, H_max, p_miss, delta` +
先验混合比 `pi_N:pi_F:pi_0`），全部在 train OOF 上拟合。

---

## 5. 与现有资产的映射（几乎零边际数据成本）

| 现有资产 | 在 GLP 中的角色 |
|---|---|
| 600-session causal teacher（covis 标签、strict no-match、counterfactual） | `psi_node` 的训练与校准数据，原样可用 |
| Phase-B feature join（进行中，job 15474001） | `psi_node` 的输入特征，原样可用 |
| `observed_frontier.py` + multi-scale union | frontier 假设生成器（proposal proxy 6/6 已验证） |
| 3-leg expert 轨迹 + Habitat pathfinder | 新标签"goal-behind-frontier"（§6.2）的来源 |
| `novel_memory_residual_v2.py` 状态机 | 外层安全壳，原样保留 |
| geometry R0（19/40）与 max-DINO 阈值基线（88.57%） | Stage 1/3 的必须击败的 reference |
| KV-cache / LingBotStream 增量计算 | 静态证据的增量求值路径 |

需要新增的只有两样：

1. **后验机制本体**（纯 Python，无 GPU，约几百行，§7 Stage 0）；
2. **goal-behind-frontier 标签**：对每个 causal prefix 状态，用 Habitat GT
   最短路判断其首次穿出的 portal 属于哪个 frontier component。标签只进
   label table，服从 V2 §6.7 的 allow-list 契约。

---

## 6. 训练与标签

### 6.1 psi_node

- 逐 candidate binary CE（covis 标签），K+1 set softmax 作为辅助损失保留
  （它已被 smoke 证明可学）；
- 输出经 scene-grouped isotonic/温度校准成 log-likelihood-ratio；
- 通过/失败标准沿用 V2 §11.2：development top-1 至少追平 max-DINO
  `82.35%`，no-match AUC ≥ 现有 `0.95` 量级。

### 6.2 psi_frontier

- 监督目标是 `P(goal behind f_j)`（goal-behind-frontier 标签的 categorical
  CE），不是 geodesic progress 回归——"哪个 portal 通向目标"比"哪个
  候选进展大"更接近因果、且天然 goal-conditioned；
- geodesic progress 保留为次级回归目标与评测指标；
- shuffled-goal 反事实约束沿用 V2 §6.6：正确 goal 的 frontier 排序必须
  显著异于错误 goal，堵住 goal-blind shortcut。

### 6.3 先验与校准参数

`pi_N:pi_F:pi_0`、carving 检测功效 d(x) 的系数、证据截断幅度、
`p_min/H_max/p_miss/delta`：全部 train-OOF 拟合 + 10-scene development
一次性校准，流程与 V2 §8 相同。

---

## 7. 实施阶段与 falsification gate

### Stage 0：后验机制本体（本地，立即可做，无 GPU）

新文件：

```text
MemNavData/goal_posterior.py        # 假设注册表、静态证据表、carving、
                                    # lineage、决策律；不 import torch/habitat
MemNavData/test_goal_posterior.py   # 性质测试
```

必须通过的性质测试：质量守恒（frontier 分裂/合并/转化前后总质量不变）；
carving 单调且有下限；空假设集/全低证据 → native；幂等重试与
graph_revision 语义与 `novel_memory_residual_v2.py` 一致；goal 切换只重置
goal-conditional 层。

### Stage 1：离线重放验证（本地 CPU，数小时，**第一道 falsification gate**）

用**现成的** DINO cosine 经 isotonic 校准充当临时 psi_node（不等
Phase-B），在 600-session teacher 上重放后验：

```text
输入: /tmp/nlsr_causal_teacher_20260807.csv（或 HPC 原件）
比较: joint localization / top-1 / no-match AUC
基线: max-DINO + train-only threshold = 88.57% / 82.35%
```

**判据：同样的 DINO 证据，后验框架若不能追平或超过单阈值基线，则框架
本身不增加信息——停下重新审视，不进入 Stage 2。** 预期的赢面来自：
多 node 证据聚合（正确簇整体质量 vs 单帧 argmax）、先验、显式
`omega_0`。同时在本地 2-scene/6-episode probe 上重放 frontier 后验，
对照单 CLS（3/6）与 goal-blind（4/6）。

**Stage 1 结果（2026-08-07 已执行，报告
`.diagnostics/goal_posterior_teacher_replay_20260807/report.json`，代码
`MemNavData/diag_goal_posterior_teacher_replay.py`）：**

| development (105 strict) | max-DINO 阈值基线 | GLP 后验（DINO-only） |
|---|---:|---:|
| joint localization | 88.57% | 88.57%（2 gain / 2 loss） |
| match accuracy | 92.38% | **94.29%** |
| conditional recall@1 | 82.35% | 82.35% |
| match AUC | 0.9387 | 0.9213 |

train-only 选出 `cluster_gap=4`（网格最小值）、`w0=64`（网格内部）。
判定：**字面通过（追平），实质结论是 DINO cosine 单证据里没有可被聚合
利用的额外信息**——train 偏好最小聚合窗即 aggregation 无增益，与
"DINO-only learned set model 输给 max-DINO"的既有结论一致。但与那个
learned model 不同（其 recall@1 掉到 67.65%），后验**没有损毁基线信息**，
这验证了机制本身无害。§0 的赢面主张（簇聚合救回 never-activated）
由此明确转移为对 Phase-B 特征质量的条件性主张，即 §8 风险 1 所述
"特征弱，后验救不了"的实证确认。Stage 2 继续，但其 Go/No-Go 改为：
Phase-B 特征上的校准似然必须先在同协议下超过 max-DINO，才谈闭环。
本次已消耗一次 development 查看；在 Phase-B 模型冻结前不得再查看
development 分割。

### Stage 2：标签与训练（Phase-B artifact 审计通过后）

- goal-behind-frontier 标签 builder 并入 Lane F 的
  `build_novel_frontier_candidates.py` 计划；
- psi_node 训练 = 既定 P1 流程，仅输出语义改为校准 likelihood-ratio；
- 全部沿用 V2 的 gradient preflight → overfit → 三 seed → 一次 development。

### Stage 3：闭环替换（第二道 falsification gate）

在 20-scene 2-leg 上做**同证据、只换决策层**的配对对照：threshold
router vs GLP。判据：

- Revisit B|A 不低于 19/31，重点考察 12 个失败里 7 个 "从未激活" 的
  episode（后验聚合弱证据应提高 recall）；
- Novel A 31/40 零退化（abstain 逐位等价，V2 §11.4 原样）；
- 若 GLP 与 threshold router 打平，框架仍保留叙事与参数压缩价值，但
  必须如实报告"闭环增益为零"。

之后才是 conditional-B/C 的 frontier 臂与 3-leg。

---

## 8. 风险与对策

1. **校准是承重墙**。未校准的似然会让后验退化成"换皮阈值 router"。
   对策：Stage 1 的 falsification gate；scene-grouped isotonic；证据截断；
   熵门兜底。
2. **frontier lineage 是最繁琐的工程**。对策：portal cell 空间 hash、
   保守合并（质量并集）、Stage 0 性质测试先行。
3. **LingBot depth 误差导致错误 carving**（把目标真实所在区域剥蚀掉）。
   对策：d(x) 乘 depth confidence、质量下限 eps 永不清零、carving 只影响
   相对排序不触发硬拒绝。
4. **相关证据重复计入**。对策：静态证据一次性计算（§3.1）、增量证据仅
   限新 keyframe、幅度截断。
5. **related work 定位**。拓扑图上的贝叶斯定位/POMDP 导航与语义探索先验
   （PONI 一系）是经典；GOAT 一系做 lifelong 多目标。差异化必须写成：
   image-goal 且 novel/revisit 状态未知、纯 RGB 自建图（无深度传感器
   SLAM）、frozen end-to-end diffusion policy 的选择性接管、校准 abstain
   的逐位回退契约。动笔前需一轮专门检索。
6. **进度风险**。GLP 不在 P0/P1 的关键路径上：Phase-B 与 psi_node 训练
   照旧推进，GLP Stage 0/1 是纯并行支线；最坏情况（两道 gate 都失败）
   损失的只是本地 CPU 时间，主线无损。

---

## 9. 论文视角的收益（为什么值得做）

1. 方法从"带阈值的 router"变成"一个分布 + 一条决策律"，自由参数
   10 → ~5；
2. Novel/Revisit 统一为同一后验的两种形态，2-leg/3-leg 同一套代码；
3. hedge-to-branch-point 是结构性新能力，可做定性 figure；
4. stagnation/coverage/confirmation 三个手写机制变成推论，方法节省的每
   一个 ad-hoc 规则都是审稿人少挑的一根刺；
5. 与 mode-collapse 发现（方向缺失）+ SR-per-leg lifelong 曲线组合成
   完整叙事：*发现缺什么 → 用后验补什么 → 能力随经验增长*。
