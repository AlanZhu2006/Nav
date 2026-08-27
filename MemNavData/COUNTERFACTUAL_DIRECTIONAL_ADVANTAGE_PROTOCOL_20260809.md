# Counterfactual Directional Advantage（CDA）协议

日期：2026-08-09

状态：**架构冻结前的 falsification 协议**。本文不授权 blind-set 评测，也不把
任何 Habitat 标签结果写成可部署方法结果。

## 0. 结论先行

当前不应继续做四件事：扩宽候选、用冻结 NavDP critic 跨方向打分、把冻结
diffusion 的简单去噪误差差值当方向能量、用 expert 前进轨迹直接训练 bearing
分类器。现有证据支持的最小方法是：

> 用固定圆周 query 表达“往哪里走”，用 goal-conditioned、同状态内训练的
> advantage adapter 判断哪个方向比 native 更可能接近目标；可靠记忆只作为
> 验证后的方向先验，置信度不足时逐位 abstain 回 native。

它暂名 **Counterfactual Directional Advantage（CDA）**。GLP 的 posterior、
carving 和 frontier lineage 不进入 v1；只有 CDA 先证明被动方向判别可学后，
才能讨论它们是否提供额外信息。不能先用复杂动态后验掩盖静态观测本身可能
不可辨识的问题。

## 1. 新增证据与边界

### 1.1 Observed-frontier 候选覆盖（描述性，不是闭环效果）

严格重放 20 scenes / 40 episodes / 1,069 个 Novel-A policy decision states，
GLB、metadata、parquet 和 Goal-A 均按冻结 manifest/overlay 校验：

- 9 条 native-A 失败、486 个状态中，reachable frontier candidate oracle 在
  ±30° 内为 **421/486 = 86.6%**；
- 现有 goal-blind utility 的 fixed top-1 在 ±30° 内为
  **303/486 = 62.3%**；保持当前朝向仅 **74/486 = 15.2%**；
- fixed top-1 相对当前朝向为 **+243/−14 states**；
- 但在 native 成功轨迹的 583 states 上方向相反：fixed top-1 约 47.5%，
  当前朝向约 74.8%，配对 **+60/−219 states**；
- 失败 episode 的前四个 plan 中，fixed top-1 在 7/9 episode 至少出现一次
  ±30° 候选；另两条要到 plan 9/30 才首次出现可用 top-1。

这些 state 高度自相关，且成功/失败分层使用了结局标签；以上数字只能定位架构，
不能做显著性或因果增益声明。它们说明 frontier provider 并非简单“死掉”，而是
**全局接管会伤害成功轨迹，选择性接管是必要条件**。

权威 artifact：

```text
.diagnostics/observed_frontier_bearing_coverage_20260809/report.json
.diagnostics/observed_frontier_bearing_coverage_20260809/states.csv
.diagnostics/observed_frontier_bearing_coverage_20260809/episodes.csv
```

### 1.2 冻结 NavDP critic 不能做 goal-direction arbitration

在上述 9 条失败 episode 的首个 plan，固定同一 diffusion seed，对 8 个等间隔
point-token 方向逐一做 read-only mixed-goal resample：

- critic 选中的 request 落入 oracle bearing ±30°：**0/9**；
- 其实际 selected trajectory 落入 ±30°：**1/9**；
- 八个方向中至少存在一个可执行 ±30° 轨迹：**8/9**；
- native selected trajectory：**5/9**。

N=9，只作为便宜的 falsification probe；但代码给出了结构性原因：
`NavDP/baselines/navdp/policy_network.py::predict_critic()` 构造四个
`nogoal_embed`，critic 只看 RGB-D 与 candidate trajectory，完全不接收
image-goal 或 point-goal embedding。因此它可以保留为**一次 request 内的
可行轨迹选择器**，不能承担 request 之间的目标方向排序。

权威 artifact：

```text
.diagnostics/navdp_critic_direction_sweep_plan0_20260809/report.json
.diagnostics/navdp_critic_direction_sweep_plan0_20260809/states.csv
.diagnostics/navdp_critic_direction_sweep_plan0_20260809/directions.csv
```

### 1.3 这修正了 STATUS 中两个过强口径

1. “goal-blind frontier 死”只被小样本闭环 null 支持；新结果显示 provider
   在失败轨迹上经常含正确方向。准确说法应是：**goal-blind 全局选择器未通过，
   provider 与选择器必须分开评估**。
2. “只剩 GLP covis 路线未被证伪”不成立。600-session covis 监督只直接训练
   `psi_node`（memory node），并不直接提供 Novel-A 的 `psi_frontier`。
   `goal-behind-frontier` 与 off-policy bearing 数据都尚未构建；二者本质上是
   同一个尚未检验的 goal-conditioned directional observability 问题。

### 1.4 冻结 diffusion 去噪差不能直接充当方向能量

在与 §1.2 完全相同的 9 个 state × 8 个 point-token request 上，新增了一个
不训练参数的 score-field falsification probe。每条固定候选轨迹在全部 10 个
DDPM training timestep 上加入配对噪声，正确 ImageGoal、零 goal 和跨场景错配
ImageGoal 看到严格相同的 `(trajectory, timestep, noise)`；方向分数冻结为：

```text
G(tau) = MSE(epsilon_null, epsilon) - MSE(epsilon_goal, epsilon)
```

结果：

- 正确 goal 选中 oracle bearing ±30° request：**1/9**；对应实际轨迹也是
  **1/9**；
- 跨场景错配 goal：request **0/9**、实际轨迹 **0/9**；正确 vs 错配仅
  **+1/−0**，exact McNemar p=1.0；
- 8 个 request 中 oracle-nearest request 的平均分数名次为 **4.56**，随机期望
  为 **4.5**；随机方向命中期望 1.375/9，而实际 1/9，Poisson-binomial
  upper-tail p=0.779；
- 正确 goal 与错配 goal 在 **6/9** states 选择同一 request；正确 goal 的
  request error 相对错配为 **2 better / 6 equal / 1 worse**；
- 正确相对错配的局部分数在 6/9 state 为正，说明 goal token 并非完全没影响
  denoiser；但该影响没有形成可用的方向排序。

服务端在采样与评分前后都做 FIFO 内容哈希，全部请求保持只读；候选间、方向间
和 goal/control 间共享评分噪声。该实验只否定上述**未校准、等权 timestep 的
denoising-error contrast**，不声称否定所有可能的 diffusion likelihood estimator。
但在新的、预注册的理论评分器通过独立 gate 之前，不再把 frozen score field
当作免费方向源，也不在这 9 个已耗尽 state 上调 timestep 权重或聚合函数。

权威 artifact 与实现：

```text
.diagnostics/navdp_goal_contrast_direction_sweep_20260809/report.json
.diagnostics/navdp_goal_contrast_direction_sweep_20260809/states.csv
.diagnostics/navdp_goal_contrast_direction_sweep_20260809/directions.csv
MemNavData/navdp_goal_contrast.py
NavDP/baselines/navdp/policy_network.py::score_imagegoal_trajectories()
```

对应 SHA256：`report.json` = `8f66849e…878`，`states.csv` =
`373d5080…411`，`directions.csv` = `c6aaae4b…1dc`。

### 1.5 X-NavDP 可作为恢复 primitive，但不是方向源或默认执行器

官方 X-NavDP commit `878740a…` / checkpoint `267089a…` 的 1062 个共享
tensor 与冻结 NavDP **全部 exact equal**；新增的是独立 fine-tuned decoder、
twin-Q 与 embodiment 模块。官方训练/评测输入仍是 PointGoal，ImageGoal encoder
虽保留在训练 checkpoint 中但未被 post-train 路径使用。

在与 §1.2 相同的 9 个 consumed plan-0 state 上，纯 PointGoal 的 oracle-nearest
request 执行结果为：post actor 7/9、byte-identical base actor 9/9、现役
mixed-token 8/9。post actor 没有默认替换价值；它的新增能力高度局部化：
`−180°` fidelity 0/9→9/9（平均 extent 0.40→1.13 m），同时 `±90°` 明显退化。
跨 request 的 X-NavDP Q 只有 4/9 命中，而且没有接收 ImageGoal，仍不能承担
goal-direction arbitration。

因此 CDA v1 不更换 executor。X-NavDP 仅登记为未来后向恢复 primitive：只有
在 train-scene off-policy 状态上同时通过 rear-direction fidelity、碰撞率和固定
时域 geodesic progress，才允许作为 `|residual|` 接近后方时的专用分支；不能
根据这 9 个已耗尽 state 调角度阈值，也不能进入 blind。

## 2. 方法：候选、目标价值、执行三层严格解耦

### 2.1 固定圆周干预基（不建显式 frontier 图）

v1 不把 TopoFocus/observed-frontier 结构搬进主干。每个 plan 的假设空间固定为：

```text
D_t = {d_native, d_0, ..., d_7, d_abstain}
d_k = relative bearing (-180° + 45° k)
```

- `d_native` 永远在场，`d_abstain` 表示保持 native，而不是一个可执行动作；
- 8 个圆周 query 只是**方向假设**，排序时不逐个调用 diffusion；只有最终被门控
  选中的一个方向才做一次 read-only mixed-goal resample；
- §1.2 已显示这一固定基在首个困难 state 的实际 trajectory execution ceiling 为
  8/9，候选问题与选择问题可以分离；
- 后方 query 仍由现有 iterative-token clip/turn 协议执行，不能用一次
  `±180°` point token 绕开已测出的 rear dead zone；
- X-NavDP 的后向 primitive 不进入 v1；§1.5 只保留了一个需独立安全/progress
  gate 的后续假设，不能用 heading fidelity 代替闭环验证；
- memory 不另建一套策略：geometry verification 通过后，把 bearing 写成同一
  圆周场上的 von-Mises/log-prior；未通过时严格为零；
- observed frontier 只保留为覆盖诊断和未来主动信息获取工具，不进入 v1 的
  方向选择器。这样避免把尚未证明有效的地图结构变成方法复杂度。

### 2.2 Goal-conditioned directional advantage

一个共享参数的小 head 对 8 个圆周 query 与 native 输出**同状态相对优势**，
而不是跨场景绝对“激活概率”：

```text
q_k = P(error(d_k, b*) <= 30° | frozen ImageGoal feature,
        observation history, egomotion, direction query)

Delta_k = q_k - q_native
logit_k' = logit_k + verified_memory_kappa * cos(d_k - b_memory)
```

其中 `b*` 只在训练 label table 中由 Habitat shortest path 产生，绝不进入模型
输入或部署日志。模型首先做同状态 listwise/pairwise 排序；绝对概率仅在
scene-grouped OOF 中校准，用于 abstention。

最小网络按可观测性逐级增加，而不是一次堆满：

- O1：冻结 NavDP ImageGoal encoder 的 current-goal feature + 共享的
  `(sin d_k, cos d_k)` query MLP；
- O2：若 O1 通过但仍有明确余量，再加入最近 8 个 decision feature 与已对齐到
  当前坐标系的 egomotion；
- native 只加入 heading、extent、heading resultant 等 proposal statistics；
- verified memory 只通过解析的圆周 prior 注入，不允许 learned head 绕开
  geometry verification。

这本质上是一个 circular readout / listwise adapter，不重训 NavDP、不预测 metric
waypoint，也不在运行时生成 8 组路径。若 O1/O2 在 scene-OOF 下不能读取方向，
就停止；不能用更大的 Transformer 掩盖不可观测性。

不能作为 cross-candidate goal score 的输入：裸 `critic_max` 或未经 OOF
验证的 denoising-error contrast。前者可以记录为 feasibility diagnostic，
但 1.2 已证明其尺度不含 goal 信息；后者在 1.4 没有形成方向排序。

### 2.3 选择性执行

```text
c* = argmax q_i
execute c* iff:
    c* != native
    AND scene-OOF ensemble LCB(Delta_c*) > delta
    AND posterior entropy / margin passes the frozen coverage target
    AND provider-specific geometry/reachability checks pass
otherwise execute native
```

direction residual 在 point-token 可操纵区间内时用现有 iterative token；后方
死区继续执行“转不完就 abstain”，不能让模型分数绕开 burst/clearance 契约。
实际运行每个 plan 最多是一次 native step + 一次被选中方向的 read-only
resample，而不是 8 路长时 eval。

## 3. 数据：先修正可观测性，后谈网络

### 3.1 权威 split

- **train：Phase-B 的 40 scenes / 480 sessions**，只在 scene-grouped OOF
  中训练、选择、校准；
- 10-scene development 与 20-scene 开发池均已耗尽，只允许诊断，不再选择
  checkpoint/阈值；
- blind 16 scenes 保持冻结，直到方法在预声明的 paired N=40 gate 通过。

### 3.2 必须包含的状态

1. **multi-yaw re-render**：在 train expert poses 上按固定 8 yaw 重渲，打破
   现有 97.5% 标签落在 ±10° 的“永远向前”shortcut；
2. **frozen native policy states**：优先收集徘徊、回环、stop/burst 和高
   native–oracle-bearing disagreement 状态。只有这一类能代表待救援分布；
3. **normal-success states**：数量不能少于 hard states，否则模型只会学会
   总是接管，重复 goal-blind frontier 的 gain/loss 交换；
4. **shuffled-goal counterfactuals**：候选几何完全相同，只换同场景 goal
   image 与对应标签；堵住 scene difficulty、朝前和 frontier utility shortcut。

multi-yaw 数据用于测试当前图像的 bearing 可学性，不能替代 native off-policy
rollout。若训练资产不足以覆盖至少多个独立 scene cluster，先补数据，不在一个
17DRP 场景上制造伪大样本。

### 3.3 标签与审计

每个 state 先用 shortest path 的首个 ≥0.3 m waypoint 定义 `b*`。对固定圆周
query 与 native proposal 记录：

- heading error、`within_15/30/45/60`；
- fixed-horizon geodesic progress（secondary label）；
- native heading/extent 与 query-nearest-bin；
- oracle path invalid / different-floor / history unavailable 等 fail-closed reason。

标签分布报告必须按 scene、state source、native outcome/progress stratum 分层；
不能再用 pooled frame count冒充独立样本量。

## 4. Observability ladder 与 falsification gates

所有模型共享完全相同的 candidate set、label、fold 和 seed：

| 层 | 输入 | 它回答的问题 |
|---|---|---|
| O0 | forward/native 与 goal-blind direction prior | 固定方向 shortcut 有多强？ |
| O1 | frozen current-goal feature + circular queries | 单帧是否含 goal-bearing 信息？ |
| O2 | O1 + observation history/egomotion | 时序能否打破 alias？ |
| O3 | O2 + verified memory circular prior | Revisit 证据能否在同一接口无损注入？ |

主要指标均以 **state 内、scene macro** 为单位：

- candidate top-1 `P(error<=30°)`；
- angular regret against candidate oracle；
- paired advantage over native 与 fixed goal-blind direction prior；
- risk–coverage：在 25/50/75/100% intervention coverage 下的 gain/loss；
- scene-grouped bootstrap CI；
- calibration error 仅对 OOF prediction；
- correct-goal vs shuffled-goal 排序差。

通过 O1/O2/O3 的必要条件：

1. scene-macro top-1 或 regret 相对前一层的 cluster-bootstrap 95% CI 下界 > 0；
2. 在预冻结 coverage 上，相对 native 的 paired gains > losses，且 CI 不跨 0；
3. shuffled-goal 后优势显著消失。若换 goal 不影响排序，判定为 goal-blind
   shortcut，哪怕 pooled accuracy 很高也停止；
4. hard-state 增益不能以 normal-success state 的净损失为代价。

若 O1/O2 都未通过，结论不是“再换 backbone”或补一张 frontier 图，而是当前
被动观测下 novel bearing 不可辨识；此时另立主动信息获取/系统探索问题，不再
承诺直接预测目标方向。O3 的 memory prior 已有 A1 闭环证据，不用 novel 失败
反向否定它。

## 5. 最小 HPC 计划

### Phase A：离线数据与 OOF（无闭环）

1. 审计 40 train scenes 的 GLB、expert episodes、已有 native rollouts 是否齐全；
2. CPU/Habitat array 生成 multi-yaw circular-bearing label tables；
3. 只对缺失的 train scene/state 跑一次 frozen NavDP rollout collection；
4. 5-fold scene OOF 跑 O0→O2，先用 linear/circular probe；O1 都失败时不建
   temporal adapter，O2 失败时不进入闭环。

这是数据生成与小模型训练，不需要把“526 episode”逐条做多臂长闭环。
统计单位始终是 scene cluster。

### Phase B：一次 consumed-development paired gate

只有 OOF gate 全过，才冻结 checkpoint、校准器、coverage、candidate 参数与
执行规则，在同机同进程跑 20-scene / 40-episode：

```text
native  vs  CDA selective intervention  vs  oracle-bearing reference
```

primary 是 CDA vs native 的 paired SR；oracle arm 只校验本次机器上的机制上限。
在运行前写死最小有意义效应、最大允许 loss、McNemar 与 scene-bootstrap 规则。
这一步只跑一次，不在结果出来后改 coverage。

### Phase C：blind 16 scenes

仅当 Phase B 通过，才把同一冻结 artifact 用于最终确认。blind 不承担消融、
剂量扫描或架构选择。

## 6. 论文主张的条件形式

现在可写的是问题发现与机制：冻结 ImageGoal diffusion proposal 在困难状态发生
方向塌缩；少量、低精度、持续 bearing 能显著恢复成功率；memory geometry 是
revisit bearing 的一个可部署来源。

CDA/GLP 只有在 OOF + paired closed loop 通过后，才能升级为方法主张。若通过，
真正有辨识度的贡献不是“又一个 frontier ranker”，而是：

> 将冻结扩散导航器的 goal alignment 与 local feasibility 解耦，在 memory、
> circular direction queries 和 native proposal 上做 calibrated counterfactual
> directional advantage；memory 作为验证后的圆周先验注入，并以 abstention
> 保留原策略的成功区域。

如果 O1–O3 失败，也要诚实收缩为：给出 bearing 的因果上限、可操纵性规格、
候选/critic 的失败分解，以及 novel-goal directional observability 的负结果。
