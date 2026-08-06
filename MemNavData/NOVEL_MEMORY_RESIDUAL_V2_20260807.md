# 冻结原生 NavDP 的 LingBot Goal-Conditioned Graph/Frontier Residual V2

日期：2026-08-07

工作树：`/home/asus/Research/Nav-graph-blind`

状态：架构与验收协议草案；允许做 train/development 离线原型，**不批准端到端旧
MemNav 长训，也不批准查看 strict blind 结果**。

## 0. 决策摘要

建议的新方法是 **NavDP-LingGraph Selective Residual（NLSR-V2）**：

1. 官方 NavDP ImageGoal policy 与 checkpoint 完全冻结，且每个 decision 总是先产生
   原生 proposal；
2. LingBot 只维护因果 pose/depth/keyframe graph，并产生 memory-node 或 frontier
   metric subgoal，不再替换 NavDP 的 RGB-D encoder、image-goal encoder、diffusion
   decoder 或 critic；
3. 一个小型 goal-conditioned set model 只学习“哪个 graph/frontier subgoal 相对原生
   proposal 有正优势、是否会伤害、pose 是否可信”；
4. native 是显式的第 0 个候选。匹配/no-match、candidate coverage、advantage、pose、
   clearance 任一不确定时都 abstain，逐位执行原生 NavDP；
5. residual 不是把新网络输出加到 action/epsilon 上，而是把一个短 metric subgoal 连同
   原 goal image 交给**同一个冻结 NavDP mixed image+point decoder**。这样 residual
   只改变高层方向，局部避障和轨迹 critic 仍由原生 policy 完成；
6. 2-leg 与 3-leg 使用同一套 goal-agnostic 状态机：高置信 memory match 走 graph，
   高置信 no-match 且原生停滞时才考虑 frontier；两者之间的模糊区一律 native。

当前判定是：

- **Go**：因果 candidate 数据生成、patch/temporal + LingBot geometry ranker、小 head
  训练、scene-disjoint calibration、shadow deployment；
- **No-Go**：继续训练旧 shared MemNav decoder、always-on coverage、仅用 DINO CLS
  排 frontier、用固定分数或 RANSAC 阈值充当 Novel router；
- **No-Go**：今晚直接启动 8 小时端到端 policy train。Novel residual 的训练标签缓存
  尚未形成，先训练 decoder 只会重复旧 shortcut。

所谓“不伤原生 Novel”必须区分三层：

- 当前纯 Python selector 能保证返回 native-default decision，但它不拥有 NavDP 调用；
- 只有待实现的 native-first orchestration wrapper 通过 FIFO 内容 hash、seed echo、调用次数和
  trajectory-byte 等价测试后，才能声称 **abstain 时输出、NavDP FIFO 和 RNG schedule 与
  原生完全相同**；现有 read-only endpoint/unit test 只证明这条路径可行，不等于集成保证；
- 对所有未来场景的 SR 不可能由一个 learned gate 数学保证，只能通过 conservative
  calibration、有限 burst 和 paired non-inferiority 测试给出统计保证。本文不会把小样本
  的“0 regression”夸大成定理。

## 1. 本地证据与 provenance

### 1.1 旧 Novel 根因已经是因果证据

本地报告：`MemNavData/NOVEL_ROOT_CAUSE_AUDIT_20260804.md`。

在 5 个 scene-disjoint MP3D 场景、10 条 start→A 路线上：

| 方法 | SR | SPL | mean final distance | mean path |
|---|---:|---:|---:|---:|
| 官方原生 NavDP | **9/10** | **0.896** | 2.120 m | 6.000 m |
| residualgate1000，旧 24-waypoint selector | 4/10 | 0.374 | 2.330 m | 10.665 m |
| residualgate1000，selector 只看将执行的前 2 点 | 7/10 | 0.626 | 2.553 m | 6.303 m |
| residualgate1000，GT-geodesic candidate oracle | 10/10 | 1.000 | 0.975 m | 4.602 m |

同 state、同 seed 只换 goal image 时，旧 MemNav 的完整 candidate tensor 变化只相当于
换 diffusion seed 的 `0.13%–3.16%`；官方 NavDP 对照为 `176.761%`。这说明旧 Novel
branch 不是“能力略弱”，而是 action 几乎不依赖 goal。

### 1.2 旧 MemNav 不是 NavDP + memory

同一报告核查出的模型血缘为：

| 项目 | 官方 NavDP | 旧 MemNav |
|---|---:|---:|
| checkpoint tensors / params | 1066 / 135.73M | 369–370 / 57.26M |
| decoder layers | 16 | 8 |
| conditioning tokens | 132 | 17 |
| image goal | 已训练 NavDP encoder | 27.247M scratch Novel branch |
| action decoder | 已训练 decoder + critic | 18.952M scratch/shared decoder |

151 个 same-name/same-shape tensor 中没有一个 exact match；第一层 decoder
self-attention 与 NavDP 的 cosine 约 `0.00045`。因此旧实验测到的是“另一套较小 policy
加 memory”，不是在冻结 NavDP 上做增量。

### 1.3 3-leg Novel-B 的缺失层级是高层 metric direction

本地报告：`MemNavData/LOCAL_MULTIGOAL_CAUSAL_AUDIT_20260806.md`。该文件当前是用户
已有 dirty 文件，本文只读引用，不修改它。

两场景六条 expert 3-leg episode 中五条到达 A。对 Goal-B 的同进程 paired metric
intervention：

| B controller | B given A | mean final B | success mean path | success mean SPL |
|---|---:|---:|---:|---:|
| native ImageGoal | 3/5 | 2.113 m | 10.151 m | 0.764 |
| privileged geodesic 1.25 m point | **5/5** | 0.981 m | 6.669 m | 0.977 |
| privileged exact final metric point | **5/5** | 0.979 m | 7.834 m | 0.886 |

所以这 5 条上并不需要 A* 才能成功：精确 final point 已是 5/5；1.25 m 路径 subgoal
主要再改善效率。冻结 NavDP 是合格的 local controller，缺的是稳定 long-horizon
metric direction。

相反，只把同一 ImageGoal diffusion pool 从 16 扩到 64 candidates，两个困难 episode
仍全部失败；其 endpoint heading resultant 分别为 `0.9834`、`0.9930`，说明更多 seed
主要增加同一错误高层方向附近的局部扰动。

### 1.4 goal-blind frontier 会交换成功样本，而不是净增益

同一 local audit 的 coverage residual 使用三次低 endpoint novelty 才触发，仍得到：

- native Goal-B：`4/5`；
- coverage residual：`4/5`；
- 一条 native failure 被救回，但一条 native success 被破坏；
- always-on frontier 更早已经破坏两个原本成功的 `1L` episode。

因此“native 停滞”只能是必要条件，绝不能直接推出“任意 coverage frontier 更优”。
这正是 V2 必须以 goal-conditioned advantage 和 uncertainty 作为第二、第三道门的原因。

### 1.5 新本地 frontier probe：候选有上限，简单排序都有灾难反例

原始 artifact（本地 privileged feasibility，不提交）：

```text
.diagnostics/goal_conditioned_frontier_probe_20260807/report.json
SHA256 12c0f172adc9ff0ee3b67c19f039509f1c48ecebd339510993729545ac59dfae
```

协议是 2 scenes / 6 expert 3-leg episodes，用 Goal-A prefix 的 RGB-D 构建 frontier map；
Habitat GT pose/pathfinder 只用于候选与 geodesic-progress 标签。所谓 DINO rank 只是目标
图与候选最近五个 context frame 的冻结 LingBot DINO CLS cosine，不是可部署结论。

结果：

- `5/6` episode 产生 frontier candidates，另 `1/6` 是 privileged proposal-proxy miss；
- 在有 candidate 的 5 条中，oracle progress-positive 为 `5/5`；
- 固定 goal-blind frontier score 选正为 `4/5`；
- 简单 DINO-context top-1 也只有 `4/5`；
- `17DRP5sb8fy/episode_0002` 中，DINO 选择的 progress 是 **`-7.293 m`**，
  而候选 oracle 是 **`+3.960 m`**；
- `1LXtFkjw3qL/episode_0002` 则相反：DINO 为 **`+3.264 m`**，固定 goal-blind
  score 为 **`-0.538 m`**，oracle 为 `+5.052 m`。

这不是“DINO 比 coverage 好”或相反，而是两者有互补信息、任何单分数都不可靠。
V2 必须先修 proposal miss，再用 goal patch/temporal evidence、graph geometry
和 uncertainty 组合；在 5 条上调一个阈值没有意义。

同日晚些时候的 multi-scale follow-up 已把“候选生成”和“候选排序”进一步分开：

```text
.diagnostics/goal_conditioned_frontier_probe_20260807/multiscale_report.json
SHA256 cc6f4e32a86c9883a4638f1a73483f6dbd7ff665a6bb543209e679020a0ee886
```

它合并 `r15/r20/r30` 三种 grid resolution 的 frontier union：privileged
**proposal-reachability proxy** 从 `5/6` 提到 **`6/6`**，且 Pathfinder
progress-positive 为 **`6/6`**；说明本地的 proposal miss 可以通过 multi-scale
proposals 修复。这里尚未对 native 与每个 candidate 做同 state、同 seed 的 H24 NavDP
rollout，所以这个 `6/6` **不是** residual useful coverage，也不能监督 coverage head。
候选变多后，简单 DINO CLS top-1 从
`4/5` 降为 **`3/6`**，固定 frontier score 也只有 **`4/6`**。例如：

- `17D/ep0`：DINO `-5.840 m`，oracle `+4.840 m`；
- 原本无 candidate 的 `17D/ep1`：DINO `+3.294 m`，goal-blind `-4.192 m`；
- `1L/ep0`：DINO `-1.099 m`，oracle `+1.168 m`。

所以 multi-scale 是 candidate generator 的 Go 信号，却让单 CLS ranker 的 No-Go 更
明确：proposal recall 提高会带来更多 hard negatives，必须使用 set-level listwise
patch/temporal + geometry，并在不确定时选择 native。

### 1.6 memory graph 有价值，但当前数字不是 final

可靠 20-scene / 40-episode 2-leg 结果记录于：

- 本地说明：`MemNavData/ONLINE_ROUTER_FAILURE_AUDIT_20260805.md`；
- 本地冻结 manifest：`MemNavData/expanded_navdp_router_eval_20260805.json`，SHA256
  `ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61`；
- 报告中记录的 HPC 原始 summary：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/all_in_one_router_eval_20260805/`
  `all1_2leg3leg_6dc745f_20260805/expanded_navdp_router/summary.json`。

| 2-leg | Native NavDP | geometry memory R0 |
|---|---:|---:|
| Novel A | 31/40 = 77.5% | 31/40 = 77.5% |
| Revisit B given A | 4/31 = 12.9% | **19/31 = 61.3%** |
| joint | 4/40 = 10.0% | **19/40 = 47.5%** |

geometry-only gain 15、loss 0，exact McNemar `p=6.1035e-5`。但 V2 不把 SIFT/RANSAC
router 当创新主体；它只把这组结果当作“memory metric route 值得保留”的 reference。

旧 reverse graph development run 从 direct `19/40` 到 graph `25/40`，六 gain、零 loss；
因旧 evaluator 没有逐 request 匹配 DDPM noise，这只是结构信号。严格 shared-prefix
protocol 已写在 `MemNavData/ELEGANT_MEMORY_GRAPH_EXPERIMENT_20260806.md`，最终 graph
数字必须重新跑。

### 1.7 当前 3-leg 不能用于否定 long-memory C

本地 manifest：`MemNavData/expanded_3leg_router_eval_20260805.json`，SHA256
`55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b`。

10-scene end-to-end 的顺序分母是：A Novel `6/10`，B Novel given A `1/6`，C Revisit
given A/B `0/1`，joint `0/10`。首要瓶颈是第二个 Novel B，只有一条 episode 真正评到
C；不能写成“LingBot 3-leg memory 全部失败”。

### 1.8 blind 仍未打开

`MemNavData/strict_graph_blind_20260806.json` 固定 16 scenes / 32 episodes，SHA256
`b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9`。这些场景不用于
训练、threshold、candidate coverage 修复或今晚 smoke；配置冻结前禁止查看结果。

## 2. 为什么旧 memory 一加到 Novel 就差

这是六个问题的组合，而不是“memory 天生伤 Novel”。

### 2.1 Base 被替换，不是被保护

旧 MemNav 重建了 image branch、current-state encoder、8-layer diffusion decoder 和
selector。即使 gate=0，执行的仍是 scratch MemNav Novel policy，不是官方 NavDP。
所以 `gate=0` 不能提供 base equivalence；继续调 gate 也无法恢复没有继承的权重。

### 2.2 训练数据允许忽略 goal

旧 loader 为了让 revisit candidate 非空，把 Goal-A current 下界推到：

```text
anchor_margin 39 + exclude_recent 83 = k_lo 122
```

inference 首次可规划是 k≈40，却从不训练 k=40..121。10 条 Novel eval route 中 4 条整段
无法产生 Goal-A row，另 2 条到 k=122 时已在 1 m success radius 内。flow run 的
Goal-A 只有 1,354 个 samples，step 2600 约等于 6,320 次 exposure，却要从零训练约
46.2M image+decoder 参数。

更根本的是，每个 current 只配同一 expert trajectory 的 future endpoint；沿 expert
方向“继续向前”已能降低 epsilon-MSE。模型没有看到同一 state 对两个目标必须走不同
方向，也没有 shuffled-goal 的有效反事实标签，于是四个 Novel tokens 被 current-state
shortcut 淹没。

旧固定 evaluator 也显示这种 objective 错位：flow→gatecurr 时 retrieval set loss 从
`0.204852` 降到 `0.134628`、gate accuracy 从 `69.44%` 升到 `80.56%`，但 Novel
action epsilon-MSE 反而从 `0.057873` 变成 `0.061428`。所以“总 loss/路由 loss 变好”
不能证明 Novel action 被保护。

### 2.3 一个 scalar gate 混淆了三个问题

旧 gate 同时试图回答：

1. complete history 中有没有目标；
2. shortlist 中哪个 node 正确；
3. pose/action residual 是否值得执行。

这些目标的正负样本、校准和错误代价都不同。已有 K+1 诊断也显示 rank 与 no-match
存在多任务冲突。V2 必须拆成 match/no-match、candidate rank、pose uncertainty、
advantage/harm 四种输出，不能再让一个 gate 决定一切。

### 2.4 Action loss 看不到 selector/controller mismatch

旧 selector 用全部 24 waypoint 评 collision，但闭环每 8 frames 重规划，只会实际提交
约 2 waypoint。它与 GT oracle 的候选一致率仅 `8/158 = 5.1%`；只改成前 2 点评分就从
`4/10` 到 `7/10`。训练的 epsilon-MSE 既不监督 selector，也不监督累积 100 次后每步
1–2 cm 的 progress regret。

### 2.5 Memory 还替换了 native current geometry 与 critic

官方 NavDP 使用自己的 RGB-D backbone 与 learned critic；旧 MemNav 使用 LingBot
predicted depth 和几何 score。于是即使 goal memory 没有启用，局部感知、候选分布和
轨迹选择也已经变化。V2 只允许 LingBot 在外部产生高层 subgoal，不让它替换 native
local-control observation path。

### 2.6 Unconditional exploration 与 ImageGoal 目标不等价

最新 frontier 因果结果已经给出一 gain、一 loss；新 probe 又给出 `-7.293 m` 的单-CLS
灾难候选。Novel 并不意味着“走向面积最大的 unknown”。只有 goal-conditioned、相对
native 有可信正优势的 exploration 才有资格成为 residual。

## 3. V2 总体架构

```text
                         ┌──────────────────────────────────────┐
RGB-D_t + goal RGB ─────►│ frozen official NavDP ImageGoal π0   │
                         │ append FIFO once; native proposal a0 │
                         └────────────────┬─────────────────────┘
                                          │ a0 always survives
RGB_t ─► frozen LingBot stream ─► pose/depth/keyframe factor graph
goal RGB ─► dense patch/temporal query       │
                                              ├─ high-conf match:
                                              │  graph-node route candidates
                                              └─ high-conf no-match:
                                                 observed frontier candidates
                                                         │
                 goal + patches + local map + covariance + native proposal
                                                         │
                                      small set rank/value/risk model
                                                         │
                    calibrated lower-bound advantage + harm upper bound
                                                         │
                              ┌────────── abstain ────────┴────── accept ┐
                              ▼                                         ▼
                      execute native a0               read-only frozen NavDP
                                                     image+point resample ai
```

关键约束：

- evaluator 不提供 `Novel/Revisit` phase；由 match posterior 的置信区间决定；
- inference 不接收 Habitat pose、goal coordinate、navmesh 或 geodesic；
- V2 deployment 不依赖 SIFT/RANSAC。现有 geometry R0 只作 reference，不作 fallback；
- V2 的 fallback 永远是 frozen native NavDP；
- 原 goal image 始终保留。point token 是短方向 residual，不会把 ImageGoal 关掉；
- residual request 必须 read-only，不可第二次 append current RGB，也不可改变 FIFO；
- 每个 proposal 的执行 commitment 仍是 8 frames，之后重新估计 posterior 和风险。

## 4. 怎样真正利用 LingBot pose、depth 与 pose graph

### 4.1 Causal node

每个持久 keyframe node 保存：

```text
node_i = {
  T_map_camera_i, Sigma_pose_i,
  DINO global + patch tokens,
  predicted depth + depth_conf,
  local surfels/free-space rays,
  timestamp/goal_epoch/visibility cone
}
```

写入依据是 pose displacement、yaw、视觉新颖度、depth confidence 与 frontier coverage，
不是固定每 N 帧均匀抽样。recent ring buffer 保留短期密集帧；long-term graph 合并冗余
node，但保留被合并帧的 patch/visibility summary。

### 4.2 Factor edges

V2 使用三类 factor：

1. LingBot streaming 相邻帧/关键帧的 relative SE(2) 与 covariance；
2. dense patch + LingBot cloud/pose-consistency head 预测的非相邻 loop factor；
3. depth ray-carving 得到的 traversability/visibility edge。

每条 factor 都携带 covariance；graph optimizer 使用 robust loss，下游将路径上的
covariance 传播到 subgoal。这里没有“20 matches / 12 inliers / ratio 0.5”之类手工
RANSAC router。已有 93-row LingBot-native development collection 中，cloud-overlap
ROC-AUC `0.743` 高于 DINO `0.610`；scene-LOSO 融合到 AUC `0.776`、session top-1
`23/25`，说明 LingBot geometry 有可学习信号，但仍需校准，不能选单一固定阈值。

### 4.3 Depth map 与 frontier

用 LingBot predicted depth/depth confidence 沿 `T_map_camera` ray-carve：

- 高 confidence surface 标 occupied；
- ray interior 标 free；
- pose/depth covariance 对 obstacle 做概率膨胀；
- free/unknown boundary 形成 connected frontier component；
- 每个 component 产生多个 approach pose，而不是只取一个 centroid；
- candidate generator 同时保留小近 frontier、大远 frontier、拓扑分支与 native heading
  邻域；本地 `r15/r20/r30` union 已把单尺度的 `1/6` miss 补到 `6/6`，下一步是在
  scene-disjoint 数据验证，而不是继续堆单尺度候选。

候选是否可达只在这个 observed graph/free-space map 内判断，不调用 Habitat pathfinder。
candidate-coverage head 预测当前集合是否可能漏掉 useful branch；coverage LCB 不足时
整个 residual abstain，而不是被迫从坏候选里选一个。

### 4.4 Goal-conditioned frontier context

未知区域没有真实 RGB，V2 不伪造“frontier image”。每个 frontier 的视觉 context 来自：

- 与该 boundary 相邻、且 viewing cone 朝向该边界的历史 keyframe patches；
- 这些 keyframe 的 temporal approach sequence；
- local depth/occupancy crop、doorway/corridor topology、frontier orientation；
- goal patches 与上述 context patches 的双向 cost volume/cross-attention；
- native endpoint、native heading、到 visited trace 的 novelty 与短期停滞统计。

如果目标图与任何 observed context 都没有可解释对应，epistemic uncertainty 应升高并
abstain；模型不能仅凭“这是厨房，所以随便选一扇门”。单 CLS 只保留全局语义，已经由
`17D/ep2 -7.293 m` 证明不足。

### 4.5 Match 后切换到 graph route

探索过程中一旦 goal-to-memory posterior 从 confident no-match 变为 confident match，
立即停止 frontier mode，定位 goal pseudo-node，并在 optimized graph 上取下一个 1.25 m
左右的短 node。接近 anchor 后恢复 image-dominant final alignment。

相比旧 reverse temporal chain，完整 pose graph 可以跨越 loop edge，避免把绕行历史
原样倒放；若 loop factor 不确定，则退化为已走过的 sequential chain，而不是猜 shortcut。

## 5. Residual 的精确定义与“不伤 base”契约

V2 的 residual 是**相对 native proposal 的高层 subgoal advantage**：

```text
a0 = frozen NavDP ImageGoal(obs FIFO, goal, depth; seed)
pi = graph/frontier metric subgoal in current [forward, left]
ai = frozen NavDP mixed ImageGoal+PointGoal(obs FIFO, goal, pi, depth; same seed)
Delta_i = V(ai) - V(a0)
```

不采用 `action_final = action_native + delta_action`，也不直接改 DDPM epsilon。原因是
waypoint/action space 的线性相加不保持 collision-free，且 residual head 一旦分布外就会
无条件污染 base。mixed decoder 本来就在官方 NavDP 多目标训练中见过 point/image token
组合，适合做冻结 local controller。

### 5.1 强不变量

1. 官方 checkpoint SHA256 固定为
   `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`；
2. 所有 NavDP parameter `requires_grad=False`，训练 artifact 记录 frozen/trainable count；
3. 每个 decision 先调用 native endpoint，current observation 只 append FIFO 一次；
4. residual resample 复用相同 FIFO 与 request seed，接口必须返回
   `memory_mutated=false`；
5. abstain 时直接执行已产生的 `a0`，不重新 sample；
6. 缺 feature、NaN、graph stale、calibration OOD、candidate miss、interval overlap、
   pose/depth 不确定任一发生都 native；
7. residual 最多连续执行 3 个 plan，每个 plan 只 commit 8 frames，然后强制重新判断；
8. goal switch 清空 posterior confirmation/latch，但不清空 LingBot graph，也不硬 reset
   NavDP FIFO。local hard-reset A/B 已显示 gain/loss 相抵，不是稳定修复。

### 5.2 已落地的独立逻辑核

本文新增：

- `MemNavData/novel_memory_residual_v2.py`；
- `MemNavData/test_novel_memory_residual_v2.py`。

它不 import Habitat、Torch、LingBot 或 dirty evaluator，只实现 native-default 的选择状态机。
默认门槛是：match LCB `>=0.90` 才允许 memory graph；match UCB `<=0.10` 才允许
frontier；candidate coverage LCB `>=0.90`；advantage one-sided 95% LCB `>=0.25 m`；
harm probability UCB `<=0.05`；pose translation p90 `<=0.30 m`、yaw p90 `<=15°`；
clearance lower bound `>=0.30 m`；frontier 需 3-plan native stagnation、同 candidate 2-plan
确认、最多 3-plan burst。

构造 threshold 时强制 `frontier match UCB < memory match LCB`，必须保留 defer gap；同一
decision 若出现重复 `(source, candidate_id)` 也直接 native 并记录歧义，不能让重复 row
改变确认或排序。

V2 context 显式携带 `session_id / goal_epoch / plan_index / graph_revision`。完全相同的
decision 重试返回缓存 decision 且不推进计数；同 plan 改输入、plan/revision 不递增、跨
session 残留状态均 fail closed。confirmation 只在严格递增 plan 且新 graph revision 上
推进。执行 residual 后，下个 plan 必须回传与
`(source,candidate_id,executed_plan,executed_revision)` 对齐的 improvement feedback；缺失、
错配、旧 feedback 重放或 graph/native-novelty/goal-evidence 全无改善时，该 plan 强制 native
并清 latch，形成至少一个完整 native-plan cooldown。

这些是保守起始值，不是用 6 条 local episode 调出的“最优参数”。正式值只能从 train
OOF 和 10-scene development calibration 冻结。

### 5.3 尚未落地的 exact-fallback orchestration

本次新增文件没有实现 NavDP HTTP/executor wrapper。提交 exact-equivalence 声明前仍须新增
集成层并故障注入验证：native endpoint 每 plan 恰好调用一次且最先调用；selector、feature
builder、ranker、serialization 任一异常都执行已缓存 `a0`；abstain 不调用 resample；FIFO
前后比较内容 hash 而不只 queue length；request seed/echo 一致；abstain arm 与 native arm
trajectory bytes 完全一致。该 gate 未通过前只能写“exact fallback feasible”，不能写
“exact fallback implemented”。

## 6. 训练数据：sample、candidate 与标签

### 6.1 Scene 角色不可混用

- 40 train scenes / 10 offline-development scenes：
  `MemNavData/router_multiscene_split_20260805.json`；
- 已反复查看的 20-scene / 40-episode navigation set：closed-loop development A/B；
- 2-scene / 6-episode local frontier probe：只做 logic/candidate smoke；
- `strict_graph_blind_20260806.json` 的 16 scene / 32 episode：一次性 final blind。

所有 normalization、threshold、temperature、conformal quantile 与 graph hyperparameter 只
能来自 train OOF + offline development，不能看 blind。

### 6.2 一个训练 row 是一个 causal decision set

```text
(scene, episode, goal_epoch, plan_index,
 causal RGB/pose/depth graph prefix,
 frozen native FIFO + native proposal,
 goal image,
 K graph/frontier proposals,
 counterfactual rollout labels)
```

prefix 必须截断在 decision 时刻；不得把未来 B/C observation、完整 expert map 或目标
坐标放进 input。标签计算可以用 Habitat GT，但导出的 deployment feature allow-list 不得
包含 `target_`、geodesic、pathfinder、goal world coordinate。

### 6.3 两种 state distribution

1. expert/source-prefix states：高效覆盖 topology 与少见 turn；
2. frozen native NavDP on-policy states：覆盖真实 FIFO、停滞、偏航与 collision recovery。

旧 decoder 在 generator trace 成功、在 NavDP closed-loop state 失败，是明确的 covariate
shift 证据。因此第二类不能省；困难 state 应通过 frozen-policy DAgger 式追加，而不是
解冻 policy 来适应 residual。

### 6.4 Candidate pool

每个 set 总含：

- candidate 0：native `a0`；
- high-confidence memory graph 的 next-node proposals；
- 多尺度 frontier approach proposals；
- native heading 左/右相邻的 reachable branches；
- 显式 dustbin：当前 candidate set 无 useful residual。

部署 `K_residual <= 32`，所以含 native/dustbin 的总 set size 不超过 34。所有 candidate
feature 必须一次 batched encode；rank 阶段不得逐 candidate 调 NavDP。只有最终获选的一个
residual 可以额外触发一次 mixed resample。

生成器先做 proposal audit，不先做 rank。对每个 state 报：candidate 数、是否存在
Pathfinder progress-positive、oracle proxy progress、source/距离/方向分桶 recall。单尺度
probe 是 `5/6`，multi-scale smoke 是 `6/6`；它们只验证 proposal reachability proxy，
不能称为 residual useful coverage。后者必须对 native 与完整候选 universe 做同 state、
同 seed 的 H24 冻结 NavDP rollout 后另行计算。

### 6.5 Counterfactual labels

对每个 residual candidate 和 native，用同 FIFO、同 diffusion seed，从同 simulator state
克隆短 rollout：

- `H_exec=8` frames：collision、clearance 与真实 commitment safety；
- `H_value=24` frames（3 个 commitment）：goal geodesic progress 与 stuck；
- 对 hard states 再做有限 closed-loop branch rollout，标终局 success/regression。

核心标签：

```text
progress_i   = d_goal(s_t) - d_goal(s_i,H)
adv_i        = progress_i - progress_native
regression_i = adv_i <= -0.25 m
harm_i       = collision OR regression_i
useful_i     = adv_i >= +0.25 m AND not harm_i
```

另有：

- `global_match / strict_no_match / ambiguous`：由完整 causal history 的 directional
  co-visibility 标注，shortlist miss 不能被伪装成 no-match；
- `anchor rank` 与连续 co-visibility；
- LingBot raw relative SE(2) 到 GT relative SE(2) 的 wrapped residual；
- depth/pose graph consistency 与 candidate coverage miss；
- path length、bearing change、frontier topology 只作连续 target/stratification。

`progress_i`、GT pose 和 goal coordinate 只存在 label table，不进入 model feature。

### 6.6 必须有同-state多-goal反事实，但 utility row 只用同场景合法 goal

对同一个 causal state 至少配：

- 本 episode 的 A/B/C goal；
- 同场景不同方向的有效 goal；
- 相似房间 hard goal；

每个 goal 都重新计算自己的 privileged candidate label，不给 shuffled goal 伪造 action
GT。训练必须让 candidate rank 随 goal 改变；validation 固定报告 correct-goal 与
shuffled-goal 的 rank/logit/activation 差异。这直接堵住旧 policy 的“只看 current 然后
继续向前”shortcut。

跨场景 goal 没有当前 Habitat scene 内的合法 target，不能伪造 native/H24 utility
rollout。它们只进入 Lane M 的 match/no-match、OOD/calibration 数据，**不进入**本文
candidate-set utility artifact，也不参与 rank/advantage/coverage loss。以后若要合并两种
artifact，必须新增显式 `utility_label_valid` mask 并升级 schema；当前 V2 不静默混用。

### 6.7 已落地的 candidate-set 数据契约

本文另新增：

- `MemNavData/novel_candidate_set_schema_v2.py`；
- `MemNavData/test_novel_candidate_set_schema_v2.py`。

这是纯 JSON-like 数据层 validator，不 import Habitat，也**不声称 collector 已实现**。
当前版本刻意只接收“同场景合法 goal + valid/reachable native rollout”的 utility set；
因此 local Pathfinder-only probe 与 cross-scene OOD row 都不能通过该 schema，这是防止
把 proxy/no-target 样本伪装成动作收益标签的预期行为。
它要求：

- candidate 0 必须且只能是 `native`；最后一个必须且只能是 `dustbin`，dustbin 除 type
  one-hot 外的 features/labels 必须全零/false，不能暗带 privileged target；
- candidate id、decision key 不可重复；
- feature 使用精确 deployment allow-list，patch/temporal/local-map relation 可以是有限数值
  vector，但 shape 在 candidate 与 dataset 间必须一致；candidate 的 7-bit presence mask
  固定依次对应 `goal_patch / goal_temporal / local_map / native_proposal relation / pose
  uncertainty / depth confidence / clearance`，set 的 6-bit mask 覆盖 stagnation 与 graph/count
  fields，且只能取数值 0/1；mask=0 时对应字段必须归零，不能把 unknown 当真实 0；
- geodesic/pathfinder/Habitat/oracle/GT/target/success 字段禁止进入 feature，只允许作为
  label；
- scene/episode/session/group/goal epoch/plan、显式 same-state `state_id`、state source、
  同场景 goal-source episode、
  environment/navmesh、prefix/FIFO/goal hash、split、source policy、candidate generator、
  feature builder、rollout labeler provenance 缺一即拒绝；
- group/scene 不可跨 train/development role；`final_reserved` 不能成为 trainable artifact；
- 同一 `state_id` 的 prefix/FIFO/plan/environment 必须逐字段一致，同一 session/goal epoch
  不能改变 goal source 或 goal content hash；这使 factual/counterfactual 的“同 state”成为
  可验证约束而不是文件名约定；
- 同一 artifact 的 dataset/split/source-policy/candidate-generator/feature-builder signature
  必须一致，同一 session/goal 的 prefix length 随 plan 单调不减；
- missing/extra key、重复 native/dustbin、ragged vector、NaN/Inf、label summary 不一致均
  fail closed；
- `advantage_h24` 必须逐 row 等于 residual progress 减 native progress；
  `regression_h24` 必须等于 `advantage_h24 <= -0.25 m`；`harm` 必须等于
  `collision_h8 OR regression_h24`，`useful`、reachable/valid 也必须一致，oracle 必须是
  最大 valid useful advantage；终局 success/failure 若未来加入，必须作为带 valid mask 的
  独立 schema label，不能复用或暗改这个 H24 定义；
- coverage 不是“set 内已有 positive”的别名：另存
  `candidate_universe_has_positive / candidate_coverage_miss / coverage_label_valid`，只有
  对完整 proposal universe 逐候选完成同 seed H24 rollout 时才置
  `coverage_label_valid=true` 并训练 coverage head；Pathfinder proposal proxy 单独写 audit
  table，并使用独立的 `proposal_proxy_*` valid/reachable/progress/positive 与
  set/universe/miss labels，绝不填入这三个 utility coverage label；
- 通过后可生成 canonical JSON SHA256，供 collector resume、trainer 与 report 互验。

这里的 allow-list 只能拒绝显式 privileged/unknown column，**不能证明** builder 没把
geodesic/GT 编码进 `goal_patch_relation` 等合法 tensor。真正的 causal/no-leakage 保证还
需要 builder code SHA 审查、prefix 截断重算、FIFO/prefix content hash 复核，以及至少一组
future-frame/goal-coordinate 注入的负向 auditor 测试；schema 不再单独宣称完成该证明。

今晚的 builder 即使能产生 parquet，也必须先逐 record 和整 dataset 通过该 schema；trainer
不能自行“容错”删除坏 row。

物理 artifact 也分三层，最后由 auditor 按 candidate id 与 provenance hash 做严格 join：

```text
candidate_features.parquet        # 只含 deployment-time inputs
proposal_proxy_labels.parquet     # Pathfinder teacher；不等于 policy utility
rollout_utility_labels.parquet    # frozen NavDP paired H8/H24 outcomes
audit/join -> train_candidate_sets.parquet
```

trainer 只能读取最后一个审计产物，不能直接打开 privileged proxy/rollout table。

## 7. 小模型与损失

### 7.1 输入与 heads

冻结 LingBot/DINO backbone。v0 的固定数据规模只有 40 train scenes × 2 episodes × 2
states × 2 same-scene goals = 320 decision sets（development 80 sets），因此先训练
**50k–200k 参数**的 masked DeepSets model；在加入更多 scene-disjoint on-policy states
并通过学习曲线审计前，不上 2–4M head：

- goal patch tokens；
- candidate adjacent-context patch/temporal tokens；
- candidate local occupancy/depth crop；
- native proposal summary 与 visited-trace novelty；
- LingBot pose/depth confidence、graph path covariance；
- candidate type、distance、bearing、topological degree。

共享 encoder 后分开输出：

1. `match head`：global match / no-match / ambiguous posterior；
2. `rank head`：candidate set 内相对 utility；
3. `advantage head`：`mu_adv, sigma_aleatoric`；
4. `harm head`：collision/regression probability；
5. `pose head`：LingBot SE(2) small residual + covariance；
6. `coverage head`：candidate set 是否可能漏掉 useful branch。

三个 seed/bootstrapped scene ensemble 提供 epistemic variance。网络不产生 waypoint，不
训练 NavDP decoder，也不需要 action epsilon-MSE。

### 7.2 初始 objective

```text
L = 1.00 L_match
  + 1.00 L_rank
  + 1.00 L_adv_nll
  + 2.00 L_harm
  + 0.50 L_pose_nll
  + 0.25 L_coverage
  + 0.25 L_goal_counterfactual
  + 0.10 L_temporal_consistency
  + 0.05 L_calibration
```

- `L_match`：strict match/no-match class-balanced BCE；ambiguous 不给二值标签；
- `L_rank`：包含 native candidate 的 advantage-weighted listwise loss；
- `L_adv_nll`：Huber heteroscedastic NLL，长尾不把均值拖坏；
- `L_harm`：对 false-safe 加高权重，不能用 class accuracy 掩盖 catastrophic tail；
- `L_pose_nll`：wrapped SE(2) residual，不直接回归 `179°/-179°`；
- `L_coverage`：仅在 `coverage_label_valid` row 上监督 universe positive 与
  candidate-coverage miss，并报告各 source bucket recall；不能用 set 内是否已有 positive
  代替 miss target；
- `L_goal_counterfactual`：同 state 下正确 goal 的 positive rank 必须胜过其他 goal；
- `L_temporal_consistency`：相邻 plan 的 match、goal pose 与 candidate identity 平滑，但
  goal_epoch 改变时不跨 boundary 平滑；
- `L_calibration`：Brier/variance regularization，最终阈值仍在独立 calibration split 选。

权重只允许在 40-train scene-grouped OOF 选择；不能根据 local 6 episodes 或 20-scene
closed-loop 的单个失败临时改 loss。

## 8. Calibration 与 uncertainty abstain

### 8.1 分开校准

- match/no-match：temperature/isotonic，按 scene group；
- advantage：ensemble epistemic + aleatoric 后做 one-sided split conformal；
- harm：输出 probability 的 upper confidence bound，而不是 point estimate；
- pose：translation/yaw 分别做 normalized-residual calibration；
- coverage：按 frontier count、scene topology、goal epoch、short/medium/long gap 分层；
- OOD：若 goal/context embedding density 或 graph uncertainty超出 calibration support，
  `calibration_supported=false`，直接 native。

### 8.2 选择规则

对 candidate `i`：

```text
LCB_adv_i = mu_i - q95 * sigma_total_i
```

只有以下全部成立才进入 temporal confirmation：

```text
memory graph: P(match)_lower >= 0.90
frontier:     P(match)_upper <= 0.10 AND native_stagnation >= 3 plans
both:         P(candidate coverage)_lower >= 0.90
              LCB_adv >= 0.25 m over H_value
              P(harm)_upper <= 0.05
              pose translation p90 <= 0.30 m
              pose yaw p90 <= 15 deg
              clearance lower bound >= 0.30 m
              graph fresh AND calibration supported
```

0.30 m 是固定 agent radius/现有 frontier novelty 的量级；pose translation p90 取其同级，
避免 uncertainty 本身跨过安全边界。0.25 m advantage 是比旧每-plan `1.63 cm` oracle
regret 更有意义的高层改善门槛。它们仍需 OOF 验证，不是物理常数。

### 8.3 独立样本数

连续 plan 高相关，不能把一次 episode 的 30 个 decision 当 30 个独立安全样本。若希望
在 59 个独立 activation episode 上观测 0 harm，则 one-sided 95% Clopper–Pearson upper
bound 才约低于 5%。今晚的小 probe 不可能给出这个保证，因此只允许 shadow/开发闭环，
不允许宣布 production-safe。

## 9. 部署状态机

每个 plan：

1. 用 `(episode, goal_epoch, plan_index)` 固定 diffusion seed；
2. 先跑 native ImageGoal，一次性 append FIFO，保存 `a0`；
3. LingBot append 同一观测，更新 pose/depth graph 与 covariance；
4. goal localizer 输出 match interval；
5. match LCB 高：产生 memory graph candidates；no-match LCB 高且 native 已停滞：产生
   frontier candidates；中间区不产生 residual；
6. 小模型对 `[native, residuals, dustbin]` 输出 rank/advantage/harm/coverage；
7. 经过第 8 节全部 threshold；同一 persistent candidate identity 在连续两次最新 graph
   revision 中都获胜才 latch；`(session, goal, plan, revision)` 重试幂等且不增加确认数；
8. 未通过：执行已缓存的 `a0`；通过：read-only mixed image+point resample 并执行前
   8 frames；
9. 每 plan 重新检查；candidate identity 改变会重新确认；连续 residual 最多 3 plan；
10. residual 后下一 plan 必须提交 typed feedback；缺失/错配或 graph displacement、native
    novelty、goal evidence 全无改善时，当前 plan 强制 native、清 latch，至少一个完整 plan
    cooldown；
11. 接近 matched anchor 后自动切回 image-dominant final alignment；
12. goal image 改变只 reset goal posterior/confirmation，不 reset persistent graph/FIFO。

任何异常都记录 explicit reason：`invalid_or_uncertain_context`、`no_eligible_residual`、
`duplicate_candidate_identity`、`confirming_residual`、`residual_burst_limit` 等，不能静默
使用零 covariance、重复 row 或旧 candidate。

延迟 gate 同样冻结：`K_residual <= 32`；abstain path 新增 p95（LingBot 增量 graph + batched
feature/set head，不含本来就要跑的 native）必须 `<=150 ms` 且 `<=20%` native-plan p95；
activation 只允许多一次 selected mixed resample，总 p95 必须 `<=2 * native-plan p95 +
150 ms`。超预算先降 K/缓存 feature，而不是增加分支网络或逐 candidate NavDP sampling。

## 10. 从 2-leg 扩展到 3-leg

### 10.1 统一语义

2-leg：

```text
A Novel: native by default；停滞且 confident no-match 时可 frontier residual
B Revisit: confident match -> graph route；不确定 -> native
```

3-leg：

```text
A Novel: 同上
B Novel: 保留 A 的完整 graph；新 goal posterior reset；不清 FIFO；必要时 frontier residual
C Revisit: goal 匹配 A-era nodes；在 A+B 累积 pose graph 上规划到 anchor
```

V2 不需要知道当前叫 A/B/C，也不读取 metadata 中的 goal kind；`goal_epoch` 只由 goal
image hash/外部合法 goal-change event 标识，用于避免把上一个目标的 posterior 带到下一个。

### 10.2 为什么 C 应用 pose graph，而不是简单 reverse replay

3-leg C 可能在 leg-A，但机器人已经走过 B detour。graph path 可以：

- 沿 sequential edge 安全回放已走路径；
- 在 learned loop factor 高置信时走拓扑 shortcut；
- 用 covariance 判断 shortcut 是否值得；
- 逐 node 重定位，不要求一次 LingBot pose 跨数百帧仍精确。

### 10.3 三层评测，避免分母塌缩

1. `conditional-B`：重放完全相同 A source prefix，只评 Novel B；每个 episode 都可评；
2. `conditional-C`：重放完全相同 A/B source prefix，只评 Revisit C，并含 oracle
   candidate/oracle pose upper bound；
3. end-to-end A→B→C：最后才报告真实 joint SR。

每层都必须共享 prefix hash、NavDP FIFO decision frames 和逐 request seed。否则 B 的路径
变化会改变 C 初态，不能把 C 差异归因给 memory。

## 11. Go / No-Go 门槛

### 11.1 Candidate generator

在任何 ranker 训练前：

- 先以 Pathfinder teacher pose 报 proposal-reachability proxy，再以部署路径
  `cam_pose_enc + ground_h_est scale + x-z` 独立重报；不能拿 teacher-pose `6/6` 代替
  deployment-pose coverage；
- development eligible state 的 proposal progress-positive recall `>=95%`；
- scene-bootstrap 95% lower bound 不低于 90%；
- 最差 scene 不低于 85%；
- native、small/large frontier、topological branch 每类单独报 recall；
- 通过 proposal gate 后，对 frozen top-K 与更宽 universe 做 paired H24 NavDP rollout；只有
  `coverage_label_valid=true` 的 state 才进入 useful-coverage 统计和 coverage loss；
- 不满足时 **No-Go ranker**，先修 candidate generation。当前 local single-scale `5/6`、
  multi-scale `6/6` 只是 teacher-pose proposal proxy，不是正式 scene-disjoint 或
  deployment-pose coverage 结论。

### 11.2 Offline model

Memory-localization head 至少超过当前 development reference：candidate top-1 `30/35`、
joint localization `101/118 = 85.6%`、match AUC `0.927`、Brier `0.095`；pose p90 相比
raw LingBot 至少降 15%，covariance-error Spearman `>0.4`。

Novel frontier head：

- progress-positive top-1 至少比单 CLS 和 goal-blind 两者高 10 pp；
- activation subset 的 mean paired advantage `>0.25 m`；
- catastrophic candidate（progress `<=-1.0 m`）activation rate 的 episode-bootstrap
  95% upper bound `<=5%`；
- calibration ECE `<=0.05`，advantage 90% interval coverage 在 0.87–0.93；
- shuffled-goal activation 不高于 correct-goal 的 10%，且不能出现 goal-swap 后 rank
  基本不变的旧 collapse；
- coverage `>=20%` 才有实际价值；若只有极少 activation，即使零错误也不算 Go。

### 11.3 逻辑/等价 gate

- abstain path：trajectory bytes、FIFO lengths/content、seed echo 100% exact；
- 同 observation 不可 double append；
- feature 缺失、NaN、stale graph、unknown source、low coverage 全部 native；
- goal switch 取消旧 confirmation；
- residual burst 必须有硬上限。

### 11.4 20-scene closed-loop development

- 共享 Goal-A trace 与 seed；native reference 仍是 Novel `31/40`；
- residual arm 的 abstained episode 必须与 native bit-exact；
- 31 个 native Novel success 中 paired regression 必须为 `0`；总 Novel A 不低于
  `31/40`；
- 若 residual 只在原生失败上激活却无 gain，则 No-Go；
- Revisit localizer + graph 至少达到既有 training plan 的门槛 `22/31`，且最多损失一个
  R0 的成功；
- graph 与 direct 必须在 strict shared-prefix runner 中重做，不能引用 confounded
  `25/40` 当 final。

### 11.5 Conditional 3-leg

- conditional-B 至少 20 个 scene/episode，relative native 有 `>=3` paired gains、`0`
  loss 才进入 end-to-end；
- conditional-C 分别报 candidate oracle、pose oracle、graph controller，不能只报 joint；
- goal switch 不清 FIFO 的主臂必须与 reset-B ablation 分开；
- end-to-end 同时报 `A SR`、`B|A`、`C|AB` 和 joint，禁止只报 `0/10 joint`。

### 11.6 Final blind

冻结 model SHA、threshold、candidate K、frontier generator、graph spacing、burst、seed
protocol 后，只运行一次 16-scene / 32-episode manifest。任一开发门槛没过都 No-Go blind。

## 12. 今夜训练链

今晚应分成可立即运行的 memory lane 与需要先产标签的 Novel lane。

### 12.1 Gate 0：本地逻辑与 provenance（已可运行）

```bash
cd /home/asus/Research/Nav-graph-blind
python -m unittest -v MemNavData.test_novel_memory_residual_v2
python -m unittest -v MemNavData.test_novel_candidate_set_schema_v2
python -m py_compile \
  MemNavData/novel_memory_residual_v2.py \
  MemNavData/test_novel_memory_residual_v2.py \
  MemNavData/novel_candidate_set_schema_v2.py \
  MemNavData/test_novel_candidate_set_schema_v2.py
sha256sum \
  .diagnostics/goal_conditioned_frontier_probe_20260807/report.json \
  .diagnostics/goal_conditioned_frontier_probe_20260807/multiscale_report.json \
  MemNavData/strict_graph_blind_20260806.json
```

任何 target file dirty、checkpoint/manifest SHA 不符或 output dir 已存在都 fail fast。

### 12.2 Lane M：LingBot-native match/pose head（现有脚本链）

现有可复用链是：

```text
slurm_lingbot_native_localizer.sbatch
    -> complete/resumable 40-scene exact LingBot rows
    -> audit_lingbot_native_localizer_artifact.py
    -> slurm_train_lingbot_native_localizer.sbatch
    -> train_lingbot_native_localizer.py
```

先完成/恢复 train collector，再以 `afterok` 提交 CPU Phase-B。已知旧 job `15430677`
因 episode CUDA cache 无界增长，在 `872/1244 rows` 后约 79 GB OOM，且旧实现未产生可
恢复 artifact；只有带 LRU=1、SQLite session transaction、signature-checked resume 的新
collector 输出通过 auditor 后才能训练。development exact rows 已记录为 job `15421650`
的 `310 rows / 78 sessions`，只用于最后一次评估/校准。

这一 lane 训练 match/no-match、anchor rank、pose residual/covariance；它不训练 Novel
frontier advantage，也不修改 NavDP。

### 12.3 Lane F：Novel frontier residual（今晚只做 cache→small-head）

第一阶段 causal sampling manifest 已实现：

```text
build_novel_candidate_manifest.py
  -> train/development only
  -> 每 scene 2 个有效 3-leg episode
  -> Goal-B switch 与 A-relative 8-frame midpoint
  -> factual + 同 scene paired-episode counterfactual goal
  -> RGB/depth/parquet-prefix/environment/navmesh/content SHA
  -> missing flow-cache inventory + atomic canonical JSON/SHA sidecar
```

本机真实 2-scene / 4-episode / 16-sample smoke 通过，flow cache 缺失为 0。冻结 local
split 是 `nlsr_local_smoke_split_20260807.json`（SHA
`8d99f67c331812eac406b5057f403e842a69b752aec6379ae35684321818ba6e`），加入显式
NavDP FIFO 后的 canonical manifest SHA 为
`33cc55f1d8c441a06ab1fa97382368b861c9a25d5ce32da85bbe9328cdf9e412`。
它 lazy-import `pyarrow` 并在缺依赖、列/shape/行数异常或非有限 pose/action 时 fail closed。
future parquet row 不改变较早 prefix hash，prefix pose/depth 改动必须改变 hash。
每个 expert state 还冻结 8-frame FIFO：前 7 个 replay RGB、当前 RGB、左侧 zero-padding
数量和内容 SHA；factual/counterfactual 必须共享完全相同的 FIFO record。

以下 candidate/label builder 与 trainer 仍未实现，不能把名称当成现有可运行脚本：

```text
[IMPLEMENTED] build_novel_candidate_manifest.py
[TO IMPLEMENT] build_novel_frontier_candidates.py
  J0-P proposal smoke: local 2 scenes/6 episodes，复现 teacher-pose single-scale 5/6、
      multi-scale 6/6，以及 DINO/goal-blind = 3/4；另报 deployment-pose arm
      这里只生成 proposal_proxy_report.json，不伪装成 candidate-set utility row
[TO IMPLEMENT] collect_novel_rollout_labels.py
  J0-U utility smoke: 同场景合法 multi-goal，对 native/top-K/universe 做同 state、同 seed
      H8/H24 frozen-NavDP rollout；每个 record 与完整 artifact 必须通过 schema
  J1 train40: causal expert + frozen-NavDP on-policy prefixes
      -> atomic candidate_set.parquet + feature cache + provenance JSON
      3-leg flow cache 当前少 train scene YmJkqBEsHnH；必须在我们的 scratch 补算并审计，
      不能把 40-train-scene 静默改成 39

[TO IMPLEMENT] train_novel_residual_ranker.py
  J2 CPU/GPU-small: 3 scene-bootstrap seeds, <=200 epochs, patience 25
      AdamW, lr 3e-4, wd 1e-4, batch 32 decision sets
      -> model seeds + OOF predictions; 不读取 development

[TO IMPLEMENT] calibrate_novel_residual.py
  J3 CPU: freeze ensemble/conformal/threshold on 10 development scenes once
      -> calibration.json + risk-coverage report

[TO IMPLEMENT] shadow_novel_residual.py
  J4 Habitat shadow: 20-scene shared-prefix，只记录 would-activate，不改 action
```

J0-P teacher-pose proxy 若不能逐条复现 `6/6`，今晚到此停止；复现后仍须分别通过
deployment-pose proposal gate 与 J0-U paired-rollout utility gate，不能拿 local `6/6`
代替。J1 必须阶段性落盘并支持 resume。J2 的 GPU 需求很小，不应为了小 head 占 H200
八小时；GPU 主要留给 LingBot feature collection。

### 12.4 今晚允许的终点

最理想的今晚输出是：

1. 完整可审计的 LingBot train artifact；
2. Phase-B memory localizer report；
3. Novel candidate coverage report 与 causal feature cache；
4. 若 coverage gate 通过，再有三 seed offline rank/advantage/harm report；
5. 只做 shadow activation，不做 final blind，不宣称 closed-loop SR 提升。

## 13. 必报日志

- frozen NavDP/LingBot/model/split/teacher/candidate-cache SHA；
- native/residual FIFO hash、request seed、goal epoch、graph revision；
- match interval、coverage interval、advantage LCB、harm UCB；
- pose translation/yaw p50/p90、depth confidence、clearance lower bound；
- candidate source/count/oracle coverage（training/development only）；
- activation/abstain reason、confirmation count、burst count、fallback rate；
- correct-vs-shuffled goal rank/sensitivity；
- Novel A、Novel B|A、Revisit C|AB、joint、SPL 与 paired transitions；
- latency 分解：LingBot stream、graph update、ranker、NavDP native、resample。

## 14. 最终建议

旧 MemNav 的失败并不要求“重新把 memory 训得更久”，而要求重新划清职责：

- NavDP 已有的 ImageGoal 与 local collision-control 能力是资产，必须冻结；
- LingBot 的独特价值是持久 pose/depth/patch graph，不是再造一个小 diffusion policy；
- Novel 的新增能力应是 goal-conditioned graph/frontier direction，并且只作为相对 native
  的 selective residual；
- uncertainty 的正确动作不是 RANSAC 工程分支，而是 abstain 到原生 NavDP；
- 3-leg 的正确实验顺序是先把 Novel-B 的 conditional denominator 做满，再评 long-memory
  C，最后才看 end-to-end joint。

因此下一次真正值得花 H200 的任务是 causal LingBot candidate/feature collection；下一次
值得训练的只是小型 rank/value/risk/pose heads。任何会更新官方 NavDP encoder、decoder
或 critic 的 run，在 V2 中都应默认 No-Go。
