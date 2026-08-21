# Pi3X Learned Relocalizer 与 Final14：最新完整状态总账

更新时间：2026-08-18 03:43 CST（Asia/Shanghai）  
文档性质：截至当前的**最新状态总账**；覆盖 2026-08-17 learned relocalizer
路线、Pi3X/LingBot-Map 分工、训练数据与泛化审计、论文 storytelling、
2026-08-15--17 三日完整时间线，以及 Final14 正式确认链。

> **2026-08-18 completion addendum：**Final14 已 `42/42` 完成，summary 与
> independent verifier 均成功，`verified=true`。正式数字、统计、功效边界和冻结后
> 决策以 `FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md` 为准。本文后续保留的
> “pending/prospective”文字属于运行前时间线，不再代表当前状态。
>
> **Latency addendum：**raw-plan 只读审计发现旧 summarizer 将首次 CEC latency 随
> cache-hit replans 重复计入。正确 Natural query-level 首次 median 为 `3.40 s`
>（`n=42`），后续 cache-hit update median 为 `0.152 ms`（`n=1346`）；只影响延迟
> 口径，不影响任何 SR 或 verifier 结论。详见
> `FINAL14_CEC_CACHE_LATENCY_AUDIT_20260818.md`。

前序项目总账仍保留在：

- `STATUS_20260814_PAPER_EVAL.md`：CEC、2-leg/3-leg、Attempt 7、Novel、
  X-NavDP 与早期跨数据集结果；
- `STATUS_20260815_NIGHTLY_PAPER_CONVERGENCE.md`：论文收敛、GOAT repair、
  Novel 因果对照；
- `NON_MP3D_EXTERNAL_EVAL_STATUS_20260816.md`：HM3D/Gibson/GOAT 等外部评测；
- `LEARNED_RELOCALIZER_NIGHT_GOAL_20260817.md`：本轮 learned route 的完整开发日志。

本文不改写已冻结协议，也不把正在运行的任务写成结果。状态标签沿用：

- **confirmed**：协议、分母、配对、统计与独立复算均通过；
- **strong internal**：证据强，但场景已用于开发或不是最终 scene-disjoint 确认；
- **underpowered**：协议有效但 scene/history 数不足；
- **mechanism**：证明能力或结构，不等于可部署方法；
- **prospective**：方法与判据已冻结，但正式结果尚未打开；
- **infrastructure-only**：只发生在任何方法结果产生之前，不能解释为科学结果。

## 0. 一页结论

### 0.1 当前最准确的项目判断

项目现在有两条清楚、但正式等级不同的路线：

1. **CEC（Certified Episodic Compass）**仍是已经获得闭环支持的
   task-training-free 主方法：DINO 提议历史位置，显式几何与 PnP 提供证据，
   certificate 决定接管或精确回退 native NavDP。
2. **Pi3X Learned Relocalizer**是 CEC 的已完成 prospective test 的 learned
   replacement candidate：
   保留 DINO 作为长历史地址簿，用 Pi3X 的因果视觉桥恢复方向，再由四个小型
   spatial proof heads 学习“证据是否足以授权接管”。它在 Final14 standard
   Revisit 达到 `19/21`，显著超过 native `4/21`，但比 CEC 的 `20/21` 少一条，且
   未通过预注册的 L2 non-inferiority 与 L3 catastrophic-bearing safety 门。

因此，当前不能写成“learned 方法已经替代 certificate”。最强而诚实的口径是：

> CEC 通过 proof-before-control 将因果历史转为高效 Revisit bearing，同时避免
> always-on raw memory 在 unsupported Novel 上的大量干扰；learned Pi3X 可以复现
> 大部分闭环 utility，但其 proof 尚不足以替代显式 certificate。

### 0.2 learned arm 到底是不是 training-free

需要分三层说：

| 组件 | 当前是否训练 | 作用 |
|---|---|---|
| NavDP diffusion policy | 预训练后冻结 | 执行 ImageGoal 或 2.5 m residual |
| DINO embedding | 预训练后冻结 | 从长历史中提取 top-8 地址 |
| Pi3X | 官方预训练权重冻结 | 多视图几何、相对方向与空间证据 |
| spatial proof heads | **在 PT1 Train40 训练** | 学习 actionability/support 与 abstention |

所以：

- CEC 是**无本项目任务训练**的方法，但不是“没有 learned component”；
- 新 arm 不是 training-free，因为四个 proof heads 是任务特定训练的；
- 它也不是端到端重新训练 NavDP：controller、DINO 和 Pi3X 都冻结，只有授权层被学。

### 0.3 Final14 最终状态

截至 `2026-08-18 03:43 CST`：

- natural-direction `21/21` 与 hard-support `21/21` histories 全部完成；
- `42/42` completion receipts 的 SHA-256 全部通过；
- policy summary `15903547` 与 independent verifier `15903548` 均
  `COMPLETED 0:0`；
- independent verifier：`verified=true`；
- natural Revisit：native `4/21`、raw `19/21`、geometry `18/21`、
  learned `19/21`、CEC `20/21`；
- CEC 对 raw 的两类 query 合计为 `28/42` 对 `21/42`，配对 `+8/-1`，
  `p=0.0391`；
- learned promotion：L1 通过，L2/L3 未通过，CEC 保持 primary；
- natural population 为 `21 histories / 10 scenes`，低于冻结的 28-history
  target，必须保留 underpowered 标签。

## 1. 为什么从 CEC 继续走向 learned relocalizer

### 1.1 CEC 已经解决了什么

冻结 CEC 的外部行为是：

```text
actual-online causal RGB history + ImageGoal
    -> DINO temporally-diverse top-8
    -> SuperPoint + LightGlue 几何对应/候选排序
    -> LingBot historical depth + PnP 相对位姿
    -> atomic geometric certificate
         pass: scale-free bearing -> fixed 2.5 m residual -> frozen NavDP
         reject/error: exact native ImageGoal NavDP
```

进入本轮 learned 工作前，关键闭环证据包括：

| 评测 | native | CEC/相关记忆臂 | 当前解释 |
|---|---:|---:|---|
| 最早 40 条 geometry memory | 4/40 | 19/40 | `+15/-0`, `p=6.1e-5`，记忆有效 |
| Fresh160 supported Revisit，B given A | 27/120 | certified 112/120 | strong internal，近饱和 |
| Attempt 7 held-out natural Revisit | 2/9 | certified 8/9 | `+6/-0`, `p=.03125`，但 underpowered |
| actual-online 3-leg NNR，C given A/B | 5/19 | role-free certified 16/19 | `+11/-0`, `p=.0009766` |

CEC 的难点不再是“它有没有用”，而是：

- 它由 retrieval、feature matching、depth replay、PnP 和手工阈值组成，论文上容易被
  评价为一条有效但工程化的 pipeline；
- 在高支持 Revisit 上，CEC 尚未显著超过简单 raw-DINO fixed bearing：
  Fresh160 为 certified `112/120`、raw direct `106/120`，配对 `+9/-3`,
  `p=.146`；Attempt 7 的 Revisit 都为 `8/9`；
- explicit certificate 的价值目前主要是开放集授权、安全拒绝和 exact fallback，
  而不是已证明更高的饱和 Revisit SR。

learned route 的目标不是为了“让论文看起来有训练”，而是用同一个可学习的几何
relocalizer 内化 CEC 的核心原则：**证据不足就 abstain**，同时删掉显式
SuperPoint/LightGlue/LingBot-depth/PnP/certificate 实现链。

### 1.2 为什么没有把整段历史直接塞进一个 Transformer

这个问题已被先前实验否决，而不是未经测试的直觉：

- true Revisit-C 上，DINO-addressed GCT 为 `18/20`；
- full-prefix candidate-free GCT 只有 `5/20`；
- 配对 `+0/-13`, `p=.000244`。

长程流式 memory 能维护近期上下文，不等于能在数百帧中做开放集内容寻址。因此
当前“统一”只统一 **post-retrieval relocalizer**：DINO 继续负责地址，Pi3X 和
learned proof 负责几何与授权。除非新实验推翻 `5/20 vs 18/20`，否则不再把
candidate-free 端到端历史搜索包装成更优雅的方案。

## 2. Pi3X 和 LingBot-Map 到底有什么区别

它们都能从单目 RGB 推断三维结构，但解决的问题不同。

| 维度 | LingBot-Map | Pi3X（当前项目用法） |
|---|---|---|
| 基本范式 | 有状态、连续流式的 feed-forward 3D mapping/SLAM backbone | 对一次有限多视图集合做联合、近似 reference-free 几何推断 |
| 输入组织 | 按时间持续输入 RGB，维护 anchor/window/trajectory state | 当前帧、若干因果 bridge/support 帧和 ImageGoal 一次联合前向 |
| 主要输出 | 连续轨迹中的相机位姿、深度/点云与内部地图状态 | 各视图相机位姿、world/local points、confidence 和 hidden tokens |
| 擅长 | 连续跟踪、在线重建、维持局部/长序列坐标关系 | query-conditioned 多视图相对几何和当前到目标的方向 |
| 不自动解决 | 不是 goal-indexed episodic database；官方方案没有显式 loop closure 搜索器 | 没有永久 memory；不会自己在数百帧历史里找 anchor |
| 当前项目的 memory | 其 streaming state 可帮助轨迹一致性；CEC 仍需外部 RGB history/DINO retrieval | 完全依赖外部 causal RGB/keyframe buffer 与 DINO shortlist |
| 单目尺度 | 当前 CEC 不信任跨段 metric translation scale | 当前部署同样只信任方向，不使用 metric range |
| 最合适角色 | 连续 VO/深度/地图 backbone；未来 metric/global subgoal 支撑 | 检索后的 episodic relocalizer 与 learned geometric witness |

更直白地说：

- **LingBot 更像持续建图和追踪者**；
- **Pi3X 更像拿到一组候选视图后，重新解一个 query-conditioned 多视图几何问题**；
- **真正的长程记忆都不是 Pi3X 本身**。当前长程记忆是因果 RGB history、索引、
  时间关系和 DINO descriptor；Pi3X 只在取出候选后工作。

### 2.1 LingBot 在旧 CEC 中承担了什么

旧 CEC 不是“LingBot 自动发现 Revisit”。真实分工是：

1. DINO 从历史中提出 top-8 位置候选；
2. SuperPoint/LightGlue 找局部对应并选几何 anchor；
3. `policy_agent.py::_certified_reference_depth` 重放选中历史帧，取得 LingBot
   reference depth/confidence；
4. 把 reference 2D 点反投影为 3D，通过 2D--3D PnP 恢复目标相对位姿；
5. atomic certificate 检查 inliers、双侧 hull coverage 和 reprojection RMSE；
6. 丢弃不可信的单目尺度，只输出单位 `[forward,left]` bearing；
7. 通过固定 `2.5 m` residual 交给 frozen NavDP，失败则 exact native fallback。

所以 LingBot 提供的是**历史深度和 episode-internal 几何坐标支撑**。它不是目标
检索器、Novel/Revisit 分类器，也不是 navigation controller。

旧设计还保留了 full-patch 滑窗和更老帧的压缩 token/state，这有利于连续 streaming，
但先前 candidate-free GCT 结果说明这类压缩 state 不能替代长程内容地址。

### 2.2 Pi3X 当前承担了什么

当前 learned arm 的正式输入输出契约为：

```text
actual-online causal RGB history + ImageGoal
    -> frozen DINO top-8
    -> 对每个候选构造：
         live current
         + b16 causal bridge
         + anchor support offsets [-8, 0, +8]
         + ImageGoal
    -> frozen Pi3X
         camera poses + point maps + confidence + hidden features
    -> raw Pi3X cross-view overlap 选择一个 proposal
    -> four learned spatial proof heads, >=2/4 model-bound votes
         accept: scale-free bearing -> fixed 2.5 m -> frozen NavDP
         reject/error: exact native NavDP
```

该 arm 的 Pi3X 前向不读取 simulator pose、simulator depth、Novel/Revisit role、
LightGlue/PnP 指标或 atomic certificate feature。代码在
`pi3x_online_relocalizer.py` 中显式声明和审计这些禁用输入。

### 2.3 一个必须保留的实现口径

方法逻辑上，learned Pi3X arm 不消费 LingBot pose/depth/GCT tokens；但当前正式五臂
进程仍需同时装载 incumbent CEC，而 DINO shortlist 也暂时通过 LingBot wrapper 内
已有的冻结 DINO trunk 计算。因此：

- 可以说“learned arm 替代了 LingBot-depth + PnP + explicit certificate”；
- 现在还不应说“整个运行时完全没有 LingBot dependency”；
- 若 Final14 通过，下一步再把 DINO encoder 从 LingBot wrapper 中独立出来，才能
  在工程依赖层面真正移除 LingBot，而不改变方法。

这项重构应在 Final14 后做，不能为了叙事在正式比较中改变运行栈。

## 3. 因果视觉桥为什么是 learned route 的关键

### 3.1 失败归因

最早 Pi3X smoke 用 evaluator trajectory 做 Sim(3) 对齐后，true-pair median bearing
error 只有 `2.35 deg`，但这种对齐部署时不可用，只能算 oracle-like upper bound。

去掉外部对齐、把 live current 加进 Pi3X，并只提供旧 anchor 周围五个局部帧时，
true-pair median bearing error 变成 `88.17 deg`。原因不是“Pi3X 没有几何能力”，
而是 current 与很久以前的 anchor 可能完全没有直接共视，联合模型无法把两套局部
坐标连接起来。

### 3.2 结构性修复

因果视觉桥从**已经真实观察过的**当前到 anchor 的轨迹中均匀采样重叠帧，再加入
anchor 附近支持帧。它不注入 GT pose 或新视角，作用相当于一条可观测的视觉坐标
传递链：

```text
current <-> recent bridge <-> ... <-> old bridge <-> anchor <-> ImageGoal
```

这不是一般性“把帧数从 8 调到 16”的调参。它针对的是经诊断的断连问题，而且增益
集中在 history gap `>96` 帧的长程样本。

### 3.3 完整 Train40 配对证据

| 配对指标 | b8 | b16 | 变化 |
|---|---:|---:|---:|
| positive candidate bearing `<=30 deg` | 585/701 | 659/701 | `+82/-8`, exact `p=1.38e-16` |
| raw-DINO session top-1 direction | 130/155 | 144/155 | `+16/-2`, `p=.001312` |
| candidate-set oracle ceiling | 149/155 | 153/155 | `+4/-0`, `p=.125` |

固定 b16 后，用 raw Pi3X cross-view overlap 在 DINO top-8 中选择 proposal，达到
`147/155` navigation-direction top-1。

这里的 target 是**方向是否可用于导航**，不只是“是否找到了 teacher 定义的同一
anchor”。早期两场景机制审计中，teacher-positive anchor top-1 仅 `4/8`，但选择的
bearing 在 `7/8` 中仍小于 `30 deg`；这解释了为什么过去某些 candidate AUC 提升
没有变成闭环提升，也要求后续始终用方向和 SR 做最终判据。

## 4. learned spatial proof 的结构

Pi3X overlap 已负责 proposal。proof head **不再改变候选和 bearing**，只回答：

> 这次 Pi3X proposal 的空间证据是否足以授权 memory takeover？

每个 b16 forward 的 frozen archive 包含 20 个 view 的证据：

- `9 x 16` 的 9-channel spatial grid：以 live-current camera gauge 表达的 world
  points、local points 和 confidence；
- 每个 view 的相对 `3 x 4` camera pose；
- Pi3X global/register descriptor；
- temporal role、relative age 与 validity mask。

坐标按 positive current-view depth 的中位数归一化，因此 proof 学的是尺度不变的
几何一致性。模型为：

```text
9-channel spatial grid/view -> shared small CNN
relative pose + global token + temporal role/age -> embeddings
all views -> 2-layer view Transformer
             -> actionability head
             -> support head
```

单个 head：

- `311,426` 个可训练参数；
- `model_dim=64`，2 个 Transformer layers，4 heads，dropout `0.1`；
- 30 epochs，AdamW，learning rate `3e-4`，weight decay `1e-3`；
- gradient clip `1.0`，support loss weight `0.25`。

部署为四成员 ensemble，总 task-specific 参数约 `1,245,704`。Pi3X 与 DINO 均冻结。

## 5. 训练数据到底够不够

### 5.1 表面样本数与真正独立样本数

Train40 archive 有：

| 单位 | 数量 |
|---|---:|
| scene clusters | **40** |
| sessions | 480 |
| DINO top-8 candidate rows | 3,840 |
| positive sessions | 155 |
| strict-negative sessions | 282 |
| ambiguous sessions | 43 |

按 candidate row 展开：

| row label | positive | negative | ambiguous |
|---|---:|---:|---:|
| session-level existence | 1,240 | 2,256 | 344 |
| navigation actionability | 1,143 | 2,353 | 344 |

1,143 个 positive action candidate 覆盖全部 40 scenes；每 scene 的数量为最少 2、
中位 24、最多 80。

这回答了“数据是不是太少”：

- 对一个冻结大 backbone 上的 31 万参数小 head 来说，candidate pair 数并非小到
  完全不可学习；
- 但 3,840 rows 高度相关，不能当作 3,840 个独立样本；
- 泛化统计的真实有效样本量接近 **40 个场景**，这仍然偏小，尤其不足以宣称
  cross-domain 或真实机器人泛化。

### 5.2 scene-crossfit 如何防止直接泄漏

正式 OOF 采用五个 outer folds：

1. 每个 outer fold 留出 8 个完整场景；
2. 剩余 32 场景内部再形成 4 个成员；每个成员在 24 场景拟合、另 8 场景校准；
3. threshold 与具体 checkpoint 绑定；
4. 对 outer 8 unseen scenes 只应用模型和阈值，不再重调；
5. 部署 ensemble 重新用四个 `30 fit / 10 calibration` 的 scene-disjoint 成员训练；
6. takeover 需要预冻结的 `2/4` 票。

development、blind、Fresh160、Attempt 7 和 Final14 都没有用于 head fitting 或
threshold calibration。

但是必须区分两种 overfit：

- **样本泄漏**：crossfit 已经较严格地避免；
- **研究者/架构 meta-overfit**：b8/b16、global-token V1--V3、spatial evidence
  等设计都在同一个 Train40 上迭代，这仍然存在，OOF 不能自动消除。

## 6. Train40 OOF 结果与泛化审计

### 6.1 冻结 `2/4` consensus 的 aggregate 结果

| 指标 | 结果 |
|---|---:|
| Pi3X overlap proposal direction top-1 | 147/155 |
| correct positive accepts | 119/155 |
| positive recall | **76.77%** |
| accepted known sessions | 125 |
| accepted precision | **95.20%** |
| strict-negative false accepts | 4/282 |
| strict-negative FPR | **1.42%** |
| accepted bearing median | 3.41 deg |
| accepted within 15/30/45/90 deg | 110/123/124/125 |
| accepted above 90 deg | **0** |

这说明 learned proof 在同为 MP3D 的 scene-held-out outer folds 上确实学到了可迁移
的空间一致性，而不只是记住候选 row。

### 6.2 每个 outer fold 的稳定性

| fold | precision | positive recall | FPR | correct/positive | accepts | `>90 deg` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.00% | 94.74% | 2.94% | 18/19 | 20 | 0 |
| 1 | 95.65% | 62.86% | 1.69% | 22/35 | 23 | 0 |
| 2 | 100.00% | 81.82% | 0% | 27/33 | 27 | 0 |
| 3 | 92.59% | 73.53% | 0% | 25/34 | 27 | 0 |
| 4 | 96.43% | 79.41% | 2.22% | 27/34 | 28 | 0 |

安全性在五折上相对稳定：precision `90--100%`，五折均无接受的灾难方向；但
coverage/recall 在 `62.86--94.74%` 间明显波动。这是“有泛化迹象但仍 scene-
sensitive”的直接证据。

### 6.3 consensus 的代价

| consensus | precision | recall | FPR | catastrophes |
|---|---:|---:|---:|---:|
| 1/4 | 86.62% | 87.74% | 6.38% | 1 |
| **2/4（冻结主点）** | **95.20%** | **76.77%** | **1.42%** | **0** |
| 3/4 | 96.55% | 54.19% | 0.71% | 0 |
| 4/4 | 96.55% | 36.13% | 0.71% | 0 |

`2/4` 不是在 Final14 上选出来的；它是在 spatial-head 结果前，依据 global-token
proof 中满足 precision/FPR/catastrophe gate 的最宽松点冻结的。

### 6.4 与旧 certificate 的同终点比较

旧 certificate 的 `122/153=79.74%` 原始 recall 使用 metric position error
`<=0.75 m` 标签；learned arm 的部署终点是 bearing `<=30 deg`，两者不能直接横比。
统一到相同 directional endpoint 后：

| directional endpoint | old certificate | learned spatial proof |
|---|---:|---:|
| correct positive accepts / 155 | 107 | **119** |
| precision | **97.27%** | 95.20% |
| strict-negative false accepts / 282 | **2** | 4 |
| FPR | **0.71%** | 1.42% |
| accepted `>90 deg` | 1 | **0** |

learned correct coverage 相对 certificate 为 `+28/-16`，McNemar `p=.0961`；
strict-negative false accepts 为 `+3/-1`, `p=.625`。因此：

- learned proof 的方向 coverage 更高、没有接受 `>90 deg`；
- certificate precision/FPR 仍略好；
- **learned 尚未被统计证明优于 certificate**；
- 这个结果足以授权 fresh closed-loop non-inferiority test，不足以直接换主方法。

### 6.5 最终部署 ensemble 的校准波动

四个 final deployment member 的校准表现为：

| member | threshold | calibration precision | recall | FPR |
|---:|---:|---:|---:|---:|
| 0 | 0.86646 | 96.88% | 83.78% | 1.30% |
| 1 | 0.98512 | 100.00% | 7.89% | 0% |
| 2 | 0.96495 | 96.77% | 76.92% | 0% |
| 3 | 0.98781 | 95.65% | 53.66% | 1.49% |

模型绑定 threshold 和 `2/4` consensus 能缓解分数尺度漂移，但成员 recall 从
`7.9%` 到 `83.8%` 的差距是明显警告。四个成员是不同 scene partition 的 ensemble，
不是四次完全独立的全流程 multi-seed replication。

## 7. 关于“是否真的可泛化”的严格答案

### 7.1 当前证据支持什么

支持：

1. frozen Pi3X spatial evidence 在 MP3D Train40 内的完整 scene-held-out folds 上
   有可重复的授权信号；
2. 该信号不是只靠 global embedding：相同 proposal 下，spatial proof 相比
   global-token proof 同时提高 precision/recall 并降低 FPR；
3. 2/4 consensus 在所有五个 unseen-scene folds 上都保持零接受 `>90 deg`；
4. positive/negative consumed closed-loop transport smoke 已证明 runtime、sticky
   rejection、fixed anchor、one-anchor update 和 exact fallback 能真实执行。

### 7.2 当前证据不支持什么

尚不支持：

1. **最终四个 deployment checkpoints** 在完全未消费场景上的闭环泛化；OOF
   模型与 final deployment refit 不是同一组 checkpoint；
2. MP3D 到 HM3D/Gibson/Replica/真实相机的 cross-domain 泛化；
3. learned arm 已经达到或超过 CEC 的 fresh SR；
4. calibration 在不同建筑分布上仍稳定；
5. 仅增加同一 40 scenes 的更多帧或候选就能解决泛化。

所以当前结论不是“数据一定太少、学不出来”，也不是“3840 pairs 已足够、可以泛化”。
准确结论是：

> 表征和任务已经学得动；当前不确定性主要来自独立 scene 数、同一 Train40 上的
> 架构迭代，以及 final refit checkpoint 尚未经过一次性 fresh test。

## 8. Final14 prospective formal test

### 8.1 冻结 arm 与生命周期

五臂严格共享 Goal-A trace、query/goal RGB、NavDP checkpoint、diffusion seed、预算
和 success criterion：

1. `native`；
2. `raw_fixed`；
3. `geometry_fixed`；
4. `certified`；
5. `learned_pi3x_spatial`。

learned lifecycle 已冻结：

- 只在 first causal query 计算 DINO top-8；
- first-query rejection 对该 goal sticky；
- first-query acceptance 后固定 anchor 和 DINO rank；
- 后续 replan 只重跑该一个 anchor，更新 current-relative bearing；
- 任意 reject/error 都 exact native fallback；
- runtime 不读取 Novel/Revisit role 或 support-band label。

### 8.2 预注册判据

**L1：Revisit utility**

- learned 对 native 必须 positive net gain；
- scene-cluster 95% CI 下界必须 `>0`。

**L2：相对 CEC non-inferiority**

- 定义 `Delta = SR_learned - SR_CEC`；
- cluster CI 下界必须 `>-10 pp`；
- paired point estimate 必须 `>=-5 pp`。

**L3：unsupported-Novel safety 与 exact fallback**

- 0 runtime-contract violations；
- 所有 abstention 必须在 seed、selected trajectory SHA、executed pose/RGB trace
  上精确复现 native；
- Novel 相对 native 不得有净损失；
- 接受 bearing 中 `>90 deg` 必须为 0。

**L4：coverage/efficiency（次要）**

- 标准与 hard-support Revisit 分开报告；
- 比较 coverage、takeover success/false takeover、first top-8 latency、one-anchor
  update latency、peak GPU memory 和 history storage；
- L4 不能挽救 L1--L3 的失败。

### 8.3 结果解释冻结规则

| Final14 结果 | 论文决定 |
|---|---|
| L1、L2、L3 全过 | learned relocalizer 可升为主方法；CEC 变为 task-training-free baseline/teacher |
| L1、L3 过，L2 不过 | CEC 保持主方法；learned 作为有前景但较弱的简化 |
| L3 不过 | learned 不具备 deployment qualification，无论 Revisit SR 多高 |
| 构造后分母不足 | 完整报告为 underpowered；不得追加 Final14、重调阈值或换 checkpoint |

Final14 即使通过，也只证明 fresh scene-disjoint **MP3D** 泛化，不自动证明跨数据集。

## 9. Final14 Attempt 1--4 基础设施审计

这些 attempt 都在正式结果读取之前处理，必须与方法失败分开。

### Attempt 1

- pre-unseal 缺 strict base manifest；
- 在 source/query/policy outcome 前失败；
- 0 科学记录，记为 infrastructure-only。

### Attempt 2

- 使用了错误的 host root，未挂载冻结 PT1 overlay；
- 0 source，0 policy outcome；
- 已取消，记为 infrastructure-only。

### Attempt 3

- frozen source manifest 本身正确：14 scenes、目标 112 sources、实际可用 80、
  4 scenes 低于 target；
- 但 zero-source scene builder 把合法空场景当错误；
- sealed bundle 缺 `materialize_online_a_traces.py` 的递归依赖；
- 取消前写入 8 条 Goal-A trace，但没有完成 materialized histories、role pairs、
  query arm 或 method comparison；
- 因此不存在可报告 SR，也没有方法污染。

### Attempt 4

已完成的 repair：

- empty source/materialized scene 会绑定显式验证过的 scene，并写出空 manifest/receipt；
- 对 repo-local Python imports 做递归闭包：122 个 Python 文件；
- 在远端正式 container + read-only overlay 中通过 import smoke 和 empty-scene test；
- 新 immutable bundle：
  `.diagnostics/learned_relocalizer_20260817/final14_learned_complete_f7acd766c0835548`；
- bundle receipt SHA-256：
  `f7acd766c08355485ea7a7349b0f181e8e7f37633dbeb86c3ff7ed75fee493d3`；
- 4,011 regular files、4,010 manifest entries、0 writable entries；
- Attempt 4 与 Attempt 3 的 scene/episode selection 完全一致；normalized selection
  SHA-256 为
  `dfbb333cae549eefe17eab55a8007b493aeea81936bf8c3b4102a11e29c3e035`；
- 没有改变 population、threshold、scene order 或方法。

HPC run root：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
final14_cec_learned_20260817/final14_learned_20260817T0715Z_attempt4
```

Job IDs：

| stage | job id | 16:02 CST 状态 |
|---|---:|---|
| Goal-A collection array | 15885114 | all 14 scene tasks complete |
| construction summary | 15885121 | completed, `0:0` |
| five-arm eval array | 15885125 | pending, `QOSGrpGRES` |
| policy summary | 15885128 | dependency pending |
| independent verification | 15885139 | dependency pending |

## 10. 下一步：只做能回答论文问题的工作

### P0：保持 Final14 冻结并等待 formal summary/verifier

现在不改：checkpoint、threshold、consensus、b16、DINO ordering、2.5 m residual、
fallback 或 population。也不从 collection 中间状态推测 SR。只有 policy summary 和
independent verification 完成后，才读取五臂科学结论。

### P1：根据 Final14 分叉，而不是继续在 Train40 调参

若 L1--L3 通过：

1. learned arm 升为论文 primary；
2. CEC 保留为 training-free baseline、教师和归因工具；
3. 做 runtime/memory、b8-vs-b16、global-token-vs-spatial、consensus ablation；
4. 把 DINO 从 LingBot wrapper 中拆出，验证 learned-only runtime 真正不依赖 LingBot；
5. 再冻结一个外部数据集协议，测试跨域。

若失败：

1. 不在 Final14 上调参或把失败样本并回训练；
2. CEC 继续作为主方法；
3. 扩充**独立训练 scenes**，而不是只从原 40 scenes 重采更多 candidates；
4. 新训练集可引入更多 MP3D train scenes，并单列 HM3D/Gibson 外部域；
5. 扩充视角、遮挡、光照和低共视 hard negatives；
6. 另建全新 untouched test，再进行下一次 prospective test。

### P2：真正回答“数据规模是否够”的最小实验包

Final14 之后再运行，避免影响当前冻结结论：

- scene-count learning curve：例如 8/16/24/32 scenes；
- 至少 3 个独立训练 seeds，报告 coverage 和 calibration 波动；
- precision--coverage/FPR--coverage 曲线，而不是只报一个 threshold；
- MP3D scene-held-out 与外部数据集严格分表；
- 只在新 scene 数增加时讨论泛化，不能用 correlated pair 数替代。

### P3：把研究实现的完整 RGB history 变成可部署 memory

当前 Pi3X arm 从磁盘读取完整 causal RGB history，科学上因果正确，但存储/延迟还不是
最终系统设计。通过 Final14 后可改为：

```text
dense recent buffer
    + sparse long-term keyframes
        {compressed RGB, DINO descriptor, timestamp, temporal adjacency}
    -> DINO address anchor
    -> sample temporal bridge from adjacency chain
    -> Pi3X + learned proof
```

这会减少存储与 I/O，但不应在 Final14 前改变，因为它可能改变 bridge 质量。

## 11. 论文定位与 storytelling：最终层级

### 11.1 不做三选一，而是分清四个层级

当前最准确、也最不容易被 reviewer 击穿的定位是：

| 层级 | 推荐表述 | 在论文中的作用 | 当前是否已经充分评测 |
|---|---|---|---|
| 长期愿景 | **toward lifelong navigation** | 解释机器人为什么应累积并复用自身经验 | 否，只能作为 motivation |
| 具体任务 | **monocular sequential ImageGoal navigation with episodic revisits** | 精确描述 actual-online 2-leg/3-leg 与 mixed-role 协议 | 是，当前主要实验都对应这一层 |
| 核心问题 | **goal-conditioned navigation loop closure** | 说明后来目标与历史位置发生对应时，要把视觉闭环变成控制 | 已有强 Revisit 证据，但不是传统 SLAM loop closure |
| 方法术语 | **causal episodic relocalization with proof-before-control** | 描述检索、因果桥、几何证据、abstention 和 bearing residual | CEC 已确认；learned 替代待 Final14 |

一句话定位：

> 本项目不是完整解决 lifelong navigation，也不是复现传统 SLAM loop closure；
> 它研究的是**面向 lifelong navigation 的、单目 sequential ImageGoal 中的
> goal-conditioned navigation loop closure**。

论文题目和摘要应优先使用 `sequential ImageGoal` 与 `episodic relocalization`；
`lifelong navigation` 放在动机和长期意义中，`loop closure` 用作直观的科学 hook。

### 11.2 为什么现在不能把任务直接叫 lifelong navigation

完整的 lifelong navigation 通常还要求：

- 跨许多 episode、天或任务持久保存 memory；
- 数十到数百个连续目标，而不只是 2/3-leg；
- 有界内存、keyframe 管理和遗忘策略；
- 环境变化后的更新、冲突处理与 stale-memory 检测；
- 长期性能随任务数和 memory size 的 scaling curve。

当前 history 主要在单 episode 内积累，Formal benchmark 仍以 2-leg/3-leg 为主，
尚未评测上述长期性质。因此可以写：

> Reusing self-acquired visual experience is a prerequisite for lifelong
> navigation.

但不能写：

> We solve lifelong navigation.

如果未来加入多目标长序列、跨 episode persistent memory、bounded buffer 和环境变化，
才有资格把 lifelong 从 motivation 升为 task claim。

### 11.3 为什么又不只是传统 loop closure

传统 SLAM loop closure 通常问“当前相机是否回到旧位置”，输出 6-DoF constraint，
再优化 pose graph。我们的问题不同：

| 传统视觉/SLAM loop closure | 本项目的 navigation loop closure |
|---|---|
| query 通常是当前帧 | query 是未来要到达的 ImageGoal |
| 判断当前是否重访旧地点 | 判断目标是否受 causal history 支持 |
| 输出相对 pose/pose-graph edge | 输出可执行的 scale-free bearing 或 abstain |
| 主要优化地图和轨迹一致性 | 主要提高冻结 controller 的闭环 SR |
| 默认 loop 应存在或离线筛选 | 必须处理 unsupported Novel 的开放集拒绝 |
| 错配污染地图 | 错配会直接改变动作，因此要求 proof-before-control |

所以 `loop closure` 描述的是核心现象，但技术名称用
`goal-conditioned episodic relocalization` 更严谨。我们不是建立一个全局 metric
pose graph，而是把目标图像与在线历史的视觉闭环转化成最小控制变量。

### 11.4 为什么不叫“单目 Novel 导航的 loop closure”

这个说法会制造两个错误期待：

1. 当前 Novel branch 并没有被新方法解决；没有可认证历史时仍由 frozen native
   NavDP 完全接管；
2. 完全 Novel 的目标在语义上没有历史闭环可关，`Novel loop closure` 本身容易矛盾。

正确顺序是：

```text
agent enters an initially unseen environment
    -> native NavDP explores a novel ImageGoal
    -> causal monocular experience accumulates
    -> a later ImageGoal may refer to an observed place
    -> goal-conditioned navigation loop closure
    -> supported: bearing residual
       unsupported: exact native fallback
```

因此我们解决的是：

> 如何把 Novel exploration 留下的单目经验，转化为以后 Revisit 目标的安全控制信息。

环境可以是 initially unseen；但方法 claim 必须是 Revisit/episodic reuse，而不是
Novel direction 已解决。

### 11.5 最强开场与核心命题

最有记忆点的开场是：

> **A revisit should not be treated as a novel navigation problem.**

展开后的论文主张：

> Frozen ImageGoal policies can execute a useful direction but discard the
> navigation value of their own past observations. We formulate experience
> reuse as goal-conditioned navigation loop closure: address a causal visual
> history, bridge long temporal gaps using only observed views, and alter the
> controller only when spatial evidence supports the inferred direction.

系统抽象为：

```text
episodic buffer remembers
    -> DINO addresses
    -> CEC or Pi3X causal bridge relocalizes
    -> geometric/learned proof accepts or abstains
    -> frozen NavDP executes
```

创新点不应写成“使用了 DINO + Pi3X + NavDP”，而应落在四个可检验命题：

1. **Content addressing is necessary**：candidate-free long-history GCT 的
   `5/20 vs 18/20` 说明 streaming state 不能自动完成开放集长程寻址；
2. **Causal visual bridge**：没有 current--anchor 直接共视时，用已观察轨迹构造视觉
   坐标传递链；b8->b16 的强配对结果直接支持这个机制；
3. **Proof-before-control**：global embedding/pose prediction 不足以安全授权动作，
   必须显式或学习地检验空间证据并允许 abstain；
4. **Minimal residual interface**：只传 scale-free bearing，保持 controller 冻结，
   所有无证据情况精确回退 native。

### 11.6 推荐标题

首选：

> **Closing the Navigation Loop: Causal Episodic Relocalization for Monocular
> Sequential ImageGoal Navigation**

更有记忆点的版本：

> **A Revisit Is Not a Novel Goal: Proof-Guided Episodic Relocalization for
> ImageGoal Navigation**

保留方法品牌的版本：

> **Episodic Compass: Goal-Conditioned Visual Loop Closure for Monocular
> ImageGoal Navigation**

如果 Final14 learned arm 通过，primary method 可以写为 Pi3X causal episodic
relocalizer，CEC 作为 task-training-free teacher/baseline；如果不通过，同一 task
story 仍成立，只是 primary implementation 回到 CEC。不能用标题和叙事覆盖
fresh qualification 的失败。

### 11.7 这个定位对评测提出的要求

若用 `navigation loop closure` 作为核心 hook，论文必须同时报告：

- **retrieval/relocalization 层**：proposal top-1、bearing CDF、precision/recall、
  FPR、`>90 deg` catastrophic accept、temporal gap；
- **开放集安全层**：unsupported Novel accept/takeover、exact fallback equality；
- **控制层**：paired closed-loop SR、gain/loss、McNemar、scene-cluster CI；
- **长期系统层**：history storage、first-query latency、update latency、memory
  coverage 随历史长度变化。

只报 loop-detection AUC 会把论文降格为定位论文；只报 SR 又无法证明增益来自可靠
重定位。当前 Final14 的五臂设计正是用同一 population 把这四层连接起来。

## 12. 2026-08-15、16、17 三天完整进展

### 12.0 三天的总体收敛轨迹

| 日期 | 当天核心问题 | 最重要结果 | 对项目方向的改变 |
|---|---|---|---|
| 8 月 15 日 | CEC 中 proposal 与 verification 谁是瓶颈；GOAT STOP 是否可迁移 | semantic-first 与 geometry-first 闭环打平；GOAT arrival 为 clean null | 停止 proposal-order 和 GOAT-STOP 分支，确定 CEC 的最小科学抽象 |
| 8 月 16 日 | raw-DINO Novel 增益是否来自真实历史；如何做外部确认 | forced-anchor attribution 未通过；Final14 冻结；HM3D 外部链建成 | Novel-DINO 停止，把预算转向 role-free Revisit、Novel safety 与外部 transfer |
| 8 月 17 日 | 能否用更统一的 learned geometry 替代显式 certificate | b16 causal bridge 与 spatial proof 成立；HM3D 外部效用确认；Final14 learned arm 提交 | 形成 CEC confirmed baseline + learned prospective replacement 的双层论文结构 |

这三天不是无序更换模块，而是依次完成：

```text
归因 CEC 已有行为
    -> 删除不能带来闭环增益的分支
    -> 验证非 MP3D 外部效用
    -> 找到显式工程链可学习化的结构性表示
    -> 冻结一次 fresh qualification
```

### 12.1 2026-08-15：论文收敛、proposal 归因与 GOAT clean null

#### A. 当天进入时的问题

Phase-2 出现一个看似矛盾的结果：raw fixed 在 mixed-role population 中达到
`27/38`，CEC 为 `21/38`；其中 Revisit 差距很小，但 raw 在 Novel 上偶然成功更多。
同时，一个明确失败例显示 geometry-first 选中的 anchor 有 522 PnP inliers、
`0.934 px` RMSE 和约 `41--42%` hull coverage，却没有带来成功。这说明：

- 局部几何自洽不等于目标语义相关；
- anchor identity、bearing 和最终闭环 utility 必须分开审计；
- 不能因为离线几何更“漂亮”就假设闭环更好。

当天冻结四个问题：proposal ordering、GOAT semantic arrival、raw Novel 因果来源、
以及论文能够防守的最小 claim。

#### B. proposal-versus-verification 离线审计

在 Attempt 7 + Phase-2 的 28 个已消费 Revisit histories 上，保持 DINO、LightGlue、
LingBot depth、PnP 和 certificate 不变，只比较三种 factorization：

| factorization | certificate coverage |
|---|---:|
| deployed geometry-first | 28/28 |
| DINO top-1 + same certificate | 28/28 |
| DINO-order first-certified | 28/28 |

geometry 改变 DINO top-1 anchor 的次数为 `21/28`，但 DINO rank-1 在 `28/28`
都能立即通过 certificate。结论不是“DINO 更好”，而是：在这批高支持 Revisit 中，
certificate 可以验证多种局部自洽 anchor，却不能决定哪一个更有控制价值。

#### C. consumed closed-loop Gate B

预先冻结规则后，只运行一次 semantic-first 与 geometry-first 的闭环比较：

| arm | success |
|---|---:|
| geometry-first CEC | 25/28 |
| semantic-first first-certified | 25/28 |

配对 `+0/-0`, McNemar `p=1.0`。虽然 first anchor 在 `21/28` 中不同，但授权 bearing
平均只差 `0.770 deg`、中位 `0.413 deg`、最大 `4.478 deg`，全部 `<5 deg`。
不同共视 anchor 最终塌缩成几乎相同方向，因此 proposal order 不可能产生 SR 差异。

执行中 task 18 在 H200 上发生 Habitat native `SIGABRT`，未产生第二 arm 或 completion；
只对该 exact index 在不变输入上重跑，并重新执行 summary/verifier。最终
independent verifier 为 `verified=true`，事件没有被当作方法结果。

**冻结决定：** semantic-first 不升级，geometry-first CEC 保留；不再运行新的
semantic-first confirmation。

#### D. GOAT first-ImageGoal semantic-arrival 正式结果

该协议只测试 20 个 GOAT scenes 的第一 ImageGoal semantic STOP，不是完整 sequential
GOAT score。修复 reset seed 的 63-bit/uint32 接口后，正式结果为：

- certified success `0/20`；
- certified STOP `0`，true STOP `0`，false STOP `0`；
- 20 条全部由 forced guard 结束；
- legacy first-zero counterfactual `1/20`；
- paired `+0/-1`, `p=1.0`；
- 预注册的至少 5 个 true certified stops gate 失败。

零 false stop 在零 coverage 下是 vacuous safety。28 次 native-zero 事件中 26 次仍在
官方到达区外，中位 goal distance `6.219 m`；真正到达的两个事件反而没过局部几何
precheck。结论是当前 semantic-arrival adapter 不能迁移到这个 GOAT first-goal
contract，但该 null 不否定 CEC 的 causal Revisit bearing takeover。

**冻结决定：** 不在 GOAT held-out 上调 threshold；以后若重启 GOAT，必须测试实际
sequential Revisit-bearing claim，而不是再次测试 first-goal STOP。

#### E. 当天形成的论文骨架

当天把 CEC 从“若干模块组合”收敛为：

> causal history proposes an episodic place hypothesis; a witness authorizes
> whether it may alter a frozen policy; the only transferred variable is a
> scale-free bearing; unsupported hypotheses preserve exact native behavior.

同时明确禁止：CEC 显著优于 raw fixed、certificate 零误激活、Novel 已解决、
semantic-first 有增益、GOAT 是完整 benchmark score、X-NavDP/learned decoder 已提升。

### 12.2 2026-08-16：Novel 归因停止门、Final14 冻结与外部评测

#### A. raw-DINO Novel cohort shift 审计

先确认 Attempt 7 和 Phase-2 的 evaluator、checkpoint、budget、success radius、seed
和 hidden-role interface 相同，不是代码漂移。Novel raw fixed 对 native 为：

| population | raw fixed | native | paired |
|---|---:|---:|---:|
| Attempt 7 | 1/9 | 2/9 | `+1/-2`, `p=1.0` |
| Phase-2 | 9/19 | 4/19 | `+6/-1`, `p=.125` |

Phase-2 仍不显著。六个 raw gain 的首方向误差为 `2.82--24.08 deg`，唯一 loss 为
`110.74 deg`，所以不是纯 CUDA 噪声；但 raw head 的方向集中在后方，而 Phase-2
恰好有 `16/19` shortest-path direction 位于后方，形成 U-turn cohort alignment。

#### B. forced-anchor 因果归因

在 19 个已消费 Phase-2 Novel queries 上，固定 current/goal 和 factual online-A
history，把 DINO factual anchor 与每 query 12 个 identity-seeded legal random anchors
比较：

| endpoint | factual advantage | scene-cluster CI | deployability |
|---|---:|---:|---:|
| shortest-path bearing error | `+4.148 deg` | `[-1.357,+8.898]` | `<=30 deg`: 10/19 vs 9.75/19 |
| direct-goal bearing error | `+2.169 deg` | `[-3.710,+7.069]` | 10/19 vs 10.0/19 |

CI 下界没有超过 0；all-eligible physical-anchor audit 也只有 `+0.048 deg`，CI
`[-2.522,+3.531]`。因此 DINO 没有稳定选择更好的历史 route location，LingBot goal
insertion 只产生异质的小修正。

**冻结决定：** `stop_novel_dino_branch_and_preserve_final14_for_cec_confirmation`；
取消 goal-shuffle 和昂贵的 600-step Novel 四臂测试。raw Novel 只作为 cohort/boundary
解释，不作为方法结果。

#### C. Final14 parent protocol 冻结

在未打开 final14 population 前，冻结了真正对应 CEC claim 的正式设计：

- role-free standard Revisit utility vs native；
- natural unsupported Novel interference 与 exact fallback；
- raw fixed、old geometry 与 CEC 的 risk--coverage；
- 单列 hard-support Revisit `covis [0.25,0.55)`；
- standard Revisit `covis [0.55,0.90]`；
- Novel front/side/rear stratification；
- goal yaw 与 route bearing 解耦，消除 Phase-2 U-turn confound。

renderer-only consumed preflight 中 standard Revisit `4/4` 可构造，hard support
`3/4`，证明两个 support bands 确实不同。没有调用 policy，也没有打开 final14。

#### D. 非 MP3D 路线选择

8 月 16 日冻结“当晚 efficacy budget 只用于非 MP3D”：

- official GOAT val-unseen 已产生 clean null，不重跑、不调参；
- Replica 在当前 `>=2 m`、frame `>=39` 长历史 contract 下没有正式 population，
  属于 constructibility failure；
- MemoNav/Gibson 只发布 episode positions，没有 evaluator code、goal rotation 或 goal
  RGB，本地也缺受许可 Gibson meshes；其 panoramic RGB-D PR/PPL 不能与 monocular
  NavDP SR 直接横减；
- 因而唯一可识别、可运行的外部路线是 outcome-disjoint HM3D val10 causal Revisit。

#### E. HM3D val10 构造与 runtime repair

从 HM3D v0.2 val 100 scenes 中先减去 36 个历史 consumed scenes，再按冻结规则取
10 scenes，每 scene 目标 4 episodes。一个 scene `q3hn1WQ12rz` 在 240 次冻结 outer
attempt 后仍为 0/4，作为显式构造 attrition 保留；其余 9 scenes 得到 36 episodes，
不替换 scene、不放宽约束。

随后依次发现并修复了结果产生前的接口问题：

1. 旧 evaluator 不认识已经冻结的两个 controller flags，也缺 raw-fixed adapter；
2. consumed smoke 缺 `depth_anything` launcher dependency；
3. 第二次 smoke 缺 evaluator 用于重建 Goal-A 的 expert RGB stream；
4. 正式 nine-scene tasks 全部完成后，旧 summary 只因 schema 名称不兼容失败，采用
   analysis-only schema repair，未重跑 episode。

每次 failure 都发生在相应 scientific output 前，并保留 receipt。最终 smoke、36
episodes、summary 和 independent verifier 全部通过。

#### F. HM3D 正式结果（任务 16 日启动，17 日完成封账）

| arm | Goal A | Revisit B given A | joint |
|---|---:|---:|---:|
| native | 21/36 | 7/21 | 7/36 |
| old geometry | 21/36 | 17/21 | 17/36 |
| raw fixed, oracle role | 21/36 | 18/21 | 18/36 |
| role-free CEC | 21/36 | 19/21 | 19/36 |

CEC 对 native：

- B given A `+12/-0`；risk difference `+57.14 pp`，cluster CI
  `[+36.36,+78.95]`；exact `p=.000488`；
- joint `+33.33 pp`，cluster CI `[+22.22,+44.44]`；
- 12 个 gain 覆盖 8/9 scene clusters；
- runtime failures 0，fallback mismatch 0。

CEC 未显著超过 geometry（`+2/-0`, `p=.5`）或 raw oracle-role（`+1/-0`,
`p=1.0`）。uncached certificate latency 中位 `5.83 s`、p95 `25.83 s`，是必须披露
的部署限制。

**当天净结论：** CEC 的 Revisit utility 已经跨 MP3D -> HM3D 外部转移；distinct
价值是 role-free authorization 和 exact fallback，而不是已证明超过所有 memory
control。Novel-DINO 和不兼容 benchmark 路线停止。

### 12.3 2026-08-17：learned relocalizer、外部结果封账与 Final14 执行

#### A. 从“换 matcher”转向“学习 proof”

目标是逐步替代 CEC 内部
`SuperPoint + LightGlue + LingBot depth + PnP + hand-set certificate`，但保留完全相同
的外部 contract：supported relative direction 或 abstain、2.5 m residual、frozen
NavDP、exact fallback。

第一候选 zero-shot MicKey 在全部 3,840 Train40 DINO pairs 上运行：

- candidate AUC `.7618`，AP `.5627`；
- positive-session top-1 `112/155`，低于 geometry `126/155` 和 raw DINO
  `115/155`；
- OOF authorization 只接受 8 sessions，其中 5 个正确，positive recall `3.2%`；
- warm latency 约 `36 ms`。

它证明 learned pose proposal 可以很快，但不能解决开放集 support/abstention；因此
MicKey 不升级，也没有盲目长训。

#### B. Pi3X 的失败、归因与 causal visual bridge

第一版用 evaluator Sim(3) 对齐得到 median `2.35 deg`，但部署不可用；去掉外部对齐、
仅用 current + 旧 anchor 周围五帧时 median 退化到 `88.17 deg`。诊断为 current 与
远端 anchor 无直接共视，坐标系断连。

于是只使用已经观察到的中间 RGB 构造 causal bridge。最初本机两场景 mechanism
smoke 把 true-pair median 降到 `5.54 deg`，但仍有长尾，需要 abstention。Train40
b8 在 RTX 5090 与 HPC 独立运行得到接近的 navigation AUC（`.89418/.89490`），
说明信号可复现。

有针对性地把 bridge 从 b8 增到 b16 后：

- positive candidate `<=30 deg`: `585/701 -> 659/701`, `+82/-8`,
  `p=1.38e-16`；
- raw-DINO top-1: `130/155 -> 144/155`, `+16/-2`, `p=.001312`；
- candidate oracle ceiling: `149/155 -> 153/155`, `+4/-0`；
- Pi3X overlap 最终 proposal top-1 `147/155`。

这成为三天里最重要的新 learned mechanism：长程重定位需要 causally observed visual
bridge，不是只比较 current/old pair。

#### C. global-token proof 为什么停止

依次测试的 global representation 路线没有同时满足安全和 coverage：

| head | precision | recall | FPR | accepted `>90 deg` |
|---|---:|---:|---:|---:|
| V1 candidatewise | 88.52% | 34.84% | 2.48% | 0 |
| V2 bound model-threshold ensemble | 90.80% | 50.97% | 2.13% | 0 |
| V3 top-8 + explicit REJECT | 83.12% | 82.58% | 5.67% | 7 |

一个 scalar b16 reliability head 虽有 AUC `.919696`、AP `.870651`，但仍接受 3 个
`>90 deg` 灾难 bearing，并且 fold 不稳。审计还确认早期 CDEC 已经测试过抽象上相同
的 top-8+NULL 问题。结论不是“MLP 不够大”，而是平均 global token 丢失了能证明
pose 的局部 correspondence structure。

#### D. spatial proof 的正结果

从同一 frozen b16 forward 导出 point/pose/confidence spatial grids，固定 Pi3X overlap
proposal，只让 311,426-parameter head 学授权。五折 scene-crossfit、2/4 consensus：

- correct positive accepts `119/155`，recall `76.77%`；
- precision `95.20%`；
- strict-negative FPR `4/282 = 1.42%`；
- accepted median bearing `3.41 deg`；
- accepted `>90 deg` 为 0；
- 五个 outer folds precision `90--100%`，但 recall `62.86--94.74%`。

同 directional endpoint 下，old certificate 为 `107/155` correct accepts、precision
`97.27%`、FPR `.71%`、一个 `>90 deg`；learned 为 `119/155`、`95.20%`、`1.42%`、
零灾难。coverage 配对 `+28/-16`, `p=.0961`，尚未显著优于 certificate，但足以进入
fresh non-inferiority test。

#### E. deployment integration 与 consumed transport tests

四个 `30 fit / 10 calibration` 成员训练后，阈值和 checkpoint hash-bound，2/4 才
接管。正式 policy/server/evaluator 已接入 sticky rejection、fixed anchor、one-anchor
update 和 exact native fallback。

positive consumed smoke 中：native B 249 steps 失败，learned 40 steps 成功；首请求
DINO top-8 + Pi3X 约 `1182 ms`，后续单 anchor 约 `149 ms`。这只是 transport，不能
计为 fresh SR。

negative counterfactual smoke 中首 proof `0/4`，之后 30 次使用 cached abstention；
native/learned 都为 247 steps，31/31 trajectory hash 和 247/247 pose/yaw/RGB 全等，
证明 fallback 是 action-path exact，而非只在最终 success 上相同。

standalone Pi3X 峰值约 `6.55 GB`；CEC/LingBot/Pi3X 共存正式进程约 `9.65 GB`。

#### F. five-arm mechanics、population repair 与正式提交

consumed five-arm dry-run 已证明 native/raw/geometry/CEC/learned 的 formal evaluator、
summary 和 independent verifier 能运行。该小 fixture 中所有 arm 都失败，不能作为
efficacy；learned 在 Novel exact fallback、Revisit takeover、零 runtime failure 和
最大 `16.32 deg` accepted bearing 上通过 transport contract。

随后发现旧 Attempt-7 builder 不完全等于 Final14 parent population：没有独立 hard
support subset、natural Novel/standard Revisit cap 和全局 front/side/rear cycle。该
问题在 final14 unseal 前被发现，重新实现了 CPU-only population contract、builder、
finalizer 与 independent audit。

Attempt 1--3 分别因 base manifest、overlay/root、zero-source/递归依赖问题在任何五臂
outcome 前失败；Attempt 4 使用 4,011-file read-only complete bundle，保持相同 14
scene/80 available source selection 后提交。其 frozen hypotheses、job IDs 和当前状态
见第 8--9 节。

**8 月 17 日最终科学状态：**

- CEC 仍是 confirmed/externally transferred primary；
- Pi3X causal bridge 是 strong internal mechanism；
- spatial learned proof 是 prospective deployment candidate；
- Final14 未出结果前，不能宣布 learned replacement 成功；
- learned route 的最大统计风险不是 3,840 pairs 太少，而是只有 40 个独立训练 scenes
  以及同一 Train40 上的 meta-overfit。

### 12.4 三天后证据矩阵发生了什么变化

| 命题 | 8 月 15 日前 | 三天后状态 |
|---|---|---|
| causal Revisit memory 能提升 frozen NavDP | MP3D 内部成立 | **HM3D 外部 scene-disjoint 也成立** |
| semantic-first proposal 能提升 CEC | 未知 | **paired null，停止** |
| GOAT first-goal semantic STOP 可迁移 | 未知 | **clean null，停止且不调参** |
| raw-DINO 能解决 Novel direction | cohort 中看似有增益 | **forced-anchor gate 未过，停止** |
| candidate-free long memory 更优雅 | 设想 | **5/20 vs 18/20 否决，DINO address 保留** |
| learned geometry 能替代 CEC | 旧 decoder/CDEC 失败 | **causal bridge + spatial proof 获得 prospective test 资格** |
| learned proof 已在 fresh scenes 超过 CEC | 无 | **未成立：19/21 对 20/21，L2/L3 promotion gates 未通过** |
| 论文主线 | 模块较多、叙事偏工程 | **goal-conditioned navigation loop closure + proof-before-control** |

三天真正完成的不是“又增加了更多组件”，而是把不可防守的分支逐个关掉，并把剩余
主线变成一个有外部证据、负结果、机制结果和 prospective test 的完整科学故事。

## 13. 关键代码、结果与冻结文档

### learned method

- `MemNavData/pi3x_online_relocalizer.py`：b16 causal bridge、Pi3X inference、
  overlap proposal 与 online bearing；
- `MemNavData/pi3x_spatial_reliability_model.py`：spatial proof 网络；
- `MemNavData/train_pi3x_spatial_reliability_crossfit_oof.py`：五折 scene-crossfit；
- `MemNavData/pi3x_spatial_proof_runtime.py`：checkpoint/threshold hash-bound runtime；
- `MemNavData/compare_pi3x_spatial_proof_to_certificate.py`：direction-aligned 比较；
- `NavDP/baselines/memnav/policy_agent.py`：DINO proposal lifecycle、CEC 与 learned
  route 的实际 policy integration。

### verified artifacts

- OOF result：
  `.diagnostics/learned_relocalizer_20260817/pi3x_spatial_head_crossfit_5090_ffcb1a682a6d9f64/full/summary.json`；
- deployment manifest：
  `.diagnostics/learned_relocalizer_20260817/pi3x_spatial_deployment_5090_9a2b99aea2673b05/deployment_manifest.json`；
- learned development ledger：
  `MemNavData/LEARNED_RELOCALIZER_NIGHT_GOAL_20260817.md`。

### Final14 documents

- prospective amendment：
  `MemNavData/FINAL14_LEARNED_RELOCALIZER_PROSPECTIVE_AMENDMENT_20260817.md`；
- Attempt 3 incident：
  `MemNavData/FINAL14_ATTEMPT3_COLLECTION_INCIDENT_20260817.json`；
- Attempt 4 repairs：
  `MemNavData/FINAL14_ATTEMPT4_COLLECTION_REPAIR_RECEIPT_20260817.json`、
  `MemNavData/FINAL14_ATTEMPT4_BUNDLE_CLOSURE_AUDIT_20260817.json`；
- Attempt 4 execution receipt：
  `MemNavData/FINAL14_ATTEMPT4_EXECUTION_RECEIPT_20260817.json`。

### official model references

- Pi3/Pi3X code: <https://github.com/yyfz/Pi3>
- Pi3 paper: <https://arxiv.org/abs/2507.13347>
- LingBot-Map code: <https://github.com/Robbyant/lingbot-map>
- LingBot-Map paper: <https://arxiv.org/abs/2604.14141>

## 14. Final14 封账后的纪律

- Train40 OOF 仍是开发证据，不是论文最终 SR；
- consumed smokes 仍是 transport/contract 证据，不是 efficacy；
- Attempt 1--6 的基础设施事故不解释为方法失败或成功；
- Final14 已关闭 MP3D empirical method-selection loop，不允许根据失败样本重调后再把
  同一 population 当测试集；
- Final14 只支持 fresh MP3D mixed-role claim；跨域 Revisit utility 由独立 HM3D
  formal result 支持，跨域 open-set safety 尚未测试；
- CEC 保持 primary；learned Pi3X 是有 utility 但未通过 deployment qualification 的
  secondary/negative result；
- 后续优先整理论文证据，并对 CEC causal feature/depth cache 做 `0` decision-mismatch
  microbenchmark，而不是继续内部调参或重复 HM3D Revisit-only 评测。

正式结果见：`MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md`。
