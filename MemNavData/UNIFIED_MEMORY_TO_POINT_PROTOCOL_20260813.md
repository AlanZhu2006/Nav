# Unified Memory-to-Point Adapter（M2P）协议草案

日期：2026-08-13（Asia/Shanghai）

状态：**train-only 设计与可证伪协议；不是方法结果，不读取新 development/blind 结果。**

## 1. 目标

目标不是继续给当前链条增加一个 learned reranker：

```text
DINO top-8 -> SuperPoint/LightGlue -> LingBot depth + PnP
             -> geometric certificate -> bearing -> frozen NavDP
```

目标是把它在部署时收缩成一个单次 learned adapter：

```text
goal RGB + current RGB + causal RGB/action memory
                    |
        Memory-to-Point Adapter (M2P)
                    |
     circular bearing posterior + unsupported mass
                    |
 accepted: 2.5 m scale-free PointGoal -> frozen official NavDP
 abstain:  exact native ImageGoal NavDP
```

目标部署图保留 **一个冻结的 geometry-pretrained streaming trunk**；本项目首选已经使用的
LingBot-Map/GCT。被删除的是显式的 DINO shortlist、SuperPoint/LightGlue、独立 depth-PnP
与手工 certificate 串联。当前完整 certificate 链仍只作为离线 privileged teacher、强诊断
基线或 shadow verifier，不把它伪装成已经被 learned adapter 替代。

## 2. 为什么不能只把旧 CDEC 放大

### 2.1 当前 CDEC 学错了最终任务

当前 pairwise CDEC 的主要 label 是 thresholded goal-surface co-visibility；所谓
`certificate_pass` 实际是：

```python
fundamental_inliers >= 32 and fundamental_query_grid_coverage >= 0.75
```

它不是最终的 LingBot-depth PnP certificate，也不是 `position error <= 0.75 m` 的
controller actionability。

480-session same-process 审计已经直接显示 label mismatch：

| proposal | teacher-positive top-1 | GT actionable | certified actionable |
|---|---:|---:|---:|
| geometry | 126 | 153 | 122 |
| CDEC | 128 | 135 | 115 |

CDEC 在训练 label 上多 2 条，却在真正 actionable 上少 18 条、certified-actionable
上少 7 条。geometry 选中的 ambiguous (`label=-1`) anchor 中 `31/41` 可执行；因此
增加同类二值 co-visibility label 可能会让模型更稳定地排斥可执行 anchor。

### 2.2 当前表示不承担位姿推理

现有 raw matcher 把 `37x37` DINO patch pool 到 `8x8`，做低秩投影、soft matching，
最后 mean/max pool 成一个 relation vector。训练只监督 session label 与上述几何代理，
没有监督 patch correspondence、SE(2) composition 或 bearing。

它能学“相似/不相似”，但没有被要求回答“目标相对当前位置在哪个方向”。

### 2.3 旧 unified MemNav 已经接触过全 PT1

旧 `MemNav_Dataset` 枚举所有有 cache 的 episode，历史长训显式使用
`MEMNAV_MAX_LEGS=0`。所以“把全 PT1 喂给旧结构”不是新的实验。

旧结构的失败面是：

- 训练并替换 shared decoder，而不是冻结 official NavDP；
- retrieval 用 DINO CLS，gate 与 action loss 并不对齐；
- goal-swap 因果灵敏度远低于 official NavDP；
- GoalSwap 长训约 `1.77 epoch` 后，正确 goal 相对错误 goal 的 gap 仍为负；
- learned pose residual 改善 translation p90，却把 bearing p90 从约 `16.8 deg`
  恶化到约 `21.1 deg`。

因此问题既有数据利用率，也有结构与目标错位；不能通过延长旧 run 解决。

## 3. 数据边界与规模

### 3.1 模型选择阶段：只用 train40

远端 PT1 实盘审计：

| family | train40 cached / `n_frames<=2048` | development | final-reserved |
|---|---:|---:|---:|
| 2-leg | 586 | 150 | 0 |
| 3-leg | 940 + 2 patched | 243 | 0 |

因此 train-only 可使用约 `1,528` episodes。当前 CDEC 只使用 `80` 个 3-leg
episodes、`480` sessions。

train40 factual revisit curves（未计入额外 Novel/counterfactual negatives）已经包含：

| support band | image pairs |
|---|---:|
| covis `>=0.50` | 109,263 |
| `[0.20,0.50)` | 73,542 |
| `[0.10,0.20)` | 33,528 |
| `<0.10` | 318,374 |
| total | 534,707 |

这些 frame pair 高度相关，不能把 `534,707` 当独立样本。训练 sampler 必须按
`scene -> episode -> role -> support band` 平衡，每个 episode 每 epoch 只取有限状态。

### 3.2 最终 refit

只有 architecture、loss、acceptance rule 在 train40 scene-OOF 完全冻结后，才允许把
全部 populated PT1（`736` 2-leg + `1,183` 3-leg = `1,919` episodes）用于一次最终
refit。此后只能在与这些 scene clusters 不相交的新池上确认，不能再用旧 development
选择模型。

## 4. 文献约束后的 M2P 架构

M2P 不是无结构的 RGB-to-angle MLP，也不从 40 个 train scene 重新学习视觉几何。
现有文献给出的共同约束是：普通 DINO/单图预训练擅长语义不变性，但宽基线相对位姿要求
patch-wise early fusion 与专门的跨视角几何预训练；单靠导航/action loss 很容易忽略 goal。

因此候选方向是 **geometry-pretrained streaming backbone + trainable goal-query
adapter**。在本项目中优先审计已经使用的 LingBot-Map/GCT；CroCo/DEBiT、MASt3R、MicKey
或 Pi3X 是接口不足时的预训练替代，不从头训练。必要的几何约束应位于一个联合前向模型
内部，而不是部署时串联多个独立算法。

这里不能把论文能力当成本地接口能力：当前可运行的 `LingBotStream` cache 对老帧主要保留
每帧 6 个 special/anchor K/V；现有 `goal_append_warm` 需要外部先给定 anchor，再重放最多
64 帧局部窗口。GCT 本体虽已有 `_set_skip_append(True)` 这一 read-only attention 原语，
仓库里仍没有经过因果性、pose 精度和 cache identity 验证的“任意 goal、整段 memory、
non-writing”查询接口。因此下述 GCT goal-query 在 S-1 通过前只是待验证设计，不是可直接
开训的实现事实。

### 4.1 输入（部署可得）

- current monocular RGB；
- goal RGB；
- causal historical RGB keyframes；
- 已执行的 velocity/action history 与时间间隔；
- 不使用 Habitat pose、GT depth、goal role 或 future frame。

### 4.2 Frozen geometry-pretrained streaming trunk

LingBot/GCT 因果地把历史 RGB 编成 anchor context、局部 pose-reference window 与压缩的
trajectory memory，并输出同一 affine map 中的 camera/geometry tokens。它已经在大规模
跨场景 3D 数据上学习了 streaming pose/depth/long-range context；第一阶段完全冻结，不能
让 PT1 的 40 个 scene 把几何先验训坏。

official NavDP checkpoint、decoder、critic 与 ImageGoal 路径同样全部冻结。DINO global
或 patch feature 只作为 cheap/direct baseline，不能再被假定为足够的几何表征。

### 4.3 Non-writing goal-query adapter

goal image 作为 **query** 对冻结的 GCT trajectory memory 做 binocular patch cross-attention；
它不能像旧 `goal_append` 那样假装 goal 是时间上相邻的新观测并写回 streaming state。
query adapter 联合输出：

```text
p(memory frame / local mode | goal, history)
p(local bearing residual | mode)
p(unsupported)
```

frame-level attention 使用整段压缩 memory；少量 posterior modes 再通过 geometry-pretrained
patch decoder 细化。coarse-to-fine 是同一网络内、由同一 task loss 联合训练的计算结构，
不是固定 DINO top-8 后再调用另一套算法。

### 4.4 Scale-free map composition

每个 memory mode 已有 GCT map pose。把 mode posterior、局部方向 residual 与当前 GCT pose
组合成 circular bearing posterior：

```text
p(theta | goal, current, causal memory), p(unsupported)
```

只需要方向，不监督/恢复不必要的 metric range；这样规避单目 scale ambiguity，并与已经
验证的固定 `2.5 m` NavDP residual 接口一致。显式 SE(2)/Sim(3) composition 是网络内的
确定性等变层，不是手工 PnP certificate。

### 4.5 Circular output and exact fallback

使用 mixture-of-von-Mises 或离散 circular bins，禁止对多峰分布直接做普通角度 MSE。
接受时输出：

```text
[x, y, z] = 2.5 * [cos(theta), sin(theta), 0]
```

并调用 official frozen NavDP PointGoal 接口；拒答时调用未修改的 native ImageGoal 路径。
目标切换时建立 posterior，后续只用 recurrent pose 更新相对 bearing，可增量缓存。

## 5. Privileged training，不是 privileged deployment

PT1 的 GT camera poses、depth 与 goal pose只生成监督：

1. **geometry audit / optional preservation**：先用相邻及 `1/4/16/64` frame 相对 SE(2)
   审计冻结 GCT；第一版只训练 query adapter/readout，因此该项只作为诊断、不给冻结 trunk
   反传。只有 S0-B 证明 GCT 是瓶颈并决定对最后若干层做 LoRA 后，才启用 preservation loss；
2. **soft memory support**：连续 reprojection/covisibility，不先阈值成唯一真值；
3. **goal location**：current-to-goal 2D direction/position；
4. **primary circular bearing**：`1-cos(theta_hat-theta_gt)` 或 von-Mises NLL；
5. **selective uncertainty**：预测误差分布和 unsupported mass；
6. **counterfactual negatives**：same-scene wrong goal、Novel-B 与 high-DINO/low-overlap
   hard negatives；
7. **temporal consistency**：同一个 goal 在不同 current time 的 world posterior 应一致。
8. **certificate structure distillation（仅辅助）**：蒸馏 LightGlue/Fundamental 的局部对应
   分布、query/reference coverage，以及 LingBot-depth PnP 的 pose、inlier/reprojection
   uncertainty；不能把最终 `pass/fail` 当唯一标签。certificate-pass 且 GT-actionable 的
   pair 可提高权重；certificate-reject 但 GT-actionable 的 pair 仍按 bearing/support 主目标
   学习，避免复制 teacher 的保守假阴性。

建议总损失：

```text
L = L_bearing
  + lambda_loc L_goal_location
  + lambda_match L_soft_support
  + lambda_teacher L_certificate_structure
  + lambda_risk L_selective_NLL
  + lambda_cons L_temporal_consistency
```

仅在后续解冻 GCT/LoRA 时才加入
`lambda_pose * L_multi_horizon_SE2`；冻结 trunk 阶段把它写进优化目标没有梯度意义。

`L_bearing` 是主目标；retrieval/covisibility 仅为辅助。不能再次让辅助 AUC 决定方法通过。

## 6. 训练课程

### Stage S-1：表示与接口可观测性门（零长训）

先冻结全部权重，只读检查当前 GCT cache 是否足以支持目标任务：

1. **oracle-anchor + frozen GCT**：给正确历史区域，测 `goal_append_warm` 的 bearing CDF，
   分离 GCT pose/局部重放误差；
2. **candidate-free read-only query**：goal 不写入持久 stream、不预给 anchor，确认它能读取
   全部因果 memory，并记录老帧只有 6-token compression 时的 recall-gap 曲线；
3. **causal/identity audit**：交换 goal 必须显著改变 posterior；交换与 goal 无关的 frame 不得
   产生同量级变化；query 前后 streaming cache 逐 tensor 相同；
4. **low-capacity scene-OOF probe**：冻结表示上的线性/极小 cross-attention probe 必须在
   held-out train scenes 上优于 DINO pooled/direct-bearing；否则不得靠扩大 adapter 掩盖
   表示不可观测性；
5. 若第 2 项在当前 GCT 接口上不可实现或第 4 项失败，切换到 CroCo/DEBiT、MASt3R、MicKey
   一类 geometry-pretrained binocular encoder 做 goal-keyframe query；禁止退回“DINO patch +
   更大 MLP + 全 PT1”。

### Stage S0：2x2 可辨识性门（便宜，必须先过）

在 train40 内固定 scene-disjoint split，把“找到哪个历史区域”和“历史坐标是否准确”正交
分开：

| anchor / coordinate | GT coordinate | frozen GCT coordinate |
|---|---|---|
| oracle support/mode | A：接口与数据上限 | B：纯地图/漂移误差 |
| learned goal query | C：纯视觉定位误差 | D：完整 feed-forward M2P |

另加 DINO pooled/direct-bearing 与当前 certificate 两个参考。A-D 全部只算离线 bearing、
support 与 selective risk，不进入 closed loop。

解释规则：若 C 弱，问题是 goal-query/cross-view 表征；若 B 弱，问题是 GCT map drift；
若 B/C 强而 D 弱，问题是两种误差组合与校准；若 A 都弱，则数据标签/接口本身有误，整条
方法立即停止。S0 不读 development。

### Stage S1：全 train40 多任务训练

- 使用全部 `1,528` train-only episodes；
- 2-leg 先提供短程 revisit，3-leg 强化长 recall-gap；
- role/band/scene balanced sampler；
- teacher-forced support/pose 逐步切换到预测 posterior；
- frozen visual trunk 起步，只允许 adapter 学习；通过后才考虑对 trunk 做 LoRA。

### Stage S2：scene-OOF selective calibration

所有 threshold、temperature 与 risk-coverage operating point只在 train40 outer folds 内
确定。允许使用 conformal upper risk bound；禁止固定 `0.5`，也禁止根据 closed-loop SR
反调 threshold。

### Stage S3：frozen closed loop

只有 S0-S2 通过后，才进入同机同进程的三臂闭环：

```text
native / current certified residual / M2P
```

## 7. 预注册停止门

S0/S1 必须同时报告：

- Revisit bearing error CDF：`<=15/30/45 deg`；
- support-conditioned coverage；
- strict no-match false activation；
- scene-macro 与 scene-cluster interval；
- S0 四格 A/B/C/D，以及 `A-B`（地图漂移）、`A-C`（goal-query）和
  `min(B,C)-D`（误差组合/校准）的差值；
- recall-gap 分层（2-leg short / 3-leg long）；
- inference latency与 memory size。

建议最低继续条件（正式运行前再由现有 train-only baseline 固定具体数值）：

1. 完整 D 在 held-out train scenes 的 `<=30 deg` selective coverage 明显高于 DINO
   pooled/direct baseline，并缩小到 A 的差距；
2. Novel/counterfactual false activation 不高于当前 certificate 的同口径参考；
3. 2-leg 的提升不能以 3-leg long-gap 崩溃换取；
4. 至少两个 seeds 方向一致；
5. 按 S0 的 B/C 分解定位瓶颈；不能在原因不明时同时扩 locator 与 GCT；若 A 本身弱，停止
   整条 M2P，不提交闭环。

## 8. 这条线与现有结果的关系

- **oracle-bearing `40/40`**：冻结方法输出为 scale-free bearing，并冻结 `2.5 m`
  PointGoal/NavDP 接口；不再学习 action decoder；
- **certified residual `112/120`**：作为当前强部署 baseline，并把对应、pose 与 uncertainty
  拆成 privileged auxiliary teacher；不只蒸馏二值 pass；
- **raw-DINO direct `106/120`**：作为最小复杂度参考和 coarse semantic prior；M2P 必须在
  相同风险下超过它，不能只超过旧 geometry router `91/120`；
- **actual-online observability `120/120 >=0.20`、`115/120 >=0.50`**：授权使用这些 Revisit
  episode 做因果 goal-to-online-memory 训练，而不是把 expert history 当部署 history；
- **CDEC 的失败与局部正信号**：保留“session 内候选相对排序可学”，但放弃 joint
  `anchors+NULL`、co-visibility top-1 和跨场景绝对阈值；bearing/support 是主目标，拒答
  单独做 scene-OOF risk calibration；
- **geometry/CDEC actionable 审计**：以 GT-actionability 修正 teacher 假阴性，禁止再让
  surrogate AUC/top-1 代替 controller-relevant 指标；
- 不训练 shared NavDP decoder，避免旧 MemNav 的 Novel 负迁移；
- 不要求八方向原地扫描；
- 不显式判断 Novel/Revisit；unsupported posterior 只决定是否提供 residual。

## 9. 当前决定

1. **不提交“旧 CDEC + 全 PT1”长训。**它不能回答工程链能否被替代。
2. 下一项代码工作应是 M2P-S-1：先证明现有 LingBot/GCT cache 能被任意 goal 以 read-only
   方式查询；通过后才复用 `goal_rel_pose` 与 camera trajectory label 建立 S0 的 2x2
   evaluator，不先训练新的 odometry/decoder。
3. 第一轮长训的科学问题只有一个：

> 不运行 DINO shortlist、LightGlue、depth-PnP 与手工 certificate 时，geometry-pretrained
> causal memory 的单个 learned goal-query，能否在未见 train scenes 上输出足够准确且可
> 拒答的 scale-free bearing？

只有这个问题通过，M2P 才有资格替代现有工程链。

## 10. 2026-08-13 S-1 口径修正：DINO 是 prior/reference，不是最终判题器

小样本 actual-online 审计补充了三项事实：

- DINO top-1 后做 frozen-GCT 局部 query，在 8 个 Revisit goal 上 bearing error 全部
  `<=15 deg`；但这正是已有 raw-DINO direct 的底层机制，不是新方法；
- 不给 anchor、只在完整因果 prefix 末端 read-only query frozen GCT，8 个 goal 中
  `7/8 <=15 deg`，说明 candidate-free query 有初步可观测性；
- 8 个 cross-scene strict no-match 的 global/local bearing disagreement 中位数约
  `105.8 deg`，但存在一个 `8.6 deg` 的偶然一致，故不能用固定一致性阈值冒充安全证书。

因此完整 train40 的 S-1 冻结为三个互不混淆的门：

1. **candidate-free observability（主门）**：在 155 个 positive session 上直接报告 global
   frozen-GCT bearing 的 `<=15/30/45 deg` CDF；`CDF@30 >=0.80` 且所有 query 前后 cache
   identity 精确成立，才允许训练低容量 scene-OOF probe；
2. **DINO-free replacement（更强、非必需门）**：global `CDF@30` 相对 DINO-local 的配对差
   不低于 `-5 pp`，scene-cluster 95% CI 下界高于 `-10 pp`，才允许讨论完全去掉 DINO；
3. **raw unsupported signal（辅助门）**：global/local agreement 对“positive 且 DINO-local
   bearing error <=30 deg”的 AUC 必须比 DINO cosine 高至少 `0.03`，且 scene-cluster CI
   下界高于 0。它只授权训练 selective head，不能单独授权长训或闭环。

这三个门的输出不得合并成一个“passed”。特别是，S-1 冻结特征结果本身永远不能授权
Selective M2P 长训；下一步必须先用 scene-OOF 的线性/极小 cross-attention probe 证明 GCT
关系特征在 frozen DINO prior 之外提供增量信息。

若后续保留 DINO，形式必须是覆盖全部因果 keyframe 的冻结 soft prior，而不是 hard top-K：

```text
pointer_logit_i = frozen_DINO_similarity_i
                  + bounded_zero_init_GCT_residual_i
```

这样初始模型严格退化为最强 raw-DINO direct，学习模块只修正 pointer、输出 circular bearing
及 unsupported mass；不能重新训练一个跨场景绝对 DINO 阈值，也不能把 DINO cosine 当作
“目标存在”的概率。

## 11. 2026-08-13 S-1 因果角色修正：短期 Novel-B 与长期 Revisit-C 不得混算

HPC 单 episode smoke（只用于接口和分析器校验，不用于效果结论）暴露了一个会颠倒架构判断的
混杂：先前所谓 155 个 positive session 同时包含 Novel-B midpoint 与真正 Revisit-C，二者的
时间跨度和因果角色不同，不能汇总成一个 candidate-free CDF。

同一条 300-frame actual-online trajectory 上：

- `goal_b_midpoint_t1/factual`：DINO anchor 的 bearing error 为 `168.3 deg`，完整因果 prefix
  read-only query 为 `0.29 deg`；teacher 的最新正 support 距 decision 为 0 frame；
- `goal_c_t0/factual`：DINO anchor 为 `1.06 deg`，完整因果 prefix query 为 `76.31 deg`；
  teacher 的最新正 support 距 decision 为 168 frames。

这两个样本不能证明哪一支总体更强，但足以否定“把所有 positive session 混起来选架构”。
它们支持的待检验假说是：完整 prefix query 可能偏向近期/工作记忆，DINO-anchored query
可能更适合长程 episodic recall。简单 global/local agreement gate 同样不成立，因为两例都
强烈 disagree，而正确分支相反。

因此后续冻结如下：

1. **唯一主单位**是 `goal_c_t0/factual`（每 episode 一条真正 Revisit-C）；
   `goal_b_t0`、`goal_b_midpoint_t1` 与 counterfactual 只作因果对照，不进入主 Revisit rate；
2. 名称固定为 **DINO-anchored GCT query** 与 **full-prefix anchor-free GCT query**，不再使用
   容易误解为空间尺度的 “local/global”；
3. 主表报告四格：both good、anchored only、full-prefix only、neither，阈值为 bearing error
   `<=30 deg`；并按最新 teacher-positive support gap（`<=32`、`33--128`、`>128/no support`）
   分层；
4. 同时报告 same-process anchored 结果和既有 production `goal_append_warm` 参考，连续角度
   不要求跨进程相等，但 exact DINO anchor 与 `CDF@30` actionability 必须一致；当前 smoke
   为 `6/6` anchor 相同、`6/6` CDF@30 判定一致；
5. 先运行 10 scenes / 20 episodes 的 **futility-only** 门。若主单位出现
   `full-prefix only = 0`、`anchored only >= 5` 且 full-prefix rate 不高于 anchored，立即停止
   full-prefix 替代；未触发只能授权完整 train40，绝不能作成功声明；
6. 完整 train40 之前不训练、不闭环、不决定融合门。只有完整分层结果显示存在稳定的
   full-prefix-only headroom，才允许做 scene-OOF 的低容量增量探针；否则保留 DINO-anchored
   路径，不为架构新颖性强行增加 learned branch。

## 12. 2026-08-13 S-1 最终止损结果

10 scenes / 20 episodes 的 futility-only 运行在真正 `goal_c_t0/factual` 上得到：

- DINO-anchored `18/20`，full-prefix `5/20`；
- 四格 `both=5 / anchored-only=13 / full-prefix-only=0 / neither=2`；
- paired `+0/-13`，exact McNemar `p=0.000244`；scene-cluster 95% CI
  `[-85,-45] pp`；
- 20 条均为 long-gap，latest positive support gap 中位数 `269` frames；
- 相反，近期 Novel-B midpoint 是 anchored `4/20`、full-prefix `17/20`。

因此预注册停止门已触发：不运行 full40，不训练当前 candidate-free GCT adapter。完整证据与
复现边界见 `M2P_S1_ROLE_STRATIFIED_RESULT_20260813.md`。

后续 candidate-set 上限审计显示，factual Revisit-C 的 DINO top-1 为 `74/80`，现有
materialized candidates oracle 为 `79/80`。直接以 controller actionability 训练的 bounded
scene-OOF residual 得到 `76/80`、`+2/-0`、`p=0.5`，未通过运行前冻结的 `77/80` 和至少
3 个 gain-scenes 门。因此 learned reranker 同样不扩大；详见
`M2P_ACTIONABILITY_RESIDUAL_OOF_RESULT_20260813.md`。

当前方法决定收敛为：**冻结 DINO 负责 episodic content address，几何 certificate 负责
自认证并输出 scale-free bearing，失败时 exact native fallback。** learned 模块不因叙事偏好
进入主路径；只有未来新增独立 mixed decisions 后才重新审计。
