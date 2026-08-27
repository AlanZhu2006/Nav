# Certified Episodic Compass：两日完整进展与论文状态

更新时间：2026-08-15 04:25 CST（Asia/Shanghai）

覆盖范围：2026-08-13 至 2026-08-14 的方法收敛、2-leg/3-leg、held-out
MP3D、学习路线、Novel/X-NavDP、跨数据集、公开 benchmark 与 HPC 状态。

2026-08-15 凌晨之后的 proposal/verification 审计、GOAT repair-only 正式链、
Novel 因果对照和论文冻结门，统一续记在
`STATUS_20260815_NIGHTLY_PAPER_CONVERGENCE.md`；以该文件为最新运行状态。

2026-08-16 的非 MP3D 外部评测、全新 HM3D val10 冻结协议、Gibson/MemoNav
可运行性审计和正式 HPC job IDs 续记在
`NON_MP3D_EXTERNAL_EVAL_STATUS_20260816.md`。共享 SSH 的已确认故障模式与
强制操作规程单列在 `HPC_SHARED_SSH_OPERATIONS_20260816.md`；后续不得仅凭
no-PTY mux channel 超时判断 HPC 或用户认证失效。

2026-08-17 的 Pi3X learned relocalizer、Pi3X/LingBot-Map 分工、proof-head
训练数据与 scene-crossfit 泛化审计，以及 Final14 Attempt 1--4 正式确认链，统一
续记在 `STATUS_20260817_PI3X_LEARNED_RELOCALIZER_FINAL14.md`；以该文件为
当前最新状态。`LEARNED_RELOCALIZER_NIGHT_GOAL_20260817.md` 保留为完整开发日志，
不替代最新结论总账。

## 0. 执行摘要

项目已经从“继续叠加记忆、方向和 learned router”收敛为一条最小主线：

> 用因果在线视觉历史提出 Revisit 重定位假设，经几何 certificate 自认证后，
> 只向冻结 NavDP 输出 scale-free bearing；证据不足则精确回退原生 NavDP。

方法名暂定为 **Certified Episodic Compass（CEC）**。

这两天最重要的变化不是又找到一个更高的内部 SR，而是：

1. 主架构已经收敛到最小、可拒绝、role-free 的 residual interface；
2. Fresh160 的记忆来源被证明是 actual-online 可观测，而不是 expert-only 泄漏；
3. actual-online 3-leg 严格复测取得强配对结果，旧 43.5% 报告的因果缺陷得到修复；
4. CDEC、candidate-free GCT、小型 residual 和 graph rescue 都经过明确停止门，未被包装成正结果；
5. 第一批 16-scene held-out MP3D Attempt 7 已完成并独立复算：CEC 对 native
   无损显著提升，但只有 9 histories / 9 scenes，低于预注册统计规模；
6. CEC 相对简单 raw-DINO fixed bearing 的显著优势仍未建立；这正是 phase-2
   power expansion 要回答的唯一 P0 问题；
7. Replica 正式构造因 benchmark population 为零而停止；GOAT-Bench 的授权 HM3D
   `val v0.2` 已下载并精确解出全部 36 个 `val_unseen` 场景，当前正在构建隔离的
  Habitat 0.2.3 环境；本机精确版本的 2-scene simulator/task contract smoke 已通过，
  HPC 不可变副本正在排队确认；仍没有 GOAT SR。

当前最诚实的结论是：

- **已成立**：Revisit memory 能显著提升 frozen NavDP；actual-online 历史确实可用；
  role-free certificate 能在受支持 Revisit 上接管，并在 held-out Novel 上拒绝和
  exact fallback。
- **尚未成立**：certificate 比简单 raw-DINO fixed bearing 带来显著更优的
  safety/utility 风险—覆盖权衡；learned 模型能替代显式几何；Novel direction source
  已可部署；公开 benchmark 泛化已经完成。

状态标签在本文中严格使用：

- **confirmed**：分母、配对、统计和独立复算均通过；
- **strong internal**：结果强，但场景已消费或不是 scene-disjoint 论文确认；
- **underpowered**：协议有效，但未达到冻结的样本/scene-cluster 门；
- **mechanism/oracle**：只证明能力或瓶颈，不是可部署方法；
- **null/negative**：按原判据未通过，不继续调参包装；
- **infrastructure failure**：任何 query outcome 前失败，不能解释为方法结果。

## 1. 当前冻结主架构

```text
current ImageGoal + actual-online causal RGB history
        |
        v
DINO temporally-diverse top-8 retrieval
        |
        v
SuperPoint + LightGlue：候选几何排序/对应
        |
        v
LingBot history depth + PnP：恢复相对位姿假设
        |
        v
atomic certificate
  - inliers >= 16
  - query inlier hull coverage >= 5%
  - reference inlier hull coverage >= 5%
  - reprojection RMSE <= 2 px
        |
        +-- pass --> discard metric scale
        |            output unit bearing
        |            fixed 2.5 m residual
        |            frozen NavDP
        |
        +-- reject -> exact native NavDP
```

### 1.1 每个部件真正承担的角色

- **DINO** 是高召回 episodic content address，不被当作安全授权。
- **SuperPoint/LightGlue** 提供跨视角局部对应，并负责候选间的几何排序。
- **LingBot** 主要为历史 RGB 估计深度，使 2-D/3-D PnP 可恢复相对位姿；它不是
  controller，也不是一个独立全局导航器。
- **Certificate** 检验一个具体 pose/bearing 假设是否被当前图像证据支持。
- **Scale-free adapter** 删除不稳定的单目尺度，只保留方向，并使用冻结的 2.5 m
  controller residual。
- **NavDP** 始终是实际控制器；CEC 不训练或替换 NavDP。

### 1.2 它不是 Novel/Revisit 二分类器

运行时不读取 `Novel/Revisit` role、instance ID、GT pose、co-visibility、未来帧或
oracle bearing。`certificate reject` 只表示“当前历史无法自认证这个定位假设”，不等于
语义上判定目标一定 Novel；`accept` 也不是读取标签后的 oracle switch。

### 1.3 当前最小性

以下组件已经从主方法移除：

- graph rescue；
- CDEC learned proposal/cascade；
- candidate-free full-prefix GCT；
- active glance/原地扫描；
- X-NavDP 全局 controller 替换；
- metric distance output；
- post-hoc co-visibility gate。

保留它们作为 ablation、mechanism 或 negative result，不再进入默认 runtime。

## 2. 全项目核心证据总表

| 证据 | 结果 | 统计/边界 | 当前判断 |
| --- | ---: | --- | --- |
| 早期 2-leg geometry memory | `4/40 -> 19/40` | `+15/-0`, McNemar `p=6.1e-5` | 最早、最干净的可部署记忆增益 |
| Fresh160 supported Revisit | CEC `112/120`, native `27/120` | `+86/-1`, `p=1.137e-24` | strong internal；高共视、接近饱和 |
| CEC vs Fresh160 raw direct | `112/120 vs 106/120` | `+9/-3`, `p=0.146` | 未证明优于简单 direct |
| train40 certificate challenge | TP/FP/FN/TN `122/9/31/318` | precision `93.13%`, recall `79.74%` | certificate 有效但不是零误激活保证 |
| actual-online NNR 3-leg | `16/19 vs 5/19` native | `+11/-0`, `p=0.0009766` | strong internal，8 scene clusters |
| held-out Attempt 7 Natural | `10/18 vs 4/18` native | `+6/-0`, `p=0.03125` | scene-disjoint 正结果，但 underpowered |
| held-out Attempt 7 Revisit | `8/9 vs 2/9` native | `+6/-0`, `p=0.03125` | held-out Revisit utility |
| held-out Attempt 7 Novel safety | `0/9` accept/takeover | `9/9` exact fallback | 干净但 N 小 |
| Double-Revisit full memory | joint `12/20` vs native `0/20` | secondary `+12/-0` | 记忆链有价值 |
| 第二次 Revisit retained-A causal contrast | `12/14 vs 8/14` | `+6/-2`, `p=0.289` | 方向正确，未确认 |
| Novel oracle bearing | `28/40 -> 40/40` | `+12/-0`, `p=0.000488` | 强可恢复瓶颈；oracle，不是方法 |
| X-NavDP controller | `21/26` vs mixed/base `20/26` | `+2/-1`, `p=1.0` | controller 不是当前主瓶颈 |
| Replica room0 pilot | CEC/native 均 `7/8` | `+1/-1`, `p=1.0` | 跨域安全证据，无净增益 |

## 3. 最新 held-out MP3D：Attempt 7

这是截至本文更新时间最新、最重要的 scene-disjoint query 结果。五臂闭环全部完成，
官方 summarizer 和不导入 summarizer 的独立 verifier 一致。

### 3.1 冻结总体与统计功效

- 16-scene held-out source pool；
- 32 条 Goal-A source；
- native Goal-A 成功 17 条，覆盖 12 scenes；
- 最终可构造 role-pair history：9 条、9 scenes；
- 每条 history 产生一个 Novel 和一个 Revisit query；
- 五臂严格配对：`native / raw_direct / raw_fixed_bearing /
  geometry_fixed / certified`；
- runtime role visibility：`none`；
- 预注册目标：至少 20 histories / 12 scene clusters；
- 实际只有 9/9，因此结果有效但必须标为 **underpowered**。

### 3.2 Natural-direction 主协议

| arm | Novel | Revisit | all | all SPL |
| --- | ---: | ---: | ---: | ---: |
| native | `2/9` | `2/9` | `4/18` | `0.119` |
| raw-DINO direct | `1/9` | `8/9` | `9/18` | `0.343` |
| raw-DINO fixed bearing | `1/9` | `8/9` | `9/18` | `0.305` |
| old geometry fixed | `2/9` | `7/9` | `9/18` | `0.328` |
| certified | **`2/9`** | **`8/9`** | **`10/18`** | **`0.380`** |

CEC 对 native：

- all：`10/18 vs 4/18`，paired `+6/-0`，风险差 `+33.3 pp`；
- exact McNemar `p=0.03125`；
- scene-cluster bootstrap 95% CI `[+16.7,+50.0] pp`；
- Revisit：`8/9 vs 2/9`，`+6/-0`, `p=0.03125`，CI
  `[+33.3,+100] pp`；
- 六个 gain 位于六个不同 scenes，全部来自 Revisit；没有 loss。

Novel fail-closed 行为：

- `0/9` Novel certificate accept；
- `0/9` Novel memory takeover；
- `9/9` fully rejected and exact native physical rollout；
- 所以 CEC 与 native 均为 `2/9`。

### 3.3 与简单 baseline 的边界

CEC 相对 raw fixed：

- all：`10/18 vs 9/18`；
- paired `+2/-1`，`p=1.0`，CI `[-11.1,+22.2] pp`；
- Revisit：二者都是 `8/9`，逐条 `+0/-0`；
- CEC 相对 old geometry：`+1/-0`, `p=1.0`。

因此本结果不能支持“certificate 提高高支持 Revisit 的成功上限”。它目前支持的是：

> 在不读取 role 的情况下，certificate 保留 Revisit utility，同时在这 9 条 held-out
> Novel 上实现无接管、exact fallback。

### 3.4 Support-controlled 协议

| arm | Novel | Revisit | all |
| --- | ---: | ---: | ---: |
| native | `1/9` | `1/9` | `2/18` |
| raw-DINO direct | `4/9` | `8/9` | `12/18` |
| raw-DINO fixed bearing | `6/9` | `8/9` | `14/18` |
| old geometry fixed | `1/9` | `7/9` | `8/18` |
| certified | `1/9` | `8/9` | `9/18` |

CEC 对 native Revisit 为 `8/9 vs 1/9`，`+7/-0`, `p=0.015625`。但该协议故意
匹配 Novel 与 Revisit 的初始最短路 bearing；错误历史方向可能偶然成为正确 Novel 方向，
因此 raw fixed 的 Novel `6/9` 不能解释为 raw memory 真正识别或解决 Novel。该协议只用于
support/safety 机制，不进入自然 SR 主排名。

### 3.5 原始产物与独立复算

远端根目录：

`/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_role_pair_20260814T051634Z_attempt7`

关键文件：

- `paper_role_pair_summary.json`；本地镜像 SHA-256
  `a59cf2ffa256b3a2a42785dc41b5e7df79b57162645275327f6c4b50b6019d90`；
- `paper_role_pair_independent_verification.json`；`verified=true`，本地镜像
  SHA-256
  `1854c55afe402b48a36483168077f90d93764e33b29a36e88e0e79aa7174bde4`；
- population receipt SHA-256
  `2ecb102f137f0ec25abd615ec544f342cb4d259a9d945fa069041a8a5bb611bc`。

## 4. 2-leg Revisit 主线

### 4.1 最早显著基础结果

geometry memory 将 joint 从 `4/40` 提升到 `19/40`，paired `+15/-0`，
McNemar `p=6.1e-5`。它仍是“历史记忆本身能修复 frozen ImageGoal policy”的最早
干净闭环证据。

### 4.2 Fresh160 supported-Revisit

共享 Goal-A 成功分母为 120：

| arm | B given shared A |
| --- | ---: |
| native ImageGoal | `27/120 = 22.50%` |
| old geometry | `91/120 = 75.83%` |
| raw-DINO direct | `106/120 = 88.33%` |
| certified scale-free bearing | **`112/120 = 93.33%`** |

CEC 对比：

- vs native：`+86/-1`, `p=1.137e-24`，cluster CI
  `[+59.32,+81.74] pp`；
- vs geometry：`+23/-2`, `p=1.943e-5`，CI `[+8.77,+27.59] pp`；
- vs raw direct：`+9/-3`, `p=0.1460`，CI `[-1.75,+12.60] pp`。

actual-online 可观测性审计确认：

- conditional-B 的 `120/120` 都在真实 online-A 历史中有 max co-visibility
  `>=0.20`；
- `115/120` 达到 `>=0.50`；
- 全 160 中低于 0.20 的 11 条全部是 Goal-A failure，不进入 B given A；
- 因此 Fresh160 不是 expert-only 泄漏，但它确实是高共视 supported-Revisit，
  不能代表开放集边界。

### 4.3 Train40 certificate challenge

40 train scenes、480 sessions 的完整 actionability 审计：

| quantity | result |
| --- | ---: |
| accepted | `131/480` |
| GT-actionable | `153/480` |
| TP / FP / FN / TN | `122 / 9 / 31 / 318` |
| precision | `93.13%` |
| precision Wilson 95% CI | `87.46%--96.34%` |
| recall | `79.74%` |
| false-accept rate among non-actionable | `2.75%` |

这否决了“certificate 在未知输入上保证零误激活”的说法。九个 metric FP 中七个
anchor-relative direction error 仍小于 30°；由于部署接口只输出 scale-free bearing，
metric FP 不能直接等同九次闭环伤害。真正安全性必须由 mixed Novel/Revisit 闭环检验，
不能靠 post-hoc support gate。

详见 `TRAIN40_CERTIFICATE_REUSE_RESULT_20260814.md`。

## 5. 3-leg 与长期记忆

### 5.1 旧 43.5% 报告的问题

旧 strict-v4 报告为：native C `4/22=18.2%`、known-role direct
`14/22=63.6%`、role-free certified `10/23=43.5%`。它不能直接作为配对结论：

1. 三个方法使用了不同 conditional denominator；
2. Goal-C 来自 expert-A，而不是 frozen NavDP 实际 online-A 观察流；
3. 因而“记忆是否在线可见”和“controller 是否成功”混在一起。

### 5.2 Actual-online Novel -> Novel -> Revisit 严格复测

从 22 条 frozen source 中，19 条在 actual-online A/B prefix 下可构造，覆盖 8 scene
clusters；五臂共享 byte-identical A/B prefix：

| C controller | success |
| --- | ---: |
| native | `5/19` |
| known-role direct | `14/19` |
| role-free certified | **`16/19`** |
| certified + equal stuck budget | `16/19` |
| certified + graph rescue | `16/19` |

CEC 对 native：

- paired `+11/-0`；
- 风险差 `+57.9 pp`；
- McNemar `p=0.0009766`；
- scene-cluster CI `[+27.8,+85.7] pp`。

Graph rescue 实际发出了 92 个历史 graph plans，但相对 equal-budget 为 `+0/-0`，
因此 null 不是“没有激活”造成的。这个结果修复了旧报告的因果解释，但使用的是已消费
source scenes，属于 strong internal，而不是最终 scene-disjoint 确认。

详见 `SHARED_ONLINE_NNR_RETEST_PROTOCOL_20260814.md`。

### 5.3 Double-Revisit fresh20

| arm | B success | C eligible | C success | joint |
| --- | ---: | ---: | ---: | ---: |
| native | `5/20` | 3 | `0/3` | `0/20` |
| full memory | `15/20` | 14 | `12/14` | `12/20` |
| memory-B / native-C | `15/20` | 14 | `8/14` | `8/20` |
| role-free certified | `17/20` | 17 | `17/17` | `17/20` |

“第二次 Revisit 是否因保留更早 A memory 而改善”的预注册主比较是：

- full-memory C `12/14` vs memory-B/native-C `8/14`；
- paired `+6/-2`；
- 风险差 `+28.57 pp`；
- McNemar `p=0.2891`；
- cluster CI `[-15.38,+64.29] pp`。

方向正确但未达到确认；不能用 secondary `17/20` 把这个特定 causal claim 替代掉。

### 5.4 为什么 graph rescue 被移除

- 在 post-outcome fresh20 机制集上，graph rescue 把 `17/20 -> 20/20`，
  `+3/-0`, `p=0.25`；
- 在更严格 actual-online NNR 上，graph、equal-budget、direct 都是 `16/19`；
- 在 2-leg 五条 certified accepted-stuck failure 上只救回 `1/5`，`p=1.0`；
- 其执行还暴露出 route-start 定位与固定 2.5 m endpoint adapter 不匹配。

所以 graph rescue 是 topology-specific 内部机制，不是稳定通用 residual。论文主方法保留
最小 CEC bearing path。

## 6. Learned 路线的完整停止结论

### 6.1 CDEC proposal/ranker

在 40 train scenes、480 sessions 的 nested scene-OOF 候选排序上：

| selector | positive-session top-1 |
| --- | ---: |
| raw DINO | `115/155` |
| geometry | `126/155` |
| learned CDEC | `128/155` |

CDEC vs geometry 是 `+10/-8`, `p=0.8145`：有弱互补偏好，但不显著。

进入同进程 LingBot PnP/certificate 后：

| proposal | GT actionable | certificate accepted | certified-actionable |
| --- | ---: | ---: | ---: |
| geometry | 153 | 131 | **122** |
| CDEC | 135 | 122 | **115** |

CDEC-only 相对 geometry 为 `+1/-8`, `p=0.0391`，显著更差。geometry-first、
CDEC-on-reject cascade 在 349 次 fallback 中只增加一个 actionable session：
`+1/-0`, `p=1.0`。因此没有运行低上限的 160-episode 长闭环。

### 6.2 Candidate-free long-prefix GCT

真实 long-gap Revisit-C（20 sessions）：

- DINO-addressed GCT：`18/20` bearing error `<=30°`；
- full-prefix anchor-free GCT：`5/20`；
- paired `+0/-13`, `p=0.000244`；
- 10 scenes 中没有一个 scene 的 full-prefix 优于 anchored。

它说明当前 GCT cache/interface 能处理近期工作记忆，却不能在数百帧历史中自行完成
long-range content addressing；不是证明所有端到端记忆模型原则上不可能。

### 6.3 DINO-preserving 小型 residual

- frozen DINO：`74/80` at `<=30°`；
- DINO + scene-OOF bounded residual：`76/80`；
- paired `+2/-0`, `p=0.5`；
- gains 只来自两个 scenes；
- candidate oracle 为 `79/80`，实际可恢复错误本身很少。

它未通过冻结的 `>=77/80` 和 gains 至少跨 3 scenes 门，故不做八小时长训。

### 6.4 正确解释

这些结果不是“学习永远无用”，而是三个更具体的结论：

1. session 内 ranking AUC/top-1 不等于进入 PnP 后的 actionability；
2. open-set activation 的跨场景绝对校准比候选相对排序更难；
3. 当前训练集中真正可恢复的独立 ranking errors 太少，扩大容量/时长不会自动产生
   闭环统计功效。

因此当前项目选择 training-free/frozen pretrained geometry 不是审美偏好，而是被
scene-OOF、同进程 PnP 和闭环门共同筛选出来的结果。

详见 `CDEC_LEARNED_EPISODIC_DIRECTION_RESULT_20260813.md`、
`M2P_S1_ROLE_STRATIFIED_RESULT_20260813.md` 和
`M2P_ACTIONABILITY_RESIDUAL_OOF_RESULT_20260813.md`。

## 7. Novel 与 X-NavDP 的当前定位

### 7.1 Novel 仍由 frozen native NavDP 接管

当 causal history 无法认证目标时，CEC exact fallback，因此它当前不试图改善 Novel。

Novel oracle bearing 的 N=40 配对结果：

- native `28/40`；
- oracle periodic yaw `40/40`；
- oracle bearing + token `40/40`；
- paired `+12/-0`, `p=0.000488`。

它证明正确方向能恢复 Novel failure，也证明 frozen NavDP 有执行这种方向的控制能力；
但 bearing 来自 Habitat geodesic oracle，所以不是部署结果。

Active-glance 三版为 `20/40 -> 24/40 -> 25/40`，而配对 native 是 `31/40`。
Margin/gating 持续减少扫描和损害，但仍没有超过 native；原地扫描路线已经停止。

### 7.2 X-NavDP

在相同 verified PointGoal 上：

| controller | B given A |
| --- | ---: |
| mixed ImageGoal + PointGoal | `20/26` |
| base pure PointGoal | `20/26` |
| official X + MPC | `21/26` |

X 相对 mixed 为 `+2/-1`, `p=1.0`。X 的 signed/reverse control 在 deep-rear
bearing 上确有专长，但全局替换会失去 mixed ImageGoal conditioning，并产生抵消 gain 的
长倒车失败。因此 controller 不是当前主要瓶颈，X 不进入默认方法。

## 8. 跨数据集与公开 benchmark

### 8.1 Replica

`room_0` pilot：

- native `7/8`，CEC `7/8`，paired `+1/-1`, `p=1.0`；
- Novel certificate `0/4` 激活，4/4 exact fallback；
- raw direct/raw fixed 都只有 `3/8`，说明无认证 intervention 可严重破坏跨域 Novel。

正式 10-scene 构造经过三次 infrastructure/constructibility 修复后，最终 sealed
population 为：

- source-only `24/40`；
- native Goal-A `23/24`；
- materializable histories `9` / 4 scenes；
- frozen role-pair 最终仅 3 histories，全部来自已消费 `room_0`；
- 排除 pilot 后 fresh population 为严格的 `0 histories / 0 scenes`。

因此 Replica formal query gate 在读取任何 outcome 前停止。正确结论是：Replica v1
房间规模与当前 `>=2 m`、frame-39 长程在线历史合约不兼容；这是 benchmark
constructibility failure，不是方法失败，也不应通过缩短任务来追求分数。

### 8.2 官方 MP3D 与 MemoNav

- MP3D Habitat archive 已下载、校验并解压；90 scenes / 90 GLBs / 90 navmeshes；
- MemoNav 官方 MP3D test 的 18 scenes 资产齐全；1/2/3-goal 各 1008 episodes；
- 公开 episode 没有 goal-view rotation/RGB，官方仓库没有完整 evaluation code；
- MemoNav 使用 RGB-D panorama，当前 NavDP 使用单目前视输入。

所以可以做 compatibility/sequential-memory extension，但在 goal render 和 sensor
contract 冻结前，不能把我们的 SR 与 MemoNav published PR/PPL 直接横减。

### 8.3 HLoc standard-localization baseline

官方 HLoc 固定在 commit `c13273bd0ecc2917a35910fd843712a1c6243193`。
causal online-history SfM smoke：

- 30 个 decision frames 中注册 19 个；
- 重建 722 个 3D points；
- CPU 用时 29.30 s；
- mean reprojection error 1.014 px；
- 未读取 pose、depth、role 或 query。

但最终 online-A endpoint 未进入最大 component，query localization 与闭环尚未运行。
HLoc 是重要 localization-backend 对照，不是已经完成的 SR baseline。

### 8.4 GOAT-Bench：已采纳，尚无分数

GOAT-Bench 比继续自造一个 Replica benchmark 更适合检验长期、跨场景 episodic memory。
公开 `val_unseen` episode 审计得到：

- 36 HM3D scenes、360 sequential episodes、2,669 subtasks；
- ObjectGoal 991、LanguageGoal 856、ImageGoal 822；
- 338/822 个 ImageGoal 在此前 task list 中出现过相同 instance，覆盖 211 episodes。

这些只是 evaluator-only task recurrence，不是 runtime Revisit label；正式方法仍只能读取
因果在线 RGB history。

已完成：

- 官方 GOAT code、episode、released checkpoint 与 36 个 `val_unseen` image cache
  全部下载并固定；
- Matterport API token 通过授权预检；官方 HM3D `val v0.2` archive 下载完成：
  `3,530,618,880` bytes、300 tar members、100 GLB、100 navmesh，SHA-256
  `04c97761cb16ed8bd6f6600d4211ab10b9d3649d981401b527f0c0264a60371b`；
- 按 GOAT `val_unseen` episode 精确选择并解压 36/36 scenes、36 GLB、36 navmesh，
  共 `1,337,123,604` bytes；72 个文件全部哈希并写入 `SEALED`；
- NavDP cumulative metric trajectory -> GOAT 0.25 m/30° discrete action adapter；
- dataset audit、adapter 与 contract-smoke helpers 共 10 项 unit tests 通过；
- 冻结官方 `SUBTASK_STOP`、raw goal rendering、causal history 和 paired-arm 口径。
- 本机隔离环境已精确使用 Habitat-Lab/Sim `0.2.3` 跑通两个冻结 scenes/episodes：
  `4ok3usBNeis:3` 与 `5cdEh9F2hJL:4`；scene/navmesh、pathfinder、两张
  `512x512` raw ImageGoal、goal rendering 前后 agent pose、临时 sensor 清理、
  `SUBTASK_STOP` 单步转移和官方 metric transition 全部通过；
- NavDP adapter contract 同次验证通过：`0.25 m` forward、`30 deg` turn、近零
  trajectory 自动映射到 `SUBTASK_STOP`；当前 helpers/adapter 合计 11 项 unit tests 通过；
- 该 smoke 明确写入 `is_navigation_score=false`，不能解释为 GOAT SR/SPL。

本机复现同时发现并修复了两个纯基础设施问题：

- simulator/task smoke 不应从 monolithic policy YAML 启动；现直接 compose 官方
  `config/tasks/goat_stretch_hm3d.yaml`，避免无关 policy backend 和 Hydra search-path
  污染；
- 2026 resolver 会为 Python 3.7 选择 OpenCV 5，而 Habitat-Lab 0.2.3 依赖 OpenCV 4
  的 `applyColorMap` shape；环境现冻结 `numpy==1.21.6`、
  `opencv-python==4.7.0.72`。

尚未完成：

- 现有 NavDP 环境含 PyTorch/Hydra 但没有 Habitat-Lab/Habitat-Sim；现有
  TransportVGGT 环境只有 Habitat-Sim 0.3.3，不能与 GOAT 官方 Habitat 0.2.3
  contract 混用；
- 因此采用独立 prefix 构建 Habitat 0.2.3 环境。环境任务 `15738230` 已在安装
  Habitat-Lab、Habitat-Sim 和 GOAT-Bench 后，因无关的 `transformers` 拉取现代
  `safetensors`、继而在 Python 3.7 下缺少 `puccinialin` 而失败；未写环境 receipt；
  依赖 smoke `15738231` 运行 `0` 秒后取消，没有运行任何 episode；
- 同次事后 preflight 还发现 HPC host 缺 `libOpenGL.so.0`。修复版移除 contract smoke
  不导入的 policy-only dependencies，并沿用既有 NavDP immutable Singularity 系统镜像
  提供 native runtime；11 项本机测试和两个 shell syntax check 均通过；
- CPU 队列提交 `15739808`/`15739821` 均在运行前取消（scheduler 返回
  `PENDING (None)` 且无 start time）；相同 immutable source 改投为环境任务
  `15740152` 与依赖的 2-scene contract smoke `15740159`；环境已在 `1:55` 内成功，
  receipt 核对 Python `3.7.12`、Habitat-Sim `0.2.3`、NumPy `1.21.6`、OpenCV
  `4.7.0` 和 core imports；
- `15740159` 在运行 episode 前因 Python 3.7 `py_compile` 向 read-only bundle 写
  `__pycache__` 而失败。已改为只读 AST syntax audit 并禁用 bytecode writes；新 smoke
  `15740384` 与 H100 复测 `15740638` 随后都在 simulator initialization 阶段无法为
  CUDA device 0 创建 EGL context，仍为 `0` episode；跨 GPU family 的相同失败反证了
  “L40S 节点问题”。动态链接对照最终定位为 wrapper 显式覆盖 `LD_LIBRARY_PATH`，把
  Singularity `--nv` 注入的 `/.singularity.d/libs/libEGL.so.1` 换成普通容器 EGL。
  修复 bundle `goat_contract_smoke_26462619d0c26df9` 提交为 `15746123`，已在 HPC
  成功完成（38 秒）。两条冻结 episode 的 scene/pathfinder、raw goal rendering、
  metric depth、离散 adapter、`SUBTASK_STOP` 与官方 metric transition 全部通过。
  smoke 只验证
  scene/navmesh load、按 `InstanceImageParameters` 的 raw goal rendering、渲染不移动
  agent、临时 sensor 清理、与 RGB 共位同内参的 metric depth、离散 adapter、
  `SUBTASK_STOP` 和官方 metric transition；不产生 SR；incident receipt 见
  `GOAT_ENV_15738230_INCIDENT_20260814.json`；
- 已冻结并提交首个 ImageGoal 的 10-scene native-NavDP runtime gate `15750812`；
  manifest SHA-256 为
  `652cbe0f731c3b817e9c1e0f5e516ae4f386d74380a7ed06c4910651357b5db5`。
  它只运行每条 episode 的首个 ImageGoal 子任务，验证 observation/action/stop/runtime
  合约，不含 ObjectGoal/LanguageGoal controller，明确不是 GOAT score。完整 sequential
  pilot 和正式 GOAT SR/SPL 仍未运行。

协议见 `GOAT_BENCHMARK_ADOPTION_20260814.md`。正式顺序是 2-scene contract smoke、
10-scene first-ImageGoal runtime gate、复现 shared non-image controller 后的完整
10-scene sequential pilot、再跑 360 个 `val_unseen` episodes；主报 ImageGoal SR/SPL 的
native-vs-CEC 配对效应，full multimodal GOAT score 只作 hybrid secondary metric。

## 9. 两日内的评测与基础设施修复

### 9.1 Single-Revisit 构造修正

旧 wrapper 错误复用了 double-Revisit builder：要求第二 anchor、anchor gap 和 B->C
距离，并允许早于 LingBot runtime window 的 frame。修正后：

- 单 Revisit 只需一个 frame-39+ runtime-eligible anchor；
- source frame `[39, end-16)`、stride 8；
- 只施加与当前 query 因果相关的 2--9 m endpoint 约束；
- V1 视角扰动、co-visibility、certificate、controller、600-step budget 和统计不变。

### 9.2 Attempt 5/6/7

- Attempt 5 在任何 query rollout 前因 benchmark construction 过约束停止；
- Attempt 6 在 collection task 的 server port precheck/bind 间发生 TOCTOU race，
  query 未启动；修复为 owned-listener retry launcher；
- Attempt 7 从全新 immutable root 重跑，所有 jobs 完成，最终结果见第 3 节；
- 所有失败目录和 receipts 均保留，没有把 partial outputs 混入正式分母。

### 9.3 Phase-2 power expansion 当前状态

Phase-2 在读取 Attempt 7 query outcome 前冻结：

- 相同 16 scenes，不换场景；
- 每场景使用未进入 Attempt 7 的 `episode_0002..0005`；
- 64 Goal-A sources；
- method、threshold、arms、seed contract 均与 Attempt 7 相同；
- 只把 `episodes_per_scene` 从 2 泛化到 4。

原 job chain：

`15729697 collect -> 15729702 construction -> 15729707 eval ->
15729708 summary -> 15729714 verify`

首次提交状态：

- collection `15729697` 全部完成；
- construction summary `15729702` 在 17 秒失败；
- exact traceback：`require(len(receipt["traces"]) == 2,
  "scene trace count changed")`；
- evaluation、summary、verification 因 `afterok` 依赖全部取消；
- 该失败发生时没有 phase-2 query arm 被运行，没有新 SR，也没有方法污染。

这是 summarizer 仍硬编码两条 trace 的基础设施泛化遗漏。冻结的唯一允许修复是：

1. 从 frozen manifest 读取 `episodes_per_scene=4`；
2. 同时修正 `source_episodes` 和 materialization count 的一致性检查；
3. 生成新的 immutable repair bundle；
4. 对已完成 64 条 Goal-A trace 做完整 hash/receipt/manifest 校验；
5. 校验通过后仅重启 construction -> eval -> summary -> verifier；
6. 不改变方法、阈值、场景、episode、arms、budget 或统计。

上述修复已于 22:56 CST 完成并恢复提交：

- 新 immutable bundle：
  `/scratch/yz11502/Research/source_bundles/paper_power_expansion_repair_c18ef4e2021ef3b5`；
- source receipt SHA-256：
  `88d5983ebfa58c9970572a112483d0a65010c13b243d1b3a0e8f67ec66958a9d`；
- 独立 bundle diff audit 确认只有
  `summarize_paper_online_a.py` 和 `slurm_paper_online_a_summary.sbatch`
  两个 executable 文件改变，policy/controller 文件改变为 false；
- construction summary `15747746` 已完成，复用原 64 条 trace，没有重跑 collection；
- inventory：Goal-A `36/64` 成功，35 条 materialized history / 15 scenes；
- sealed role-pair population：19 histories / 12 scenes，两种 protocol 身份完全一致；
- phase-2 单独仍低于预注册 20 histories（差 1 条），因此单独报告必须标注
  underpowered；不得为补齐 1 条做第三次自适应扩样；
- evaluation `15747763` 已进入队列，`15747767` summary、`15747768` verifier
  保持 `afterok` 依赖；截至 22:58 CST evaluation 因 `QOSGrpGRES` pending，尚无
  query outcome。

修复协议和提交 receipt：
`PAPER_POWER_EXPANSION_REPAIR_PROTOCOL_20260814.md`、
`PAPER_POWER_EXPANSION_REPAIR_SUBMISSION_RECEIPT_20260814.json`。

### 9.4 SSH 身份纪律

本次核查确认正确 HPC account 为 `yz11502`。曾发现一个可用但属于 `yz11445` 的共享
ControlMaster；身份守卫阻止正式项目上传。误创建的一个空 GOAT 目录已用 `rmdir` 清理，
没有文件或 job 写入该账户。当前所有项目读取均显式绑定 `yz11502` socket。

## 10. 论文贡献应如何表述

### 10.1 当前可以写的主张

1. **Low-bandwidth episodic interface**：长程记忆不替换 controller，只提供一个
   scale-free bearing residual。
2. **Role-free open-set authorization**：系统不读取 Novel/Revisit 标签，而由可验证的
   几何证据决定接管或 abstain。
3. **Causal online memory**：正式 2-leg/3-leg 结果使用 frozen policy 的真实观察流，
   不是 expert demonstration memory。
4. **Exact fallback contract**：拒绝时恢复 byte-identical native path，使 safety 能通过
   paired closed-loop 直接检验。
5. **Evidence-driven minimalism**：学习、graph rescue、active scan 和 controller swap
   都有明确 negative/null gate，而不是在主图里堆模块。

### 10.2 当前不能写的主张

- “certificate 保证零 false positive”；
- “CEC 显著超过 raw-DINO fixed bearing”；
- “learned decoder 已替代 geometry”；
- “Novel navigation 已解决”；
- “oracle bearing 是可部署模块”；
- “Replica/GOAT 已证明跨数据集 SR 泛化”；
- “Double-Revisit retained-A benefit 已确认”；
- “X-NavDP 显著优于普通 PointGoal controller”。

### 10.3 工程拼接风险的真实边界

如果论文只写成“DINO + LightGlue + LingBot + PnP + NavDP”，确实有工程拼接风险。
真正需要强调的不是组件列表，而是被实验证明的设计原则：

> 在 frozen ImageGoal policy 上，episodic memory 的有效接口不是第二套 planner，而是一个
> 可拒绝、低带宽、scale-free 的方向残差；科学问题是其 open-set 风险—覆盖权衡。

这个 story 是否足以成为论文方法，取决于 phase-2 能否证明 certificate 相对 always-on
raw-DINO 在保留 Revisit utility 的同时显著降低 Novel harm。若 phase-2 仍然只是同 SR、
无足够 safety discordance，certificate 只能被表述为可靠工程外壳，而不是已证明优于简单
baseline 的核心算法贡献。

## 11. 当前唯一 P0 与停止规则

### P0：修复并完成 phase-2 power expansion

它回答唯一尚未解决、且直接决定论文主线的问题：

> CEC 是否相对 raw-DINO fixed bearing 形成更好的 Novel safety / Revisit utility
> 风险—覆盖权衡？

正式判读必须同时报告：

- Natural-direction all/Novel/Revisit SR 与 SPL；
- CEC vs native、raw fixed、old geometry 的 paired `+/-` 和 exact McNemar；
- scene-cluster bootstrap CI；
- certificate Novel accept/takeover、exact fallback；
- Revisit activation/recall；
- Attempt 7 单独结果、phase-2 单独结果以及按 scene cluster 合并的结果。

### 结果后的决策树

- 若 CEC 保持 Revisit utility，且显著减少 raw fixed 的 Novel loss：冻结为论文主方法；
- 若 CEC 与 raw fixed 在 utility/safety 上都近似相同：论文主张降为 training-free
  causal episodic bearing baseline，不再声称 certificate 是效果来源；
- 若 certificate recall 明显损失 Revisit 且 safety 没有补偿：退回 raw fixed/known-Revisit
  结果作为分析，不继续调阈值；
- 不允许第三次自适应 MP3D 扩样、不打开 blind、不读 development 调 operating point。

### P1/P2（不抢 P0）

1. 等待隔离 Habitat 0.2.3 环境任务 `15736735`，随后自动执行 2-scene
   official-contract smoke `15736912`；未通过前不报分。
2. HLoc 完成 endpoint + query localization/abstention；只有接口通过才增加 secondary
   closed-loop arm。
3. MemoNav 先冻结 goal-image render 和 sensor adapter，再决定是否做兼容性 extension。

明确不做：新一轮八小时 learned reranker、恢复 graph rescue、继续 active glance、全局
X-NavDP 替换、根据 Attempt 7 调 certificate threshold。

## 12. 关键文档与产物索引

### 方法与主要结果

- 本总账：`MemNavData/STATUS_20260814_PAPER_EVAL.md`
- 论文评测协议：`MemNavData/PAPER_EVALUATION_PROTOCOL_20260814.md`
- construction amendment：`MemNavData/PAPER_CONSTRUCTION_AMENDMENT_20260814.md`
- phase-2 协议：`MemNavData/PAPER_POWER_EXPANSION_PROTOCOL_20260814.md`
- phase-2 repair：
  `MemNavData/PAPER_POWER_EXPANSION_REPAIR_PROTOCOL_20260814.md`
- Fresh160 online observability：
  `MemNavData/CERTIFIED_RELOCALIZATION_ONLINE_OBSERVABILITY_RESULT_20260813.md`
- train40 certificate：`MemNavData/TRAIN40_CERTIFICATE_REUSE_RESULT_20260814.md`
- actual-online NNR：`MemNavData/SHARED_ONLINE_NNR_RETEST_PROTOCOL_20260814.md`
- Double-Revisit：`MemNavData/SHARED_ONLINE_DOUBLE_REVISIT_FRESH_PROTOCOL_20260813.md`
- learned CDEC：`MemNavData/CDEC_LEARNED_EPISODIC_DIRECTION_RESULT_20260813.md`
- GCT role audit：`MemNavData/M2P_S1_ROLE_STRATIFIED_RESULT_20260813.md`
- learned residual：`MemNavData/M2P_ACTIONABILITY_RESIDUAL_OOF_RESULT_20260813.md`

### 外部 benchmark

- external baseline matrix：`MemNavData/EXTERNAL_BASELINE_MATRIX_20260814.md`
- HLoc protocol：`MemNavData/HLOC_ONLINE_HISTORY_BASELINE_PROTOCOL_20260814.md`
- Replica protocol：`MemNavData/REPLICA_CROSS_DATASET_PROTOCOL_20260814.md`
- GOAT protocol：`MemNavData/GOAT_BENCHMARK_ADOPTION_20260814.md`

### Attempt 7 / phase-2 远端路径

- Attempt 7：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_role_pair_20260814T051634Z_attempt7`
- phase-2 run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_power_expansion_20260814_pre_result`
- phase-2 immutable source bundle：
  `/scratch/yz11502/Research/source_bundles/paper_power_expansion_915f6c6a30837ee5`
- phase-2 failure log：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/paper_A_sum_15729702.err`

## 13. 最终一句话判断

截至 2026-08-14 17:55，项目已经证明“因果在线 Revisit memory 能显著修复 frozen
NavDP”，也第一次在 underpowered held-out mixed-role 测试中证明 CEC 可以无损拒绝 Novel、
接管 Revisit；但论文最关键的增量尚未完成：**certificate 是否比简单 raw-DINO fixed
bearing 提供统计上更优的开放集 safety/utility trade-off**。下一步只修 phase-2
summarizer 并完成预冻结扩样，不再增加新架构。

## 14. 2026-08-15 增量：GOAT stopping 归因与到达证书

这一节晚于上面的 2026-08-14 总账，是当前最新状态；它不改变 Revisit 论文主线，
而是解决公开 GOAT runtime adapter 暴露出的独立接口错误。

### 14.1 GOAT pilot 发现的不是导航 SR，而是 STOP 语义错误

第一批 GOAT runtime pilot 已完成 10/10 条记录，但 official first-subtask SR 为 0/10。
其中 9/10 条所谓“自主停止”发生在离目标 2.56--14.87 m 的位置。根因不是 critic 太弱，
而是 adapter 把 NavDP 返回的 selected zero trajectory 直接映射成 `SUBTASK_STOP`。

代码审计确认 NavDP 会把预测终点 `<0.5 m` 的候选统一钳成零轨迹，而 GOAT 的成功半径
是严格 `<0.25 m`。因此零轨迹在信息论上不能区分真到达与 `(0.25,0.50] m` near miss；
生产语义必须是 `zero = abstain/replan`，不能是 `zero = STOP`。

### 14.2 同一批 939 states 的 candidate-consensus 审计已经否决

冻结 train40 population：40 scenes、80 episodes、160 goals、939 states，其中 160 个
`<0.25 m` arrival、779 个 non-arrival；每状态 4 个固定 seed、每 query 16 candidates。

- selected-zero persistence AUC：0.7087；
- all-candidate zero fraction AUC：0.7244；
- top-4 zero fraction AUC：0.7164；
- 冻结网格没有任何一个 operating point 达到 0 false positive；
- 最保守的“所有候选均为零”仍为 TP=56、FP=43；
- 43 个 FP 中 39 个位于 `(0.25,0.50] m`。

所以“多 seed / 多候选一致收缩”不能恢复已被 0.5 m clipping 丢掉的分辨率。完整协议、
repair provenance 与 sealed report hash 见
`MemNavData/NAVDP_ARRIVAL_CONSENSUS_PROTOCOL_20260815.md`。

### 14.3 当前在跑的最小解：metric image-goal arrival certificate

新审计不增加 navigation controller，只在 native NavDP 本来要输出零轨迹时运行：

```text
native sample-0 zero proposal
    -> current RGB <-> goal RGB LightGlue correspondences
    -> frozen Fundamental/PnP geometry certificate
    -> causal LingBot depth + frozen first-prefix monocular scale
    -> certified predicted distance <= frozen threshold
       yes: candidate STOP
       no: abstain/replan
```

它使用与 consensus 审计完全相同的 939 states；不读 simulator depth，GT distance 仅在
推理后由 summarizer 使用。阈值网格、0-FP/TP>=20/scene>=10 通过门和 tie-break 已在看
正式输出前冻结，见 `MemNavData/LINGBOT_PNP_ARRIVAL_PROTOCOL_20260815.md`。

工程 smoke（首个 lexicographic episode）已完成：12/12 行、10/12 通过 2D precheck、
8/12 通过完整 certificate，纯采集耗时 46.9 s。两条真到达的 GT/预测距离分别为
`0.007/0.004 m`、`0.019/0.021 m`；两条 near miss 分别为 `0.371/0.369 m`、
`0.393/0.387 m`。这只证明实现的坐标与尺度没有显然错误，不作为效果证据。

截至提交时的 immutable job chain：

- formal 8-shard collection：`15756118`；
- deterministic summary：`15756133`；
- independent verifier：`15756312`；
- collection bundle SHA：
  `7aa18c2a9608994ed20d2ed7af812d68da6e712561366fa90b144b4b66818ed3`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_pnp_arrival_20260815/run_7aa18c2a9608994e_20260814T182158Z`。

正式结果即使通过 train gate，也只允许冻结一个阈值去做一次 disjoint GOAT
confirmation；不能直接写成公开 benchmark 提升，不能读取 GOAT 后回调阈值。

### 14.4 当前两条 HPC 主任务的关系

- `15747763` 仍是论文主线的 phase-2 power expansion，回答 CEC 是否显著优于
  raw-DINO 的 safety/utility trade-off；
- `15756118` 是 GOAT adapter 的 arrival-safety blocker，回答冻结 NavDP 何时可以安全
  发出官方 STOP。

二者问题不同、数据不同、结论不能混算。调度上均限制并发，目标是让两条线继续而不
取消任何已完成结果。

## 15. 2026-08-15 最新结算：Phase-2 已完成，GOAT 到达确认已冻结提交

本节是全文最新口径，取代第 9.3、11、13 和 14.3 中仍写为“在跑/P0”的状态描述。

### 15.1 Phase-2 power expansion 最终结果

修复后的完整 query、summary 和 independent verifier 均已完成；`verified=true`。
冻结 population 为 19 histories / 12 scenes，低于预注册 20/12 一条 history，因此
phase-2 单独仍标为 underpowered，且不做第三次自适应补样。

Natural-direction 主协议（19 Novel + 19 Revisit）：

| arm | Novel | Revisit | all |
| --- | ---: | ---: | ---: |
| native | `4/19` | `1/19` | `5/38` |
| raw-DINO fixed bearing | `9/19` | `18/19` | `27/38` |
| old geometry fixed | `4/19` | `19/19` | `23/38` |
| certified | `4/19` | `17/19` | `21/38` |

Certified 对 native：

- all `21/38 vs 5/38`，paired `+16/-0`，风险差 `+42.1 pp`；
- exact McNemar `p=3.0518e-5`；
- scene-cluster 95% CI `[+32.14,+50.0] pp`；
- Revisit `17/19 vs 1/19`，同样 `+16/-0`；
- Novel `4/19 vs 4/19`，逐条 exact fallback。

Open-set 行为：

- Natural 与 support-controlled 两个协议均为 `0/19` Novel certificate accept、
  `0/19` Novel takeover；
- `19/19` Novel fully rejected exact-native；
- `19/19` Revisit 都发生 certificate activation；
- runtime failure plans 为 0。

但决定论文增量的简单 baseline 比较没有通过：

- Natural certified vs raw fixed：`21/38 vs 27/38`，paired `+1/-7`，
  `p=0.0703`，风险差 `-15.8 pp`，cluster CI `[-28.57,-2.94] pp`；
- Revisit：`17/19 vs 18/19`，`+0/-1`, `p=1.0`；
- certified vs geometry：`21/38 vs 23/38`，`+0/-2`, `p=0.5`；
- support-controlled 中 raw fixed 的 Novel 方向被构造性匹配放大，总 SR
  `30/38`，不能作为自然导航主排名，但再次说明该协议只适合 safety/support 分析。

所以 phase-2 给出的结论不是“CEC 失败”，而是更窄、更严格的分解：

1. Revisit memory utility 在新 source 上再次成立；
2. certificate 的 Novel abstention 和 exact fallback 成立；
3. certificate **没有**被证明比简单 raw-DINO fixed bearing 提高 SR，且当前 coverage
   代价会损失 1--2 条 Revisit；
4. 因而 certificate 目前应定位为 open-set safety authorization，而不能声称是更高
   Revisit 成功率的来源；“显著更优 safety/utility trade-off”这一预设论文主张未建立。

Sealed artifacts：

- summary：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_power_expansion_20260814_pre_result/paper_role_pair_summary.json`；
- summary SHA-256：
  `0cddac3a919035c28a56ecfaffbc54eac6bfeea3b15c63b09eaffdbbe831b3fd`；
- independent verifier SHA-256：
  `347c9d4406456dc35ee48759a3ec95bab8152b9391ee9024ac77b14c0da0152e`。

### 15.2 LingBot/PnP train-only arrival audit 最终结果

80 episodes / 40 MP3D train scenes、939 exact states 已全部采集并独立复算：

- strict `<0.25 m` arrival：160；non-arrival：779；
- 冻结 tie-break 选出 predicted distance `<=0.075 m`；
- `native zero AND PnP certificate`：TP/FP/FN/TN = `76/0/84/779`；
- precision `1.0`，recall `0.475`，TP 覆盖 38 scenes；
- PnP-only 同阈值：TP `114`、FP `0`、recall `0.7125`，覆盖 40 scenes；
- train gate 通过，唯一合法下一步为不调阈值的 disjoint GOAT confirmation。

Formal report：

`/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_pnp_arrival_20260815/run_7aa18c2a9608994e_20260814T182158Z/merged/report.json`

- report SHA-256：
  `13f265b200f02c877557bdc18a846688274961ddc451ead463dbcb319d528373`；
- independent verifier SHA-256：
  `ffb2576ef25f1a0ff571d66640ec7cddd611417858a35d4e15de1c6ef2ea7dfd`；
- verifier：`verified=true`。

### 15.3 已冻结的 GOAT 在线确认

已完成代码与 20 项本地单测：

- NavDP zero trajectory 改为 typed `ARRIVAL_PROPOSAL`，adapter 永远不能直接发
  `SUBTASK_STOP`；
- 证书拒绝时先使用同一 16-candidate batch 中 critic 最高的可执行 motion；
- 若整批无 motion，最多三次同观测、固定 seed、FIFO 不变的只读 resample；
- online sidecar 只使用因果 RGB、current-goal LightGlue/PnP、LingBot depth 和
  first-64-frame causal scale；禁止 pooled-scale fallback 和 simulator depth；
- 只有 native zero、certificate accept、strict scale available、预测距离
  `<=0.075 m` 四者同时满足才授权官方 STOP；GT 在冻结决策后才读取。

正式 population 是 20 个 HM3D `val_unseen` scenes 各一条 ImageGoal-first episode，
与已消费十场景 runtime pilot 完全 scene-disjoint。主 gate 预注册为：0 false certified
stops、至少 5 true certified stops、且覆盖至少 5 scenes。这是 ImageGoal semantic-stop
confirmation，不是 full GOAT score。

协议：`MemNavData/GOAT_CERTIFIED_ARRIVAL_CONFIRMATION_PROTOCOL_20260815.md`。

不可变提交：

- source bundle：
  `/scratch/yz11502/Research/source_bundles/goat_certified_arrival_1060f43722e7a2c2`；
- source receipt SHA-256：
  `1060f43722e7a2c261e1b2c7224b42142ff460d7ac4cd860dfc9834484ea3d24`；
- smoke `15759008`；formal array `15759010`；summary `15759014`；
  independent verifier `15759020`；
- formal manifest SHA-256：
  `3120625b8b6e86d9d517f08dd4d3b366c0d417cfdb738b50dc01834186458b79`；
- formal run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_certified_arrival_20260815/formal_20260814T200728Z`。

截至本次更新时间，smoke 已通过提交与资源合法性检查，因 Slurm 资源槽位等待；formal
及两个后处理任务保持严格 `afterok` 依赖，尚未产生 GOAT confirmation outcome。

### 15.4 当前项目决策

论文主线现在应分成两个不混淆的结论：

- **utility**：因果在线 Revisit bearing 对 frozen NavDP 有大幅、跨构造重复的闭环增益；
- **authorization**：geometry certificate 提供 role-free Novel abstention / exact fallback，
  但目前没有证据表明它提高 supported-Revisit 的 SR 上限。

因此接下来不再训练 learned ranker 或补第三批 MP3D。近期唯一运行中的证伪实验是 GOAT
semantic-stop confirmation；其结果只决定 7.5 cm arrival layer 是否能进入公开 GOAT
runtime，不反向改写 Revisit phase-2 的负比较结果。

## 16. 2026-08-15：GOAT arrival 负结果与 sequential-Revisit 正式协议

### 16.1 first-ImageGoal arrival 已正式失败并关闭

20 scenes / 20 episodes 的 frozen GOAT arrival confirmation 已完成且独立复算：

- certified success `0/20`；certified STOP `0`；
- forced guard stop `20/20`；
- 28 次 native-zero 中 26 次实际仍在官方 `<0.25 m` 成功区外；
- preregistered gate 要求至少 5 个 true certified STOP、覆盖至少 5 scenes，实际均为 0；
- `do_not_claim_deployable_goat_semantic_stop`，禁止在这 20 条上重调阈值。

完整结果：`MemNavData/GOAT_CERTIFIED_ARRIVAL_FORMAL_RESULT_20260815.md`。

### 16.2 为什么改测 sequential Revisit

arrival branch 测的是“当前视图能否授权 GOAT STOP”，并未测试 CEC 已成立的核心机制。
新协议直接测 released GOAT 序列中同一 instance 再次成为 ImageGoal 时，因果在线历史能否
经原有 certificate 输出 bearing 并改善 frozen controller。

dataset-only 审计为：36 scenes / 360 episodes / 822 ImageGoals，其中 338 个 exact
repeated ImageGoal 分布在 211 episodes、36 scenes。正式集排除本机已消费的 `4ok...`、
`5cd...`，其余 34 scenes 各按 outcome-blind 规则冻结一条最早 recurrence episode。

### 16.3 正式口径

- 主比较：34 条全部 ITT，official GOAT vs role-free CEC；
- CUDA、released stochastic sampling、seed 100、每臂官方 5000 action budget；
- CEC 只可在 certificate accept 后覆盖非 STOP motion；
- 官方 `SUBTASK_STOP` 必须逐次原样执行；GOAT 成功不要求朝向，因此正式方法不含 U-turn；
- reject/error 必须 action/pose exact fallback；
- target reach、prior success、candidate/certificate coverage 都只作诊断，不能事后过滤分母；
- primary test 为 paired risk difference + two-sided exact McNemar + scene-cluster CI；
- 独立 verifier 从 raw JSON 复算。

短历史兼容性已修正：certificate 的 DINO shortlist 不再被无关的 learned decoder
`S+W=40` warm-up 阻塞；八帧 scale block 后即可提出 causal candidates，但几何阈值完全不变。
已消费场景端到端 CUDA smoke 为 28 步两臂严格一致、0 accept、exact fallback；只证明链路，
不计论文 SR。

冻结协议：`MemNavData/GOAT_SEQUENTIAL_REVISIT_FORMAL_PROTOCOL_20260815.md`。当前 48 项
测试和 Python 3.7/Bash 审计通过。冻结时 HPC 提交曾因没有以 `yz11502` 身份认证的 live
SSH master 而 fail-closed；连接恢复后的正式提交记录见 16.4。

提交前又完成 5000-step 容量路径审计：以正式 `MEMNAV_MAX_FRAME_NUM=6000` 启动、
5001-step reset 后连续写入 2052 帧，实际跨过 1024/2048 RoPE 边界；384.2 秒完成、
无异常、结束显存约 24.0 GiB。临时 105 MiB JPEG 缓存已删除，GPU 回到 255 MiB。
同时补齐 immutable bundle 的动态 import closure，并把重试日志改为包含 Slurm job id，
避免重提同一 manifest index 时覆盖先前的故障证据。
正式 manifest 还新增 17/17 的 native-first / CEC-first 冻结顺序平衡，消除固定 arm
顺序可能带来的 CUDA/环境 warm-state 混杂；每臂 seed 与起点配对规则不变。
已消费场景的 `CEC->native` 反向顺序 CUDA smoke 同样得到 28 步动作/位姿 exact pairing，
证明配对不依赖 native-first 的运行偶然性。

### 16.4 Sequential-Revisit 正式 HPC 提交（2026-08-15）

`yz11502` 共享 SSH 恢复后，提交前 dry-run 再次完整通过：48 项相关测试通过，
远端 base bundle、四个 checkpoint/weight、GOAT/HM3D 环境以及两个冻结 manifest 的
SHA-256 均与预注册值一致。随后生成新的只读 source bundle 并提交完整依赖链：

- source bundle：
  `/scratch/yz11502/Research/source_bundles/goat_sequential_revisit_356c19b2e4bb9549`；
- source receipt SHA-256：
  `356c19b2e4bb954983b24e88401b4fd9231d857bdb400b3f9c187823b0551368`；
- smoke job：`15781705`，仅运行已消费 scene 的 manifest index 1；
- formal array：`15781715`，34 scenes，array `0-33%4`；
- summary：`15781725`；independent verifier：`15781737`；
- smoke root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/smoke_20260815T110413Z`；
- formal root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/formal_20260815T110413Z`。

依赖关系已从 Slurm 独立读取确认：
`15781705 -> afterok:15781715 -> afterok:15781725 -> afterok:15781737`。
远端 `submission.json` 记录 `selected_scenes=34`、
`method_or_threshold_selection_allowed=false`，以及 formal manifest SHA-256
`aaedc6fb0c6d3787b5c8c61eed2c2d943320f595f9b1783f881febc544121397`。

截至 2026-08-15 07:13 EDT，smoke 尚未运行，状态为 `PENDING (Priority)`；Slurm
当时估计 11:22 EDT 启动。30 分钟、1 小时、2 小时和 8 小时时限的只读
`sbatch --test-only` 均给出相同或更晚的预测起点，说明等待来自 fair-share/预约优先级，
而不是 smoke 沿用 8 小时时限导致。因此保留原 job 和全部 receipt，不取消、不重投；
正式结果必须等 smoke、34-scene array、summary 和 verifier 四段全部 `afterok` 完成后报告。

### 16.5 首次 smoke 基础设施失败与冻结修复

Smoke `15781705_1` 于 07:12:44 EDT 在 H100 `gh002` 启动，并在 3 分 06 秒后以
exit code `1:0` 结束。MemNav 和 NavDP 两个服务均成功加载、绑定端口；官方 GOAT runner
在创建环境或执行任何 episode 前退出：

```text
ModuleNotFoundError: No module named 'clip'
```

该次 smoke 的 `episodes/` 为空，stdout/stderr 主日志为空；错误只写入 job-scoped
`logs/episode_001_15781705.log`。因此它是运行环境依赖闭包遗漏，不是方法结果，且没有消费
formal population。下游 `15781715` 显示 `DependencyNeverSatisfied`，summary `15781725`
和 verifier `15781737` 也没有运行。

根因是本地 GOAT 环境通过 `.pth` 隐式暴露 OpenAI CLIP，而 HPC frozen env 中没有该
package；原提交只冻结了项目源码和权重，没有把这个源码依赖收入 bundle。修复不改变
GOAT policy、checkpoint、seed、manifest、动作或 certificate：

- vendor 本地已验证的 `openai/CLIP` commit
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` 的五个运行文件；
- 在隔离 runner 的 `PYTHONPATH` 中显式加入该只读目录；
- bundle manifest、submission receipt 和每个 task 的 environment log 都记录 upstream
  commit；
- task 启动前验证 `upstream.json` 与五个文件齐全；
- 本地 48 项协议测试、Bash 语法、Python 3.7 import/tokenize smoke 均已通过。

旧链不可作为结果。下一次提交必须先用新的只读 source bundle 通过 smoke，再释放同一份
34-scene frozen formal manifest；不得根据本次基础设施失败修改方法或 population。

### 16.6 CLIP 依赖闭包修复后的重提

进一步审计发现 HPC 用户 cache 中也没有官方 `RN50.pt`；若仅 vendor Python package，
OpenAI CLIP 会在计算节点尝试联网下载。因此把本机已有的官方 RN50 资产上传至：

`/scratch/yz11502/Research/datasets/goat_bench_20260814/data/goat-assets/checkpoints/openai_clip/RN50.pt`

文件为 255,827,503 bytes、权限 `0444`，SHA-256 为 OpenAI URL 中公布的：

`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`。

默认 `/home/yz11502/.cache/clip/RN50.pt` 是精确指向该只读资产的 symlink。Job 启动前
同时校验 symlink target、文件 SHA、vendored CLIP commit 与五个源码文件；dependency
receipt、source manifest、submission receipt、每个 task environment log 均记录相应
provenance。修复版完整 dry-run 再次通过 48 项测试、Python 3.7 import/tokenize 和全部
远端依赖哈希。

新的正式链：

- source bundle：
  `/scratch/yz11502/Research/source_bundles/goat_sequential_revisit_d395581288a89546`；
- source receipt SHA-256：
  `d395581288a89546e95e0e61d77ba5d4d23b68acf81e01c54aed46739603cfd3`；
- smoke：`15784790`；formal array：`15784792`；summary：`15784794`；
  verifier：`15784795`；
- smoke root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/smoke_20260815T125524Z`；
- formal root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/formal_20260815T125524Z`。

该链继续使用完全相同的两个 manifest、34-scene ITT population、seed、arm order 和方法；
唯一变化是补全原本隐式存在于本机环境、但 HPC 缺失的官方运行依赖。首次失败链保留为
incident 证据，不复用任何输出目录。

### 16.7 第二次零-episode 依赖失败与 CPU dependency gate

修复版 smoke `15784790` 于 09:02:28 EDT 启动，MemNav/NavDP 服务和 vendored OpenAI
CLIP 均已进入初始化，但在任何 episode 创建或动作执行前因 CLIP 的传递依赖缺失退出：

```text
ModuleNotFoundError: No module named 'ftfy'
```

该 job 用时 2 分 12 秒、exit code `1:0`；run root 中只有 job-scoped 环境/服务/异常日志，
没有 `episodes/` 输出。formal `15784792` 和 summary `15784794` 因 `afterok` 未满足取消，
因此第二次 incident 同样没有消费 formal population，也不构成方法结果。

随后没有继续逐个碰撞式补依赖，而是对 OpenAI CLIP 的完整 import closure 做了隔离审计，
把以下与本机已验证环境完全一致的只读运行依赖收入 immutable bundle：

- `ftfy 6.1.1`；
- `regex 2024.4.16`（module version `2.5.141`，Python 3.7 ABI）；
- `wcwidth 0.2.14`。

本机临时隔离环境先完成了 tokenizer、RN50 CPU load 和 GOAT lean-config 的精确调用顺序
测试。为避免第三次占用 GPU 才发现依赖错误，正式链新增 CPU-only dependency preflight：
它使用与 GPU job 相同的 Singularity、GOAT Python、bundle、manifest scene 和环境变量，
验证所有依赖来源及哈希，加载 102,007,137 参数的官方 RN50，并注册官方
`goat_bench.models.clip_policy`；它不创建 Habitat episode，不执行任何方法决策。

第三次 immutable chain 为：

- source bundle：
  `/scratch/yz11502/Research/source_bundles/goat_sequential_revisit_0dc991351847119b`；
- CPU dependency preflight：`15785472`；
- GPU smoke：`15785480`；
- formal array：`15785482`（34 scenes，array `0-33%4`）；
- summary：`15785483`；independent verifier：`15785484`；
- smoke root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/smoke_20260815T132014Z`；
- formal root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/formal_20260815T132014Z`。

CPU preflight `15785472` 已于 09:25:17 EDT 完成，elapsed 39 秒、exit code `0:0`。
机器可读 receipt 为 `preflight/dependency_preflight.json`，记录 `complete=true`、
`episodes_executed=0`；RN50 CPU load 用时 1.959 秒，模型参数量和权重 SHA-256 均与
冻结值一致，官方 GOAT CLIP policy 导入成功。stderr 中只有 Gym 维护状态警告，不是异常。

因此截至该时间，运行时依赖闭包已通过同容器 CPU gate；GPU smoke 处于正常调度等待，
formal/summary/verifier 继续由严格 `afterok` 链保护。两个冻结 manifest、34-scene ITT
population、seed、arm order、certificate 阈值和动作语义从首次提交起均未改变。

### 16.8 官方 GOAT CUDA allocator 兼容修复与第四链

第三链 CPU preflight 通过后，GPU smoke `15785480_1` 于 H200 `gh122` 启动。它已越过
此前的 `clip` / `ftfy` import 失败，成功初始化 dataset、GOAT simulator 和 task，但在
官方 CLIP policy 首次迁移到 CUDA 时、任何导航动作和方法决策发生前退出：

```text
RuntimeError: Unrecognized CachingAllocator option: expandable_segments
```

根因不是 H200 或 checkpoint，而是一个跨环境兼容错误：同一 job 内的 MemNav/NavDP 服务
使用较新的 PyTorch，适合 `expandable_segments:True`；冻结 GOAT 环境却是
PyTorch `1.13.1+cu117`，不认识该较新 allocator 选项。该变量此前在 sbatch 顶层 export，
因而被官方 GOAT 子进程错误继承。旧 arrival confirmation 没有实例化官方 GOAT policy，
所以不会触发这条 CUDA 初始化路径。

修复保持两个服务的显式 allocator 配置不变，但取消 job-global export，并用
`env -u PYTORCH_CUDA_ALLOC_CONF` 启动官方 GOAT 进程。HPC 同一 Singularity/同一 GOAT
Python 的隔离验证确认：即使父进程设置该变量，子进程中也为 `None`，torch 版本仍为
`1.13.1`。相关冻结提交测试从 48 增至 49 项并全部通过；扩展相关回归共 81 项通过。

第四个 immutable chain：

- source bundle：
  `/scratch/yz11502/Research/source_bundles/goat_sequential_revisit_ccbbb8418e4bc275`；
- source receipt SHA-256：
  `ccbbb8418e4bc275a9d3bb786968b9d7279cb6cf493235b16269a1dc0d1dcf4a`；
- CPU dependency preflight：`15787764`；
- GPU smoke：`15787765`；
- formal array：`15787766`（34 scenes，array `0-33%4`）；
- summary：`15787768`；independent verifier：`15787770`；
- smoke root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/smoke_20260815T134809Z`；
- formal root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/formal_20260815T134809Z`。

第四链 CPU preflight 已完成（39 秒，exit `0:0`）。receipt 新增并验证：
`torch_version=1.13.1`、`pytorch_cuda_alloc_conf=null`、`episodes_executed=0`；官方 RN50
仍为 102,007,137 参数且 SHA-256 不变，官方 GOAT CLIP policy 注册成功。

GPU smoke `15787765_1` 随后提前在 A100 `ga010` 启动，并于 2 分 29 秒后以 `0:0`
完整结束。它跑完一条两臂配对 episode：official policy device 为 `cuda:0`，执行顺序为
CEC→native，28 个 override 前动作/位姿全部配对，CEC 0 accept，两臂 target success 均为
0；该场景是已消费工程 smoke，`paper_claim_authorized=false`，不计论文 SR。原始 JSON 的
sidecar SHA 校验通过。该 smoke 证明官方 GOAT CUDA policy、两个服务、环境 reset、配对
rollout 和写盘链路均已实际打通，而不只是 import 成功。

smoke 成功后 formal `15787766` 已由 Slurm 自动释放；最先两个 task `0/1` 分别在 A100
`ga010/ga022` 启动。此时尚未读取、汇总或解释 formal outcome；必须等待 34 tasks、summary
和 independent verifier 全部完成。

前三次 smoke 均保留为基础设施 incident：前两次未创建 episode，第三次创建 GOAT 环境
但未执行动作；三次均未运行 34-scene formal array。不得把任何一次 incident 当作 SR，
也不得据此改变 population、方法、阈值、seed 或 arm order。

### 16.9 Sequential-Revisit formal 最终结果与 actionability 归因

第四链已全部完成：formal 34/34 tasks、summary `15787768`、independent verifier
`15787770` 均为 `COMPLETED 0:0`。34 raw JSON 和 34 SHA sidecars 齐全，verifier 从 raw
结果独立复算并给出 `verified=true`。

Confirmatory ITT：

- official GOAT：`4/34 = 11.76%`；
- role-free CEC：`4/34 = 11.76%`；
- paired `+0/-0`；risk difference `0 pp`；
- exact McNemar `p=1.0`；cluster CI `[0,0] pp`。

但 post-hoc actionability audit 证明这是退化 no-op，而不是一次有效干预后的性能 null：

- 5 次 certificate accept（target 4、pre-target 1）全部发生在 official
  `SUBTASK_STOP`；
- actionable non-STOP accept：0；NavDP plan：0；executed override：0；
- first-override episode：0；
- native/CEC 逐步动作与位姿 exact：34/34。

冻结协议要求 official STOP 原样执行，因此 certificate 获得可靠几何证据时已经没有 motion
控制窗口。此前的 `mechanistic_coverage_gate_passed=true` 只证明 observation-level
candidate/certificate constructibility，没有要求 control-level actionability，是本次协议的
真实设计缺口。34 scenes 已消费，禁止在其上降阈值、改 STOP contract 或重筛 population。

准确结论是：当前 strict certificate 在 GOAT sequential recurrence 上没有可执行覆盖；本次
结果不能支持 CEC 提升 GOAT，也不能反推 bearing 无效。GOAT 应作为 external limitation，
论文正向主证据仍来自 MP3D actual-online Revisit。

完整结果与解释：`MemNavData/GOAT_SEQUENTIAL_REVISIT_FORMAL_RESULT_20260815.md`。
