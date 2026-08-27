# MemNav / Directional Adapter 全量展开版（归档）

更新时间：2026-08-10 18:59（CST）

主工作树：`/home/asus/Research/Nav-graph-blind`

状态：**归档展开版**。当前简明 canonical 快照见 `STATUS_20260810.md`。

本文取代 `STATUS_20260809.md` 和更早的 `LATEST_TRAINING_RESULTS_20260807.md`
作为当前统一总账。旧文仍保留实验历史；凡与本文的确定性 R0、B0、X-NavDP R2、
O1、active-glance 或 5090 bearing×memory factorial 冲突，均以本文为准。

本文严格区分：

- **方法结果**：部署输入不含 Habitat goal oracle，完整闭环计 SR；
- **oracle 机制结果**：用于测上限、能力分解或接口兼容性，不能当方法 SR；
- **离线/observability 结果**：证明表示中有信息，不等于闭环增益；
- **development signal**：在已耗尽数据上的方向性结果，不能当最终确认；
- **负结果**：同样进入总账，禁止通过换口径消失。

阅读导航：若只需要当前决策，读 §0、§3、§4、§18 和 §21；若需要论文证据链，读
§5–§13；若需要复现/审计，读 §14、§16、§17 和 §20。

---

## 0. 一页结论

1. **当前唯一显著、oracle-free、闭环有效的方法结果仍是几何记忆。**
   Canonical deterministic R0 上，2-leg joint SR 从 `3/40` 提升到 `20/40`，
   配对 `+17/-0`，exact McNemar `p=1.5259e-5`，scene-cluster 95% CI
   `+27.5~+57.5 pp`。历史 HPC 独立运行也得到 `4/40 -> 19/40, +15/-0`。

2. **方向是 Novel-A 的强可恢复瓶颈，但目前仍是 oracle 机制结论。**
   独立 N=40 同机配对中，native `28/40`，periodic oracle yaw 与 oracle-bearing
   token 均为 `40/40`，配对 `+12/-0`，`p=4.88e-4`。它证明“方向给对时冻结
   NavDP 能执行”，没有证明部署时能得到该方向。

3. **B0 已把 Revisit memory 的 controller 接口压缩成 bearing + 固定局部半径。**
   Geometry 的原始 metric PointGoal 为 `20/26`，固定 `2.5 m`、只保留 bearing 后仍为
   `20/26`，配对 `+1/-1, p=1.0`。因此总体记忆增益不要求把估计距离暴露给
   controller；但不能声称距离对每条 episode 都无关。

4. **最新 5090 factorial 证明 Novel direction 与 Revisit memory 在同一条真实 joint
   rollout 中可组合。** 全 native 为 `8/40`，A 使用 oracle-bearing、B 使用 geometry
   memory 后为 `29/40`；直接配对 `+22/-1`，`p=5.72e-6`。更关键的是，在 bearing
   改写 A 路径后，memory 仍把 B 从 `13/40` 提升到 `29/40`，`+16/-0`，
   `p=3.05e-5`。但 `29/40` 仍包含 Habitat oracle bearing，不能写成部署方法 SR。

5. **X-NavDP 不是更强的全局 Revisit controller。** 完整 R2 上，mixed `20/26`、
   base pure PointGoal `20/26`、official X+MPC `21/26`。X 对 mixed 为
   `+2/-1, p=1.0`；共同成功样本上路径更短、SPL 更高，并提供真实 reverse capability，
   但产生一条新的 500-step side-rear loss。默认 controller 仍应是 mixed NavDP。

6. **3-leg 目前首先被第二个 Novel-B 卡住，不能用 `0/10 joint` 否定 long memory。**
   End-to-end 中 B|A 只有 `1/6`，使自然 C 分母只有 1。固定 A/B source prefix 的严格
   conditional-C 为 native `4/10`、direct geometry `8/10`、reverse graph `6/10`；
   direct 对 native `+4/-0, p=0.125`，是有希望但未显著的 N=10 signal。

7. **O1 首次证明 train-scene expert states 上存在单视角、goal-conditioned 的方向
   observability。** 部署主模型只输入一张当前 RGB + ImageGoal；31 scenes scene-OOF
   上，off-axis selected direction 相对 camera-forward 的 progress 为 `+0.218 m`，
   CI `[+0.150,+0.285]`，goal-swap NLL contrast 也显著为正。但 exact C8 bin 仅
   `17.74%`、circular error `85.5°`，这是一条弱但真实的 progress signal，不是高精度
   bearing solver，更不是闭环 SR。

8. **Raw image similarity 不能直接充当 Novel compass。** 四向 active-glance V1
   将 Novel-A 从 `31/40` 降到 `20/40`，配对 `+2/-13`，`p=0.00739`；margin 与 gated
   版本把干预和伤害降下来，但最终仍只有 `25/40`，没有超过 native。

9. **Learned router 的价值仍停留在相对排序。** Phase-B candidate AUC 约
   `0.95~0.96`，高于 DINO `0.91~0.92`；但 top-1 为 `25/34 vs 26/34`，OOF activation
   `84.5% < 87.3%`，GLP Stage 2 也未通过。跨场景绝对分数校准仍是主要失败面。

10. **项目的核心 open problem 已收敛为一个问题：**从部署可见的当前 RGB/历史与
    ImageGoal 中，产生可周期更新、带 confidence、能 abstain 的 Novel bearing。
    Memory 已经是 Revisit bearing source；O1 只通过了上游 observability gate，下一步
    必须验证 expert-state readout 能否迁移到 frozen-policy 自然/失败状态。

---

## 1. 任务、术语与统计分母

### 1.1 Novel / Revisit 的含义

本文的 Novel/Revisit 是相对**同一条 episode 的历史 memory**而言，不是训练/测试场景
是否见过。

2-leg：

```text
start -> Goal A（Novel）-> Goal B（Revisit，目标区域在 A 路径中出现过）
```

3-leg：

```text
start -> Goal A（Novel）
      -> Goal B（Novel detour）
      -> Goal C（Revisit A-era memory after detour）
```

因此：

- 2-leg joint 必须同时完成 A 和 B；
- `B|A` 的分母只包括 A 成功的 episode；
- 3-leg `C|AB` 只包括 A、B 都成功的 episode；
- `0/10 joint` 不能自动解释为 C memory `0/10`。

### 1.2 三种常见评测协议

| 协议 | 前缀 | 后续控制 | 能回答什么 | 不能回答什么 |
|---|---|---|---|---|
| replayed-prefix conditional | 专家/固定 source prefix | policy 控制目标 leg | 隔离某一段能力 | 不能算真实 joint SR |
| executed-prefix conditional | policy 实际跑前缀，成功后继续 | policy 控制后续 | 更接近系统，但分母条件化 | 不能忽略前缀失败 |
| end-to-end joint | 全部 leg 均由系统执行 | 全链路 | 最终完整成功率 | 单个低值不能定位故障段 |

### 1.3 当前关键数据集合

| 集合 | 用途 | 当前状态 |
|---|---|---|
| 40 train scenes | 训练、scene-grouped OOF | 可用，但不是最终泛化确认 |
| 10 development scenes | trainer/self-eval、Stage 2、P1 calibration | **已耗尽** |
| 20-scene closed-loop pool | 40 episode Novel/Revisit、bearing、factorial、失败解剖 | **重度开发/consumed** |
| 4 final-reserved scenes | 预留 | 当前原始 2-leg 目录无兼容 episode |
| 16 blind scenes | 最终 scene-generalization | **尚未打开** |
| 526 条 3-leg expert episodes | 3-leg train/eval pool | episode 数，不是 526 个独立 scene cluster |
| 535 条 2-leg expert episodes | 2-leg train/eval pool | 同上 |

所谓“526 pool”中，真正与 20-scene pool、训练场景均不重叠的统计单位只有约 16 个 scene
cluster。以后禁止用 episode 数伪装独立样本量。

### 1.4 当前 canonical R0 分母

- 20 scenes × 2 episodes = 40；
- deterministic R0 Novel-A 成功 `26/40`；
- 因而 Revisit-B conditional denominator 为 26；
- 14 条 A 失败在只重放 B 的 R2/B0 臂中固定为 implied joint failure；
- R0、R2、B0 共享 Goal-A trace、episode seed 和 deterministic plan seed。

### 1.5 不能混算的 Native-A 绝对值

| 运行 | Native-A | 用途 |
|---|---:|---|
| 历史 HPC memory run | 31/40 | 历史配对复现 |
| oracle direction paired run | 28/40 | Novel direction mechanism |
| deterministic R0 | 26/40 | R2/B0 controller/interface 底座 |
| 5090 factorial | 30/40 | 同运行 bearing×memory compatibility |
| active-glance worktree | 31/40 | active source negative control |

同 checkpoint 在不同机器/进程路径下会因 CUDA/扩散采样非确定性产生绝对 SR 漂移。
正式效应必须写成同机、同进程族、逐 episode 配对的 gain/loss；禁止把不同表中的
`SR_A` 与 `SR_B|A` 相乘。

---

## 2. 当前系统与能力分解

### 2.1 原生 NavDP 在 Novel 段做什么

ImageGoal 会条件化 diffusion trajectory generator：当前 RGB 与 goal RGB 进入 paired
ImageGoal encoder，生成局部轨迹候选。当前 critic 路径不读取 ImageGoal/PointGoal
embedding，主要在同一个 request 内按局部可行性/安全性选轨迹。

因此原生 Novel 探索依赖：

1. ImageGoal-conditioned imitation prior；
2. 当前 RGB-D 的局部可行空间；
3. goal-agnostic critic 的候选内安全选择。

它没有显式全局地图、frontier search 或长期方向变量。困难状态下，diffusion 候选可能
出现 heading mode collapse。

### 2.2 当前 geometry memory 做什么

```text
Goal image
   -> DINO 历史候选检索
   -> SIFT/RANSAC 几何验证
   -> 估计 verified anchor 的 local aux pose
   -> PointGoal token
   -> frozen mixed ImageGoal+PointGoal NavDP
```

它在当前 Habitat sensor/controller contract 下不使用 Habitat goal oracle，因此属于
oracle-free benchmark method。它尚不是“真实单目硬件已经部署”的同义词；内部仍可能
使用当前系统提供的深度、位姿链和几何信息。

### 2.3 当前目标架构

```text
                        geometry memory evidence（Revisit）
                                     |
current RGB/history + ImageGoal -----+---- Novel visual/history evidence
                                     |
                                     v
                       camera-relative C8 belief
                            + confidence R
                                     |
                         low confidence: abstain
                                     |
                         high confidence: bearing
                                     |
                           fixed local radius 2.5 m
                                     |
                                     v
                       existing mixed NavDP controller
```

约束：

- Novel/Revisit 不是两套 controller，而是统一 bearing belief 的两种 evidence source；
- adapter 只输出局部 bearing distribution + confidence，不输出完整地图；
- abstain 时逐位回退 native；
- direction 必须流式/周期更新；
- fixed radius 删除当前没有 aggregate 必要性的自由度；
- geometry 内部仍可使用 metric pose，只是不把距离暴露给 controller；
- X/MPC 不在默认路径，只保留为未来可能的 deep-rear recovery primitive。

---

## 3. 证据等级总表

### 3.1 A 级：N=40 严格闭环配对

| 结果 | 数字 | 类型 |
|---|---|---|
| Deterministic geometry memory R0 | joint `3/40 -> 20/40`，`+17/-0`，p=`1.53e-5` | 方法结果，consumed pool |
| 历史 HPC geometry memory | joint `4/40 -> 19/40`，`+15/-0`，p=`6.10e-5` | 方法复现，绝对 rollout 不同 |
| Novel-A oracle bearing | `28/40 -> 40/40`，`+12/-0`，p=`4.88e-4` | oracle mechanism |
| Top-K K=1 vs K=8 | `18/40 vs 18/40`，`+1/-1`，p=`1.0` | 干净 null |
| Active-glance V1 | `31/40 -> 20/40`，`+2/-13`，p=`0.00739` | oracle-free source 负结果 |
| Bearing×memory combined vs all native | `8/40 -> 29/40`，`+22/-1`，p=`5.72e-6` | 含 oracle 的 compatibility mechanism |

### 3.2 B 级：scene-OOF / 百级离线决策

| 结果 | 数字 | 边界 |
|---|---|---|
| Phase-B candidate ranking | AUC `~0.95-0.96` vs DINO `~0.91-0.92` | 相对排序有信息 |
| Phase-B top-1 | `25/34` vs DINO `26/34` | 没有转成更好 anchor |
| OOF activation | `84.5%` vs max-DINO `87.3%` | 仍未过基线 |
| Calibration | threshold migration loss `13.6 pp -> 6.4 pp` | 改善但未解决 |
| O1 single-view off-axis progress | `+0.218 m`, CI `[+.150,+.285]` | train-scene expert-state observability |

### 3.3 C 级：机制测量

- Native candidate heading resultant `R≈0.98-0.99`，显示方向 mode collapse；
- point-token transfer function：±60° 忠实，±90–105° 峰值，±150° 崩塌，
  165–195° 零输出；
- 每 plan 实际可执行转角约 22°；
- 走近后 cloud overlap 约从 0.049 增到 0.432，DINO AUC 从 0.735 增到 0.923；
- frontier candidate proposal 在 9 条 A failure 的 486 个状态中，421 个状态至少含一个
  ±30° 候选，但 selector 不可靠。

### 3.4 D 级：N<20 轶事/开发信号

- N=6 direction token：`1/6 -> 3/6`；
- N=5 executed-A bearing：`3/5 -> 4/5`；
- N=6 3-leg symmetry：`b2_rendered 1/6 -> b1_matched 3/6 -> b2_turned 4/6`；
- N=10 conditional-C：`4/10 -> 8/10`，p=`0.125`；
- N=9 X-NavDP/critic/diffusion score-field probes；
- N=5 goal-blind frontier：`4/5 vs 4/5`。

这些结果可以形成机制假设，不能单独承担论文主效果。

---

## 4. 最新 5090：Novel-A bearing × Revisit-B memory factorial

### 4.1 唯一问题

此前 oracle bearing 与 memory 来自不同运行，不能用 `SR_A × SR_B|A` 拼 joint。
Factorial 直接回答：

> Bearing 改变 A 的路径、到达姿态与沿途 memory 后，geometry memory 的 B 增益是否仍然
> 保留？两者能否在一次真实 2-leg rollout 中组合？

### 4.2 四臂与结果

20 scenes × 2 episodes：

| Arm | A SR | B given A | Joint | Mean A path | Mean B path |
|---|---:|---:|---:|---:|---:|
| `a_native__b_native` | 30/40 | 8/30 = 26.7% | 8/40 = 20.0% | 8.387 m | 9.004 m |
| `a_native__b_geometry` | 30/40 | 23/30 = 76.7% | 23/40 = 57.5% | 8.387 m | 4.343 m |
| `a_bearing__b_native` | 40/40 | 13/40 = 32.5% | 13/40 = 32.5% | 5.966 m | 12.870 m |
| `a_bearing__b_geometry` | 40/40 | 29/40 = 72.5% | 29/40 = 72.5% | 5.966 m | 6.663 m |

Bearing-on 是 Habitat oracle source：每 4 plans 刷新，并使用 development 上预先冻结的
episode-persistent ±30° bias；它不是 deployable source。

### 4.3 配对对比

#### Memory effect after native A

- `8/40 -> 23/40`；
- `+15/-0`；
- risk difference `+37.5 pp`；
- exact McNemar `p=6.1035e-5`；
- scene-cluster 95% CI `+22.5~+52.5 pp`。

#### Memory effect after bearing-modified A

- `13/40 -> 29/40`；
- `+16/-0`；
- risk difference `+40 pp`；
- `p=3.0518e-5`；
- CI `+25~+57.5 pp`。

这是 factorial 最重要的结果：bearing 改写 A prefix 后，memory 的零损失显著效应仍然
完整存在。

#### Bearing effect with geometry B

- `23/40 -> 29/40`；
- `+7/-1`；
- net `+6`，risk difference `+15 pp`；
- exact McNemar `p=0.0703125`；
- scene-cluster CI `+5~+25 pp`。

方向一致，cluster interval 为正，但 exact McNemar 未过 0.05。准确表述是“强正向 joint
compatibility signal”，不是单独显著的方法增益。

#### Bearing effect with native B

- `8/40 -> 13/40`；
- `+8/-3`；
- net `+5`，risk difference `+12.5 pp`；
- `p=0.2265625`；
- CI `-2.5~+25 pp`。

#### Complete combined arm vs all native

- `8/40 -> 29/40`；
- `+22/-1`；
- net `+21`，risk difference `+52.5 pp`；
- `p=5.7220e-6`；
- CI `+37.5~+67.5 pp`。

这是强组合机制结果，不是 deployable `72.5%` 方法结果。

### 4.4 独立运行时审计

在汇总后，使用与 producer/summarizer 独立的 auditor 对远端原始 CSV 与 plan payload
进行了完整检查：

- scene directories：20；
- episode keys：40；
- arm rows：160；
- native-A Leg-A identity pairs：40/40 exact；
- bearing-A Leg-A identity pairs：40/40 exact；
- 完整 `leg_a_plans`、A success/steps/path、seed、Goal-A/B hash、protocol/manifest/input
  hash 与 arm rotation 均通过；
- 因此相同 A treatment 下的 B treatment 差异可以作因果解释。

### 4.5 必须披露的协议偏差

1. 原 JSON 要求四臂共享长期存活 server pair；32 GB 5090 在第二臂出现累积 OOM，正式
   runner 改为每个 arm position 启动一对全新 server。A-prefix exact identity audit
   提供了较窄的 causal salvage，但不能声称逐字满足 persistent-process 条款。
2. Protocol JSON 写 `graph_spacing_m=1.25`；实际 hash-pinned runner 传
   `--graph_subgoal_spacing_m 0.0`，plan 回显 `graph_subgoal_enabled=false`。评测的是
   direct geometry aux-pose baseline，不是 reverse graph。

### 4.6 权威远端工件

```text
host: 5090
/home/cv/memnav_eval/Nav-bearing-next/.diagnostics/
  bearing_memory_factorial_formal_20260810/
```

- summary SHA256：`b07c410c17e0a0aea5d484a7248fc01dec7fe7ed3a8ea0b889af3e0ba72ad986`；
- protocol SHA256：`551201b6c00952de79d1d225ae0972da810415a63c77db1f141162ebdb1050f1`；
- frozen commit：`d5e2a9395a1cd026b323bfb2b027695237e1376b`；
- 所有 20 scenes 于 2026-08-10 17:36 CST 完成；总墙钟约 3 小时 57 分。

---

## 5. Revisit / Geometry memory 完整结果线

### 5.1 历史 HPC 20-scene 结果

| Endpoint | Native | Geometry |
|---|---:|---:|
| Novel-A | 31/40 | 31/40 |
| Revisit-B given A | 4/31 | 19/31 |
| Joint | 4/40 | 19/40 |

配对 joint `+15/-0`，McNemar `p=6.1035e-5`。这是第一份强方法结果，但缺少当前 R0 级别
的稠密 trace/seed 审计，因此 absolute count 不再作为 canonical。

### 5.2 Deterministic R0 canonical

| Endpoint | Native | Geometry |
|---|---:|---:|
| Novel-A | 26/40 = 65.0% | 26/40 = 65.0% |
| Revisit-B given A | 3/26 = 11.5% | 20/26 = 76.9% |
| Joint | 3/40 = 7.5% | 20/40 = 50.0% |
| Conditional-B mean SPL | 0.0384 | 0.6229 |
| Conditional-B mean final distance | 6.937 m | 1.785 m |

配对：

- `+17/-0`；
- risk difference `+42.5 pp`；
- exact McNemar `p=1.52587890625e-5`；
- 100,000 次 scene-cluster bootstrap CI `+27.5~+57.5 pp`；
- 20 scenes、40 episodes、training overlap 空；
- shared Goal-A trace 与 deterministic seed contract 全通过。

Router 运行统计：

- Revisit-B 激活 `21/26 = 80.8%`；
- Novel-A false-activation episodes `0/40`；
- selected candidate rank p50/p95/max 为 `1/2/2`；
- geometry verification latency mean/p50/p95 为 `6.53/0.304/26.65 ms`。

R0 的 6 条 geometry residual：router inactive 4 条，router active 2 条。因此只换 controller
的最大净上限约为 2/40；更大的杠杆仍在 activation/direction supply。

### 5.3 B0 fixed-radius bearing interface

B0 仍运行完整 geometry router，只在 controller 边界把非零 aux-pose 投影到固定 2.5 m：
该半径由 21 条 active episode 的 first-active radius 做 episode-balanced median
`2.513 m` 后事先四舍五入冻结，没有扫描 radius。

| Interface | B given A | Implied joint |
|---|---:|---:|
| Native ImageGoal | 3/26 | 3/40 |
| Metric geometry + mixed | 20/26 | 20/40 |
| Fixed 2.5 m bearing + mixed | 20/26 | 20/40 |

Fixed vs native：`+17/-0, p=1.5259e-5`。

Fixed vs metric：

- `+1/-1, p=1.0`；
- conditional CI `-11.1~+11.1 pp`；
- all-40 joint CI `-7.5~+7.5 pp`；
- retained metric successes `19/20`；
- 322 active plans 最大 radius error `8.88e-16 m`；
- 最大 bearing error `2.84e-14°`；
- 5 个 inactive controls 的 success/steps/path/final distance 完全一致。

互换的两条均是 deep-rear：

| Episode | Metric | Fixed | First raw request |
|---|---|---|---|
| `rPc6DW4iMge/episode_0000` | fail, 500 steps | success, 209 steps | 4.782 m, +166.99° |
| `ac26ZMwG7aT/episode_0000` | success, 314 steps | fail/stuck | 3.377 m, -155.41° |

不能用两条样本事后发明 radius/bearing selector。

### 5.4 旧 19/40 cohort 的真共视失败分解

对 8 条 B failures 用生成器 `_goal_world_pts` + `covis_frac` 重渲真共视：

| 类别 | n | 现象 | 所属修复层 |
|---|---:|---|---|
| 目标从未被 A 路径观察 | 1 | max covis 0.47 | 当前 memory 不可修 |
| 正例全在 anchor margin 外 | 2 | 可达正例为 0 | memory architecture |
| 正例未进 top-8 | 2 | DINO shortlist miss | retrieval |
| alias 通过并锁存 | 1 | 错 anchor latch | decision |
| 正例进池但 RANSAC 拒绝 | 2 | verifier false negative | decision |

决策层最多直接修 3 条；检索 2、memory architecture 2、不可修 1。旧的“floor filter
误删真匹配”来自距离+朝向代理错误，已撤回。

### 5.5 Top-K 候选链

K=1 与 K=8：

- `18/40 vs 18/40`；
- `+1/-1`；
- p=`1.0`；
- A 段逐条一致。

候选多样性/采样宽度不是当前主要瓶颈，禁止继续以“再多采几个 candidate”为主线。

### 5.6 Reverse graph

旧 2-leg development：direct `19/40`，graph `25/40`，`+6/-0`，sign-test
`p=0.03125`。但两臂 diffusion request 次数不同，后续 noise 未严格配对。

严格 conditional-C 中：direct `8/10`，graph `6/10`，graph 反而更差。节点切换、重复
定位和局部误差可能抵消短 subgoal 优势。因此：

- direct verified anchor 仍是默认；
- graph 只是结构假设；
- 5090 factorial 实际也没有启用 graph。

---

## 6. Novel direction 完整结果线

### 6.1 Heading mode collapse

在早期 N=6 困难 episode 中，64 个 native candidates 的 endpoint heading resultant
`R≈0.98-0.99`：看似有 64 个候选，实际几乎全朝同一方向。

Direction token 在 4 个塌缩状态中唤出 2 个真实进展模态，使 R 降到约 0.5。这个结果解释
失败形态，但因样本来自同一批 N=6，不是独立显著证据。

### 6.2 Novel-A oracle bearing N=40

| A arm | Success | vs native |
|---|---:|---:|
| Native | 28/40 | -- |
| Oracle periodic yaw | 40/40 | +12/-0 |
| Oracle bearing + token | 40/40 | +12/-0 |

- paired risk difference `+30 pp`；
- exact McNemar `p=0.00048828125`；
- scene-cluster CI `+15~+47.5 pp`；
- 12 gains 分布于 9 scenes；
- token 在 35/40 episode 激活，共 159 次，无 burst exhaustion。

正确表述：持续正确局部 shortest-path bearing 足以恢复这批数据上的全部观测到的 A
failures，且 token execution 能兑现信息。

错误表述：已有部署 Novel module、控制永远不是瓶颈、joint 天花板等于 100%、oracle arm
是方法 arm。

### 6.3 Bearing dose response（当前 N=38 正式读数）

| Dose | Result | Paired transfer |
|---|---:|---:|
| 每周期刷新 | 38/38 | +11/-0 |
| 每 4 周期刷新 | 38/38 | +11/-0 |
| 仅起点一次 | 26/38 | +2/-3 |
| 持续 ±30° bias | 38/38 | +11/-0 |
| 持续 ±60° bias | 25/38 | +6/-8 |

方向必须是可更新的 flow；start-only 不是可接受替代。N=38 是当前冻结读数，旧状态稿中
scene 11 的 N=40 汇总尚未形成新的 canonical report，不能擅自补写。

### 6.4 Point-token steerability transfer function

- request ±60°：大体忠实；
- request ±90–105°：实现转角达到 65–100° 峰值；
- request ±150°：开始崩塌；
- request 165–195°：16/16 extent=0；
- 每 8 帧 plan 实际转角约 22°。

已经排除“弦长/圆弧导致角度看起来减半”的误读：后半段位移方向约等于 chord，
path≈extent，轨迹近直线。Mixed token 是局部转向接口，不是全向控制器。

### 6.5 Goal-blind frontier 与 frontier coverage

小样本闭环：

- goal-blind frontier：`4/5 vs 4/5`，救一条毁一条；
- DINO-CLS frontier ranking：`3/6`，低于 goal-blind `4/6`。

但 proposal coverage 并不差。在 9 条 native-A failures 的 486 个状态中：

- candidate set 至少含一个 ±30° 候选：`421/486 = 86.6%`；
- fixed goal-blind top-1 within ±30°：约 `62.3%`；
- current heading within ±30°：约 `15.2%`。

因此主要缺口更像 goal-conditioned selection + takeover timing，而不是 frontier provider
完全没有方向候选。

### 6.6 NavDP critic 与 diffusion score-field

9 个 failure-enriched consumed states：

- critic 跨 8 request 选中 oracle ±30°：`0/9`；
- critic 最终 executed heading：`1/9`；
- native executed：`5/9`；
- execution ceiling：`8/9`。

源码上 critic 不接 goal embedding，只能在给定 request 内选可行 trajectory，不能跨方向
承担 ImageGoal relevance。

配对 diffusion denoising goal-contrast score-field：

- 命中 `1/9`；
- oracle-nearest request 平均排名 4.56；
- 随机期望 4.5。

冻结 diffusion 的等权 denoising error 不能直接当方向 likelihood。

### 6.7 现成 expert bearing 标签为何不能直接训练

73 episode / 1434 decision states 中：

- 97.5% relative bearing 在 ±10°；
- 100% 在 ±30°。

生成器让 expert camera 基本朝向运动方向，直接监督只会学成“永远向前”，覆盖不到 native
徘徊/off-policy states。需要 yaw-balanced privileged supervision 与真实 policy-state
transfer test。

---

## 7. Active-glance 三个版本

### 7.1 机制

每个 plan 观察当前 yaw 与另外三个相隔 90° 的 yaw，用只读
`imagegoal_similarity/current_goal_cos` 最大值选择朝向，再交给 native NavDP。扫描/朝向
切换在模拟中不支付完整物理旋转成本，因此是 raw similarity source 的乐观 ceiling，
不是严格硬件部署。

### 7.2 结果

| Version | Active SR | Native | Gains/Losses | p | Mean glances | Mean turns | Turn sum/ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 no margin | 20/40 = 50.0% | 31/40 | +2/-13 | 0.00739 | 32.25 | 14.40 | 1948.5° |
| V2 margin | 24/40 = 60.0% | 31/40 | +1/-8 | 0.03906 | 29.85 | 3.275 | 398.25° |
| V3 margin+gate | 25/40 = 62.5% | 31/40 | +1/-7 | 0.07031 | 7.225 | 2.95 | 362.25° |

Scene-cluster intervals：

- V1：`-47.5~-7.5 pp`；
- V2：`-32.5~-2.5 pp`；
- V3：`-30.0~-2.5 pp`。

每次修复都按预测方向减少 intervention 和 harm，但没有一个版本超过 native。V2/V3 又是
读取此前 outcomes 后在同一 consumed pool 上运行的 diagnostic，p 值不具独立确认性。

V1 的 1290 plans 全部触发 glance，576 次选择非零转向；全体累计转角 77,940°，约
216.5 圈，而最佳 view 相对当前 view 的平均 cosine 优势只有 0.0143。机制把微弱噪声级
相似度差放大成反复大角度状态改变。

冻结结论：停止 raw-similarity unconditional active glance；不继续在 consumed pool 上
sweep margin、trigger、view count 或 scan schedule。

---

## 8. CGC failure audit 与 O1 observability

### 8.1 原始 CGC job 不是模型负结果

HPC job `15527318` 在 1 分 51 秒后 exit 2，发生在 teacher build，未进入 feature
extraction/training。

全图审计：

- factual reachable：160/160；
- swapped-goal reachable：124/160；
- 36 invalid groups 分布于 9 scenes，原因是 paired goals 位于不同 navmesh islands；
- 另 1 group 的 C8 `try_step` 投影到不同楼层；
- 合法 causal pairs：123 physical groups / 31 scenes / 246 rows；
- 合法标签不退化：goal swap 会改变 best C8 bin。

因此 failure 是 teacher domain connectivity 定义不成立，不是“CGC 学不会”。不能给
unreachable goal 人造 penalty，也不能只删单行后继续宣称原协议成立。

### 8.2 O1 的唯一问题

> 冻结 NavDP ImageGoal representation，在部署时只看一个当前 camera view 时，是否包含
> 跨场景可线性读出的相对 C8 direction/progress signal？

### 8.3 输入与监督

- 31 个合法 train scenes；
- 123 physical groups；
- 246 goal-conditioned rows；
- 984 rendered RGB views；
- 每个 physical state 的 8 yaw 仅用于 yaw-balanced supervision 与信息对照；
- label 为 1 m geodesic progress ring；
- Habitat geometry 只生成 teacher，不进入模型输入；
- same-state swapped goal 为 goal-conditioning control；
- development、20-scene consumed、final-reserved、blind 均未读取。

### 8.4 部署主模型

```text
one current RGB + ImageGoal
        -> frozen NavDP paired ImageGoal encoder
        -> mean-pooled 384-D feature
        -> LayerNorm(no affine) + Linear(384,8)
        -> camera-relative C8 logits
```

训练：5 scene folds × seeds `11/29/47` × 300 epochs，AdamW，lr `3e-4`，weight decay
`1e-4`，batch 32。结果读取后禁止追加 MLP、Transformer、LoRA、spatial head、温度或
encoder fine-tuning。

### 8.5 Formal O1 结果

H100 smoke `15561233`：1 分 12 秒，工程通过。

H100 formal `15561742`：9 分 26 秒，`COMPLETED 0:0`。

Primary single-view：

| Cohort/metric | Result |
|---|---:|
| All starts exact-bin | 21.37% |
| All starts circular error | 81.2° |
| All starts selected vs camera-forward progress | +0.126 m，CI `[+.0527,+.1943]` |
| Off-axis exact-bin | 17.74% |
| Off-axis circular error | 85.5° |
| Off-axis selected vs camera-forward progress | +0.218 m，CI `[+.150,+.285]` |
| Off-axis gains/losses beyond ±0.25 m | 184/118 |
| Positive/negative mean-progress scenes | 26/5 |
| Goal-swap NLL increase | +0.174，CI `[+.0895,+.276]` |
| Same-state factual goal pair changes best bin | 31/61 = 50.8% |
| Changes best bin by at least 90° | 22/61 = 36.1% |

五个预注册 gate 全部通过，冻结决定：

```text
go_single_view_global_to_train_scene_frozen_policy_state_gate
```

### 8.6 View-dose 信息对照

| Accumulated views | Exact-bin | Off-axis progress | Gains/Losses |
|---:|---:|---:|---:|
| 1 | 21.37% | +0.218 m | 184/118 |
| 2 | 22.78% | +0.277 m | 200/111 |
| 4 | 27.82% | +0.414 m | 234/85 |
| 8 | 45.16% | +0.619 m | 271/75 |

Full-ring shared linear information control exact-bin 约 46.77%、circular error 47.9°。这些是
信息上限，不能用于事后选择部署 view dose。

### 8.7 O1 的关键限制

- Primary 是 3-leg expert manifest 中 kind=`novel` 的 `goal_b_t0`；
- 它是 Novel-B，不是 Revisit-C；
- 它不是 20-scene Novel-A failure；
- expert state 不是 frozen policy 自己走偏后的状态；
- pass 只授权 train-scene frozen-policy state-transfer gate；
- 不授权 development/blind、closed-loop SR 或论文方法 claim。

---

## 9. O1 P1 frozen-policy state-transfer（下一步，尚无结果）

### 9.1 唯一问题

当 frozen native NavDP 离开 expert trajectory，进入自己的自然 rollout states 后，冻结
O1 readout 选择的一米方向，是否比 NavDP 自己实际选择并准备执行的 trajectory heading
带来更高 geodesic progress？

### 9.2 冻结范围

- 31 O1 train scenes；
- 61 factual Novel-B `goal_b_t0` episodes；
- frozen native NavDP only，O1 不参与 state collection；
- max 500 execution steps，commitment horizon 8；
- sampled plan indices：`0,1,2,4,8,16,32,48,60`；
- primary natural indices：`1,2,4,8,16,32,48,60`；
- O1 ensemble 只允许使用对应 held-out fold 的 3 个 seed checkpoints；
- confidence threshold 固定为 expert OOF median resultant：
  `R >= 0.3464830160970278`；
- selection 不得读取 teacher、未来 success/failure、geodesic distance 或 critic score。

### 9.3 Primary 对比

```text
delta_progress = progress[O1 C8 argmax]
               - progress[native executable trajectory nearest C8 bin]
```

关键 cohort：all natural、native opportunity、native sufficient、native-success episode、
native-failure episode、confidence-selected。

只有 representation transfer、goal swap、selected safety、native-success non-harm 与
native-failure opportunity 全部通过，才授权 train-scene selective Novel-B closed loop。

### 9.4 当前状态

- protocol 已在 target-state outcome 前冻结；
- input authority、fold/checkpoint SHA、rollout backend 与 collector 已实现/在实现；
- 尚未提交正式 P1 job；
- 尚无 policy-state 或 closed-loop 结果；
- 即使 P1 pass，也不能直接外推 Novel-A，Novel-A 需独立冻结协议。

---

## 10. X-NavDP 完整审计与结果

### 10.1 Checkpoint/source attribution

- official commit：`878740a2011856d0e3782dd6ccd880fd2eccd70f`；
- post-train checkpoint SHA：
  `267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24`；
- 与原 NavDP 共享的 1062 tensors 全部 exact equal；
- PointGoal eval model 1329 tensors 对 checkpoint missing=0、shape mismatch=0；
- checkpoint 额外 357 tensors 来自该 eval 类未构造的 ImageGoal/pixel/旧 critic 模块。

因此 X 的公开 post-training 主要作用于 PointGoal actor、twin-Q、embodiment 与 signed
control，不应冒充 ImageGoal post-training。

### 10.2 N=9 direction execution probe

同一 RGB-D、seed、8 fixed directions、每方向 8 candidates：

- mixed-token oracle-nearest request：8/9；
- X post actor：7/9；
- base pure PointGoal：9/9；
- X vs mixed `+1/-2, p=1.0`；
- post vs base `+0/-2, p=0.5`；
- X cross-request Q oracle hit：4/9。

X post actor 新增 `-180°` 模态：base 0/9 -> post 9/9，但 `±90°` fidelity 明显下降。
Q 不输入 ImageGoal，不能解释为 Novel direction source。

### 10.3 Rear short-horizon 与 G1b online-router gate

最初自然 rear conditioning S1 failed；将 PointGoal 半径预先限制为 2 m 后，N=8 short
horizon signed recovery 相对 forward：8 wins/0 losses，median progress difference
约 +2.01 m；相对 rotate-first：7 wins/0 losses/1 tie，median 约 +1.10 m。

这证明 bounded signed executor 的 rear capability，但不是 full-episode SR。

随后 G1b 在 5 个 consumed scene cluster 的 10 条固定 episode 上重放同一 Goal-A
trace；其中 7 条到达 A：

| Controller | B given A | Mean SPL | Mean B path | Mean B steps |
|---|---:|---:|---:|---:|
| Existing memory mixed | 6/7 | 0.7330 | 5.361 m | 153.6 |
| X unbounded, forward-only | 6/7 | 0.7982 | 5.065 m | 148.4 |
| X bounded, forward-only | 6/7 | 0.7617 | 5.252 m | 159.0 |
| X bounded, signed | 6/7 | 0.8571 | 4.225 m | 116.7 |

Bounded-signed 相对 mixed 的平均 SPL `+0.1241`、路径 `-1.136 m`、步数 `-36.9`，
且执行 244 个 reverse frames，没有 blocked translation/non-navigable acceptance。但四臂
成功均为 `6/7`；唯一共同 failure 的 router 从未激活，不存在 controller 可修复
机会。因此 G1b 只证明成功轨迹的效率与 signed-recovery mechanism，不证明 SR 增益。

此时本机尚无 acados/casadi，bounded-signed 使用符合 signed-velocity contract 的
pure-pursuit surrogate。它促使后续将官方 MPC 真实移植，但本身不能冒充 official
executor result。

### 10.4 D1：历史 failure opportunity 不稳定

一次 post-hoc D1 从历史 20-scene artifact 中选出 5 条“A 成功、B 失败、router
active 且首个 bearing 在后方”的 episode。在 contemporary deterministic protocol 下：

- eligible controller failures：`0/5`；
- 3 条仍到达 A 的轨迹现在都完成 B；
- 另 2 条现在 A 就失败，根本不进入 Revisit-B；
- 因此 X arms 没有启动。

可归因的结论不是 X 成功或失败，而是旧 artifact 缺少当前级别的 per-plan seed
与稠密 Goal-A trace，历史 failure 不能冒充 contemporary paired opportunity。这个 null
直接促成 deterministic R0，重新建立了 `26/40` A 分母与 2 条稳定 router-active
residual failures。

### 10.5 R1：定向 residual opportunity gate

在 R0 事先固定的 2 条 router-active failures 上：

| Episode | Mixed | Base pure PointGoal | Official X + MPC |
|---|---:|---:|---:|
| `pLe4wQe7qrG/episode_0001` | fail, 206 steps | fail, 206 steps | success, 100 steps |
| `rPc6DW4iMge/episode_0000` | fail, 500 steps | success, 196 steps | success, 389 steps |

- official X rescues：`2/2`；
- base pure PointGoal rescues：`1/2`；
- 首个 active router state 的 step、anchor、aux-pose 与 bearing 三臂 exact；
- 两条首 bearing 为 `-157.585°` 与 `+166.990°`；
- X/MPC 全部 solve status 为 0，无 blocked/non-navigable 接受。

R1 证明存在真实 deep-rear controller opportunity，并授权 complete R2；但它读取 R0
failures 后条件选样，`2/2` 不是 unbiased SR estimate。而且第二条 base 也救活，不能
把两条都归因于 X post-training。

### 10.6 Official MPC port（E1/E2）

本机最初缺少 acados/casadi，后来移植兼容 acados 0.5.1 + casadi 3.7.0，使用发布的
BatchMPC controller：

- N=9 static fidelity：9/9 within 30°；
- mean rollout request error 1.37°；
- solver status 全零；
- prediction bridge error < `2e-9 m`；
- one-case Habitat E2 official/signed 两臂均成功；
- signed velocity、bounded command、navmesh safety contract 均通过。

E1 在 9 个固定 static states 上还验证了 candidate/Q replay `9/9` exact、前 8 个
controls 为 `72/72` 负速度，acados 预测与独立 unicycle integration 最大误差
`1.70e-9 m`。E2 的一条 Habitat paired smoke 中，signed surrogate 与 official MPC
均成功，official 的 5/5 solves 、40/40 controls 均通过安全契约。

所以后续 X 负结果不是因为仍在使用错误 pure-pursuit 模拟器。

### 10.7 Complete R2

| Controller | B given A | vs mixed | Implied joint |
|---|---:|---:|---:|
| Existing mixed ImageGoal+PointGoal | 20/26 | -- | 20/40 |
| Base pure PointGoal | 20/26 | +1/-1，p=1.0 | 20/40 |
| Official X + MPC | 21/26 | +2/-1，p=1.0 | 21/40 |

Official X CI `-8.0~+17.4 pp`，没有通过零损失门。

两个 gains 均在 `|bearing|>=150°` deep-rear；loss 在 `-115.55°` side-rear，跑满
500 steps，并包含 483 reverse-control frames。

19 条共同成功中，X 相对 mixed：

- mean path `-1.57 m`；
- mean SPL `+0.136`。

这些是 success-conditioned efficiency，不能转换成 SR claim。当前默认仍是 mixed；
不能在 consumed set 上事后部署 `|bearing|>=150°` selector。

X-NavDP README 中类似 `61.20% -> 84.28%` 的数字属于它自己的 Isaac-Lab
PointGoal/multi-embodiment benchmark。本项目测的是 MP3D ImageGoal-Revisit 中“几何
memory 已经给出 PointGoal 后由谁执行”；R2 既不复现也不反驳官方数字，只能说在
当前任务分布和接口下没有观察到显著的全局 SR 优势。

---

## 11. 3-leg / long-memory 完整状态

### 11.1 Generator 与 benchmark 修复

已完成：

- B distance 上界，使 A/B 距离约束更对称；
- Novel-B goal image 按 expert 实际到达朝向重锚；
- 重算 covis；
- camera-facing yaw bug 修复（旧实现约偏 90°）；
- clearance-aware near-reversal turn；
- 每帧 turn/navmesh safety gate；
- 526 条 3-leg + 535 条 2-leg camera-facing audit 全部干净。

### 11.2 为什么 A/B 不能当成同难度两次试验

旧 10 条 expert 3-leg：

| | A leg | B leg |
|---|---:|---:|
| mean frames | 141.6 | 274.9 |
| median | 130 | 201 |
| range | 82–207 | 112–871 |

B 平均约 A 的 1.94 倍，并且曾有：距离无上界、max covis<0.1、goal image terminal yaw
不同、从 on-policy A arrival 状态出发等额外困难。因此 `B|A=1/6` 不能直接写成“NavDP
第二次 goal switch 必然失败”。

Carry FIFO 与 hard reset 在小样本中均为 4/5，只是 gain/loss 互换，不能用简单 FIFO
污染解释。Metric subgoal upper bound 在相同 post-A states 上从 native 3/5 到 5/5，
更支持高层 direction 缺失。

### 11.3 End-to-end 10-scene

| Stage | Native | Geometry router |
|---|---:|---:|
| A Novel | 6/10 | 6/10 |
| B Novel given A | 1/6 | 1/6 |
| C Revisit given A/B | 0/1 | 0/1 |
| Joint | 0/10 | 0/10 |

它首先揭示连续 Novel exploration/困难 B distribution；自然 C denominator=1，无法估计
long-memory SR。

### 11.4 Conditional-C five-arm 与严格 direct/graph

第一份 five-arm：native 4/10、geometry top-1 7/10、temporal top-K 7/10、oracle anchor
8/10、oracle metric point 9/10。

后续严格 shared-prefix/per-request-seed：

| C controller | SR | mean SPL | mean final distance |
|---|---:|---:|---:|
| Native | 4/10 | 0.2049 | 6.174 m |
| Direct geometry gap16 | 8/10 | 0.5300 | 2.875 m |
| Graph gap16 | 6/10 | 0.3325 | 4.518 m |
| Oracle-anchor direct | 7/10 | 0.5502 | 3.014 m |
| Oracle-anchor graph | 5/10 | 0.3627 | 4.450 m |
| Oracle point | 7/10 | 0.5657 | 2.911 m |

Native→direct `+4/-0, p=0.125`；direct→graph 1 gain/3 losses，p=0.625。只能说 direct
memory 有强方向性信号，不能说 graph 更好或 long-memory 已显著解决。

### 11.5 Novel-B upper bound / symmetry N=5-6

同一 post-A state：

| Arm | B given A |
|---|---:|
| Native ImageGoal | 3/5 |
| 1.25 m geodesic subgoal | 5/5 |
| Exact final metric point | 5/5 |

短 geodesic subgoal 还提高成功路径效率，说明冻结 NavDP 是合格 local controller；困难 B
主要缺稳定高层 direction/target。

四不对称 N=6：`b2_rendered 1/6 -> b1_matched 3/6 -> b2_turned 4/6`，同一小样本，
不显著，只作机制线索。

---

## 12. Learned memory router / GLP / patch 全量结论

### 12.1 Phase-B 数据与 checkpoint

- frozen exact LingBot/DINO inputs：49,373；
- causal teacher：600 sessions = 480 train + 120 development；
- emitted candidate rows：17,845；
- counterfactual rows：8,923；
- 首个正式 checkpoint SHA：
  `1232a426458cedf36869304116a2dd5c779bbcdaca587f76abd5ed3572164f2c`；
- 3 seeds，20-D v3 feature；
- train/dev artifact SHA 与 source contract 已审计。

Session 结构：

| Teacher session class | Count |
|---|---:|
| Shortlist 内有 positive | 189/600 = 31.5% |
| Strict no-match | 345/600 = 57.5% |
| Ambiguous | 66/600 = 11.0% |

在 shortlist 真有 positive 的条件下，recall@1/4/8/16/32 分别为
`76.19/88.36/93.12/96.83/100%`。这同时说明候选集有上限、top-1 仍可改进，
而且 strict no-match 过半，显式 dustbin/abstention 不是可选项。

Factual Revisit-C 的 positive coverage 为 train `75/80=93.75%`、development
`19/20=95%`；Factual Novel-B 只有 train `24/160=15%`、development `3/40=7.5%`。
因而 memory candidate 对 Revisit 常存在、对 Novel 通常应 no-match；用一个平均 scalar
gate 同时承担两种语义会天然混淆。

### 12.2 Candidate ranking 有信息

Consumed development：

- pooled strict candidate AUC：Phase-B `0.9605` vs DINO `0.9208`；
- 另一个独立协议约 `0.9535` vs `0.9103`；
- within-session micro pair AUC：`0.9441` vs `0.8191`；
- scene-bootstrap delta lower95 > 0。

因此几何/causal features 中确实有相对排序信息。

### 12.3 但 top-1 与 activation 没提升

- global-positive session top-1：Phase-B `25/34`，DINO `26/34`；
- Phase-B wins/losses：5/6；
- candidate shortlist 本身漏 2 个正例 session；
- direct Phase-B joint localization：72.7%；
- GLP posterior：70.9%；
- max-DINO：87.3%；
- OOF calibrated activation：84.5%，仍低于 87.3%。

### 12.4 Calibration

- in-sample threshold：0.171；
- OOF threshold：0.553；
- development oracle threshold：0.830；
- threshold transfer loss：13.64 pp -> 6.36 pp；
- OOF/isotonic development accuracy：84.5%；
- oracle ceiling：90.9%；
- isotonic 无额外改善；
- negative mean score：in-sample 0.011，OOF 0.065，development 0.212。

排序是相对判断，尺度可消；activation 是绝对判断，跨场景尺度漂移致命。

### 12.5 GLP

- Stage 0：`goal_posterior.py`、carving、frontier lineage、goal-switch fail-closed 与 18 项
  property tests 完成；
- Stage 1：DINO 单证据下与 max-DINO 打平，88.57%，框架无害但无新信息；
- Stage 2：GLP 70.9%、Phase-B 72.7%、max-DINO 87.3%，未通过；
- ψ_frontier 与完整 goal-behind-frontier 方法尚未成为闭环方法。

### 12.6 RANSAC Simpson 悖论与 patch pooled trap

44 个真实验证 anchors（15 positives）分层：

| Channel | Pooled AUC | Episode-internal AUC |
|---|---:|---:|
| CLS cosine | 0.628 | 0.333 |
| RANSAC pass | 0.375 | 0.405 |
| Patch affine residual q90 | 0.777 | 0.429 |
| Patch mutual match fraction | 0.697 | 0.143 |
| Patch best match q90 | 0.575 | 0.048 |

Episode-internal 只有 21 positive-negative pairs，不能证明通道反向有害；准确结论是没有
证据表明它们能在同一 episode alias 中正向判别。Pooled patch 优势主要来自 episode
难度差异。

RANSAC 按 episode outcome 分层后显示，pass/fail 更多反映整个 episode 难度，而不是 anchor
正确性；在最需要 memory 的困难 episode 中，它反而频繁关闭 memory。

### 12.7 Learned router 五次尝试的统一定位

| 尝试 | 结果 | 当前定位 |
|---|---|---|
| 旧 flowgate/gatecurr | gate acc 提升，top-1/action/pose 未动 | 激活头未转成行为 |
| DINO-only set model | 67.65% < raw DINO 82.35% | feature/训练不足 |
| Phase-B session decision | calibrated 84.5% < 87.3% | calibration failure |
| Hybrid P0 learned rerank | 在 floor→RANSAC→first-pass 架构中无决定权 | 插入位置错误 |
| Patch/temporal verifier | offline 96.3%，未有闭环优势 | pooled shortcut |

结论：学习头目前只能负责“候选排序可能有信息”，不能负责“是否激活”，更没有方法 SR
增益。

---

## 13. 历史训练模型与早期控制实验

| 模型 | 已完成结果 | 最终定位 |
|---|---|---|
| `flowgate2600` | 旧 shared baseline | GateCurr 受控起点，不再主推 |
| `gatecurr600` | 两场景 conditional Revisit `6/20 -> 10/20`；gate acc 约 69→81% | 历史方向性证据；top-1/action/pose 未同步 |
| `residualgate1000` | Novel `2/10 -> 4/10`；selector 对齐后 7/10 | 证明 visual branch 不应关闭，但低于 native 9/10 |
| `novelgs_res1000_early40_w025` | GoalSwap gap `-0.000442`，目标 +0.05 | objective 接通但 goal-conditioning 未学成 |
| DINO-only NLSR | train smoke overfit；dev 67.65% vs raw 82.35% | 工程可学，不授权 checkpoint |
| Geometry router | 两次 N=40 显著配对增益 | 当前默认 memory baseline |

下面保留历史细节，目的是防止旧数字在后续又被当成 current baseline。

### 13.1 GateCurr：门控改善，但排序、动作与 pose 未同步改善

固定离线对照：

| Metric | `flowgate2600` | `gatecurr600` |
|---|---:|---:|
| Retrieval set loss | 0.204852 | 0.134628 |
| Retrieval top-1 | 89.58% | 87.50% |
| Gate accuracy | 69.44% | 80.56% |
| Revisit gate accuracy | 54.17% | 72.92% |
| Action epsilon-MSE | 0.086110 | 0.087323 |
| Aux x-y MSE | 4.376839 | 4.577455 |
| Revisit position error | 1.5487 m | 1.6070 m |
| Camera rotation error | 13.617° | 13.074° |

五个 unseen scenes / 10 条 replay-A conditional Revisit-B 上，SR `2/10 -> 6/10`、
SPL `0.105 -> 0.395`、mean final distance `2.998 -> 1.882 m`，McNemar
`p=0.21875`。两个 scenes / 20 条上，SR `6/20 -> 10/20`、`+4/-0`，SPL
`0.1371 -> 0.3583`。这证明 continuation 能改变 memory-to-action 闭环，但场景
数太少，且 continuation 同时更新 retrieval/projection/decoder，不能把收益只归因于
gate scalar。

### 13.2 Novel 负迁移、ResidualGate 与 selector 时域错配

五个 scene-disjoint scenes / 10 条 start-to-A 的历史对照：

| Method | SR | SPL | Final distance | Path |
|---|---:|---:|---:|---:|
| Official native NavDP | 9/10 | 0.896 | 2.120 m | 6.000 m |
| `flowgate2600`, forced Novel | 3/10 | 0.300 | 3.475 m | 7.811 m |
| `gatecurr600`, forced Novel | 2/10 | 0.065 | 3.974 m | 13.905 m |
| `residualgate1000`, 24-point selector | 4/10 | 0.374 | 2.330 m | 10.665 m |
| Residual, executed-prefix selector | 7/10 | 0.626 | 2.553 m | 6.303 m |

Residual 把 visual weight 从 `1-gate` 改为恒等于 1，证明不应因 memory gate 关闭
ImageGoal branch。Selector 原本给 24 个 waypoints 全部打分，而闭环每 8 frames 只执行
约前 2 个；把风险打分对齐 executed prefix 后 `4/10 -> 7/10`。这是 N=10
机制结果，不授权把 2 硬编码为通用规则。

Goal-image 因果检查还显示旧 shared MemNav 对目标极不敏感：同 state/history/noise 只交换
goal image，flowgate 的 goal-swap output RMS 只有 seed-change RMS 的
`0.13%~0.14%`，residualgate 为 `0.25%~3.16%`，原生 NavDP 约 `176.8%`。旧 MemNav
也不是“原生 NavDP 外挂 memory”：它替换了 encoder/decoder/critic，没有 exact fallback。

### 13.3 GoalSwap 训练负结果

`novelgs_res1000_early40_w025` 约训练 1112 steps / 1.77 epochs，目标是让正确 goal
相对 same-scene 错 goal 的 denoising gap 至少 `+0.05`。实际：

| Metric | Result |
|---|---:|
| 107 logged gaps mean | -0.000442 |
| Last-20 gap mean | -0.001780 |
| Last-20 margin-loss mean | 0.051780 |
| Target gap | +0.050000 |

Objective 链路已接通，但错 goal 的误差没有高于正确 goal；这不是 Novel 修复成功。

### 13.4 Patch/temporal 的历史离线能力

旧 pair-verifier development：

| Feature | Accuracy | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|
| DINO cosine | 84.4% | 0.950 | 0.917 | 0.135 |
| Patch | 95.4% | 0.990 | 0.971 | 0.041 |
| Patch + temporal | 96.3% | 0.991 | 0.973 | 0.032 |

但 33 个 positive sessions 中三者 top-1 都是 `24/33`；它首先改善的是 verifier/rank。
后续 temporal-NMS top-32 task-aligned development 中，raw DINO / pointwise /
listwise top-1 为 `24/35` / `29/35` / `30/35`，MRR 为 0.766 / 0.868 / 0.895。
这些是 consumed development 结构信号，从未转成新闭环 SR；后来的 episode-internal
分层又显示 pooled 优势可由 episode 难度 shortcut 产生。

### 13.5 早期 geometry/router 与 terminal U-turn 工程结果

在 5 scenes / 10 episodes 的早期 corrected geometry-router 运行中，A 三臂均为
`9/10`，B|A 的 geometry / advantage-hard-gate / native 为 `8/9` / `7/9` /
`3/9`，joint 为 `8/10` / `7/10` / `3/10`。相同 geometry routing 换回
`gatecurr600` direct controller 也是 `8/9`，所以不能把收益归因于 PointGoal
controller；主要来自长期记忆几何和 fail-closed routing。该小样本后来已被两次
N=40 geometry 结果取代。

必须保留一条撤回：旧 `HYBRID_POSE_NAVDP_EXPERIMENT_20260804.md` 的 B `9/9`、
SPL 0.830 不能当 `gatecurr600` 成绩。当时 server `chdir` 后相对 checkpoint 路径
失效，loader 又静默返回随机初始化 heads。后续已改为路径绝对化、缺文件 fail-fast、
trainable missing=0 强审计。

另一条早期 terminal U-turn 机制线在 3 scenes / 30 replay-A episodes 中：B 导航先到达
`13/30`；对这 13 条，forward-only terminal path 完成 `12/13`，最终位姿达标
`11/13`；12 条完成动作的 yaw error median `161.57° -> 2.86°`，直接图像
cosine mean `0.8725 -> 0.9393`。这证明近终点回转/视角对齐可执行，但交接当时依赖
GT distance，且没有新的导航 SR 分母，所以只是 terminal mechanism/engineering 成果。

Learned reliability 蒸馏在 4-train -> 5-unseen scenes、22,267 image pairs 的严格零误触
阈值下 100% 回退 geometry，没有进入 live policy。可复用的工程收益是 same
goal-anchor geometry cache：平均复验延迟 `24.283 ms -> 0.068 ms`，判定不变。

### 13.6 共享训练 lineage 的中间读数与未闭合账项

历史 replay-A `SR_B_given_A` 中间 checkpoint：

| Checkpoint | Unseen SR/SPL | Seen SR/SPL | Seen-Unseen gap |
|---|---:|---:|---:|
| `gs0_sym-5700` | 71% / 0.439 | 56% / 0.307 | -15 pp |
| `gs25_sym-6100` | 70% / 0.413 | 56% / 0.305 | -14 pp |
| `gs25_vscale-6200` | 51% / 0.262 | 51% / 0.258 | 0 pp |

GoalSwap 没超过 `gs0`；`vscale` 的零 gap 是 seen/unseen 一起降到约 51%，不是更好
泛化。2026-08-07 曾记录 final `6100/6270/6270` 的 2-leg 和 3-leg jobs 在运行，
但当前本地证据树中没有完整权威 pooled report。因此不能把当时 partial summaries
补写成 final result；这是一笔明确的“无可审计最终结论”，而不是默认成功或失败。

明确停止继续长训旧 shared decoder。失败主要不是 action epsilon-MSE，而是方向、候选覆盖、
alias 与跨场景校准。

---

## 14. 数据与工程审计成果

### 14.1 Generator / camera-facing

- genuine multi-stop 2-leg/3-leg generator 完成；
- 3-leg B max-distance、goal-yaw、covis 与 terminal semantics 修复；
- camera-facing audit 具有判别力：修复前平均约 71°，修复后约 0.3°；
- 526 条 3-leg + 535 条 2-leg 全池通过；
- `arrive=True` 曾被发现是死参数，已避免把无效修复写成结果；
- trajectory safety、navmesh segment、turn-rate 与 floor confinement 均进入 fail-closed。

### 14.2 Phase-B 链路修复的六个真实问题

1. `session_max_covis` 错取自 DINO shortlist，而非 teacher 全集；
2. 修复标签后 SQLite checkpoint 未同步；
3. `negative_threshold` 实际为 0.2，而非先前误记的 0.1；
4. audit job 未挂载 squashfs overlay，检查从未真正执行；
5. `YmJkqBEsHnH` 3-leg 原始数据缺失，字节级校验又过严；
6. trainer 引用了不存在的测试模块，旧训练路径实际没有成功执行。

修复保留 preRepair artifact、source diff SHA 与 fail-closed preflight，避免“修了但没接上”。

### 14.3 Baseline fairness

自动 trainer 曾用固定 0.5 threshold 评估集中在 0.9 左右的 DINO cosine，把 DINO 压到
21.7%，制造“79.2% vs 21.7%”假优势。公平基线必须在 train 内选择 threshold，并在
scene-disjoint split 迁移。

---

## 15. 已证伪或明确停止的方向

1. 继续增加 candidate K：正式 N=40 null；
2. raw goal-image cosine active glance：V1 显著有害，V2/V3 未救回；
3. 用 NavDP critic 跨方向选目标：critic goal-agnostic，0/9；
4. 冻结 diffusion denoising score 当 likelihood：约随机；
5. 直接在现成 expert trajectory 训 bearing：标签 97.5% 在 ±10°，会退化为 always-forward；
6. 把 X-NavDP 全局替换 mixed：R2 `+2/-1, p=1.0`；
7. 在 consumed pool 上事后学 `|bearing|>=150°` X-router；
8. sweep fixed radius：B0 只授权统一接口，不授权 selector；
9. 继续训练旧 shared decoder；
10. 用 pooled AUC 作为 deployment gate；
11. 在 development/blind 上调 O1 threshold/head/view dose；
12. 在 Novel-B 尚未解决前盲目扩大 3-leg joint，只会得到更多不可解释的 0。

---

## 16. 方法论纪律

### 16.1 每个增益必须同时报告

- N；
- scene cluster 数；
- paired gains/losses；
- exact McNemar/sign test；
- scene-cluster interval；
- 是否 oracle；
- 是否 consumed/development；
- 是否完整 closed loop。

N<20 一律标为轶事级。

### 16.2 配对纪律

- 同 episode、Goal-A、goal image、success radius、step budget；
- 同机、同 process contract 或显式跨进程 prefix identity audit；
- 每个 diffusion request 由 `(episode, leg, plan index)` deterministic seed 约束；
- 不允许因两臂 request 次数不同而静默漂移后续 noise；
- 所有 inactive controls 应逐字段一致。

### 16.3 分层纪律

- pooled candidate AUC 只作筛子；
- 必须检查 episode-internal pair；
- success-conditioned SPL 不能转换成 SR claim；
- conditional SR 不能冒充 joint；
- oracle upper bound 不能冒充 deployment method。

### 16.4 Scene budget

- 10 development 已耗尽；
- 20-scene pool 已耗尽；
- O1 只使用 train scenes；
- 16 blind 必须等完整方法冻结后一次性打开；
- blind 上不能选择 threshold、radius、controller、head 或 view dose。

---

## 17. 当前计算资源与作业状态

### 17.1 HPC

截至 O1 完成后的最后一次成功查询，用户队列为空。本文末次实时重查时
`torch` SSH 在 12 秒内未建立连接，因此下表是已完成作业账本，不冒充当下队列快照。

| Job | 状态 | 时间 | 结论 |
|---|---|---:|---|
| `15527318` CGC | FAILED fail-closed | 1:51 | teacher connectivity bug，未训练 |
| `15541189_[0-19]` X G1 A100 array | COMPLETED | 每 scene 约 8–18 min | 已汇总进入 X 归因线 |
| `15541190` X summary | COMPLETED | 0:14 | 工程完成 |
| `15561233` O1 smoke H100 | COMPLETED | 1:12 | preflight/teacher/OOF 工程通过 |
| `15561742` O1 formal H100 | COMPLETED | 9:26 | 五个 O1 gate 全通过 |

### 17.2 共享 5090

- bearing×memory 20 scenes 全部完成；
- 运行约 13:39–17:36 CST；
- 当前无 evaluator/server；
- GPU 0%，约 571 MiB 基础占用。

### 17.3 本机

- NavDP `21000` 与 MemNav `21002` 已按用户要求关闭；
- 两端口已释放；
- 项目 GPU server 显存从约 16.8 GiB 降到约 254 MiB；
- 当前没有闭环 evaluator/training process。

---

## 18. 当前优先级

### P0：固化已完成 factorial

- 把 5090 summary SHA、四臂结果、80 个 A-prefix identity pairs 与两处协议偏差写入
  canonical 状态；
- 本文已完成该信息固化；
- 若后续复制 artifact，必须保持远端原始目录只读。

### P1：运行 O1 frozen-policy state-transfer

这是当前唯一合理主任务：

- 不扩 head；
- 不重训 O1；
- 不读 development/blind；
- 在 61 条 Novel-B native rollout 中收集自然 states；
- 比较 O1 direction 与 native executable direction；
- 使用冻结 `R>=0.346483` abstention；
- 同时检查 native-success safety 与 native-failure opportunity。

### P2：P1 通过后才做 selective Novel-B closed loop

必须 paired source-on/off，并报告：

- gains/losses；
- abstention coverage；
- refresh frequency；
- native-success non-harm；
- native-failure rescue；
- invalid/undefined direction。

### P3：Novel-A 单独冻结验证

O1/P1 domain 是 Novel-B，不得偷换为 Novel-A。若 Novel-B transfer 通过，仍需独立冻结
Novel-A causal state protocol，再决定是否进入完整 2-leg adapter。

### P4：方法冻结后一次性 blind

最终 16-scene blind 同时比较：

1. native；
2. geometry memory；
3. fixed-radius memory bearing interface；
4. Novel source + abstention；
5. 完整 Novel+Revisit bearing adapter。

X/MPC 只有独立 recovery gate 通过才可作为预注册额外臂。

### P5：3-leg

只有 Novel-B source 与 conditional-C denominator 都稳定后，才扩大 end-to-end 3-leg。

---

## 19. 当前可声称与不可声称

### 19.1 可以安全声称

1. Geometry memory 在两次独立 N=40 paired runs 中均带来显著、零配对损失的 Revisit/
   joint 增益；canonical deterministic effect 为 `+17/-0`。
2. Oracle local bearing 在 N=40 上恢复全部观测到的 Novel-A failures，token execution
   完整兑现该信息。
3. Bearing update frequency 与 angular bias 存在明确 dose response；start-only 不够。
4. 在 consumed R0 上，一个固定 2.5 m bearing interface 保留 geometry memory aggregate
   effect。
5. 5090 factorial 证明 geometry memory 在 oracle-bearing 改写后的 A prefix 上仍保持
   `+16/-0`，方向与记忆可组合。
6. Official X/MPC 改善共同成功轨迹效率并提供 reverse capability，但没有全局 SR
   promotion 证据。
7. Raw similarity active glance 在 N=40 上显著有害，即使 scan 被赋予乐观零物理成本。
8. O1 证明 frozen NavDP representation 在 train-scene expert Novel-B states 上包含弱但
   显著的单视角 goal-conditioned progress signal。
9. Candidate-level learned ranking 含信息，但 activation/top-1 尚未超过公平 DINO baseline。

### 19.2 当前不可声称

1. 已解决 Novel navigation；
2. 已有部署可用的 visual bearing source；
3. Oracle `40/40` 或 factorial `29/40` 是方法 SR；
4. joint 天花板已变成 100%；
5. metric distance 对每条 episode 都无关；
6. X-NavDP 把 SR 提升到 80% 或显著优于 mixed；
7. O1 linear head 本身构成充分论文创新；
8. O1 expert-state OOF 可直接外推 Novel-A/frozen-policy failure；
9. Reverse graph 已显著优于 direct；
10. 3-leg `0/10` 证明 long memory 失败；
11. 当前任何结果授权打开 blind。

---

## 20. 权威工件索引

### 20.1 Canonical local reports

```text
.diagnostics/deterministic_memory_rebaseline_20260810/report.json
  SHA256 1f1d1db456ac748de073bce8ab6f0b2f74dc9c11cec150724bd84d7b31decf1f

.diagnostics/memory_bearing_only_b0_20260810/report.json
  SHA256 0856b6f6626dc68a33b024504651fa3a31382d801bd78671fde3fcedb665a86b

.diagnostics/xnavdp_recovery_g1b_full_20260809/report_audited.json
  SHA256 54313c1b4c4db14f30ba75a7390c8a0735dda35af50e517af9e2834a63d2444b

.diagnostics/xnavdp_active_failure_d1_20260810/eligibility.json
  SHA256 93bcb383f64b9e5e1981c45334339b655c3f4ac9841c719930b7ea6242f05b6c

.diagnostics/xnavdp_r0_residual_r1_20260810/report.json
  SHA256 7f31a7505a6a00e57d809178b64a97a20693be3c051ffd387f9fccafd762e8ec

.diagnostics/xnavdp_official_mpc_e1_20260810/report.json
  SHA256 2b2cd61e2b4ff4ab3f9661aba39746dfb7a6b6a5a64fb5afc05e6286ea786e4a

.diagnostics/xnavdp_official_mpc_e2_20260810/report.json
  SHA256 77eeed9eb6ec158f0409e547f89b0018e4f76110b7dbca12c5c7ab9e7c04464d

.diagnostics/xnavdp_r0_complete_r2_20260810/report.json
  SHA256 a0ada486dbb834bfaffadc25cd3a89ea8e3a5ffa05fc3aff6e3898c9cba53f8b

.diagnostics/phase_b_decision_units_20260809/report.json
.diagnostics/phase_b_calibration_20260808/report.json
.diagnostics/glp_stage2_20260808/report.json
.diagnostics/observed_frontier_bearing_coverage_20260809/report.json
.diagnostics/navdp_critic_direction_sweep_plan0_20260809/report.json
.diagnostics/navdp_goal_contrast_direction_sweep_20260809/report.json
```

### 20.2 Active-glance worktree

```text
/home/asus/Research/qw2440-active-glance/.diagnostics/
  active_glance_formal_20260810/summary.json
    SHA256 617e1b4a2cf96ca32d398763b0f2f0a09d9130f59cacc757565620b862cf8085
  active_glance_margin_formal_20260810/summary.json
    SHA256 2e7ab8e3e0764bf1d6a2ad3112af6f84f291e231e3c10a1c29b0b6595419a2c9
  active_glance_gated_formal_20260810/summary.json
    SHA256 e7a1b5f6b1f86adec36df1e6d90c2cd2486c0731de2871de1613967cc08ff0cf
```

### 20.3 O1 HPC

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  orbit_distilled_subgoal_o1_20260810/
    o1_formal_20260810a_job_15561742/
      teacher/report.json
      teacher/dataset.jsonl
      oof/report.json
      oof/oof_predictions.npz
      oof/fold_checkpoints/
```

Final hashes：

- teacher report：`7ae287731b096f39ff777c8e1ecf9dd1cbafe3ca1ccdbae63f1ccc0d8f627d03`；
- OOF report：`d40cd4675ec997f831e197116a9a69af641c60fa3dc703d5a1e503ece3cde648`；
- predictions：`82ce38b127bb33c0112d8b7102302ee092ad293060df5f6b7caf8a6d3e5effd9`。

### 20.4 5090 factorial

```text
/home/cv/memnav_eval/Nav-bearing-next/.diagnostics/
  bearing_memory_factorial_formal_20260810/
```

- summary SHA：`b07c410c17e0a0aea5d484a7248fc01dec7fe7ed3a8ea0b889af3e0ba72ad986`；
- protocol SHA：`551201b6c00952de79d1d225ae0972da810415a63c77db1f141162ebdb1050f1`；
- commit：`d5e2a9395a1cd026b323bfb2b027695237e1376b`。

### 20.5 关键设计与协议

```text
MemNavData/POINT_TOKEN_STEERABILITY_20260808.md
MemNavData/GOAL_POSTERIOR_DECISION_LAYER_20260807.md
MemNavData/GLP_LITERATURE_REVIEW_20260808.md
MemNavData/COUNTERFACTUAL_DIRECTIONAL_ADVANTAGE_PROTOCOL_20260809.md
MemNavData/CYCLIC_GOAL_COMPASS_PROTOCOL_20260809.md
MemNavData/XNAVDP_REVISIT_CONTROLLER_PROTOCOL_20260809.md

.worktrees/xnavdp-revisit-g1/MemNavData/
  DETERMINISTIC_MEMORY_REBASELINE_RESULT_20260810.md
  MEMORY_BEARING_ONLY_B0_PROTOCOL_20260810.md
  MEMORY_BEARING_ONLY_B0_RESULT_20260810.md
  XNAVDP_RECOVERY_G1B_RESULT_20260810.md
  XNAVDP_ACTIVE_FAILURE_D1_PROTOCOL_20260810.md
  XNAVDP_ACTIVE_FAILURE_D1_RESULT_20260810.md
  XNAVDP_R0_RESIDUAL_R1_PROTOCOL_20260810.md
  XNAVDP_R0_RESIDUAL_R1_RESULT_20260810.md
  XNAVDP_OFFICIAL_MPC_E1_PROTOCOL_20260810.md
  XNAVDP_OFFICIAL_MPC_E1_RESULT_20260810.md
  XNAVDP_OFFICIAL_MPC_E2_PROTOCOL_20260810.md
  XNAVDP_OFFICIAL_MPC_E2_RESULT_20260810.md
  XNAVDP_R0_COMPLETE_R2_PROTOCOL_20260810.md
  XNAVDP_R0_COMPLETE_R2_RESULT_20260810.md
  CGC_FORMAL_FAILURE_AUDIT_20260810.md
  ORBIT_DISTILLED_SUBGOAL_O1_PROTOCOL_20260810.md
  ORBIT_DISTILLED_SUBGOAL_O1_RESULT_20260810.md
  O1_POLICY_STATE_TRANSFER_P1_PROTOCOL_20260810.md
  o1_policy_state_transfer_p1_inputs_20260810.json
```

---

## 21. 最终判断

项目已经有三块可信成果：

1. **部署侧 Revisit 方法**：geometry memory 的 N=40 显著闭环增益；
2. **Novel 能力分解**：oracle bearing N=40 证明方向是强可恢复瓶颈，token interface 能
   兑现；
3. **统一接口与组合性**：B0 证明 fixed-radius bearing 足以保留 memory aggregate effect，
   5090 factorial 证明方向改写 A prefix 后 memory 仍显著有效。

真正没有完成的是最难的中间步骤：

> 从部署可见的单视角/历史信息中，产生在 frozen-policy 自然与失败状态上仍然有效、可周期
> 更新、可校准并能 abstain 的 Novel bearing。

O1 已经排除了“表示里完全没有方向信息”这一悲观假设，但 `17.74%` exact-bin 和
`85.5°` circular error 也表明它远未成为高精度 solver。当前最科学的动作不是扩大模型、
继续调 active-glance、重跑 X 或提前打开 blind，而是执行已冻结的 O1 P1 policy-state
transfer。只有它通过，才有资格进入 selective closed loop、独立 Novel-A 验证和最终
blind。
