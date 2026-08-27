# Nav-graph-blind / MemNav 项目总账（截至 2026-08-12）

更新时间：2026-08-12 18:35 CST  
状态：2026-08-11 至 2026-08-12 主工作区与外部 3-leg 工作区已同步；本项目 certified relocalization 四臂正式任务已完成并独立复算，外部 NRN 任务仍按带时间戳快照记录。  
本文定位：当前 canonical 全量状态。旧状态文档保留实验历史；若口径冲突，以本文及各冻结协议/原始报告为准。

---

## 0. 一页结论

### 0.1 目前真正成立的结论

1. **Revisit memory residual 是项目最确定、最可部署的正向能力。**
   - 历史 2-leg：native `4/40 -> geometry 19/40`，配对 `+15/-0`，exact McNemar `p=6.10e-5`。
   - deterministic R0：native `3/40 -> geometry 20/40`，`+17/-0`，`p=1.53e-5`。
   - 2026-08-11 fresh-episode replication：在共享 Novel-A 成功的 118 条中，raw-DINO direct
     将 Revisit-B 从 geometry 的 `93/118` 提高到 `109/118`；joint `93/160 -> 109/160`，
     `+20/-4`，`p=0.0015438795`，scene-cluster joint 95% CI `[+5,+15.625] pp`。
   - 2026-08-12 certified 四臂：共享 A 成功 120/160；certified Revisit `112/120=93.33%`，
     raw direct `106/120=88.33%`，geometry `91/120=75.83%`，native `27/120=22.50%`。
     certified 对 geometry `+23/-2, p=1.94e-5`，对 native `+86/-1, p=1.14e-24`；对 direct
     为 `+9/-3, p=0.146`，cluster CI 跨 0，因此不能声称显著优于 direct。

2. **RANSAC/SIFT 有真实几何信息，但不应继续担任统一 hard gate。**
   - 真 Revisit-C 中 pass precision `90.9%`，但 recall 只有 `66.3%`；
   - Novel 起点 pass precision 只有 `34.2%`；
   - known-Revisit fresh replication 中，绕过其否决与重选反而显著更好。
   - 正确角色是 telemetry、正支持、局部位姿质量或 certificate 的组成部分，而不是
     `pass = Revisit / reject = Novel` 的裁判。

3. **Novel 的方向是强可恢复瓶颈，但目前仍只有 oracle 机制结果。**
   - 同机 N=40：native Novel-A `28/40`，oracle periodic yaw 与 oracle-bearing token 都为
     `40/40`，`+12/-0`，`p=0.000488`。
   - 这证明给对方向时冻结 NavDP 具备完成能力；不证明部署时能得到方向。
   - Active-glance V1/V2/V3 最好 `25/40`，仍低于 native `31/40`；不能把原地扫描或 raw
     similarity 当作已解决的方向源。

4. **静态 Novel/Revisit selector 尚未解决，而且当前 benchmark 本来就提供 goal kind。**
   - Factorized F2/F8、Phase-B、GLP、OOF calibration、top-K、时序 latch 都没有在最终
     decision unit 上稳定超过公平基线。
   - 当前 benchmark 的最简强架构应直接使用 metadata：`Novel -> native NavDP`，
     `Revisit -> memory residual`。
   - 不读取 goal-kind 是更严格的开放集部署扩展，不应反过来阻塞当前论文主结果。

5. **X-NavDP 没有在本任务中形成新的整体 SR 杠杆。**
   - 相同 verified PointGoal 上：mixed `20/26`、base pure PointGoal `20/26`、official
     X+MPC `21/26`；X 相对 mixed 为 `+2/-1, p=1.0`。
   - 官方约 `61.20% -> 84.28%` 属于其 Isaac-Lab PointGoal / multi-embodiment benchmark，
     不能直接外推到本项目的 monocular ImageGoal-Revisit 闭环。

6. **外部旧 3-leg 结果存在严重构造混杂，不能再解释成纯 role/leg 效应。**
   - 标准 `Novel -> Novel -> Revisit`：`A=69/100`，`B|A=16/69`，`C|AB=13/16`。
   - role-swap `Novel -> Revisit -> Novel`：`A=66/100`，`B|A=54/66`，`C|AB=9/54`。
   - 两套 episode 不配对，而且旧后段 Novel 无 9 m 上限、初始 bearing 更难、目标图不绑定
     专家实际到达帧；标准 NNR 中 `42/69` 条可评 B 超过 9 m。
   - 已升级为 `multileg_v4_role_paired_20260812`：A/B 同 episode 距离差不超过 0.50 m，
     Goal C 必须只 anchor 在 leg A 且 leg B 最大共视不超过 negative threshold。
   - 本机不同场景首条 v4 数据通过独立 Habitat/pathfinder 审计与冻结 NavDP 闭环；这是实现
     smoke，不是 SR 结果。详见 `MULTIGOAL_ROLE_PAIRED_V4_RESULT_20260812.md`。

### 0.2 当前最合理的系统

```text
goal-kind 已知的 benchmark

Novel goal
  -> frozen native ImageGoal NavDP

Revisit goal
  -> raw DINO top-1 historical proposal
  -> LingBot causal metric pose
  -> legacy metric residual（当前最强 known-role reference）
  -> frozen mixed ImageGoal + PointGoal NavDP
```

若部署时没有 goal-kind：

```text
goal switch 时只做一次 certified episodic relocalization
  pass -> 输出可认证的 scale-free bearing
       -> fixed 2.5 m adapter
       -> frozen mixed ImageGoal + PointGoal NavDP
  fail -> Unsupported / Unknown，回退 native ImageGoal NavDP
```

这里不需要强行输出“Novel”这个语义标签。定位失败只表示当前历史不足以支持接管。

### 0.3 当前正式任务状态

截至 `2026-08-12 18:35 CST`：

- 本项目 certified relocalization 四臂闭环：20 scenes / 160 episodes 已完整结束；正式 report
  audit 为 ok，独立 raw CSV 复算一致。结果见 0.1 与 5.5。
- 外部工作区 NRN short-memory reset：正式 job `15642434` 运行中；当前只有 2 个完整 scenes / 20
  episodes 可比较，尚无净增益。

外部 NRN 的中间数字仍禁止写成最终结论。

---

## 1. 任务、名词与评测单位

### 1.1 2-leg 主基准

```text
start -> Goal A (Novel) -> Goal B (Revisit)
```

- `SR_A`：Novel A 成功率；
- `SR_B|A`：只在 A 成功的 episode 上计算 Revisit B；
- `joint = A and B`；
- 主项目中最可靠的 memory 结果都来自这一严格配对协议。

### 1.2 标准 3-leg

```text
start -> A (Novel) -> B (Novel) -> C (Revisit)
```

3-leg joint 同时受两个 Novel 段和一个 Revisit 段限制。joint 低不能自动归因于 memory；必须分别
报告 `SR_A`、`SR_B|A`、`SR_C|AB`。

### 1.3 外部 NRN role-swap

```text
start -> A (Novel) -> B (Revisit) -> C (Novel)
```

它不是新方法，而是归因实验：把 Revisit 与第二个 Novel 的顺序互换，以判断标准 3-leg 的低
第二段成功率究竟来自“第二段位置”还是“Novel goal role”。

### 1.4 Known role 与 role-free deployment

- 当前生成器 metadata 已明确标记 goal `kind`；主 benchmark 可以、也应该读取该标签。
- role-free 是更难的开放集扩展：系统必须从历史证据决定是否允许 memory 接管。
- `不读取标签` 本质上已经增加了一个 unknown-goal support 问题；它不是验证 Revisit 方法的
  必要前置条件。

---

## 2. 证据等级与口径纪律

| 等级 | 含义 | 当前代表结果 |
|---|---|---|
| A：严格闭环配对、完成审计 | 可以作为方法主结果或强内部确认 | geometry memory、fresh160 raw direct |
| B：冻结机制/上界 | 证明瓶颈或能力，不是部署方法 | Novel oracle bearing 40/40 |
| C：离线/资格证据 | 决定是否值得进闭环 | RANSAC train40、Phase-B、PnP certificate |
| D：定向小样本/机制案例 | 只能提出或排除假设 | 单场景 U-turn、active source 早期样本 |
| Running | 分母未冻结 | certified formal、NRN reset formal |

固定纪律：

1. 每个增益同时报告 `N`、配对 `+/-` 与显著性；
2. 跨机器或跨代码版本的 absolute SR 不拼接，只使用同机同进程的配对差；
3. candidate AUC、session existence、anchor top-1、closed-loop SR 是不同 decision unit；
4. development 已消费，不再用于新阈值或架构判定；
5. old 20-scene pool 与 fresh160 的同一 scene clusters 均已用于内部决策，不是 paper-final fresh
   scene confirmation；
6. oracle bearing 必须明确写 `given Habitat oracle bearing`；
7. 运行中 partial 只用于 transport/sanity，不做正式统计决策。

---

## 3. 进入这两天前的基线总账

### 3.1 Revisit geometry memory

| 实验 | native | memory | 配对 |
|---|---:|---:|---:|
| 历史 HPC 2-leg | 4/40 | 19/40 | `+15/-0`, `p=6.10e-5` |
| deterministic R0 | 3/40 | 20/40 | `+17/-0`, `p=1.53e-5` |

R0 中 Novel-A 为 `26/40`；在 A 成功条件下，native Revisit 为 `3/26`，geometry 为
`20/26`。这是项目最硬的可部署 memory baseline。

### 3.2 候选宽度 null

- K=1 vs K=8：joint `18/40 vs 18/40`；`+1/-1`，`p=1.0`。
- 结论：候选多样性不是总体瓶颈；不能继续用扩大 K 解释所有失败。

### 3.3 Novel oracle direction

| arm | Novel-A SR | 相对 native |
|---|---:|---:|
| native | 28/40 | — |
| periodic oracle yaw | 40/40 | `+12/-0` |
| oracle bearing + token | 40/40 | `+12/-0` |

- paired risk difference `+30 pp`；exact McNemar `p=0.000488`；scene-cluster 95% CI
  `[+15,+47.5] pp`；
- 12 gains 分布于 9 scenes；token 35/40 episode 激活，共 159 次，无 burst exhaustion；
- 只证明方向信息的可恢复上界与执行层能力。

### 3.4 Active-glance 负结果

| arm | SR | native | paired | p | avg glances | avg turns | turn/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 no margin | 20/40 | 31/40 | `+2/-13` | 0.00739 | 32.25 | 14.40 | 1948.5° |
| V2 margin | 24/40 | 31/40 | `+1/-8` | 0.03906 | 29.85 | 3.275 | 398.25° |
| V3 margin+gate | 25/40 | 31/40 | `+1/-7` | 0.07031 | 7.225 | 2.95 | 362.25° |

每次修复都减少干预与伤害，但没有超过 native。冻结结论：停止 raw-similarity active glance；
部署时不要求机器人先原地转一圈。

### 3.5 X-NavDP

| Revisit controller | B success / 26 | joint / 40 |
|---|---:|---:|
| mixed ImageGoal+PointGoal | 20/26 | 20/40 |
| base pure PointGoal | 20/26 | 20/40 |
| official X + official MPC | 21/26 | 21/40 |

X 相对 mixed `+2/-1, p=1.0`。它有真实 signed-velocity / reverse primitive，但本任务的主要
误差不在 executor 上，不能用“更新 controller”替代定位、方向与 role 分解。

### 3.6 Learned router / GLP

- Phase-B candidate AUC：learned `0.9535` vs DINO `0.9103`；
- session oracle threshold 基本持平，但 train 最优阈值 `0.397` 到 dev `0.807`；
- formal decision unit：model `72.7%`，DINO `87.3%`；
- OOF calibration 缩小迁移损失但仍未超过 DINO；GLP Stage 2 未通过。

结论：几何特征能帮助 session 内相对排序，但跨 scene absolute activation 校准失败；模型的
价值在“哪个 anchor”，不在“是否激活”。

---

## 4. 2026-08-11：Revisit 线的完整进展

### 4.1 train40 RANSAC / geometry expert 全审计

数据：40 train scenes、480 sessions、14,172 candidates；development/blind 未读取。

| 指标 | 结果 |
|---|---:|
| stable-support precision | 76.20% |
| positive candidate recall | 58.78% |
| hard-rejected positives | 622，跨 37 scenes |
| positive sessions activated | 124/155 |
| first-pass correct anchor | 102/155 |
| strict no-match false activation | 67/281 |

按 causal role：

| state | pass precision | positive recall |
|---|---:|---:|
| Novel 起点 `goal_b_t0` | 34.2% | 44.4% |
| Novel 中途 `goal_b_midpoint_t1` | 50.0% | 35.9% |
| 真 Revisit `goal_c_t0` | 90.9% | 66.3% |

scene-grouped OOF：

| unit | DINO | geometry | DINO+geometry |
|---|---:|---:|---:|
| candidate ROC-AUC | 0.8395 | 0.8802 | 0.9041 |
| session-macro AUC | 0.8123 | 0.8884 | 0.9081 |
| existence ROC-AUC | 0.8854 | 0.8856 | 0.9146 |
| positive-session top-1 | **116/155** | 107/155 | 113/155 |

含义：geometry signal 真实，但“更高 AUC”没有变成更好的 top-1。五次 RANSAC seed 在全部
14,172 rows 上完全一致，运行时无需重复五次。

### 4.2 known-Revisit direct 小规模配对

共享 Goal-A trace；只改变 Revisit B 的 memory 激活/候选规则。

| arm | A | B given A | joint |
|---|---:|---:|---:|
| geometry router | 29/40 | 21/29 | 21/40 |
| raw-DINO top-1 direct | 29/40 | 26/29 | 26/40 |

- paired `+6/-1`，`p=0.125`；scene-cluster joint CI `[0,+25] pp`；
- 六个 gain 分布于六 scenes；一 loss 位于 `pLe4wQe7qrG/episode_0001`；
- 方向明确但小样本未满足替换门，因此触发 fresh replication。

### 4.3 fresh160：raw-DINO direct 架构门通过

数据为旧 20 scene clusters 上重新生成的 8 episodes/scene，共 160；生成 acceptance 条件不变，
仅将 attempt budget 从 6 倍提高到 600 倍，最终 160/160 完成并通过 manifest/hash 审计。

共享 Novel-A：三臂均 `118/160 = 73.75%`。

| arm | Revisit B given A | joint | B SPL | B final distance |
|---|---:|---:|---:|---:|
| geometry router | 93/118 = 78.81% | 93/160 = 58.13% | 0.5565 | 1.843 m |
| raw-DINO direct | **109/118 = 92.37%** | **109/160 = 68.13%** | **0.7959** | **1.153 m** |
| native | 31/118 = 26.27% | 31/160 = 19.38% | 0.1136 | 6.223 m |

Primary direct minus geometry：

- `+20/-4`；joint `+10.0 pp`；conditional B `+13.56 pp`；
- exact McNemar `p=0.0015438795`；
- joint scene-cluster 95% CI `[+5,+15.625] pp`；conditional CI `[+6.42,+21.49] pp`；
- gains 分布于 12 scenes，losses 分布于 2 scenes。

Direct minus native：`+79/-1`，joint `+48.75 pp`，`p=1.34e-22`。

机制拆分：

- geometry 未激活的 22 条：geometry `9/22`，direct `20/22`，`+11/-0`；
- geometry 已激活的 96 条：geometry `84/96`，direct `89/96`，`+9/-4`。

结论：增益超过一半来自 hard-gate false negative；激活后绕过 RANSAC 重选也为净正。正式
decision 为 `replace_geometry_hard_gate_then_seek_fresh_scene_confirmation`。这不是 fresh-scene
或 blind 结果，160 条现已消费。

### 4.4 actionability 与 front-support 负结果

`pLe` loss 暴露：memory bearing 接近 `±166–176°`，PointGoal forward 分量会被裁到 `[0,10]`，
但这不代表所有 behind bearing 都应回退 native。

零参数规则 `forward < 0 -> native`：

- post-hoc T0 在 `pLe`：`+1/-0`；
- formal T1 前 4 scenes / 8 episodes：direct joint `6/8`，front-support `2/8`，`+0/-4`；
- 按预注册 loss=0 安全门提前停止。

原因：mixed decoder 仍能使用 lateral sign 与 goal image 完成 U-turn；native fallback 删除了
有用 memory guidance。endpoint、critic、behind、早期 bearing slope 均未形成可靠 safety gate。

### 4.5 unknown-goal factorized selector F2/F8

40 train scenes、436 extreme sessions：155 positive、281 strict no-match。

| system | correct support | correct anchor | wrong anchor | strict FP |
|---|---:|---:|---:|---:|
| hard geometry H | 365/436 | 93/155 | 14 | 9/281 |
| F2 seed 1 | 359/436 | 85 | 12 | 7 |
| F2 seed 2 | 365/436 | 89 | 12 | 5 |
| F2 seed 3 | 365/436 | 88 | 13 | 4 |

F8 correct anchor `90/89/90`，仍未超过 H 的 93，且一个 seed strict FP=11。主要失败是
34–38 条有 positive proposal 却 abstain。继续换 MLP、调 temperature、扩大 top-K 不再有信息
增量。

### 4.6 natural-stream hard pilot

3-leg 在这里仅作为旁路诊断容器，动作全部由 native NavDP；router takeover=0。

- Novel false positive：同一个错误 anchor 连续 `63/63` plans 通过 geometry；
- Revisit false negative：正确 DINO top-1 连续 `35/35` plans 被 geometry 拒绝；
- 重复静态检索不会产生独立新证据，temporal latch 只会把稳定错误锁得更牢。

冻结决定：停止扩大这条 temporal selector；返回 known-Revisit 主线。

---

## 5. 2026-08-12：开放集定位与 certified residual

### 5.1 MRC 为什么暂停

MRC 原意：DINO top-1 proposal 后，用 `[-4,0,+4]` 历史局部片段和冻结 LingBot 多视角几何
判断 goal 是否能在共同 3D 坐标中注册。

旧 90-row / 25-session / 22-scene artifact：

| signal/model | AUC / OOF AUC |
|---|---:|
| raw DINO | raw 约 0.596；scene-LOO 0.568 |
| cloud overlap | raw 0.737 |
| DINO + geometry scene-LOO | 0.735 |

但进一步审计发现：

- scene identity 对 geometry score 的影响大于 label；cloud overlap scene 内 z-score AUC 可从
  `0.737` 变为 `0.917`，说明主要是相对量；
- 三视角 signal correlation `0.77–0.82`，effective N 仅约 `1.14–1.18`；
- deployment top-1 proposal 在 155 positive sessions 中只有 `115` 条命中 positive，上限
  `74.2%`；
- 旧 sampler 丢弃单类 session，无法估计 strict no-match tail；
- signal 更像 pose stability，而非 teacher `covis >= 0.5` 的 goal-surface existence。

24-session A100 smoke 用时 `28m13s`；full 480 线性约 `9h24m`。由于观测设计不能区分 scene
nuisance、shared hallucination、proposal miss 与 label/action utility 错位，full collection 被
冻结暂停，而不是为了躲避负结果。

### 5.2 更正确的问题：可认证重定位，而不是二分类

冻结 v2 流程：

```text
ImageGoal
  -> DINO causal top-8 proposal
  -> SuperPoint + LightGlue + Fundamental-MAGSAC
  -> LingBot causal depth / pose
  -> reference 2D-3D + PnP
  -> bilateral geometric certificate
       inliers >= 16
       query hull >= 5%
       reference hull >= 5%
       reprojection RMSE <= 2 px
```

通过证书才输出 historical localization；失败输出 Unsupported/Unknown 并回退 native。

### 5.3 HPC certificate v1 与 v2

| run | data | TP/FP/FN/TN | decision |
|---|---|---:|---|
| v1 job `15633271` | 24 sessions | 8/2/1/13 | fail；发现单边 coverage 漏洞与短向量 actionability 定义问题 |
| v2 job `15634113` | 24 sessions / 19 design-disjoint train scenes | **8/0/1/15** | qualification gate pass |

v2：precision 100%，recall 88.9%，accepted 覆盖 6 scenes，中位位置误差 `0.131 m`。相对 LingBot
direct pose 仅 5/8 改善，sign-test `p=0.727`；因此不能声称 PnP 显著改善 pose，最硬价值是
fail-closed certificate。

### 5.4 单目尺度边界与 runtime v3

真实 runtime smoke：

- predicted/GT bearing `174.28° / 174.61°`；
- true distance `6.54 m` 却被在线 scale 估成 `15.29 m`；
- 像素内点、覆盖与重投影 RMSE不能认证 monocular metric scale。

因此 v3 收缩为 **scale-free bearing**：

- v2 accepted 8/8 bearing error `<4.45°`，median `2.35°`；
- 固定 2.5 m adapter 与 metric interface 在 B0 都是 `20/26`，`+1/-1, p=1.0`；
- contract 明确 `metric_distance_certified=false`；
- first call `2.09 s`，cache `0.15 ms`；移动更新、reset 清理、accepted mixed、rejected native
  均通过本机 smoke。

### 5.5 正式四臂闭环：完整结果

协议：20 scenes × 8 episodes；四臂共享一条 Goal-A trace；Williams arm order；同场景、同进程、
同 seed：

1. native；
2. old geometry router；
3. known-Revisit raw-DINO direct；
4. certified relocalization bearing residual。

原 array `15641052` 的 tasks 4–7 在 `gl055` 被 system uid 0 于 payload 前取消，repair array
`15642562` 已补齐且全部 exit 0；不是方法失败。summary `15642571` 因 Habitat 环境缺少 pytest
在读结果前失败，analysis-only replacement `15645446` 仅改测试解释器后完成；冻结 rollout、
summarizer 与统计规则均未改变。

20 scenes / 160 episodes 完整结果：

| arm | shared A | joint / B given A |
|---|---:|---:|
| native | 120/160 | 27/160 = 27/120 = 22.50% |
| geometry | 120/160 | 91/160 = 91/120 = 75.83% |
| raw direct | 120/160 | 106/160 = 106/120 = 88.33% |
| certified | 120/160 | **112/160 = 112/120 = 93.33%** |

完整 paired certified：

- vs native `+86/-1`，conditional `+70.83 pp`，`p=1.14e-24`，cluster CI
  `[+59.32,+81.68] pp`；
- vs geometry `+23/-2`，conditional `+17.50 pp`，`p=1.94e-5`，cluster CI
  `[+8.87,+27.64] pp`；
- vs raw direct `+9/-3`，conditional `+5.00 pp`，`p=0.146`，cluster CI
  `[-1.74,+12.60] pp`；
- 1,544 accepted plans，runtime failure=0；115/120 episode takeover，5/120 fail-closed native。

正式结论：certificate residual 显著优于 native 与 old geometry router，并与最强的 known-role
raw direct 基线统计持平、数值正向；不能声称显著优于 direct。其额外价值是 scale-free、
fail-closed 的部署接口。median uncached latency `5.01 s`、p95 `26.83 s`，仍需工程优化。

report SHA256：
`0e41a6d9b339d143229ba405b04802654d2053b5d641a03ed2d09aefc1a589f4`；audit 为 ok。
独立 raw-CSV reader 未导入项目 summarizer，复算出相同的四臂计数、paired +/- 与 exact McNemar。

此外，该正式集的 Goal-B 本身都是 Revisit。因此它证明 certificate bearing 在
Revisit 闭环可执行；role-free no-match safety 仍来自独立离线证书，尚未形成完整 mixed-goal
closed-loop 证明。

---

## 6. 外部工作区 3-leg 同步（qw2440，2026-08-11 至 08-12）

### 6.1 来源与审计范围

外部只读工作区：`/scratch/qw2440`。本次直接读取了 sbatch、生成器、evaluator、raw
`metric.csv`、plan JSON 与 Slurm 全用户账本；没有修改同学文件。

关键任务：

| job | 任务 | 状态/耗时 |
|---:|---|---|
| `15610208` | `pose_diag_3leg` | completed，36m50s |
| `15612533` | `gen_uturn_check` | timeout，45m18s；不构成结果 |
| `15623885` | NRN formal generation | completed，1h08m45s |
| `15626797` | NRN formal eval | completed，6h35m01s |
| `15642434` | NRN short-memory-reset formal | running at snapshot |

标准 NNR 与 NRN 均使用同一 patched `gs0_sym-6100_branchbias.ckpt`，SHA256：
`46bb1792efc74dcd7ed5adabacee461ee79b66d9967113858000f37f01e0894e`；同一 10 个 unseen
MP3D scenes，每场景 10 episodes。NRN 的 episode 是重新生成的，不与标准 NNR 逐条配对。

### 6.2 标准 NNR 的 episode-level pooled 结果

标准顺序：`Novel A -> Novel B -> Revisit C`。

| stage | exact pooled result |
|---|---:|
| Novel A | `69/100 = 69.0%` |
| Novel B given A | `16/69 = 23.19%` |
| Revisit C given AB | `13/16 = 81.25%` |
| joint | `13/100 = 13.0%` |

Router 行为：

- Novel B：69 个 eligible episode 中只在 1 个 episode false-activate；plan `105/6096` active；
- Revisit C：16 个 eligible episode 中 13 个激活；plan `669/835` active。

原 `agg_3leg.py` 报告的 `B|A=0.2086`、`C|AB=0.75` 是先算每 scene 条件率再等权平均；本文
使用 raw CSV pooled 分母，二者口径不同，不是数据冲突。

### 6.3 targeted pose / U-turn diagnostic

job `15610208` 在 `17DRP5sb8fy`、checkpoint `gs0_sym-5700` 上重跑 10 episodes，目标是解释
“anchor 稳定、C 仍失败”的案例。

两个典型失败的第一条 C plan：

| episode | GT bearing | selected first-waypoint bearing |
|---|---:|---:|
| `0006` | `+176.5°` | `-0.8°` |
| `0009` | `-175.2°` | `-1.0°` |

在需要近 U-turn 的 plans 中：

- episode 0006：28 plans 的 mean `|model bearing| = 2.22°`；
- episode 0009：19 plans 的 mean `|model bearing| = 6.84°`。

这证明了一个具体失败模式：**真实目标几乎在正后方时，最终选择的控制 waypoint 仍系统性
接近正前方**。它没有独立区分错误来自 LingBot pose、memory fusion 还是 diffusion decoder，
所以不能笼统写成“pose recovery 全部失败”。

跨标准 unseen100 的事后 bearing audit也一致：

- leg A success / fail episode 的 mean absolute bearing error：`23.0° / 91.2°`；
- leg B success / fail：`71.5° / 97.6°`；
- leg C success / fail：`71.1° / 108.5°`。

这些是 outcome-conditioned 描述，不是可部署 gate。

### 6.4 NRN role-swap 的正式结果

新顺序：`Novel A -> Revisit B -> Novel C`。

| stage | exact pooled result |
|---|---:|
| Novel A | `66/100 = 66.0%` |
| Revisit B given A | `54/66 = 81.82%` |
| Novel C given AB | `9/54 = 16.67%` |
| joint | `9/100 = 9.0%` |

Router 行为：

- Revisit B：66 个 eligible episode 中 60 个激活；plan `1115/1676` active；
- Novel C：54 个 eligible episode 中 2 个 false-activate；plan `75/4751` active。

最重要的角色对照：

| role | 标准 NNR | role-swap NRN |
|---|---:|---:|
| Revisit | C：13/16 = 81.25% | B：54/66 = 81.82% |
| 第二个 Novel | B：16/69 = 23.19% | C：9/54 = 16.67% |

这组数支持：标准 3-leg 的瓶颈不是“第二段天然难”或“C 段天然易”，而是 **Revisit 有 memory
支持，Novel 没有；两个 Novel 串联会压低 joint**。但因为两组 episode 重新生成、不是 paired
counterfactual，不能对 23.19% vs 16.67% 或 81.25% vs 81.82% 做 McNemar，也不能称论文级
因果确认。

### 6.5 NRN 的 short-memory reset 修复：当前未见净增益

修复只在第二个 Novel C 开始前清空 NavDP 自己的短期 observation window；不清除 MemNav
episodic map。假设是 controller FIFO 仍携带 A/B 末尾的异目标帧，导致新 ImageGoal 污染。

截至快照，两完整 scenes / 20 episodes：

- A `15/20`；Revisit B `15/15`；C eligible 15；
- original C `3/15`，reset C `3/15`；
- paired `+1/-1`，其余 13 相同；
- A/B outcome 逐条一致，说明修复确实只作用于 C lifecycle。

因此当前不能说 reset 有效。正式 job `15642434` 完成前保留假设；若全量仍 null，应停止这条
goal-scoped FIFO 修复，而不是调 reset 时机。

### 6.6 外部工作区的两个审计陷阱

1. NRN evaluator 沿用了标准 NNR 的 summary key 名：把 B 写成 `novel_B...`、C 写成
   `revisit_C...`，但 metadata 与 runtime require 明确是 B=Revisit、C=Novel。本文按 raw
   `goal.kind` 解释，不按旧字段名解释。
2. 原 false-activation 辅助脚本把未尝试下游 leg 也放进分母；本文只在上游成功、该 leg
   eligible 的 episode 中计算 activation。

### 6.7 对本项目的直接意义

- 3-leg 不要求另建一套 memory 架构；它是同一 `Novel native / Revisit residual` 组合的连续
  压力测试。
- joint 低时先拆三段，不要训练 memory 去修复 Novel 失败。
- 真正值得同步的是 state lifecycle：persistent episodic map 应跨 goals 保留；NavDP 的短期
  controller memory 是否应 goal-scoped reset，仍由正在运行的配对实验决定。
- 这组外部数据与本项目 fresh160 的 `Revisit direct 92.4%` 并不数值冲突：scene、episode、
  checkpoint/evaluator 与前置成功条件不同，只能比较各自同协议的条件效果。

---

## 7. 相关但不是这两天的新线：Nav-long3leg

本机 worktree：`/home/asus/Research/Nav-long3leg`；最后 commit 为 2026-07-20，8 月 11–12 日无
新增提交。它测试 route-disagreement/long-decision curriculum 的离线 action 学习，不是当前
闭环 3-leg 同学线。

此前文档停在 step 100，但远端 step 200/full-DDPM jobs 实际已完成。64 条固定离线样本的
独立 comparator：

- full diffusion action MSE `0.077952 -> 0.089999`，treatment **恶化 15.45%**；
- 13 improved / 51 worsened；paired bootstrap delta CI `[+0.00617,+0.01752]`；
- Goal-C `0.096384 -> 0.106710`，恶化 10.71%，CI 跨零；
- easy-turn 恶化 23.39%；hard-turn 只有约 0.95% 小改善且 CI 跨零；
- rank loss、aux direction/range 与 goal sensitivity 有改善，但没有转成 action fidelity。

结论：不要恢复“按 route disagreement 过采样再长训”的旧方向。它是典型的 auxiliary metric
变好、最终 diffusion action 变差。

---

## 8. 为什么过去很多实验效果不好

### 8.1 把五个不同 latent 当成一个 selector

```text
DINO similarity      -> 外观像不像
RANSAC                -> 单 pair 局部几何是否可解释
LingBot/PnP           -> pose/bearing 是否可恢复或自洽
teacher covis         -> goal surface 是否被历史覆盖
closed-loop utility   -> residual 是否优于 native
```

这些量相关但不等价。用一个 scalar threshold 同时回答 existence、anchor、pose 与 actionability，
会自然产生“离线很好、闭环不增益”。

### 8.2 相对排序可学，绝对激活跨 scene 漂移

Phase-B、F2、MRC 都重复出现：session 内 AUC/排序提高，但 train-to-unseen threshold 迁移失败。
scene texture、尺度、运动与重复结构给 absolute score 带来 nuisance；加模型容量不能增加缺失的
观测。

### 8.3 静态重复不等于时序证据

对固定 goal 与固定历史 pair 连续做 63 次相同判断，不会把 false positive 变成 true negative。
真正的时序必须引入新 current view、odometry、pose-chain 或主动验证，而非 latch/hysteresis。

### 8.4 Oracle 信息过强但不可部署

Habitat oracle 每次知道当前位置到目标的 geodesic bearing；真实单目机器人只有当前 RGB、goal
image 与 causal memory。`40/40` 说明控制能力，不说明 direction source 已解决。

### 8.5 Controller 不是当前第一瓶颈

mixed/base/X 在同一 PointGoal 上几乎持平。替换 executor、叠加 rotate/bearing/gate，若不先修
定位/role/信息来源，只是在强 controller 周围继续调接口。

### 8.6 过早追求 joint 会混淆分母

3-leg joint 是多个条件概率的乘积。若 Novel B 只 20%，即使 Revisit C 80%，joint 仍低。先报告
各 leg 条件 SR，才能知道该优化哪一支。

---

## 9. 冻结的架构决定

### 9.1 当前默认 benchmark 架构

```text
Known Novel
  frozen native NavDP

Known Revisit
  raw DINO top-1
  -> causal LingBot relative pose / bearing
  -> fixed-radius residual
  -> mixed NavDP
```

- RANSAC 不再 hard-veto 或主导候选重选；
- X-NavDP 不作全局默认替换；
- 不增加 learned bearing head；
- 不把 active glance 作为必需部署动作。

### 9.2 候选的论文方法形式

更优雅的形式不是“DINO + RANSAC 门控器”，而是：

> **Certified Episodic Relocalization Residual**：episodic map 只在能够从历史中自认证恢复 goal
> bearing 时提供一个 scale-free residual；否则冻结 ImageGoal policy 保持原样。

其创新点应落在：

1. 不是显式 Novel/Revisit 二分类，而是可执行 bearing 的 fail-closed certificate；
2. memory 与 native policy 的接口只有一个方向 residual，不替换整个导航栈；
3. persistent episodic state 与 goal-scoped controller state 分离；
4. 配对闭环证明 residual 的价值，negative controls 证明 hard geometry gate、active glance 与
   controller replacement 并非增益来源。

正式四臂结果已经表明 certificate 对 raw direct 为 `+9/-3`，但 `p=0.146`、cluster CI 跨 0；
因此不能声称 certificate 显著优于 raw direct。它当前成立的价值是显著优于 native/old geometry，
同时把最强 Revisit 能力收缩成 scale-free、fail-closed 的部署接口。在 mixed no-match 闭环完成前，
仍不能声称 role-free system 已完全解决。

---

## 10. 下一步优先级

### P0：冻结四臂结论，并完成 role-correct 3-leg 接线

1. known-role strongest reference 固定为 raw-DINO direct + legacy metric residual；
2. deployable candidate 固定为 certified scale-free bearing + 2.5 m residual；不再调 certificate
   threshold 或 adapter radius；
3. v4 3-leg 中 A/B 必须保持 native 并写因果记忆，C 才允许 known-Revisit residual；另设
   certified role-free arm，让 A/B/C 都自行 pass/abstain；
4. 外部 NRN reset 完成后只看同 episode C paired `+/-`，不看未配对绝对 SR。

### P1：最小的新闭环确认

- 在 v4 role-paired 3-leg 上先做小规模三臂：native / known-role direct / role-free certified；
- 主要读 `SR_A`、`SR_B|A`、`SR_C|AB` 与 A/B false takeover，不用低 joint 掩盖 C；
- 只在本机 transport smoke 通过后，把生成与闭环放到 HPC；不复用旧 confounded 3-leg 分母；
- 若 NRN reset null：停止 FIFO reset；若显著正向且零 loss，再将 controller short memory
  设为 goal-scoped。

### P2：论文级新确认

1. **fresh scenes**：与旧 20 clusters、train40 和当前外部 10 scenes 明确不相交；
2. **role-free mixed-goal safety**：同一冻结系统同时遇到 Novel no-match 与 Revisit，验证
   rejected->native、accepted->residual；
3. 仍按 scene cluster 做 paired bootstrap，不打开 blind。

### P3：论文叙事

- 主线：episodic relocalization residual 显著改善 Revisit；
- 机制：方向是可恢复瓶颈，冻结 policy 能兑现可信 bearing；
- 组合性：3-leg role-swap 显示 Revisit 能力可跨位置复用，joint 主要受重复 Novel stages 限制；
- 边界：Novel oracle 是 upper bound，开放集 certificate 仍需 mixed closed-loop confirmation。

---

## 11. 明确停止或不重复的事项

- 不再扩大 DINO/RANSAC top-K；
- 不再扫 Phase-B/GLP/F2/F8 的 threshold、temperature 或 MLP 宽度；
- 不把 RANSAC reject 当 Novel；
- 不恢复 front-support behind fallback；
- 不恢复 unconditional active glance；
- 不再用 old route-disagreement curriculum 长训；
- 不把 X-NavDP 官方 PointGoal 数字写成本项目 ImageGoal 增益；
- 不在 development 或 blind 上试架构；
- 不在正式任务结束前用 partial 数字做 paper claim；
- 不把 3-leg joint 单值当作 memory 方法质量。

---

## 12. 关键文档与原始收据索引

### 主项目

- `STATUS_20260810_FULL.md`：8 月 10 日前完整总账；
- `REVISIT_GEOMETRY_EXPERT_RESULTS_20260811.md`：train40 RANSAC 全审计；
- `REVISIT_KNOWN_PHASE_ABLATION_RESULT_20260811.md`：known-Revisit direct 小规模配对；
- `REVISIT_FRESH_CONFIRMATION_RUN_20260811.md`：fresh160 正式结果与独立审计；
- `REVISIT_FRONT_SUPPORT_RESULT_20260811.md`：behind fallback 负结果；
- `SELECTOR_DEEP_AUDIT_AND_NEXT_STEP_20260812.md`：selector 全去重审计；
- `UNKNOWN_GOAL_NATURAL_STREAM_HARD_PILOT_RESULT_20260812.md`：时序 latch 负结果；
- `MRC_SIGNAL_ATTRIBUTION_AND_LITERATURE_20260812.md`：MRC signal/root-cause 审计；
- `OPEN_SET_RELOCALIZATION_STATUS_20260812.md`：PnP certificate v1/v2 与 runtime 边界；
- `CERTIFIED_RELOCALIZATION_CLOSED_LOOP_PROTOCOL_20260812.md`：正式四臂冻结协议；
- `CERTIFIED_RELOCALIZATION_CLOSED_LOOP_RUN_20260812.md`：正式任务运行收据。

### 外部 3-leg 只读来源

- `/scratch/qw2440/eval_hybrid_pose_3leg.sbatch`；
- `/scratch/qw2440/pose_diag.sbatch` 与 `/scratch/qw2440/pose_diag/results/`；
- `/scratch/qw2440/eval_nrn/generate_threeleg_nrn.py`；
- `/scratch/qw2440/eval_nrn/eval_3leg_habitat_nrn.py`；
- `/scratch/qw2440/eval_nrn/formal_results/*/metric.csv`；
- `/scratch/qw2440/eval_nrn_fix/eval_3leg_habitat_nrn.py`；
- `/scratch/qw2440/eval_nrn_fix/formal_results_fix/`；
- Slurm jobs `15610208`, `15623885`, `15626797`, `15642434`。

---

## 13. 最终判断

项目并不是“所有效果都不好”。更准确的状态是：

- **Revisit 方法已经很好，而且得到两轮显著闭环支持；**
- **过去效果不好的主要是试图用同一批静态视觉分数解决 unknown-goal existence、pose 与
  actionability 三个问题；**
- **Novel direction 的上界极强，但部署方向源尚未建立；**
- **外部 3-leg role-swap 进一步表明 memory 能力随 Revisit role 迁移，低 joint 主要来自重复
  Novel 段，而不是 Revisit 本身失效。**

因此接下来不应继续堆模块或长训一个新 selector。最有价值、也最优雅的收束是：把当前显著的
Revisit residual 写成一个 **可认证、可 abstain、scale-free 的 episodic relocalization adapter**；
用冻结 native NavDP 保住 Novel，用严格的 fresh-scene mixed-goal 闭环确认安全边界。
