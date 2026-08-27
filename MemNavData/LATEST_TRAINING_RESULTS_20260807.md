# MemNav / NLSR 最新训练与评测成果总报告

更新时间：2026-08-07 19:47 CST

个人子仓库：`/home/asus/Research/Nav-graph-blind`

核对提交：`3c53e437be03899859f16cfbdbd0951612b8dcad`

适用范围：截至上述时间已经落盘并完成审计的本地/HPC 结果，以及仍在运行任务的状态

母目录约束：本文和相关修改均位于个人子仓库，不修改 `/home/asus/Research/Nav`

---

## 0. 一页结论

目前最可靠的结论不是“继续把旧 MemNav 训练更久”，而是重新划分系统职责：

```text
冻结的官方 NavDP
    负责原生 ImageGoal 与局部避障/控制

冻结的 LingBot + DINO 长期记忆
    负责持久 keyframe、pose/depth graph 和候选生成

小型概率定位/价值模型
    负责 match/no-match、候选排序、pose 修正/不确定度、utility/risk

高置信 Revisit
    沿 memory pose graph 产生短程 subgoal

Novel 或不确定
    byte-equivalent 回退到原生 NavDP
```

到目前为止，最强且统计上最清楚的闭环结果是 20 场景、40 条 2-leg episode 的几何
memory 基线：原生 NavDP joint SR 为 `4/40 = 10.0%`，DINO + SIFT/RANSAC 几何 memory
为 `19/40 = 47.5%`；在 Goal A 已成功的 31 条 episode 上，Revisit B 从 `4/31` 提升到
`19/31`，配对结果为 memory-only gain 15、loss 0，exact McNemar
`p=6.1035e-5`。这证明“可靠长期记忆几何”确实能带来很大收益。

但它仍不是最终方法：

- 旧 MemNav 的 GateCurr 能改善条件式 Revisit 闭环，却没有改善 retrieval top-1、平均
  action epsilon-MSE 或 pose 长尾；不能把所有提升归因给 curriculum 本身。
- 旧 MemNav 的 Novel 分支明显弱于原生 NavDP，核心不是 LingBot 转弯漂移，而是它并未
  继承 NavDP、几乎忽略 goal image，并且 selector 的评分时域与实际执行时域错配。
- ResidualGate 保护了 visual branch，但没有恢复原生 NavDP；GoalSwap 训练也没有学出
  正的 goal-conditioning gap。
- learned patch/temporal 模型已经能做较好的候选验证和重排，但尚未稳定取代 DINO
  candidate proposal 或 RANSAC 最终确认。
- 当前 3-leg `0/10 joint` 主要先被第二个 Novel 目标 B 卡住，只有一条 episode 真正进入
  Revisit C，不能据此声称“LingBot 长程 C memory 全部失败”。
- 2026-08-07 已完成新的 600-session causal teacher，并用真实数据证明 K+1/no-match
  objective 可训练；但 DINO-only set model 的候选 top-1 低于原始 DINO 基线。因此还没有
  可报告的新 NLSR checkpoint、W&B 长训或闭环 SR。

当前优先级是把新的 causal teacher 与 patch/temporal、LingBot overlap/pose/depth 和
metric-scale feature 做严格 join，生成 trainer-compatible 审计 artifact，再按
gradient preflight -> 小样本 overfit -> 三 seed 训练 -> 一次 development -> shadow ->
闭环的顺序推进。现在不应提交 final blind，也不应继续延长旧 shared decoder。

---

## 1. 证据等级

本文把结果分成三类，避免把 smoke、development 和最终结论混写。

### 1.1 已确认（Confirmed）

满足以下至少一类条件：严格配对闭环、完整 artifact 审计、可复现因果干预，或已有足够
明确的失败保护。

1. 20-scene/40-episode 2-leg 几何 memory 相对 native NavDP 的提升：`4/40 -> 19/40`。
2. 旧 MemNav Novel 分支存在 goal-image 弱敏感和 selector horizon mismatch。
3. 原生 NavDP Novel 在旧五场景十路线对照中明显优于 MemNav：`9/10` 对 `2--4/10`。
4. GateCurr 的 retrieval set loss 和 gate classification 改善，但 retrieval top-1、action
   epsilon-MSE 和平均 pose 指标没有同步改善。
5. 600-session causal teacher 已完整生成、带内容哈希并通过审计；其
   `deployment_approved=false` 状态也已确认。
6. DINO-only K+1 objective 梯度、dustbin/no-match 和真实标签 overfit 均机械可训练，但
   其 development candidate top-1 目前不够好。
7. 旧 Phase-B 任务 `15440645` 在真实 artifact 审计处正确失败，没有产生训练 checkpoint。

### 1.2 方向性/开发证据（Directional / Development）

这些结果有价值，但样本量、场景复用或随机性控制不足以作为最终性能声明。

1. GateCurr 在两个场景、20 条条件式 Revisit B 上由 `6/20` 提升到 `10/20`。
2. Reverse graph 从 direct `19/40` 提升到 `25/40`，但旧 evaluator 没有逐次严格配对 DDPM
   noise。
3. Patch/listwise reranker 在 development 上能把正确候选平均排名和 top-1 提高。
4. LingBot-native cloud/pose consistency 在 development candidate verification 上有信号，
   但不同场景的绝对尺度和阈值稳定性仍不足。
5. 本地 Novel-B privileged metric target 能把 `3/5` 提升到 `5/5`，说明冻结 NavDP 的
   local controller 本身有能力，缺的是高层 metric direction；样本仍只有五条 eligible。
6. 十场景 conditional-C development 评测中，native NavDP 为 `4/10`，geometry
   memory 为 `7/10`，更严格的 same-prefix/per-request-seed 对照中 direct-gap16 为
   `8/10`、graph-gap16 为 `6/10`。这是 development 证据，不是 final blind。

### 1.3 尚未验证（Not Yet Verified）

1. 新 NLSR-V2 是否优于几何 router 的闭环 SR。
2. learned localizer 是否能完全移除 RANSAC，而不增加 false activation。
3. strict noise-paired reverse graph 是否仍能保持 `25/40`。
4. 3-leg conditional-C 在更大、scene-disjoint 且一次性冻结的 blind 场景上是否仍能
   保持 development 中的收益。
5. goal-conditioned frontier residual 是否能净提升 Novel-B 而不伤已有成功。
6. shared `6100/6270/6270` 最终 checkpoint 的完整 seen/unseen/3-leg pooled 结果。
7. final-reserved 16-scene/32-episode blind 表现。

---

## 2. 评测语义与不可混比项

### 2.1 Novel、Revisit 与 3-leg

2-leg：

```text
start -> Goal A (Novel) -> Goal B (Revisit，目标区域在 A 路径中出现过)
```

3-leg：

```text
start -> Goal A (Novel) -> Goal B (Novel detour) -> Goal C (Revisit A-era memory)
```

这里的 Novel/Revisit 是相对当前历史 memory 而言，不是训练/测试场景是否见过。

### 2.2 三种常见协议

| 协议 | Goal A | 后续 leg | 能回答什么 | 不能回答什么 |
|---|---|---|---|---|
| replayed-A conditional-B | 回放专家/固定轨迹 | policy 控制 B | 隔离 Revisit B 能力 | 不能算真实 joint SR |
| executed-A conditional-B | 原生 policy 实际跑 A，仅 A 成功才评 B | policy 控制 B | 更接近真实系统，但分母是 `B|A` | 不能忽略 A 失败 |
| end-to-end joint | A/B/C 都由系统执行 | 全链路 | 最终真实成功率 | 低 joint 不能直接定位是哪一段失败 |

因此：

- `gatecurr600 6/10` 是 replay A 后的 Revisit B，不等于完整 2-leg `6/10 joint`。
- shared `gs0/gs25` 表中的 `SR_B_given_A` 也是 replay A 的条件指标。
- `19/40` 是完整 2-leg joint，因为 A 由 native NavDP 执行，B 只有在 A 成功后继续。
- 3-leg 的 `C|AB` 分母只有真正完成 A 和 B 的 episode；当前是 `0/1`，不是 `0/10` 的
  long-memory 估计。

### 2.3 SR、SPL 与单步 loss

- SR：是否在规定距离/预算内到达，闭环行为优先级最高。
- SPL：成功基础上惩罚绕路；失败通常贡献 0。
- action epsilon-MSE：随机 timestep/noise 上的 denoising 回归，不等价于闭环方向正确。
- retrieval set loss：正例集合总概率，不保证部署时 `argmax` 的 top-1 是正例。
- gate accuracy：是否判断为 Revisit，不保证 decoder 知道怎样利用该 memory。
- aux loss：旧 empirical aux head 在 GateCurr 中被冻结，曲线变化不能解释为 aux 在学习。

任何跨表比较都必须先检查：场景、episode、A 是 replay 还是 executed、success radius、
step budget、`exclude_recent`、candidate K、DDPM seed、controller 和 checkpoint 是否一致。

---

## 3. 历史模型和当前角色

| 名称 | 来源/机制 | 现在如何看待 |
|---|---|---|
| `flowgate2600` | 旧 shared MemNav baseline | GateCurr 的正确受控 baseline |
| `gatecurr600` | 从 flowgate2600 暖启动，500-step gate teacher curriculum/all-leg | Revisit 行为有方向性改善，但不是最终基线 |
| `residualgate1000` | visual branch 始终保留，memory 作为 residual | 缓解 Novel 负迁移，但没有恢复 native NavDP |
| `novelgs_res1000_early40_w025` | early Goal-A sampling + GoalSwap margin | GoalSwap 未学成，不能作为成功改进 |
| geometry router | DINO proposal + SIFT/RANSAC verification + frozen NavDP | 当前最可靠闭环 memory baseline |
| reverse graph | verified anchor 后沿历史 pose chain 走短 subgoal | 有强开发信号，需严格 noise-paired 重跑 |
| NLSR-V2 | frozen NavDP + probabilistic LingBot graph residual + abstention | 当前主线，尚无新闭环 checkpoint |

---

## 4. GateCurr：机制、结果和正确解释

### 4.1 为什么引入 curriculum

旧模型训练初期直接用预测 gate 控制 decoder：

```text
revisit attention bias = log(g_pred)
novel attention bias   = log(1 - g_pred)
```

当早期 `g_pred` 偏低时，即使训练端已经选到 GT-positive anchor，正确 revisit token 仍被
attention mask 压低，action loss 很难教会 decoder 使用 memory。因果干预也显示：修正错误
anchor 后第一跳只变化 `0.23--0.66 deg`；强制 gate=1 虽然使输出变化更明显，却没有稳定
改善方向。

GateCurr 使用：

```text
g_used = (1 - r) * g_pred + r * y_GT
```

`r` 在前 500 optimizer steps 从 1 线性降到 0：先保证正确分支打开，让 decoder 学会使用
memory；随后完全切换到预测 gate，适配真实 inference。gate BCE 始终监督 `g_pred`。

### 4.2 固定离线对照

正确对照是 `flowgate2600` 与 `gatecurr600`，不是 `flowgate1600`。

| 指标 | flowgate2600 | gatecurr600 | 结论 |
|---|---:|---:|---|
| retrieval set loss | 0.204852 | **0.134628** | 明显下降 |
| retrieval top-1 | **89.58%** | 87.50% | 没有改善 |
| gate accuracy | 69.44% | **80.56%** | 改善 |
| revisit gate accuracy | 54.17% | **72.92%** | 改善 |
| action epsilon-MSE | **0.086110** | 0.087323 | 基本持平/略差 |
| aux x-y MSE | **4.376839** | 4.577455 | 没有改善 |
| revisit position error | **1.5487 m** | 1.6070 m | 没有改善 |
| camera rotation error | 13.617 deg | **13.074 deg** | 小变化，CI 跨 0 |

多正例 set loss 奖励“分给所有正例的总概率”，部署却取单个 `argmax`。所以 set loss 降低
与 top-1 不升并不矛盾。在 48 个 revisit 样本中，41 个两者都命中、4 个两者都漏、2 个
从 hit 变 miss、1 个从 miss 变 hit。

### 4.3 闭环结果

五个 unseen scene、十条 replay-A Revisit B smoke：

| 指标 | flowgate2600 | gatecurr600 |
|---|---:|---:|
| SR | 2/10 | **6/10** |
| SPL | 0.105 | **0.395** |
| mean final distance | 2.998 m | **1.882 m** |

样本只有 10，McNemar `p=0.21875`，属于强方向性信号，不是最终统计结论。三个共同失败
强制 oracle-positive anchor 后仍为 `0/3` rescue，说明剩余错误不主要是 learned top-1。

两个场景、20 条 replay-A Revisit B 配对：

| 指标 | flowgate2600 | gatecurr600 |
|---|---:|---:|
| SR | 6/20 | **10/20** |
| SPL | 0.1371 | **0.3583** |
| mean B path | 10.737 m | **6.989 m** |
| mean final distance | 3.219 m | **2.438 m** |

配对为 4 gain、0 loss，但场景只有两个；它支持“continuation 改善了条件式 Revisit
闭环”，不能把 gain 全部归因给 gate scalar 或 curriculum，因为 continuation 同时更新了
retrieval、projection 和 decoder，而且没有“相同 600 steps、无 curriculum”的严格控制。

### 4.4 GateCurr 的最终定位

GateCurr 的真实价值是证明：即使 retrieval anchor 和平均 gate 几乎相同，下游
memory-to-action 映射也能改变闭环结果。它没有证明：

- retrieval top-1 已修好；
- aux pose 已修好；
- 更硬地把 gate 设为 1 会更好；
- Novel 能力不受损；
- 继续训练旧 shared decoder 会持续提升。

此外，旧实验的 empirical aux head 和 LingBot input 均冻结/no-grad；`w_aux_pose * aux_loss`
会进入日志标量，但没有有效梯度更新这条 head。因此 W&B aux 曲线只能看作数据分布/诊断，
不能称为学习曲线。其 position error 呈明显长尾：median `0.523 m`、mean `1.607 m`、p90
`5.013 m`，四个样本贡献 77.5% aux MSE；这更像稀疏 scale/pose tail，而不是全局 axis
仍然翻转。

主要来源：

- [RETRIEVAL_CAUSAL_DIAGNOSTIC_20260802.md](RETRIEVAL_CAUSAL_DIAGNOSTIC_20260802.md)
- [UNSEEN_SCENE_PAIRED_SMOKE_20260803.md](UNSEEN_SCENE_PAIRED_SMOKE_20260803.md)

---

## 5. Novel 负迁移、ResidualGate 与 selector

### 5.1 纯 Novel 闭环结果

五个 scene-disjoint 场景、十条 start-to-A：

| 方法 | SR | SPL | mean final distance | mean path |
|---|---:|---:|---:|---:|
| 官方原生 NavDP | **9/10** | **0.896** | 2.120 m | 6.000 m |
| flowgate2600，强制 Novel | 3/10 | 0.300 | 3.475 m | 7.811 m |
| gatecurr600，强制 Novel | 2/10 | 0.065 | 3.974 m | 13.905 m |
| residualgate1000，旧 24-point selector | 4/10 | 0.374 | 2.330 m | 10.665 m |
| residual，selector 只评 executed 前 2 points | **7/10** | **0.626** | 2.553 m | **6.303 m** |

ResidualGate 的改动是：

```text
旧 complementary: visual weight = 1 - gate, revisit weight = gate
新 residual:      visual weight = 1,        revisit weight = gate
```

它避免 gate 偏高时把 image-goal branch 关掉，所以 `2/10 -> 4/10` 有恢复；但 base policy
并不是官方 NavDP，因此仍远低于 `9/10`，也没有“gate=0 时严格等于 native NavDP”的结构
保证。

### 5.2 根因一：旧 MemNav 几乎不看 goal image

同 state、同 history、同 DDPM noise/timestep、同 gate，只交换正确/错误 goal image：

- flowgate 的 goal-swap output RMS 只有 seed-change RMS 的 `0.13%--0.14%`；
- residualgate 的比例约 `0.25%--3.16%`；
- 原生 NavDP 同类对照为 `176.8%`。

这不是路径断线，因为输出有非零变化；而是 goal 条件相比 current-state shortcut 和
diffusion noise 太弱。旧 MemNav 也不是“NavDP 外接 memory”：它只有约 57.26M 参数、
8-layer decoder、17 conditioning tokens，未继承官方 NavDP 的 135.73M/16-layer policy
权重。所谓 pretrained Novel image backbone 也没有加载原 NavDP image encoder。

### 5.3 根因二：Goal-A sampling 太晚

旧 loader 由 `window=32`、`num_scale=8`、`exclude_recent=83` 推出：

```text
anchor margin = 39
Goal-A k_lo   = 39 + 83 = 122
```

但 inference 第一次可规划大约在 `k=40`。十条评测路线中，4 条完全生成不了 Goal-A
training row，另有 2 条到 `k>=122` 时已经在 1 m success radius 附近。模型于是更容易学
“沿当前视角继续向前”，而不是学习远距离 goal-conditioned direction。

### 5.4 根因三：selector 时域错配

模型生成 24 个 waypoint，但闭环每 8 simulator frames 重规划，实际只执行约前 2 个
waypoint。旧 selector 却用 24 点全部未来碰撞/净空来打分。只把 collision scoring 对齐
到执行前缀后，SR 从 `4/10 -> 7/10`，配对 3 gain、0 loss。该结果样本小，不能把数字 2
永久硬编码；正确实现应由：

```text
committed_waypoints = ceil(exec_horizon_frames / pred_digit)
```

动态推导，并把远端 waypoint 作为折扣风险而不是一票否决。

### 5.5 为什么不再继续训练 shared decoder

旧系统同时更换了 NavDP 的 image encoder、current geometry、decoder 和 critic。Revisit
更新会通过共享 decoder 破坏 Novel；epsilon-MSE 又看不到 selector/controller mismatch。
因此当前主线改为冻结完整官方 NavDP，而不是继续在 `gatecurr600`、`residualgate1000` 上
长训。

主要来源：

- [NOVEL_ROOT_CAUSE_AUDIT_20260804.md](NOVEL_ROOT_CAUSE_AUDIT_20260804.md)
- [NOVEL_NAVDP_PAIRED_EVAL_20260804.md](NOVEL_NAVDP_PAIRED_EVAL_20260804.md)

---

## 6. GoalSwap：具体机制与负结果

### 6.1 机制

对于 Novel row，固定：

- 当前图像与完整 history；
- diffusion noise 和 timestep；
- gate 与 expert action/noise target；

只把正确 goal image 换成同一 MP3D 场景、另一 episode 的 goal。优先同一 goal type，要求
错误目标距当前至少 `0.5 m`，且与正确目标 bearing 相差至少 `30 deg`。

定义：

```text
E_correct = MSE(epsilon(correct goal), expert noise)
E_wrong   = MSE(epsilon(wrong goal),   expert noise)
gap       = E_wrong - E_correct
L_swap    = max(0, 0.05 - gap)
L_total  += 0.25 * L_swap
```

它不为错误 goal 伪造 action GT，而只要求正确 goal 比错误 goal 更能解释真实 expert
noise。当 `gap >= 0.05` 后 hinge 为零，避免鼓励错误输出无限变大。

### 6.2 训练配置与结果

Run：`memnav_novelgs_res1000_early40_w025_20260804`

读取 checkpoint：约 `checkpoint-1100`

实际训练：约 1112 steps / 1.77 epochs

| 指标 | 结果 |
|---|---:|
| 107 个 logged gap mean | `-0.000442` |
| last-20 gap mean | `-0.001780` |
| last-20 margin loss mean | `0.051780` |
| 目标 gap | `+0.050000` |

错误 goal 的 denoising error 并没有高于正确 goal；最后阶段 gap 仍略为负。输出 RMS 会
变化，不代表变化方向正确。结论是 GoalSwap objective 已接通，但旧 shared decoder 在该
训练规模和结构下没有学出有用 goal conditioning。不能把这个 run 写成 Novel 修复成功。

相关实现：

```text
InternNav/internnav/dataset/memnav_dataset_lerobot.py
InternNav/internnav/model/basemodel/memnav/memnav_policy.py
InternNav/internnav/model/basemodel/memnav/goal_swap.py
InternNav/internnav/trainer/memnav_trainer.py
InternNav/scripts/train_memnav/submit_novel_goalswap_8h.sh
```

---

## 7. Retrieval / Router：DINO、learned verifier 与 RANSAC

### 7.1 三个不同问题

必须区分：

1. **分类/verification**：给定一对 query/candidate，判断是否真匹配。
2. **retrieval/ranking**：在同一 session 的很多候选中，把正确帧排到 top-1/top-K。
3. **routing**：综合 match/no-match、pose 可靠性和动作收益，决定是否启用 memory。

W&B 上很高的 DINO CLS accuracy 往往是 pair 或 gate 分类准确率，不等于正确历史帧一定
排在 top-1，更不等于启用该帧后导航动作会更好。

### 7.2 learned patch/temporal 的已有能力

旧 binary verifier development 结果：

| 方法 | Accuracy | ROC-AUC | AP | Brier lower is better |
|---|---:|---:|---:|---:|
| DINO cosine | 84.4% | 0.950 | 0.917 | 0.135 |
| Patch | 95.4% | 0.990 | 0.971 | 0.041 |
| Patch + temporal | **96.3%** | **0.991** | **0.973** | **0.032** |

但在对应 33 个有 positive 的 session 中，top-1 都是 `24/33`；mean positive rank 则由
`3.79 -> 2.00 -> 1.64`。所以它首先是优秀 verifier/reranker，不是独立 global retriever。

随后在 40-train/10-development、temporal-NMS top-32 的 task-aligned co-visibility
协议中，listwise 结果为：

| 排序 | correct top-1 | mean first-positive rank | MRR |
|---|---:|---:|---:|
| raw DINO | 24/35 = 68.57% | 3.086 | 0.766 |
| pointwise patch+temporal | 29/35 = 82.86% | 2.800 | 0.868 |
| listwise patch+temporal | **30/35 = 85.71%** | **2.229** | **0.895** |

这是明确的 development 结构信号，但五/十个 development 场景已经被反复查看，模型仍
标记 `deployment_approved=false`，不能当 final blind。

### 7.3 为什么 RANSAC+SIFT 有用

DINO 擅长整体语义/外观，可能把相似走廊、墙面或相邻重复帧排在前面。SIFT + Essential
Matrix RANSAC 要求几十个局部 correspondence 共同满足同一个相机运动模型，因此能拒绝
很多 perceptual alias。20-scene 闭环的大增益证明了这种可靠验证的价值。

但 RANSAC teacher 与导航 co-visibility 目标也不完全一致。完整 task-aligned relabel 中，
旧 SIFT teacher 对极端正负标签 precision 高但 recall 低，会拒绝低纹理真 revisit，也可能
因共享背景接受目标表面不可见的候选。因此当前定位是：

```text
DINO/temporal-diverse candidate proposal
    -> learned patch/LingBot rerank and calibrated confidence
    -> 不确定候选才用 RANSAC fail-closed confirmation
```

目标不是永久依赖 RANSAC，而是在 learned model 通过跨场景、零/低误触发审计前保留安全
fallback。

### 7.4 temporal-NMS 和 top-K 的作用

失败样本 `cV4RVeZvu5T/episode_0000` 中，正确簇在 frame 53--90，raw DINO top-1 却是
frame 209；正确帧直到 raw rank 20 才出现。temporal-NMS gap=16 后，前 3 个是
`209, 125, 69`，RANSAC 依次拒绝前两个并接受 frame 69。该单 episode 的 C 从 final
distance `4.4185 m` 失败变成 `0.9923 m` 成功；第一次三候选验证 `127.3 ms`，cache 复验
`0.62 ms`。

这说明 candidate diversity 能救回“相邻错误帧占满 top-K”的失败，但 gap=4/8/16 的选择
必须由 train OOF 决定，不能根据一个开发样本写死。

主要来源：

- [PATCH_TEMPORAL_ROUTER_TRAINING_20260805.md](PATCH_TEMPORAL_ROUTER_TRAINING_20260805.md)
- [TASK_ALIGNED_RETRIEVAL_ROUTER_20260805.md](TASK_ALIGNED_RETRIEVAL_ROUTER_20260805.md)
- [PROBABILISTIC_MEMORY_LOCALIZATION_PLAN_20260806.md](PROBABILISTIC_MEMORY_LOCALIZATION_PLAN_20260806.md)
- [ONLINE_ROUTER_FAILURE_AUDIT_20260805.md](ONLINE_ROUTER_FAILURE_AUDIT_20260805.md)

---

## 8. 冻结 NavDP + 几何 memory：当前可靠基线

20 个场景、每场景 2 条 2-leg episode：

| 指标 | Native NavDP | Geometry memory + frozen NavDP |
|---|---:|---:|
| Novel A | 31/40 | 31/40 |
| Revisit B given A | 4/31 = 12.9% | **19/31 = 61.3%** |
| Joint | 4/40 = 10.0% | **19/40 = 47.5%** |

配对：

```text
both success:       4
geometry-only:     15
native-only:        0
both failure:      21
exact McNemar p: 6.1035e-5
Novel false activation: 0/40
```

这个结果支持两点：

1. 长期 memory geometry 是主要新增价值，不是两个旧场景上的偶然波动。
2. 冻结 native NavDP 能保住 Novel ceiling `31/40`，避免旧 shared decoder 的负迁移。

但 12 个 eligible B failure 中，7 个从未激活 memory，5 个激活后仍失败；其中至少 3 个
是错误/弱 anchor，其余约 2 个更像 direct point-goal/local-control mismatch。下一步需要
概率 candidate set、pose uncertainty 和 graph subgoal，而不是单一阈值。

在相同五场景十路线的小型 controller ablation 中：

| Controller | Goal B given A | Joint |
|---|---:|---:|
| gatecurr600 direct decoder | 8/9 | 8/10 |
| geometry router + NavDP point-goal | 8/9 | 8/10 |

这说明大增益来自“可信 memory 几何”，尚不能证明更换成 NavDP point-goal controller 本身
是必要增益。direct arm 知道 A-to-B phase boundary，并不是可部署 router。

来源：

- [ONLINE_ROUTER_FAILURE_AUDIT_20260805.md](ONLINE_ROUTER_FAILURE_AUDIT_20260805.md)
- [GATECURR_DIRECT_CONTROLLER_AB_20260804.md](GATECURR_DIRECT_CONTROLLER_AB_20260804.md)

---

## 9. Reverse graph、3-leg 与 Novel-B

### 9.1 Reverse graph 的开发信号

| Controller | Novel A | Revisit B given A | Joint |
|---|---:|---:|---:|
| direct point-goal, gap16 | 31/40 | 19/31 | 19/40 |
| reverse graph, gap16 | 31/40 | **25/31** | **25/40** |

配对为 6 gain、0 loss，sign-test `p=0.03125`。其直觉是：不要求一次预测跨越整个长距离，
而沿历史 pose chain 逐个走 1.25 m 左右的短 subgoal，并在每个 node 重新定位。

该数字目前只能称为 structural signal。旧 evaluator 只在 episode reset 时 seed 一次；
direct/graph 的早期 diffusion 调用次数不同，后续 noise 可能不同。严格 runner 已设计为共享
Goal-A trace/JPEG hash、重建相同 FIFO，并按 `(episode, leg, plan index)` 固定每次 DDPM
seed；正式结论必须来自该协议。

### 9.2 当前 end-to-end 3-leg

十场景结果：

| 阶段 | Native | Geometry router |
|---|---:|---:|
| A Novel | 6/10 | 6/10 |
| B Novel given A | 1/6 | 1/6 |
| C Revisit given A/B | 0/1 | 0/1 |
| Joint | 0/10 | 0/10 |

这组结果首先说明第二个 Novel B 很弱，导致 C 的分母塌缩为 1。它不能证明长程 Revisit C
整体失败。正确实验必须分三层：

1. causal replay A prefix，只评 conditional-B；
2. causal replay A/B prefix，只评 conditional-C，并加入 oracle candidate/oracle pose；
3. 最后才跑真实 end-to-end joint。

### 9.3 已完成的十场景 conditional-C

后续已经用固定的 causal source A/B prefix 直接重放到 C，使 10 条 episode 全部有
资格评估 Revisit C。该协议是 conditional diagnostic，不是完整 3-leg joint SR。

第一份五臂结果：

| C controller | C SR | mean SPL | mean final distance |
|---|---:|---:|---:|
| native NavDP | 4/10 | 0.2180 | 7.262 m |
| geometry top-1 | 7/10 | 0.4938 | 4.638 m |
| geometry temporal top-K | 7/10 | 0.4938 | 3.660 m |
| oracle anchor | 8/10 | 0.6520 | 2.808 m |
| oracle metric point | 9/10 | 0.7446 | 2.564 m |

它表明 memory 对长间隔 C 有真实价值；`7/10 -> 8/10` 之间仍有 candidate/anchor
缺口，`8/10 -> 9/10` 之间仍有 metric pose/控制长尾。由于样本只有 10 条，
top-K 相对 top-1 的 SR 持平不能用来固定 candidate gap/K。

随后的严格 direct/graph 对照共享 source prefix、FIFO、geodesic 与逐 request seed：

| C controller | C SR | mean SPL | mean final distance |
|---|---:|---:|---:|
| native | 4/10 | 0.2049 | 6.174 m |
| direct-gap16 | **8/10** | **0.5300** | **2.875 m** |
| graph-gap16 | 6/10 | 0.3325 | 4.518 m |
| oracle-anchor direct | 7/10 | 0.5502 | 3.014 m |
| oracle-anchor graph | 5/10 | 0.3627 | 4.450 m |
| oracle point | 7/10 | 0.5657 | 2.911 m |

native 到 direct 是 4 gain、0 loss，但 McNemar `p=0.125`；direct 到 graph 是 1 gain、3 loss，
`p=0.625`。因此可以说 direct memory 有强方向性信号，不能说 reverse graph 已经比
direct 更好。该 job 的 GPU rollout 全部完成；原 `summary.json` 因 summarizer 把字符串
`True` 当作数字解析而失败，后续只在修复后的 summarizer 上恢复 aggregate，没有重跑
或改写 rollout。

证据路径：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/three_arm_conditional_20260806/
  full_condonly_c4946eb_20260806_v1/conditional_c_five_arm/summary.json

/scratch/yz11502/Research/Nav-axis-uturn-results/graph_conditional_c_20260806/
  graph_condc_gap16_full_v3_job_15440630/summary_recovered_5fd4e3d.json
```

### 9.4 本地 Novel-B 因果诊断

两场景六条 3-leg episode 中 5 条到达 A。增加相同 diffusion 分布的候选从 16 到 64，仍
不能救回两个困难 `17DRP5` episode；失败样本中只有约 15%--23% 候选能产生短期进展，
候选 endpoint heading 高度集中。这说明问题不是单纯 diffusion seed 不走运，而是没有
生成正确的高层方向。

只替换 Goal-B metric target 的 privileged 上限：

| Arm | Goal B given A | mean final B | mean success path | mean success SPL |
|---|---:|---:|---:|---:|
| native ImageGoal pair | 3/5 | 2.113 m | 10.151 m | 0.764 |
| 1.25 m geodesic subgoal | **5/5** | 0.981 m | **6.669 m** | **0.977** |
| exact final metric point | **5/5** | **0.979 m** | 7.834 m | 0.886 |

因此困难 Novel-B 的主要缺失层级是稳定的长期 metric target/direction；A* 不是达到 5/5
的必要条件，但短 geodesic subgoal 能提高效率。冻结 NavDP 已经是合格 local controller。

随后 goal-blind observed-frontier residual 在五条 eligible 上与 native 都是 `4/5`：它救回
一个失败，却破坏另一个成功。结论是 generic exploration 不等于 ImageGoal；不能继续调
一个固定 stagnation threshold，必须训练 goal-conditioned frontier/graph-subgoal ranker，
并在不确定时保持 native。

来源：

- [ELEGANT_MEMORY_GRAPH_EXPERIMENT_20260806.md](ELEGANT_MEMORY_GRAPH_EXPERIMENT_20260806.md)
- [LOCAL_MULTIGOAL_CAUSAL_AUDIT_20260806.md](LOCAL_MULTIGOAL_CAUSAL_AUDIT_20260806.md)

---

## 10. 当前主线：NLSR-V2

NLSR-V2 的目标不是再训练一个替代 NavDP 的大 decoder，而是训练一个小型、可校准、会
abstain 的长期记忆 residual 系统。

### 10.1 推理架构

```text
ImageGoal + current RGB-D
        |
        +--> frozen official NavDP --> native proposal a0
        |
        +--> frozen LingBot/DINO causal graph
                 |
                 +--> temporal-diverse candidate set
                 +--> explicit no-match/dustbin
                 +--> candidate match/rank
                 +--> pose residual + covariance
                 +--> utility/advantage + harm/coverage

uncertain / no eligible residual
        -> execute cached a0 exactly

high-confidence memory match
        -> verified anchor -> reverse graph short subgoals

high-confidence no-match + native stagnation
        -> goal-conditioned frontier residual

near matched anchor
        -> return to image-dominant final alignment
```

### 10.2 强不变量

1. 官方 NavDP encoder、decoder、critic 全部冻结。
2. abstain path 必须与 native trajectory bytes、FIFO 和 diffusion seed 完全一致。
3. memory 不确定、feature 缺失、NaN、stale graph、OOD 或 coverage 低时必须回退 native。
4. goal change 只 reset goal posterior/confirmation，不清 persistent graph，也不默认清
   NavDP FIFO。
5. model 不直接预测 waypoint；它选择/修正 graph/frontier residual，由 frozen NavDP 做
   collision-aware local control。
6. Novel/Revisit 不依赖 evaluator phase label，而由 match/no-match posterior 决定。

### 10.3 小模型 heads

初版是约 50k--200k 参数的 masked set model，冻结所有大 backbone：

- match/no-match/ambiguous head；
- candidate listwise rank head；
- advantage mean/uncertainty head；
- harm head；
- wrapped SE(2) pose residual/covariance head；
- candidate coverage-miss head。

训练标签严格分开：co-visibility/pose target 只作 label，不能进入 feature；跨场景 goal 可
用于 match/no-match 和 OOD，但没有合法同场景 action utility 时不能伪造 rollout label。

### 10.4 为什么它比旧 gate 更合适

旧 scalar gate 同时承担了“是否有 match”“哪个 anchor 正确”“memory action 是否优于
visual action”三个不同问题，并用 complementary fusion 直接压制 visual branch。NLSR
把它拆成独立 head，并显式建模 dustbin、pose uncertainty、utility 和 harm；任何一个
条件不够可靠都 abstain，而不是让一个平均 gate 值决定整条 action path。

完整设计与 Go/No-Go 门槛见：

- [NOVEL_MEMORY_RESIDUAL_V2_20260807.md](NOVEL_MEMORY_RESIDUAL_V2_20260807.md)
- [CURRENT_BASELINE_TRAINING_V2_20260806.md](CURRENT_BASELINE_TRAINING_V2_20260806.md)

---

## 11. 最新 causal teacher artifact

### 11.1 已完成任务

| Job | 名称 | 状态 | 用时 |
|---:|---|---|---:|
| `15461734` | `nlsr_teacher_a` | COMPLETED | 00:11:14 |
| `15461785` | `nlsr_teacher_relay` | COMPLETED | 00:00:02 |
| `15461909` | `nlsr_teacher_b` | COMPLETED | 00:12:07 |

Artifact：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/
  nlsr_routes_41c1773_20260807_v1/
  causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/
```

### 11.2 数据规模与契约

```text
49,373  frozen exact LingBot DINO embedding inputs
600     causal sessions
480     train sessions
120     development sessions
17,845  emitted candidate rows
8,923   counterfactual candidate rows
```

全部 candidate frame 严格早于 decision frame；factual/counterfactual、Goal B/C、train/dev
均有完整覆盖。标签只来自 metadata covis curve 或 rendered-goal depth reprojection，不用
DINO/RANSAC 自报结果作 label。

### 11.3 session 分布

| 类别 | 数量 | 比例 |
|---|---:|---:|
| shortlist 内有 positive | 189/600 | 31.5% |
| strict no-match | 345/600 | 57.5% |
| ambiguous | 66/600 | 11.0% |

在“shortlist 内确实有 positive”的条件下：

| K | conditional recall@K |
|---:|---:|
| 1 | 76.19% |
| 4 | 88.36% |
| 8 | 93.12% |
| 16 | 96.83% |
| 32 | 100% |

这说明 top-K candidate generation 总体有较高上限，但 top-1 和最差场景仍需 learned
ranking；同时，超过一半 session 是 strict no-match，显式 dustbin 不是可选项。

### 11.4 Novel B 与 Revisit C 的差异

Factual Revisit C：

| Split | positive coverage | conditional top-1 | conditional top-4 |
|---|---:|---:|---:|
| train | 75/80 = 93.75% | 88.00% | 94.67% |
| development | 19/20 = 95.00% | 89.47% | 100% |

Factual Novel B：

| Split | positive coverage |
|---|---:|
| train | 24/160 = 15.0% |
| development | 3/40 = 7.5% |

因此最新数据支持当前架构语义：Revisit C 通常真的存在可用 memory candidate；Novel B
通常应输出 no-match，并继续 native ImageGoal 或经过严格 utility/risk 审核的 frontier
residual。把两者塞进同一个“平均 gate”会天然混淆。

### 11.5 内容哈希

```text
teacher.csv
fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313

audit.json
cd7091bb8c398ac5a7f558003a3fb467cc450a9b65e4416e29b7d27e1fad14c6
```

Artifact 状态明确为：

```text
status=audited_not_deployment_approved
deployment_approved=false
```

---

## 12. 最新真实数据 learnability smoke

该 smoke 直接读取上述真实 600-session teacher，排除 66 个 ambiguous session，用 7 个
causal/deployable DINO 与候选分布特征训练显式 K+1 candidate/no-match model。它只验证
objective 是否可学，不是最终 LingBot geometry model。

### 12.1 梯度与小样本 overfit

```text
encoder gradient norm:       0.6521
rank head gradient norm:     0.9772
no-match head gradient norm: 0.9997
```

16-session overfit：

| 指标 | 结果 |
|---|---:|
| initial CE | 3.2611 |
| final CE | 1.1896 |
| joint accuracy | 100% |
| match accuracy | 100% |
| candidate recall@1 | 100% |

所以 loss、gradient、dustbin 和 set packing 机械上没有断路；模型能记住真实标签。

### 12.2 完整 strict session 开发结果

```text
train strict sessions:       429
development strict sessions: 105
ambiguous excluded:           66
```

| 方法 | joint localization | match accuracy | candidate recall@1 |
|---|---:|---:|---:|
| max-DINO + train-only threshold | **88.57%** | 92.38% | **82.35%** |
| DINO-only learned set model | 86.67% | 92.38% | 67.65% |

DINO-only learned model的 match ROC-AUC `0.95195`、AP `0.93472`、Brier `0.06579`，说明
no-match classification 已有信号；但 ranking 明显弱于 raw DINO top-1。结论不是 objective
失败，而是输入特征不足：正式模型必须加入 directional patch/temporal、LingBot-native
overlap/refinement/depth confidence、metric pose/scale 和 uncertainty。

本机临时证据：

```text
/tmp/nlsr_set_objective_smoke.py
/tmp/nlsr_set_objective_smoke_result.json
/tmp/nlsr_causal_teacher_20260807.csv
```

结果 JSON SHA256：

```text
1633959ae6dc26d083d2788c800c1b8011a891183de6fdc23bad9b118ce6c436
```

重要边界：目前没有因此产生新 NLSR model checkpoint，也没有对应新 W&B run，更没有
闭环 SR。

---

## 13. 旧 Phase-B 为什么没有训练成功

旧 `lb_phase_b` job `15440645`：

```text
State:    FAILED
Elapsed:  00:00:46
ExitCode: 1:0
```

它先通过 18 个 synthetic/unit tests，然后在真实 artifact audit 正确停止：

```text
expected rows: 1244
actual CSV:    1240 rows (1241 lines including header)
session max covis drift: 0.166666...
no positive candidates
no negative candidates
no selected-positive sessions
no strict no-match sessions
```

所以不能说“旧 Phase-B 已经训练过，只是效果不好”；真实 backward/训练根本没有开始。

更重要的是，新 teacher 与旧 trainer 的 ABI 不同：

- 新 teacher 提供 causal DINO shortlist、task-aligned co-visibility、strict no-match 和
  factual/counterfactual label；
- 旧 trainer 期待 1244-row collector 中的 LingBot overlap/refinement/depth confidence、
  metric pose target、external scale quality 和 selected-positive session。

不能把新 `teacher.csv` 直接塞给旧 trainer。必须先 join 新 teacher 与 deployment-valid
patch/LingBot feature cache，并重新生成完整、可审计、trainer-compatible artifact。

---

## 14. shared `lg154/qw2440` 最新对照

### 14.1 checkpoint 的正确含义

三个 lineage 都使用 `tc500` curriculum：

```text
memnav_wsfrz_gs0_tc500_sym_k40
memnav_wsfrz_gs25_tc500_sym_k40
memnav_wsfrz_gs25_tc500_vscale_k40
```

- `gs0`：GoalSwap weight 0，不是“无 curriculum”。
- `gs25`：GoalSwap weight 0.25。
- `sym/vscale`：fusion/value scaling variant。

### 14.2 已完成的中间 checkpoint 结果

评测是 replay Goal A 后的 `SR_B_given_A`，不是完整 joint 2-leg SR。

| Checkpoint | Unseen SR / SPL | Seen SR / SPL | Seen - Unseen SR gap |
|---|---:|---:|---:|
| `gs0_sym-5700` | **71% / 0.439** | 56% / 0.307 | -15 pp |
| `gs25_sym-6100` | 70% / 0.413 | **56% / 0.305** | -14 pp |
| `gs25_vscale-6200` | 51% / 0.262 | 51% / 0.258 | 0 pp |

正确解释：

1. `gs25_sym` 没有超过 `gs0_sym`，所以目前没有证据表明 GoalSwap 改善 Revisit B。
2. `vscale` 同时把 seen/unseen 压到约 51%，零 gap 是共同性能崩塌，不是泛化改善。
3. scene 间方差很大，pooled 平均会掩盖某些 scene 的 40--50 pp 差异；应继续报告每场景
   SR/SPL 与 scene bootstrap，而不是只看总均值。
4. 这些共享 checkpoint 与我们 NLSR 主线回答不同问题：前者仍是旧 shared MemNav
   decoder 的条件式 Revisit 训练，后者冻结 NavDP 并显式建模 match/no-match/utility/risk。

### 14.3 当前 final sweep 状态（截至 2026-08-07 16:15 CST）

| Job | 任务 | 状态 | 已运行 |
|---:|---|---|---:|
| `15448593` | 2-leg unseen final | RUNNING | 04:09:16 |
| `15448594` | 2-leg seen final | RUNNING | 03:58:41 |
| `15441614` | 3-leg unseen | RUNNING | 12:32:51 |
| `15441615` | 3-leg seen | RUNNING | 12:22:05 |

2-leg final checkpoint 是 `gs0_sym-6100`、`gs25_sym-6270`、
`gs25_vscale-6270`。当前目录已有部分 scene summary，但任务未结束；不能把部分结果拼成
final pooled score。3-leg 日志同样只能作为运行状态，不能在任务完成前下结论。

共享路径：

```text
/scratch/qw2440/eval_checkpoints_frozen/
/scratch/qw2440/eval2leg_unseen/
/scratch/qw2440/eval2leg_v1/
/scratch/qw2440/eval2leg_results_unseen10/
/scratch/qw2440/eval2leg_results_seen10/
/scratch/qw2440/slurm_logs/
```

---

## 15. W&B 应该怎样读

### 15.1 Retrieval

优先看并同时报告：

```text
eval/retrieval_set_loss
eval/retrieval_top1_hit
eval/candidate_recall_at_K
eval/mean_first_positive_rank or MRR
eval/no_match AUC/AP/Brier
```

train retrieval loss 波动较大通常来自每 batch 的 episode、正例数量、候选长度和 hard
negative 难度变化；它不自动表示优化不稳定。固定 eval set 的 top-1/Recall@K 才能回答
部署 retrieval 是否改善。

### 15.2 Gate

`gate accuracy` 只回答 “Novel/Revisit 分类”，不能回答：

- top-1 anchor 是否正确；
- memory pose 是否准确；
- decoder 是否会使用 memory；
- 使用 memory 是否优于 native action。

旧 GateCurr 已经展示了 gate accuracy 大幅提高而 action epsilon-MSE 不升的分离现象。

### 15.3 Aux / pose

旧 empirical aux head 冻结时，aux 曲线不是训练效果。正式 NLSR 应报告：

- raw/corrected translation median、p90；
- wrapped yaw median、p90；
- covariance-error correlation 与 coverage；
- metric scale source/quality bucket；
- pose tail 对实际 graph/controller 的影响。

### 15.4 Goal conditioning

仅看 `action_loss_novel` 不够。必须同时看：

```text
goal_swap_error_gap       target > 0
goal_swap_output_rms      只表示有变化，不表示方向正确
correct-vs-shuffled rank/activation
fixed same-state same-seed causal sensitivity
Novel closed-loop SR/SPL
```

当前 GoalSwap run 的 gap 为负，因此不能因普通 action loss 下降就宣称模型学会 image goal。

---

## 16. 当前阻塞点

### 16.1 Memory localizer 数据 ABI 正在接通，尚未完成审计

新的 causal teacher 已有正确 labels 和 no-match 分布，Phase-B 的 patch/LingBot deployment
feature join 已于提交 `3c53e43` 开始正式采集。当前阻塞点已从“ABI 完全未接通”
收窄为“join 尚未完成并通过最终 artifact audit”。旧 Phase-B artifact 仍缺行且与
teacher 语义漂移，不能复用。新 join 在 exact cover、no-match、positive、finite value、
scale/route receipt 和 content hash 全部通过前，trainer 不应产生正式 checkpoint。

### 16.2 DINO-only ranking 不够

真实数据 smoke 中，learned DINO-only candidate recall@1 为 67.65%，低于 max-DINO 的
82.35%。需要加入真正解决候选关系的 feature，而不是继续增加同类 scalar MLP 深度。

### 16.3 Novel frontier 只有 privileged upper bound

metric target 5/5 证明方向层值得做，goal-blind frontier 4/5 对 4/5 又证明 generic
exploration 不够。正式 candidate/utility collector 和 goal-conditioned ranker 尚未完成。

### 16.4 Reverse graph 最强数字仍有 noise confound

`25/40` 不能作为 final claim，必须使用 strict shared-prefix/per-request seed runner 重跑。

### 16.5 3-leg 分母不足

当前 end-to-end 中只有一条进入 C，所以 joint `0/10` 仍不能估计 C 的长期记忆
能力。十场景 conditional-C 已完成，显示 direct memory 的 development SR 可从 native
`4/10` 提高到 `7--8/10`；下一个分母问题是 conditional-B 和更大的 scene-disjoint
conditional-C。在这两层通过前，不应只扩充同样的 joint 评测并继续得到不可解释的 0。

### 16.6 Final blind 尚未获准

所有 learned router/localizer 输出仍是 `deployment_approved=false`。在 model、threshold、
candidate K、graph spacing、burst、seed protocol 未冻结前，不能运行最终 blind 并反复根据
结果调参。

---

## 17. 下一步精确执行顺序

### P0：完成真实 NLSR memory localizer artifact

1. 以新 teacher 的 `session_id/candidate identity/content hash` 为 authority。
2. 对同一 causal candidate 生成 directional patch/temporal feature。
3. 生成 LingBot cloud overlap、pose consistency/refinement、depth confidence、metric pose 和
   external scale quality。
4. 严格 join，禁止 target/covis/error/GT pose 进入 feature allow-list。
5. 审计 train/development scene role、candidate exact cover、strict no-match、selected positive、
   finite values 和全部内容哈希。

### P1：训练前门槛

1. 在真实 artifact 上执行所有 head 的 finite nonzero gradient preflight。
2. 对 16--32 个平衡 session 做小样本 overfit，要求 match、rank、dustbin 和 pose head 都能
   拟合。
3. 三 seed、scene-grouped internal split 训练；stopping/threshold 只看 train-internal。
4. development 只在模型和阈值冻结后运行一次。
5. 若 candidate top-1、no-match calibration 或 pose p90 未超过既有 reference，停止闭环。

### P2：shadow 与闭环

1. 先运行 no-action shadow，统计 would-activate、false activation、coverage 和延迟。
2. abstain episode 必须与 native bit-exact。
3. 20-scene paired closed loop：native、geometry R0、learned+RANSAC fallback、learned-only。
4. 报 Novel A、Revisit B|A、joint、SPL、paired transitions、memory utilization 和 p95 latency。

### P3：Graph 与 3-leg

1. strict noise-paired direct vs reverse graph。
2. 至少 20 条 conditional-B，拆解 goal switch/FIFO/high-level direction。
3. 至少 20 条 conditional-C，加入 candidate oracle、pose oracle、graph controller 上限。
4. 只有 conditional-B/C 通过后再跑 end-to-end 3-leg。

### P4：Novel frontier 支线

1. 先做 deployment-pose candidate coverage，不用 Habitat goal coordinate 作 feature。
2. 同 state、同 seed 收集 native/candidate H8/H24 utility 与 harm label。
3. 训练 goal-conditioned rank/advantage/risk head；不确定时完全回退 native。
4. 先 shadow，达到足够 coverage 且 zero/low harm 后再启用 action residual。

### P5：最终 blind

模型、特征 schema、所有阈值和 runner 全部冻结并写 SHA 后，才运行一次
16-scene/32-episode blind manifest。任何 development 门槛未过均 No-Go。

---

## 18. 代码、artifact 与复现索引

### 18.1 旧 MemNav 训练代码

```text
InternNav/internnav/dataset/memnav_dataset_lerobot.py
InternNav/internnav/model/basemodel/memnav/memnav_policy.py
InternNav/internnav/model/basemodel/memnav/goal_swap.py
InternNav/internnav/trainer/memnav_trainer.py
InternNav/scripts/train_memnav/
```

### 18.2 NLSR-V2 当前代码

```text
MemNavData/novel_memory_residual_v2.py
MemNavData/novel_candidate_set_schema_v2.py
MemNavData/nlsr_set_ranker.py
MemNavData/train_nlsr_set_ranker.py
MemNavData/build_manifest_causal_dino_embeddings.py
MemNavData/build_manifest_causal_covisibility_teacher.py
MemNavData/assemble_manifest_causal_covisibility_teacher.py
MemNavData/build_causal_ground_scale.py
MemNavData/external_causal_scale_contract.py
MemNavData/build_memory_graph_candidate_artifact.py
MemNavData/phase_b_feature_schema.py
MemNavData/phase_b_deployment_inference_contract.py
```

### 18.3 关键 evaluator / runner

```text
MemNavData/eval_2leg_habitat.py
MemNavData/eval_3leg_habitat.py
MemNavData/run_strict_graph_2leg.sh
MemNavData/run_graph_conditional_c.sh
MemNavData/run_all_in_one_router_eval.sh
NavDP/baselines/navdp/navdp_server.py
NavDP/baselines/navdp/policy_agent.py
```

### 18.4 关键 checkpoint 哈希

```text
flowgate2600
debd079c6f578e9c6e2c1f0e70f6dc8fc2c2230785c28d6da2fae118a665b38b

gatecurr600
9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7

official NavDP
3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947

LingBot-Map long
832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409

LingBot commit
7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
```

### 18.5 Frozen blind manifest

```text
MemNavData/strict_graph_blind_20260806.json
SHA256 b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9
```

### 18.6 本报告的主证据文档

```text
MemNavData/RETRIEVAL_CAUSAL_DIAGNOSTIC_20260802.md
MemNavData/UNSEEN_SCENE_PAIRED_SMOKE_20260803.md
MemNavData/NOVEL_ROOT_CAUSE_AUDIT_20260804.md
MemNavData/NOVEL_NAVDP_PAIRED_EVAL_20260804.md
MemNavData/GATECURR_DIRECT_CONTROLLER_AB_20260804.md
MemNavData/PATCH_TEMPORAL_ROUTER_TRAINING_20260805.md
MemNavData/TASK_ALIGNED_RETRIEVAL_ROUTER_20260805.md
MemNavData/ONLINE_ROUTER_FAILURE_AUDIT_20260805.md
MemNavData/PROBABILISTIC_MEMORY_LOCALIZATION_PLAN_20260806.md
MemNavData/ELEGANT_MEMORY_GRAPH_EXPERIMENT_20260806.md
MemNavData/LOCAL_MULTIGOAL_CAUSAL_AUDIT_20260806.md
MemNavData/CURRENT_BASELINE_TRAINING_V2_20260806.md
MemNavData/NOVEL_MEMORY_RESIDUAL_V2_20260807.md
```

---

## 19. 16:15 版可对外表述（由第 20 节补充）

当前可以严谨地说：

> 我们确认了长期视觉记忆对 ImageGoal 回访导航具有显著价值。在 20 场景、40 条严格
> 2-leg 闭环评测中，可靠的几何 memory route 将 frozen NavDP 的 joint SR 从 10.0%
> 提升到 47.5%，且产生 15 个配对 gain、0 loss。进一步诊断显示，旧端到端 MemNav 的
> 主要限制不是单一 retrieval loss 或 LingBot 转弯误差，而是 memory-to-action 耦合、
> Novel base-policy 负迁移、显式 no-match 缺失，以及长程 pose/utility 不确定性。我们
> 因此转向 NLSR-V2：冻结原生 NavDP，以 LingBot pose/depth graph 提供长期候选，并用
> 可校准的 candidate-set model 学习定位、pose 修正和 selective residual；不确定时严格
> 回退原生 ImageGoal。新的 600-session causal teacher 已完成审计，真实数据 objective
> 已通过梯度和 overfit smoke，但完整 feature join、新模型 checkpoint 与最终闭环结果
> 尚未完成。

当前不应声称：

- `25/40` reverse graph 已是 final；
- learned router 已完全替代 RANSAC；
- GoalSwap 已解决 goal conditioning；
- 3-leg 失败证明 LingBot long-memory 无效；
- shared final `6100/6270/6270` 已完成；
- NLSR-V2 已有新的 W&B/SR 提升。

---

## 20. 2026-08-07 晚间增量核查

本节整理今日晚间对 20-scene 2-leg、3-leg、历史训练与最新 Phase-B 任务的交叉
核对。它是本报告的最新时间快照；如果与前文的“当前运行状态”冲突，以本节为准。

### 20.1 代码、路径与时间边界

```text
个人子仓库: /home/asus/Research/Nav-graph-blind
正式运行快照: 3c53e437be03899859f16cfbdbd0951612b8dcad
远程结果根: /scratch/yz11502/Research/Nav-axis-uturn-results
核对时间: 2026-08-07 19:47 CST
```

本文档与所述代码都位于个人子仓库；没有把本轮修改写入母目录
`/home/asus/Research/Nav`。工作树中的其他 dirty 文件为用户已有内容，本次只更新本报告。

### 20.2 20-scene 2-leg 跑完时的真实基线

协议是 `start -> A Novel -> B Revisit`，20 个场景、每场景 2 条 episode：

| 指标 | Native NavDP | Geometry memory + frozen NavDP |
|---|---:|---:|
| Novel A SR | 31/40 = 77.5% | 31/40 = 77.5% |
| Revisit B SR given A | 4/31 = 12.9% | **19/31 = 61.3%** |
| Joint SR | 4/40 = 10.0% | **19/40 = 47.5%** |
| Revisit B mean final distance | 7.035 m | **2.586 m** |
| Revisit B mean SPL | 0.0350 | **0.4377** |

配对为 15 个 geometry-only gain、0 个 native-only loss，McNemar exact
`p=6.1035e-5`，Novel A false activation 为 `0/40`。这份 `19/40` 是当前最可靠的
广场景闭环 memory 基线。

31 条 eligible B 中的 12 条失败可拆成：

```text
7 条: memory route 从未激活
5 条: route 激活后仍失败
       至少 3 条是错误/弱 anchor
       其余约 2 条是长 direct point-goal 或 local-control mismatch
```

原先 5 个 audit scene 中，eligible Revisit 激活 `10/10`、成功 `8/10`；新增 15 scene
中的 21 条 eligible 只激活 `13/21`，激活后成功 `10/13`，未激活时仅 `1/8`
成功。因此 20-scene 后的第一优先级是 localization/router recall，不是继续压低
action epsilon-MSE。

### 20.3 20-scene 之后的修改时间线

| 阶段 | 直接问题 | 已实现改动 | 证据/当前状态 |
|---|---|---|---|
| Online candidate fix | raw DINO top-1 被相邻错误帧占据 | complete-history score、temporal-NMS、top-8 RANSAC、two-plan latch | 单 episode 从 4.4185 m 失败到 0.9923 m 成功；尚无新 20-scene 总分 |
| Anchor consistency | 验证后 pose query 可退回 raw top-1 | verified-anchor latch 强制绑定 pose query，cached reverify | 机械路径已通过单测与因果复测 |
| Learned localization | binary accuracy 不等于 session top-1/no-match | task-aligned co-visibility、pointwise/listwise set rank、dustbin | development 有信号，`deployment_approved=false` |
| Long direct control | 一次给较远 point-goal 有长尾 | reverse pose graph，约 1.25 m 短 subgoal | 旧运行 `19/40 -> 25/40`，但存在 DDPM noise confound |
| Causal evaluation | 不同 arm 的 FIFO/调用数/noise 不同 | shared A trace、FIFO replay、per-request seed、hash audit | strict runner 已实现；strict 2-leg 总分尚未完成 |
| LingBot-native factors | RANSAC 工程重、难学习不确定性 | cloud overlap、pose consistency/refinement、depth/scale quality | development 信号明确，暂未可完全替代 RANSAC |
| Safe residual redesign | 旧 scalar gate 同时承担 match/rank/utility | NLSR-V2：frozen NavDP + explicit no-match/rank/pose/utility/harm | 代码/数据契约已实现，新 checkpoint 尚未产生 |

关键 commit 时序：

```text
20b81f4  verify temporally diverse retrieval candidates
0e9cff2  paired top-K and conditional-C protocol
c558287  probabilistic graph ablation
c29a666  strict causal graph memory protocol
8f0ed5c  deployment-style LingBot loop factors
0bceb69  optional graph terminal loop alignment
6ef3195  shadow virtual-loop arrival diagnostics
b3f1516  safe frozen-NavDP residual contract
9ae2328  audited LingBot-native localizer trainer
a55ae5d  formal causal teacher stages
31838ef  audited Phase-B feature join
3c53e43  serialize Phase-B GPU stages through relay
```

ResidualGate、NovelGS、GateCurr 与最初 Patch router 早于 20-scene 闭环结果，不应被记作
“20-scene 之后新加的改动”。它们在后续的作用主要是提供反例和暖启动血缘
诊断，不是当前基线。

### 20.4 候选与控制路径具体如何变化

20-scene 原运行：

```text
raw DINO top-1
    -> 只验证一帧 SIFT/Essential-RANSAC
    -> 通过两次后启用 direct LingBot point
    -> frozen NavDP
```

立即修复后：

```text
complete-history DINO scores
    -> deterministic temporal-NMS
    -> 最多 8 个时间多样候选
    -> SIFT/Essential-RANSAC 逐个 fail-closed 验证
    -> 同 anchor 连续两次通过
    -> latch selected anchor + cached reverify
    -> direct/graph point-goal
    -> frozen NavDP
```

目标中的 learned 系统：

```text
DINO + patch/temporal + LingBot candidate set
    -> P(global match/no-match/ambiguous)
    -> candidate listwise rank
    -> pose residual + covariance
    -> utility/advantage + harm/coverage
    -> 不确定: byte-equivalent native fallback
    -> 高置信: graph/frontier residual -> frozen NavDP
```

之所以不再用一个 learned gate，是因为“memory 中有没有目标”、“候选中哪个正确”、
“pose 是否可靠”、“memory action 是否比 native 好”是四个不同问题。旧 gate 将它们
压成一个 scalar，并用 `1-g` 抑制 visual branch，容易同时造成 Revisit 不激活和 Novel
负迁移。

### 20.5 Reverse graph 和 virtual loop 的正确状态

Reverse graph 的目的是将长 direct point-goal 分成约 1.25 m 的历史节点。旧 20-scene
development 运行中，direct `19/40`、graph `25/40`，6 gain、0 loss，但两臂在之前
产生的 diffusion request 数不同，所以该数字不是 final。

后续已经实现 strict shared-prefix/per-request-seed runner。十场景 conditional-C 严格对照
中，direct 为 `8/10`、graph 为 `6/10`，说明 graph 在长程 C 上也可能因节点切换、
重定位和局部执行误差而退化。目前默认仍应以 verified-anchor direct point 作为主基线，
graph 作为待严格验证的控制臂。

Virtual-loop/terminal alignment 已有可选代码和 no-action arrival shadow，用于检查到达后的
目标朝向一致性。它未启用在当前 position-SR 主基线，没有被训练成 checkpoint，也没有
可对外声称的 SR 增益。

### 20.6 为什么 3-leg 的 Novel A 与 Novel B 不是同一难度

今日对 generator 和 evaluator 的逐行核对表明，`A 6/10` 与 `B|A 1/6` 不能直接解释为
“NavDP 天然不会处理第二个 goal”。A/B 存在明显的 benchmark 难度不对称：

1. A 的 start-to-A geodesic 被限制在 `3--9 m`；B 在整层随机采样，只有 `>=3 m` 的
   下界，没有上界。
2. B 被强制与 A 整段历史 `max covis < 0.1`，是更严格的完全未见目标。
3. A 的 goal image 直接取 expert leg A 最后一帧；B 使用在 B 位置单独渲染的
   `goal_1.jpg`，而 trajectory generator 不执行 terminal-yaw alignment。
4. A 从精确数据起点/起始 yaw 开始；B 从 policy 进入 A 的 1 m success radius 后的
   实际位置/yaw 开始，已经带有 on-policy 累积误差。

同一批 10 条 expert 3-leg 轨迹的帧数统计：

| | A leg | B leg |
|---|---:|---:|
| mean frames | 141.6 | 274.9 |
| median frames | 130 | 201 |
| range | 82--207 | 112--871 |

相同生成速度下，B 段总帧数是 A 的 `1.94x`；最极端的一条是 `82 -> 871`。
距离不是唯一原因，因为 112/147-frame B 也有失败，但它证明两段不可当成两次
同难度试验。

已有因果对照排除了两个简单解释：

- carry FIFO 和 B 前 hard reset 都是 `4/5`，gain/loss 互换，所以不是简单的“A FIFO
  污染 B”；
- 同一 post-A 状态下，native ImageGoal B 为 `3/5`，精确 final metric point 和 1.25 m
  geodesic point 都是 `5/5`，所以 frozen NavDP 的局部控制能力存在，缺的主要是
  困难 B 状态的高层方向。

真正能隔离 goal order 的下一个实验应是 matched order-swap：同一个目标、相同起点、
距离、路径转角和目标 yaw，分别作为第一和第二 goal；同时分开 exact-A/policy-A
与 carry/reset FIFO。在此之前，`B|A=1/6` 应被表述为“更难 B 分布上的失败”，
而不是“NavDP 第二次 goal switch 必然失效”。

### 20.7 历史已训模型的最终定位

| 模型/实验 | 已完成结果 | 当前定位 |
|---|---|---|
| `gatecurr600` | 两场景 conditional Revisit `6/20 -> 10/20`；gate accuracy 升，top-1/action/pose 未同步升 | 历史方向性证据，不再继续长训 shared decoder |
| `residualgate1000` | Novel `2/10 -> 4/10`；selector 对齐 executed prefix 后 `7/10` | 证明 visual 不应被 gate 关闭，但仍低于 native `9/10` |
| `novelgs_res1000_early40_w025` | 约 1112 steps / 1.77 epochs；mean GoalSwap gap `-0.000442`，目标 `+0.05` | objective 接通但 goal-conditioning 未学成 |
| DINO-only NLSR smoke | 16-session 100% overfit；development top-1 `67.65%` vs raw DINO `82.35%` | 证明 loss/gradient 可学，也证明 feature 不足；未批准 checkpoint |
| 20-scene geometry R0 | joint `19/40` | 当前最强可靠基线，但它不是新训练 checkpoint |

旧 Phase-B job `15440645` 在真实 artifact audit 处因 1240/1244 行、label/selection 语义漂移而
正确失败，真实 backward 没有开始。不能将它记作“训了但效果差”。

### 20.8 600-session teacher 与最新 Phase-B 状态

已完成的 causal teacher：

```text
600 sessions = 480 train + 120 development
17,845 emitted candidate rows
8,923 counterfactual rows
49,373 frozen DINO embedding inputs
189/600 shortlist-positive
345/600 strict no-match
66/600 ambiguous
```

在 shortlist 内真有 positive 的条件下，recall@1/4/8/16/32 为
`76.19/88.36/93.12/96.83/100%`。这一分布证明 explicit dustbin/no-match 是必须的，也证明
candidate top-K 有足够的上限用于 learned reranking。

当前正式任务：

| Job | 任务 | 19:47 CST 状态 |
|---:|---|---|
| `15474001` | `nlsr_pB_feat`，Phase-B LingBot feature collection/join | RUNNING `01:15:51` on `ga016` |
| `15474003` | `nlsr_pB_stage`，后续 stage relay | PENDING on dependency |

核对时 collector 进度：

```text
sessions = 85/480
rows     = 195/1098
allocated GPU memory ~= 13.6 GiB
peak allocated       ~= 16.6 GiB
peak reserved        ~= 17.0 GiB
```

该 job 当前是 feature collection，不是 model optimizer 训练，所以尚无新 checkpoint、W&B loss 或
闭环 SR。输出使用 session-atomic SQLite checkpoint、单写者 lock 和可恢复落盘。日志开始处
有一条：

```text
Failed to load pretrained weights: [Errno 2] No such file or directory: ''
```

它未导致进程退出，之后仍持续产生数值有限的 rows 与 checkpoint；当前只能定位为
非致命预警，不能在最终 audit 通过前假定所有 feature 都正确。后续必须依次通过：

```text
exact candidate/session cover
-> finite/non-constant feature audit
-> teacher/route/scale/content-hash receipt audit
-> all-head gradient preflight
-> balanced 16--32 session overfit
-> three-seed scene-grouped training
-> one-shot development
-> no-action shadow
-> paired 20-scene closed loop
```

### 20.9 截至本次更新的精确结论

已经可以说：

1. 2-leg 中可靠 memory geometry 相对 native NavDP 有显著配对收益，`4/40 -> 19/40`。
2. 剩余主要瓶颈先是 candidate/localization recall，其次是 wrong/weak anchor 与长 direct-control
   tail，不是简单的 action loss 太高。
3. 3-leg joint `0/10` 主要先被更难的 Novel B 卡住；conditional-C 显示 direct memory
   可从 native `4/10` 提高到 `7--8/10`。
4. Reverse graph 有 2-leg 开发信号，但 strict conditional-C 中低于 direct，目前不能宣称
   graph 已稳定改善长程规划。
5. 旧 GateCurr/ResidualGate/NovelGS 均不是当前应继续长训的主线；冻结 NavDP、显式
   no-match 和 selective residual 才是目前最符合证据的设计。
6. 最新 Phase-B 正在采集真实 LingBot features，尚未开始可报告的正式模型训练。

尚不能说：

- temporal top-K 已将完整 20-scene 2-leg 从 `19/40` 提高到新数字；
- reverse graph `25/40` 是 strict final；
- learned localizer 已可以无 RANSAC 上线；
- Phase-B 已训出新 checkpoint；
- NLSR-V2 已在 W&B 或 Habitat 闭环中超过 R0。
