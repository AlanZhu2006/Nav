# REVISIT Geometry Expert：全量结果与架构决策

日期：2026-08-11（CST）  
状态：40-scene train-only 资格审计完成；不构成 deployment 或论文确认。

> **2026-08-11 架构口径修正：** 当前 2-leg/3-leg benchmark 的 metadata 明确给出
> Novel/Revisit goal kind。因而下面的全局 `Semantic existence expert E` 只适用于未来
> “部署时 goal kind 未知”的扩展任务，不是当前 benchmark 的必要模块。当前主线改为：
> Novel leg 只用原生 NavDP；已知 Revisit leg 才使用 memory。`goal_b_t0` 的 34.2% 是
> RANSAC 全局路由资格的负对照，不是 Novel branch 的组成部分。下一项冻结实验见
> `REVISIT_KNOWN_PHASE_ABLATION_PROTOCOL_20260811.md`。
>
> **后续结果：** 已知 Revisit direct 在同进程配对中把 conditional B 从 `21/29` 提到
> `26/29`，但为 `+6/-1, p=0.125`，未满足零损失替换门。更重要的是，唯一回退的 anchor
> 实际接近目标轨迹，错误暴露在 place-to-action 链而非单纯检索层。完整结果与修正架构见
> `REVISIT_KNOWN_PHASE_ABLATION_RESULT_20260811.md`；本文第 6–8 节的 existence-expert
> 方案仅保留为未知 goal-kind 扩展，不再是当前 benchmark 的下一主线。

## 1. 先固定当前最强 baseline

当前证据最强的 REVISIT 方法仍是 geometry router，而不是 X-NavDP 或 learned gate：

| 同机同进程 R0 | Novel A | Revisit B given A | joint |
|---|---:|---:|---:|
| native | 26/40 | 3/26 | 3/40 |
| geometry router + mixed NavDP | 26/40 | **20/26** | **20/40** |

geometry vs native 为 `+17/-0`，exact McNemar `p=1.5259e-5`。固定 2.5 m bearing 与
metric waypoint 都是 `20/26`；base PointGoal、mixed、official X+MPC 是
`20/26、20/26、21/26`，X 的单条净增益不显著。因此下一杠杆是 activation / anchor，
不是换 controller、回归更准距离或再叠 bearing head。

## 2. 正式任务收据

- Slurm job：`15583693`，`COMPLETED`，exit `0:0`；
- partition/QoS：`cpu_short/cpu48`；16 CPU，64 GB；
- elapsed：1m42s；evidence extraction 70.53s；
- 输入：40 train scenes、480 sessions、14,172 causal candidates；
- labels：1,509 positive、11,279 negative、1,384 ignore；
- development/blind：未读取；
- query/candidate：逐文件按 causal teacher SHA256 验证；
- 单测：6/6；
- 产物四项 `sha256sum -c` 全部通过。

HPC 产物根：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  revisit_geometry_expert_20260811/train40_ransac5_v1/
```

本机摘要：

```text
.diagnostics/revisit_geometry_expert_20260811/
  evidence_report.json
  analysis_report.json
  FINAL_SHA256SUMS
```

冻结协议：`REVISIT_GEOMETRY_EXPERT_PROTOCOL_20260811.md`。

## 3. 全量主结果

### 3.1 RANSAC 不是可靠的全局 binary expert

在 task extreme labels 上：

| 指标 | 结果 |
|---|---:|
| stable-support precision | **76.20%** |
| positive candidate recall | **58.78%** |
| hard-rejected positives | 622，跨 37 scenes |
| positive sessions activated | 124/155 |
| first-pass correct anchor | 102/155 |
| strict no-match false activation | 67/281 |

所以预注册 Gate A 没过：把任意 RANSAC pass 当“目标在 memory 中”的正证据并不够安全；
把 reject 当负证据又会漏掉大量真实 positives。

### 3.2 但连续 geometry evidence 确实有信息

五折 scene-grouped OOF：

| 指标 | DINO | geometry | DINO + geometry |
|---|---:|---:|---:|
| candidate ROC-AUC | 0.8395 | 0.8802 | **0.9041** |
| episode 内 session-macro AUC | 0.8123 | 0.8884 | **0.9081** |
| session existence ROC-AUC | 0.8854 | 0.8856 | **0.9146** |
| positive-session top-1 | **116/155** | 107/155 | 113/155 |

关键不是“geometry 无用”，而是：它显著提高 pairwise / existence 判别，却没有被 pointwise
logistic objective 转成正确 top-1。fusion 对 DINO 为 `+17/-20`，exact McNemar
`p=0.743`，scene-cluster CI `[-9.86,+7.29] pp`；预注册 Gate B 因而失败。

这与 Phase-B 的历史结果同构：candidate AUC 会升，最终 anchor/activation 不一定升。

### 3.3 五次 RANSAC seed 完全一致

14,172 rows 中：

- `unstable=0`；
- `inlier_std_nonzero=0`；
- 状态为 stable support 1,791、estimable reject 7,649、insufficient matches 4,728、
  insufficient features 4。

因此当前 OpenCV 路径的主要不确定性不是随机 seed，而是**证据是否存在及证据对应什么
latent**。运行时没有理由做五次 RANSAC；一次即可。

## 4. 最关键的分层：RANSAC 测的是 place，不是 goal-surface existence

按 causal state：

| state | 语义 | pass precision | positive recall | positive session support | strict false support |
|---|---|---:|---:|---:|---:|
| `goal_b_t0` | Novel 起点 | **34.2%** | 44.4% | 9/10 | 32/143 |
| `goal_b_midpoint_t1` | Novel 途中 | **50.0%** | 35.9% | 26/40 | 28/102 |
| `goal_c_t0` | 真正 Revisit | **90.9%** | 66.3% | 89/105 | 7/36 |

这解释了表面矛盾：

- RANSAC 在真正 revisit 上是高精度、有限召回的几何 expert；
- 在 novel/counterfactual 上，它会匹配同一房间、墙面或背景，虽然目标表面没有被观察过；
- 因而“几何一致”是 place consistency，不等于 task-aligned memory existence。

当前 geometry router 在闭环中仍然很强，是因为实际 Revisit B 正好处于它擅长的条件分布；
但若让它同时判断 Novel/No-match，它的语义就错了。

## 5. 候选宽度解释了 top-K 零结果

RANSAC support 随 DINO rank 变宽而失去纯度：

| geometry 预算 | pass precision | pass-priority top-1 | vs DINO | strict false support |
|---|---:|---:|---:|---:|
| top-1 | 91.1% | 116/155 | +0/-0 | 5/281 |
| top-2 | **92.7%** | 117/155 | +3/-2 | 7/281 |
| top-4 | 91.2% | 117/155 | +3/-2 | 11/281 |
| top-8 | 88.1% | 119/155 | +5/-2 | 26/281 |
| top-32 | 76.2% | 120/155 | +7/-3 | 67/281 |

扩 K 会换来少量 top-1 gain，同时快速增加 no-match 假 support；所有 paired gain 均不显著。
这与已有 40-episode top-K 闭环 `18/40 vs 18/40, p=1` 一致。不能继续靠“多验几个候选”
优化 baseline。

## 6. 未知 goal-kind 扩展架构：Factorized Revisit Posterior

RANSAC 应该是 expert，但角色必须条件化。最终不是两个 action controller，也不是两个 scalar
gate，而是一个显式分解的 episodic posterior：

```text
goal + causal memory
        |
        +--> Semantic existence expert E
        |      p(memory contains goal surface)
        |      explicit dustbin / no-match
        |
        +--> Conditional geometry expert G
               top-2 DINO candidates only
               stable RANSAC support / insufficient=unknown
               candidate place-consistency likelihood

P(no-match) = 1 - p_exist
P(anchor=i) = p_exist * softmax(rank_logit_i + log L_geo_i)
        |
LingBot raw relative pose (不学 bearing)
        |
fixed 2.5 m directional residual
        |
frozen mixed NavDP executor
```

### Expert E：只回答“memory 里有没有”

- 输入是候选集合，而不是某个候选的最大 scalar；
- 使用 frozen DINO / Phase-B task features、set margin、候选一致性和显式 dustbin；
- loss 是 session-level existence BCE；positive 与 strict no-match，ambiguous ignore；
- operating point 用 scene-grouped nested OOF / risk control，不再拿 pointwise score 的固定 0.5。

### Expert G：只回答“给定存在，哪个 anchor 几何可信”

- 只验证 top-2，理由是独立闭环 top-K null、R0 selected rank max 2，以及本次 top-2
  precision 92.7%；
- stable pass 是 likelihood boost，不是绕过 E 的强制 activation；
- insufficient matches/features 是 unknown，即 `L_geo=1`；
- estimable reject 在新模型通过前也不能硬置零；
- conditional rank loss 只在 positive sessions 上做 listwise marginal NLL，不能再用 pointwise
  classification 代替 top-1 目标。

### 为什么这比“learned + RANSAC 两票制”更好

两票制无法表达本次发现：同一个 RANSAC pass 在 Revisit C 是 90.9% precision，在 Novel B
却只有 34–50%。因子化 posterior 明确区分：E 决定 latent 是否存在，G 只在该条件下提供
anchor likelihood。它既保留当前唯一显著的可部署结果，也解释并修复其 residual。

## 7. 下一实验（冻结顺序）

### P0：离线 Factorized OOF，不再长时间提 feature

现有 14,172-row evidence 已足够；再与 1,098-row Phase-B task feature 表按
`(session_id, candidate_path)` 对齐：

1. 单独训练 session existence head；
2. 单独训练 positive-session listwise anchor head；
3. geometry 只对 deployment top-2 提供 support/unknown likelihood；
4. 5 folds × 3 seeds，scene-grouped；development 不用。

必须同时报告：

- existence ROC/AP/Brier/ECE 与最差 scene；
- nested-OOF frozen operating point 的 false activation / miss；
- anchor top-1 相对 DINO 的 paired wins/losses；
- joint localization，而不是 candidate AUC。

停止条件：anchor top-1 不优于 DINO，或 joint localization 不优于公平校准的 DINO，就不进
闭环。

### P1：planning-stream shadow

只记录 posterior，不控制机器人。检验同一 temporal cluster 是否跨 plan 持续；learned-only
override 必须通过连续证据累积，不能单步越过 RANSAC unknown。

### P2：三臂闭环

结构和 operating point 冻结后，在 consumed 20-scene pool 同机同进程运行：

1. native；
2. 当前 hard geometry router；
3. factorized expert router。

主要终点仍是 conditional Revisit SR 和 paired `+/-`；通过才进入 fresh non-blind confirmation。

## 8. 现在明确不做

- 不训练 bearing head；正确 anchor 条件下 LingBot raw direction 已约 89–91% 落在 ±30°；
- 不继续换 PointGoal / X-NavDP controller；当前空间至多一条且不显著；
- 不扩大候选 K；已同时被 40-episode 闭环与本次 precision-risk 曲线否定；
- 不让 RANSAC hard pass 单独决定 activation；
- 不把 RANSAC reject 当 task negative；
- 不再用 candidate AUC 作为进入闭环的门。

## 9. 结论边界

以下四点必须和主结果一起报告：

1. 本次 `false positive` 是相对 goal-surface co-visibility teacher；它不自动等于闭环 harmful。
   一个只匹配到目标附近背景的 anchor 仍可能给 NavDP 有用的方向，当前 geometry router
   `+17/-0` 正是不能把二者混同的理由。
2. 480 sessions 来自冻结 expert trajectories 的六个 causal state/variant，不是在线 planner
   每一步的真实分布；planning-stream persistence 仍需 shadow collection。
3. scene-grouped OOF 防止同 scene 直接泄漏，但这 40 scenes 已参与本次架构选择；它们只能做
   internal development，不能再充当论文确认集。
4. top-2 是根据既有闭环 top-K null、R0 selected-rank 以及本次风险曲线选择的下一协议；其
   92.7% precision 是 post-hoc architecture evidence，必须在冻结后的 fresh pool 重验。

一句话结论：**保留 RANSAC，但把它从“全局裁判”降为“存在条件下的几何 expert”；真正要
学习和校准的是 memory existence，真正要优化的是 conditional listwise anchor，而不是方向
或 controller。**
