# REVISIT V2：Factorized Dual-Expert Revisit

日期：2026-08-10（CST）  
状态：**已暂停，不再作为当前冻结架构。** 2026-08-11 的修正是：在证明
RANSAC 对 task co-visibility 的条件信息之前，不能预先决定让它退出 runtime，也不能继续
把它当 hard veto。当前资格审计、预注册判据与正式 HPC 任务见
`REVISIT_GEOMETRY_EXPERT_PROTOCOL_20260811.md`。本文件以下内容仅保留为历史提案。

## 1. 修正

`verified_bearing_v1` 只把 controller 边界变干净了；它仍依赖
`DINO -> SIFT/essential-RANSAC -> LingBot pose`，所以是可靠 baseline，不应被包装成最终
方法。V2 的目标是让 SIFT/RANSAC 退出部署路径，同时不重犯 learned gate 的校准错误。

不采用“Novel controller + Revisit controller”的两个 action experts。现有证据显示：

- R0 六条 residual failure 中四条是 router inactive，只有两条是 active controller failure；
- mixed/base/X 在 R2 为 `20/26, 20/26, 21/26`；换 controller 的净空间很小；
- 旧 shared decoder、learned activation 与 GLP 都没有把离线分数转成闭环增益。

因此两个 expert 应分解 **memory inference**，共享同一个 mixed NavDP executor。

## 2. 两个 expert

### Expert L：Loop / Existence expert

问题：目标是否存在于 memory；若存在，是哪个 temporal cluster/anchor？

输入：

- frozen DINO/LingBot goal token；
- temporal-diverse memory candidate tokens；
- candidate 的局部 patch correlation 与邻域支持。

输出：

- listwise anchor logits；
- 显式 dustbin / no-match probability；
- task-aligned covisibility estimate。

监督不是 SIFT pass/fail，而是离线 goal-surface covisibility：`>=0.5` positive、`<=0.1`
negative，中间 ignore。它直接回答“这个 anchor 是否看到了目标表面”。

### Expert R：Bearing Reliability expert

问题：给定 anchor 后，LingBot 提议的 bearing 是否可信；不从零重新学习方向。

输入：

- frozen LingBot raw bearing；
- anchor 及相邻 temporal offsets 的 pose hypotheses；
- per-hypothesis direction、尺度、depth confidence、patch/cloud consistency；
- current planning state 的因果 pose/history tokens。

输出：

- bearing-valid probability；
- raw bearing 的小 circular residual；
- von-Mises concentration或等价 circular uncertainty。

这样 Expert R 学的是“何时信 LingBot”，不是把一个已经较准的几何量重新回归一遍。

## 3. 融合不是另一个黑盒 gate

对 candidate `i`：

```text
p_loop(i), p_dustbin = Expert_L(goal, memory)
q_i(theta), r_i      = Expert_R(current, goal, candidate_i)
w_i                  = p_loop(i) * r_i
q(theta)             = sum_i normalize(w_i) * q_i(theta)
```

最终 confidence 由三个可审计量构成：

1. `1 - p_dustbin`：memory 中是否存在目标；
2. candidate posterior 是否集中；
3. circular bearing posterior 的 resultant / inter-anchor agreement。

scene-grouped OOF 只用于冻结一个风险约束的 abstention operating point，例如约束错误激活
上界，而不是最大化 accuracy。低置信度直接回退 native NavDP，不再 fallback 到 RANSAC。

因此部署路径为：

```text
DINO high-recall retrieval
        |
  Expert L: anchor/no-match
        |
  Expert R: bearing validity + uncertainty
        |
 circular posterior / risk-controlled abstain
        |
 fixed 2.5 m bearing residual
        |
 existing mixed NavDP
```

SIFT/RANSAC 仅保留为历史 baseline 和离线误差分析，不进入 V2 runtime，也不作为训练标签。

## 4. 现有数据是否支持

对 repaired Phase-B rows 的新审计：

| split | scenes | sessions | positive sessions | strict no-match | positive rows |
|---|---:|---:|---:|---:|---:|
| train | 40 | 480 | 155 | 281 | 224 |
| consumed dev | 10 | 120 | 32 | 76 | 93 |

正确 candidate 条件下的 LingBot raw direction：

| split | median error | within ±30° |
|---|---:|---:|
| train | 3.65° | 88.8% |
| consumed dev | 2.42° | 91.4% |

这强烈支持“校准 existing bearing”，不支持再训练一个从 RGB 端到端猜方向的 head。

但数据还不够直接长训：

- train positive bearing 为 front/side/rear=`167/41/16`，rear 明显稀疏；
- consumed dev 为 `77/15/1`，不能评估 deep-rear generalization；
- 当前 CSV 的 translation/rotation dispersion coverage 为 0，Expert R 所需的多 hypothesis
  一致性根本没有被收集。

所以现在直接跑八小时模型，很可能只得到另一个“永远相信前方”的 expert。

## 5. 正确执行顺序

### D0：补数据，不训练

在 40 个 train scenes 内，为每个候选收集 temporal offsets 的 LingBot pose hypotheses：

- 保存每个 hypothesis 的 raw bearing、scale、depth confidence、patch/cloud evidence；
- labels 使用 causal GT relative bearing/covisibility，仅用于训练；
- rear 样本必须来自 Habitat 真实重渲染或真实 rollout state，禁止旋转 RGB 像素伪造；
- development 不再参与结构、阈值或 early stopping。

D0 的通过条件是：dispersion 字段非空、scene 分布可审计、rear 有足够的 scene coverage，且
所有输入在部署时因果可得。

### D1：分别训练两个 expert

- frozen backbone；
- Expert L 使用 listwise anchor+dustbin objective；
- Expert R 使用 validity loss + circular residual NLL；
- scene-grouped OOF；先各自通过，不联合微调；
- 禁止 shared decoder、action loss 和 learned scalar activation score。

### D2：shadow 融合

在完整 planning stream 上只记录 posterior，不控制机器人。验收单位是 session/episode：

- no-match false activation；
- correct-anchor recall；
- accepted bearing 的 ±30° precision/coverage；
- scene-macro 与最差场景；
- 相对 geometry baseline 的 gains/losses，而不是 pooled candidate AUC。

### D3：闭环

先在 consumed pool 做机制配对但不作论文确认；所有结构和 operating point 冻结后，再做
fresh non-blind confirmation。主三臂：native、RANSAC baseline、dual-expert V2。blind 最后
一次性打开。

## 6. 真正的创新点

不是“用两个网络替换 SIFT”，而是把 episodic navigation 写成一个可验证的后验分解：

> 一个 expert 判断记忆里有没有/在哪里，另一个 expert 判断由该记忆诱导的方向是否可信；
> 二者只产生带不确定性的 directional residual，原导航策略保持冻结。

若最终闭环成立，论文主张会比 RANSAC router 更完整：**memory as a risk-controlled
direction posterior，而不是 place-match heuristic 或第二套 controller。**
