# Unknown-goal Natural-stream：困难 Pilot 结果

日期：2026-08-12（CST）  
状态：定向 pilot 已结束；支持停止扩大“仅靠时序一致性判断 Novel/Revisit”的路线。该结果是机制诊断，**不是闭环方法或 SR 结果**。

## 1. 先澄清：这次用了 3-leg，但没有在优化 3-leg SR

每条真实 rollout 的结构是：

```text
Leg A：Novel，建立历史
  -> Leg B：Novel / no-match，检查是否会把旧历史误认成当前目标
  -> Leg C：Revisit，检查是否能找回真实历史 anchor
```

三个 leg 的动作均由冻结的 native NavDP 产生；memory/router 只旁路记录 top-8 的
DINO/RANSAC 证据，`router takeover = adapter takeover = 0`。A 或 B 失败后，下游 leg 被严格
记为 censored，不能当作 negative。

因此 3-leg 在这里是一个**诊断容器**：同一自然轨迹里同时提供 B 的 no-match 负对照和 C 的
Revisit 正例。它不等于当前主方法必须是 3-leg，也不授权比较 joint SR。若 benchmark 已提供
Novel/Revisit phase，主 Revisit 方法仍应在 known-Revisit / 2-leg 配对协议上确认。

## 2. 冻结与完整性

- 四条 hard target 与判据在读取 natural-stream 结果前冻结；协议 SHA256：
  `298bc36c323b0c7897240a980100e2518d880144e165e337c94d08420f38d8b5`；
- frozen analyzer SHA256：
  `b3dcf45b29fd5d31e450348017f389ce9ce5c013ae5276a06cfe066628b44853`，与协议记录一致；
- geometry 只按现役部署阈值重建：matches `>=20`、inliers `>=12`、ratio `>=0.5`；
- teacher 只使用 recorded causal pose 与离线 Habitat depth，positive `>=0.5`、strict negative
  `<=0.2`，本轮没有调阈值；
- development、consumed 20-scene pool 和 blind 均未用于选择或判定；
- 四个有正式 manifest 的 run，其 input receipt、shadow report、preflight report 和 teacher 共
  `16/16` 个文件本地复算 SHA256 全通过；
- 两份合并 audit 的 canonical content self-hash 均独立复算通过：
  `735e3ddb...569fae`（hard）与 `36fc8f21...c4f62a`（hard + easy control）。

`1pXnuDYAj8r` 的 rollout 在 A 失败，因此没有 memory candidate/teacher artifact。旧 summarizer
错误地把“因果截断后候选为零”当成 contract failure；修复只允许显式 censored 的零候选 run，
不改 rollout、阈值或 frozen analyzer。修复后该 run 的 post-hoc contract 通过，但仍严格记为
inconclusive，不放入 hard evidence 合并表。

## 3. 运行结果

| Target | Native A/B/C | natural plans（有候选） | 关键观察 |
|---|---:|---:|---|
| `1pXnuDYAj8r/ep1` | 0 / censored / censored | 48（0） | A 失败；目标困难 C 未被观察，inconclusive |
| `JeFG25nYj2p/ep1` | 1 / 0 / censored | 90（63） | B 的 strict-negative false geometry support 连续持续 63 plans |
| `YVUC4YcDtcY/ep0` | 1 / 1 / 0 | 92（65） | C 有 35 个 positive plans；geometry 连续 miss 35/35，DINO top-1 35/35 命中 positive |
| `5ZKStnWn8Zo/ep1` | 1 / 0 / censored | 75（39） | B 有 39 个 strict-negative plans，但静态 false pass 在自然流中消失 |
| `17DRP5sb8fy/ep1`（easy control） | 1 / 1 / 0 | 61（51） | C 的 20 个 positive plans 中 geometry 与 DINO top-1 均 20/20 正确 |

Hard pilot 中有 teacher evidence 的三个 run 合计：

| 读数 | 数量 |
|---|---:|
| natural plans | 257 |
| candidate plans / trials | 167 / 1,332 |
| positive-support / strict-negative plans | 35 / 132 |
| positive 上 geometry correct / miss | 0 / 35 |
| positive 上 DINO top-1 correct / miss | 35 / 0 |
| strict-negative 上 geometry false support | 63 |
| frozen residual-opportunity plans | 98 |

加入独立 easy control 后：318 plans、218 candidate plans、1,523 trials；positive-support 55、
strict-negative 163；geometry correct/miss 为 20/35；DINO top-1 positive hit 为 55/55；geometry
false support 为 63。

这些 plan 是同一 episode 内高度自相关的连续观测，**不能把 318 当成独立样本量**。有效范围
只有四条带 teacher 的 episode（其中 hard positive 主要来自 `YVUC4YcDtcY`），所以 `55/55`
不能写成“DINO 已解决 unknown-goal”。

## 4. 最硬的机制结论

### 4.1 “等两帧 / 连续两次一致就激活”不是安全解法

两个相反故障都能长时间稳定：

1. `JeFG25nYj2p` 的 Novel B 中，同一个错误历史 anchor（A 的 frame 158）连续 `63/63`
   plans 通过 RANSAC hard gate。其单 plan 证据为 DINO `0.8596544`、matches `28`、inliers
   `17`、ratio `0.6071`，但 task co-visibility label 为 `0.0`。时间 latch 会把这一错误变得
   更自信，而不是排除它。
2. `YVUC4YcDtcY` 的 Revisit C 中，真实 positive anchor 连续出现 `35/35` plans，但现役
   geometry 同样连续 miss `35/35`。等待自然视角改善没有救回它，RANSAC hard veto 会持续
   丢掉正确检索。

因此问题不是“单帧太抖”。简单 persistence、hysteresis 或两帧投票没有新增可辨识信息：它既
会锁住稳定假阳性，也无法穿过稳定假阴性。

### 4.2 RANSAC 不适合继续担任统一的存在性硬门

本 pilot 不是对 RANSAC 总体 precision/recall 的统计估计，但它给出了两个明确反例：

- 局部几何一致不等于该 anchor 对当前 goal 的任务相关支持；
- 局部几何失败也不等于最相似 anchor 错误。

这与独立的 fresh known-Revisit 160-episode 结果一致：绕过 RANSAC veto、直接使用 raw-DINO
top-1 后，conditional B 从 `93/118` 提升到 `109/118`，joint 从 `93/160` 提升到
`109/160`；配对 `+20/-4`，exact McNemar `p=0.0015438795`，scene-cluster joint RD 95% CI
为 `[+5,+15.625] pp`。

更准确的架构角色应是：DINO/任务证据负责“选哪个 anchor”，RANSAC 作为可选的正支持、姿态
质量或 abstain 证据；不再让它一票否决 Revisit 检索。

### 4.3 DINO 排序值得保留，但 unknown-goal existence 尚未解决

观察到的 positive plans 上 DINO top-1 全部正确，是支持“DINO 排序比 hard geometry veto 更
合适”的方向性证据；它只覆盖一个 hard-positive episode 加一个 easy control。Novel B 上的
top-1 本来就必然是某个历史 frame，排序正确性本身不能回答“当前 goal 是否存在于历史”。

所以 unknown-goal 的缺口不是再训一个更强 anchor ranker，而是一个**role-free existence / task
relevance** 判定：何时允许使用任何历史。当前证据不足以证明这个量能从现有 DINO/RANSAC
时序里可靠学出。

## 5. 对当前架构的决策

当前主线应回到有显著闭环证据的 known-Revisit residual，而不是扩大 unknown-goal 3-leg
temporal router：

```text
benchmark phase = Novel
  -> frozen native NavDP；memory 不接管

benchmark phase = Revisit
  -> task-aligned / DINO top-1 检索 anchor
  -> RANSAC 只作支持、位姿质量或 abstain 证据，不作统一 hard veto
  -> 可执行的相对方向/残差交给冻结 NavDP
  -> 不确定或不可执行时退回 native
```

这不是声称最终 residual 已完成。fresh confirmation 仍有 `4` 条 direct 相对 geometry 的配对
损失；下一步应对这四条做预注册的**动作可执行性/安全 abstention 归因**，而不是再发明一个
Novel/Revisit 门控器或直接长时 eval。

如果未来确实要求在没有 phase metadata 的部署中自动区分 unknown goal-kind，则需要新的、跨
scene 可观察的 existence 信号，并先通过 scene-grouped OOF 风险门。当前 pilot 明确否定的是
“仅靠现有证据的时间一致性就够了”，不是证明所有 unknown-goal router 都不可能。

## 6. 决定与下一步

1. **停止扩大本 3-leg temporal pilot**：不提交 6–8 小时训练，也不做闭环 SR；当前已获得其
   分叉点答案。
2. **主线用 known-Revisit 配对协议**：保留 native Novel，把直接检索带来的 `+20/-4` 作为
   当前最强 learned-free 改进起点。
3. **先审计四条损失再设计 residual expert**：只寻找能在不牺牲 20 条 gains 的前提下识别
   4 条 harmful takeover 的可部署 actionability 信号；先做已有 artifact 的反事实分析，避免
   重复长评测。
4. **RANSAC 降级而非删除**：验证它能否贡献相对姿态质量、局部朝向可信度或 abstention；不
   再以 hard match/no-match gate 使用。
5. 只有 train-scene nested OOF 风险门通过，才授权 fresh paired closed-loop confirmation。

## 7. 基础设施注记

- 初次 `1pXnuDYAj8r` job `15628740` 落到 H200 后，MemNav `memory_step` 约需 `19–21 s`，在
  6m51s 后被精确取消；partial artifacts 保留，不进入结果。
- 排除 H200 后重跑 job `15628873`，rollout 有效但 A 失败；Slurm 的 FAILED 仅来自旧
  zero-candidate summarizer 假设，不是 rollout 或数据损坏。
- 其余 hard jobs：`15628741`（JeFG，A100，4m28s）、`15628742`（YVUC，L40S，5m53s）、
  `15628743`（5ZK，A100，3m53s）均完成。

本地审计产物位于：
`.diagnostics/unknown_goal_natural_stream_hard_pilot_20260812/`。
