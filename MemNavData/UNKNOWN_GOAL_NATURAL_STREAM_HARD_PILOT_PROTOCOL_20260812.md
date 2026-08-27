# Unknown-goal Natural-stream：困难 Strata Pilot 协议

日期：2026-08-12（CST）  
状态：目标与分析代码已在读取 natural-stream 结果前冻结。

## 1. 问题

正式 overlay 预检证明采集与因果 teacher 可用，但 `17DRP5sb8fy` 的有效 C 样本过于容易：
DINO top-1 和现役 hard geometry 都已正确。该 pilot 只回答：

> train-only 静态审计中发现的 DINO/RANSAC 困难，是否会在原生 NavDP 自然运动的连续
> planning stream 中持续？

它不是训练、闭环方法比较或 SR 声明。

## 2. 冻结输入

- causal/Phase-B rows SHA256：
  `193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b`
- geometry evidence SHA256：
  `ef9374b31a63e4012525e0e68e7749b94569b6b263162143ee516bd4b8260463`
- natural-stream source receipt SHA256：
  `9c7a5140fcf0623bb3187235a30245fcdac362506b956c50fb13ad03b99adc09`
- collection launcher SHA256：
  `23b95fe7dbc385714670b88c6cc768578e5dbc436dfef292520b33738a224594`
- frozen analysis：`analyze_unknown_goal_natural_stream_hard_pilot.py`
  SHA256 `b3dcf45b29fd5d31e450348017f389ce9ce5c013ae5276a06cfe066628b44853`
- teacher thresholds：positive `0.5`、strict negative `0.2`；不在 pilot 上调节；
- reconstructed deployed geometry：matches `20`、inliers `12`、ratio `0.5`。

development、consumed 20-scene pool、blind 不读取。

## 3. 预先选择的四条 episode

| Scene / episode | 静态 factual B | 静态 factual C |
|---|---|---|
| `1pXnuDYAj8r / episode_0001` | strict no-match，无 hard pass | positive rank 3；top-1 wrong；positive geometry miss |
| `JeFG25nYj2p / episode_0001` | strict no-match，无 hard pass | positive rank 4；top-1 ambiguous；positive geometry miss |
| `YVUC4YcDtcY / episode_0000` | strict no-match；rank-7 false hard support | positive rank 0；positive geometry miss |
| `5ZKStnWn8Zo / episode_0001` | strict no-match；rank-4 false hard support | easy positive control；top-1 与 geometry 均正确 |

已完成的 `17DRP5sb8fy / episode_0001` 保留为独立 easy pipeline control，不重新运行。

## 4. 采集契约

- 每条独立 one-episode job，可并行；
- 原生 NavDP ImageGoal 动作流；
- router/adapter 阈值不可达，takeover 必须为 0；
- 每 plan 记录最多 top-8 的 DINO 与 raw RANSAC evidence；
- teacher 仅用 recorded causal pose + offline Habitat depth；
- A/B 失败导致的下游 C 必须标为 censored，不能当 negative；
- contract、输入与最终 artifact hashes 任一失败即停止该条。

## 5. 冻结读数

按 plan 报告：

1. teacher positive / strict-negative / ambiguous；
2. DINO top-1 是否命中 positive；
3. 现役 geometry first-hard-pass 是否命中 positive；
4. positive 但 geometry miss、positive 但 top-1 miss；
5. strict-negative 上 geometry false support；
6. 同一 positive/false-support anchor 跨 plan 的出现次数与连续 persistence；
7. A/B/C reachability 与因果 censoring。

A/B/C 只用于结果分层，禁止进入 learned decision feature。

## 6. 判定边界

- 若 C 未到达：记为 inconclusive/censored，不把缺记录解释成 classifier 成功或失败；
- 若静态困难在自然流中立即消失：优先考虑“等待/自然视角改善”，不训练复杂 temporal head；
- 若同一困难 positive 在多个 plan 中持续，而连续 evidence 提供额外区分：才授权扩大
  train-scene shadow collection与 scene-grouped temporal OOF；
- 若困难持续但现有 deployable evidence 没有区分信息：停止该路线，不以模型容量掩盖
  observability 缺失；
- 无论 pilot 结果如何，均不直接授权闭环或论文声明。

