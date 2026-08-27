# Unknown-goal Top-8 Set Uncertainty：冻结协议

日期：2026-08-11（CST）  
状态：在查看该实验结果前冻结。只允许 40 train scenes；不读取 development、consumed
20-scene pool 或 blind。

## 1. 问题

上一轮 factorized support 使用 deployment top-2 的集合特征。它比公平 DINO 更安全，但
在真正存在 memory support 时过度 abstain，三个 seeds 的 correct anchor 为 85/89/88，
低于 hard geometry 的 93。

最新 VPR uncertainty 工作指出，完整 retrieval score distribution 可以估计候选歧义。这里
检验一个严格隔离的问题：

> top-8 是否可以只作为 uncertainty observation，提高 unknown-goal existence；而不把更多
> candidate 交给导航或改变 anchor ranker？

这与已有 top-K closed-loop null 不矛盾：旧实验测试“扩大可执行候选链”；本实验只测试
“集合分布是否提供置信信息”。

## 2. 唯一改动

基线 F2 保持上一轮全部设计：

- top-2 set existence；
- top-2 conditional pairwise anchor ranker；
- nested scene-OOF risk-matched threshold。

新系统 F8 只在 existence 输入中增加、且预先固定如下 top-8 summary：

- DINO：mean、std、top1-top4、top1-top8、top2-tail-mean；
- matches：`log1p` mean/std/max；
- inliers：`log1p` mean/std/max；
- inlier ratio：mean/std/max；
- hard-pass count 与 first-pass rank（无 pass 记为 8）；
- essential-available rate、pose-recovered rate、geometry pass rate：各自 mean/max。

F8 不读取 `label`、`covisibility`、phase、state name 或 goal role。conditional ranker、候选
top-2、模型族、正则、fold、seed、校准器、risk budget 均不变。

## 3. 数据与公平性

- geometry table 的 480 sessions 均至少有 ranks 0--7；
- top-8 仅是已有 memory 检索/验证结果，不需要主动 360° glance；
- F2 直接读取已冻结正式产物，不重新选择其结果；
- F8 使用与 F2 相同的 outer/inner folds 和 seeds 20260811/12/13；
- hard geometry H 仍是 top-8、DINO floor 0.88、first hard pass 的 decision-unit reference。

## 4. 预注册判定

F8 只有在三个 seeds 全部满足时才通过：

1. strict-no-match false activation ≤ H；
2. wrong-anchor activation ≤ H；
3. correct-anchor activation > H；
4. correct-anchor activation > 同 seed F2；
5. correct-support decisions > 同 seed F2。

若通过：top-8 distribution features 进入 natural-stream temporal collector，但仍不授权闭环。

若未通过：停止扩充单时刻 feature，直接进入 natural-stream temporal evidence collection；
不得继续试 top-4/top-16、softmax temperature、MLP 宽度或 post-hoc threshold。

## 5. 输出边界

这是已被多次查看的 train scenes 上的模型开发实验，不是独立确认。任何 AUC/coverage 提升
都不能写成论文最终结果，也不能覆盖 N=40 geometry memory 的闭环结论。
