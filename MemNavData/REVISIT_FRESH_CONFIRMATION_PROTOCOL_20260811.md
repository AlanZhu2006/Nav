# REVISIT：raw-DINO direct 对 geometry hard gate 的新 episode 确认

日期：2026-08-11（CST）  
状态：v2 协议冻结；待 HPC 生成与评测。v1 在任何 arm evaluation 前因两个小场景的固定
生成抽样预算耗尽而 fail-closed；v2 仅增加候选抽样预算，不改变 episode 接受条件或统计规则。

## 唯一主问题

在 benchmark 已声明 Goal-B 为 Revisit 时，去掉 RANSAC/SIFT **硬激活门**、直接使用
raw-DINO top-1，是否在全新 episode 上稳定优于当前 `memory_geometry`？

这不是再次优化旧 40 条。旧池只得到 `+6/-1, p=0.125`，不足以决定架构。本实验在同一
20 个、与 50 个训练场景不重叠的 scene cluster 中重新生成 8 条 episode，共 160 条。
因此它是 **fresh-episode replication**，不是 fresh-scene 或 blind confirmation。

## 因果配对

每条 episode 只运行一次原生 ImageGoal Novel-A，冻结逐帧 RGB 哈希、位姿、动作和逐 plan
seed。随后三臂逐像素重放这条 A trace：

1. `geometry_router`：raw-DINO 候选 + 固定 SIFT/RANSAC hard gate；
2. `known_revisit_direct`：同一 raw-DINO top-1、同一 LingBot pose、同一 mixed NavDP，
   唯一去掉 geometry veto；
3. `native`：原生 ImageGoal NavDP，作为方法增益参照。

三臂共享 scene、episode、Goal-A 成败、Goal-B 图像、起点状态、NavDP FIFO 重放和逐 plan
diffusion seed。scene index 轮换三臂的 6 种顺序。不能加入 fixed-radius、front-support、
learned ranker、X-NavDP 或新阈值。

## 数据身份

- 场景顺序与旧 20-scene manifest 完全一致；
- generation seed 为 `2026081200 + scene_index`，同时显式 seed Habitat simulator、
  PathFinder 和 NumPy；
- 每条 requested episode 最多允许 600 次 `make_episode` 调用；这是对 v1 固定 6 次预算的
  pre-outcome 可行性修复。scene、seed、geodesic、co-visibility、heading、轨迹平滑和碰撞
  接受标准均不变；
- 每场景必须恰好 8 条完整 episode，Goal-B 必须标为 Revisit；
- 新 metadata、parquet、goal 图哈希不得与该场景旧 `episode_0000/0001` 相同；
- development、4-scene final-reserved 和 16-scene blind 均禁止读取。

机器生成的数据在 inference 开始前由独立 CPU stage 形成不可覆盖 manifest；评测只接受
该 manifest 中列出的文件和 SHA256。

## 统计与冻结决策

主比较是 `known_revisit_direct - geometry_router` 的 paired joint/conditional-B risk
difference、`+/-` 数、two-sided exact McNemar，以及以 scene 为 cluster 的 100,000 次
bootstrap 95% CI。`native` 只作为预注册次比较。

只有同时满足以下条件才移除 geometry hard gate：

- direct 风险差为正；
- exact McNemar `p < 0.05`；
- scene-cluster 95% CI 下界 `> 0`；
- direct 不差于 native。

反向满足同一标准则保留 geometry。其他结果一律是
`inconclusive_keep_geometry_and_do_not_retune_on_these_episodes`。本池运行后即消费，不能
再用于阈值、adapter 或安全规则选择。

机器可读协议见 `revisit_fresh_confirmation_protocol_20260811.json`。
