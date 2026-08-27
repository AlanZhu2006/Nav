# REVISIT：已知阶段下的 RANSAC 必要性消融

日期：2026-08-11（CST）  
状态：协议冻结；只使用已经 consumed 的 20-scene / 40-episode 池；不是论文确认。

## 问题

当前 2-leg benchmark 在生成和评测协议中已经明确规定：Goal A 是 Novel，Goal B 是
Revisit。当前最强 `memory_geometry` 实现却为了支持未知阶段部署，在两个 leg 都运行自动
memory router，并允许 SIFT/RANSAC 决定是否启用 memory。

新完成的 geometry evidence audit 表明，RANSAC 在真正 Revisit 上是高精度但有限召回的
证据：pass precision `90.9%`，positive recall `66.3%`。R0 的 6 条 Revisit 残差中有 4 条
从未激活；这 4 条都有接近目标位置的 DINO memory proposal，其中两条被 `0.88` floor
挡住、两条被低纹理 RANSAC 拒绝。

本实验只回答：**当任务协议已经声明当前目标是 Revisit 时，RANSAC 是必要的安全 expert，
还是过度保守的 activation veto？**

## 两臂

每个 scene 先执行一次 Goal A，并冻结完整图像哈希 trace。Goal B 两臂共享同一 trace、
episode seed、逐 plan diffusion seed、goal image、Habitat scene、LingBot、pose recovery 和
mixed NavDP controller。

1. `geometry_router`：raw DINO gap-16 top-8；visual floor `0.88`；SIFT/RANSAC
   `matches>=20, inliers>=12, ratio>=0.5`；连续两次确认后 latch。
2. `known_revisit_direct`：在已知 A→B Revisit 边界直接调用相同 raw-DINO retrieval 和
   LingBot pose；不运行 RANSAC activation gate。pose 暂不可用时仍 fail closed 到原生
   ImageGoal。

禁止 learned reranker、X-NavDP、oracle、terminal refine、graph、bearing head 和任何新阈值。
`success_dist=1m`、`max_steps=500`、`exec_horizon=8`、server trajectory selector 不变。

## 主要统计

- 主要分母：共享 Goal A 成功后的 Revisit B；
- 同时报告 joint SR、SPL、最终距离、path length；
- direct 相对 geometry 的配对 gain/loss/tie、exact McNemar 和 scene-cluster bootstrap CI；
- 所有对照必须同机、同一对长驻 server、同进程顺序执行。

## 冻结决策

- `gain>0 且 loss=0`：direct 进入 fresh non-blind confirmation；RANSAC 降为可选诊断，不再
  是 Revisit activation gate；
- `loss>gain` 或 `loss>=2`：RANSAC 是必要安全 expert；下一步只研究 cluster-local
  multi-view geometry，不能直接移除；
- 其余：结论不充分；只授权 cluster geometry 消融。

无论哪一分支，本 consumed-pool 实验都不授权 blind evaluation 或论文方法声明。

## 实现

- `run_local_phase_b_p0_20scene.sh`：新增只改变该因子的
  `RUN_KNOWN_REVISIT_DIRECT=1` arm；
- `summarize_revisit_phase_ablation.py`：严格收据、配对统计和冻结分支；
- `test_summarize_revisit_phase_ablation.py`：bootstrap 与决策单测。
