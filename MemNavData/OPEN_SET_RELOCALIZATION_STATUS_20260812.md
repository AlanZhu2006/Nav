# Novel / Revisit 开放集定位：最新状态

更新时间：2026-08-12

## 一句话结论

当前最好的解法不是训练一个二分类器硬猜 `Novel` 或 `Revisit`，而是把问题改写为：

> **历史地图能否为当前 ImageGoal 产生一个有几何证书的 metric PointGoal？**

能定位就交给现有 revisit controller；证据不足就标记
`Unknown/unsupported` 并回退 native ImageGoal NavDP。定位失败不等于语义 Novel。

这一方案的 v2 双边几何证书已经通过 19 个设计集外 train 场景的冻结 HPC
actionability gate；这授权下一步闭环比较，但**还不是 SR 增益**。

### 2026-08-12 runtime 边界修正（闭环前完成）

真实 runtime smoke 发现：证书接受的 PnP 方位几乎准确（`174.28°` 对 GT
`174.61°`），但在线 ground-scale 把 GT `6.54 m` 错放大为 `15.29 m`。像素内点、双边
覆盖与重投影 RMSE 无法认证单目尺度，因此下文历史性的“certified metric PointGoal”表述只在
HPC v2 使用的外部 causal-scale artifact 条件下成立，**不能直接外推到在线执行器**。

闭环 runtime v3 已收缩为 scale-free bearing：8 个 v2 confirmation accepted rows 的
bearing error 全部 `<4.45°`（median `2.35°`），再经已闭环验证的
`verified_bearing_v1` 投影到固定 `2.5 m`。B0 中 fixed bearing 与 metric 接口均为
`20/26`（配对 `+1/-1, p=1.0`）。runtime 明确写出
`metric_distance_certified=false`，不再计算或暴露在线 metric scale；拒绝仍回退 native。

本机真实传输已通过：首次 certificate `2.09 s`、缓存 `0.15 ms`；移动后更新方向、reset
清缓存、accepted→mixed NavDP 与 rejected→native NavDP 都完成端到端烟测。正式协议见
`CERTIFIED_RELOCALIZATION_CLOSED_LOOP_PROTOCOL_20260812.md`。

## 为什么改写问题

- 一个 teacher co-visibility 仅 0.006 的旧“近 Novel”样本，实际可以定位到
  0.26 m / 2.9°；换成几何排序 anchor 后达到约 0.05 m / 0.19°。
- 因而 `covis < threshold` 不是部署意义上的 Novel；少量但高质量的对应点已经
  足以恢复相机位姿。
- 反过来，匹配失败只能说明当前历史证据不足，不能证明环境从未访问。

## 冻结架构

```text
ImageGoal
   │
   ├─ DINO：从历史帧提议 top-8（只负责高召回）
   │
   ├─ SuperPoint + LightGlue + Fundamental-MAGSAC
   │     └─ 在 top-8 内选一个几何支持最强的 anchor
   │
   ├─ LingBot-Map：anchor 的 causal depth + camera pose + metric scale
   │
   ├─ 2D(reference) → 3D(map) + PnP
   │     └─ 直接恢复 goal camera pose / relative PointGoal
   │
   └─ fail-closed certificate
         ├─ 通过：现有 PointGoal+ImageGoal revisit controller
         └─ 不通过：Unknown，回退 native ImageGoal NavDP
```

证书 v2 固定为同一个中心假设同时满足：PnP 内点不少于 16、query 与 reference
内点凸包覆盖都不少于 5%、重投影 RMSE 不超过 2 px。不会把不同邻帧的最大值
拼成一个虚假证书。

这个设计没有新导航策略，也没有另起一个语义分类网络。定位 expert 同时完成
“是否可信”和“往哪里去”，输出仍是主架构已经支持的 PointGoal 接口。

## 本机最终验证

数据限制：train-only，本地只有 2 个 scene；不能声称跨场景泛化或 SR 增益。

### 精确可执行集合

- 原始候选源：24 sessions。
- 同时要求 teacher evidence、`anchor >= 8`、LingBot camera cache 可执行后：
  **23 sessions / 167 top-8 candidate pairs**。
- 另 1 个 session 没有合法候选，按协议输出 Unknown，不补假候选。

### 图像几何证据

| 信号 | candidate-level ROC-AUC |
|---|---:|
| DINO cosine | 0.9491 |
| Fundamental inliers | 0.9910 |
| Fundamental query grid coverage | 0.9986 |

co-visibility 只用于这里的事后描述，未进入候选排序。

### Metric PnP actionability

GT-only actionability 定义：位置误差 ≤0.75 m、旋转误差 ≤30°，且非近目标的方向
误差 ≤30°。

| 结果 | 数字 |
|---|---:|
| GT 可定位 | 13/23 |
| 证书接受 | 11/23 |
| true positive / false positive | **11 / 0** |
| false negative / true negative | 2 / 10 |
| precision | 100%（Wilson 95% CI 74.1%–100%） |
| recall（对 GT 可定位） | 84.6%（Wilson 95% CI 57.8%–95.7%） |

在 11 个被安全接受的样本上：

- 中位位置误差：LingBot direct pose **1.329 m → PnP 0.052 m**；
- 中位方向误差：**85.51° → 1.41°**，10/11 改善，two-sided sign-test
  `p=0.0117`；
- 中位旋转误差：9.17° → 0.30°。

### 必须诚实保留的 null

早先 top-DINO 单候选的同类证书也是 11 TP / 0 FP。几何 top-8 排序在本地没有
增加最终证书通过数；它找到了额外准确的稀疏位姿，但冻结安全门选择 abstain。
因此目前证明的是：

1. LightGlue + LingBot depth + PnP 能安全地产生高精度 PointGoal；
2. 几何排序离线优于 DINO，但尚未证明带来 closed-loop 净增益；
3. 是否跨场景保持零假阳性，必须由 HPC 确认。

## 已否决路线

- **Colored/geometric ICP**：不是相机定位器。正例 refinement 将位置误差
  1.11→3.00 m、方向误差 12.98→131.5°；严格负例 colored fitness 反而达到
  0.867。单目稠密云会在重复墙面/地面上取得很高表面对齐分数。
- **SIFT PnP 主 expert**：测试正例失败，负例产生小规模伪内点；保留为负对照。
- **仅多视图一致性门**：约 9 m 的错误位姿在三个相邻 anchor 上仍稳定一致；
  重复结构可以“稳定地错”。空间支持和 PnP 证书不能被一致性替代。
- **强制 Novel/Revisit 二分类**：监督标签与实际可定位性不一致，问题定义错误。

## HPC 第一次冻结审计（v1，已完成但未通过）

- job：`15633271`，A100-SXM4-80GB，正常完成，耗时 28m24s；
- source bundle SHA：
  `3b6fc2bc1e13a5794c51e1760507cd3ad706e1b62ac86fdc7818fbf2f340a75e`；
- bundle 验证通过，HPC 内 34/34 tests 通过；
- Stage A：480 train sessions × 8 candidates = 3,840 图像对；
- Stage B：冻结 24 sessions / 20 scenes，三邻帧 causal LingBot-PnP；
- development 与 final-reserved 均不读取。

v1 原始 gate **未通过**：GT actionable 9/24；证书 TP/FP/FN/TN =
8/2/1/13，precision 80%。失败不是执行或合同问题：24/24 rows、20 scenes、所有
provenance/causal receipt 均通过。

两个 FP 的机制不同：

- `VLzqgDo317F` 是 16 m 的重复木门/墙板别名；query coverage 11.46%，但
  reference coverage 只有 1.93%。这暴露了 v1 只检查 query 支持的不对称漏洞。
- `JF19kD82Mey` 的目标位置误差只有 0.354 m，且视觉上确为同房间相邻视角；
  50.3° bearing error 来自仅 0.429 m 的短真值向量。PointGoal 与 benchmark 都只
  使用位置/距离，把它按 bearing 判为不可执行是审计定义错位。

因此 v2 只做两项结构性修正：增加同为 5% 的 reference coverage；GT-only
actionability 改为与 metric PointGoal 对齐的位置误差 ≤0.75 m。v2 在旧数据上的
11 TP/0 FP（local23）与 9 TP/0 FP（HPC24）都只是 post-hoc 设计证据。

## HPC 独立冻结确认（v2，已通过）

- job：`15634113`，A100-SXM4-80GB，正常完成，耗时 27m28s；bundle SHA：
  `f73016d5ecfebfa7c0cd28e34ea80890c81832d714365ff712b6d5d58ad57ecb`；
- 24 sessions / 19 scenes；19 个场景均未进入 local 设计或 v1 HPC；
- 每场景先 hash 取一条，再 label-blind hash 取 5 条额外 session；
- 冻结后才查看分布：8 positive、14 strict-negative、2 ambiguous；
- bundle 验证、节点内 tests、Stage-B preflight 和最终 contract receipt 均通过；
- 原始 rows SHA：
  `eed09072d87973bc7232571dcf93fe517596e752e0ac7fa2af43486be80f73c5`。

冻结 v2 结果：

| 结果 | 数字 |
|---|---:|
| GT metric-PointGoal actionable | 9/24 |
| 证书接受 | 8/24 |
| TP / FP / FN / TN | **8 / 0 / 1 / 15** |
| precision | **100%**（Wilson 95% CI 67.6%–100%） |
| recall | 88.9%（Wilson 95% CI 56.5%–98.0%） |
| certified actionable scene | 6 |

8 个接受样本的位置误差中位数为 0.131 m；对应 LingBot direct pose 为
0.154 m，5/8 改善、3/8 变差，sign-test `p=0.727`。因此不能声称 PnP 显著改善
已接受样本的位姿；它当前最硬的价值是**安全证书与可执行 PointGoal**。

唯一 FN 的位置误差为 0.401 m，但 reference coverage 为 4.59%，略低于冻结的 5%
门槛；这是保守拒绝，不放宽。另有一条 teacher-negative（covis 0.184）被安全接受，
位置误差 0.212 m，直接验证了“低共视 ≠ 不可定位”。

我用不导入 summarizer 的独立脚本从每行唯一 `offset=0` 假设重算，得到相同的
8/0/1/15、6 scenes 和误差中位数；本机 43 个定向测试与 `git diff --check` 通过。

旧 job `15633241` 在方法运行前因 bundle 漏装一个测试 fixture 而 fail-closed；它是
packaging failure，不是实验结果，已由新 SHA bundle 修正。

HPC effectiveness gate：0 false positive、至少 5 个 certified actionable session、
至少覆盖 5 scenes、且中位位置误差优于 LingBot direct pose。通过只授权下一步
同机同进程闭环三臂比较，不直接构成 SR 结果。

## 下一步

1. 实现最小 runtime adapter，把 certified map goal 转成现有 metric PointGoal 接口。
2. 做 native / 当前 geometry router /
   certified relocalization router 的同机同进程闭环配对。
3. 闭环前不再调整 v2 证书；若出现回归，按定位接受/拒绝与 controller 执行分层归因，
   不回到语义 Novel/Revisit 阈值调参。

完整冻结协议见 `OPEN_SET_RELOCALIZATION_PROTOCOL_20260812.md`。
