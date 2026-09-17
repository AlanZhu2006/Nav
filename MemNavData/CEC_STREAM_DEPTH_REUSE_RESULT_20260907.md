# 主在线几何复用于 CEC：三历史本机诊断结果

日期：2026-09-07。研究／仿真侧诊断完成；未修改论文、正式方法或真机服务。

## 1. 结论先行

**保存主在线 LingBot 流的历史深度，值得作为减少重复几何计算的下一步，但尚不能
替换 canonical CEC。**

3 条已消费历史、9 个预定图像对，两个深度来源均为 9/9 certificate accept，
0 次授权改判。输出方向差中位数 **0.0273°**，最大 **0.0836°**。主在线状态和
first40 scale receipt 完全一致；没有增加第二份 dense KV 状态。

另一方面，深度本身**不是逐元素等价**：三条历史的平均相对差分别为 3.69%、
1.66%、2.61%。因此这是估计器输入变体，不是可以直接继承旧 SR 的透明缓存优化。

本次没有执行 controller，没有新的导航 SR，也没有 Novel 误激活测量。

## 2. 实验究竟比较了什么

正式默认：历史 RGB → 首次查询时 dense causal replay → anchor depth → PnP。

本轮变体：同一主在线 LingBot 流 → 在 anchor 被观察时保存 depth/confidence →
首次查询直接读取该帧 → 同一个 PnP。

候选 anchor、RGB prefix、LightGlue 匹配数组、参考／当前 camera pose、PnP seed、
证书阈值均固定。匹配只运行一次，两个分臂收到相同数组。未引入检索头训练、
新的门控、其他 planner 或第二份 dense 几何流。

每条历史做普通 ingest 和带一次 materialization 的 ingest，跨历史交替先后顺序。
同一 GPU、同一模型进程、相同 reset seed；flow gate 使用原 episode length。
两种 ingest 的主 KV、camera pose、DINO 特征、最后聚合特征和 first40 receipt
一致，materialization 与查询也没有改变被检查的主在线状态。

完整预定构造见 [protocol](CEC_STREAM_DEPTH_REUSE_PROTOCOL_20260907.md)。

## 3. 逐历史数据

| 已消费历史 | anchor / current | 图像对 | canonical / online accept | 最大方向差 | 深度平均相对差 |
|---|---:|---:|---:|---:|---:|
| gxdoqLR6rwA / episode_0000 | 80 / 120 | 3 | 3 / 3 | 0.0364° | 3.689% |
| pLe4wQe7qrG / episode_0000 | 120 / 160 | 3 | 3 / 3 | 0.0836° | 1.661% |
| yqstnuAEVhm / episode_0001 | 160 / 200 | 3 | 3 / 3 | 0.0606° | 2.607% |

深度平均相对差为 `mean(abs(online - canonical) / canonical)`，只在两者均为
正深度的像素上计算，不是相对 GT depth 的误差。

9 对输出方向差：均值 0.0349°，中位数 0.0273°，最大 0.0836°。
从既存 trace 离线取 GT，仅用于事后评分：

| 方向估计 | GT 直线方向误差均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| canonical replay | 0.4681° | 0.4577° | 1.5433° |
| saved online depth | 0.4478° | 0.4453° | 1.4598° |

两者平均差很小，不能声称定位准确率提升。目标到 current 的 GT 平面距离为
1.38–2.25 m；本轮没有接近零距离的病态方向项，但也**不是长距离测试**。

## 4. 耗时与资源：必须区分冷、热查询

以下是 CUDA 同步计时，模型已经加载；证书查询**不含共享的 SuperPoint／LightGlue
匹配、不含 DINO shortlist、不含 NavDP 推理或网络传输**。

| 历史 | 首次 canonical 查询 | 对应 saved-online 查询 | 历史时刻保存一次深度 |
|---|---:|---:|---:|
| gxdoqLR6rwA | 11.741 s | 16.13 ms | 13.45 ms |
| pLe4wQe7qrG | 19.330 s | 10.67 ms | 13.05 ms |
| yqstnuAEVhm | 27.248 s | 5.24 ms | 13.30 ms |

这里测到的主要收益是**消除新 anchor 首次查询的全历史重放**。同 anchor 的后续
canonical 查询原本也会命中缓存；不能把首次冷查询的加速比套到每次重规划。

- 每个 anchor 保存 depth + confidence：2,146,592 bytes，约 2.05 MiB CPU 数据。
- 仅保存预选的 1 个 anchor／history；当前代码没有替正式运行时选择全历史保存策略。
- 若将这一表示保存 1,000 帧，单这两张数组约 2 GiB；这是线性存储估算，不是实测
  完整部署内存。不能把本轮单 anchor 成本当作全历史检索的免费缓存。
- 两种分臂都没有常驻第二份 dense state。相较既有 eager 双流方案，这条变体
  不需要再跑一份 dense aggregator；并不是声称当前 canonical 默认一直占着第二份状态。
- GPU 为本机 RTX 4090；试验进程已正常退出，未终止或修改其他常驻服务。

## 5. 为什么深度不同，而方向可以很接近

本轮证据只表明：在这些高支持局部图像对中，深度变化没有造成显著的方向变化。
一种合理解释是 PnP 只需从邻近历史 anchor 恢复一个较小的相对偏移；当 current
离 anchor 更远时，该偏移的数值变化未必显著改变 current-to-goal 的方向。

这不是“归一化能消除全部几何误差”。非均匀深度误差可能改变 PnP inliers、
旋转、相对位移或 certificate；主流累计漂移也不会被单位方向自动消除。
本轮所有图像都容易匹配，不能据此推断跨视角或低支持样本。

## 6. 验证与限制

独立进程从落盘的 matches、depth、pose 重新执行同一冻结 PnP／certificate 实现，
**18/18 分臂记录复算一致，verified=true**。这是独立数据／执行入口复算，不是
另行实现一套 PnP 算法。

主要限制：

1. 三个目标都取自各自 RGB 历史，是真正重复 JPEG，不是重渲染的独立视角目标。
2. 每次只给一个预选 anchor，未覆盖 DINO top-8 排序与不同 depth 导致候选改变。
3. 无 Novel／无匹配负例；9/9 accept 不能用于估算 false activation。
4. 只有 3 个场景；9 对不是 9 个独立场景，不能用来声明等价或普遍泛化。
5. 短固定 RGB prefix，不是实际 rollout，也没有长程绕障、漂移或到达测量。
6. 单次耗时，不报告稳定延迟分位数或 Jetson／真机端到端延迟。

## 7. 接下来做什么

**投稿主表先不改。** SPL 补跑仍是论文计量收尾；主方法现有闭环证据保持原定义。

这个支线下一步应复用已消费／训练侧**真实目标图**做小型配对，包含：

- 不同视角的 supported Revisit；
- 低共视、证书接近边界的候选；
- 无匹配／Novel 负例；
- 至少几条较长 prefix。

保持完整 DINO top-8 与同一匹配输入，重点看证书改判、anchor 改变和方向失真，
同时明确历史保存覆盖率／内存成本。先完成这一步，再决定是否值得小规模导航
复测；本次不启动长训，不提交新长时间闭环，也不把小样本当作替换门。

## 8. 工件与当前 HPC

- 驱动：`MemNavData/benchmark_cec_stream_depth_reuse.py`
- 独立复算：`MemNavData/verify_cec_stream_depth_reuse.py`
- 冻结 manifest：`.diagnostics/cec_stream_depth_reuse_20260907/pilot_v2/manifest.json`
- 全量结果：同目录 `summary.json`
- 复算收据：同目录 `independent_verification.json`
- 每条历史：`ingestion.json`、`depths_and_poses.npz`、三个 `matches_*.npz` 与
  `query_*.json`；原始运行日志为 `run.log`。

`pilot_v1` 因旧归档使用六位 JPEG 编号，在数据预检退出；没有模型结果。
修复文件名解析后使用新目录 `pilot_v2`，样本和方法选择不变。

本轮严格使用 `alantorch` / `yz11502` 共享 SSH 的 PTY 检查 SPL 作业：

- `17057430` preflight completed；
- `17057431_17` 与 `17057432_0..3` completed：合计 **5/21 histories**；
- 剩余 `17057432_[4-16,18-20]` 因 **QOSGrpGRES** 排队；
- `17057433` 汇总等待 dependency，无本轮新增失败。

这不是完整新 SR／SPL。待 21 条历史全部完成并核验后，Table III 才同时替换
新 SR 和精确 SPL；本轮没有取消、重提或改变 HPC 冻结任务。
