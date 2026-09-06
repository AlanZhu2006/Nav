# Sparse Monocular History Route：开发诊断

日期：2026-09-03

状态：**已完成并拒绝为统一精度修复；不授权闭环评测。**

## 问题

Index39 的冻结 route-tangent 确认在方向输出上通过，但逐帧 history route 的终点误差为 1.552 m，超过预注册的 1.25 m。一个可能原因是 1,189 个约 2–3 cm 的视觉边连续积分放大了相关误差。

本诊断复用系统已有的 sparse-depth cadence：从 CEC anchor 开始，每个绝对帧号 8 的固定节拍直接估计一条 history-to-history RGB + height-scaled monocular depth + LightGlue/PnP SE(2) 边。关键帧选择不读取位姿、距离、role 或结果；query motion、累计路径预算、route-tangent readout 和 2.5 m controller residual 均保持不变。该设计不会把反向 query 图像匹配到正向历史，因此不重复已经证伪的 180° DINO/PnP route-address 方案。

## 结果

三个样本都已被此前机制开发或确认消耗，因此这里只是 post-hoc development。

| History | Sparse edge chain | Dense tangent within 30° | Sparse tangent within 30° | Dense median | Sparse median | Dense endpoint | Sparse endpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 147/147 | 67/72 | 64/72 | 7.38° | 9.82° | 1.026 m | 1.215 m |
| 39 | 236/236 | 90/105 | 92/105 | 11.64° | 5.55° | 1.552 m | 1.557 m |
| 40 | 208/208 | 81/99 | 72/99 | 17.44° | 15.75° | 0.401 m | 1.083 m |

所有稀疏链均完整，但效果不一致：index39 的方向更好，index16 略差，index40 的 ±30° coverage 从 81.8% 降至 72.7%，低于既有 80% 机制标准。Index39 的 endpoint 误差也几乎不变。

## 决策

固定 stride-8 证明较大基线 PnP 可以把历史边数和深度请求约降到逐帧方案的八分之一，但不能稳定提高路线方向或终点精度。因此：

- 不进行 stride sweep；
- 不把稀疏结果与 dense 结果按样本择优；
- 不将它冻结为新论文方法；
- 不据此提交闭环 SR。

该负结果进一步表明，index39 的 1.552 m endpoint failure 不是单纯由“小基线边数量过多”造成。若继续改善绝对 route shape，需要真正的长程约束；仅改变相邻边采样密度不够。

## 原子产物

- index16：`sparse_history_index16_stride8_dev_v1/sparse_monocular_history_route.json`，SHA256 `7efe5760f801ab2b017799ef39de6fcfd7727b9260f195e2c9410f09cbdebc47`
- index39：`sparse_history_index39_stride8_dev_v1/sparse_monocular_history_route.json`，SHA256 `ed1f0229033d70ba69b44522c421c2817beb628937faa2413b5bddd6e90e6abf`
- index40：`sparse_history_index40_stride8_dev_v1/sparse_monocular_history_route.json`，SHA256 `d8a8f80e7e75dd8fceac6202e34b40eece17ea3a9294e3fb11280d7736bb5d2f`

产物根目录：`.diagnostics/mono_adjacent_motion_20260903/`。
