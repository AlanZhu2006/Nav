# Monocular Route-Tangent Compass：冻结确认结果

日期：2026-09-03

结论：**预注册机制门未通过；不授权 fresh 闭环 SR。** 该结果不能通过修改阈值、删除失败项或读取导航结果后重解释为通过。

## 1. 冻结身份与信息边界

- 冻结协议：`MONOCULAR_ROUTE_TANGENT_COMPASS_PROTOCOL_20260903.md`
- 协议 SHA256：`4200787139e73c9d27354292eccd25a8337282a31a8f1c49ae78fceba7c9a3d4`
- confirmation freeze SHA256：`3fb0c2d1401107e43230bf077f3fa8d383f4b41c99da80db59fd7c56059ba3cc`
- population index：39
- scene：`LT9Jq6dN3Ea`
- episode/history：`survey_0435`
- causal history：1948 帧
- authorized historical anchor：67
- query set：215 帧，其中 105 个平移帧、109 个原地转向帧，另含 route origin
- 运行时未读取 simulator depth、GT/global pose、轮速里程计、role label、导航 outcome 或 SR；相机安装高度是唯一尺度先验。

远端输入按共享 SSH 手册通过既有 ControlMaster 连接镜像。本地复核了 Goal-A、online-A trace、scene、navmesh 和全部 1948 张 RGB 的 SHA256；无输入哈希不一致。

## 2. 一次性结果

Query 相邻运动链：

- 214/214 对接受；PnP 214/214；链完整；
- 局部路线方向误差中位数：0.837°；
- 转向帧伪平移 P95：0.00643 m；
- query 路径长度相对偏差：-5.56%。

History route-tape：

- 1880/1880 条边接受，链完整；
- 其中 1189 条视觉边、691 条 exact-RGB identity 边；
- 预测路线长度：31.637 m；scorer 路线长度：34.011 m；相对偏差：-6.98%；
- 路线位置误差中位数：0.490 m，P90：0.819 m；
- **历史终点误差：1.552 m。**

Route-tangent readout：

- translated bearing：90/105 在 ±30° 内，即 85.71%；
- translated bearing 中位误差：11.643°；P90：60.244°；
- query 最终 progress fraction 误差：0.00218；
- projection 单调且遵守累计路径预算；
- 每个 controller payload 的范数严格为 2.5 m；
- 不含 distance regime、visual gate、endpoint fallback 或 native fallback。

## 3. 冻结门逐项结果

| 预注册检查 | 结果 |
|---|---:|
| Query motion chain complete | PASS |
| History motion chain complete | PASS |
| No forbidden runtime inputs | PASS |
| Query path-length bias ≤ 15% | PASS |
| History path-length bias ≤ 15% | PASS |
| History endpoint error ≤ 1.25 m | **FAIL：1.552 m** |
| Translated bearing within ±30° ≥ 80% | PASS：85.71% |
| Translated bearing median error ≤ 20° | PASS：11.643° |
| Projection monotone and path-budgeted | PASS |
| Controller residual norm = 2.5 m | PASS |
| No regime/gate/fallback | PASS |

总判定：`passed=false`，`fresh_closed_loop_construction_authorized=false`，`closed_loop_sr_computed=false`。

## 4. 结果解释

这次确认没有否定 route-tangent 的核心方向机制：两个直接方向门均通过，而且 query/historical motion chain 与尺度偏差均合格。它否定的是更强的联合主张——当前逐帧单目 SE(2) 累积在该 34 m 路线上还能同时满足 1.25 m 的绝对历史终点误差。

事后归因只用于设计下一版机制，不能改变本次判定。预测与 scorer 的历史终点向量夹角约为 19.20°；将整条预测路线按真值长度统一缩放后，终点误差仍为约 1.49 m。因此失败主要不是单一尺度系数，而是长链中相关方向/形状误差的累积。下一步若继续，应在已消耗样本上开发能够抑制长链漂移的视觉重锚定或稀疏关键帧路线表示，再对新的未读样本和新冻结协议做确认；不得只把 1.25 m 阈值放宽。

## 5. 原子产物

- query set SHA256：`a02b0750cd798d84595dbcb153988e373ff5eb382262d18e83a1fead41a73840`
- query-motion audit SHA256：`e256929f2f6fbbcbf30d3b543a3d72adeb8838c2d60e66dabed95b60b69e24c8`
- history-route audit SHA256：`60d2cdcbb44ae9da98270f4df6c04d8c52bb616944906a42ef7beb3f2d4112a9`
- tangent replay SHA256：`da8d558d4a6e9eabdb27da81a284db3a3944ceb8751be064149e61e9be5009bc`
- final gate SHA256：`76c8717e6e33bfee299b304fe8679fbdeb924e4b1eaa78d72c36db96e44f671d`

所有路径均位于 `.diagnostics/mono_adjacent_motion_20260903/`，正式 gate 为 `route_tangent_index39_gate_attempt1/route_tangent_confirmation_gate.json`。
