# 单目深度栅格修复：本机完整配对结果

日期：2026-09-08。状态：8/8 查询完成，独立复算通过，8 个第一视角视频完成。

## 结论

逆转 LingBot 的 resize/pad、恢复到原 RGB 栅格，是有源码及实际张量依据的接口修复。
在这两条已消费历史上，native 从 0/2 到 1/2；CEC 保持 1/2。
**不能据此称整体 SR 提升，也不能称深度修复对 CEC 无作用。** 分母为两条历史，而不是八条独立样本。
后向 PointGoal 引起的零轨迹停滞没有被深度修复解决，需要独立接口处理。

## 固定比较

两条历史为 gxdoqLR6rwA、pLe4wQe7qrG，均来自此前实际 metric-NavDP-A。
每历史共享 A、起点、目标、首张 RGB，分别运行 native / CEC × old square / source RGB raster。
第二历史反序；每臂上限 600 tick、8 tick 重规划，1 m 平面 GT 到达。
查询均为 LingBot 单目深度、正确 RGB 通道、bounded pursuit + 一次标准 try_step。
没有同时加转向、改证书、改 2.5 m 或换 controller。不是 full-mono-A 的新外部确认。

| 历史 | 方法 | 原方形深度：成功 / tick / 实走 m | 对齐深度：成功 / tick / 实走 m |
|---|---|---:|---:|
| gxdoq | native | 0 / 600 / 15.701 | 0 / 600 / 22.525 |
| gxdoq | CEC | 1 / 115 / 3.905 | 1 / 125 / 4.103 |
| pLe | native | 0 / 210 / 2.290 | 1 / 360 / 9.097 |
| pLe | CEC | 0 / 151 / 0.000 | 0 / 151 / 0.000 |

- native：+1/−0，双侧 exact McNemar p=1.0。
- CEC：+0/−0，无不一致对；p=1.0，不能解释成等效检验通过。
- gxdoq 的 CEC 对齐后用了更多步；native 虽同为失败，路径明显不同。
- pLe 的 CEC 两臂均无平移，修正深度不能恢复已被前向坐标裁剪损失的后向目标语义。

## 独立审计

- 源码/权重运行期间无变化，`changed_source_files=[]`。
- 8 条实际路径、末步后终点、SR、SPL 均由逐动作坐标重算。
- 291 份真实模型规划的 producer/readout 深度 NPZ 逐像素复算，源 RGB 尺寸一致。
- 几何已激活时，old 为 `[1,518,518,1]`，source 为 `[1,270,480,1]`。
- 原始缓存、单目尺度及 PnP 历史坐标不变；改的是 NavDP dense readout。
- 无 sensor depth 消费；仍使用 Habitat 标准 NavMesh 碰撞及理想低层状态，不是无 GT 物理实验。
- 全部成功、失败视频已保存，10 fps 表示控制 tick，不包含模型推理等待时间。

## 结果路径

项目根目录下：

```
.diagnostics/habitat_minimal_repair_20260908/depth_raster_bridge_v1/
  manifest.json
  source_snapshot/
  summary.json
  independent_verification.json
  evaluation/
  depth_raster_artifacts/
  first_person/
```

后续保持正确 RGB 与对齐 depth；不以这两条的 SR 决定回滚正确输入。
前向 PointGoal 域适配见 `HABITAT_FRONT_GOAL_BRIDGE_PROTOCOL_20260908.md`。
本结果未写入论文，也未改变任何生产默认或旧正式结果。
