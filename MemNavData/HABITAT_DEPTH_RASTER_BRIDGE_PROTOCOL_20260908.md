# 单目深度栅格修复的下一阶段本机对照

日期：2026-09-08。状态：真实四臂 8-tick smoke 与两历史八臂完整闭环均完成并独立核验。
完整结果见 `HABITAT_DEPTH_RASTER_BRIDGE_RESULT_20260908.md`。
这是读取当前六臂部分结果后的新增接口检查，不冒充原六臂的预注册内容。

## 问题

LingBot 输出在自身 pad-518 坐标上；NavDP RGB 使用原帧的独立 letterbox。
在真实缓存 payload 中已经确认 RGB padding 含非零预测深度。
修复是逆 resize/pad 回原帧栅格，不改变深度单位、height scale 或模型权重。

## 固定局部设计

使用现有四历史 manifest 中顺序最前的两条：gxdoqLR6rwA、pLe4wQe7qrG。
两者都已消费，只做接口诊断，不做显著性/泛化确认。共享旧 actual metric-NavDP-A；
不称新 full-mono-A。每条 natural Revisit 查询运行四臂：

1. native / corrected RGB / bounded standard executor / old square depth；
2. native / corrected RGB / bounded standard executor / source-RGB-raster depth；
3. CEC / corrected RGB / bounded standard executor / source-RGB-raster depth；
4. CEC / corrected RGB / bounded standard executor / old square depth。

第二历史倒序。8 条 rollout 均在本机；各历史四臂使用同一组 model process。
旧 square-depth 臂也重新配对运行，不跨进程偷用上一阶段结果作为严格对照。
预算 600 tick、每 8 tick 重规划、1 m 平面 evaluator 到达；所有其他设置与六臂一致。
不同时加转向、不动 CEC 阈值和 2.5 m，不调用 sensor depth。

## 启动前与核验要求

- 先等正在运行的六臂结束，不能修改其封存源码。
- 用实际 Flask reset/plan/depth 入口验证每个 reset 固定 raster 模式；不是只测纯函数。
- 原 JPEG、depth token、原始尺度、裁剪坐标和最终 actor 深度栅格要可对账。
- 新模式中几何已激活的 depth 从 518 方形变回 source RGB 尺寸；bootstrap zero 仍保持 zero。
- NavDP encoder 本身保持未改；不同模式出现不同计划是允许结果，不能要求首次计划相同。
- 各臂完成后独立重算实际位移/SPL/终点，并保存所有成功与失败的第一视角视频。

这是后续有限的输入修复检查，不是正式重跑 Table I--IV 的授权或完成证明。
后向 PointGoal 的独立处理依赖前一阶段归因，不能与本次 raster 对照混在一个新“修复臂”里。

## 启动收据

- 前一阶段六臂任务先完成 24/24、独立核验和全部视频，之后才扩展 runner/verifier。
- `depth_raster_smoke_v1`：4/4 完成，4 份原始/转换后深度 NPZ 独立复算通过，4 视频完成。
  native、CEC 各自两臂的首帧原始 depth PNG 与原始张量 SHA 完全相同。
  输出形状分别为 `[1,518,518,1]` 与 `[1,270,480,1]`。短测不是 SR 结果。
- `depth_raster_bridge_v1`：按本协议 first-two histories × four arms，600 tick 上限。
  模型、源码、输入清单和每次实际深度转换归档；由同一 supervisor 负责复算、视频和私有服务收尾。

启动命令（项目根目录）：

```bash
PYTHONPATH=/home/asus/Research/Nav-graph-blind \
/home/asus/miniconda3/envs/memnav/bin/python -u \
MemNavData/run_habitat_minimal_repair_local.py local \
  --study depth_raster --histories 2 --max-steps 600 \
  --out .diagnostics/habitat_minimal_repair_20260908/depth_raster_bridge_v1
```

状态取运行目录 `progress.json`，同时核对 supervisor/owned process 是否仍在；
不能仅凭旧 progress 文件宣称运行正常。当前没有任何 HPC 或真机任务由本协议启动。
