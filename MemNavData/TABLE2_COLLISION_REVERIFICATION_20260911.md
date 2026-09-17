# Table II：碰撞环境复算修复（2026-09-11）

范围：只修离线验证，不改导航运行时、候选、阈值、预算或成功判据。

## 发现与本机复现

正式 A 来源 `29hnd4uzFmX/episode_0005`，数组 `17357681_21`，执行了 164 步，
原始记录 `reached=0`、实际路径 `1.9573274919885881 m`、最终距离
`4.151704415660452 m`、SPL=0。任务随后在验证动作 40 的碰撞落点时失败，
不能将 Slurm FAILED 直接解释成新的导航失败，也不能删掉该来源。

原始归档 SHA-256：
`ccff850082f8e8c908dbe71ace33fe058a7d083236e1e0ba9c7457f7d7b571db`。

本机读取同一归档、相同场景 GLB，以无相机渲染的 CPU 测试复现：

| 检查 | 结果 |
|---|---|
| 独立重算的控制请求 vs 原始请求 | 动作 40 完全相同；全程最大误差 1.11e-16 m |
| 现场方式重建后的 NavMesh vs 原归档 | 文件逐字节相同 |
| 重建后直接使用的连通区域数量 | 2 |
| 保存后重新加载的连通区域数量 | 3 |
| 动作 40 起点/终点所在区域，直接重建 | 0 / 0 |
| 动作 40 起点/终点所在区域，重新加载 | 0 / 2 |
| 直接重建后复放全部 164 个请求 | 164/164 落点逐位相同，最大误差 0 |

NavMesh 文件 SHA-256（本机重建与原 HPC 归档相同）：
`ea19d0d87efe2d3a26b68d739303a470864e1b3a8f0c088013e4207f6e064144`。

这不是 controller 请求错误、日志漏点或 NumPy/Math 三角函数精度造成的误差。
问题是旧 verifier 假设“加载文件后的连通性”必定等价于“执行时现场重建的连通性”。
Habitat 0.3.3 在建立岛屿表后处理零面积多边形，`tryStep` 又检查起终点连通性；
这与此次前后连通区域不同的复现一致。
[官方实现](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/src/esp/nav/PathFinder.cpp#L862)。

## 修复方式

1. 原 GPU 任务、原 bundle、plan 和失败归档不改。
2. 单独封存 CPU 复核工具。依据源 GLB 哈希及归档内 NavMesh settings 重建执行环境，
   必须证明重建后保存的 NavMesh 与原归档逐字节相同。
3. 将此执行环境显式传给独立 rollout verifier；位置容差仍为 1e-7 m，
   同时复验完整任务的深度、图像、目标、轨迹末点、SR、SPL 和分支协议。
4. 成功后逐项确认原始索引中的所有文件均未改变；将原失败目录移至
   `failed_attempts/17357681_21_collision_context`，不删除。
5. 生成补充验证收据和可供下游读取的新完整归档。原失败日志与原索引同时保留。
   原始 summary 数字保持不变，绝不把失败改成成功。

这不是导航重跑，不消耗 GPU，不接受“略微放宽容差即可”的做法。
该复核只能恢复记录完整且故障确属此类的任务；别的任务故障不能套用此修复。

代码：`verify_rebuilt_collision_context.py`、`reverify_table2_collision_task.py`、
`slurm_table2_collision_reverify.sbatch`；`verify_repaired_fullmono_local.py` 仅增加
显式 `execution_pathfinder` 审计依赖参数，默认行为保留，未修改正在运行的 frozen bundle。

## 当前验证状态

- 本机原归档 164 步执行请求/落点复现：通过。
- 相关单元测试：初步 24 passed，完整 Table II/最终后处理回归 69 passed；
  CLI import 和 sbatch 语法：通过。
- HPC CPU 作业 `17365241`：`COMPLETED / 0:0`，22 秒，全任务复核通过。
- 再次使用独立归档复算器检查恢复后的记录：通过，`reached=0` 未改变。
- 原归档保存在 `failed_attempts/17357681_21_collision_context/`，重新计算的 SHA
  与上列原始 SHA 完全相同。没有执行第二次导航。
- 正式 A reducer `17357682` 尚在等待主数组，恢复早于 reducer 启动完成；
  不需要重提数组或取消已运行来源。原 Slurm 失败状态保留，由补充 CPU 复核收据解释。

补充工具封存于正式 run 的 `collision_reverification_v1/`；
其 `REVERIFICATION.sha256` 的 SHA 为
`bb07fc26e1e831cf3d629d181e91dfebdf561f6ddbd2ba24cb101dbbaf78bff9`。
此处进度回填不修改那份已封存的工具/说明副本。
