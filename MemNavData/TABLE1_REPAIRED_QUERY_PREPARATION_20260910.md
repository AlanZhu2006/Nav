# Table I 修复版 controller 对照：准备记录

后续更新：用户要求加入 NoMaD 后，三-controller正式设计与执行状态分别转到
`TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md` 和
`TABLE1_REPAIRED_THREE_CONTROLLER_STATUS_20260910.md`。
下文保留早期两-controller准备过程；其中保留distance>7掩码的方案已由新协议明确替代。

2026-09-10，北京时间。状态：原总体已定位、ViNT RGB 输入修复已做真实权重 CPU 验证；
**尚未完成新版 ViNT 闭环集成，未提交 Table I 补跑。** 本文件不是正式冻结运行协议。

## 1. 为什么做这一项

会议主表要求 NavDP/ViNT × HM3D/MP3D 的同 controller、同查询 native/GEM 配对。
原四组实验已经完成，但使用旧输入/执行链。9月10日完成的新 HM3D actual-mono-A
结果不能替代 MP3D 或 ViNT，也不能与旧版 native 横减。

本项复用原 Table I 的实际在线历史和原始查询，隔离修复版 query controller。
不重新生成有利目标，不要求另采一批 A，不称修复版 A+query 全流程结果。
新的 full-mono 全流程证据来自另一批 45 histories / 90 queries，不能合并分母。

## 2. 已由 HPC 原始封存记录核对的总体

| 数据集 | 历史 | 场景 | Novel / Revisit 查询 | 两 controller × native/GEM 的总 rollout |
|---|---:|---:|---:|---:|
| HM3D | 28 | 21 | 28 / 28 | 224 |
| MP3D | 42 | 25 | 42 / 42 | 336 |
| 工作量合计，不是一个科学总体 | 70 | 不跨数据集合并 | 140 | 560 |

原 manifest 的历史条目字段名为 `episodes`，不是 `histories`。每条含两类 query，
每个 controller 内做两臂配对。跨 controller 的绝对 SR 不直接解释为纯模型能力差。

HM3D manifest：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914/population/natural_direction/manifest.json
SHA256 f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01
```

MP3D manifest：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/natural_direction/manifest.json
SHA256 a33f210fdd0cfa84e82c4d403ac79056dcc7959cd1ce84bf62bec8c5632deb69
```

两份文件均在本轮通过共享 `alantorch` / `yz11502` 直接读取并重算 SHA；
条目数和 scene 数分别为 28/21、42/25。
MP3D 的最终入口来自正式运行 `formal_20260829T085025Z/sealed_inputs/experiment_inputs.sha256`，
构造验证 SHA 为 `618c409f7c7c62ad739687935cdd6f2e564e96aed6ccf6059d887d795c3e953e`。
早期 `construction_20260829T050401Z_6813d501` 是扩源前版本，不能误用为最终 42-history 总体；
两者哈希不同本身不是数据损坏。

本轮只核对小型总体/封存收据，**未重新校验全部大型历史归档**。正式提交前仍需逐项验证。

## 3. ViNT：本轮实际完成的修复与验证

此前执行审计已经指出：旧 `vint_server.py` 将 JPEG 解码为 RGB 后转成 BGR，
`vint_agent.py` 却直接用 `PIL.Image.fromarray` 和 RGB normalization。
当前图、目标图、context-only replay 都受影响。

新增独立入口 `MemNavData/vint_rgb_input_bridge.py`：在 agent 边界恢复 RGB，
同时覆盖 `step_imagegoal`、`observe`、`step_nogoal`。保留权重、resize、
normalization、预测距离、原有 `distances > 7.0` 轨迹掩码和轨迹生成器。
只用于新版对照，未修改原 server、原 agent 或生产服务。

真实权重 CPU 对照的两条路径：

```text
6帧正确RGB context + 当前RGB + goalRGB → 原ViNT
同一组输入经旧server式BGR交换 → 新bridge恢复RGB → 同一个原ViNT
```

结果：waypoints 与 trajectory 的最大绝对差均为 **0.0**，最终 context 逐像素相同。
这确认颜色修复在 agent/model 边界的等价性；不是 HTTP 端到端或 Habitat 闭环验证，
也不证明修复后 SR 一定提高。

- 权重 SHA：`155fd72de2e98ae0e2fef9404072e1aefa79dae5f7f2411d4bcf7e384b83aa1f`。
- 原 server SHA：`5d7737984ddbcb2f81c7cc0d0e3ccf8719e5390c9d383c98dbf88399e0d1e77a`。
- 原 agent SHA：`3e31861250cd473f2d1cb019907b1660496d4ea61084e334a2b7e5feec0c141a`。
- 结果目录：`.diagnostics/dino_imagegoal_pilot_20260910_78edUj/vint_rgb_parity_cpu/`。
- 核心脚本：`MemNavData/audit_vint_rgb_bridge_cpu.py`。

环境注意：应调用 `.diagnostics/controller_portability_20260821/envs/vint/bin/python`。
虽然该解释器是 symlink，直接调用其 base conda 目标会丢失 venv 的包路径。
本轮第一次用 base 解释器在导入 `efficientnet_pytorch` 时失败，之后改用已有 venv 即通过；
未安装/升级任何共享依赖，没有把导入失败计作导航失败。

## 4. 提交前仍需完成的最小工作

1. 将新 ViNT RGB 入口接入现有 portability hub，并采用修复版 bounded execution。
   当前 NavDP 诊断 wrapper 有专用 depth/receipt 假设，不能直接改 controller 名就当作 ViNT 集成。
2. 本机固定小样本完成 HTTP、context replay、真实转向、重新观测规划与两臂整段导航；
   检查 GEM reject 时仍走同一 ViNT native 请求，而不是改用 NavDP。
3. 统一采用一次 Habitat `try_step` 碰撞响应、正确零/短轨迹处理；不恢复自定义落点投影/30%重试。
   仍是运动学仿真及 GT 距离评分，不称真实机体避障或自主视觉 STOP。
4. 保留两个 controller 的既有接口区别：NavDP 接认证 bearing 的 PointGoal residual；
   ViNT 接认证历史 anchor ImageGoal 与物理朝向适配。不得把 ViNT 描述为原生 PointGoal policy。
5. 小测运行通过后再冻结代码、实际输入、预算和整块 arm 顺序；HPC exact 环境预检通过后提交。
   同 query 的两臂必须同节点/同模型服务配对，报告原始目标 SR 与实际位移 SPL。

这里的放行条件是接口/执行正确，不是“GEM 必须赢”。600 ticks 可保留原表预算，
每元素时长由完整小测实测决定；按 HPC 手册显式指定分区、account、QOS 和时限。
不直接重用绑定旧 source bundle 的 `submit_hm3d_table1_controller_portability_hpc.sh`。

## 5. 本轮边界

目前 HPC 只继续既有 coverage 四臂数组；本机 DINO-imagegoal 小测另见
`DINO_IMAGEGOAL_LOCAL_STATUS_20260910.md`。没有新开 Table I 作业、没有修改论文表格，
没有更换默认 GEM、停止真机服务或提交 Git。
