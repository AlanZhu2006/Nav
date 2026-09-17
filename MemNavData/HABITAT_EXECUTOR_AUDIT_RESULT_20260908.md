# Habitat 执行器本机调查：首轮结果（2026-09-08）

## 结论先行

本机已完成三项检查：原始长距离轨迹的位移可达性检查、固定合成命令压力测试，以及一个真实 Revisit 查询的四臂闭环。闭环结果已由独立脚本复算通过。

**目前可以确定：原执行器使用 GT NavMesh；但不能因此直接断言 CEC 的成功率是由自定义投影或缩步额外抬高的。** 本次唯一闭环案例中，换为 Habitat 标准 `try_step` 后，native 仍失败，CEC 仍成功；同一策略的实际坐标序列和所选规划输出完全一致。

这个案例没有触发明显的贴边修正，因此也不能用它排除困难环境中的执行器影响。`try_step` 同样使用 GT NavMesh，并不是无 GT 或真实机器人动力学对照。

本轮只运行本机独立诊断，没有提交 HPC，没有发送机器人动作，没有改动正式执行器或论文，没有 commit / push。独立诊断服务已退出；现有真机服务未关闭或重置。

## 1. 本次究竟比较什么

原执行器见 `MemNavData/eval_2leg_habitat.py` 的 `pursuit_step`：

1. 按同一 pure-pursuit 控制律计算一次前进与转向，默认最大前进量为 0.0376 m。
2. 对命令终点执行 `snap_point`。
3. 若投影不合法或平面修正超过 0.06 m，尝试 30% 步长。
4. 再次失败则只改变朝向，不平移。

本轮比较：

| 执行方式 | 内容 | 是否控制真实闭环 |
|---|---|---|
| `legacy_snap` | 原函数直接执行，上述投影／缩步规则保持不变 | 是 |
| `try_step` | 保留相同控制律，以当前点和命令终点调用 Habitat 标准碰撞步进，允许滑移 | 是 |
| `try_step_no_sliding` | 相同命令，但禁用滑移 | 否，仅逐步记录对照落点 |

比较中没有更换 NavDP、CEC、目标、步长、转向限制或到达阈值。诊断重建的 legacy 输出每步与原函数比较；legacy 臂直接返回原函数结果，而不是用近似实现替代它。

本次检查的是执行合约，不是新方法的正式成功率确认。标准 NavMesh 步进也不等于 IsaacSim 物理仿真或 Go2 全身碰撞检查。

## 2. 原始长距离轨迹：2,352 次实际位移

对象为既有 HM3D 长距离案例 `eF36g7L6Z9M / 005`，三个臂全部检查。使用与原始结果绑定的 `.basis.navmesh`，没有用另一个不同机器人半径的资产 NavMesh 替代。

NavMesh SHA-256：

`11912dc9b99291f2e08a13f9181a02b8b32822f59696f9a052440111e543817c`

| 已记录策略 | 位移数 | 相对新朝向的侧移 >1 mm | 步长 >0.0386 m | 实际落点与标准步进结果相差 >1 mm | 与禁滑移结果相差 >1 mm |
|---|---:|---:|---:|---:|---:|
| mono native | 973 | 185 | 2 | 0 | 120 |
| CEC endpoint | 626 | 274 | 2 | 0 | 33 |
| CEC route tangent | 753 | 52 | 0 | 0 | 48 |

这里的标准步进输入是“已记录起点 → 已记录实际终点”。因此零差异只说明这些已实现位移可由标准 NavMesh 滑移重现，**不是原始预测命令的反事实重放**。

旧日志保存了所选轨迹的 SHA，但没有保存该轨迹的完整坐标，无法据此还原原始命令。不能由“落点可达”推出“执行器等价”，也不能由侧移次数推算成功率虚增多少。

原记录结果没有被重新解释或改写：native 失败、endpoint 失败、route tangent 成功。

证据：

- `.diagnostics/habitat_executor_audit_20260908/recorded_longrange_005/summary.json`
- 同目录各臂的逐位移记录。

## 3. 合成命令压力测试：21,168 次探针

在上述全部 2,352 个动作前状态上，固定生成半径 2.5 m 的方向目标：

`−180°, −90°, −60°, −30°, 0°, 30°, 60°, 90°, 180°`。

使用同一控制律比较三个执行器。正负 180° 对应重复方向；这里保留固定角度网格，不把这些探针当作独立统计样本。

| 状态来源 | 合成命令数 | legacy 对命令的修正 >1 mm | legacy 与标准滑移相差 >1 mm | legacy 与禁滑移相差 >1 mm | legacy / 标准最大差异 |
|---|---:|---:|---:|---:|---:|
| native | 8,757 | 1,816 | 11 | 1,827 | 3.61 cm |
| endpoint | 5,634 | 2,648 | 13 | 2,445 | 2.81 cm |
| route tangent | 6,777 | 466 | 5 | 494 | 2.31 cm |

结果：

- legacy 与标准滑移在 29 / 21,168 次探针中相差超过 1 mm，约 0.137%。
- 全部探针走 `full_snap`；没有触发 30% 缩步或原地阻塞分支。
- 全部 legacy 实际终点均能由标准滑移从起点到达，误差不超过 1 mm。
- 禁止滑移时差异明显增加，但这还不是禁滑移闭环结果，更不能等同于真实 Go2 的行为。

这证明两种执行规则并非完全等价，也纠正了“本批结果中一直靠缩步脱困”的猜测。0.06 m 投影容限大于 0.0376 m 最大单步长度，值得继续检查它对分支触发的影响；本轮只报告观测到零次缩步，不宣称该分支在所有状态都不可能发生。

**这些是人为固定的合成命令，不是旧 NavDP 命令，不产生导航 SR。** 单步差异少也不能推出长程累计影响小。

证据：

- `.diagnostics/habitat_executor_audit_20260908/synthetic_heading_stress/manifest.json`
- `.diagnostics/habitat_executor_audit_20260908/synthetic_heading_stress/summary.json`
- 同目录各臂 `*_probes.jsonl`。

## 4. 真实四臂闭环：已经跑完，不是离线估计

### 4.1 固定条件

- 在既有、已经使用过的四历史 manifest 中，按顺序选择第一条：`gxdoqLR6rwA / episode_0000`；不是根据本轮结果挑选成功案例。
- 仅测试其自然 Revisit 查询，不测试 Novel 安全性；运行时不读取角色标签。
- 共用真实 online metric-NavDP-A 的 240 帧 RGB 历史。查询阶段使用 LingBot 单目深度，因此不是“actual-mono-A 全链路”实验。
- 历史参考深度固定为 canonical 来源，避免同时改变历史深度缓存实现。
- 同机、同一组驻留模型，各臂按协议 reset / seed；最大 600 步，每 8 次执行动作重规划。
- 固定 2.5 m residual；到达仍为评测器的 GT 平面距离 <1 m，不检验部署自主 STOP。
- 顺序：native/legacy → native/try_step → CEC/try_step → CEC/legacy。
- 本轮保存完整所选规划、候选、逐动作命令、三个执行器落点及最后动作后的实际坐标。

初始最短路距离为 3.4913 m。只有 **1 history / 1 scene**，四条 rollout 不代表四个独立样本。

### 4.2 完整结果

| 策略 | 执行器 | 到达 | 步数 | 实际行走距离 | 终点到目标的平面距离 |
|---|---|---:|---:|---:|---:|
| mono native | 原 snap | 否 | 600 | 22.2249 m | 5.7730 m |
| mono native | 标准 try_step | 否 | 600 | 22.2249 m | 5.7730 m |
| mono CEC | 标准 try_step | 是 | 117 | 3.9950 m | 0.9773 m |
| mono CEC | 原 snap | 是 | 117 | 3.9950 m | 0.9773 m |

逐步检查：

- 全部 1,434 次动作没有 >1 mm 的命令投影修正；没有缩步或阻塞分支。
- 全部动作的 legacy / 标准滑移 / 禁滑移落点均无 >1 mm 差异。
- native 的两条实际坐标序列完全相同；CEC 的两条实际坐标序列也完全相同。
- 每对策略的所有所选规划输出完全一致；共享 A replay、首次规划一致。
- 独立 verifier 从逐动作记录重算完整路径、最终距离、成功标签，并在保存的执行 NavMesh 上复算每条命令；`verified=true`。

因此，这一案例中的 CEC 相对 native 增益**不是由自定义 snap / 30% 重试相对于标准滑移额外造成的**。

但这一案例没有实际触发明显贴边修正，所以它没有回答“窄通道、贴墙或家具密集处，两种执行器是否改变 SR”。也没有排除两种执行器共同的 GT NavMesh 保护对仿真—真机差距的贡献。

### 4.3 证据与完整记录

根目录：

`/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_executor_audit_20260908/closed_loop_v1`

- `manifest.json`：选择规则、控制变量、代码／模型哈希。
- `summary.json`：四臂完整结果，`completed=true`。
- `independent_verification.json`：独立复算，`verified=true`。
- `evaluation/gxdoqLR6rwA/<policy>__<executor>/executor_actions.jsonl`：逐动作命令与对照落点。
- 同臂 `full_plan_outputs.jsonl`：完整预测规划及候选。
- 同臂 `execution.navmesh`：本轮实际使用的 NavMesh。
- 同臂 `terminal_measurements.json`：最后动作后位置和实测路径。

关键哈希：

| 对象 | SHA-256 |
|---|---|
| 原四历史 manifest | `191473c90ab7eefff54c7ae752e2c03bc723ea3415d9b97b42c105b5e62a8848` |
| 本轮 manifest | `7ede7009f1bd4c1d374a712cc46a4ba410a8c23e5429e6feff9d5794036b2465` |
| 本轮 summary | `d991fcef69ad0fbc281f650b3e342e59c758cc63f406b0775c61d889af2b2d64` |

## 5. 对之前判断的修正

| 问题 | 本轮后的准确状态 |
|---|---|
| 执行器是否用了 GT 几何？ | 是；原执行器与标准滑移都用了。 |
| 自定义投影是否等于 NavDP 官方物理执行？ | 不是；本轮也没有将其改造成物理执行。 |
| 是否发现穿越标准 NavMesh 不可达区域的已记录位移？ | 在检查的 2,352 次实际位移中没有发现；不是全项目排除。 |
| 30% 缩步是否解释本次成功？ | 不解释；四臂闭环和本轮合成探针均未触发它。 |
| CEC 的已有正式增益是否因此全部失效？ | 没有这种证据；不能从实现风险直接推导数值作废。 |
| 是否已经证明仿真执行没有问题？ | 没有。贴边闭环、滑移依赖和全身物理约束仍待检验。 |
| 能否据此证明真机安全？ | 不能。全身碰撞、深度时效、速度和停止距离均未在此验证。 |

此前若把“用了 GT NavMesh”直接解释为“CEC 依赖额外智能绕障而成功”，证据是不够的。本次结果要求把实现事实、局部反事实和总体 SR 影响分开报告。

## 6. 下一步最有信息量的本机检查

1. 从上述已记录的命令差异中，预先固定有实际接触／贴边可能的起点与方向，先对齐原命令、三种落点和局部几何；不按新 SR 选择案例。
2. 在固定困难查询上运行相同 native/CEC 配对，增加禁滑移执行条件，记录原计划是否持续压向障碍、滑移是否提供侧向推进，以及成功判定前的实际路径。
3. 只有观察到有实质意义的闭环差异，才决定哪些正式表格需要补跑或调整执行协议。当前不启动全量 HPC 重跑，也不直接替换主方法。

禁滑移仍不是机器人全身物理仿真。即使这组结果也相同，真机近障碍保护、局部路径碰撞检查和自主到达仍应单独验证。

上述为下一步建议，**尚未运行下一轮困难查询闭环**。

## 7. 新增诊断代码与复查命令

- `MemNavData/habitat_executor_audit.py`：执行规则及已记录位移检查。
- `MemNavData/test_habitat_executor_audit.py`：4 个测试，全部通过。
- `MemNavData/stress_habitat_executor_commands.py`：固定角度合成探针。
- `MemNavData/run_habitat_executor_audit_local.py`：独立本机模型服务及四臂闭环。
- `MemNavData/verify_habitat_executor_audit.py`：独立逐动作复算。
- `MemNavData/HABITAT_EXECUTOR_AUDIT_PROTOCOL_20260908.md`：本轮运行前范围与固定条件。

复查已完成结果：

```bash
PYTHONPATH=/home/asus/Research/Nav-graph-blind /home/asus/miniconda3/envs/memnav/bin/python -m pytest MemNavData/test_habitat_executor_audit.py -q

/home/asus/miniconda3/envs/habitat/bin/python MemNavData/verify_habitat_executor_audit.py .diagnostics/habitat_executor_audit_20260908/closed_loop_v1
```

新一轮必须使用新的输出目录，不能覆盖本轮 manifest、日志或结果。已有用户修改均予保留；本轮未整理或提交其他工作线文件。
