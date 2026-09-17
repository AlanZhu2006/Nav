# 前向 PointGoal 域适配：本机候选修复协议

日期：2026-09-08。状态：四臂短测与四历史 16 臂完整查询均完成并核验；视频齐全。
结果见 `HABITAT_FRONT_GOAL_BRIDGE_RESULT_20260908.md`。下文保留原冻结设计。
这是读取执行器配对后设计的消耗型机制试验，不是新场景确认或预期增益承诺。

## 为什么还需要这一层

`process_pointgoal` 把负 forward 裁成 0。只修 tracker 的零参考保持，并没有恢复原目标的方向语义。
域适配只解释已经发出的 PointGoal：前方目标不变；后方目标先用限幅转向动作改变真实仿真视角，
在新画面重新定位/规划。它不更换 NavDP，不增加 critic 门、certificate 门或失败搜索。
不能用虚拟旋转旧图像、旧轨迹来代替转动。

“后向”严格取输入分量 `<0`，来自模型现有前向输入域，不从 SR 拟合角度门限。
每次提交目标请求时都可触发，不限于被事后挑出的两个失败，也不读取 Novel/Revisit role。
方向在一次转向期间锁定于低层 heading 参考，不因每帧重抽样左右切换。

## 固定设计

四个既有、已消费历史，保持原顺序：gxdoq、pLe、yqst、mJX。
每历史四臂，第二、四历史反序：

1. raw fixed memory / bounded standard / corrected RGB / source-raster mono / heading off；
2. 同上 / heading on；
3. CEC / 同一输入执行配置 / heading on；
4. CEC / 同一输入执行配置 / heading off。

共 16 个完整查询，分母四历史；不加入 native 来扩大本轮主比较。
本轮回答各记忆来源的 off/on 效应，不拿上一进程 native 的终局拼成新的严格配对。
raw 使用已有 `phase + raw_fixed_bearing_v1` 查询臂，对所有查询一视同仁，不接收 runtime role。
两种记忆来源使用完全相同的域适配，避免只给 CEC 增加控制能力。

- 共享旧 actual metric-NavDP-A；不称新 full-mono-A 或新执行器端到端历史。
- 上限 600 tick，包含所有转向；最大每 tick 4.5°，继承既有角速度限制。
- 位置保持、yaw 按限幅命令逐动作更新。每次转向都有新 RGB；没有一步跳转 160°。
- 当前是 Habitat 运动学圆形代理，转动期间无平移；这不验证 Go2 的全身转动扫掠安全。
- 原候选、证书和 2.5 m 不变；后方规则不依赖候选是否置零，也不依赖 critic。
- 转向期间每帧进入 LingBot；仅原定每 8 tick 的决策帧重放到 NavDP FIFO，不做 diffusion。
- 转向结束后的新画面必须重新规划，不执行转向前的参考。
- 600 tick、原停滞退出和 GT 平面 1 m 到达均保留；不是自主 STOP。
- 始终使用已确定的正确 RGB 与 source-raster depth，不根据深度消融的 SR 临时选旧输入。

两臂共同收紧模型输入边界：从送给 MemNav 的表单移除旧通用 executor odometry 字段。
当前 endpoint-bearing 估计并不消费它们，但不再仅以“收到后未使用”作为隔离手段。
理想位姿仍保留在低层 tracker 和 evaluator 中，不称整个仿真无 GT。
原始/实际发送字段与删除字段名称会记录，且对 on/off 两臂相同；原图像、goal 和深度交易不变。

## 诊断实现的明确代价

为沿用同一请求与日志链，当前私有 hook 在原请求返回后识别后向 PointGoal。
因此首个 NavDP sample 仍会计算、存档和计入计划数，但其轨迹明确不执行。
这没有伪造模型返回，也没有隐藏额外计算；不能将当前实现称为无冗余推理的部署优化。
实际单步转向仍消耗总动作预算，仿真推理等待不推进世界时钟。

## 核验

先跑首历史四臂、64-tick 上限的接口短测，包含完整转向和转后重规划；短测不计入完整 SR。
完整试验保留：

- 每个动作类型（轨迹跟踪或 heading）、前后状态与原请求；
- 初始共享 A、首张 RGB、同策略首条模型输出；
- 完整规划、单目深度原始/转换后 tensor；
- 每帧 LingBot 已产生的 pose，只读采样，不额外 forward 或位姿纠正；
- 转向期间 FIFO 重放次数、无重复规划/append；
- 首次转后 PointGoal 与相机朝向估计变化、平移估计变化；
- 实际路径/SPL/终点，全部成功和失败第一视角视频。

纯转动时 LingBot 仍可能估计虚假平移；相机高度先验不能消除这一风险。
若仍失败，将按记录区分表示、几何与局部控制，不恢复零参考下的隐式前进。

范围：不触碰生产默认、论文、HPC 或真机。等当前 depth-raster 冻结任务结束后才扩展公共私有 runner。

## 启动记录

深度八臂已全部完成、核验并渲染，之后才修改私有 runner。
`front_goal_smoke_v1` 在第一个查询初始化时被空输出目录检查拒绝：诊断 hook 提前写入了执行循环副本。
没有发生策略查询或动作，不能算导航失败。保留原目录，修复为初始化完成、进入查询时才物化私有循环；
新增对应回归测试后，在新目录 `front_goal_smoke_v2` 重新短测。未放宽原 evaluator 的空目录要求。

`front_goal_smoke_v2` 完成 4/4：raw、CEC 的 off/on 首次状态、完整模型输出分别配对；
两种 on 臂都用 35 动作完成约 157° 转向，在动作 35 以新图重规划。
LingBot yaw 与实际差约 1.31° / 1.26°，但仍有约 0.381 / 0.356 m 的尺度化虚假平移；
转后目标均落到前方，不能由这个短测推断后续 SR。
转向期间每帧 append 一次、仅 8/16/24/32 帧进入 NavDP replay，没有 diffusion。

64-tick 短测不覆盖原 150-tick 停滞窗口。完整试验前的逐分支审查补上了 pending-turn 分支
相同的实际位置停滞退出条件，防止 `continue` 跳过原评测退出；不增加转身专用预算。
新增两个 CPU 分支测试验证应退出及不应退出，原短测 64 tick 的轨迹不受此条件影响。
完整试验将使用新的源码快照 `front_goal_bridge_v1`，不是覆盖短测结果。

完整试验启动命令：

```bash
PYTHONPATH=/home/asus/Research/Nav-graph-blind \
/home/asus/miniconda3/envs/memnav/bin/python -u \
MemNavData/run_habitat_minimal_repair_local.py local \
  --study front_goal --histories 4 --max-steps 600 \
  --out .diagnostics/habitat_minimal_repair_20260908/front_goal_bridge_v1
```

启动前统一入口 `run_habitat_contract_preflight.sh` 实跑 146 passed / 12 dependency warnings。
后向模块及接入检查为其中 21 项，不将单元数量当导航样本数量。
