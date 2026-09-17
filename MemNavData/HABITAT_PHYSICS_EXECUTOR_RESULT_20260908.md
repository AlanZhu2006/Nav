# Habitat 物理执行迁移：本机进度与证据（2026-09-08）

## 当前结论

已经在独立环境中跑通 **官方 base NavDP MPC → 速度命令 → Habitat Bullet 动态刚体**。
不调用 NavMesh 修正运动；但这是圆柱碰撞代理，**不是 IsaacSim Dingo 或 Go2 全身动力学复现**。

- 六项基础运动／碰撞检查通过。
- 官方 MPC 的直线、左右弧线跟踪和墙体阻挡检查通过。
- 四个既有 MP3D 场景起点的零速度落地＋短段前进检查通过。
- 独立脚本复算了原始运动、碰撞和图像差异，`verified=true`。
- 真正 NavDP native / CEC × pure-pursuit / MPC 的首个四臂查询已完成并独立复算通过：
  两个 native 臂均未到达，两个 CEC 臂均到达。**仅 1 history / 1 scene，不是总体 SR 确认。**

本轮未修改正式执行器、论文或真机，没有提交 HPC，没有 commit / push。
已有真机服务保留，不向机器人发送请求。

## 1. 为什么先做这些检查

旧主线执行器是 pure-pursuit 式跟踪加 NavMesh 投影，不是 MPC。
此前本机一条真实 Revisit 查询中，原 snap 与标准 `try_step` 的结果相同：
native 两臂失败，CEC 两臂成功。但该查询没有触发明显贴边修正，且两种执行都使用 NavMesh，
不能由此判断去掉 NavMesh 之后的效果。

本轮把接触交给 Bullet：控制器只给前进／转向速度，碰撞改变实际运动，
而不是把预测落点改成可行走点。场景碰撞网格属于仿真环境，不提供给 MPC 或 CEC 规划。

## 2. 固定的执行合约

| 项目 | 当前诊断 |
|---|---|
| 环境 | Habitat-Sim 0.3.3，Bullet 构建，独立 Conda prefix |
| 代理 | 动态圆柱，半径 0.30 m、高 1.50 m、质量 20 kg |
| 含义 | 对齐旧 NavMesh 的名义导航包络，不代表真实机器人机体 |
| 时间 | 命令 0.1 s；物理子步 1/240 s，每命令 24 子步 |
| 驱动 | 理想平面／yaw 速度伺服，保留重力方向速度；不改写运行中位置 |
| 上限 | 0.376 m/s、π/4 rad/s，由旧单帧幅度换算用于本次对照 |
| MPC | 原仓库 `NavDP/utils_tasks/tracking_utils.py`，N=15、T=0.1 s，应用第 1 索引控制 |
| MPC 输入 | 固定参考轨迹与理想当前位姿；没有障碍或 NavMesh 输入 |
| 相机 | 实际代理底部以上 0.5 m，yaw-only；不模拟机体抖动与曝光延迟 |

每子步设速度并不等于轮胎扭矩／轮地动力学，roll/pitch 角速度也受到这个理想伺服影响。
这层只能检验控制—接触接口，不能据此宣称机器人安全或真实运动性能。

官方实现来源：

- [NavDP ImageGoal evaluator](https://github.com/InternRobotics/NavDP/blob/master/eval_imagegoal_wheeled.py)
- [NavDP MPC tracker](https://github.com/InternRobotics/NavDP/blob/master/utils_tasks/tracking_utils.py)
- [Habitat rigid-object API](https://aihabitat.org/docs/habitat-sim/habitat_sim.physics.ManagedRigidObject.html)

本机 MPC 源码 SHA-256：
`003e606926c03df83a93890cb555f66e1743cd3a836b23eca6dc470f9f51159a`。
复查源码版本应以本轮 manifest 中的实际哈希为准，不依赖未来变化的 GitHub master。

## 3. 六项基础结果

输出：`.diagnostics/habitat_physics_executor_20260908/primitive_v3/`。

| 测试 | 实测 | 判定 |
|---|---|---|
| 自由落体 | 0.35 s 下落 0.5937 m，理论 0.6009 m | 通过 |
| 直行 | 4 s，理论终点误差 4.32 cm | 通过，原 5% 容差不变 |
| 恒曲率转弯 | 4 s，理论终点误差 4.10 cm | 通过 |
| 正面墙体 | 前进约 1.150 m 后被阻挡；最大接触穿透 1.04 mm | 通过 |
| 侧缘细柱 | 922 个子步出现障碍接触；最大穿透 1.21 mm | 通过 |
| 零速度停止 | 停止后额外平面位移 0.79 mm | 通过 |

侧缘碰撞出现真实接触响应和横向运动；不是“物理仿真绝不允许滑移”。
关键区别是接触解算改变运动，而不是策略查询 NavMesh 并选取修正落点。

失败版本完整保留：

- `primitive_v1`：JSON 的 NumPy bool 序列化错误；转弯理论终点还没有绑定实际初始 yaw。
- `primitive_v2`：新诊断脚本先冻结 STATIC 再设置位置，导致墙／地板仍在原点；不能作为正确物理场景结果。
- `primitive_v3`：先放置再冻结，并检查实际初始位置；未放宽任何通过阈值。

以上是本轮新测试脚本的问题，**不是从旧 CEC 主实现中发现的同一个障碍放置错误**。

## 4. 官方 MPC 接触闭环

输出：`.diagnostics/habitat_physics_executor_20260908/mpc_tracking_v1/`。

| 固定参考轨迹 | 求解次数 | 终点误差 | 求解中位延迟 | 结果 |
|---|---:|---:|---:|---|
| 1 m 直线 | 50 | 0.21 cm | 2.56 ms | 跟踪通过 |
| 左弧线 | 50 | 7.44 cm | 2.25 ms | 跟踪通过 |
| 右弧线 | 50 | 7.50 cm | 2.34 ms | 跟踪通过 |
| 指向墙后的 2 m 直线 | 80 | 85.01 cm | 2.35 ms | 墙体阻挡通过，并未到达参考终点 |

最后一行不是导航成功。它说明：即使参考轨迹穿过墙，Bullet 也会阻挡代理；
该 MPC 不会凭空改成绕墙路线，因为它没有障碍约束或全局规划输入。

这些是同一 CPU 设置下的 230 次小规模求解，不包含模型推理、通信或渲染，
不能当作端到端部署延迟。每条轨迹的 p95 约 2.40–2.66 ms。

## 5. 真正场景起点：发现高度对照问题

使用原有四历史 manifest 的全部四个起点，没有新选场景：

| 场景 | 被动落地检查 | 零速度伺服落地检查 | 相机高度相对旧起点变化 |
|---|---|---|---:|
| gxdoqLR6rwA | 通过 | 通过 | −9.57 cm |
| pLe4wQe7qrG | 横向漂移 2.92 cm，未过 2 cm 判据 | 通过 | +1.55 cm |
| yqstnuAEVhm | 通过 | 通过 | −3.63 cm |
| mJXqzFtmKg4 | 通过 | 通过 | +1.72 cm |

零速度落地使用与运行时停止命令相同的伺服，保留垂直重力和物理接触；
不向原点传送，不读取障碍来调姿，也不放宽判据。

NavMesh 表面和实际扫描碰撞地面并不完全重合。相机真实落地后高度变化，
首张图像也变化。因此不能把新物理结果与旧 NavMesh 绝对 SR 相减，
然后全部解释成投影或碰撞执行的影响。

这也不是证明旧 NavDP 高度先验失效：此处比较的是两种仿真地面定义，
不是估计 LingBot 深度的误差。

证据分别在 `scene_starts_v1`（被动）与 `scene_starts_v2`（零命令），
每场景包含完整位姿／接触记录及三张第一视角 RGB。

## 6. 首轮四臂查询：已完成，独立复算通过

首个历史 `gxdoqLR6rwA/episode_0000`，按既有顺序选定，不依结果选择：

1. native + pure-pursuit + Bullet；
2. CEC + pure-pursuit + Bullet；
3. CEC + 官方 base MPC + Bullet；
4. native + 官方 base MPC + Bullet。

四臂共用实际 online metric-NavDP-A 历史，查询阶段使用 LingBot 单目深度。
在查询边界开始物理执行，各臂共享零速度落地初始化并核验首张 RGB。
保持 600-step 上限、8-step 重规划、1 m 平面成功判据及固定 2.5 m residual。

主要比较为**相同物理执行条件下 CEC 对 native**；其次是相同策略下两个跟踪器。
不是新旧执行器无混杂的绝对 SR 比较，不是 full-physics-A，不是自动 STOP。

所有子步保存真实位置、速度和接触，不进行运行中 NavMesh 运动修正。
若代理倾斜超过预先声明的 5°，本条物理诊断判无效退出，不替换成旧执行或无碰撞移动。

首次启动 `query_four_arm_v1` 在模型查询前因继承旧环境的 pip-vendored requests 路径失败；
已改为新环境自身的 requests，并在模型启动前加入解释器导入检查。
`query_four_arm_v2` 随后在读取源 Parquet 时发现 PyArrow 缺失，同样没有执行导航动作。
已用独立 wheel 补齐 PyArrow 14.0.2，并在启动模型前实际读取该历史的 Parquet 和内参。
曾尝试 Conda 求解，但它提出大范围降级依赖，已在环境事务前中断；原 Habitat、NumPy、
OpenCV、CasADi 版本未改变。修复后的当前运行目录为 `query_four_arm_v3`。

### 6.1 逐臂完整结果

| 策略 | 跟踪器，均为 Bullet 执行 | 到达 | 命令数 | 实际路径（命令边界积分） | 终点到目标 |
|---|---|---|---:|---:|---:|
| mono native | 原 pure-pursuit 控制律 | 否 | 600 | 21.3346 m | 18.4705 m |
| mono CEC | 原 pure-pursuit 控制律 | 是 | 125 | 3.9911 m | 0.9923 m |
| mono CEC | 官方 base NavDP MPC | 是 | 131 | 4.2325 m | 0.9881 m |
| mono native | 官方 base NavDP MPC | 否 | 600 | 21.7146 m | 19.9759 m |

这是同一条查询的四次处理，不能按 N=4 计算显著性，更不能称为“总体 100%”。
每个跟踪器中的 CEC/native 比较均为单案例 +1/−0，只作机制检查。

四臂全部 1,456 次命令、34,944 个物理子步；记录了命令后终点，不存在缺失末步坐标问题。
按物理子步积分的路径依次为 21.3357、3.9919、4.2369、21.7187 m，
与上表命令边界积分差异很小；没有将命令速度积分冒充实际路径。

最大机体倾斜依次为 3.82°、2.04°、2.30°、4.71°，未触发预声明的 5° 无效判据。
MPC 实际查询中求解中位延迟为 CEC 3.05 ms、native 2.82 ms，仍不包括模型推理／通信。

### 6.2 配对与独立核验

- 四臂落地后的完整机体状态（位置、姿态、速度）完全一致。
- 四臂首张 RGB 的逐字节哈希完全一致：
  `d71f2df3690f519213e4682ce2774aa4a858fdaac68ea905fc1e056b0fe4cd96`。
- 同一策略在两种 tracker 下的第一条所选 NavDP 轨迹完全一致。
- 共享 A replay 的哈希检查通过；A replay 不重新抽样 diffusion。
- 运行时角色隐藏，查询策略没有消费模拟器 depth。
- 运动函数不接收目标 GT 或障碍查询结果，不调用 NavMesh 修正运动。
- 独立 verifier 从原始子步重算终点、路径、命令幅度、成功标签与机体倾斜，`verified=true`。
- 四臂共用原历史参考深度 `canonical`；没有顺便切换到新的深度缓存实现。

收据：

`.diagnostics/habitat_physics_executor_20260908/query_four_arm_v3/independent_verification.json`

### 6.3 能得出什么，不能得出什么

**本例的 CEC 增益并不需要旧的 NavMesh 运动投影才能出现。**
在动态接触执行下，用纯追踪或原版 MPC 都保留了 native 失败 / CEC 成功的成败差异。

**本例也没有显示换 MPC 就能恢复 native。** MPC 是局部轨迹跟踪器，
它不自动提供目标定位、长期路线或带障碍约束的路径规划。
不能由一次 MPC 路径稍长得出它更差，也不能因“优化控制”更复杂就预设它提升 SR。

仍未证明：

- 旧论文全部 SR 都不受执行近似影响；只有 1 个已消费场景的查询对照。
- 两种物理 tracker 在总体上等价。
- 能穿过真实机器人的所有窄门或绕开薄桌腿；代理不是 Go2 全身结构。
- 对真实模型／通信延迟的鲁棒性：本轮同步推进物理步，计算等待期间没有模拟继续运动。
- 可部署的自主到达／停车：仍使用评测器 1 m 平面 GT 到达事件。
- 全物理、全单目 A 端到端：第一段历史来自既有 metric-NavDP-A。

### 6.4 第一视角视频

四臂均按实际保存的机体位姿重渲染，不重新执行策略，不插值、转向美化或修改成败标签。
每个视频首帧须与运行时 RGB 逐字节一致才输出；播放速度为 10 帧/s，
对应命令 0.1 s 的模拟时间，不对应模型等待在内的墙钟时间。

输出目录：

`.diagnostics/habitat_physics_executor_20260908/query_four_arm_v3/first_person_v1/`

四个视频已全部完成，分别为 601、126、132、601 帧；首帧均与运行时 RGB 哈希一致。
`render_receipt.json` 已生成，记录源位姿文件和输出视频哈希。

本次专用模型服务及 evaluator 均已退出；原真机驻留服务没有关闭或重置。

### 6.5 接下来的最小范围

先按既有四历史 manifest 顺序补另外三个 consumed 场景，以相同 Bullet 执行下的
CEC/native 配对为主，不改 certificate、半径或增添兜底策略。
遇到实际身体代理失稳，先作为物理建模问题报告，不填成方法失败或回退旧执行器。

只有这一步成立，才确定正式论文中哪些表格需要同协议补跑；完整物理端到端还要
重新收集 actual-physical-A。当前不提交大规模 HPC，不用这四条替换现有论文表格。

## 7. 代码、环境与复算

- `MemNavData/habitat_physics_probe.py`：基础驱动／碰撞测试。
- `MemNavData/habitat_physics_scene_probe.py`：四场景实际落地和短段运动。
- `MemNavData/habitat_mpc_tracking_probe.py`：原版 MPC 的固定参考轨迹闭环。
- `MemNavData/verify_habitat_physics_probes.py`：独立数据复算，不加载模型或测试 runner。
- `MemNavData/test_habitat_physics_probe.py`：静态放置／墙体阻挡两项回归测试。
- `MemNavData/run_habitat_bullet_query_local.py`：独立本机模型服务及四臂导航诊断。
- `MemNavData/verify_habitat_bullet_query.py`：四臂独立复算。
- `MemNavData/render_habitat_bullet_query.py`：实际物理位姿第一视角重渲染。

独立环境：

`/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_bullet_env_20260908`

总证据目录：

`/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_physics_executor_20260908`

独立复算收据：`independent_verification_v1.json`，`verified=true`。
MPC 的逐命令终点、控制和接触已复算；子步最大 tilt 没有独立副本，报告明确注明该限制。
新增两项回归测试均通过；所有新 Python 脚本语法编译通过。

旧 NavMesh 审计结果仍见 `MemNavData/HABITAT_EXECUTOR_AUDIT_RESULT_20260908.md`，未被覆盖。
