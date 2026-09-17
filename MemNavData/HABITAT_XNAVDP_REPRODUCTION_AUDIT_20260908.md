# X-NavDP 复现边界与 RGB 适配错误审计（2026-09-08）

## 结论

后续完成情况见 `HABITAT_XNAVDP_RGB_REPAIR_RESULT_20260908.md`：四历史 RGB 对照已闭合并验证；
修复已从私有诊断入口同步到共用 X 服务。不是仅在测试包装中修复。

**不能把上一轮称为“严格复现 X-NavDP”。** 使用了官方 posttrain actor、Q/RTC 和 MPC 源码，
但运行的是迁移到我们任务上的 Habitat 物理诊断，而非原作者 Isaac 的完整评测。
本轮进一步确认了一个非预期错误：X actor 的 RGB 红蓝通道交换。
先修复这个输入错误再判断能力；不能用旧结果给 X 下性能结论。

## 已确认错误：两端编码约定没有配套

官方本地源码固定于 `.diagnostics/xnavdp_official_878740a2011856d0/NavDP/`，
commit `878740a2011856d0e3782dd6ccd880fd2eccd70f`。

官方链路：Isaac `camera_rgb_raw_data` 的 RGB 数组 → OpenCV `imencode` →
服务端 PIL RGB 解码 → RGB 转 BGR → actor。OpenCV 对 raw RGB 的编码解释与服务器转换
相互抵消，actor 最终拿到 RGB 顺序。

本项目旧链路：Habitat RGB → PIL 正常 RGB JPEG → 同样的服务端 RGB 转 BGR → actor。
少了客户端对应交换，actor 实际拿到 BGR。问题同时影响短期历史回放与当前帧。

通过实际官方 `eval/src/client_utils.py::pointgoal_step`（仅拦截 HTTP，不改编码）和
`eval/src/policy_server.py::navdp_step_xy` 的图像解码语句测试：

- 原始像素 `[220,60,20]`：官方 actor 保持 `[220,60,20]`。
- 本项目旧 actor 约 `[20,60,219]`，通道反转已证实，不是对失败轨迹的猜测。
- 修复后 actor 与相同 PIL JPEG 的 RGB 解码逐像素一致。
- 四个实际查询首帧也通过；JPEG 编解码库不同，不能声称与官方 JPEG 位级一致。

离线证据：`.diagnostics/habitat_physics_executor_20260908/xnavdp_rgb_contract_v1/summary.json`。
测试：`audit_xnavdp_observation_contract.py`、`test_xnavdp_rgb_contract.py`。

初次修复在 X 私有单目诊断适配器解码后恢复 RGB；随后已将正确解码落实到共用 X 服务，
私有包装默认不再交换通道。LingBot/CEC 的原 JPEG 字节和深度交易绑定不动。
旧 `legacy_bgr` 仅用于配对复现，不根据场景、结果或置信度切换。

上一轮 verifier 证明了测量、初态、证书及传输图像 SHA 的配对，却没有验证 actor 的通道约定。
**传输 SHA 相同不等于模型实际观测语义正确。** 新测试补齐的是这个明确缺口，不宣称全部接口从此无误。

## 已对齐与未严格复现的项目

| 项目 | 本轮检查 | 结论 |
|---|---|---|
| X actor | 官方 posttrain 权重，1329 个模型张量全部覆盖，无缺失/形状不符；8 candidates | 保留官方核心 |
| 目标预处理 | 保留负 forward，官方 radial cap；不采用 base NavDP 的负向裁剪 | 保留官方核心 |
| 选择/短期记忆 | 官方 Q、RTC、历史大小、机器人状态坐标转换；额外刚体变换测试通过 | 保留核心，理想里程计需披露 |
| MPC | 直接加载官方 `BatchMPCController`；`BatchMPCNEWController` 在公开 eval 中只是兼容别名 | 不是另写一个伪 X MPC |
| MPC 限幅 | 本机匹配对照用 0.376 m/s、π/4 rad/s；公开 wheeled 配置为 0.5、0.5 | 有意改配置，非原配置复现 |
| 执行节奏 | 本机固定每 8×0.1 s 命令重规划，推理时暂停物理；官方异步线程更新动作队列 | 非官方完整时序，非实时性验证 |
| RGB | 旧代码 BGR；本次恢复 RGB | 确认并修复的适配错误 |
| 深度 | 本机 LingBot 单目深度；官方 RGB-D PointGoal | 有意改变输入分布 |
| 目标 | 本机 CEC 方向固定投影 2.5 m；官方当前真实相对 PointGoal | 有意改变目标契约 |
| 具身/动力学 | 本机 Habitat Bullet 圆柱、速度伺服；使用 wheeled embedding | 不是 Dingo、Go2 或 Isaac 原始具身复现 |
| 相机 | 本机既有相机配置；没有完整复刻各官方具身相机 | 域差异不能归成 X 模型退化 |
| 历史 A | 旧 actual-online metric A；只在查询段做单目物理执行 | 不是 full-mono、full-physics A |
| GT | 无 NavMesh 运动投影；低层仍用理想位姿，到达由 evaluator GT 判定 | 非完全无 GT 系统，GT 不给目标方向 |
| 到达 | 本机 <1 m；官方 eval reward 使用 <0.5 m 且速度 <0.25，启动具身缩放后的到达计时 | 非同一成功定义，不能横减 published SR |

官方来源：[X-NavDP 论文](https://arxiv.org/html/2607.28560v2)、
[发布代码](https://github.com/InternRobotics/NavDP/tree/878740a2011856d0e3782dd6ccd880fd2eccd70f/baselines/x-navdp)。
本审计只把直接查到的实现作为事实，不把论文数字当作本迁移条件的保证。

到达定义进一步查到：YAML 虽写 `arrival_threshold: 1.0`，实际
`reward_utils.py::goal_arrival_reward_eval` 使用 `curriculum_utils.py` 中的
`DEFAULT_DISTANCE_THRESHOLD=0.5` 和 `DEFAULT_VELOCITY_THRESHOLD=0.25`。
因此不能仅根据 YAML 声称我们的 1 m 与官方相同。官方另有到达延时，并非一次距离比较。

## 旧结果怎样保留

`xnavdp_stack_v2` 四个 X 查询为 3 到达、1 未到达；它是**旧 BGR 条件**的真实测量，
不是 RGB 修复后结果。原 NavDP 转身臂有一条 5° 代理失效，不能删样本后称 100%。
原目录、视频、原 independent verifier 均保留，不覆盖或混合进新结果。
旧 X 的 77 次查询都被 CEC 授权、都走 X，没有实际 native fallback；
但架构仍保留 CEC 证据不足时的 native 分支，不能说整个方法不存在 fallback。

## 下一步与范围

先做同四历史 X-only 的旧/新颜色配对；协议为
`HABITAT_XNAVDP_RGB_REPAIR_PROTOCOL_20260908.md`。
不要同时加入 GT bearing / metric-distance 臂，不要依据新 SR 调 MPC 参数。
只有输入语义正确后，才有必要解释还剩多少目标距离、路线、物理或深度限制。

潜在关联：base NavDP 服务端也存在 RGB→BGR，而本项目 evaluator 用 PIL JPEG。
本次未完成 base NavDP 上游全链路及训练输入的独立核对，不能直接推广错误结论或静默修改主方法。
将其列为独立后续核查；本次只比较 X 与自身，避免把不同修复程度当作算法胜负。

本轮修改了共用 X 图像适配，但不变更 actor 权重、NavDP/LingBot 策略、论文数字、真机服务或 HPC，不提交 Git。
