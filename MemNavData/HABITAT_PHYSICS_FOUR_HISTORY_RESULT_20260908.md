# 四历史无 NavMesh 运动投影诊断（2026-09-08）

## 最终状态

原定 4 个历史 × 4 臂均已尝试完成：12 个正常结束、4 个触发代理失稳。
其中正常结束包括到达和未到达，不等于 12 个成功。
失稳后补齐其他臂的两个历史，重复 native/PP 的全部规划和动作文件均与第一批逐字节一致。

| 策略 / 跟踪器（各 4 个历史） | 到达 | 正常结束但未到达 | 代理失稳 |
|---|---:|---:|---:|
| native / PP | 0 | 2 | 2 |
| CEC / PP | 3 | 0 | 1 |
| CEC / MPC | 2 | 2 | 0 |
| native / MPC | 0 | 3 | 1 |

这是全部计划对象的状态分布，不把失稳臂赋成 SR=0，也不剔除它们后报告完整病例 SR。
仅 4 个既有、已消费历史，没有总体显著性/等价性推断。两个没有失稳的历史中，
两种 tracker 下均保留 native 未到达、CEC 到达的差异；其余历史暴露了接触和零轨迹执行问题。

**最关键新发现：旧 PP 在零平移参考下仍生成最低速前进；官方 MPC 不会这样做。**
两个后向 CEC 查询都在 MPC 臂连续输出零平移参考并停滞，且首次估计方向本身接近 GT 直线方向。
所以下一步应先解决后向目标/零轨迹执行语义，而不是把“换 MPC”当成通用提升方案。

## 结论范围

本轮只回答：在同一 Habitat Bullet 动态接触执行下，CEC/native 的差异是否仍出现，
以及换成官方 base NavDP MPC 是否比原 pure-pursuit 更好。不是论文正式 SR，
不是 IsaacSim Dingo / Go2 动力学复现，也不是自主 STOP 验证。

运动没有调用 NavMesh 做落点投影、缩步或方向修正。仿真仍用场景碰撞网格计算接触，
仍有理想位姿供跟踪和评测，到达仍为 GT 平面距离 <1 m；不能称整个实验“无 GT”。
共享的 A 是旧 actual-online metric-A，查询阶段才是单目控制。

## 第一批：两个完整四臂历史、两个失稳历史

原有第 0 条不重跑；本轮补原清单剩余三条。四臂顺序和参数均未调整。

| 历史 | native / PP | CEC / PP | CEC / MPC | native / MPC |
|---|---|---|---|---|
| gxdoqLR6rwA（复用已完成） | 未到达，600 命令 | 到达，125 命令 | 到达，131 命令 | 未到达，600 命令 |
| pLe4wQe7qrG | 第 48 命令失稳 | 未运行 | 未运行 | 未运行 |
| yqstnuAEVhm（本轮新增） | 未到达，200 命令 | 到达，122 命令 | 到达，115 命令 | 未到达，197 命令 |
| mJXqzFtmKg4 | 第 270 命令失稳 | 未运行 | 未运行 | 未运行 |

“失稳”严格指任何子步倾斜达到运行前固定的 5°；本次分别为 5.853°、5.026°。
它是这个物理代理的有效性中止，不赋值为方法 SR=0，也不是成功。
原 runner 在一个臂异常后退出整个历史，所以其余六臂当时确实没有运行。
不能把剩下两个完整历史包装成四历史总体结果，也不能隐去失稳案例。

两个完整历史各自独立复算通过：同一落地状态、同一首张 RGB，
同一策略的 PP/MPC 首条所选规划一致；实际路径、最终落点、成功标签和命令幅度通过核验。

### 新完整历史的实际路径

| 策略 / 跟踪器 | 到达 | 命令数 | 实际平面路径 | 最终目标距离 |
|---|---:|---:|---:|---:|
| native / PP | 否 | 200 | 1.8179 m | 4.4465 m |
| CEC / PP | 是 | 122 | 3.6200 m | 0.9739 m |
| CEC / MPC | 是 | 115 | 3.8694 m | 0.9936 m |
| native / MPC | 否 | 197 | 1.7748 m | 4.4468 m |

600 是最大命令预算，不是强制执行满 600。native 的两臂触发原有停滞终止，
没有修改该规则来延长 CEC 或缩短 native。

## 从逐动作记录能看出什么

在 yqstnuAEVhm 上，使用事后统一的描述规则“前进速度 >0.1 m/s、该 0.1 s 命令实际位移 <5 mm”：

- native / PP：141 次；
- native / MPC：109 次；
- CEC 两臂：均为 0 次。

这项统计没有参与控制，是实际命令与实际位移的比较，不是把命令速度积分当路径。
接触记录也显示 native 持续有较高机体位置的受力接触。
CEC 在该例改变方向后仍能推进；仅换 MPC 没有把 native 恢复为到达。
这不证明所有失败都归因于路线，也不证明 CEC 无碰撞：CEC 本身仍有接触记录。

场景是一个统一扫描碰撞 stage，日志中 `obstacle=false` 来自未提供独立障碍 ID 列表，
**不能据此称“没有碰撞”**。离线高度/力筛选只是描述，不是家具类别或正式碰撞率。
原记录缺接触法向，因此不把所有地面接触算障碍碰撞。

首轮失稳视频中，pLe4wQe7qrG 的代理已靠近柱体，mJXqzFtmKg4 则处于桌椅之间。
该视觉检查支持继续分析完整机体接触，而不是把所有失败归为深度完全看不到障碍。
但第一视角本身不能区分预测路线碰撞与跟踪偏离，更不能替代正式机体碰撞检查。

## 第二批覆盖补充

为完成本来安排的四臂，对两个中断历史单独重开同样的四臂进程组。
调度仅改成“该臂失稳退出后仍运行下一臂”；实际 `evaluate()` 函数 AST 完全相同，
其他模型/物理源码哈希不变，5°、2.5 m、速度、重规划频率不变。
第一批结果保留；第二批不是新独立样本，也不是从多次运行中择优。

输出：`.diagnostics/habitat_physics_executor_20260908/complete_arms_v2/`。

当前已确认 pLe4wQe7qrG 的重复 native/PP：完整 `plans.jsonl` 和
`physical_actions.jsonl` 与第一批逐字节一致，仍在第 48 命令失稳。
该历史第二批四臂已全部尝试：

| 臂 | 结果 | 命令数 | 说明 |
|---|---|---:|---|
| native / PP | 失稳，5.853° | 48 | 逐字节复现第一批 |
| CEC / PP | 失稳，5.973° | 35 | 不能算成功或填作正常 SR=0 |
| CEC / MPC | 未到达 | 151 | 实际仅移动 0.0155 m，停滞退出 |
| native / MPC | 失稳，5.871° | 69 | 更换 MPC 并未避免本例失稳 |

mJXqzFtmKg4 也已补齐：

| 臂 | 结果 | 命令数 | 实际路径 | 最终目标距离 |
|---|---|---:|---:|---:|
| native / PP | 失稳，5.026° | 270 | 7.2052 m（截断前缀） | 不赋导航结束指标 |
| CEC / PP | 到达 | 124 | 3.6227 m | 0.9798 m |
| CEC / MPC | 未到达 | 151 | 0.0111 m | 3.0045 m |
| native / MPC | 未到达 | 479 | 10.8109 m | 2.1567 m |

没有把第 270 命令之后未运行的 native/PP 假想成失败或成功。
MPC 在本例避免了代理失稳，但没有到达目标，不能将“更稳定”表述为“SR 更高”。

### 新确定的问题：零平移轨迹的执行语义不同

pLe4wQe7qrG 的 CEC 首次实际送给 NavDP 的 point goal 是
`[-2.4207269510, -0.6245646713]`，长度 2.5 m，处于机器人后侧。
初始 bearing alignment 按原配置为 off。PP/MPC 收到的第一条 NavDP 轨迹完全一致，
24 个点的 x、y 全部为零；这不表示已到达（评测距离约 3.26 m，`navdp_stop_evidence=false`）。

两种跟踪器对此表现不同：

- 原 PP 的速度式 `v_max × (0.48 + 0.52 × (1 + cos(alpha))/2)` 不随路径长度归零。
  本例首条命令仍是 0.3729 m/s，实际移动 3.47 cm。此后新视角又产生非零预测，
  但最终碰到物理代理失稳；不能将首条移动称为忠实执行零平移参考。
- MPC 根据零位移参考求解，首条速度仅约 0.0000296 m/s，角速度近零。
  后续 19/19 条所选轨迹的 x、y 都为零，最终 151 次命令停滞。
  **该 CEC/MPC 失败不是“前进命令被障碍挡住”，而是没有收到非零平移参考。**

所以本轮识别到的差异不只有 NavMesh：还包括零轨迹时 PP 隐含的最低速推进。
这是实际记录与现有控制公式直接对上的局部证据，不是所有历史失败的统一解释。
不能将该零轨迹诊断自动扩展成“NavDP 想停车”、也不能用 GT 到达来控制下一步动作。
本轮不从结果中调整控制；后续应先统一“零参考/后向 goal”的接口语义，再决定扩大物理 SR。

mJXqzFtmKg4 独立出现相同的闭环模式：初始 point goal 为
`[-2.4916430034, 0.2042428541]`（175.314°，2.5 m）。CEC/MPC 19/19 条规划的
平移点全零，151 次命令仅移动 1.11 cm；CEC/PP 则有 2/16 条零平移规划，
期间仍移动了约 0.2818 m，后续到达。该 PP 结果并不是完全由非零 NavDP 轨迹解释的执行过程。

为区分初始定位方向错误和后向输入问题，只在运行结束后将估计 bearing 与评测器真实目标
**直线方向**对照（不是测地线路线首段，未将 GT 回送模型）：

| 场景 | 首次 CEC bearing | GT 直线 bearing | 绝对角误差 |
|---|---:|---:|---:|
| pLe4wQe7qrG | −165.533° | −164.722° | 0.811° |
| mJXqzFtmKg4 | +175.314° | +176.828° | 1.514° |

由首条实际位置/yaw 与 `goal_xz_evaluator_only` 计算：
`ego_forward = −sin(yaw)·dx − cos(yaw)·dz`，
`ego_left = −cos(yaw)·dx + sin(yaw)·dz`，再取 `atan2(ego_left, ego_forward)`。
两例 `navdp_stop_evidence` 均为 false。证据支持“首次方向基本正确但在后方、随后得到零平移输出”，
不支持把这两例的 MPC 停滞先归为 LingBot 漂移，也不是证明全程定位无误或所有后向目标必然失败。

### 完整核验

- 全部已尝试臂的实际初始化/首张 RGB 对齐，同一策略的首条所选规划对齐。
- 12 个正常结束臂从动作记录复算路径、终点和成功标签；其角色隐藏、单目深度及 shared-A 收据通过。
- 4 个中断臂只检查已保存动作前缀，不伪造最终成功标签或缺失的正常结束收据。
- 两个补齐历史与原始尝试的 `evaluate()` AST 一致，所有冻结源码快照通过哈希校验。
- 重复 native/PP 的规划和物理动作文件全部逐字节复现；没有选择更好的复跑结果。
- 16 个不同场景/臂组合均有第一视角视频；中断前缀也保留，并非只导出成功视频。

总审计入口：

`.diagnostics/habitat_physics_executor_20260908/complete_arms_v2/combined_diagnostic_audit.json`

该文件保留 `histories`（第一批原始状态）和 `completion_attempts`（两个覆盖补充），
最终覆盖统计见 `coverage_status_counts_after_completion`，不是把重复尝试当成 6 个历史。

## 已有视频与证据入口

所有路径相对于项目根目录 `/home/asus/Research/Nav-graph-blind`：

- 第一批全状态及逐动作审计：
  `.diagnostics/habitat_physics_executor_20260908/remaining_three_v1/offline_motion_audit.json`。
- 完整新历史、四个第一视角视频：
  `.diagnostics/habitat_physics_executor_20260908/remaining_three_v1/history_02_yqstnuAEVhm/first_person_v1/`。
- 柱体附近的中断视频：
  `.diagnostics/habitat_physics_executor_20260908/remaining_three_v1/history_01_pLe4wQe7qrG/aborted_first_person_v1/native__pure_pursuit__aborted_partial.mp4`。
- 桌椅间的中断视频：
  `.diagnostics/habitat_physics_executor_20260908/remaining_three_v1/history_03_mJXqzFtmKg4/aborted_first_person_v1/native__pure_pursuit__aborted_partial.mp4`。
- 两个补齐历史的全部四臂视频，包括 MPC 停滞与 CEC/PP 到达：
  `.diagnostics/habitat_physics_executor_20260908/complete_arms_v2/history_01_pLe4wQe7qrG/first_person_v1/`；
  `.diagnostics/habitat_physics_executor_20260908/complete_arms_v2/history_03_mJXqzFtmKg4/first_person_v1/`。

视频根据保存的实际位置/yaw 重渲染，首张 RGB 与记录核对；没有重新调用策略。
10 fps 对应每命令 0.1 s 的模拟时间，不包含模型推理等待，不能据此估算部署实时速度。
失稳臂最后一帧是该命令后的重建视角，不声称策略实际看过该越界状态。

## 当前不做的外推

- 不由两个完整历史宣布“去掉投影完全无影响”或“CEC 总体 100%”。
- 不由一个代理失稳宣布原论文所有 SR 作废。
- 不把纯跟踪 MPC 当成含障碍约束的绕障规划器。
- 不为过关而放宽倾斜阈值、改变机体/相机参数或恢复 NavMesh 运动保护。
- 本轮不改论文，不提交 HPC，不修改/控制真机，不 commit/push。

## 下一步：先小范围处理接口，不直接扩大正式 SR

1. 单独定义并验证零平移参考的执行语义：没有可执行平移不应被跟踪器暗中改成持续前进，
   也不能仅凭零轨迹宣布到达。核对官方实现与现有控制公式，避免再混淆跟踪、探索和 STOP。
2. 在这两个已暴露问题的后向目标上，测试将已认证 bearing 转入可执行朝向后再交 MPC。
   使用已实现的非 GT 转向接口、记录实际转向代价；仍需验证，不能在修复前预测 SR。
3. 接触/代理有效性是另一问题：不放宽 5° 来过关，也不把高圆柱当作真实底盘。
   若要进入正式物理导航评测，需明确采用的稳定底盘/约束模型和完整碰撞语义。

上述是下一项实验建议，本轮没有执行新的转向干预或替换正式 PP/MPC。
本轮自建 21680/21681 模型服务已退出；18888/8888 真机驻留服务未关闭或重置。

## 代码入口

- `run_habitat_bullet_query_local.py`：`--history-index` 选择同池历史；
  `--continue-after-invalid-physics` 只补齐被失稳连带跳过的其他臂。
- `run_habitat_bullet_remaining_local.py`：第一批顺序运行，保留首条旧结果。
- `summarize_habitat_bullet_extension.py`：原始/补齐尝试分开，独立复算运动与零轨迹统计。
- `render_habitat_bullet_partial.py`：重渲染中断历史，不调用模型、不填成功标签。
- 原 `verify_habitat_bullet_query.py` 和 `render_habitat_bullet_query.py`：用于完整四臂历史。
