# 真机 Revisit 目标生命周期自动化（2026-08-24）

## 结论

此前缺少的不是导航模型，而是目标的控制平面：候选采集、支持度判断、目标安装、
阶段切换和审计收据彼此分离。现在将其收敛为一个可重试的目标提交事务：

```text
memory_recording
  -> 自动候选探测（只读支持查询）
  -> 候选帧注册但不写入长期记忆
  -> 显式任务边界：begin_revisit
  -> RTX 原子执行 score -> select -> freeze goal -> NavDP warm-up
  -> Jetson 校验目标 SHA-256 并在线安装
  -> revisit_query
  -> 每步携带 installed_goal_sha256，CEC/NavDP 只能使用已提交目标
```

机器人短暂停下并不等于任务从 recording 切到 revisit，因此仍保留一次显式
`begin_revisit` 任务边界。它可以由操作员、任务调度器或未来的上层 mission manager
触发；不应由“静止若干秒”这类启发式自动猜测。触发之后不再需要人工评分、拷贝文件、
重启 adapter 或手动安装目标。

## 自动候选

- Jetson 每 24 个已记录记忆帧发起一次候选探测，最多注册 6 张；
- RTX 的 `/goal_candidate_support` 只读端点一次编码目标图，直接与缓存的整段历史
  DINO CLS 比较，再对 top-1 anchor 做 SIFT/epipolar 几何验证；每张候选只允许读取
  其拍摄边界至少 16 帧以前的历史，该因果上界在后续 prepare 复算中保持不变；
- 接受门为 `inliers >= 16` 且 `max_cos <= 0.90`；阈值仍属于待真机无运动标定的
  provisional contract；
- 只有通过的候选才进入 candidate pool；候选 RGB 本身不进入长期记忆；
- 接受后额外跳过 4 个相邻相机帧，降低目标与历史形成近乎完全重复视图的风险；
- 上传响应若不确定，该 RGB 直接丢弃，不能再作为 memory frame 写入，避免网络故障
  破坏候选/历史互斥。

手动 `/capture_goal_candidate` 仍保留，作为受控实验覆盖，不是正常主流程。

## 原子 prepare

RTX `/prepare_revisit` 在同一串行临界区内完成：

1. 重新对全部已注册候选做 frozen support 检查；
2. 只在 `provisional_weak_covis` 集合内，按几何内点数、内点率、DINO 支持度和
   candidate id 确定性排序；
3. 没有合格候选时原样停留在 `memory_recording`，不 warm-up、不安装目标；
4. 选中后执行已有 stride-8 NavDP FIFO warm-up，并严格验证 queue length；
5. 冻结 candidate id、原始 JPEG 和 SHA-256；
6. 一次返回候选分数、选中目标、目标 JPEG、warm-up indices/queue lengths。

该端点是幂等的。如果 RTX 已提交而 HTTP 响应丢失，再次调用会返回同一目标和同一
提交收据，不会打开第二个 goal session，也不会重复 warm-up。

## 在线安装与审计

- Jetson 解码目标 JPEG，复算 SHA-256，通过后在内存中替换 `_image_goal`，无需重启；
- `/navdp/image_goal` 立即显示新目标；
- 每个 `/imagegoal_step` 都携带 `installed_goal_sha256`；RTX 忽略可能陈旧的客户端
  goal upload，始终使用自己提交的目标字节；
- `/navdp/status` 持续包含阶段、帧数、候选数、active goal、完整 begin receipt、CEC
  takeover/reason/anchor 和单目 depth receipt；
- `/navdp/cec_receipt` 以 transient-local JSON 发布 candidate、prepare 和每次 plan 的
 详细收据；
- RViz marker 显示 phase、候选数、active goal 以及最近一次 CEC 决策。

## 代码位置

研究侧新增只读批量支持查询：

- `NavDP/baselines/memnav/policy_agent.py::goal_candidate_support`
- `NavDP/baselines/memnav/memnav_server.py::/goal_candidate_support`

发布/真机侧：

- `/home/asus/Research/Memnav_Realworld/deployment/gpu/realworld_cec_hub.py`
- `/home/asus/Research/Memnav_Realworld/deployment/go2/navdp_client.py`
- `/home/asus/Research/Memnav_Realworld/deployment/go2/navdp_ros_node.py`
- `/home/asus/Research/Memnav_Realworld/ARCHITECTURE.md`
- `/home/asus/Research/Memnav_Realworld/RUNBOOK.md`

## 已完成验证与剩余边界

- 工作站发布栈测试（除本机缺 ROS 的 `test_rgbd_sync.py`）：82 passed；
- Jetson `test_navdp_client.py`：8 passed（禁用不兼容的第三方 pytest plugin
  autoload）；
- Python 语法检查和启动脚本 `bash -n`：通过；
- 覆盖无合格候选不变更状态、近重复候选不注册、原子目标覆盖陈旧上传、目标 SHA
  校验和 prepare 幂等重放。

2026-08-24 camera-only 真链烟测也已完成：

- RTX preflight 全部通过，Jetson 以单入口启动 D435i、SSH tunnel 和禁用态 adapter，
  `go2_bridge=false`；
- `/navdp/status` 实测 `enabled=false`、`phase=memory_recording`、RGB-D fresh、
  `cmd_vx=cmd_wz=0`，并显示新增 auto-selection 参数；
- 自动候选探测收据实测 `frames_total=120`、`frames_swept=16`、
  `scoring_ms=101.58`，静止相机候选得到 `max_cos=0.9983`、114 matches、0
  fundamental inliers，因此 `registered=false`；
- 在候选池为空时调用 `begin_revisit` 返回明确失败，随后状态仍为
  `memory_recording`、active goal 仍为空、速度仍为零；
- 测试结束后 Jetson 与 RTX tmux session 均已关闭，没有残留相机、adapter、策略或
  Go2 bridge 进程。

尚未声称正向真机闭环完成。静止相机无法产生非退化的 Revisit 候选；下一步需要在
adapter disabled、无 Go2 bridge 条件下由手柄完成一次有视角变化的 A→B 录制，验证
candidate acceptance、正向 prepare、目标图在线切换和哈希确认。通过后才做系绳低速
运动。自动到达 evaluator 是独立 termination 层，不应与本次目标生命周期提交混为同一个
安全权限。

2026-08-24 首次有运动录制还发现了一个时间窗缺陷：whole-history top-1 总被仅早 8
帧的近重复图像占据，导致 374 帧中没有候选注册。真实序列离线复算显示 frame 336 在
冻结的 16 帧因果间隔下为 `cos=0.894, inliers=48`，本应通过原门限。现已将每个候选的
支持上界固定为 `captured_after_frame - 16`；后续记录的帧也不能污染 prepare 复算。
同时真机 MemNav buffer 改为每次服务启动使用独立时间戳目录，避免进程内 episode 计数
重置时覆盖上一条 runtime trace。

## 2026-08-24 正向禁用态真链复测

修复后重新由手柄完成真实视角变化的 A→B 录制，全程 `go2_bridge=false`、
`enabled=false`：

- 自动注册 1 个候选；候选拍摄边界为 frame 216，冻结支持 ceiling 为 200；
- prepare 复算时历史已增长到 285 帧，但仍只读取 ceiling 200 及以前；
- 冻结门结果为 DINO cosine `0.85793`、SIFT/fundamental `84/111` inliers，
  inlier ratio `0.75676`；
- `/prepare_revisit` 成功切换到 `revisit_query`，NavDP warm-up 为 8/8 帧，
  `queue_lengths=[8]`；
- Jetson active SHA、RTX committed SHA 和 selected-goal SHA 均为
  `6b834ff1db5b27ca41796a57c1ebfeca4af42c1ad1b842afaf656c6160fd5899`；
- 后续真实 `/imagegoal_step` 成功消费该目标，CEC 返回
  `certificate_accepted`、`navdp_image_point_mix`，scale-free bearing 为
  `[0.9288, -0.3706]`，固定 residual 为 `[2.3220, -0.9264] m`；
- 单目 depth receipt 有效且 `metric_depth_sensor_consumed=false`；
- `/navdp/cmd_vel` 始终为零，未进行底盘运动。

这证明候选采集、冻结、prepare、目标在线安装、SHA 确认和首个 mixed NavDP plan 的
软件控制平面已经正向闭环。但在线 CEC 最终使用 anchor 215，而构造门的非近邻支持
anchor/ceiling 为 200。故该结果只能称为真实传感器 transport/control-plane smoke；在
在线 CEC 也消费同一冻结 ceiling 前，不能将其作为“非近重复 Revisit”论文结果。

## 2026-08-24 Q→R 首次自主 Revisit 及归因

首次系绳低速 Q→R 运行使用预先冻结的 R 图、真实 Q 处当前观测和完整 causal FIFO。
启动前检查为 172 个 LightGlue good matches、134 个 fundamental inliers，CEC 选择
anchor 229；长程 bearing 约为左后方 110.8°，与独立 Q→R 几何方向一致。

该轮机器人自主移动 3.01 m，辅助位姿距离从 3.507 m 降到最低 1.498 m，随后由路径长度
上限自动 estop。正式结果必须记为失败：`safety_abort_path_length_limit`，不能因为末态
图像相似而改写成功标签。停止后的只读末态审计显示：

- 39 good matches、28 fundamental inliers，object-level match 通过；
- exact-view 仅因中心偏移 0.2193 超过 0.18 而拒绝；
- 直接 current→R PnP 仍有 161 inliers，预测距离约 0.695 m；
- 直接局部 bearing 约为 -178°，即目标几乎在机器人正后方；
- 同期长程 LingBot bearing 却约为 -32°。

因此失败不是“历史里找不到 R”，也不是底盘没有运动，而是终端控制合约断裂：长程
history→anchor→goal 位姿链在近目标时漂移；固定 2.5 m residual 不随距离衰减；而冻结
NavDP 的 point-token 在后向 bearing 上已知不可操纵。原始收据位于 Jetson：

`/home/nvidia/twork/NavDP/runtime/go2/experiments/q_to_r_cec_mixeda_20260824T1248Z/`

## 证据分级的 direct-bearing 接管（v2）

最初 v1 设计把通过证书的单目 PnP translation 当成 metric relative pose，并允许它
锁存局部模式与授权 STOP。后面的 S→Q 实测回放证伪了这项权限：几何证书能证明两幅图
共视、相对方向可恢复，却不能同时证明 first-40 单目尺度在该真机序列上达到厘米精度。
因此 v1 的 metric-local/STOP 合约已经作废，不能部署。

v2 不增加 matcher 或结果后调阈值，而是让每类证据只获得其确实支持的权限：

```text
Novel: native NavDP                 Revisit: long-range CEC
                 \                 /
          direct current->goal LightGlue + LingBot/PnP
                       |
             certificate 是否通过？
              /                    \
          否：原路回退            是：只读取 scale-free bearing
                                      |bearing| <= 60 deg
                                        -> 归一化到已验证的 2.5 m residual
                                        -> frozen NavDP image+point decoder
                                      |bearing| > 60 deg
                                        -> Jetson 原子有界转向

metric translation norm：仅写入诊断收据，无控制权限、无 STOP 权限
proof 丢失：回到 native/long-range CEC，不永久 hold
STOP：等待独立、scale-free 的 visual-convergence proof，当前 fail closed
```

关键实现：

- `deployment/gpu/revisit_local_pose_adapter.py`：纯函数 bearing 权限仲裁；
- `deployment/gpu/realworld_cec_hub.py`：长程 CEC 与 direct-local PnP 的优先级；
- `deployment/go2/terminal_motion_override.py`：Jetson 原子转向的唯一运动权限边界；
- `deployment/go2/navdp_ros_node.py`：原子 phase switch、turn 执行和持久 receipt；
- `NavDP/baselines/memnav/policy_agent.py`：在任何 metric scale 查询之前显式输出
  `predicted_scale_free_relative_xy`；
- `NavDP/baselines/memnav/memnav_server.py::/local_pose_query`：低延迟真机局部位姿端点。

GOAT 的 `/arrival_query` 继续保持 strict-first-64 合约。真机 `/local_pose_query` 只复用
full-mono NavDP 已在 causal frames 0..39 冻结并校验的 MDTEC scale，不读取传感器 metric
depth、不使用 pooled/oracle fallback，也不做第二次 LingBot prefix replay；但其 metric
字段从 v2 起明确只用于诊断。两种 scale 数值彼此接近不等于它们与真实世界绝对尺度
一致。

## 历史 v1 禁用态真链验证（已被 v2 权限收紧取代）

全新 reset 后完成两轮 disabled shadow smoke：

- `begin_revisit` 的推理槽现在在 idle check 时原子预留；实测第 10 次自然空隙成功提交，
  phase 切换后 `last_error=""`，不再出现 stale Novel request；
- v1 direct local proof 曾锁存，输出距离 0.607--0.619 m、bearing -176.6° 至
  -177.8°；这些距离现只保留为历史诊断，不能用于控制或到达；
- 仲裁稳定选择 `terminal_atomic_turn`，shadow command 为 `vx=0, wz=-0.35`；
- adapter 处于 `enabled=false, estop=true`，实际 `cmd_vx=cmd_wz=0`；
- 去除重复 64 帧尺度 replay 后，全新 reset 的首个 terminal receipt 从约 90 秒降到
  19.37 秒；其中随后 direct-local PnP 实测 0.039 秒，剩余首次延迟来自 CEC anchor 深度
  materialization，缓存后整轮约 0.5 秒；
- adapter 现在默认 `estop_on_start=true`，重启后无需依赖人工补锁。

这些测试只证明当时 v1 transport/control shadow 的实现一致性，不证明 metric scale。
当前代码已经升级为 schema `cec_direct_bearing_handoff_v2_20260824`；旧 schema 收据不能
通过新的 Jetson 权限边界。

## R→Q Novel 首段失败与 Go2 控制契约修复

首次 R→Q 自主首段按 native Novel arm 运行，CEC receipt 始终为
`novel_recording_native_only`，因此该轮不涉及错误 memory takeover。结果是严格失败：

- `timeout`，90 秒内未到达；
- 辅助距离 `1.167 m -> min 1.019 m -> final 1.022 m`；
- 路径长度仅 `0.615 m`；
- final yaw error `2.068 rad`；
- visual final 仅 1 个 good match、0 inlier；
- evaluator 自动 estop，随后再次显式 `set_enabled=false`；
- 原始收据与 5.9 MiB rosbag 位于
  `/home/nvidia/twork/NavDP/runtime/go2/experiments/r_to_q_native_fullmono_20260824T134318Z/`。

该轮暴露的是执行器合约偏差，不应归因于 CEC。进程审计显示 adapter 被临时覆盖为
`max_linear_mps=0.15`、`max_angular_rps=0.35`，而正式 TinyNav/Go2 验收值应为
`0.30/0.55`。桥本身确实保留了 `min_cmd_v=0.10`、`min_cmd_w=0.20`，问题不是漏掉
最小速度，而是漏掉了 TinyNav 在最小角速度 floor 之前的 `8 deg` 航向死区：NavDP
pure-pursuit 的微小正负角速度被逐样本放大成 `+/-0.20 rad/s`，形成左右 hunting。

启用窗口的 bag 独立审计：

- `/navdp/cmd_vel` 1393 samples，其中 1039 个非零转向 samples；
- 1011/1039 = 97.31% 低于 Go2 `0.20 rad/s` floor；
- 实际命令发生 43 次左右符号翻转；
- 127 次 active path updates 中，旧控制器 127 次全部转向、55 次翻转；
- 恢复 `8 deg` 死区后，98/127 个微小航向误差归零，仅 29 次真实转向、9 次翻转；
- 修复后 path-level median forward command 为 `0.299 m/s`。

代码修复保留硬件门槛，只恢复正确顺序：controller `8 deg` heading deadband，随后
Go2 bridge `0.10/0.20` floor。正式 profile 还会拒绝遗留的非 `0.30/0.55` 覆盖；若确需
短距 commissioning，必须显式使用 `NAVDP_CONTROL_PROFILE=acceptance`，不能混入正式
episode。离线审计工具为
`/home/asus/Research/Memnav_Realworld/deployment/go2/audit_turn_gate_bag.py`。

截至修复时机器人保持 `enabled=false, estop=true`。由于失败轮已使机器人离开冻结 R
起点，修复后的正式 R→Q 不能直接续跑；必须由现场人员重新放回 R，随后新建 causal
history、evaluator 和 rosbag，才能形成可比较复测。

## S→Q Full-Mono 复测与到达问题最终归因

在恢复 `0.30/0.55` controller、`8 deg` heading deadband 和 Go2 `0.10/0.20`
velocity floor 后，S→Q 首个有效命令为 `vx=0.297 m/s, wz=0`，此前的左右 hunting 没有
复现。该轮同时启用了 frame/SHA/token 绑定的单目深度事务，MemNav append 与 NavDP
消费同一 JPEG 的收据完全一致，没有再出现跨进程 SHA race。

正式 evaluator 仍将该轮记为失败：

- `arrival_mode=visual`，termination 为 `operator_stop`；
- initial/min/final distance：`1.226 / 0.993 / 3.729 m`；
- path length：`18.54 m`；
- exact-view 与 pose success 均为 false；
- 原始结果：
  `/home/nvidia/twork/NavDP/runtime/go2/experiments/s_to_q_fullmono_transaction_gatefix_20260824T1420Z/evaluation.json`。

这说明执行器和 RGB-depth 事务已经修复，但 native NavDP 仍从 Q 附近驶过，旧 SIFT
arrival verifier 没有在经过时确认到达。RTX causal buffer 保存了 431 帧；对同一 Q 目标
做只读离线审计得到：

- SIFT 只有 3 帧达到 `good>=30, homography inliers>=20`，没有满足完整三连门；
- LightGlue 在 300--305 与 324--329 形成两个强共视窗口；frame 326 有 339 matches、
  298 fundamental inliers；
- 完整 `LightGlue -> current LingBot depth -> PnP -> atomic certificate` 在 frames
  325--328 连续通过，单次查询约 `31--74 ms`；
- v1 metric distance 依次为 `0.769, 0.520, 0.358, 0.125 m`；但独立 evaluator 证明
  整轮真实最近距离仍为 `0.993 m`。因此 frame 328 的 metric translation 至少低估
  `7.9x`，不能授权近距离控制或 STOP；
- 同期 terminal yaw residual 为 `3.1, 21.1, 32.7, 43.4 deg`，也清楚表明视角没有
  收敛到目标图。

用 v2 同轨迹重放，frames 325--328 全部只产生 `bearing_local`，分别输出固定 2.5 m
point residual；frame 329 证书拒绝后精确回到 native。五帧全部
`metric_control_authority=false`、`stop_authorized=false`。这证明“看到高共视后自动修正
方向”已经具备离线可执行链路，但尚未证明“自动到达并停止”。后者必须由独立视觉收敛
模块解决，不能继续复用单目 PnP translation norm。

当前验证：发布侧 bearing/Hub/Jetson 权限测试 `43 passed`；研究侧 scale/terminal
contract `14 passed`；Python compile 通过。以上修改只完成离线回放和单元测试，没有重新
启动 RTX 服务、Jetson adapter 或底盘。真机仍应保持停止状态，v2 未经新的 disabled
shadow 验证前不得用于运动。
