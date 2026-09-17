# 真机 CEC 与仿真闭环差异审计（2026-09-07）

## 结论先行

这次不能统一解释为“CEC 找错记忆”“LingBot 漂移”或“NavDP 本来就不行”。
原始记录支持三个更具体的结论：

1. **pair_004：低评分替代轨迹与 CEC 朝向纠正发生执行语义冲突**，再叠加真实 RGB-D 间断与近障停车。
2. **pair_006：固定 bearing residual、整段局部轨迹执行、末端视图确认没有组成收敛闭环**。同组 native 也出现局部轨迹尾部停滞，存在两臂共用的执行器问题；但不能因此免除 CEC 末端设计的责任。
3. **仿真与真机不是同一执行/成功合约**：前者短前缀重规划、持续更新几何、由 GT 距离结束；后者在本批次执行完整局部路径，运动期间不刷新策略/几何，并要求视觉到达信号或人工终止。

目前最应优先修的是这三个接口之间的关系，不是重新训练 relocalizer，也不是把所有 fixed bearing 一律改成未经独立验证的 metric distance。

## 1. 范围、版本与原始数据

本轮通过共享 SSH `unitree-dog` 直接读取 `/home/unitree/MemNav-RealWorld`，同时读取本机 GPU 仓库 `/home/asus/Research/MemNav-RealWorld` 和仿真研究仓库。没有启动/恢复机器人，没有修改真机代码、参数、原始记录或人为结果，没有运行真机单元测试，没有提交或推送。

真机和 GPU 仓库 HEAD 均为 `95afe64ce9047150107a8eba1a34829fb8a70cd1`，但工作区均有未提交修改，且执行器文件并非两机完全相同。本轮没有覆盖这些修改。

先读取最新 `runtime/go2/experiment_pairs/index.json`。其登记的有效完成组为 `001/002/003/004/006/009`。`CURRENT_STATUS.md` 的主快照仍是 08-30，不能用它代替 09-06 运行记录判断最新结果。

关键样本：

| 项目 | pair_004 | pair_006 |
|---|---|---|
| Episode | `episode_20260906T072141_928425Z` | `episode_20260906T100849_093505Z` |
| CEC run | `m_episode_20260906T072141_928425Z_cec_20260906T072738Z` | `m_episode_20260906T100849_093505Z_cec_20260906T101601Z` |
| 历史 | 171 帧 sealed survey | 153 帧 sealed survey |
| 活动窗口 | UTC 07:29:29.165–07:31:04.190 | UTC 10:17:41.461–10:21:01.885 |
| CEC 结果 | 人工 failure；operator stop | 人工 failure；operator stop |
| Baseline 结果 | failure；depth unavailable stop | failure；持续低进展后人工 stop |

006 明确采用**重新采集后当前保存的这一轮**，不套用旧轮撞墙解释。两组使用的冻结相机高度均为 0.42 m，NavDP 为 monocular sidecar；D435i depth 不进入策略，只进入本地安全链路。

本地审计副本目录：

`/home/asus/Research/Nav-graph-blind/.diagnostics/realworld_sim_audit_20260907_VeP2Ch/`

两组各三份 `cec_receipt/status/rgb_arrival_status.jsonl` 的 SHA 均与各自 finalized capture manifest 完全一致（6/6）。`summarize_receipts.py` 仅离线统计已复制 JSONL；复算结果为 `recomputed_summary.json`。没有下载或重新处理数 GB 的原始 MCAP，004 的 MCAP 帧间隔统计引用其原有离线分析，而命令时长、低评分次数、anchor、006 关键事件均独立复算。

## 2. 最新总体情况：既有收益，也远未稳定

| 有效登记组 | CEC 人工结果 | Mono-native 人工结果 | 到达/终止备注 |
|---|---|---|---|
| 001 | success | failure | 保留既有人工标签，本轮未重新确认其自动停止 |
| 002 | success | failure | CEC 为人工 stop；不代表自动到达 |
| 003 | success | failure | CEC 自动到达未触发，实际先发生 trajectory stale |
| 004 | failure | failure | 侧向回退/朝向冲突及深度问题 |
| 006 | failure | failure | 末端动作、重规划和视图确认未衔接 |
| 009 | success | failure | CEC 有原始自动到达 latch，并获人工确认 |

登记表描述性计数为 CEC 4/6、native 0/6；**不能写成冻结总体上的正式 SR，更不能写成自动到达 4/6**。这些是人工指定有效组，含 retry、版本演进、人工起点确认和不同终止时机，不满足仿真正式配对总体的统计条件。

009 是不能遗漏的新正结果：UTC `14:23:09.631704`（北京时间 22:23:09.632）视觉模块发出 latch，55 good matches、46 inliers、coverage 0.2242/0.0979、center offset 0.1579、scale 0.7473、RMSE 1.0387 px。随后状态为 disabled、estop、zero。它证明到达信号可以真实触发停车，不能证明各类接近方向都已解决。

## 3. pair_004：有正确执行的初始转身，但随后动作语义冲突

### 3.1 原始收据能够确认的事件链

1. 首次 CEC 方向约 −160.4°，显式 rear-goal heading turn 约 8–9 秒完成。
2. UTC `07:29:38.837554`，memory bearing 已是 +7.43°，但 critic 最大值低于本轮冻结阈值 −2.0，上游返回重复 `[0,-1]`。
3. `policy_agent.py:332` 不是从规划器得到这条正常路径：它在低评分时将 x 全置零、y 置为原候选 y 均值的符号。
4. 测量式路径执行器补上原点、去重并变换到世界系，把它解释为右侧 1 m 的终点。其追踪律先转向该端点。
5. 下一次 `07:29:45.460976`，目标相对方向变成 +77.29°，CEC 又要求重新朝向目标。

即：**朝向目标 → 低评分替代路径使其转向侧面 → CEC 再纠正回来**。不能把它叫作 anchor oscillation。

活动窗口 191 条状态均为 anchor 15、takeover true。7 次低评分回退发生在 step `30,32,35,36,37,38,39`；部分受 CEC 原地转向覆盖，不能声称每条侧向路径都被完整执行。

### 3.2 时间与停止原因

按约 2 Hz 状态做左端时间积分，使用互斥类别：有前进命令 / 无前进但有角速度 / 零命令。

| 类别 | 时间 |
|---|---:|
| 仅转向命令 | 34.74 s |
| 有前进命令 | 9.99 s |
| 零命令 | 50.30 s |
| 合计 | 95.03 s |

RGB-D 恢复 9 次，共 23.17 s；这是上述窗口的交叉标记，**不另加到合计**。这些都是指令/状态时间，不是 GT 实际运动时间。用户转述的 35/58/19 秒不能作为这段互斥分解。

原有 MCAP 分析记录：同窗 RGB 最大间隔 0.534 s，aligned depth 最大间隔 2.702 s，6 个深度间隔超过 2 s。存在真实接收间断，但尚不能区分传感器发布、DDS、同步或处理负载各自贡献。

本轮实际查看保存图像：初段前方为绿篱，后段画面接近叶片；最后净空约 0.20–0.22 m，确实是近障背景，不是到达。活动段最小状态净空 0.204 m。最终人为结果与终止原因仍是 failure / operator stop，不改写为另一种自动终止。

### 3.3 不能从本组推出什么

- anchor 不变不等于定位绝对正确；但记录没有支持“频繁换错记忆”作为主因。
- 证书不是可通行路线证书。目标方向可能指向绿篱后方，需局部避障绕行；仅凭此运行无法确认可行路线是否曾在候选中。
- critic 阈值 −2.0 和较早仿真入口默认值不是同一概念下可直接横比的数值；本轮不据此认定阈值是根因，也不调阈值。

## 4. pair_006：不是低评分回退，而是闭环更新与末端目标不一致

### 4.1 定位证据没有转为近目标动作

UTC `10:19:35.779876`（北京时间 18:19:35.780），step 37：

| 收据字段 | 值 |
|---|---|
| 当前到目标直接 PnP 证书 | accepted |
| PnP inliers / RMSE | 155 / 0.9873 px |
| query/ref hull coverage | 0.0776 / 0.0641 |
| 预测距离 | 0.3183048 m |
| 预测位置 bearing | −3.0106° |
| 目标相机相对朝向 | `terminal_yaw_right_deg=54.6592°` |
| 距离控制权限 | false |
| 停车权限 | false |
| 真正送入 NavDP 的 PointGoal | `[2.49655, -0.13130, 0]`，模长 2.5 m |
| NavDP 选中路径端点 | `[2.52639, -0.22698]` |

该条证据在本地 `pair_006/cec_receipt.jsonl:39`。定位输入原图已在 GPU buffer 找到，并以 receipt 的图像 SHA 完全对上：

`/home/asus/Research/MemNav-RealWorld/runtime/gpu/buffer/run_20260906T064125Z_505918/ep_0029/189.jpg`

SHA：`ec8afb818337c327395bbebba0fbd5bb37bddfcf4af15daec43c49bb980a9c04`。

这不是“机器人明知 GT 已到却不停车”：0.318 m 仍是预测，证书只验证重投影/支持，不能保证距离正确。本轮没有独立 GT，不能断言已穿过真实目标点。但可以确认：**系统获得了一份近距离定位假设，却仍只生成远固定半径的控制请求，且没有末端收敛处理。**

### 4.2 关键新归因：不是推理慢，而是执行期间不重规划

step 37 到 step 38 间隔约 23.31 s。到 UTC `10:19:55.398935`，相同计划已老化 19.622 s，Go2 里程计投影到这条局部路径的进度约 2.461 m，仍在追踪其剩余约 0.094 m。此值是本地 odometry/path projection，不是目标 GT 距离。

随后 step 38 的历史目标方向变成 −157.44°，再次大转身。step 40 的旧计划最长又保留了 50.363 s；状态中 `last_inference_s` 仍约 0.78 s。

因此：

- Go2 位姿反馈层仍在闭环跟踪，不能称“所有控制完全开环”；
- **目标视觉与 CEC 更新被完整局部路径阻塞，形成长时间的目标反馈空窗**；
- 单次推理不到一秒不代表目标闭环一秒更新；配置 planning rate 也不等于实际规划频率。

代码直接证据：`navdp_ros_node.py` 的 `_request_inference` / `_snapshot_inference_input` 在 trajectory active 或 heading turn active 时返回；`TrajectoryExecution.start()` 安装整条路径；只有局部终点完成/被中断才进入下一轮。活动路径还绕过普通 trajectory age 检查。

006 活动窗 200.42 s 内仅收到 14 个新规划收据；402 条状态全部为 anchor 23，低评分回退为 0。历史接管存在，但不能将每条 status 当成一次新的定位。

### 4.3 位置方向和目标视角方向不是同一个量

本次假设同时说“目标位置近乎在前方”和“目标相机朝向在右侧约 54.7°”。这两者可以同时成立。

`revisit_local_pose_adapter.py` 的 atomic turn 依据是**平移 bearing 是否超出 PointGoal 支持范围**，不是目标相机的 rotation；`yaw_right` 被保留在 receipt，却不用于末端视图对齐。因此 ±160° 重定向不是完成 ImageGoal 视角对齐的证据。

停车用的是另一条 SIFT/homography 视觉模块，而不是 CEC 的 SuperPoint/LightGlue/PnP certificate。006 在 enabled 窗内 1846 条到达状态发布中，matched / confirmed 都为 0（不是 1846 个独立样本）。主要失败原因是 good matches 不足、homography 失败、inliers 不足。部分 center mismatch 的同一观测还具有很大的 scale/rotation 残差，不能只降低 center 阈值就称修复。

在 step 37 周围，到达模块约只有 15–22 good matches，明显不满足它自己的匹配条件；直接 PnP 的 155 inliers 来自另一 matcher，不能将两个算法的计数当成相同信号。

### 4.4 局部路径尾部停滞是另一个共享缺口

旧执行完成容差为 0.08 m；局部速度又随路径尾段缩小，记录出现剩余约 0.1–0.18 m、持续约 0.1–0.17 m/s 指令，却几乎不增加路径进度。006 native 同样曾在剩余 0.1629 m、vx 约 0.143 m/s 持续约 45 秒。

这支持执行尾段与底盘可实现运动之间不匹配，而不是单凭“发出了 vx”证明机器人在前进。究竟是 gait dead zone、地面/腿部负载还是 odometry 误差的精确贡献，尚缺受控低速实测；不宜简单断言把最低速度提高即可。

## 5. 仿真与真机的关键差异

| 层级 | 论文主要仿真协议 | 本批真机 |
|---|---|---|
| NavDP 目标输入 | 原 ImageGoal + 可选 2.5 m bearing residual | 相同核心输入；额外直接当前视图 PnP 分支 |
| 局部执行 | `exec_horizon=8` 后重规划，每步命令上限 0.0376 m | 先追踪完整约 2–3 m 路径，再重新观测/规划 |
| NavDP 目标反馈 | 最多约 0.301 m 的命令前缀后更新；碰撞/转向另计 | 可等待二十余秒，尾段停滞时甚至约 50 s |
| LingBot 更新 | 非规划步仍 `srv_memory(frame)`；运动中持续更新 | 当前实现推理入口被运动阻塞，没有独立运动中几何写入通道 |
| 转向 | 仿真追踪/专门合约，主表 terminal U-turn off | CEC rear bearing 的额外 IMU 原子转向 |
| 运动模型 | 理想运动学 + NavMesh snap/碰撞处理 | Go2 步态、实际位姿反馈、加速度、硬件速度响应 |
| 成功/停止 | GT 位置进入 1 m 半径立即结束，无需视觉 STOP | 独立视觉匹配 latch 或人工停止；还受安全/新鲜度终止影响 |
| 深度安全 | 无真实 RGB-D 掉帧链路 | D435i ROI 安全依赖；即使策略单目也可能因 depth gap 暂停 |
| 历史来源 | Final14 的实际 metric-A；其他 full-mono 表使用 actual mono-A | 手柄 survey 封存、重新定位起点后第二轮查询 |

仿真代码：`eval_2leg_habitat.py` 的 `pursuit_step`、`step % exec_horizon`、非规划步 `srv_memory` 和 `benchmark_goal_distance`；Final14 runner 冻结 success 1.0 m、horizon 8，`eval_shared_online_role_pairs.py:172` 明确禁止该 position-SR 协议启用 terminal U-turn。

**2.5 m 是策略条件，不应与一次必须走完的路程混为一谈。** 同一 PointGoal 放在频繁重规划和完整路径追踪两种系统中，闭环含义不同。这是此前“沿用同一个 NavDP，所以效果应当相同”的主要遗漏。

## 6. LingBot 漂移与 metric：证据目前支持到哪里

两组均有 first-40 高度尺度 receipt，camera height=0.42 m；004 scale=3.54057，006 scale=2.97086，均标记 valid、未 clamp。这反驳“没有读取高度/一直 zero depth”作为简单解释，但不证明深度或全程位姿完全正确。

真机运动过程中不送连续几何观测，会增加跨视角重定位压力；这种更新方式可能加重 drift，但两条失败记录没有独立 GT 可分离 drift 与执行超程。

另外，bearing 本身在接近目标时容易受位置误差影响。对 `b=v/||v||`，一阶扰动满足 `δb ≈ (I−bbᵀ)δv/||v||`。这是误差传播关系，不是本次测得的 drift 大小。即便总体尺度可用，v 很小时方向仍可能大幅变化；固定投影又保留了非零 residual，不能自行产生到达后的静止平衡。

所以“有高度先验”不能自动解决末端到达；“只保留 bearing”也不能取代终端收敛机制。当前证据不足以断言改用 metric distance 必然改善或必然更差。

## 7. 最新代码已经改了什么，哪些不能重复宣称待修

真机工作区在 004/006 之后已有以下修改，本轮只审阅，没有再实现：

| 已存在改动 | 当前能确认的效果/边界 |
|---|---|
| 局部完成容差 0.08 → 0.15 m；4 s / 2 cm stagnation replan | 009 收据中存在这些参数，且实际出现 `stalled_replan`，随后继续；但不同场景不能单独归因 SR 改善 |
| 低 critic sentinel 改为 hold-and-replan | 代码已存在，不再把替代侧向点直接追踪；004 失败发生在此修改前。未看到其在同类 004 场景的修复后有效性证据 |
| RGB-D / pose / control callback groups 分离、多线程 executor | 代码存在；不能据此认定所有 depth gaps 已消失 |
| 安全 ROI 对超出 5 m 的有效深度做截断，而非全部当无效 | 修复一种 open-view depth-unavailable 来源；不等于相机/传输间断也修复 |
| 已知近障时禁止转动和前进 | 更保守的安全处置，不会自动生成绕行路线 |

低 critic hold 在 `terminal_motion_override` **之前**执行。这避免危险替代轨迹，但若某次 rear bearing 请求也引出低 critic，有可能连有证据的转向都被先挡住。它是需要离线检查的优先级问题，不是已经观察到的本次失败原因；单纯反复停住重采样也不能被称为已恢复探索能力。

009 的成功发生在局部尾段修复后，其日志无低 critic，也没有 low-critic guard 收据，因此不能拿 009 证明 004 的低评分路径问题已解决。

## 8. 下一步：先闭合最小控制环，不再叠加大模块

### P0-A：恢复与仿真相符的 receding-horizon 含义

保留 NavDP 输出完整轨迹，但执行层消费短的、可测进度前缀，而非等待虚拟远端点完整到达。运动期间仍应推进同一 causal RGB 几何流；policy FIFO 按预定规划频率更新，不要与几何观察频率混在一起。

首先以 006 已存轨迹/状态进行离线调度推演：标出原本应重新感知的时刻，验证不会再次让 0.8 s 推理对应 20–50 s 无目标更新。离线推演只能验证事件顺序和响应时机，不能声称已产生修复后的真实 SR。

不要只把 `planning_rate_hz` 调大：active-path 的直接 return 不移除，这个数字不会改变实际行为。也不要绕过新鲜度/碰撞保护追求连续运动。

### P0-B：明确路径、搜索意图和终端目标的不同语义

正常轨迹与低 critic 搜索意图应使用明确不同的返回语义，不能再靠一个重复坐标同时充当“正常路径”和“没把握”。保留现有安全停止，但要在合法安全范围内确认搜索动作、已认证朝向与重观测的顺序；拒绝危险轨迹不等于获得绕行策略。

先用 004 收据离线走一遍调度状态，不在机器人上重新盲试 ±90°。只有执行语义恢复后，仍然长期朝向绿篱、不能绕行，才值得进一步做路线/controller 归因。

### P0-C：完成末端接近—视角对齐—停止，而非降低一个阈值

- 长程 bearing 继续承担引导职责，不能要求它自身完成 STOP。
- 当直接局部几何提供稳定目标假设时，控制的任务应变成局部误差收敛，而不是无条件复用远 2.5 m 目标直到人工喊停。
- 近目标平移、目标相机朝向和到达确认需要同一目标定义；位置方向和目标相机 yaw 不能互相代替。
- 不以单次 0.318 m PnP 估计直接授权停止，先核对尺度误差、跨帧稳定性及近/远负样本。
- 以 003 的人工已到/自动未到、006 的直接 PnP 正证据/视觉停车负证据、009 的自动到达正证据建立小型终端校准集合，避免只改到 006 能过。
- 与 native 配对时使用一致的成功定义和终端规则，不能把停车器差异归为 CEC 记忆增益。

### 后续才是定位与路径增强

在目标反馈周期、终端任务和传感器停顿可解释之后，才用已封存轨迹加独立距离/朝向标注，测分段定位误差。若确认长基线 drift，再考虑连续几何更新或局部重定位；若方向对但无路，再评估历史路径/route cue。现在直接增加 learned router、全局图或大规模训练都不能解决已发现的动作/到达接口矛盾。

## 9. 对仿真和论文结论的影响

仿真 Revisit SR 仍是其冻结协议下的有效结果，不因真机失败自动作废。但它证明的是**到达位置邻域的导航效用**，不是任意起点下完全自主的 ImageGoal 末端对齐与 STOP。

真机当前正确表述：有多组人工确认的 CEC Revisit 收益，009 有自动停止证据；低评分路径、目标更新周期、末端收敛和传感器可靠性仍影响稳定性。不能把人工 success 全部记为自动成功，也不能用两组失败推翻所有已有定位增益。

最小而有效的重构方向是：**保留已验证的记忆定位，恢复短前缀目标反馈，再闭合终端任务。** 不是再增加一个高层 expert 来遮住执行与到达的断层。

## 10. 快速证据入口

- 004 原分析：[analysis.md](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/pair_004/analysis.md)
- 006 当前 pair registry：[pair.json](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/pair_006/pair.json)
- 两组独立统计：[recomputed_summary.json](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/recomputed_summary.json)
- 006 原始视频本地副本：[pair006_cec_revisit_rgb.mp4](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/pair_006/pair006_cec_revisit_rgb.mp4)
- 真机当前执行器快照：[trajectory_execution.py](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/robot_code/trajectory_execution.py)
- 真机当前调度快照：[navdp_ros_node.py](../.diagnostics/realworld_sim_audit_20260907_VeP2Ch/robot_code/navdp_ros_node.py)

视频帧仅作定性查看，不用编码视频时间轴替代 JSONL/相机 header 的事件时间。以上诊断副本默认不进入 Git；原始证据仍保留在真机和 GPU 仓库中。
