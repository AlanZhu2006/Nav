# Pi3X 连续当前状态读出：本机闭环开发对照

日期：2026-09-06。状态：**4条查询 × 3臂全部完成，12条记录独立复算通过**。
结论：连续状态减少了部分Pi3X绕行，但三臂均4/4成功；保留为开发机制，不替换正式CEC。

## 1. 问题与唯一方法变化

这不是重跑 Pi3X 的正式替换评测，也不是新增训练。先前异常查询中，错误方向往往发生在
后续重新定位时，而不是首次识别历史目标时。本轮检验：**是否应该让连续视觉状态负责
当前相机，让稀疏重建只负责历史到目标的关系？**

三臂均保持冻结 controller、候选、输入图、proof 权重及阈值：

| 臂 | 当前状态 | 目标关系 / 授权 |
|---|---|---|
| 原 Pi3X (`joint`) | 每次由当前图、历史桥接帧和目标图共同重建 | 原 Pi3X 关系、原 learned proof |
| 连续状态 (`shared_state`) | 从连续 LingBot 状态对齐到共同历史坐标 | 同一个 Pi3X 目标关系；在变换后的几何上重新计算原 proof |
| 原 CEC (`certified`) | 连续 LingBot 状态 | 原显式几何定位与 certificate |

不增加候选、不改阈值、不增加转向策略、不用 GT 纠正模型。所有研究变动在隔离诊断
wrapper 中，生产 CEC 和论文默认方法没有替换。

与更早的两例/四例离线读出不同，本轮重新运行检索、原proof和真实动作，后续RGB由
各臂实际轨迹产生。保留动态目标关系是有原因的：
[先前四例读出](PI3X_SHARED_STATE_FOUR_QUERY_RESULT_20260906.md)中，弱支持Novel的
动态目标关系优于固定首次关系。因此本轮不把“持续当前状态有效”外推为“所有目标只估计
一次更好”，也不是重复宣称离线角度改善已兑现到SR。

### 共同坐标对齐

以已观察到的历史 anchor `h` 和 `h±8` 基线，将 LingBot 当前 pose 映射到 Pi3X 表示：

```text
a = ||pP(h+8) − pP(h−8)|| / ||pL(h+8) − pL(h−8)||
Q = RP(h) RL(h)^T
RP_new(t) = Q RL(t)
pP_new(t) = pP(h) + a Q [pL(t) − pL(h)]
```

当前 world points 也使用同一个新 pose 与原 local points 重新计算；然后才进入原
spatial proof。不是先保留旧 accept 再替换方向。目标图和历史图的 Pi3X 预测保持原样。

这项操作对两个重建各自的公共 Sim(3) 变换有相应不变性，但不消除实际 anchor 误差或
非刚性漂移。旧 proof 面对新几何分布，其校准表现不能自动继承原测试结论。

## 2. 样本与对照边界

四条已经消费、按已知异常选择的开发查询，三个场景：

1. `8WUmhLawc2A / episode_0000`：Revisit。
2. `V2XKFyX4ASd / episode_0004`：Revisit。
3. `PuKPg4mmafe / episode_0001`：Revisit。
4. `PuKPg4mmafe / episode_0004`：弱支持 Novel，不是完全无共视的 Novel。

原20260817冻结记录中这四条 Pi3X 是3成功、1失败；本轮必须使用本机重新配对结果，
不能把旧失败与新成功跨运行拼成增益。

- 每个配对块重放同一 actual-online A RGB、同一 NavDP FIFO；目标与场景资产哈希一致。
- 三臂在同一 server/模型进程内分别 reset，再独立执行；每次 planning 的 seed 按原步号固定。
- 600 步、`exec_horizon=8`、固定2.5 m residual；到达采用原仿真1 m位置 first-crossing。
- Novel/Revisit role 不进入 runtime；GT 用于仿真初始化、成像与离线评分，不进入 relocalizer。
- **controller depth 使用原20260817的 simulator metric depth。** 这是只改读出的归因实验，
  不是 full-mono 正式评测，也不是现实部署的自动到达验证。
- 使用本机 Torch2.8/CUDA12.8 环境，不声称与先前HPC不同环境逐位复现。

因此有效实验单位是4条查询、3个场景；不能把 hundreds of replans 作为独立样本。

## 3. 固定观测预检：已完成

159个相同原观测时刻，原 proof 在每个变体的实际输入上重算：

| 读出 | accept plans | endpoint bearing 中位误差 | P95 | >90° |
|---|---:|---:|---:|---:|
| 原 Pi3X | 143 | 12.54° | 97.47° | 18 |
| 连续当前状态 | 141 | 2.28° | 23.28° | 0 |

两臂的 accept 集合不同。这支持进一步闭环检验，但不直接证明 SR 提升；角度参考是
当前位置到真实目标的直线方位，不是 geodesic 第一段。

原始输出：`.diagnostics/pi3x_shared_state_closedloop_20260906/proof_preflight_summary.json`。

## 4. 完整闭环结果

每格为“步数 / 实际平面路程”，全部进入相同1 m成功区：

| 查询 | 原Pi3X | 连续当前状态 | 原CEC |
|---|---:|---:|---:|
| 8W / 0000，Revisit | 175 / 5.31 m | 170 / 5.27 m | 168 / 5.27 m |
| V2 / 0004，Revisit | 112 / 3.70 m | 115 / 3.78 m | 115 / 3.89 m |
| PuK / 0001，Revisit | 310 / 10.27 m | 163 / 5.20 m | 164 / 5.24 m |
| PuK / 0004，弱支持Novel | 491 / 16.50 m | 243 / 9.00 m | 306 / 9.26 m |
| 成功数 | 4/4 | 4/4 | 4/4 |

连续状态对原Pi3X、对CEC均为 `+0/−0，exact McNemar p=1.0`。
12条运行的runtime failure均为0。不能与旧环境的Pi3X 3/4跨运行计算增益；
旧失败发生在8W查询，本机原Pi3X自身也已成功，没有将其计为新读出救回。

两种Pi3X读出的首次anchor逐例相同：`140 / 39 / 85 / 8`，后续获授权时也不切换。
CEC的候选选择来自其原几何规则，并不要求与Pi3X相同；例如前两条分别选146和39。

### 4.1 路程改善是真实的，但来自哪几条必须说清

四例平均实际平面路程为8.94 / 5.81 / 5.92 m。**两个明显的缩短均来自同一个PuK场景**；
另外两个场景差异很小，V2中连续状态还多3步、约0.08 m。因此不把均值差写成跨场景
稳健提升，也不据此做新方法的统计确认。

已完成的第三条轨迹可作具体解释：原Pi3X实际平面路程10.27 m，连续状态5.20 m，
CEC5.24 m。原Pi3X在接近目标区域前多绕了一圈，连续状态去除了这段额外绕行；
三者共同的起始转弯仍在，并非把全部控制路径都变短。三臂最终都进入相同1 m成功区，
所以这是该例的路径改善，不是SR增益，也不是相对CEC的新收益。

[该例的实际轨迹与方向图](../.diagnostics/pi3x_shared_state_closedloop_20260906/case_02_executed_paths.png)。
图中没有障碍地图，不能单靠这张图断言任何一段是全局最短路线。

第四条原Pi3X在目标附近又形成额外回环；连续状态与CEC没有出现这段额外回环，路程分别
9.00与9.26 m。相对CEC的差异远小于相对原Pi3X的差异：
[第四条实际轨迹](../.diagnostics/pi3x_shared_state_closedloop_20260906/case_03_executed_paths.png)。
这两张图是在完整轨迹上作解释，不是新增样本，也没有按它们修改方法。

### 4.2 实际执行过程中获授权方向的描述统计

| 臂 | accepted plans | endpoint中位误差 | P95 | 最大误差 | >90° |
|---|---:|---:|---:|---:|---:|
| 原Pi3X | 118 | 10.25° | 121.12° | 168.45° | 10 |
| 连续当前状态 | 81 | 1.55° | 8.38° | 13.81° | 0 |
| 原CEC | 96 | 3.11° | 11.70° | 25.43° | 0 |

三臂路径和accept时刻不同，这不是相同观测的逐帧对照；也不是N=118/81/96的独立
统计。第3节才是固定观测预检。81次接管比118次少，也包含路程变短导致规划次数减少，
不能直接解释成更严格的授权。

## 5. 资源修复与复算约束

第四条先后三次尝试被同卡其他工作区的显存峰值打断。保留全部原日志，不把 OOM 算作
方法失败，也不从不完整尝试中抽取成功臂拼成配对。

最终资源方案是权重交替驻留 CPU/GPU，并把 LingBot aggregator 的完整 K/V 按层暂存
主机内存：不减少历史、不改变窗口、dtype、attention 顺序或候选。该方案仅为共用GPU时
完成同一计算，传输开销明显，不能用于比较部署时延。

前三条使用原完整 `closedloop/`；第四条只使用
`closedloop_resource_repair3/` 的完整三臂。资源安排在每个三臂块内相同。

检查包括：

- pose与world points一致变换；两侧公共坐标变换；退化基线；原joint路径不变；
- 权重搬运与小型两层SDPA完整缓存搬运的输出一致性；
- 从实际 rollout 位置独立重算首达、末距、实际路程，而非使用命令速度积分；
- 从真实目标与当前 yaw 重算 endpoint angle，与独立 atan2 算法核对；
- 每个 learned plan 与实际服务端 response 对齐，核对 mode、proof分数、accept和执行方向；
- role隐藏、源A重放、goal/scene身份、K和seed配对、2.5 m方向接口。

最后一条原Pi3X在资源修复前后均491步成功；全部491步 `x/y/z/yaw` 最大差为0，
62次规划的seed、接管、bearing和选中trajectory SHA一致。连续状态臂的243步位姿
和31次规划也一致。这是资源改变未改变这两臂结果的实测证据；不将它扩写成所有
网络算子在所有未来输入上的无条件逐位保证。

## 6. 去留决定

**保留连续状态读出作为学习分支的机制原型；不替换生产CEC，不宣布新SR，不提交长训。**

本轮兑现的是两条开发查询的路径改善，而不是SR增益。它支持把“持续当前定位”和
“稀疏目标关系”分开负责，不能推出learned relocalizer已经与CEC在更大开放集上等效。
主方法本来就采用连续当前状态，因此这项结果也不是给主方法增加第三个模型的理由。

后续学习若继续，最小可检验方向是**学习历史 anchor 到目标图的关系读出，并复用连续
当前状态**。它与重新训练 DINO selector 或把整段历史直接交给 decoder 是不同问题。
是否训练、是否扩样，要由本轮闭环及额外未用于设计的测试决定；本报告不自动启动长训。

相应的原型保留代码位于本目录下述诊断根；没有用它改写原Pi3X正式替换“未通过”的决定。

## 7. 复现入口

- 总账：`MemNavData/OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md`。
- 诊断根：`.diagnostics/pi3x_shared_state_closedloop_20260906/`。
- 协议：`PROTOCOL.md`、`RESOURCE_REPAIR.md`。
- 变体：`shared_state_readout.py`、`server_wrapper.py`。
- 单例执行：`run_one_query.py`；最后配对块：`run_resource_repair3.py`。
- 独立复算：`independent_closedloop_recount.py`。
- 最终机器可读结果：`closedloop_independent_verification.json`，`verified=true`，12 rows。
- 测试记录：`.diagnostics/overnight_project_audit_20260906/readout_and_residency_tests.xml`。

本轮不操作真机、不替换正式方法、不改论文标题/Abstract/Introduction、不commit或push。

最终结果JSON SHA-256：
`1699b32d02a1b02085fcb13a1ea93c067c5fb090b24b62a07b463e0b3ef0a937`。
每条raw result及8组learned响应序列的来源哈希/匹配关系保存在该JSON。
完成时间14:22（Asia/Shanghai）；本任务18891/8891两个服务器已结束，其他工作区进程保留。
