# CEC Proof-Carrying Controller Portability

日期：2026-08-21（Asia/Shanghai）  
状态：统一 CEC 契约、四种 accepted-branch adapter、逐动作授权与 ViNT proof-bound anchor 接口均已完成；四臂、两场景 mixed-role 本机 gate 已独立复算通过，尚无新统计闭环 SR。

## 1. 冻结问题

本支线不再比较“裸 ViNT / 裸 iPlanner / 裸 ViPlanner 与 CEC+NavDP 谁更强”。正式问题统一为：

> 同一个 CEC 开放集证明，能否通过很薄的 controller-native adapter，把同一段因果在线历史的 Revisit 价值交给不同的冻结 controller，同时在证据不足时保持完全相同的 mono NavDP fallback？

因此主比较中的每个 arm 都必须经过 CEC。Controller 只在 CEC **接受后**不同；CEC 拒绝后所有 arm 使用同一个冻结 mono NavDP，不允许各自选择有利 fallback。

这比“所有模型都吃同一个 PointGoal”更准确。不同 controller 的原生目标接口本来就不同，公平条件应是：

- 相同 causal RGB history；
- 相同 target RGB；
- 相同 CEC shortlist、LightGlue/PnP certificate 与 accept/reject；
- 相同 scale-free direction proof；
- 相同 reject fallback；
- 只改变 accepted proof 如何投影到 controller 的原生接口。

## 2. 统一架构

```text
actual-online causal RGB history + current RGB + target RGB
                         |
                         v
        one frozen CEC proof (role label invisible)
        DINO top-8 -> LightGlue -> LingBot depth/PnP -> certificate
                         |
              +----------+-----------+
              |                      |
          reject / error          accept
              |                      |
     shared mono NavDP          proof-bound adapter
     original ImageGoal          | NavDP mixed goal
     exact fallback              | ViNT verified anchor ImageGoal
                                 | iPlanner 2.5 m PointGoal
                                 | ViPlanner 2.5 m PointGoal
                                 ` EGO local metric-map goal (diagnostic)
```

### 逐动作 proof authorization

CEC 的正式语义不是 query-level gate，而是 **proof before every action**：

- 每个 decision frame 都用当时可见的 causal history、current RGB 和 target RGB 重新产生 certificate；
- certificate accept 只授权当前一个 action，并允许新的证据在下一步更新 anchor 与 bearing；
- certificate reject 或 certificate endpoint 的可恢复错误，只让当前 action 精确回退 shared mono NavDP；
- causal history 写入失败，或在 action 已授权后 controller/proof identity 出错，要求 reset，不能用 fallback 隐藏基础设施失败；
- target JPEG 改变只增加 query index，不锁定之后的 accept/reject 状态。

为了使逐动作切换保持因果，两个有短时状态的 controller 都持续 shadow update：替代 controller 接管时，fallback NavDP 仍接收当前 observation；fallback 接管时，ViNT 的 observation-only endpoint 仍更新其 RGB context。该 endpoint 不采样轨迹，也没有动作权。iPlanner/ViPlanner 的 accepted side 是 stateless 的，无需伪造历史状态。

这与当前 CEC 论文主方法一致：certificate 是逐动作安全授权，而不是一次判定整段 query 的 Novel/Revisit 标签。它也避免了 query latch 把一次早期误拒绝永久化，或把一次早期误接受扩散到整个 rollout。

## 3. 同一个 proof 的四种投影

| Accepted controller | CEC 投影 | Controller 输入 | 能支持的主张 |
| --- | --- | --- | --- |
| NavDP | `bearing_mixedgoal` | 原始 ImageGoal + 归一化 2.5 m `[forward,left]` | 保持当前正式 CEC 方法不变 |
| ViNT | `verified_anchor_imagegoal` | CEC 选中并由 SHA-256 绑定的历史 anchor JPEG | proof-carrying episodic goal 可迁移到另一视觉 controller |
| iPlanner | `bearing_pointgoal` | 2.5 m PointGoal + LingBot mono depth | certified bearing 可由另一 learned local planner 执行 |
| ViPlanner | `bearing_pointgoal` | 2.5 m PointGoal + RGB semantics + LingBot mono depth | 语义局部规划能否改善 bearing 兑现 |
| EGO-Planner | `bearing_metric_map_goal` | bearing 投影的局部 metric goal + odometry/occupancy | 非同传感器 path-quality diagnostic；不进入 headline SR |

### ViNT 的严格边界

ViNT 不支持 PointGoal，不能谎称它与 NavDP/iPlanner 消费相同 bearing token。CEC 接受后，它接收的是**被同一 proof 授权的历史 anchor 图**：

1. `certified_relocalize` 返回 `selected_anchor` 和该 JPEG 的 SHA-256；
2. `/certified_anchor_image` 只允许读取当前 goal 的 cached accepted proof 所授权的 anchor；
3. goal bytes、anchor index、causal goal boundary、cached proof digest 和磁盘 JPEG digest 任一不一致即拒绝；
4. controller proxy 再次核验上传到 ViNT 的 goal bytes 与 proof digest 一致。

所以这不是任意 memory retrieval，也不是读取 role 标签。它测试的是更一般的 **proof-carrying episodic goal interface**。若该臂失败，只能否定这种 ViNT 投影，不能否定 bearing executor portability。

### PointGoal 单位

CEC 原始输出仅是 LingBot 单目坐标中的 scale-free `[forward,left]` 方向。统一 adapter 每步重新归一化为：

```text
norm([goal_x, goal_y]) = 2.5 m
```

禁止使用 monocular translation norm、Habitat pose 或 oracle distance。NavDP 使用原有 mixed-goal endpoint；iPlanner/ViPlanner 使用原生 PointGoal endpoint。

## 4. Mono sensor 与 fallback 公平性

主支线全面采用单目条件：

- CEC：causal RGB + frozen LingBot；
- NavDP accepted/fallback：LingBot monocular sidecar；
- iPlanner/ViPlanner：同一 sidecar 输出的当前深度 PNG；
- ViNT：只消费 RGB，旧 wrapper 中“读取但不用 depth”的伪依赖已经移除；
- client/simulator metric depth 不允许进入 policy。

当前 RGB 每步只向 CEC stream 写入一次。iPlanner/ViPlanner 所需 depth 通过 read-only `/monocular_depth_query` 取得，并核验 current-image SHA、depth PNG SHA、first-40 scale receipt 与 `metric_depth_sensor_consumed=false`。

CEC reject 后的 fallback 始终是同一个原始 target ImageGoal + mono NavDP。这样 mixed Novel/Revisit 总 SR 的差异只可能来自 accepted branch；Novel 安全不会因某个 controller 有更强的私有 fallback 而虚高。

## 5. 正式比较与统计口径

### 主表：all-CEC mixed-role

同一 frozen mixed Novel/Revisit query population、同一初始状态、相同步数/距离/碰撞与到达判据：

1. CEC -> NavDP；
2. CEC -> ViNT verified anchor；
3. CEC -> iPlanner；
4. CEC -> ViPlanner。

运行时不读取 role。Novel/Revisit 分层只能在 rollout 完成后用于分析。

每个 arm 必须同时报告：

- 总 N、scene clusters、Novel/Revisit 后验分层数量；
- total SR、SPL、最终距离、碰撞率、路径长度；
- CEC accept/reject action coverage，以及 episode-level any-accept coverage；
- accepted-action 后验分层与 episode conditional SR（两者不能混淆）；
- 相对 CEC->NavDP 的 paired gain/loss、exact McNemar；
- scene-cluster bootstrap 95% CI；
- CEC、depth sidecar、controller 与端到端 decision latency；
- proof SHA、anchor SHA、controller source/checkpoint SHA receipt。

Native NavDP、native ViNT 以及 raw-memory 可以保留为上下文/消融，但不属于“CEC controller portability”主比较，不能与上述因果差值混为一个结论。

### EGO-Planner

EGO-Planner 需要 odometry、3-D occupancy 与 metric local goal，而且面向四旋翼。即使其 goal 也由 CEC proof 产生，它仍与单目 ImageGoal 主系统不满足同一感知/运动合约。当前只允许作为隔离 ROS workspace 中的 trajectory feasibility、path quality 与 latency diagnostic；不进入 headline SR，不复制 GPL 代码到主仓库。

## 6. 代码边界

- 纯契约与 proof 投影：`MemNavData/controller_portability_contract.py`；
- checkpoint/source/trajectory 审计 proxy：`MemNavData/controller_portability_proxy.py`；
- all-CEC 逐动作授权、context shadow 与 HTTP orchestration：`MemNavData/cec_controller_portability_hub.py`；
- CEC proof-bound history JPEG：
  - `NavDP/baselines/memnav/policy_agent.py::certified_anchor_image`；
  - `NavDP/baselines/memnav/memnav_server.py::/certified_anchor_image`；
- 隔离环境构建：`MemNavData/setup_controller_portability_envs.sh`；
- frozen machine-readable receipt：`MemNavData/CONTROLLER_PORTABILITY_PREFLIGHT_20260821.json`。

当前测试覆盖：

- proof 字段/单位/角色泄漏；
- 四种 accepted adapter；
- shared reject fallback；
- 逐动作 accept/reject、anchor 更新与错误边界；
- alternate takeover 时 shadow fallback，fallback 时 shadow ViNT context；
- ViNT goal JPEG 双重 SHA 绑定；
- iPlanner mono-depth sidecar；
- proxy checkpoint/source receipt 与 finite trajectory；
- MemNav cached proof 的 anchor authorization。

截至本次修改，相关五个 test files 共 **95 passed**。

## 7. 官方资产与本机 preflight

| 资产 | SHA-256 | 本机状态 |
| --- | --- | --- |
| ViNT `vint.pth` | `155fd72de2e98ae0e2fef9404072e1aefa79dae5f7f2411d4bcf7e384b83aa1f` | strict load、finite trajectory 通过 |
| iPlanner `iplanner.pth` | `685f16cde28d05249d50d24ed79ab4bdc94b3fbbcb99c8dbaed31039d11633b9` | strict load、finite、左右符号通过 |
| ViPlanner `viplanner.pt` | `2fd5219cfb160e5035d43319632b3d975637a0e770c4d455a26d3124a15ca87b` | 56 tensors strict match、finite、左右符号通过 |
| Mask2Former R50 | `54df384aa7f293a7fb13aff779d687d8216545ebdcf8b38216b6927136406536` | mmdet 3.3.0 inference 通过 |

RTX 4090 合成输入 model-only warm latency：iPlanner median 1.70 ms、ViNT 11.30 ms、ViPlanner 完整 Mask2Former+planner 约 51.39 ms。它们不含 CEC、HTTP、Habitat、sidecar 与执行器，不能写成系统 latency。

ViPlanner 使用隔离环境 `.diagnostics/controller_portability_20260821/envs/viplanner-py310-cu118`（Torch 2.0.1+cu118、mmcv 2.0.0、mmdet 3.3.0）。EGO 源码仅隔离锁定到官方 commit。

## 8. 执行顺序与停止规则

1. ~~官方 checkpoint strict load、finite trajectory、左右符号~~；
2. ~~统一 CEC proof adapter、proxy receipt、逐动作 authorization~~；
3. ~~ViNT proof-bound anchor、移除伪 depth 输入~~；
4. ~~本机 train-scene 两场景短闭环：每场景至少一个 CEC accept Revisit 和一个 CEC reject Novel~~；
5. 八场景 latency/failure pilot，冻结每个 controller 的执行/到达契约；
6. 只有前五项通过才提交至少 20 scene clusters 的 HPC paired confirmation。

不允许通过调 2.5 m 半径、改 success radius、按结果筛 episode 或给不同 fallback 来“救”某个 controller。若 iPlanner 与 ViPlanner 都不能在同一 certified bearing 下稳定执行，则论文只声称 CEC 对视觉 diffusion controller 有效；若 ViNT verified-anchor 失败，则保留为接口负结果，不反推 CEC proof 无效。

### 已完成的本机 mixed-role gate

冻结了两个 train-scene cluster、每个场景一条 Novel 和一条 Revisit query；每条 query 独立 reset 并精确重放同一 actual-online A prefix。四个 controller 均使用 hub v2，同一个 query 的 CEC proof SHA 与 anchor 在四臂间完全相同：

| Controller | Novel | Revisit | Accepted projection |
| --- | --- | --- | --- |
| NavDP | 2/2 action fallback | 2/2 action takeover | 2.5 m mixed ImageGoal/PointGoal |
| iPlanner | 2/2 action fallback | 2/2 action takeover | 2.5 m mono PointGoal |
| ViNT | 2/2 action fallback；RGB context shadow | 2/2 action takeover | proof-bound anchor JPEG |
| ViPlanner | 2/2 action fallback | 2/2 action takeover | 2.5 m mono PointGoal + semantics |

完整矩阵是 2 scenes × 2 roles × 4 controllers = 16 个独立 query rollouts；每条只给 8 simulator steps，因此每条恰好产生一个 controller decision。所有计划均满足：

- `cec_controller_portability_hub_v2` + `per_action`；
- Novel 不接管，当前动作由 shared mono NavDP 消耗配对 diffusion seed；
- Revisit 接管，四臂的 proof SHA/anchor 相同；
- iPlanner/ViPlanner 的 residual norm 为 2.5 m；
- ViNT anchor JPEG 有 SHA-256 proof binding；
- 替代 controller 接管时 fallback NavDP context 被 shadow；ViNT fallback 时自身 RGB context 被 shadow；
- `metric_depth_sensor_consumed=false`，无 runtime role 字段，无 replay diffusion sample，无隐藏 runtime failure。

独立审计输出：`.diagnostics/controller_portability_20260821/local_v2_mixed_two_scene_audit.json`，`verified=true`，SHA-256 `bf19a73fbb0a01d2ecd18bd8f89763dde846a8f20d89eb94e137a251c3cfc4e6`。

这只完成了 takeover/fallback 的真实闭环接口门，**不提供 SR 证据**。因为每条 query 只有 8 steps，表中的 2/2 是 CEC 分支覆盖，不是导航成功率。

### 早期 accepted-branch 开发 smoke

同一个 train-scene、同一个 frozen actual-online A/B prefix、同一个 Revisit-C 上，已完成以下开发 smoke：

| Controller | CEC 计划 | 结果 | 可得结论 |
| --- | ---: | --- | --- |
| NavDP | 33 | CEC 链路完整；该已知困难 episode 仍失败 | 验证端到端链路；不构成新 SR |
| iPlanner | 2 | 2/2 accept，2.5 m mono PointGoal 正常执行 | accepted adapter 可执行 |
| ViNT | 2 | 2/2 accept，proof-bound anchor JPEG 双 SHA 校验通过 | accepted adapter 可执行 |
| ViPlanner | 1 | 1/1 accept，mono depth + semantics + PointGoal 正常执行 | accepted adapter 可执行 |

后三项分别只运行 16、16、8 simulator steps，目的只是查接口、单位、proof receipt、context shadow 和传感器泄漏；它们的 episode success 均不能解释为 controller 性能。上述开发输出使用 hub v1；逐动作语义已冻结为 `cec_controller_portability_hub_v2`，正式运行只接受 v2。

当前下一步是第 5 项：先在 HPC 做小规模 latency/failure pilot，确认隔离环境、checkpoint、CEC latency 和完整 rollout 时长，再冻结正式 array 的时限与并发。现阶段仍没有 ViNT/iPlanner/ViPlanner 的统计闭环 SR，不能把 smoke 写成方法增益。

本机分段延迟 smoke（`gxdoqLR6rwA/episode_0000`、iPlanner、每个 query 仅一个决策）已经验证埋点链路：Novel reject 决策总耗时 890.05 ms；Revisit 首次 accept 决策总耗时 20639.20 ms，其中 certificate 为 19937.56 ms。该单点只用于说明首次 proof 是当前延迟主体，不代表分布或部署速度。HPC pilot 必须区分首次认证与后续缓存命中，并按 controller 报告完整分布。

## 9. 固定来源

- ViNT: https://github.com/robodhruv/visualnav-transformer (`dca79815b704e5aa9c6bdc3082351f9e3b2848c2`, MIT)
- iPlanner: https://github.com/leggedrobotics/iPlanner (`4a8d823ff9d09c3f626b727e7e00484b38f80d49`, MIT)
- ViPlanner: https://github.com/leggedrobotics/viplanner (`6fcf3c60f6fa3b28b3a11af054d6033825923789`, BSD-3-Clause)
- EGO-Planner: https://github.com/ZJU-FAST-Lab/ego-planner (`bfda51284c8c1b476043255a8145ef925a3778a5`, GPL-3.0)
- NavDP baseline API/checkpoints: `NavDP/README.md`。
