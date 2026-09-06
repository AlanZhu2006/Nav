# CEC canonical architecture（2026-09-01）

## 1. 一句话定义

> **One causal RGB stream, two synchronized geometric readouts, one
> proof-gated authority boundary, and one frozen navigation policy.**

Certified Episodic Compass（CEC）不是第二个 planner，也不是 Novel/Revisit
分类器。它是冻结导航策略外侧的一个低带宽、可拒绝的 episodic-memory
接口：历史证据充分时只授权一个单位 bearing；证据不足时完全不改变原生
ImageGoal 请求。

完整系统的主方法固定为：

```text
current RGB I_t -> atomic frame transaction -> episodic RGB history
      |                         |
      |                         v
      |                frozen LingBot geometry stream
      |                    /                     \
      |        dense current-frame readout    sparse episodic witness <- Goal G
      |        monocular depth D_hat_t         proposal -> proof -> b_t or ⊥
      |                    \                     /
      +---------------------\-------------------/------------------+
                                                                  |
Goal G -----------------------------------------------------------+-->
                                                                  |
                                  frozen NavDP policy <------------+
                         RGB + goal + D_hat_t + optional 2.5 b_t
                                             |
                                  diffusion trajectories + critic
                                             |
                                      executed trajectory
                                             |
                                      next causal RGB frame
```

主方法不输出 metric waypoint、全局地图、历史路径或机器人动作。CEC 能跨过
控制边界的唯一新增信息是二维单位方向。

## 2. 三种状态必须分开

系统共享同一条因果时间轴和相同 RGB 身份，但运行时并不是一个混合的
“memory cache”。有三类不同状态：

| 状态 | 保存内容 | 用途 | 生命周期 |
|---|---|---|---|
| Episodic RGB history | JPEG、frame index、SHA、DINO descriptor，以及与该帧绑定的几何 receipt | 长程目标检索和历史 witness | 跨 goal 持续保留 |
| LingBot streaming geometry | streaming KV、camera head state、当前/历史 pose 与 depth readout | 当前单目深度、历史 2D--3D witness、当前相对 bearing | 整个 episode 连续推进；禁止由 query 任意重置 |
| NavDP observation state | 当前 RGB/depth FIFO、goal embedding、diffusion seed | 轨迹采样和 critic 选择 | 按冻结 NavDP 合约更新 |

DINO 检索读取显式的 causal RGB/descriptor history。LingBot 的 KV cache 用于
连续几何推理，不能代替一个可寻址、可审计、可冻结 candidate ceiling 的 episodic
database。二者由 frame index 和图像 SHA 对齐，而不是互相冒充。

## 3. Goal session 与因果边界

当新 goal image `G` 在时刻 `tau_g` 安装时：

1. 冻结本次 goal 可检索的上界 `candidate_ceiling = tau_g - 1`；
2. 只有此前真实观察过的帧可以成为历史候选；
3. goal image 本身不写入 memory；
4. 当前 goal 执行期间的新 RGB 仍推进 LingBot 和 NavDP，用于当前 depth/pose，
   但不会反过来成为同一 goal 的候选；
5. goal 切换时清除 goal-conditioned shortlist、proof 和 sticky decision，长期 RGB/
   geometry history 不清除。

因此 A -> B -> A 或更多 leg 可以利用所有此前 observation，但不会利用未来帧或
当前 query 产生 trivial self-match。Novel/Revisit 标签仅用于评测分组，从不传给 runtime。

## 4. Dense short-range readout

### 4.1 目的

Dense branch 只回答：**冻结 NavDP 在当前时刻应看到怎样的 depth observation？**
它不做记忆检索，也不决定目标方向。

### 4.2 单目尺度 receipt

LingBot 输出当前相对深度 `D_t^rel`。系统只使用最初 40 个因果 RGB observation 和
已知相机安装高度估计一次尺度：

```text
s_hat_40 = Scale(I_0:39, LingBot camera/depth state, camera height)
D_hat_t  = s_hat_40 * D_t^rel
```

尺度 receipt 在第 40 帧冻结，之后不更新；禁止未来帧、整段轨迹 pooled scale、
simulator depth 或 expert pose。当前 depth payload 必须与当前 JPEG、frame index 和
receipt hash 原子绑定。

运行语义：

- frame `< 40`：向 NavDP 传 zero depth，同时继续积累合法 causal prefix；
- receipt 合法且 frame `>= 40`：传 frame-bound monocular depth；
- receipt 无效：传 zero depth，不秘密回退 metric depth；
- RGB append、stream identity 或 transaction 本身损坏：整个 geometry stream 不可信，
  要求 reset/stop，而不是伪装成普通 CEC reject。

已知相机高度在 canonical 方法中只用于这条 dense scale receipt。它不自动证明
长基线 PnP translation norm 可靠。

## 5. Sparse long-range CEC

Sparse branch 分为 proposal、witness 和 authority 三个不能互换的阶段。

### 5.1 Proposal：寻找历史地址

```text
goal image G
    -> frozen DINO descriptor
    -> cosine ranking over eligible causal history
    -> top-K=8 + temporal NMS
```

DINO 的职责是高召回地提出可能的历史地址。它没有控制权限；最高相似度不等于
“目标一定被访问过”，也不等于该候选的位姿可用。

候选随后由 SuperPoint + LightGlue correspondence 和 Fundamental-MAGSAC 的
epipolar support、coverage、match quality 做确定性排序。proposal score 本身仍不能
触发控制。

### 5.2 Witness：构造可检查的相对位姿

对候选历史帧 `I_h`：

1. 读取与 `I_h` 原子绑定的 LingBot historical depth；
2. 将匹配的历史 keypoint 从 2D lift 到 3D；
3. 使用 goal-image 上对应的 2D keypoint 运行 PnP-RANSAC；
4. 得到 goal-camera hypothesis `T_G`；
5. 与当前 streaming camera state `T_t` 组合，得到当前坐标系的原始平移向量
   `v_t`。

当前 operational certificate 固定为：

```text
PnP valid
inliers >= 16
query hull coverage >= 0.05
reference hull coverage >= 0.05
reprojection RMSE <= 2 px
```

这是一个冻结的 operational evidence check，不是形式化安全证明。reject 的含义是
“现有历史无法充分支持该 hypothesis”，不是“该目标在语义上一定是 Novel”。

### 5.3 Authority：只传最小充分控制量

对通过 certificate 的非零 `v_t`：

```text
b_t = v_t / ||v_t||
p_t = rho b_t,  rho = 2.5 m
```

只有 `b_t` 的方向信息被保留。`||v_t||` 被记录用于 audit，但 canonical controller
不消费它。2.5 m 是事先冻结的局部 residual，不是对真实目标距离的估计。

接受后的 goal pose/witness 被缓存；后续 replan 只用最新 `T_t` 更新 current-relative
bearing，避免每步重新做完整 relocalization。首次 reject 也是本 goal session 的合法
sticky abstention；切换 goal 后才重新建立 candidate ceiling 和 proof。

## 6. 唯一动作专家：frozen NavDP

CEC accept 时，NavDP 接收：

```text
current RGB observation
current monocular depth D_hat_t
original ImageGoal G
optional PointGoal residual 2.5 b_t
```

同时保留 ImageGoal 和 PointGoal 是有意设计：bearing 只提供长程方向偏置；原始 goal
appearance、局部避障、轨迹生成和 critic 选择仍由冻结 NavDP 完成。CEC 从不直接输出
动作或轨迹。

CEC reject 时，系统执行相同当前 RGB、相同 mono depth、相同 ImageGoal、相同 seed/FIFO
下的 native ImageGoal 请求，不发送 PointGoal。评测中的 exact fallback 由返回 seed、
selected trajectory 和 executed trace 对账，而不是只比较最终 SR。

因此这不是 action-level mixture of experts：dense branch 和 sparse branch 不竞争动作，
也不存在 learned router 在两个 policy 之间切换。更准确的名称是
**dual-timescale geometric sidecar with proof-gated residual authority**。

## 7. 两类失败契约

```text
CEC evidence reject / handled proof failure
    -> memory has no authority
    -> unchanged mono-native ImageGoal request

shared RGB/LingBot stream failure or stale frame transaction
    -> both current depth and memory geometry are untrusted
    -> reset_required / actuator stop
```

这两个路径不能合并。前者是正常开放集 abstention；后者是系统状态损坏。

实机上的 D435i aligned depth 只允许进入 Jetson 近障急停安全层，不进入目标条件、
CEC、NavDP observation encoder 或轨迹生成。因此准确表述是
“monocular navigation policy with an independent local depth safety layer”，而不是整台
机器人不存在深度传感器。

## 8. Metric distance 的正确位置

Canonical CEC 仍是 scale-free bearing。原因不是系统完全没有尺度，而是 dense depth
scale 与 long-range translation norm 的误差路径不同：

- first-40 height receipt 能为当前局部 depth 提供有用尺度；
- 长基线 `||v_t||` 同时承受 historical depth、camera-state drift、matching/PnP 和
  coordinate composition 误差；
- 当前 certificate 的像素 inlier/coverage/RMSE 只认证几何支持，不认证 metric norm。

现有离线审计表明 first-40 metricization 有信息，但还没有证明将 distance 暴露给
NavDP 能提高闭环 SR。因此必须区分：

| 分支 | Controller payload | 地位 |
|---|---|---|
| `verified_bearing_v1` | `2.5 b_t` | canonical paper method |
| `verified_bounded_metric_v1` | `min(d_hat_t, 2.5)b_t` | integration/trust-region ablation；不是完整 metric 方法 |
| future `verified_metric` challenger | `d_hat_t b_t`，只受 NavDP 原生输入域约束 | 尚未正式闭环评测；不得写入主方法 |

如果 full-metric challenger 与 fixed bearing 持平，结论应是“可靠 direction 已足够，
metric norm 没有额外控制价值”，而不是“系统没有真正恢复尺度”。如果它显著改善，才
授权在独立总体上验证 metric authority。

## 9. 什么属于论文主方法

### 属于 canonical architecture

- 一个因果 RGB stream；
- 显式 episodic RGB/descriptor history；
- 冻结 LingBot streaming geometry；
- first-40 causal monocular depth receipt；
- DINO top-8 temporal proposal；
- SuperPoint + LightGlue + Fundamental-MAGSAC；
- historical-depth PnP operational certificate；
- scale-free bearing + fixed 2.5 m residual；
- exact mono-native fallback；
- 一个冻结 NavDP trajectory policy。

### 只属于 baseline / ablation / negative result

- raw-DINO always-on memory；
- old SIFT/RANSAC geometry router；
- metric-depth and zero-depth controller arms；
- bounded/full metric residual challenger；
- X-NavDP controller replacement；
- Pi3X/CDEC learned relocalizer；
- graph rescue、active glance、oracle bearing；
- GOAT adaptation experiments。

它们可以解释主方法为何这样设计，但不能出现在 canonical architecture 图中。

## 10. 论文应采用的主叙事

问题不是“如何再训练一个更强的导航 policy”，而是：

> 在持续 ImageGoal navigation 中，如何让一个已经具备局部控制能力的冻结 policy
> 使用历史经验，同时避免未经支持的 memory 干扰？

对应贡献应写成：

1. **Problem formulation**：把 episodic reuse 表述为 open-set action authorization，
   而不是已知角色的 Revisit routing 或 Novel/Revisit classifier；
2. **Method**：proposal 找地址，geometric witness 建立证据，authority 只释放单位
   bearing；
3. **System composition**：同一 causal RGB/LingBot state 同时提供 dense local depth 和
   sparse long-range memory，最终只保留一个 frozen diffusion policy；
4. **Evaluation discipline**：分别报告 supported Revisit utility、unsupported-goal
   interference、exact fallback、full-mono composition 和多 goal 累积。

论文不应把 novelty 写成 DINO、LightGlue、PnP 或 LingBot 的简单相加。真正的方法对象是
**从 retrieval hypothesis 到 control authority 的受限接口和因果生命周期**。

## 11. 代码对账

| 功能 | 当前核心实现 |
|---|---|
| LingBot causal stream、RGB/DINO/KV/goal-session lifecycle | `NavDP/baselines/memnav/policy_agent.py` |
| first-40 scale 与 frame-bound mono depth | `MemNavData/monocular_depth_runtime.py` |
| certificate 输出到 fixed-bearing authority | `MemNavData/revisit_bearing_adapter.py` |
| 仿真 role-free paired evaluator | `MemNavData/eval_shared_online_role_pairs.py` |
| 实机 proof/fallback orchestration | `MemNavData/realworld_cec_hub.py` |
| frozen NavDP mixed ImageGoal/PointGoal controller | `NavDP/baselines/navdp/policy_agent.py`、`policy_network.py` |

旧 learned decoder、X-NavDP、graph rescue 与 metric challenger 即使仍保留代码路径，也不应
被主方法入口默认启用。
