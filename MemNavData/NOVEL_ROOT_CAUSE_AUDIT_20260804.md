# MemNav Novel 分支弱于 NavDP：完整根因核查

日期：2026-08-04
工作树：`/home/asus/Research/Nav-axis-uturn`
范围：只核查 start→Goal-A 的纯 Novel 闭环；不把 revisit、terminal U-turn 或
LingBot 长程 relocalization 混入结论。

## 1. 最终结论

目前 Novel 效果差不是单一的 “loss 太高” 或 “训练不够”，而是两个已经用因果实验
确认的主问题：

1. **Novel decoder 几乎不使用 goal image。** 固定完全相同的 live state 与
   diffusion seed，只把正确 Goal-A 图换成同场景另一条 episode 的错误 Goal-A 图，
   MemNav 全部 16 条候选的变化只相当于普通换 seed 变化的 `0.13%–3.16%`；原始
   NavDP 同类对照为 `176.8%`。所以当前所谓 Novel policy 主要在依赖 current-state
   和“继续向前”的轨迹先验，而不是根据目标图决定方向。
2. **碰撞 selector 与实际 controller 的执行时域不一致。** 模型输出 24 个
   waypoint，一个 waypoint 对应约 4 个数据帧；当前闭环每 8 帧重规划，实际只提交
   约前 2 个 waypoint 的时间，但旧 selector 用未来全部 24 点的碰撞/净空评分挑路。
   把评分临时限制到前 2 点后，同一 checkpoint 的 SR 从 `4/10` 提升到 `7/10`，且
   原先成功的 4 条没有一条退化。

这两个问题互相放大：diffusion 的 goal 条件很弱，16 条候选主要由采样噪声形成；
selector 又只懂安全、不懂目标进度，并且过度关注本次根本不会执行的远期段，最终
会持续选择“看起来远期更空，但不一定朝目标”的轨迹。

以下因素不是这组纯 Novel 失败的主因：retrieval top-1、predicted gate、aux
translation/yaw、revisit goal pose、LingBot 长程 rotation drift。实验强制
`gate=0` 且跳过 goal-pose tower，这些量没有进入 action 生成。

## 2. 评测协议

使用 5 个与训练场景不重叠的 Matterport3D 场景，每场景固定 2 条 start→A 路线：

```text
s8pcmisQ38h  e9zR4mvMWw7  rqfALeAoiTq  zsNo4HB9uLZ  yqstnuAEVhm
```

- 共 10 条路线；场景由固定 seed `20260803` 从 `available - training_scenes` 选出；
- Habitat-Sim 0.3.3；相同 RGB、goal、起点、navmesh、pure-pursuit controller；
- success distance `1.0 m`，max 500 frames，execution horizon 8；
- 每次 diffusion 生成 16 条候选；
- MemNav 强制 decoder gate `0`，`gate_skip_below=0.001`；
- 原始 NavDP 与 MemNav 都按 `[x forward, y left, theta]` 解码；
- 下表是内部配对 benchmark，不是 Habitat 官方 ImageNav leaderboard。

原始 NavDP 使用 Habitat metric depth 与 learned critic；MemNav 使用 LingBot predicted
depth/current-state 与几何 selector。因此 `9/10` 对 `4/10` 是完整系统对比，不是只换
一层 goal encoder 的公平消融。

## 3. 所有关键闭环结果

| Policy / intervention | SR | SPL | mean final distance | mean path |
|---|---:|---:|---:|---:|
| 原始 NavDP | **9/10** | **0.896** | 2.120 m | 6.000 m |
| flowgate2600，强制 Novel | 3/10 | 0.300 | 3.475 m | 7.811 m |
| gatecurr600，强制 Novel | 2/10 | 0.065 | 3.974 m | 13.905 m |
| residualgate1000，旧 24 点 collision selector | 4/10 | 0.374 | **2.330 m** | 10.665 m |
| decgate5570，旧 24 点 collision selector | 4/10 | 0.370 | 3.278 m | 9.796 m |
| residual，完全关闭 collision、endpoint medoid | 1/10 | 0.094 | 3.617 m | 11.693 m |
| residual，collision 只评分前 2 waypoint | **7/10** | **0.626** | 2.553 m | **6.303 m** |
| residual，GT-geodesic oracle 选候选 | 10/10 | 1.000 | 0.975 m | 4.602 m |
| residual，错误 goal + 旧 server selector | 6/10 | 0.461 | 1.606 m | 11.080 m |
| residual，错误 goal + GT oracle selector | 10/10 | 1.000 | 0.973 m | 4.602 m |

注意：

- oracle 每 8 帧偷看真实目标，只用于定位问题，不能部署；
- 正确/错误 goal 的独立闭环进程会受 CUDA 非 bitwise deterministic 与路径累积影响，
  所以 `4/10` 对 `6/10` 只说明错误目标没有稳定变差；真正定性的证据是下一节的
  同 state、同 seed、逐 tensor A/B；
- 前 2 点 selector 的 7/10 相比旧 selector 是 paired `3 gain / 0 loss`，但 n=10
  的 exact two-sided McNemar `p=0.25`。它是很强的工程信号，不是最终统计结论。

## 4. 核心证据一：模型几乎不看 goal image

### 4.1 同 state、同 seed 的因果干预

诊断过程对每个 `k` 做三次完全独立 reset/replay：

```text
A: 正确 Goal-A image + seed S
B: 错误 Goal-A image + seed S
C: 正确 Goal-A image + seed S+1
```

Gate 固定为 0；A 与 B 的当前帧、完整历史、LingBot state、DDPM 初始 noise 都完全
相同，唯一变量是 goal image。比较完整 `[16,24,3]` candidate tensor 的 RMS：

| Checkpoint / scene | k | goal-swap RMS | seed-change RMS | goal / seed |
|---|---:|---:|---:|---:|
| flowgate2600 / s8 | 40 | 0.001019 | 0.709003 | **0.144%** |
| flowgate2600 / s8 | 122 | 0.000923 | 0.698324 | **0.132%** |
| residual1000 / s8 | 40 | 0.002080 | 0.822787 | **0.253%** |
| residual1000 / s8 | 122 | 0.002738 | 0.872169 | **0.314%** |
| residual1000 / e9 | 40 | 0.024090 | 0.929100 | **2.593%** |
| residual1000 / e9 | 122 | 0.028718 | 0.908657 | **3.160%** |
| residual1000 / rq | 40 | 0.017614 | 0.965447 | **1.824%** |
| residual1000 / rq | 122 | 0.021125 | 0.993349 | **2.127%** |
| 原始 NavDP / s8 | 0 | 0.492821 | 0.278807 | **176.761%** |

结论：

- MemNav 的 goal 路径不是断线；换图会产生有限非零变化；
- 但这个变化比 diffusion noise 小约 30 到 750 倍，decoder 实际上主要由
  current-state 与 noise 控制；
- k=122 已进入现有 Goal-A 训练支持区，goal sensitivity 仍然没有本质恢复，所以
  “只补前 122 帧”是必要修复，但不会单独解决问题；
- flow checkpoint 比 residual 更不敏感，说明塌缩在 gate curriculum 之前已经
  存在；residual routing 稍有改善，但没有恢复真正的 image-goal policy；
- 原始 NavDP 在同一对 s8 goal 图上对换图高度敏感，排除了“这两张图太相似”或
  “diffusion 天生对条件只应有微小响应”的解释。

### 4.2 错误 goal 的 oracle 反证

正确 goal + oracle 和错误 goal + oracle 都是 10/10；每条路径长度只差毫米级。
在错误 goal 的 158 次规划中，平均 `99.8%` 的候选在接下来 8 个 controller frame
内都能让真实目标的 geodesic 略微下降。

这说明 oracle 10/10 不能被解释为“Novel 已经生成正确目标路线”：oracle 实际上是
从带强 forward prior 的随机候选里，每 8 帧偷看 GT 做小幅 steering。它证明候选集
有足够局部运动覆盖，但不证明模型理解目标图。

## 5. 核心证据二：selector 的时域错配

旧流程：

```text
16 条 goal 条件很弱的候选
    -> 用 LingBot 当前 predicted depth 构造障碍点
    -> 对每条候选全部 24 waypoint 计算碰撞比例 + clearance gain
    -> 取最高安全分
    -> controller 其实只执行 8 frame，然后重新规划
```

当前几何分数没有 goal progress 项：

```text
-5 * collision_fraction + 0.5 * (last_clearance - first_clearance)
```

在 oracle 诊断的 158 次规划中：

- 旧 server selector 与 GT oracle 只一致 `8/158 = 5.1%`，接近 16 选 1 的随机
  一致率 `6.25%`；
- oracle 有 `94.3%` 的规划选择了更低的 predicted collision score；
- server 相对 oracle 的 8-frame geodesic regret 平均只有 `1.63 cm`、median
  `0.63 cm`，但连续 100 多次重复后会累积成数米偏差；
- oracle 平均牺牲 `0.681` collision score，却在 Habitat navmesh 中完整 10/10
  到达，说明旧分数对远期风险的权重和本次短执行不匹配。

`pred_digit=4` 时，24 waypoint 约覆盖 96 source frames；`exec_horizon=8` 只提交约
2 个 waypoint 周期。用 opt-in 诊断开关只评分前 2 点后：

```text
SR  4/10 -> 7/10
SPL 0.374 -> 0.626
path 10.665 m -> 6.303 m
paired outcomes: 4 both-success / 0 regression / 3 gain / 3 both-fail
```

完全用 endpoint medoid 只有 1/10，说明碰撞信息确实有价值；正确方向不是“删掉
selector”，而是让 safety horizon 与 controller commitment 对齐，并让 goal-aware
信号决定安全候选之间的排序。

## 6. 为什么会学成“不看 goal”

### 6.1 这不是“NavDP 外接 memory”，而是一套从零训练的新 policy

原始 NavDP 与 MemNav 的实质差异：

| 项目 | 原始 NavDP | 当前 MemNav |
|---|---:|---:|
| checkpoint tensors / params | 1066 / 135.73M | 369--370 / 57.26M |
| decoder layers | 16 | 8 |
| conditioning tokens | 132 | 17 |
| current geometry | NavDP RGB-D encoder + metric depth | LingBot tokens + predicted depth |
| goal encoder | 原 checkpoint 已训练 image encoder | 新 NovelBranch |
| candidate selector | learned critic | predicted-depth geometric score |

把 MemNav key 去掉 `core.` 后，能与 NavDP 对上的 same-name/same-shape tensor 共
151 个，但 **0 个 exact match**。第一层 decoder self-attention 权重与 NavDP 的 cosine
只有约 `0.00045`，说明 MemNav 没有继承原始 NavDP policy 权重。

Novel backbone 名为 `imagegoal_encoder.pretrained`，但这里的 `pretrained` 只是
`DepthAnythingV2` 中新构造的 `DINOv2` 成员；DINO constructor 随即执行
`init_weights()`。`NavDP_ImageGoal_Backbone` 没有加载
`depth_anything_v2_vits.pth`，MemNav 训练也没有加载原 NavDP image encoder。

当前 checkpoint 的参数分布：

| 组 | 参数 | 占 MemNav checkpoint |
|---|---:|---:|
| Novel image branch | 27.247M | 47.6% |
| diffusion decoder | 18.952M | 33.1% |
| current-state projections/compressors | 10.519M | 18.4% |
| retrieval | 0.525M | 0.9% |

也就是说，需要学习 image-goal 与 action 映射的约 46.2M 参数基本从零开始。

还有一个未来做权重迁移时必须处理的细节：原 NavDP 数据把 6 通道组织为
`[goal,current]`，MemNav NovelBranch 是 `[current,goal]`。当前从零训练内部是一致的，
不是现有 bug；但直接加载原 NavDP patch-conv 时必须交换两个 3-channel half，或者
统一输入顺序。

### 6.2 Goal-A loader 只训练非常晚的 current frame

现有常量：

```text
anchor margin = num_scale + window - 1 = 8 + 32 - 1 = 39
exclude_recent = 83
Goal-A k_lo = 39 + 83 = 122
```

这源于 loader 强制 `E(k)=[39,k-83]` 非空。后果：

- inference 第一条有效计划在 k=40；训练却从不见 k=40..121；
- Goal-A leg 少于约 126 帧时，整条 Novel goal 被 dataset 丢掉；
- 长 leg 也只从很晚的位置采样，剩余目标常常已经很近；
- train `encode_memory` 对 Novel row 仍然计算一次无意义的 revisit goal pose，因此
  loader 才人为要求候选集合非空；inference 在低 gate 下会直接跳过该 tower。

本次 10 条 scene-disjoint eval route 的 Goal-A switch：

| route | switch | 当前 loader 是否有 Goal-A sample | k=122 后 expert remaining |
|---|---:|---|---:|
| s8/0000 | 135 | 有，但只有 9 个 k | 0.451 m |
| s8/0001 | 234 | 有 | 4.171 m |
| e9/0000 | 208 | 有 | 3.168 m |
| e9/0001 | 102 | **整条丢弃** | - |
| rq/0000 | 178 | 有 | 2.067 m |
| rq/0001 | 99 | **整条丢弃** | - |
| zs/0000 | 97 | **整条丢弃** | - |
| zs/0001 | 212 | 有 | 3.345 m |
| yq/0000 | 143 | 有，但只有 17 个 k | 0.752 m |
| yq/0001 | 122 | **整条丢弃** | - |

所以 4/10 eval route 在这种 loader 下完全不能产生 Goal-A training row；另外 2 条
进入支持区时已经在 1 m success radius 内。

flow run 的 trainer state 显示：

- 4456 个 goal samples，其中 Goal-A 1354；
- step 2600 = 4.668 epochs，global batch 8；
- 因而纯 Goal-A 约只有 `1354 * 4.668 ~= 6320` 次 sample exposure；
- 训练场景 50 个，却要从零训练 27.25M Novel encoder + 18.95M decoder。

### 6.3 目标图对 action loss 是可绕过的

每个 Goal-A current 只与“同一 expert trajectory 未来的终点图”配对；current camera
本来就沿 expert path 朝前。模型没有看到：

- 同一个 current 对应两个不同有效 goal、因此动作必须不同；
- 同一个 current 搭配错误 goal 时不应输出同一动作；
- 原 NavDP teacher 在同 state/goal 下应该输出什么。

于是最容易降低 epsilon-MSE 的捷径是：从 current-state/depth 推断局部可行方向，
沿训练轨迹的 forward prior 输出动作，忽略只有 4 个 token 的 Novel condition。
同 state goal-swap 实验正好观测到了这个捷径。

### 6.4 优化配置还有两个次要但确定的问题

配置写了 `weight_decay=1e-4` 与 `warmup_ratio=0.05`，但自定义 trainer 实际：

```python
torch.optim.Adam(params, lr=lr)              # 没有 weight_decay
LinearLR(start_factor=1.0, end_factor=0.5)   # 没有 warmup
```

同时 scratch Novel、current-state、decoder、retrieval 全部共用 `1e-4`，没有冻结阶段或
差分 learning rate。这不是 goal collapse 的唯一原因，但对小数据训练 27M scratch
vision backbone 很不合理。

## 7. 为什么 W&B loss 看起来还能下降

W&B 中的 `action_loss_novel` 是 teacher-forced 数据上的 one-step epsilon-MSE，不是：

- 闭环 SR；
- goal-swap sensitivity；
- 16 条 candidate 中最终选中哪条；
- selector 的 collision/progress loss。

当前 trainer 根本没有 selector loss。最大的闭环瓶颈因此不会出现在 W&B 任意一张
loss 图里。

另外，loader 给 Novel loss 的多数是 late/easy current，current-state shortcut 足以
让 epsilon-MSE 很低。固定 72-sample evaluator 已经显示 gate curriculum 后：

```text
Novel action epsilon-MSE: flow 0.057873 -> gatecurr 0.061428（轻微变差）
retrieval set loss:       0.204852 -> 0.134628（改善）
gate accuracy:            69.44% -> 80.56%（改善）
```

这正说明 retrieval/gate 可以变好而 Novel action 与闭环不变甚至变差。Aux 只在 revisit
row 统计且当前 empirical head 无优化梯度，也不能解释纯 Novel SR。

## 8. 已排除或降级的假设

### 8.1 “主要是 LingBot 转弯/长程 pose drift”

不是这组纯 Novel 的主因。强制 gate=0 后 goal-pose/revisit tower 被跳过，aux pose、
retrieved anchor、LingBot relative yaw 都没有进入 decoder。LingBot predicted depth 与
current tokens仍会影响局部避障，所以它们可能贡献次要误差，但不能解释换 goal 几乎
不改变 action。

### 8.2 “retrieval loss 还不够低”

不是这组实验的问题。Retrieval 只留下反事实日志，action 使用固定 gate=0。继续只调
retrieval loss 不会让 Novel 学会看 goal。

### 8.3 “axis / theta / aux 还错”

Waypoint 的 `[x forward,y left]` 在 server 和 Habitat client 之间已经用纯 forward /
pure left round-trip 核对；theta 对两边的 waypoint-to-world 都没有参与位置控制。旧
axis 修复对训练标签很重要，但现在 4/10 的直接证据是 goal-insensitivity 和 selector
horizon，不是残留的 action-axis 翻转。

### 8.4 “再多训几个 epoch 就会好”

现有 `decgate5570` 纯 Novel 仍是 4/10，与 residual1000 的 4/10 基本相同；flow 与
residual 的 goal sensitivity 都很低。因此 available evidence 不支持不改数据/结构、
只延长当前 objective。`decgate5570` 不是 residual1000 的严格同 lineage continuation，
所以这里的准确表述是“训练时长本身不充分”，不是数学上证明任何长训都不可能改善。

## 9. 修复优先级

### P0-A：先修 selector/controller contract

不要把 `2` 永久硬编码为通用值。Server 应获得 controller 的真实 commitment：

```text
committed_waypoints = ceil(exec_horizon_frames / pred_digit)
```

推荐评分：

1. 对 committed prefix 做强 collision veto；
2. 对更远 waypoint 只做 discount safety cost，而不是一票否决当前正确方向；
3. 在安全候选之间使用 goal-aware progress/density score；
4. 记录 selector score、被选 candidate、实际执行 prefix，确保 train/eval 可核查。

本次 `MEMNAV_COLLISION_HORIZON_WAYPOINTS=2` 只是一项已验证的诊断开关，默认仍为 0
（历史 24 点行为）。扩大 seed/scene 之前不应直接宣称最终修复。

### P0-B：恢复真正的 base-policy-preserving ImageNav

最稳妥的结构不是继续让 8-layer scratch decoder 同时承担 Novel 与 revisit，而是保留
并冻结原始 NavDP image-goal base：

```text
epsilon_final = epsilon_navdp + gate * delta_epsilon_memory
```

由结构保证：

```text
gate = 0  =>  输出严格等于原始 NavDP
```

Memory 只学习有界 residual；训练初期只更新 retrieval、gate、memory residual。
这会直接消除当前 “Novel 名字相同但实际上不是 NavDP” 的问题。

如果暂时保留现架构，至少：

- 从原 NavDP checkpoint 加载 image encoder，而不是随机 DINO；
- 处理 RGB/BGR preprocessing 与 `[goal,current]` / `[current,goal]` channel-half 映射；
- 用原 NavDP 对相同 current/goal/noise/timestep 做 denoising distillation；
- 增加 `gate=0` consistency，禁止 revisit 训练破坏 base output。

### P1：修 Goal-A sampling 与 no-candidate training

1. Goal-A 从 inference 首次可规划的 `k=40` 开始采样，而不是 122；
2. 允许 `cand_mask` 全空的 Novel row；
3. 对无 candidate row 跳过 goal-append/revisit pose tower，revisit token 置为 neutral，
   与 inference 一致；
4. 按 goal distance/turn complexity 分桶采样，显式 oversample early、far、turn rows；
5. 不再丢弃 45--125 帧的短 Goal-A leg。

只改 `k_lo` 不够：当前 `encode_memory` 会把 anchor clamp 到 `[39,k-1]` 并无条件算
goal pose；必须同时完成 no-candidate fast path。

### P1：让 goal 不再是可忽略变量

- 同一 current 构造多目标训练/teacher pairs；
- 加入错误 goal 对照，监控 output sensitivity，而不是为错误 goal 随意伪造 action GT；
- 使用 NavDP teacher 的 trajectory/noise target 做 goal-conditioned distillation；
- validation 固定报告：正确 goal loss、shuffled-goal loss、两者 output RMS、Novel
  closed-loop SR；
- early stopping 同时约束 Novel 与 revisit，不再只看总 action/retrieval loss。

### P2：优化器与训练策略

- 改为 AdamW 并真正传入 `weight_decay`；
- 使用真实 warmup + decay；
- pretrained base 先冻结，memory/new heads 用较高 LR，base 若解冻则用更低 LR；
- 在 8 小时长训前先要求 local goal-sensitivity smoke 通过，否则长训只会继续学习
  current-state shortcut。

## 10. 下一轮验收标准

在提交新长训前，至少满足：

1. 同 state/same seed 的 correct-vs-swapped goal 输出差异不再低两个数量级，并用原
   NavDP teacher 作相对标尺；
2. 5-scene × 2-route × 至少 3 seeds 上，prefix-aligned selector 稳定优于 full24；
3. Novel scene-disjoint SR 不低于冻结的 NavDP base；
4. revisit 另报 2-leg/3-leg SR、retrieval hit、gate、pose 长尾，不能用 Novel 提升掩盖；
5. W&B 增加 goal-sensitivity 与 selector panel，而不只看 epsilon-MSE。

## 11. 代码和结果位置

诊断代码：

```text
MemNavData/eval_2leg_habitat.py
MemNavData/run_novel_navdp_ab.sh
MemNavData/diagnose_novel_goal_sensitivity.py
NavDP/baselines/memnav/policy_agent.py
```

关键生产代码：

```text
InternNav/internnav/dataset/memnav_dataset_lerobot.py
InternNav/internnav/model/basemodel/memnav/memnav_policy.py
InternNav/internnav/model/encoder/navdp_backbone.py
InternNav/internnav/trainer/memnav_trainer.py
InternNav/internnav/model/basemodel/memnav/collision_check.py
```

结果：

```text
.diagnostics/novel_navdp_ab_20260803/results/full_default/navdp
.diagnostics/novel_navdp_ab_20260803/results/full_forced_novel_v2/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/audit_decgate5570_forced_novel/decgate5570
.diagnostics/novel_navdp_ab_20260803/results/audit_medoid_selector_full/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/audit_collision_horizon2_full/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/audit_oracle_selector_full/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/audit_server_selector_goal_swap/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/audit_oracle_selector_goal_swap/residualgate1000
.diagnostics/novel_navdp_ab_20260803/goal_sensitivity_*.json
```

本轮所有代码修改都只在个人子工作树 `/home/asus/Research/Nav-axis-uturn`；没有修改
母目录 `/home/asus/Research/Nav`。

## 12. 诊断改动的安全边界与验证

本轮没有修改训练 objective，也没有把 oracle 或前 2 点选择强行设为生产默认值：

- evaluator 默认仍为 `trajectory_selector=server`、`leg1_goal_source=own`；
- `MEMNAV_COLLISION_HORIZON_WAYPOINTS=0` 默认保留历史的全部 24 点评分；
- oracle 只能在 Habitat evaluator 中显式启用，不能进入部署 server；
- goal-sensitivity 工具只连接外部 server、reset/replay 并写 JSON，不启动训练。

最终验证：

```text
python -m py_compile ...                                  PASS
bash -n MemNavData/run_novel_navdp_ab.sh                  PASS
git diff --check                                          PASS
InternNav/tests/unit_test/test_memnav_collision.py        6 passed
```

母目录当前自身存在其他未提交文件，不能把它描述为 clean；这里能确认的是，本轮所有
补丁、报告和诊断输出的写入路径均位于 `Nav-axis-uturn` 子工作树。
