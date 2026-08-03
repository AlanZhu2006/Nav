# MemNav Novel 分支与原始 NavDP 配对评测

日期：2026-08-04

工作树：`/home/asus/Research/Nav-axis-uturn`

目的：回答“MemNav 的 Novel 能力是否仍弱于 NavDP，以及 gate curriculum 是否已经解决该问题”。

## 1. 结论先行

这个问题尚未解决，而且现在已经从推断变成了闭环证据。

在 5 个不属于 MemNav 训练场景的 Matterport3D 场景、每场景 2 条固定路线，共 10 条 start→Goal A 路线上：

- 原始 NavDP：SR `9/10 = 0.90`，SPL `0.896`；
- flowgate2600 的纯 Novel：SR `3/10 = 0.30`；
- gatecurr600 的纯 Novel：SR `2/10 = 0.20`；
- residualgate1000 的纯 Novel：SR `4/10 = 0.40`；
- residualgate1000 按当前 soft gate 部署、低于 0.5 时跳过 pose tower：SR 仍为 `4/10 = 0.40`。

因此：

1. gate curriculum 改善了此前“revisit 信息进不了 decoder”的问题，但没有保护 Novel 基础能力；
2. residual fusion 是目前最好的 MemNav 版本，相比 gatecurr600 有恢复，但仍明显未达到原 NavDP；
3. 当前 `gate_skip_below=0.5` 不是严格的“低 gate 完全退回 Novel”，也没有提高最终 SR；
4. 最大结构问题不是 LingBot 转弯 pose，而是 MemNav 的所谓 Novel 分支并不是被完整保留下来的原 NavDP policy。

样本只有 10 条，所以这是明确的工程诊断，不是足以发表的统计结论。NavDP 与 residual 的配对 exact McNemar 检验为 `p=0.125`；应扩大场景和 diffusion seed 后再报告最终数字。

## 2. 什么叫本实验中的“纯 Novel”

每条 episode 只执行 start→Goal A，并在到达或用完预算后停止。Goal A 出现前不存在已经走过的目标区域，因此它是这个 episode 的视觉新目标。

对 MemNav 使用：

- `--gate_override 0`：decoder 实际使用的 gate 固定为 0；
- `--gate_skip_below 0.001`：不重算 revisit goal pose；
- 原模型预测出来的 gate 仍单独记录，但不参与动作生成。

这隔离的是 **MemNav 自己的 Novel 路径**，不是把模型替换成原始 NavDP。LingBot 当前状态、MemNav 的 Novel encoder、共享 diffusion decoder 和 MemNav 候选轨迹选择仍照常工作。

第一次目录 `full_forced_novel` 无效：当时 evaluator 只把 `gate_override` 传给 Leg B，没有传给 Goal A。发现 actual gate 不为 0 后立刻停止，并修正作用域。`gate_override_preflight_v3` 用 42 帧验证了：

- decoder actual gate：严格为 `0.0`；
- 原始 predicted gate：`1.496e-8`，独立记录。

正式有效结果目录是 `full_forced_novel_v2`。

## 3. 严格对照设置

### 3.1 场景与路线

从可用 MP3D 场景中排除 50 个 MemNav 训练场景，再以 seed `20260803` 固定抽取：

1. `s8pcmisQ38h`
2. `e9zR4mvMWw7`
3. `rqfALeAoiTq`
4. `zsNo4HB9uLZ`
5. `yqstnuAEVhm`

每个场景使用同一批 2-leg generator 生成的前两条 episode，但本实验只跑 Goal A。场景、起点、目标图、目标位置、Habitat controller 和 seed 在各模型之间完全相同。

### 3.2 闭环参数

| 参数 | 值 |
|---|---:|
| Habitat-Sim | 0.3.3 |
| Goal A mode | policy-driven |
| success distance | 1.0 m |
| max frames | 500 |
| execution horizon | 8 frames |
| base seed | 20260803 |
| MemNav DDPM samples | 16 |
| MemNav window / scale frames | 32 / 8 |
| retrieval | trained head |
| exclude recent | 83 |
| terminal U-turn / visual refine | off / off |

500 帧超过这 10 条 expert 轨迹中最长帧数的两倍。MemNav 需要先积累 41 帧才能首次规划，这段时间机器人不移动；它占用 frame budget，但不增加 path length。原 NavDP 可以从第一帧规划。这是完整系统的真实差别之一，应在更大评测中同时报告 decision budget。

### 3.3 Checkpoint

| 模型 | SHA-256 | 说明 |
|---|---|---|
| 原始 NavDP | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` | 官方/主仓库 checkpoint，保留 learned critic |
| flowgate2600 | `debd079c6f578e9c6e2c1f0e70f6dc8fc2c2230785c28d6da2fae118a665b38b` | curriculum 前基线 |
| gatecurr600 | `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7` | 从 flowgate2600 暖启动，gate curriculum/all-leg |
| residualgate1000 | `b11938a3030a67a3077355d5b9873821cfe3ed9530be81f52e747848300cc2ae` | residual fusion；8 小时任务保存的 checkpoint-1000 |

Residual 任务计划 5570 step，8 小时 walltime 到达时实际约 step 1027，最后落盘 checkpoint 是 step 1000。因此它是目前最好但尚未完整训练结束的版本，不能把“训练不够”排除；不过在继续长训前必须先加入 Novel preservation 约束，否则共享 decoder 可能继续发生负迁移。

## 4. 结果

### 4.1 总体

| 模型/干预 | SR | SPL | 最终距离均值 | 路径长度均值 | predicted false-revisit plans |
|---|---:|---:|---:|---:|---:|
| 原始 NavDP | **0.90** | **0.896** | 2.120 m | 6.000 m | N/A |
| flowgate2600，强制 Novel | 0.30 | 0.300 | 3.475 m | 7.811 m | 47/264 = 17.80% |
| gatecurr600，强制 Novel | 0.20 | 0.065 | 3.974 m | 13.905 m | 112/525 = 21.33% |
| residualgate1000，强制 Novel | **0.40** | **0.374** | 2.330 m | 10.665 m | 25/369 = 6.78% |
| residualgate1000，预测 gate，pose skip < 0.5 | 0.40 | 0.336 | **2.164 m** | 10.521 m | 1/371 = 0.27% |

强制 Novel 行中的 false-revisit 是“模型本来会如何预测”的反事实日志；actual gate 已被固定为 0，因此这些 false positive 没有进入动作生成。

NavDP 唯一失败路线的终点误差为 12.33 m，抬高了它的平均最终距离。因此平均距离不能替代 SR/SPL：NavDP 在其余 9 条都进入了 1 m，Residual 则有多条停在 1.5–3 m。

### 4.2 逐路线成功配对

`1` 表示在 1 m 内成功，`0` 表示失败。

| 场景 / episode | NavDP | flow Novel | gatecurr Novel | residual Novel | residual deployed |
|---|---:|---:|---:|---:|---:|
| e9 / 0000 | 1 | 0 | 0 | 0 | 0 |
| e9 / 0001 | 1 | 0 | 0 | 1 | 1 |
| rqf / 0000 | 1 | 1 | 0 | 0 | 0 |
| rqf / 0001 | 1 | 1 | 0 | 1 | 1 |
| s8 / 0000 | 1 | 0 | 0 | 1 | 1 |
| s8 / 0001 | 1 | 0 | 0 | 0 | 0 |
| yq / 0000 | 1 | 0 | 1 | 0 | 1 |
| yq / 0001 | 1 | 1 | 0 | 0 | 0 |
| zs / 0000 | 1 | 0 | 1 | 0 | 0 |
| zs / 0001 | 0 | 0 | 0 | 1 | 0 |

Residual pure Novel 在 `zs/0001` 成功而 NavDP 失败，说明新表征不是毫无价值，二者存在互补路线；但 NavDP-only 成功 6 条、Residual-only 成功 1 条，整体差距仍大。

不同进程的 CUDA kernel 没有开启 bitwise deterministic 模式。虽然显式 DDPM seed 相同，`residual Novel` 与 `residual deployed` 的单条路线互换不能被当作严格的 gate 因果效应。可以可靠使用的是：两种模式总体都只有 4/10，部署 soft gate 没有形成可见的总体提升。

## 5. 为什么 gate curriculum 没有保护 Novel

Gate curriculum 解决的是另一个问题：训练初期先用 GT gate 打开正确 branch，让 decoder 学会“怎样使用 revisit”，再逐渐回到 predicted gate。它没有保证“gate=0 时仍等于原 NavDP”。

MemNav 的 Novel 路径与原 NavDP 有四个实质差别：

1. 当前观测来自 LingBot streaming token 和 depth feature，并经过新的 compressor；
2. 目标条件来自新的 DINO current/goal patch-pair `NovelBranch`；
3. Novel 与 revisit 共用一个重新训练的 diffusion decoder；
4. MemNav 没有 NavDP 原来的 learned critic，16 条候选轨迹改用当前深度的几何碰撞评分。

因此即使 gate=0，运行的也不是原始 NavDP。Revisit/all-leg 更新可以通过共享 decoder 破坏 Novel，离线 action MSE 的小变化在闭环中会累积成很大的 SR 差异。gatecurr600 从 flow 的 3/10 降到 2/10，就是当前最直接的负迁移信号；Residual 把它恢复到 4/10，但没有建立“绝不弱于 base”的结构保证。

## 6. `gate_skip_below` 的准确含义

`gate_skip_below=0.5` 只控制是否执行昂贵的 LingBot goal-insert / camera-pose tower：

- 低于阈值且没有 cache 时，revisit readout 被置零；
- predicted soft gate 本身仍传入 decoder；
- decoder 仍为 revisit token slot 添加 positional embedding，并用 `log(gate)` 做 attention bias；
- 一旦某个 goal/anchor 的 pose 已经进入 cache，后续低 gate 也能取到它。

所以它是计算优化和软抑制，不是严格的 fail-closed 路由。Residual fusion 只保证 visual branch 的 bias 不再乘 `log(1-g)`，但 cross-attention 中仍存在 revisit slots；数学上不能保证：

```text
policy(gate≈0) == policy(Novel-only base)
```

这也是下一版不能只继续调 threshold 的原因。

## 7. 下一步最有效的改法

### 7.1 Safe memory augmentation（首选）

保留并冻结一条**完整原始 NavDP image-goal base policy**，包括它的 RGB-D encoder、diffusion decoder 和 learned critic。Memory 只预测有界 residual：

```text
epsilon_final = epsilon_navdp + gate * delta_epsilon_memory
```

或者在最终 waypoint 上做同样的 residual。这样由结构直接保证：

```text
gate = 0  =>  MemNav 输出严格等于原 NavDP
```

这比“把 visual token 永远保留在共享 attention 里”更强，因为后者不能阻止归一化、位置 token 和共享权重产生干扰。创新点可以概括为 **base-policy-preserving long-term memory**：记忆只在有证据时增益，不能破坏原有 Novel 能力。

### 7.2 Novel consistency / distillation

在同一 noisy action、同一 timestep 下加入：

```text
L_safe = || epsilon_mem(g=0) - stopgrad(epsilon_navdp) ||^2
```

初期冻结 base，只训练 gate、retrieval 和 memory residual；后期如需联合微调，也保留 `L_safe`。验证集必须同时 early-stop 两类指标：Novel closed-loop SR 和 revisit closed-loop SR，不能只看总 action loss。

### 7.3 先隔离 critic/selector，再开长训

当前 A/B 比较同时改变了 denoiser 和候选轨迹 selector。最便宜的下一项诊断是：固定同一批 MemNav diffusion candidates，分别用：

1. 当前单帧几何碰撞 selector；
2. 原 NavDP learned critic；
3. oracle GT-mesh collision score（只用于诊断）。

如果原 critic 能显著恢复 Novel SR，先修 selector；否则主修 base denoiser/conditioning。这个实验比直接再跑 8 小时更能减少不确定性。

### 7.4 扩大评测

完成结构修复后至少运行：

- 20 个 scene-disjoint 场景；
- 每场景至少 4 条 Goal A；
- 每条路线 3 个 diffusion seed；
- 同时报 SR、SPL、final distance、decision count、近目标失败率；
- 再单独报告 2-leg/3-leg revisit，避免 Novel 提升掩盖记忆退化。

## 8. 结果与代码位置

有效结果：

```text
.diagnostics/novel_navdp_ab_20260803/results/full_default/navdp
.diagnostics/novel_navdp_ab_20260803/results/full_forced_novel_v2/flowgate2600
.diagnostics/novel_navdp_ab_20260803/results/full_forced_novel_v2/gatecurr600
.diagnostics/novel_navdp_ab_20260803/results/full_forced_novel_v2/residualgate1000
.diagnostics/novel_navdp_ab_20260803/results/full_residual_deploy_skip05/residualgate1000
```

无效、禁止引用：

```text
.diagnostics/novel_navdp_ab_20260803/results/full_forced_novel
```

评测代码：

```text
MemNavData/eval_2leg_habitat.py
MemNavData/run_novel_navdp_ab.sh
NavDP/baselines/memnav/memnav_server.py
NavDP/baselines/navdp/navdp_server.py
```

所有本轮代码修改和新增报告都位于子工作树 `/home/asus/Research/Nav-axis-uturn`。本轮没有向母目录 `/home/asus/Research/Nav` 写入；这里只读加载了原 NavDP checkpoint 和 LingBot 权重。母目录本来已有的未提交文件不属于本实验产物，也不会被本次 commit 纳入。
