# MemNav metric pose × frozen NavDP controller：真实两段闭环实验

日期：2026-08-04
工作目录：`/home/asus/Research/Nav-axis-uturn`（没有修改母目录 `/home/asus/Research/Nav`）
基线提交：`edca2ddc05b8f0ef09bd7442c53d404b0dcb3a9c`

## 1. 结论

当前最有效的方案不是继续让小数据训练的 MemNav diffusion decoder 从头学习
waypoint，而是把两个已经分别验证可靠的模块组合起来：

1. MemNav/LingBot 从长期 memory 中检索历史帧，并恢复目标相对 metric pose；
2. 冻结的官方 NavDP 使用 image + point-goal mixed decoder 做局部避障与动作生成。

在 5 个 scene-disjoint MP3D 场景、10 个 episode 的相同 Habitat 闭环协议下：

| 指标 | Hybrid pose | 纯官方 NavDP |
|---|---:|---:|
| Goal A SR | 9/10 = 0.90 | 9/10 = 0.90 |
| Goal B SR，给定 A 成功 | **9/9 = 1.00** | 3/9 = 0.333 |
| 两段联合 SR | **9/10 = 0.90** | 3/10 = 0.30 |
| Goal B SPL，给定 A 成功 | **0.830** | 0.0828 |
| Goal B SPL，A 失败记 0 | **0.747** | 0.0746 |
| Goal B 平均路径，给定 A 成功 | **3.92 m** | 12.23 m |
| Goal B 平均控制步数 | **126.8** | 339.2 |

成对比较为 hybrid 赢 6、平 4、输 0。唯一没有执行 B 的样本是
`zsNo4HB9uLZ/episode_0001`：官方 NavDP 在第一段就没有到达 A，最终距 A
12.33 m，因此它同时限制了两种方法的联合 SR。

逐 episode 数值保存在
`MemNavData/HYBRID_POSE_UNSEEN_RESULTS_20260804.csv`。

## 2. 模型与评测口径

### 2.1 Checkpoint

| 模型 | 路径 | SHA256 |
|---|---|---|
| MemNav `gatecurr600` | `.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt` | `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7` |
| 官方 NavDP | `/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt` | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` |

母目录中的 NavDP checkpoint 只读使用，没有修改母目录代码或权重。

### 2.2 场景与 episode

- `e9zR4mvMWw7`
- `rqfALeAoiTq`
- `s8pcmisQ38h`
- `yqstnuAEVhm`
- `zsNo4HB9uLZ`
- 每个场景 2 个 episode，共 10 个。

### 2.3 闭环协议

- Goal A：官方 NavDP image-goal policy；
- 去程的每一帧同时、且仅一次写入 MemNav 长期 memory；
- Goal B：
  - baseline 使用官方 NavDP image-goal；
  - hybrid 使用 MemNav auto retrieval + LingBot metric pose，再交给官方
    NavDP `navdp_step_ip_mixgoal`；
- `success_dist = 1.0 m`；
- `max_steps = 500`；
- `exec_horizon = 8`；
- trajectory selector 使用 server 自带 critic；
- 不启用 terminal U-turn 或 visual refine；
- scene 内 episode seed 分别为 `20260803`、`20260804`；
- MemNav：`W=32`、`S=8`、`exclude_recent=32`、trained retrieval head、
  `flow_gate=auto`、ground scale 上限 6.0。

因此 hybrid 与纯 NavDP 的 Goal-A 轨迹和失败样本完全配对，B 的差异只来自是否加入
memory metric pose。

## 3. 新控制链

```text
goal image
    │
    ▼
DINO retrieval ──► history anchor
                       │
current LingBot pose ──┼──► relative metric aux_pose = [forward, left]
goal LingBot pose ─────┘
                       │
                       ▼
       frozen NavDP image + point-goal mixed decoder
                       │
                       ▼
              collision-scored waypoint
```

这里 gate 仍被记录用于诊断，但不再决定是否让准确 pose 进入动作 decoder。这样避免了
旧 gate 在在线路径上接近 0、把 revisit 信息直接关掉的问题；同时保留官方 NavDP 已经
训练好的 RGB-D 局部避障能力与 Novel image-goal 能力。

`NavDP/baselines/memnav/memnav_server.py` 新增 `/posegoal_step`：

```text
append current frame
→ retrieve anchor
→ LingBot goal re-insert
→ relative metric pose
→ return（不运行 MemNav diffusion）
```

它与原 `/imagegoal_step` 使用完全相同的 retrieval、pose、scale 和 cache 路径，只提前
跳过不再使用的 MemNav waypoint decoder。

如果返程刚开始时还没有合法的长期候选帧，或 oracle anchor 暂时落在
`exclude_recent` 内，`/posegoal_step` 会明确返回错误而不是伪造一个被 clamp 的 pose。
评测器此时让官方 NavDP 临时使用原生 image-goal，并正常推进其内部状态；下一次
MemNav pose 合法后再自动切回 image + point-goal mixed controller。

## 4. 为什么旧方案在真实路径上失败

以 `s8pcmisQ38h/episode_0001`、seed `20260803` 为因果诊断样本。

### 4.1 旧 gate 关闭了 revisit

- 自动 retrieval 下第一步 gate：约 `0.00043`；
- 全返程平均 gate：约 `0.00054`；
- 原 gate 最终距 B：`33.94 m`；
- 强制 `gate=1` 最终距 B：`33.88 m`。

所以 gate 是问题之一，但单独强开 gate 不能修复 decoder。

### 4.2 固定 83 帧排除区间不适配 NavDP 在线速度

- 实际路径中离 B 最近的是 frame 160，距 B `0.175 m`；
- A 在约 frame 208 到达，正确 revisit gap 只有约 48 帧；
- `exclude_recent=83` 使 frame 160 在返程开始时不合法；
- 强制 frame 160 时，服务器要等约 40 步才允许它进入候选集。

正式 9 条成功去程中的真实最近 gap 范围为 31–100 帧。固定 83 会排除其中大多数
在线正确段；本次使用 32，并依靠 sticky anchor 避免立即自匹配。32 仍是实验参数，
最终应改成基于空间/视差的自适应排除，而不是再固化另一个帧数常量。

### 4.3 LingBot pose 实际上是准的

同一时刻，强制正确 anchor 后：

- LingBot aux：`[-1.489, 0.251] m`；
- Habitat 近似真值：`[-1.729, 0.327] m`；
- 2D translation error：`0.251 m`；
- direction error：`1.15°`。

这排除了该样本上的 axis、metric scale 和长程 rotation drift 作为主因。

### 4.4 失败发生在旧 MemNav action decoder

虽然 aux 很准，旧 decoder 的首个采样轨迹终点约为
`[-4.10, -2.54] m`，长度 4.82 m，方向相对真实目标偏约 43°。在
`exclude_recent=32 + 正确 anchor + gate=1` 下，它走 6.81 m 后仍离 B 2.73 m，
没有净进展。

但同一 checkpoint 在生成器原轨迹上使用 `GT anchor + gate=1` 能成功，SPL 0.841。
这说明 decoder 只适应训练时 A*/生成器的专家状态；NavDP 实际闭环产生的小幅偏位和
视觉变化形成 imitation-learning covariate shift。

### 4.5 新组合的单样本因果对照

| 方法 | B 成功 | B 路径 | SPL |
|---|---:|---:|---:|
| 旧 MemNav，自动 gate | 否 | 15.58 m 后仍发散 | 0 |
| 旧 MemNav，正确 anchor + gate=1 | 否 | 6.81 m 后停滞 | 0 |
| 纯官方 NavDP image-goal | 是 | 14.63 m | 0.190 |
| 新 hybrid，正确 anchor | 是 | 2.697 m | 1.000 |
| 新 hybrid，自动 retrieval | 是 | 2.658 m | 1.000 |

自动 retrieval 第一帧选 143，而 metric-nearest 为 160；两者 aux 分别约
`[-1.522, 0.243]` 和 `[-1.489, 0.251] m`。因此 retrieval 不必命中唯一帧，只需命中
同一共视局部段。

## 5. 与旧结果的关系

旧 `gatecurr600` 的 unseen-scene B 结果是 6/10、平均 SPL 0.395，但它使用
**GT leg-1 replay**，不是策略实际走出的 memory。新 hybrid 在更严格的真实 NavDP
路径上，对所有可执行 B 的 episode 达到 9/9，条件 SPL 0.830。

因此旧 replay SR 不能再被解释为真实端到端上限；它同时混合了：

- 生成器轨迹与训练分布一致；
- gate/Novel 分支可能完成部分返回；
- 没有暴露真实 policy path 的候选间隔和 covariate shift。

## 6. 代码改动

- `MemNavData/eval_2leg_habitat.py`
  - 新增 `hybrid_oracle` 的真实双服务执行；
  - 新增 `hybrid_pose`；
  - 记录真实在线 memory frame 与 Habitat 位置；
  - 新增 `gt_path_nearest` 因果干预；
  - 支持独立 Goal-A success radius；
  - 保存 aux、anchor、controller 与路径诊断；
  - pose 不可用时 fail-safe 回退到官方 NavDP image-goal。
- `NavDP/baselines/memnav/policy_agent.py`
  - `plan(..., pose_only=True)` 在 relative pose 后提前返回。
- `NavDP/baselines/memnav/memnav_server.py`
  - 新增 `/posegoal_step`。
- `NavDP/baselines/navdp/policy_agent.py`
  - 修复 mixed-goal 低 critic fallback 中对 NumPy 数组调用 `torch.sign` 的 500 错误，
    改为 `np.sign`。

## 7. Pose-only 功能等价 smoke

正式 10-episode sweep 在 pose-only 优化前完成；当时 MemNav 先算出同样的 aux，再额外
运行一个未使用的 diffusion decoder。优化后对 `s8pcmisQ38h/episode_0000` 重跑：

| 路径 | SR/SPL | B 路径 |
|---|---:|---:|
| 完整 MemNav plan，再丢弃其 waypoint | 1 / 1.00 | 2.694 m |
| 最终 `/posegoal_step` | 1 / 1.00 | 2.655 m |

两次运行使用相同 episode 与 seed；diffusion 闭环路径存在厘米级数值波动，但成功率和
SPL 不变。服务日志确认决策请求实际进入 `/posegoal_step`，而非
`/imagegoal_step`。最终版本也把 pose-only 不使用的 depth/current-state feature 计算
移到了旧 decoder 路径；旧 `/imagegoal_step` 兼容性 smoke 仍正常返回 `(24, 3)`
trajectory。

### 7.1 候选边界 fail-safe smoke

`rqfALeAoiTq/episode_0001` 的实际最近 anchor 是 frame 44；返程第一次决策时
合法上界只有 43（gap 31，小于 `exclude_recent=32`）。实际控制记录为：

| 返程 step | controller | 结果 |
|---:|---|---|
| 0 | `navdp_image_fallback` | 明确记录 `forced anchor 44 outside eligible range [39, 43]` |
| 8 及以后 | `navdp_image_point_mix` | anchor 合法后自动切回 metric-pose 控制 |

该样本最终 Goal B 成功，SPL 0.644。这个测试验证了边界条件不会输出错误 pose、不会冻结
NavDP 的时序状态，也不会因为一次候选不足而中断整段导航。

## 8. 依赖与提交前检查

以后提交任务前至少执行：

```bash
test -x /home/asus/miniconda3/envs/memnav/bin/python
test -x /home/asus/miniconda3/envs/habitat/bin/python
test -f .diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
test -f /home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt
test -d .diagnostics/unseen_scene_eval_20260803/episodes
test -d .diagnostics/unseen_scene_eval_20260803/assets

/home/asus/miniconda3/envs/memnav/bin/python -m py_compile \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py \
  NavDP/baselines/navdp/policy_agent.py
/home/asus/miniconda3/envs/habitat/bin/python -m py_compile \
  MemNavData/eval_2leg_habitat.py
git diff --check
```

启动 MemNav 时，本实验必须显式使用 `--exclude_recent 32`；不能依赖服务器仍为数据集
默认的 83。两台服务必须分别使用端口，例如 MemNav 18899、NavDP 18898。

## 9. 当前限制与下一步

1. `hybrid_pose` 当前知道 A→B 的 phase boundary；真实部署仍需学习/验证 router。
2. 评测只有 5 个 unseen scenes、10 个 two-leg episodes，需要扩大场景与 seed。
3. `exclude_recent=32` 解决了本批在线帧率不匹配，但应升级为 pose/flow-based spatial
   exclusion 与置信度约束。
4. 官方 NavDP point-goal 预处理会把负 forward 坐标裁到 0；mixed decoder 依靠 image
   token 完成回头。后续可显式加入 forward-only U-turn 状态机，降低 e9/ep0 等效率长尾。
5. 最合理的训练对象已从“整套共享 diffusion decoder”缩小为：
   - retrieval confidence / phase router；
   - metric-pose reliability；
   - pose-progress trajectory reranker。
   官方 NavDP backbone 和 decoder 应冻结，以结构性避免 Novel forgetting。
