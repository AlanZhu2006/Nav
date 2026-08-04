# MemNav × NavDP 自动几何可靠性路由：审计、修复与正式结果

> **2026-08-04 learned-router follow-up:** 使用 4 个训练/校准场景和 5 个完全
> held-out 场景（22,267 个图像对）尝试把几何 verifier 蒸馏到已有 DINO CLS。
> relation head 和 cosine-only 在严格 scene-disjoint 零错误阈值下都需要 100% 回退
> 几何，因此 learned head 不接入部署。相同 goal-anchor 的第二次几何确认现在使用
> per-episode 结果缓存，判定不变，实测平均延迟由 24.283 ms 降到 0.068 ms。完整
> 方法、负结果和依赖检查见
> `MemNavData/LEARNED_RELIABILITY_ROUTER_20260804.md`。

> **2026-08-04 controller follow-up:** 在完全相同的 10 条 episode 上，使用冻结
> NavDP 完成 Goal A、再由 gatecurr600 原生 diffusion decoder 直接控制 Goal B，得到
> 同样的联合 SR `8/10` 和条件 Goal-B SR `8/9`。因此不能把这里相对纯 NavDP 的提升
> 归因于 point-goal controller；当前最清楚的贡献是长期记忆几何和无需 phase 先验的
> 自动可靠性路由。完整成对结果见
> `MemNavData/GATECURR_DIRECT_CONTROLLER_AB_20260804.md`。

日期：2026-08-04
工作目录：`/home/asus/Research/Nav-axis-uturn`
基线提交：`f4cbcb726f6848e6e41bb342d4985eca5af93f7b`
母目录 `/home/asus/Research/Nav` 未修改。

## 1. 结论

这轮工作解决了 Hybrid controller 已知的阶段先验：系统不再由 evaluator 告诉它
“现在是 Goal A（Novel）还是 Goal B（Revisit）”，而是从在线视觉证据自动决定是否
启用长期 memory。

最终路由为：

```text
当前图 + goal image
        │
        ▼
score-only DINO history retrieval
        │  visual cosine >= 0.88
        ▼
SIFT ratio test + calibrated essential-matrix RANSAC
        │  matches >= 20, inliers >= 12, ratio >= 0.50
        │  连续两次确认
        ▼
MemNav/LingBot metric pose -> frozen NavDP image + point-goal controller

任一步不满足：fail closed -> frozen NavDP 原生 image-goal
```

在 5 个 scene-disjoint MP3D 场景、每场景 2 个 episode 的相同 Habitat 闭环协议上：

| 指标 | 最终 geometry router | advantage 硬门槛 | 纯官方 NavDP |
|---|---:|---:|---:|
| Goal A SR | 9/10 = 0.900 | 9/10 = 0.900 | 9/10 = 0.900 |
| Goal B SR，给定 A 成功 | **8/9 = 0.889** | 7/9 = 0.778 | 3/9 = 0.333 |
| 两段联合 SR | **8/10 = 0.800** | 7/10 = 0.700 | 3/10 = 0.300 |
| Goal B SPL，给定 A 成功 | **0.623** | 0.420 | 0.0828 |
| Goal B SPL，A 失败记 0 | **0.561** | 0.378 | 0.0746 |
| Goal B 平均路径，给定 A 成功 | **4.53 m** | 6.34 m | 12.23 m |
| Goal B 平均控制步数 | **139.1** | 183.8 | 339.2 |

纯 NavDP 数值来自之前同一批 episode 的冻结官方 baseline；它不依赖 MemNav
checkpoint，因此不受下面的路径 bug 影响。逐 episode 原始数值保存在
`MemNavData/GEOMETRIC_ROUTER_RESULTS_20260804.csv`。

## 2. 先修复了一个会使历史结论失效的 checkpoint bug

### 2.1 错误链

旧启动命令给 MemNav server 传入相对路径：

```text
.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
```

但 `memnav_server.py` 在构建 agent 前会切换到
`NavDP/baselines/memnav`。相对路径随之失效，而旧
`MemNavPolicy.from_pretrained` 的行为是：路径不存在就静默跳过加载，返回随机初始化的
retrieval、aux 和 action heads。

直接证据是同一 checkpoint、同一输入在服务重启后 trained retrieval anchor 从 176
变成 135，projection score 也从 0.0846 变成 -0.0511；raw DINO score 则逐位相同。

### 2.2 修复

现在：

1. server 在 `chdir` 前把 checkpoint、InternNav root 和 buffer root 绝对化；
2. checkpoint 或 InternNav root 不存在时立即报错；
3. `MemNavPolicy.from_pretrained` 对非空但不存在的 checkpoint 直接报错；
4. 加载后若任何 trainable parameter 缺失，直接报错；
5. 正式运行日志明确出现：

```text
[memnav] loaded .../gatecurr600.memnav.ckpt: missing=2612 unexpected=0
```

缺失的 2612 项属于单独加载的冻结 LingBot 部分；trainable missing 已由新检查确认是 0。

因此旧报告 `HYBRID_POSE_NAVDP_EXPERIMENT_20260804.md` 中 Hybrid 的 9/9、SPL
0.830 不能作为 `gatecurr600` 成绩；该报告顶部已加入失效说明。它仍能说明组合架构
在某组权重下可工作，但不能用于 checkpoint 对比。

## 3. 为什么不能只用 retrieval/gate score

### 3.1 模型 gate 不能承担部署判定

正确加载 checkpoint 后，Novel 与 Revisit 的 predicted gate 都约为 `1.49e-8`。它在
gate curriculum 中主要承担 decoder conditioning，不是一个经过部署校准的 loop
closure detector。

### 3.2 DINO advantage 也不是充分条件

最初使用：

```text
best_history_cosine - current_goal_cosine >= 0.04
```

它在短校准集上有效，但把完整轨迹 memory 装满后，9 个真正 Novel 的跨 episode
目标中有 5 个会连续通过 DINO 预筛。原因是相似房间、走廊和材质具有很高的语义
相似度，但并非同一地点。

同一批 9 个负例经过几何验证后：

- DINO 预筛通过：5/9；
- 几何任意一次通过：0/9；
- 几何连续两次通过：0/9；
- 最强负例只有 17 matches / 8 inliers / 0.471 ratio。

这说明 DINO 适合做高召回候选生成，不适合单独决定 loop closure。

## 4. 几何 router 的关键设计

### 4.1 Goal-session candidate ceiling

每个 goal 第一次查询时记录 `goal_start_frame`，以后候选上限固定为
`goal_start_frame - 1`。这样边界附近的旧帧会在满足 `exclude_recent=32` 后变得可用，
但追逐当前 goal 时刚采集的返回路径永远不能成为自己的 retrieval 候选，消除了
self-match。

### 4.2 两阶段计算

第一阶段 `/retrieval_probe_step` 只追加当前帧并计算 retrieval score，不进行昂贵的
LingBot goal-pose warm append。只有几何确认并 latch 后，才通过
`/posegoal_query` 为已经追加的同一帧恢复 metric pose，不会 double append。

### 4.3 几何确认

`/retrieval_verify` 在 CPU 上执行：

1. SIFT 特征；
2. Lowe ratio 0.75；
3. 使用 episode 相机内参估计 essential matrix；
4. RANSAC + `recoverPose` 统计几何内点；
5. 阈值为 20 matches、12 inliers、0.50 inlier ratio；
6. 连续两个 planning step 通过后 latch；latch 前保持官方 NavDP image-goal。

最终 9 个可执行 Revisit 的首次确认统计为：

| 统计 | matches | inliers | ratio |
|---|---:|---:|---:|
| 最小 | 22 | 13 | 0.516 |
| 中位数 | 52 | 32 | 0.615 |
| 最大 | 171 | 160 | 0.936 |

9 个 Goal A 的自动路由激活次数为 0；9 个可执行 Goal B 都在 step 8 完成第二次确认并
启用 memory。

## 5. 为什么删除 advantage 硬门槛

`memory_advantage` 保留为 ablation。它要求历史图必须比当前图高 0.04，导致 5/9
个真实 Revisit 延迟 24--72 帧才开始确认。

最清楚的因果样本是 `rqfALeAoiTq/episode_0000`：

| 模式 | memory 激活 | 结果 | SPL | 最终距离 |
|---|---:|---:|---:|---:|
| advantage 硬门槛 | step 80 | 失败 | 0.000 | 4.661 m |
| geometry-only decision | step 8 | 成功 | 0.740 | 0.968 m |

它在 step 0 的历史绝对相似度是 0.990，几何为 39/24/0.615，但当前画面本身与目标
也很像（0.974），所以 advantage 只有 0.016。几何已经证明是同一地点，再要求
history 必须明显胜过 current 只会延迟有效信息。

另外：

- `e9/episode_0001` SPL：0.516 -> 0.800；
- `yq/episode_0000` SPL：0.385 -> 0.872；
- `yq/episode_0001` SPL：0.589 -> 1.000；
- `s8/episode_0000` 仍成功，但 SPL：0.458 -> 0.360。

所以提前切换显著提高总体 SR/SPL，但并非每条路径都更短。下一步适合研究几何确认后
image-goal 与 point-goal 的软融合一致性，而不是恢复一个会漏掉真阳性的硬 advantage。

## 6. 剩余失败说明了什么

最终唯一在 Goal A 成功后仍失败的是 `e9zR4mvMWw7/episode_0000`：

- step 0/8 已有 52 matches、32 inliers、0.615 ratio；
- step 8 就启用了 mixed controller，不是 retrieval 延迟；
- 最终距离 1.909 m；
- 后期在两个观测状态间来回振荡，aux pose 也在约
  `[2.3, 0.3]` 与 `[1.7, -0.7]` 间交替。

因此当前主要剩余瓶颈是 metric point-goal 与 NavDP 局部闭环的稳定性/停滞恢复，而
不是“LingBot 转弯完全丢 pose”或 router 找不到回访。可靠性路由已经把错误源进一步
压缩到控制耦合层。

## 7. 模型、协议与复现

### 7.1 Checkpoint

| 模型 | 路径 | SHA256 |
|---|---|---|
| MemNav gatecurr600 | `.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt` | `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7` |
| 官方 NavDP | `/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt` | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` |

### 7.2 协议

- 场景：`e9zR4mvMWw7`、`rqfALeAoiTq`、`s8pcmisQ38h`、
  `yqstnuAEVhm`、`zsNo4HB9uLZ`；
- 每场景 `episode_0000`、`episode_0001`；
- `success_dist=1.0 m`、`max_steps=500`、`exec_horizon=8`；
- Goal A 与 router 拒绝状态：冻结 NavDP image-goal；
- router 接受状态：MemNav/LingBot metric pose + 冻结 NavDP image/point mix；
- MemNav：`W=32`、`S=8`、`exclude_recent=32`、raw DINO anchor、
  `flow_gate=auto`、ground scale max 6.0；
- 不启用 terminal U-turn、visual refine 或 Habitat oracle selector。

客户端核心参数：

```bash
--server_backend hybrid_pose \
--leg1_mode policy \
--hybrid_route memory_geometry \
--router_visual_floor 0.88 \
--router_min_matches 20 \
--router_min_inliers 12 \
--router_min_inlier_ratio 0.50 \
--router_confirm_plans 2 \
--max_steps 500 --exec_horizon 8
```

## 8. 主要代码位置

- `NavDP/baselines/memnav/policy_agent.py`
  - score-only retrieval；
  - goal-session candidate ceiling；
  - retrieval/visual margin 和 anchor 诊断；
  - SIFT/essential geometric verifier；
- `NavDP/baselines/memnav/memnav_server.py`
  - checkpoint 路径绝对化和 fail-fast；
  - `/retrieval_probe_step`、`/retrieval_verify`、`/posegoal_query`；
- `MemNavData/eval_2leg_habitat.py`
  - `memory_advantage` ablation；
  - 最终 `memory_geometry` 自动路由；
  - 相机内参传递和完整 plan 日志；
- `InternNav/internnav/model/basemodel/memnav/memnav_policy.py`
  - checkpoint 不存在/缺少 trainable 权重时 fail-fast。

## 9. 当前边界与下一步

1. 10 个 episode 仍是小规模证据，需要扩展更多 unseen scenes；
2. 当前有 9 个“已有长 memory 后的新目标”负例 probe，但下一步还应做完整 3-leg
   `Novel -> Revisit -> Novel` 闭环；
3. SIFT 对低纹理、强动态和大视角差可能失效；fail-closed 保证不会错误启用 memory，
   但可能产生 false negative；
4. 优先解决 `e9/ep0` 的两状态振荡：可尝试 point-goal temporal filtering、
   image/point directional-consistency weighting 和 stuck-triggered image-goal fallback；
5. 在这些控制逻辑本机验证前，没有必要用 8 小时训练掩盖 inference 层错误。
