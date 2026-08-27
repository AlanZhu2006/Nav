# REVISIT 可靠基线：Verified Bearing Residual Adapter v1

日期：2026-08-10（CST）

> 2026-08-10 更新：本文件冻结的是可靠 controller/interface baseline。SIFT/RANSAC
> source 不够优雅，不再作为最终学习方法；RANSAC-free 双 expert V2 见
> `REVISIT_DUAL_EXPERT_V2_20260810.md`。

## 1. 决策

项目重心回到 **REVISIT**。当前可靠基线不再表述成“MemNav 估计完整目标位姿并替换
NavDP”，而表述成：

> 可靠的 revisit memory 给冻结 ImageGoal policy 提供一个经过几何验证的、低带宽的
> bearing residual；证据不可靠时，逐 planning step 精确回退 native policy。

X-NavDP 继续保留为 controller capability 研究，不进入默认方法路径。

## 2. 冻结架构

```text
current RGB/history + ImageGoal
              |
              +-------------------------------> native ImageGoal NavDP
              |                                      ^
              v                                      | abstain
       memory retrieval                              |
              |
       SIFT/RANSAC geometry gate -- reject ----------+
              |
            accept
              |
     raw relative translation [x,y]       （只保留在 audit）
              |
              v
       unit bearing [x,y] / ||[x,y]||
              |
       frozen local radius 2.5 m
              |
              v
  existing mixed ImageGoal + PointGoal NavDP
```

它只有三个可解释模块：

1. **source**：memory retrieval + geometry 负责“有没有可信方向”；
2. **adapter**：只做归一化、固定半径和 fail-closed abstention；
3. **executor**：始终是现有 mixed NavDP，不切换新的 learned controller。

数学上，geometry 返回相对平移 `t`。仅当 router 已激活且 `t` 有限、非零时：

`b = t / ||t||,  z = 2.5 b`。

controller 接收 `(current RGB, ImageGoal, z)`；否则只接收原生
`(current RGB, ImageGoal)`。原始 `||t||` 不进入 canonical controller。

## 3. 为什么这是当前最优雅、也最诚实的形式

### 3.1 它删除了一个没有证据支撑的自由度

B0 在同一 deterministic R0 上比较 metric PointGoal 与固定 2.5 m：

| Interface | B given A | vs metric |
|---|---:|---:|
| metric geometry + mixed | 20/26 | -- |
| fixed 2.5 m bearing + mixed | 20/26 | `+1/-1, p=1.0` |

固定接口保留 metric 的 `19/20` successes。相对 native 为 `+17/-0,
p=1.5259e-5`。因此 aggregate 方法效果需要的是方向，不需要把不稳定的距离估计暴露给
controller。2.5 m 来自 first-active radius 的 episode-balanced median 2.513 m 后事先
四舍五入；没有 sweep radius。

### 3.2 它不制造第二套导航系统

memory 不输出地图、长航点序列或速度；adapter 不学习；mixed NavDP 仍保留 ImageGoal
上下文和原生避障/局部规划能力。失败时不是 episode 级永久切换，而是下一次 planning
step 可以重新验证并 abstain。

### 3.3 它把真正未解决的问题暴露出来

R0 的 6 条 residual failure 中，4 条 router inactive，只有 2 条是 router-active
controller failure。当前主要空间是 **memory evidence recall / alias rejection**，不是再换
controller，也不是扩大 diffusion candidate K。

## 4. X-NavDP 为什么没有把官方能力转成整体 SR

Complete R2 的同机同前缀结果：

| Controller | B given A | vs mixed |
|---|---:|---:|
| mixed ImageGoal+PointGoal | 20/26 | -- |
| base pure PointGoal | 20/26 | `+1/-1` |
| official X + official MPC | 21/26 | `+2/-1, p=1.0` |

新生成的 consumed-R2 机制审计按 first-active bearing 描述：

| 区域（只用于描述） | N | mixed | base | X | X 负速度比例 |
|---|---:|---:|---:|---:|---:|
| `|bearing| < 60°` | 0 | 0 | 0 | 0 | -- |
| `60° <= |bearing| < 135°` | 4 | 4 | 4 | 3 | 68.0% |
| `|bearing| >= 135°` | 17 | 15 | 15 | 17 | 93.7% |
| router inactive | 5 | 1 | 1 | 1 | -- |

解释是：

- X 的 signed-control/post-training 确实形成了 **deep-rear 专门能力**；它不是没兑现；
- 全局替换 mixed 会丢掉 mixed 的 ImageGoal 条件，并把倒车能力施加到不合适的区域；
- `gxdoqLR6rwA/episode_0001` 首次 bearing 为 `-115.55°`，X 产生 483 个 reverse
  frames 后失败，是净增益从 +2 被抵消为 +1 的直接原因；
- 两个 gain 都在 `|bearing| >= 150°`，但这是看过结果后的描述，不能据此上线阈值。

因此 X 只能进入一个**未来、独立预注册**的 capability-envelope 实验。当前数据不授权
global X、不授权 angle gate、不授权 blind。

## 5. 已执行的工程改动

- `revisit_bearing_adapter.py`
  - 固化 `verified_bearing_v1`、2.5 m 语义常量；
  - malformed / non-finite / zero / inactive 全部 fail closed；
  - 每步输出 raw point、unit bearing、controller point、takeover/reason 审计；
  - 保留 `legacy_metric` 仅用于旧结果复算。
- `eval_2leg_habitat.py`
  - 新增 `--revisit_adapter verified_bearing_v1`；
  - canonical 模式强制 `hybrid_pose + automatic geometry router + navdp_mixed`；
  - 禁止 phase-oracle 和 X controller 混入 canonical adapter；
  - episode/summary 增加 takeover 与 abstain 计数。
- `analyze_revisit_xnavdp_capability.py`
  - 独立复算 R2 的 bearing 区域、三 controller outcome 和 reverse-control 比例；
  - 输出中机器可读地写死“不授权 selector / blind”。
- 新增 8 项 adapter/归因单元测试；连同 5 项既有 X contract 测试共 13 项，全部通过。

另外把新 adapter 重放到 R0 的 26 条 Goal-A-success metric rollouts：361/361 个有效
active decision 均成功转换，最大 fixed-radius 误差 `4.44e-16 m`，最大方向叉积误差
`2.22e-16`。B0 fixed arm 自己记录 322 个 active plans；两臂执行后轨迹和 replan 次数会
分叉，所以 planning-step 数不应强行相等，严格比较单位仍是 26 个 paired episodes。

Canonical 调用的关键参数为：

```bash
--server_backend hybrid_pose \
--hybrid_route memory_geometry \
--revisit_controller navdp_mixed \
--revisit_adapter verified_bearing_v1
```

## 6. 接下来的优先级

### P0：冻结 adapter，不再调 radius/controller

已完成代码化。B0 已经闭环验证等价语义，所以不需要为了“再看一次 20/26”立刻做一次
长 eval。只需在下一次正式 run 前做 1 scene transport smoke，确认新 audit contract。

### P1：只攻 geometry evidence，不动 controller

按 residual failure taxonomy 做 source-side 分叉：

1. `target never observed`：不可恢复，必须 abstain；
2. `positive outside margin/top-8`：检索召回问题；
3. `alias latch`：跨 planning-step 一致性问题；
4. `RANSAC false reject`：verification recall 问题。

下一项实验必须先离线证明它改变了哪一类 failure，再进入闭环。已证伪的“单纯扩大 K”
和跨场景漂移的 learned activation 不重启。

一个值得研究、但尚未授权的方向是把多个**独立通过 geometry 的 bearing**形成 circular
consensus，再以 resultant/conflict 决定 abstain；它利用的是方向一致性，不是事后挑一个
更好 anchor。当前日志没有为每个通过候选保存 pose，先补纯 shadow evidence，不能直接
上线。

### P2：fresh confirmation

方法、radius、gate、controller 全冻结后，才在与 consumed 20-scene pool 不相交的集合做
一次性确认。blind 仍保持关闭。主比较只需要：

1. native ImageGoal；
2. verified-bearing residual adapter。

metric-distance arm、base PointGoal、X 都是已完成 attribution，不进入主确认，避免把 eval
扩成调参矩阵。

## 7. 允许写出的主张

当前允许：

> On the consumed paired benchmark, geometry-verified revisit memory yields a
> significant oracle-free closed-loop gain. Replacing its metric displacement
> with a fixed-radius bearing preserves the aggregate gain, supporting memory
> as a directional residual rather than a replacement navigation stack.

当前不允许：fixed bearing 与 metric 完全等价、X 在 deep-rear 必然更优、角度 gate 可部署、
blind/generalization 已确认、真实单目硬件已经部署。
