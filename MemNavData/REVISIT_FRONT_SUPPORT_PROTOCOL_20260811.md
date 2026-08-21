# Revisit `navdp_front_support_v1` 冻结协议

日期：2026-08-11（CST）  
状态：consumed-pool controller-interface 反事实；零训练、零调参；不授权 blind 或论文声明。

## 1. 唯一问题

known-Revisit direct 相对 geometry router 为 `+6/-1`。唯一 loss `pLe/0001` 的前四个
memory PointGoal 全在后半平面，而冻结 NavDP 的 `process_pointgoal` 会把 forward 分量截到
`[0,10]`。本实验只问：

> 不把超出冻结 PointGoal encoder 支持域的后向向量送入 mixed controller，是否能保留 direct
> 的 recall 增益，同时消除 native-success harm？

这不是 bearing 头、actionability classifier、RANSAC 阈值扫描或 executor 调参。

## 2. 冻结决策

对 known Revisit 的每个 replan，先取得与 direct arm 相同的 temporal-DINO/LingBot
camera-relative PointGoal `p=(forward,left)`：

```text
pose missing / invalid       -> native ImageGoal
forward < 0                  -> native ImageGoal for this replan
forward >= 0                 -> existing mixed ImageGoal+PointGoal NavDP
```

- 边界 `0` 直接来自冻结代码的 `clip(x,0,10)`，不是从 7 条 episode 拟合；
- 不使用 endpoint、critic、RANSAC、oracle、GT goal 或 geodesic；
- 不永久锁存。下一 replan 重新计算 PointGoal 并重新判断；
- forward-supported 路径逐字保留 metric PointGoal，不引入固定半径；
- fallback 调用现有 native ImageGoal 路径，FIFO 每个决策点仍只追加一次 observation；
- 两臂沿用相同模型权重、server process、episode seed 和逐计划 diffusion seed。

## 3. 配对臂

| arm | Goal-A | Goal-B |
|---|---|---|
| `geometry_trace_source` | 冻结 native NavDP，并写出共享 trace | 仅为运行器复用，B 不进入本协议的正式比较 |
| `known_revisit_direct` | 逐像素重放同一 Goal-A trace | `legacy_metric`：有效 pose 即 mixed 接管 |
| `front_support_residual` | 逐像素重放同一 Goal-A trace | `navdp_front_support_v1`：后向逐计划回退 native |

正式运行必须在同一台机器、同一对长期存活的 MemNav/NavDP server process 中顺序完成 trace
源和两条正式比较臂；
禁止拿旧运行的 26/29 与新运行直接配对。

## 4. 分阶段执行

### T0：传输检查

只跑 `pLe4wQe7qrG` 的两个 episode，两臂同进程。必须满足：

- Goal-A trace、seed、geo-A、A outcome 完全相同；
- support arm 至少实际记录一次 `pointgoal_behind_navdp_support`；
- fallback plan 的 controller 为 native ImageGoal；
- 无 HTTP fallback、seed mismatch 或 FIFO contract error。

T0 只检查实现和 `pLe` 机制，不作统计结论，也不据结果改规则。

### T1：完整 consumed pool

T0 通过后，用相同冻结代码直接跑既定 20 scenes / 40 episodes。主分母为共享 A-success 的
conditional Goal-B，同时报告 joint A∧B。

## 5. 预注册统计与门

- paired `direct -> support`：both/direct-only/support-only/neither；
- exact two-sided McNemar；
- scene-cluster bootstrap 95% CI；
- conditional B、joint、SPL、final distance；
- fallback episode/plan 数、首次进入支持域的 step；
- native-success harm 单独列出。

架构继续门：

```text
support_only > direct_only  且  native-success harm = 0
```

若只得到持平，它仍可作为更正确的接口防护，但不形成方法增益；若产生新 loss，立即停止，不再
叠加 rotate-first。只有 `pLe` 仍失败且无新 harm 时，才另立协议测试显式转向执行器。

## 6. 严禁事项

- 不根据 T0/T1 扫描角度、endpoint 或 critic 阈值；
- 不把 7 条 post-hoc discordant episode 当统计分母；
- 不跨机器、跨 server process 拼接配对结果；
- 不在 development/blind 上试规则；
- 不把“后向 fallback”表述成学会了方向或提升了 NavDP 本体能力。
