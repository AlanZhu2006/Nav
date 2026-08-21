# Unknown-goal Natural-stream Shadow：Trace-complete Smoke 结果

日期：2026-08-11（CST）  
状态：**采集契约通过；不是模型、SR 或方法结果。**

对应协议：`MemNavData/UNKNOWN_GOAL_NATURAL_STREAM_SHADOW_PROTOCOL_20260811.md`  
运行目录：`.diagnostics/unknown_goal_natural_stream_smoke_v2_20260811`

## 0. 结论

在 1 条 3-leg episode 上，系统让原生 NavDP 正常运动，同时以 shadow 方式记录 unknown-goal
memory support 证据。router 和 adapter 均未接管动作；所有 planning state、当前观测和候选
anchor 均可回连到 evaluator 保存的 rollout/memory trace 与 buffer 图像。因此，下一步可以在
train scenes 采集自然时序证据，而不需要原地转圈，也不会把 learned router 的动作干预混入
训练数据。

这轮只验证数据链和因果契约。单条 episode 的 `A=1, B=1, C=0` 不作任何 SR 或方法声明。

## 1. 审计结果

| 项目 | 结果 |
|---|---:|
| episodes | 1 |
| planning states | 99 |
| leg A / B / C plans | 19 / 17 / 63 |
| plans with retrieval candidates | 80 |
| full top-K verification plans | 80 |
| geometry trials | 604 |
| max candidate pool | 8 |
| malformed trials | 0 |
| missing trace frames | 0 |
| missing buffer images | 0 |
| router active plans | 0 |
| adapter takeover plans | 0 |
| contract pass | **true** |

哈希：

- `report.json`：`4610bf9dd5b9e175913fd8eabb53818d75dcf530e47611ae98a52f67c22b56cb`
- `episode_0000_plans.json`：`d46a534ad41536960f3cf2bfda071b8f733c87de5b0491d7917752165e13b2e9`

## 2. 相比第一次 smoke 的关键修正

第一次 smoke 已证明能旁路记录 DINO/RANSAC 候选证据，但 plan 文件没有保存 evaluator 内部
已经存在的 `rollout_trace` 和 `memory_trace`。这会使当前 frame、候选 anchor、机器人位姿
无法做严格的因果对齐，因此不足以授权 HPC 扩采。

本轮在 `eval_3leg_habitat.py` 中为每个 leg 保存：

- `rollout_traces`：自然执行轨迹及逐步 pose；
- `memory_traces`：写入 episodic memory 的 frame 与 pose；
- shadow 证据里的当前 `step/frame_idx` 与每个 candidate `anchor_frame_idx`。

汇总器现在 fail-closed：任一 trace、frame 或 buffer image 对不上就失败；任一 router active
或 adapter takeover 也失败。4 个契约单元测试全部通过。

## 3. 它对 learned router 的意义

已有 single-state F2/F8 的失败主要是 positive support 上过度 abstain，而非选错 anchor。
继续扩大静态 top-K、调 threshold 或加 MLP 不会改变可观测性。自然 stream 提供的是新的
证据维度：同一 anchor 是否在机器人正常位移后的多个视角中持续出现、几何假设是否相容、
pose/bearing 是否随运动一致。

下一正式步骤不是闭环长评测，而是只在 train scenes 扩采 shadow stream，然后做与 F2/F8/H
完全相同的 nested scene-OOF 比较。只有时序模型在三个 seeds 上同时提高 correct-anchor
coverage，且不增加 strict FP/wrong anchor，才授权 action expert 或闭环评测。
