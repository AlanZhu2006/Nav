# Novel-A oracle-bearing mechanism gate（冻结协议）

冻结时间：2026-08-08（任何正式 20-scene outcome 产生之前）

机器可审计协议：`novel_a_bearing_gate_protocol_20260808.json`，SHA256 `4006f9a62b8376c6a55a6394f0bce026739d2c7c968712b542b15b7f1158b6c8`。

## 1. 问题与证据等级

本实验只回答两个分层问题：

1. **机制上限**：在 Novel A 闭环中周期性提供正确的局部 geodesic bearing，并以零成本理想转向执行，能否提高 `SR_A`？
2. **执行兑现**：同一 bearing 能否通过冻结 NavDP 的 mixed image+point 接口，在相同物理预算内兑现为 `SR_A` 增益？

第二问只有在第一问为正时才解释。`ideal_periodic_yaw` 是 privileged upper bound，不能称为可部署方法；`oracle_token_periodic` 仍使用 Habitat oracle bearing，也不能称为完整部署系统，它只检验现有 token actuator。

冻结的 20-scene/40-episode 集已经用于多次项目诊断，因而是**内部机制门**，不是 blind confirmation。若结果值得推进，最终主张必须在未用于开发的 526-pool 子集上按本协议预先冻结后确认。

## 2. 固定样本与共同设置

- manifest：`expanded_navdp_router_eval_20260805.json`
- manifest SHA256：`ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61`
- Goal-A input overlay：`novel_a_bearing_inputs_20260808.json`，SHA256 `401d43723a37465fa00778fd21b27eecbe46cf114abb074a3582b524451ce901`（补冻结 parent manifest 未逐图记录的 40 张 expert-arrival Goal-A image）
- 样本：manifest 固定的 20 scenes × 每场景 2 episodes = 40 episodes；不按既有成功/失败筛选
- 只跑 start → Novel A；到达 A 后立即停止，不运行 Revisit B
- frozen NavDP checkpoint SHA256：`3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`
- Goal-A image：generator expert arrival frame `rgb/{switch_idx-1}.jpg`
- success：当前位置到 A 的平面欧氏距离 `< 1.0 m`
- 每臂 budget：最多 500 physical simulator steps；`exec_horizon=8`
- controller：现有 pure-pursuit 参数不变（`v_max=0.0376 m/frame`、`r_min=0.40 m`、`max_turn=4.5°/frame`、`lookahead=0.7 m`）
- stuck rule：150-step displacement `<0.10 m` 时终止
- trajectory selector：server-selected；禁止 Habitat candidate oracle
- 每一 scene 的三臂在**同一 NavDP server 进程、同一 evaluator 进程**内运行；每臂前 `navigator_reset`
- episode seed：`20260803 + episode_index_within_scene`
- native plan seed：`diffusion_plan_seed(episode_seed, leg=0, native_plan_index)`；三臂同 plan index 使用相同 seed
- arm order 按 `(scene_index + episode_index) mod 3` 对三臂循环移位，减少固定顺序/热状态混杂；输出记录实际顺序
- 任何 seed echo、FIFO read-only 声明、episode/asset hash 或完整覆盖审计失败，整批 fail closed，不报告 SR

## 3. 三臂干预

### `native`

完全冻结的 NavDP ImageGoal。每个决策点 append 当前 RGB 一次，执行 server-selected trajectory 最多 8 physical steps。

### `ideal_periodic_yaw`

每个正常决策点、调用 ImageGoal **之前**：

1. 用 Habitat shortest path 的首个距当前位置至少 `0.30 m` 的 waypoint 计算 desired yaw；
2. 若 wrapped residual `|desired_yaw-current_yaw| > 20°`，将 evaluator yaw 零成本设置为 desired yaw；位置、physical step、path length 均不变；
3. 在转后的真实 rendered observation 上进行一次与 native 同 seed 的 ImageGoal plan，并正常执行。

该臂记录每次 yaw teleport、累计绝对角度和干预次数。其 SPL 仅作 privileged diagnostic，不与可部署 SPL 等同。

### `oracle_token_periodic`

每个正常决策点：

1. 先进行与 `native` 相同的一次 ImageGoal call；它是本决策点唯一的 FIFO append，并缓存 native trajectory 作为 abstain fallback；
2. 计算相同 oracle geodesic bearing。若 residual `≤20°` 或 token 已 fail-closed，直接执行缓存的 native trajectory，不进行 resample；
3. 否则把请求角裁到 `sign(residual) × min(|residual|,100°)`，半径固定 `2.0 m`；
4. 对同一 observation/FIFO 调用 read-only `mixgoal_resample`，使用与步骤 1 相同的 diffusion seed；必须返回 `memory_mutated=false`、FIFO content fingerprint 前后逐项一致且 seed echo 一致；
5. 执行 mixed decoder 的 server-selected trajectory 最多 8 physical steps；下一决策点重新计算 bearing。

最多允许 10 个连续 token plans。若 10 plans 后 residual 仍大于 20°，该 episode 后续永久 abstain 到每个决策点已经缓存的 native trajectory，并记录 `max_burst_exhausted`。所有 token movement 都计入同一 500-step/path budget；不存在额外预算。

NavDP critic 只在每个候选集合内部维持 server 原有选择，并作为 shadow diagnostic 记录。由于源码中的 critic 将 goal embeddings 置零，本实验禁止把不同 bearing 请求的 critic 值当作 goal-direction router。

## 4. 结局、统计与解释顺序

### Primary：机制上限

`ideal_periodic_yaw` vs `native` 的 episode-level paired `SR_A`：

- 完整报告 40 对的 gain/loss、净 risk difference；
- exact two-sided McNemar（discordant-pair sign test）；
- 各臂 Wilson 95% CI；
- 以 scene 为 cluster 的 deterministic percentile bootstrap 95% CI（20 scenes，100,000 resamples）。

### Conditional secondary：执行兑现

仅在 primary 方向为正时，按相同统计报告 `oracle_token_periodic` vs `native`。同时报告 token activation、abstain、burst exhaustion、累计 token path、残差变化及 native-success harm。

### 固定决策区间

- **Go（进入未见 526 pool 扩样，不等于论文确认）**：ideal gain ≥4、loss ≤1，且净增益 ≥3/40。
- **No-Go（不把方向升为项目主线）**：ideal gain ≤2，或 ideal 净增益 ≤0。
- **Ambiguous**：其余情况；不建 frontier ranker，先在不重叠样本复验。
- **论文级正向结论**：不能由本内部集单独给出；至少要求冻结后的未见池复验方向一致，并报告预先指定的 paired inference。

Token 臂不会替代 primary gate。理想臂为正而 token 臂不为正，结论是“方向上限存在但当前 actuator 未兑现”，不是“方向方法成功”。

## 5. 禁止事项

- 不只重跑既有 9 条 native A failures；必须全 40 对以计入 harm。
- 不依据 smoke 或部分 scene outcome 修改 20-scene 参数、门限、arm 定义或 decision rule。
- 不使用 development/blind set 做任何选择。
- 不把旧机器结果与新结果拼成 paired table；所有正式配对来自同一 scene task 的同机同进程三臂。
- 不把 critic 的跨-bearing 分数解释为目标相关证据。
- 不在看完部分正式结果后提前停止或扩展 arm。
