# Unknown-goal Natural-stream Memory Support Shadow：协议

日期：2026-08-11（CST）  
状态：先做单 episode 本机 smoke；通过后才允许扩为 train-scene 数据采集。

## 1. 目的

静态 F2 与 F8 均未超过 hard geometry，且 F8 已按协议停止。下一步不再增加静态 feature，
而是在**原生 NavDP 自然运动**产生的 planning stream 上记录同一 goal-memory hypothesis 的
连续证据。

本阶段只验证采集契约：shadow 是否完全不改变动作，是否能在每个 planning step 保存可供
离线时序建模的 top-8 DINO/RANSAC 连续证据。

## 2. 不干预实现

复用现有 `memory_geometry` 自动 router，但把部署决策阈值设置为不可达：

- `router_min_matches = 1,000,000,000`；
- `router_min_inliers = 1,000,000,000`；
- `router_confirm_plans = 100,000`；
- `router_visual_floor = -1.0`，仅为了完整记录 DINO top-8；
- `router_verify_top_k = 8`。

因此每个 trial 的原始 matches/inliers/ratio 被记录，但 `router_active` 永远为 false；实际动作
始终来自 frozen NavDP ImageGoal。MemNav 只接收同一正常 observation stream，不产生额外
转圈或 viewpoint action。

## 3. Smoke

- scene：`17DRP5sb8fy`；
- episode：`episode_0000`；
- true 3-leg；
- deterministic plan seeds、server trajectory selector、goal-switch FIFO carry；
- 输出只作采集可行性，不作性能读数。

Smoke 必须满足：

1. 所有 A/B/C planning records 的 `router_active=false`；
2. 所有 `revisit_adapter_takeover=false`；
3. 有候选的 planning records 全部验证完候选池，且 pool/trials 均不超过 8；
4. trial 含 candidate identity、DINO score、matches、inliers、inlier ratio；
5. 三条 leg 都有 planning stream；
6. 每条 leg 同时保存 natural `rollout_trace` 与 server-frame-indexed `memory_trace`，每个
   query/candidate frame 都可回溯到非干预轨迹 pose；
7. evaluator 完成且输出 JSON 可独立复算。

## 4. Smoke 后的正式采集边界

若 smoke 通过，正式采集仍只使用 40 train scenes 的原生 3-leg stream。需要另行增加 causal
teacher，按每个 decision-frame 的真实 goal-anchor covisibility/pose usefulness 打标签；不得
用 A/B/C role 直接当标签。训练、window、hysteresis 与 operating point 仍做 nested
scene-OOF。

若 smoke 不通过，先修 logger/不干预契约，不提交 HPC。
