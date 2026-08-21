# MDTEC Raw-Depth Gate D Protocol — 2026-08-19

状态：**前瞻冻结，尚未读取任何 Gate D 闭环结果。** 继承
`MONOCULAR_DUAL_TIMESCALE_EXPERT_PROTOCOL_20260818.md`；40-scene Gate C 已由独立
verifier 授权 `raw_lingbot_depth`，并淘汰 `latent_adapter`。本协议只规定获准后的已消费
闭环开发门，不改变既有 CEC 结果。

## 1. 要回答的问题

在保持 NavDP checkpoint、goal encoder、diffusion decoder、critic、随机种子和执行器不变
时，用因果单目 RGB 流中的 frozen LingBot raw depth，能否替代 simulator metric depth，且
不会因 40 帧 RGB-only bootstrap 使 Novel ImageGoal 闭环明显退化？

这是 controller 可消费性问题，不是 CEC/Revisit 效果测试。Gate D 通过前不得把系统称为
完整单目导航器，也不得运行 CEC-on-monocular 的正式比较。

## 2. 冻结架构

```text
causal RGB stream
  -> one frozen LingBot-Map stream
       frame 0..39: explicit zero depth
       after receiving frames 0..39:
         replay exactly this prefix once
         freeze first40_scale_receipt
       frame index >=40:
         frozen LingBot relative depth * frozen scale_hat
         invalid scale -> explicit zero depth
  -> unchanged NavDP process_depth + RGB-D encoder
  -> unchanged NavDP image-goal encoder + diffusion decoder + critic
```

约束：

- LingBot sidecar 与 CEC 共用一份 live map；不得启动第二个 LingBot stream；
- NavDP 是唯一 action-generating policy；
- `first40_scale_receipt` 必须来自每个 arm 自己实际走出的 RGB 0..39 与 camera poses；
- `whole_episode_ground_cache_consumed=false`；
- sidecar depth 必须与当前 JPEG SHA-256 绑定；不匹配时 fail closed；
- `monocular_sidecar` 模式中上传的 simulator depth 必须被 NavDP server 忽略，并报告
  `metric_depth_sensor_consumed=false`；
- frame 39 即使已完成冻结仍输出 bootstrap zero；frame index 40 才允许激活 raw depth；
- 不允许 pooled constant、teacher prefix、expert pose、future frame 或 oracle scale 回退。

## 3. 已完成、但不属于闭环结果的前缀接口门

本地真实 41 帧 smoke 已在读取任何 Gate D SR 前完成：

- frame 0、7、39 深度逐像素全零；
- 第 40 个 observation 生成一次不可变 first-40 receipt；
- frame index 40 首次输出 raw LingBot metric depth；
- 重复 query 的 depth/scale SHA 完全一致且不增加 stream count；
- NavDP wire smoke 中分别上传 0.2 m 与 4.0 m metric-depth 图片，同 seed 的 trajectory 与
  critic 逐值相同，证明上传 sensor depth 未被消费。

这些只授权正式闭环，不是 SR 证据。

## 4. 冻结开发人口与三臂

人口继承已经消费的 MP3D Novel-A 机制池：

- `expanded_navdp_router_eval_20260805.json`；
- SHA-256 `ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61`；
- 20 scenes、每 scene 2 episodes、共 40 episodes；
- Goal-A 输入继承 `novel_a_bearing_inputs_20260808.json`，SHA-256
  `401d43723a37465fa00778fd21b27eecbe46cf114abb074a3582b524451ce901`；
- checkpoint SHA-256
  `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`。

三臂在同一 GPU、同一已加载 MemNav sidecar 和同一已加载 NavDP 进程中逐 episode 配对：

1. `metric_teacher`：NavDP 消费请求中的 Habitat metric depth；
2. `zero_depth`：NavDP 忽略请求 depth，始终消费全零 depth；
3. `raw_first40`：0..39 全零，frame>=40 消费 SHA-bound LingBot raw depth。

arm 顺序按 `(scene_index + episode_index) mod 3` 旋转。每臂 reset 两个 server，使用相同
episode seed 和逐 plan deterministic diffusion seed。成功半径 1.0 m，max steps 500，
execution horizon 8，server trajectory selector；只跑 Novel Goal-A。

## 5. 必须报告的闭环与部署量

每臂、每 episode：

- SR、SPL、path length、steps、termination reason；
- `metric_depth_sensor_consumed_any`；
- 是否观察到 frame index 40；
- first40 scale valid/clamped/scale_hat/freeze latency；
- 到首次 raw activation 的实际 path length；
- bootstrap 与 active plan 数；
- current-image/depth/scale receipt SHA 一致性；
- server PID/start receipt，证明三臂同进程。

汇总：

- raw vs metric、zero vs metric、raw vs zero 的 paired gain/loss；
- exact McNemar；
- scene-cluster bootstrap 95% CI；
- frame-40 survival、scale-valid/clamped 比例；
- 按 `reached_frame40` 与 `scale_valid` 分层的 SR，仅作诊断，不改主分母。

## 6. 冻结决策

Gate D 有两层口径：

### 6.1 继续 CEC-on-monocular 的工程授权

必须同时满足：

1. 40/40 raw arm 均 `metric_depth_sensor_consumed_any=false`；
2. 无 JPEG/depth receipt mismatch、无第二次 scale freeze、无 future/pool fallback；
3. raw SR 风险差点估计不低于 metric teacher `-10 pp`；
4. paired `(losses - gains) <= 4`；
5. raw 在至少一个功能闭环量上严格优于 zero：SR 更高，或 SR 相同且 SPL 更高；
6. 所有到达 frame 40 且 scale-valid 的 episode 都实际进入
   `raw_lingbot_metric_depth`，不能悄悄继续 zero。

通过才允许在相同单目 controller 上测试 CEC Revisit bearing/abstain。失败则停止全系统
单目升级，CEC 保持“monocular episodic sidecar + RGB-D controller”的论文口径。

### 6.2 论文中的统计 non-inferiority

只有 raw-minus-metric 的 scene-cluster 95% CI 下界高于 `-10 pp`，才可写“在该已消费
MP3D 开发池上达到预设 10 pp non-inferiority margin”。工程授权不自动等于统计确认，且
本人口已消费，不能称 held-out confirmation。

## 7. 明确不做

- 不重新训练 Adapter；
- 不调 scale、critic、stop、success 或 bootstrap 阈值；
- 不把 frame<40 改成估计 depth；
- 不先看少数 episode 再筛人口；
- 不在本 Gate 中启用 CEC、bearing、oracle、X-NavDP 或 graph rescue；
- 不把 Gate C latent 指标写成 SR，也不把本已消费 Gate D 写成外部泛化。

