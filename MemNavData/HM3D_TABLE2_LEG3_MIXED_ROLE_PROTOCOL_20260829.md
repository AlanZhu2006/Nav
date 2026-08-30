# HM3D Table 2：actual-mono A/B shared-prefix Leg-3 mixed-role protocol

冻结时间：2026-08-29 14:32（Asia/Shanghai）。本协议补会议 Table 2 的唯一实质缺口：
在两段真实 Novel 导航之后，用同一因果前缀比较 Leg-3 Novel 与 Revisit，而不是再跑一条
已经验证过的两段 Revisit。

## 1. Estimand

冻结序列为：

```text
actual mono NavDP: Novel A -> Novel B
                         |
                         +-> new Novel C: native vs CEC
                         +-> new Revisit C: native vs CEC
```

Leg-3 的两种 role 与两条 treatment arm 都从完全相同、逐帧哈希验证的 actual A/B
endpoint 出发。运行时只收到 goal RGB，不读取 `Novel/Revisit` 标签。

主问题有两个：

1. 历史已经积累到 A+B 后，CEC 能否提高 Revisit-C？
2. 对完整 A+B 历史仍不支持的 Novel-C，CEC 能否零误接管并 exact fallback？

## 2. 为什么不直接复用旧 Goal-C

旧 lifelong 链已经运行过其中一批 Revisit-C；再次运行同一 query 不能称为
result-blind formal experiment。本协议只复用旧 C 结果之前就封存的 22 条 actual A/B
成功前缀，重新构造两个从未运行过的 Leg-3 query，并把旧 Goal-B/Goal-C 的 RGB 与
pose+yaw identity 列为 forbidden。旧 C outcome 不参与构造、筛选或调参。

## 3. Prefix contract

- A、B 都由同一 frozen NavDP 和 causal LingBot monocular sidecar 实际执行；
- 不读取 simulator metric depth；
- A/B 的每个 physical RGB observation 原样保留；
- 只在 NavDP 原本做 diffusion decision 的帧恢复其短期 FIFO；
- 每个 query arm 开始前重置服务并重放完全相同的 A/B prefix；
- 该拼接是 observation replay，不是 expert trajectory，也不新增导航动作。

22 条前缀来自已经封存的 v4 population，覆盖 15 scenes；它们是在任何旧
`C/B2/C2` outcome 之前按 `A reached && B reached` 冻结的。

## 4. New query construction

在完整 A+B 历史上独立构造：

- Revisit-C：从 actual A/B observation 周围做受控小位姿扰动；要求 combined-A/B
  co-visibility 在 `[0.55, 0.90]`；
- Novel-C：同层可导航点、B endpoint 到目标测地距离 `2--9 m`，对完整 A+B history
  的最大 co-visibility `<0.10`；
- Novel 优先方向按 population index 在 front/side/rear 间轮转，失败后仅按冻结顺序
  尝试另外两个方向；
- 两个新 query 都必须与旧 B/C 的 JPEG SHA 和 pose+yaw identity 不同；
- construction 不启动 policy，也不读取任何 Leg-3 outcome。

Prospective power gate：至少 16 histories、10 scene clusters，且 front/side/rear
各至少 3 条。若独立 verifier 未通过，controller rollout 不得提交，也不降低阈值。

## 5. Runtime arms

每条 history 运行两臂：

- `mono_native`：相同 frozen NavDP，不允许 memory takeover；
- `mono_cec`：相同 NavDP + frozen CEC，certificate 通过时只注入 unit bearing 与固定
  `2.5 m` residual，拒绝时 exact native fallback。

共同设置：600 steps、horizon 8、1 m success radius、deterministic diffusion seeds、
无 terminal U-turn、无 visual refine、无 role label。

## 6. Reporting discipline

Table 2 应将两类量分开：

- Leg 1/2：报告来源 population 的 factual waterfall；
- Leg 3：报告 `C | A,B` 的 Novel/Revisit paired SR、gain/loss、McNemar、scene-cluster
  bootstrap，以及 CEC accept/false-takeover/false-reject。

不能把条件于 A/B 成功后选出的 Leg-3 分母冒充无条件三段 joint SR。现有 B2
`all_prior vs initial_leg_only` 仍是“后加入 B memory 是否有边际价值”的补充机制实验，
不替代本协议的 mixed-role Table 2。
