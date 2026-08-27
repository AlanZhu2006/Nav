# Fresh HM3D actual-online Full-Mono mixed-role confirmation

冻结时间：2026-08-20 22:00（Asia/Shanghai），早于本协议中任何新 Goal-A、native、raw
或 CEC outcome。机器可读双生协议：
`MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json`。

## 1. 研究问题

在此前从未读取导航 outcome 的 HM3D scenes 上，完整的 causal RGB-only 系统能否同时
复现三件事：

1. CEC 相对 mono native 的 Revisit utility；
2. CEC 在 unsupported Novel 上的 abstention / exact fallback；
3. CEC 相对 always-on raw memory 的 utility--interference 权衡。

本实验不是新训练、阈值搜索或 controller 调参。CEC、NavDP、LingBot、DINO、
LightGlue/PnP、2.5 m residual、first-40 scale、600-step budget、1 m success radius 和
三个 arm 均保持不变。

## 2. Fresh scene identity

HM3D val archive 固定包含 100 scenes，member-list SHA-256：
`087eb023ae868c7a67cde44badf88d2bdd0f8ed9f016846dd169a981a8adbc61`。

排除集合是：

- 2026-08-16 consumed audit 中的 36 scenes；
- 先前 heldout-val10 protocol 中的全部 10 scenes，包括当时未成功生成 episode 的
  `q3hn1WQ12rz`。

排除 46 scenes 后，按 archive 五位索引升序得到 54-scene fresh reserve。完整 identity、
archive index 和顺序已经逐项写入 JSON 协议，运行时不得重排、替换或按 scene
constructibility/SR 选择身份。

## 3. Source generation 与 actual mono Goal-A

54 scenes 各尝试生成 4 条冻结 two-leg source episode，共 216 个 target sources。生成参数
与旧 HM3D 构造保持一致：3--9 m Goal-A、fixed attempt multiplier 60、seed
`2026082100 + scene_rank`。如果某个 scene 在固定尝试预算内无法生成完整 4 条，记为
pre-navigation constructibility attrition；不重试、不放宽条件、不以别的 scene 替换。

对每条 constructible source 只运行一次 actual mono Goal-A：

- `hybrid_route=native_sidecar`；
- `navdp_depth_source=monocular_sidecar`；
- frame 0--39 为 zero depth，此后只用冻结 first-40 RGB scale receipt；
- simulator metric-depth read 必须为 0；
- seed 为 `2026082200 + 100 * scene_rank + episode_rank`。

Goal-A 失败仍保留在 intention-to-construct 总账。只有成功、native-control audit 通过且
history 长度满足 frame-39/end-margin 合约的 trace 才能产生查询 history。

## 4. Query 构造

构造阶段只能读取 source generation、actual mono Goal-A trace 和 simulator geometry，不能
读取三个 query arm 的任何 outcome。每个 eligible history 同时构造：

- Natural Novel：对 online-A eligible frames 的 max co-visibility `<0.10`；
- Revisit：max co-visibility `[0.55,0.90]`；
- 两者 query geodesic 均为 2--9 m；
- 每个 history 一对 Novel/Revisit；
- selected prefix 中所有成功构造的 history 全部进入评测。

## 5. 冻结扩样规则

为了避免上一轮 8 histories / 7 scenes 的 underpowered 分母，同时不允许看到 query SR 后
扩样，54 scenes 在同一 immutable DAG 中完成 source generation、Goal-A 和 construction。
Finalizer 只按固定前缀依次检查构造分母：

```text
30 -> 36 -> 42 -> 48 -> 54 scenes
```

选择第一个同时达到 `>=24 histories` 和 `>=15 scene clusters` 的前缀；若 54 scenes 仍
未达到，则使用全部 54 并诚实报告 underpowered。该决策只能读取所有 arm 共享的
pre-query constructibility，禁止读取 mono native/raw/CEC query outcome。后续 reserve
scene 的 Goal-A outcome 即使已被计算，也不能让 finalizer 跳过固定前缀或挑选个别 scene。

## 6. 三个同机配对 arm

每个 history 在同一 array element、同一 GPU、同一持久 server pair 中运行：

1. `mono_native`：不使用 memory；
2. `mono_raw_fixed`：DINO raw top-1，always-on 2.5 m fixed-radius bearing；
3. `mono_cec`：DINO top-8 + frozen geometry/PnP certificate；accept 输出相同 2.5 m
   bearing，reject/error exact fallback 到 `mono_native`。

每个 arm 独立 reset，重放完全相同的 actual mono Goal-A RGB history。Novel/Revisit 只用于
离线分层分析，绝不传给 policy。arm 顺序按 sealed history index 轮换。

## 7. 预注册统计口径

Primary：

1. Revisit 上 `mono_cec - mono_native` 的 paired gain/loss、risk difference、exact
   McNemar 和 scene-cluster bootstrap CI；
2. role-balanced total 上 `mono_cec - mono_raw_fixed` 的同类统计；
3. Novel certificate takeover、fully rejected exact-native 和 runtime failure 审计。

Secondary：CEC-vs-raw 的 Novel/Revisit 分层结果、raw Revisit 的 -10 pp non-inferiority
描述性检查，以及按初始最短路方向 front/lateral/back 的预注册分层。不会基于这些结果
重新选方法或阈值。

## 8. 允许与禁止的论文结论

若 verifier 通过，本实验可以支持 fresh HM3D scene 上的 full-monocular mixed-role
confirmation。它仍不能支持 formal safety、Novel solved、mono-vs-metric equivalence、官方
GOAT score 或 backend/controller agnosticism。

无论结果如何，都必须报告：216 target source 的 generation attrition、actual mono Goal-A
success、最终 prefix、history/scene 分母、三个 arm 的 Novel/Revisit/overall SR、所有 paired
统计、takeover/fallback、metric-depth zero-read receipt 和 independent raw-distance recount。
