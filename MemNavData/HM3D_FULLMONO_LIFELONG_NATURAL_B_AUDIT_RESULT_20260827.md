# HM3D full-mono lifelong direct Natural-B 构造审计结果（2026-08-27）

## 结论

冻结的 54-scene direct Natural-B 构造审计已完整结束，并通过独立 raw-fragment
复算。它得到 `99 candidates / 61 recipient histories / 35 scenes`，超过预注册的
`96 candidates / 15 scenes` reference gate。因此，结果盲地构造
`A -> natural Novel-B -> Revisit-A -> Revisit-B -> Revisit-A` 五段序列，在现有
actual-online full-monocular HM3D source pool 上是可行的。

这不是导航 SR，也不授权直接读取或筛选 B/C/B2/C2 结果。下一阶段必须先把候选、
source-history 依赖、统计单位和五段执行顺序冻结成新的 v4 protocol，再运行一次 factual
mono B。

## 冻结输入与运行记录

- parent actual-online full-mono pool：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6`
- v3 source construction：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/formal_20260826T141733Z_375f0b68`
- audit bundle：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_audit_e2832e17231534e3`
- formal run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/natural_b_audit_formal_20260827T125748Z_e2832e17`
- scene array：`16464404_[0-53%4]`
- seal：`16464407`
- independent verifier：`16464666`

54/54 scene tasks、seal 和 verifier 全部 `COMPLETED / exit 0`。单场景 GPU 时间为
26 秒至 1 分 54 秒，说明本轮主要等待来自并发 GPU 配额，而不是算法运行异常。

## 构造合约

每条已经能从 actual-online A 构造 controlled Revisit-C 的 history，最多确定性提出 4 个
Natural-B：

- A-end 到 B：geodesic 2--9 m；
- B 到 controlled C：geodesic 2--9 m；
- B 对完整 online-A history 的最大共视严格 `<0.10`；
- 同一 source history 内候选平面距离至少 2 m；
- 同场景 navmesh、冻结相机高度；
- B 由 navmesh 直接渲染，不借用另一个 successful online history；
- 不读取 B/C/B2/C2 policy/navigation outcome。

## 最终 sealed 数字

| 项目 | 数量 |
|---|---:|
| scene fragments | 54 |
| materialized actual-online A histories | 130 |
| controlled-Revisit-C constructible histories | 80 |
| 至少有一个 Natural-B 的 recipient histories | 61 |
| Natural-B candidate histories | 99 |
| constructible scene clusters | 35 |
| 无 Natural-B 的 controlled-C histories | 19 |

候选最大 online-A 共视为 `0.0897222`，中位数和最小值均为 `0`，所有候选满足严格
`<0.10` Novel contract。方向 strata 为 front `20`、side `22`、rear `57`。这个分布不是
均衡设计；后续报告必须给分层结果或协变量审计，不能把 99 条描述为方向均匀样本。

reference gate：

- candidates：`99 >= 96`；
- scenes：`35 >= 15`；
- `met=true`；
- `evaluation_authority_conferred=false`。

## 独立复算

独立 verifier 不导入 production constructor/aggregator，逐个检查：

- 54 个 scene fragment 及 SHA-256 sidecar；
- scene index、scene、recipient、candidate identity 唯一性；
- 每条候选的 A-to-B/B-to-C 距离、Novel covis、方向 stratum 与 yaw contract；
- candidate/recipient/scene/status/stratum 总数；
- summary 与 raw recount 一致；
- query/navigation outcomes 均未读取，且没有 evaluation authority。

结果为 `verified=true`，99/61/35 与官方 summary 完全一致。

一个明确限制：scene fragment 没有序列化候选 `_position`，所以 independent verifier 不能
从 JSON 重新计算同一 recipient 内的 2 m pairwise separation；该约束由已哈希封存的
constructor 执行。v4 正式 materialization 应序列化 floor position，并让下一版 verifier
独立重算这一谓词。

## 哈希

- `summary.json`：
  `4edbb3f076360063f3dd267a62d90c09d0cd1425973404e47b41bb4cc04ad60f`
- `independent_natural_b_verification.json`：
  `52b15b5e05f21e5ab3bc460f351bdc2068d2bdf121a1eae08646a2d3dd591ab7`
- `submission.json`：
  `85aa593db105f41829a5e0da58daeca45644fb5b12789b56a6cdd771774c26f7`

## 科学含义与下一门

这项结果解决的是 benchmark constructibility，而不是方法效果。它说明可以在不借用其他
轨迹、不看策略结果的前提下，得到足够多的真正 Novel-B；随后 B 被实际 mono NavDP 走过，
才会在同一 episode 内从 Novel 变成可回访记忆。这个 state transition 是 5-leg 实验真正
有意义的部分。

v4 必须依次过三道门：

1. 冻结并 materialize 99 个 B/C query asset，保留 61 个 source-history 与 35 个 scene
   cluster 依赖；
2. 对每个候选只运行一次 factual mono B，并在不知道后续结果时封存成功且得到 factual-B
   support 的 population；
3. 用同一 factual C prefix 配对比较 `all_prior`、`initial_leg_only` 与 exact-native fallback，
   主终点是 B2，C2 作为 older-memory retention 的五段终点；所有统计以 scene 为主 cluster，
   并附 source-history clustered sensitivity。
