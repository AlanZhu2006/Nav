# HM3D Table II Leg-3 mixed-role constructibility audit

审计日期：2026-08-30（Asia/Shanghai）。本审计只读取冻结的 query-construction
receipts，不读取或运行任何 Leg-3 policy outcome，也没有修改 Novel/Revisit 阈值。

## 1. 结论

原 prospective gate 正确停止。22 条 actual-mono `Novel A -> Novel B` 成功前缀中，
只有 8 条、6 个 scenes 能同时构造一条新 unsupported Novel-C 和一条 supported
Revisit-C；front/side/rear 为 `4/0/4`，未达到冻结的 `16 histories / 10 scenes /
每方向至少 3`。因此：

- `formal_policy_evaluation_authorized=false`；
- 没有 Leg-3 controller rollout，也没有新的 SR；
- 不能降低 `<0.10` Novel covis、`[0.55,0.90]` Revisit covis 或方向门后再称为同一
  formal experiment；
- 这个结果是 benchmark constructibility null，不是 CEC 或 NavDP 的性能失败。

## 2. 14 条 attrition 的组成

| attrition | histories |
|---|---:|
| accepted role pair | 8 |
| 完整 A+B history 后找不到新 unsupported Novel-C | 13 |
| 找不到标准支持带 Revisit-C | 1 |

8 条 accepted pair 的 Novel 来源中，6 条来自 deterministic local polar grid，2 条
来自 frozen seeded-uniform fallback；Revisit anchor 来源为 A history 3 条、B history
5 条。可接受前缀长度为 179--601 帧，中位数 349.5 帧。

## 3. Novel-C 失败不是 sampler 或 attempt budget 不足

13 条 Novel attrition 对 front/side/rear 三个方向均完整执行 5,000 次尝试：

- 总 attempts：`195,000`；
- deterministic local：`2,366`；
- frozen uniform fallback：`192,634`；
- terminal rejection counters 对全部 195,000 次尝试逐项闭合。

通过同层、可导航、clearance、2--9 m、方向和 paired-distance 等前置门，最终到达
history-support 检查的候选为：

| direction | reached final support check | rejected because covis `>=0.10` |
|---|---:|---:|
| front | 881 | 881 |
| side | 2,847 | 2,847 |
| rear | 2,932 | 2,932 |
| **all** | **6,660** | **6,660** |

因此 side stratum 为 0 不是没有采到 side candidate。2,847 个 side candidate 已经
通过所有前置几何门，但都仍与完整 factual A+B history 共视。增加相同 sampler 的尝试
次数不会解决这个问题。

其余 terminal rejects 为：floor mismatch 99,130、clearance 49,768、wrong direction
15,717、outside 2--9 m 15,374、unreachable 4,790、identity separation 1,471、
paired-distance 1,331，以及 duplicate/non-navigable 759。上述计数只用于解释构造
funnel，不能转成导航效果。

唯一 Revisit attrition 共检查 3,456 个 grid proposals，其中 1,256 个到达完整支持
评分，全部未落入 `[0.55,0.90]`。原 receipt 没有拆分 `<0.55` 与 `>0.90`，因此不对
更细原因作推断。

## 4. 科学含义

在小型室内环境中，两段较长且成功的实际在线轨迹可能已经覆盖大部分仍可在 2--9 m
内到达的视觉空间。此时强行要求每条 prefix 同时存在“严格无支持 Novel”和“高支持
Revisit”，不是一个中性的抽样条件，而会把大量真实 lifelong prefix 排除。

这与 CEC 的运行语义也不同：CEC 从不读取二元 Novel/Revisit 标签；它只判断当前历史
证据是否足以授权。因此，随历史增长更自然的量是 historical-support coverage，而不是
人为保证每一 leg 都有一对二元角色。

## 5. 后续决策

当前不做：

- 不在 8 条 gated population 上偷跑 controller；
- 不增加同一 5,000-attempt sampler 的预算；
- 不放宽 covis、距离、方向或 scene-count gate；
- 不把 `8/22` 写成 SR。

论文中应保留该 constructibility null，并把 continual 证据拆成两部分：

1. factual leg waterfall / prefix survival，说明历史是如何真实形成的；
2. 在冻结共享 prefix 上比较 retained-history scope，回答新增经历能否成为以后可调用的
   记忆。

当前等待的 HM3D B2 实验正回答第二个问题，但只有 17 histories，必须继续标为
underpowered mechanism evidence。若会议硬性要求完整的 Leg-3 Novel/Revisit SR 矩阵，
唯一不改 estimand 的补法是 prospectively 扩展新的 actual A+B successful prefixes，
或转到更大、仍有 unsupported 空间的数据集；不能从现有 22 条内部继续调 constructor。
按当前 `8/22` constructibility 点估计，达到 16 条至少约需 44 条 successful A+B
prefix，这只是资源规划值，不是保证或 power calculation。

## 6. 可复算资产

- audit：`HM3D_TABLE2_LEG3_CONSTRUCTIBILITY_AUDIT_20260830.json`；
- independent verification：
  `HM3D_TABLE2_LEG3_CONSTRUCTIBILITY_AUDIT_VERIFICATION_20260830.json`；
- analyzer：`audit_hm3d_table2_leg3_constructibility.py`；
- verifier：`independent_verify_hm3d_table2_leg3_constructibility_audit.py`；
- source root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table2_leg3_mixed_role_20260829/construction_repair_20260829T064841Z_8e909a5b`。

独立 verifier 为 `verified=true`；它复核 22/22 completion SHA、population/verifier
SHA、全部 funnel 计数和 outcome-visibility 边界。
