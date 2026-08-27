# HM3D full-mono lifelong v3：构造性审计结果

审计日期：2026-08-27（Asia/Shanghai）。本文件只分析已经封存的 v3 构造过程，不读取、
不运行任何 factual-B、C 或 B2 导航结果。

## 1. 结论

v3 在导航前的 power gate 正确停止：最终只有 `52 histories / 19 scenes`，未达到冻结的
`96 / 15` 双门槛，因此 `factual_B_authorized=false`。这不是 Slurm、CUDA、环境依赖或
24-frame temporal sampling 的失败，而是当前 benchmark construction 本身的结构性
attrition。

最关键的两个瓶颈是：

1. 130 条 actual-online A 历史中，50 条在 runtime 可用锚点 `frame >= 39` 之后，距 A
   endpoint 的最大测地范围仍小于 2 m，无法构造非平凡 Revisit-C；
2. 剩余 80 条可构造 C 的历史中，只有 33 条能从同场景“另一条成功 online-A”找到同时
   满足 A-to-B、B-to-C 距离带与 Novel covis `<.10` 的 donor。

因此不能靠继续把同一 donor trace 采得更密来修复，也不能把 330 个高度相关 temporal
frames 当成 330 条独立历史。

## 2. 不可变运行与复算

源运行：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_fullmono_lifelong_power_v3_20260826/
  formal_20260826T141733Z_375f0b68
```

v3 construction：

```text
array: 16401203_[0-53%4]   54/54 completed
seal:  16401233            completed
```

结果盲审计：

```text
bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_lifelong_constructibility_audit_9954e16eadbc7c0d

SOURCE_BUNDLE.sha256 SHA-256:
9954e16eadbc7c0df71fda2a8fc0e2b7322e4b30d6d812e48c7a5586c95baea7

scene array: 16440094_[0-53%8]
aggregate:   16440095
```

审计重新生成了 54/54 scene 的 navmesh 几何和全部候选门，并逐一复现封存的 donor/frame
identity。`all_sealed_selections_reproduced=true`；264/264 文件哈希通过；
`query_policy_outcomes_read=false`、`navigation_outcomes_read=false`。

聚合结果：

```text
source materialized actual-A histories       130
controlled Revisit-C constructible            80
sealed Novel-B candidates                      52
sealed candidate recipients                    33
sealed scene clusters                          19
construction target met                     false
```

## 3. Recipient 级 waterfall

| 约束阶段 | 至少存在一个候选的 recipient |
|---|---:|
| actual-online A history | 130 |
| runtime-eligible Revisit-C source | 80 |
| same-floor donor | 73 |
| A-to-B reachable | 73 |
| A-to-B in 2--9 m | 68 |
| B-to-C in 2--9 m | 59 |
| B 对 recipient A 的 max covis `<.10` | 33 |

50 条短历史的 runtime-window 最大测地范围分布为：min `0.638 m`、Q1 `1.085 m`、
median `1.344 m`、Q3 `1.687 m`、max `1.995 m`。全部真实低于 2 m，不是 frame stride
漏采。`frame 39` 是当前 `8-step FIFO + 32-frame LingBot window` 的首个合法 runtime
anchor，不能为了补分母事后下调。

## 4. Candidate 级 waterfall

24 个 deterministic temporal samples 共产生 3,840 个跨历史 proposals。按第一个失败门：

| first rejection | 数量 |
|---|---:|
| floor mismatch | 847 |
| A-to-B unreachable | 72 |
| A-to-B outside 2--9 m | 1,255 |
| B-to-C outside 2--9 m | 563 |
| recipient history support not Novel | 773 |
| eligible | 330 |

330 个 eligible temporal proposals 最终只对应 33 个 recipient；应用每 recipient/donor
上限、方向优先及 2 m 平面去重后得到 52 个 sealed candidates。33 个 eligible recipient
全部仍被保留，说明 `330 -> 52` 没有损失 recipient coverage；它主要去除了同一轨迹上
空间高度相关、成功半径重叠的帧。放宽 2 m separation 只会制造伪分母。

## 5. Goal-A 长度诊断

Goal-A 初始测地距离与 C 可构造性高度相关：

| Goal-A 距离 | histories | C constructible |
|---|---:|---:|
| 3--4 m | 41 | 6 |
| 4--5 m | 37 | 22 |
| 5--6 m | 22 | 22 |
| 6--7 m | 12 | 12 |
| 7--8 m | 8 | 8 |
| 8--9 m | 10 | 10 |

但更长 A 同时更难由 frozen mono NavDP 完成。在全部 196 个原始 Goal-A attempts 中，
3--4 m 成功率为 `42/52=80.8%`，而 8--9 m 只有 `10/27=37.0%`。因此“只把 A 变长”
不是免费修复；它增加可用历史范围，却降低产生该历史的概率。

## 6. 当前科学决策

以下修复均被否决：

- 不下调 `frame >= 39`、A/B/C 的 2 m 下限或 Novel covis `<.10`；
- 不放宽 2 m candidate separation；
- 不增加同一 donor 的 temporal samples；24-frame 已经足够暴露全部候选区域；
- 不在 gate 未通过时运行 factual-B 或读取 partial SR。

下一步先审计一个更基础的 benchmark 假设：Novel-B 只是“agent 尚未亲历的目标”，是否
必须来自另一条成功 online-A history。Final14 已使用过确定性 navmesh render 来生成
自然 Novel query；旧 expert-vs-online 缺陷约束的是 Revisit-C 必须真实存在于 A history，
并不要求 Novel-B 本身来自 online history。因而已经实现独立的 direct-natural-B
constructibility audit：

- A 仍是 actual-mono online trace；
- C 仍复用 v3 封存的 actual-A controlled Revisit；
- B 在同一 navmesh、同一相机高度确定性渲染；
- A-to-B 与 B-to-C 均为 2--9 m；
- B 对 A 全历史 max covis 严格 `<.10`；
- 每条 C-constructible A 最多一个候选；
- 不运行导航、不授权 evaluation。

单场景 GPU smoke `16441089_1` 已在 L40S 上用 70 秒完成，程序和文件收据正常。该小
场景的 3 条 C-constructible histories 均未找到自然 B，主要被同楼层/clearance/可达性
挡住；它只是 implementation smoke，N=1 scene 不用于选择阈值或外推总体可行性。CPU
smoke `16441094_1` 在 49 秒内按预期因无 EGL/CUDA device 失败，证明 RGB-D render
确实需要 GPU；这不是科学或代码失败。

第一版全量 audit `16441206/16441207` 在尚未启动时被取消：它每条 A 只允许一个 B，
理论上最多 80 candidates，数学上永远不能检验原有 96-candidate reference gate。修正后
每条 A 最多保留 4 个 deterministic natural-B，且任意两个位置平面距离至少 2 m；没有
放松 A/B/C 距离、covis、clearance 或 runtime-history 条件。

拆分诊断后的不可变 multi-candidate audit bundle：

```text
bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_lifelong_natural_b_audit_7327a6919acafef8
receipt SHA-256:
7327a6919acafef8a60d84142015422d60f11da8d5820f0b505d2504dc7a27de

multi-candidate GPU smoke: 16441408_1 (waiting for GPU quota)
```

smoke 通过后才提交新的 54-scene array。只有全量 audit 达到原有 `96 candidates / 15
scenes` reference gate，才会考虑另行冻结 v4；审计本身不授予 evaluation 权限。
