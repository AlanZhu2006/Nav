# Final14 CEC Cache 与延迟口径审计

日期：2026-08-18（Asia/Shanghai）  
性质：**冻结结果上的只读 post-hoc implementation audit**；不改变任何 SR、SPL、
配对统计、certificate 决策或 Final14 verifier 结论。

## 1. 审计结论

CEC 并没有在每次 NavDP replan 时重新执行完整重定位。每个 query 恰好执行一次
uncached certificate，之后固定该次 absolute goal pose 或 abstention，只重新计算当前
相对 bearing。Final14 的 84 个 certified plan 文件中，两个 protocol 各有 42 个 query，
全部满足：

```text
第 1 次 certificate request：cached = false
其余 request：             cached = true
```

真正的部署瓶颈是**每个新 goal 的第一次认证**，尤其是为较晚的历史 anchor 顺序重放
LingBot dense causal depth；后续更新已经低于 1 ms。

## 2. 发现的汇总口径问题

runtime 的 cache-hit response 会保留第一次结果里的
`certified_relocalization_uncached_ms`，这是为了让原始首次认证 receipt 始终可审计。
旧 Final14 summarizer 却按“字段非空”收集 latency，因此把同一个首次耗时在每次后续
replan 中重复了一遍：

- Natural：真实 `42` 个首次 query 被扩成 `1388` 个值，放大 `33.05x`；
- Hard-support：真实 `42` 个首次 query 被扩成 `1461` 个值，放大 `34.79x`。

这只影响 latency distribution 的口径，**不影响任何导航结果或方法比较**。正式
summary 与 independent verification 文件保持冻结，不被重写；未来 summarizer 已改为
按 `certified_relocalization_cached` 显式分流。

## 3. 从原始 plans 复算的正确延迟

### Natural-direction

| 指标 | n | median | p95 | mean | max |
|---|---:|---:|---:|---:|---:|
| 首次 uncached certificate | 42 | `3.404 s` | `26.348 s` | `7.820 s` | `45.219 s` |
| 后续 cached bearing update | 1346 | `0.152 ms` | `0.428 ms` | `0.217 ms` | `0.539 ms` |

首次 query 按 role：

| role | n | median | p95 | mean |
|---|---:|---:|---:|---:|
| Novel | 21 | `0.817 s` | `19.854 s` | `4.684 s` |
| Revisit | 21 | `10.639 s` | `26.690 s` | `10.956 s` |

首次 query 按最终 certificate 决策：

| 决策 | n | median | p95 | mean |
|---|---:|---:|---:|---:|
| accepted | 23 | `7.817 s` | `25.969 s` | `10.221 s` |
| rejected | 19 | `0.547 s` | `22.391 s` | `4.914 s` |

### Hard-support

| 指标 | n | median | p95 | mean | max |
|---|---:|---:|---:|---:|---:|
| 首次 uncached certificate | 42 | `3.054 s` | `20.829 s` | `6.731 s` | `45.949 s` |
| 后续 cached bearing update | 1419 | `0.158 ms` | `0.474 ms` | `0.225 ms` | `0.596 ms` |

Hard-support 的 duplicated Novel 只属于 instrumentation；上述数字可用于实现审计，
不能把 duplicated Novel 当成新的科学样本。

## 4. 为什么有 20--45 秒长尾

当前第一次 CEC query 依次做：

1. 使用已在线保存的 DINO CLS 从 causal history 冻结 top-8；
2. goal SuperPoint feature 只提取一次，对 top-8 reference 分别做
   SuperPoint + LightGlue；
3. 用 fundamental support 排序；
4. 对选中的一个 candidate 从 scale block 后顺序重放 RGB 到 anchor，恢复 dense causal
   LingBot depth/confidence；
5. PnP 与 atomic certificate；
6. 缓存 absolute goal pose 或 sticky abstention。

Natural 中 selected-anchor index 与首次 latency 的 Pearson 相关为 `r=0.817`；
Hard-support 为 `r=0.664`。最慢 Natural query 的 anchor 为 `307`，耗时 `45.22 s`；
最慢 accepted Revisit anchor 为 `274`，耗时 `40.06 s`。这与顺序 replay 成本随历史位置
增长一致。

Revisit 通常有足够几何支持，会进入 depth/PnP，所以 median 明显高于大部分在早期证据
检查处拒绝的 Novel。不能据此说 matcher 没有成本，但现有证据明确把主要长尾指向
selected-anchor depth replay。

## 5. 当前已经存在的 cache

- 每帧 DINO CLS 在线计算并存于 CPU；
- goal DINO feature 缓存；
- 第一次 causal query 的 top-8 shortlist 冻结并缓存；
- goal SuperPoint feature 在 top-8 matching 内复用；
- 每个 goal 的最终 absolute pose 或 abstention 缓存；
- 后续 replan 只做 current-relative bearing，Natural median `0.152 ms`。

因此不能再笼统地说“CEC 没有 cache”。缺的是**第一次 query 所需历史局部特征与
dense causal depth 的写入时预计算/有界存储**。

## 6. 优化优先级

### P0：已完成，不改变方法

- 修正未来 latency summarizer，只在 `cached=false` 记录首次耗时，在 `cached=true`
  记录更新耗时；
- 增加独立 raw-plan audit，严格检查每个 query 恰好一次 uncached request；
- 保存本次复算 JSON 与 SHA-256。

### P1：下一项应做的 decision-equivalence microbenchmark

给 memory writer 增加受控的 causal geometry cache，但先不跑长闭环：

1. 在历史写入时缓存 reference SuperPoint features；
2. 建立与当前 dense replay 同状态语义的 depth cache，或保存有界 dense replay
   checkpoints 后只重放短 suffix；
3. 在既有 frozen traces 上同时运行 legacy lazy replay 与 cached path；
4. 必须逐项满足：top-8 相同、geometry order 相同、PnP/certificate accept 相同、
   selected anchor 相同、bearing 数值在冻结 tolerance 内、最终动作相同；
5. 只有 `0` 个 decision mismatch 后，才报告 latency/memory scaling 并考虑替换正式
   runtime。

注意：live navigation stream 使用 flow-gated sparse KV，而 certificate depth 使用
dense causal replay；直接把当前 flow-gate 内临时算出的 depth 存下来，**尚不能假定
decision-equivalent**。

### 暂不做：会改变科学方法的“优化”

- 减少 top-K；
- 只保留 sparse keyframes；
- 改 certificate threshold；
- 用 learned proof 替换 certificate；
- 根据 Final14 failures 选择 replay 长度。

这些都会改变 coverage 或决策，必须作为新方法在新的 untouched population 上预注册，
不能伪装成无损加速。

## 7. 可复现文件

- 审计脚本：`MemNavData/audit_final14_cec_latency.py`；
- 单元测试：`MemNavData/test_audit_final14_cec_latency.py`；
- 复算结果：
  `.diagnostics/learned_relocalizer_20260817/final14_attempt7_formal_result_20260818/final14_cec_latency_audit_20260818.json`；
- 结果 SHA-256：
  `aef1c45599ee34b20e0e52c7466ae753a492f21924e7e4d616b4872e443bfe9f`；
- 冻结科学结果：`MemNavData/FINAL14_CEC_PI3X_FORMAL_RESULT_20260818.md`。

## 8. P1 后续结果

P1 decision-equivalence microbenchmark 已完成，详见
`MemNavData/CEC_LATENCY_OPTIMIZATION_RESULT_20260818.md`。简要结论：

- selected-anchor final depth cache 与 reference SuperPoint LRU 可严格复用；
- special-prefix/短 suffix replay 和 multi-frame block replay 均产生数值偏差，已排除；
- 可选 exact eager dense writer 将单条端到端认证从 `11.7408 s` 降到
  `22.84 ms`，但代价为约 `0.176 s/frame` 写入开销和 `7.59 GiB` 额外 CUDA
  allocated，因此保持显式开关、默认关闭；
- 后续又在两个不同 scene、121/161 帧轨迹上复验，normal/eager 在线状态 SHA-256
  均一致，anchor depth/confidence 逐元素零误差，19.39/27.09 秒 replay 分别降到
  0.266/0.215 毫秒。完整代价与边界见上述 P1 文档。
